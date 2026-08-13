"""Generate small synthetic CSVs that mimic the *public* schemas of ToN-IoT
(Train_Test_Network.csv) and Edge-IIoTset (ML-EdgeIIoT-dataset.csv).

WHY THIS EXISTS
----------------
The real datasets are too large to bundle and are license-gated downloads
(see each config's `dataset.source_note`). This script exists purely so the
codebase can be smoke-tested end-to-end without the real data, and so
reviewers/graders can see the pipeline run. It is NOT a substitute for
running on the real datasets, and results produced on this synthetic data
have no scientific meaning — `run_all.py` and the README both say so loudly.

DESIGN
------
For each simulated "asset" we generate a benign background traffic stream,
and for a configurable fraction of assets we additionally inject an attack
*campaign*: a period of IAD (recon) traffic, followed by a period of LMEP
(lateral movement/escalation/persistence) traffic, followed by a short IMP
(impact) burst. This gives the downstream impact forecaster genuine
pre-impact signal to learn from, which is the only way to smoke-test that
stage of the pipeline meaningfully.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TONIOT_IAD = ["scanning", "password", "xss"]
TONIOT_LMEP = ["backdoor", "injection", "mitm"]
TONIOT_IMP = ["dos", "ddos", "ransomware"]

EDGE_IAD = ["Fingerprinting", "Port_Scanning", "Vulnerability_scanner", "Password", "XSS"]
EDGE_LMEP = ["Backdoor", "SQL_injection", "Uploading", "MITM"]
EDGE_IMP = ["DDoS_UDP", "DDoS_ICMP", "DDoS_TCP", "DDoS_HTTP", "Ransomware"]


def _asset_ip(i: int) -> str:
    return f"192.168.{1 + i // 250}.{1 + i % 250}"


def _peer_ip(rng: np.random.Generator) -> str:
    return f"10.0.{rng.integers(0, 5)}.{rng.integers(1, 250)}"


def _simulate_asset_records(
    rng: np.random.Generator,
    asset_id: str,
    t0: float,
    duration_s: float,
    record_rate_hz: float,
    is_attacked: bool,
    iad_labels: list[str],
    lmep_labels: list[str],
    imp_labels: list[str],
) -> list[dict]:
    """Return a list of raw-record dicts (schema-agnostic core fields) for one asset."""
    n_records = max(20, int(duration_s * record_rate_hz))
    times = np.sort(rng.uniform(t0, t0 + duration_s, size=n_records))

    if is_attacked:
        # Campaign occupies the last ~55% of the timeline: IAD -> LMEP -> IMP,
        # leaving the first ~45% purely benign so the forecaster has
        # negative (pre-attack) examples too.
        campaign_start = t0 + duration_s * 0.45
        iad_end = t0 + duration_s * 0.65
        lmep_end = t0 + duration_s * 0.90
        # IMP is a short terminal burst.
        imp_start = t0 + duration_s * 0.93
    records = []
    for t in times:
        label = "normal" if not is_attacked else None
        if is_attacked:
            if t < campaign_start:
                label = "normal"
            elif t < iad_end:
                label = rng.choice(iad_labels)
            elif t < lmep_end:
                label = rng.choice(lmep_labels)
            elif t >= imp_start:
                label = rng.choice(imp_labels)
            else:
                label = "normal"
        records.append({"asset_id": asset_id, "t": t, "label": label})
    return records


def _build_common(n_assets: int, attacked_frac: float, seed: int, hours: float, rate_hz: float, iad, lmep, imp):
    rng = np.random.default_rng(seed)
    t0 = 1_700_000_000.0  # arbitrary epoch anchor
    duration_s = hours * 3600
    n_attacked = int(n_assets * attacked_frac)
    attacked_flags = np.array([True] * n_attacked + [False] * (n_assets - n_attacked))
    rng.shuffle(attacked_flags)

    all_records = []
    asset_ips = [_asset_ip(i) for i in range(n_assets)]
    for i, asset_ip in enumerate(asset_ips):
        recs = _simulate_asset_records(
            rng, asset_ip, t0, duration_s, rate_hz, bool(attacked_flags[i]), iad, lmep, imp
        )
        all_records.extend(recs)

    df = pd.DataFrame(all_records)
    df = df.sort_values("t").reset_index(drop=True)

    n = len(df)
    # Destination mix: a meaningful share of traffic targets ANOTHER
    # monitored asset (not just an external peer) so the asset-time
    # interaction graph actually has edges to learn from. Attack traffic
    # (IAD/LMEP especially — recon + lateral movement are inherently
    # asset-to-asset) is biased toward internal targets.
    internal_prob = np.where(df["label"] == "normal", 0.30, 0.65)
    is_internal = rng.uniform(size=n) < internal_prob
    dst_ips = np.empty(n, dtype=object)
    for idx in range(n):
        if is_internal[idx] and len(asset_ips) > 1:
            candidates = [a for a in asset_ips if a != df["asset_id"].iat[idx]]
            dst_ips[idx] = candidates[rng.integers(0, len(candidates))]
        else:
            dst_ips[idx] = _peer_ip(rng)
    df["dst_ip"] = dst_ips
    df["src_port"] = rng.integers(1024, 65535, size=n)
    df["dst_port"] = rng.choice([22, 23, 80, 443, 502, 1883, 8080, 5000], size=n)
    df["proto"] = rng.choice(["tcp", "udp", "icmp"], size=n, p=[0.7, 0.25, 0.05])
    df["duration"] = np.clip(rng.exponential(1.5, size=n), 0, 60)
    df["src_bytes"] = rng.integers(0, 5000, size=n)
    df["dst_bytes"] = rng.integers(0, 5000, size=n)
    df["src_pkts"] = rng.integers(1, 50, size=n)
    df["dst_pkts"] = rng.integers(1, 50, size=n)
    # attack traffic tends to have higher packet/byte volume — gives the RF
    # something real to learn, not just the label leaking through directly.
    attack_mask = df["label"] != "normal"
    df.loc[attack_mask, "src_bytes"] += rng.integers(2000, 20000, size=attack_mask.sum())
    df.loc[attack_mask, "src_pkts"] += rng.integers(20, 200, size=attack_mask.sum())
    return df


def generate_toniot(out_path: Path, n_assets: int, attacked_frac: float, seed: int, hours: float, rate_hz: float) -> None:
    df = _build_common(n_assets, attacked_frac, seed, hours, rate_hz, TONIOT_IAD, TONIOT_LMEP, TONIOT_IMP)
    n = len(df)
    rng = np.random.default_rng(seed + 1)

    out = pd.DataFrame()
    out["ts"] = df["t"]
    out["src_ip"] = df["asset_id"]
    out["dst_ip"] = df["dst_ip"]
    out["src_port"] = df["src_port"]
    out["dst_port"] = df["dst_port"]
    out["proto"] = df["proto"]
    out["service"] = rng.choice(["-", "http", "dns", "ssl", "dhcp"], size=n)
    out["duration"] = df["duration"]
    out["src_bytes"] = df["src_bytes"]
    out["dst_bytes"] = df["dst_bytes"]
    out["conn_state"] = rng.choice(["SF", "S0", "REJ", "RSTO"], size=n)
    out["missed_bytes"] = 0
    out["src_pkts"] = df["src_pkts"]
    out["src_ip_bytes"] = df["src_bytes"] + rng.integers(0, 100, size=n)
    out["dst_pkts"] = df["dst_pkts"]
    out["dst_ip_bytes"] = df["dst_bytes"] + rng.integers(0, 100, size=n)
    out["dns_query"] = "-"
    out["dns_qclass"] = 0
    out["dns_qtype"] = 0
    out["dns_rcode"] = 0
    out["dns_AA"] = rng.choice(["T", "F"], size=n)
    out["dns_RD"] = rng.choice(["T", "F"], size=n)
    out["dns_RA"] = rng.choice(["T", "F"], size=n)
    out["dns_rejected"] = rng.choice(["T", "F"], size=n)
    out["ssl_version"] = rng.choice(["-", "TLSv12", "TLSv13"], size=n)
    out["ssl_cipher"] = "-"
    out["ssl_resumed"] = rng.choice(["T", "F"], size=n)
    out["ssl_established"] = rng.choice(["T", "F"], size=n)
    out["ssl_subject"] = "-"
    out["ssl_issuer"] = "-"
    out["http_trans_depth"] = 0
    out["http_method"] = rng.choice(["-", "GET", "POST"], size=n)
    out["http_uri"] = "-"
    out["http_referrer"] = "-"
    out["http_version"] = rng.choice(["-", "1.1"], size=n)
    out["http_request_body_len"] = rng.integers(0, 500, size=n)
    out["http_response_body_len"] = rng.integers(0, 500, size=n)
    out["http_status_code"] = rng.choice([0, 200, 301, 404, 500], size=n)
    out["http_user_agent"] = "-"
    out["http_orig_mime_types"] = "-"
    out["http_resp_mime_types"] = "-"
    out["weird_name"] = rng.choice(["-", "dns_unmatched_reply"], size=n, p=[0.95, 0.05])
    out["weird_addl"] = "-"
    out["weird_notice"] = rng.choice(["F", "T"], size=n, p=[0.97, 0.03])
    out["type"] = df["label"]
    out["label"] = (df["label"] != "normal").astype(int)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"[toniot] wrote {len(out):,} rows -> {out_path}")


def generate_edgeiiotset(out_path: Path, n_assets: int, attacked_frac: float, seed: int, hours: float, rate_hz: float) -> None:
    df = _build_common(n_assets, attacked_frac, seed, hours, rate_hz, EDGE_IAD, EDGE_LMEP, EDGE_IMP)
    n = len(df)
    rng = np.random.default_rng(seed + 2)

    ts = pd.to_datetime(df["t"], unit="s")
    frame_time = ts.dt.strftime("%b %d, %Y %H:%M:%S.000000000")

    out = pd.DataFrame()
    out["frame.time"] = frame_time
    out["ip.src_host"] = df["asset_id"]
    out["ip.dst_host"] = df["dst_ip"]
    out["arp.opcode"] = rng.integers(0, 3, size=n)
    out["arp.hw.size"] = 6
    out["icmp.checksum"] = rng.integers(0, 65535, size=n)
    out["icmp.seq_le"] = rng.integers(0, 100, size=n)
    out["icmp.unused"] = 0
    out["http.content_length"] = rng.integers(0, 2000, size=n)
    out["http.response"] = rng.choice([0, 1], size=n, p=[0.8, 0.2])
    out["http.request.method"] = rng.choice(["0", "GET", "POST"], size=n)
    out["http.request.version"] = rng.choice(["0", "HTTP/1.1"], size=n)
    out["http.tls_port"] = rng.choice([0, 443], size=n)
    out["tcp.ack"] = rng.integers(0, 2**16, size=n)
    out["tcp.checksum"] = rng.integers(0, 65535, size=n)
    out["tcp.connection.fin"] = rng.choice([0, 1], size=n, p=[0.9, 0.1])
    out["tcp.connection.rst"] = rng.choice([0, 1], size=n, p=[0.9, 0.1])
    out["tcp.connection.syn"] = rng.choice([0, 1], size=n, p=[0.8, 0.2])
    out["tcp.connection.synack"] = rng.choice([0, 1], size=n, p=[0.8, 0.2])
    out["tcp.flags"] = rng.integers(0, 255, size=n)
    out["tcp.flags.ack"] = rng.choice([0, 1], size=n)
    out["tcp.len"] = df["src_bytes"]
    out["tcp.seq"] = rng.integers(0, 2**16, size=n)
    out["tcp.srcport"] = df["src_port"]
    out["tcp.dstport"] = df["dst_port"]
    out["udp.port"] = rng.integers(0, 65535, size=n)
    out["udp.stream"] = rng.integers(0, 50, size=n)
    out["udp.time_delta"] = rng.exponential(0.5, size=n)
    out["dns.qry.name"] = rng.choice(["0", "device.local", "broker.local"], size=n)
    out["dns.qry.name.len"] = rng.integers(0, 30, size=n)
    out["dns.qry.qu"] = rng.choice([0, 1], size=n)
    out["dns.qry.type"] = rng.choice([0, 1, 28], size=n)
    out["dns.retransmission"] = rng.choice([0, 1], size=n, p=[0.95, 0.05])
    out["mqtt.conack.flags"] = "0"
    out["mqtt.conflag.cleansess"] = rng.choice([0, 1], size=n)
    out["mqtt.len"] = rng.integers(0, 200, size=n)
    out["mqtt.msgtype"] = rng.integers(0, 14, size=n)
    out["mqtt.hdrflags"] = "0x00000000"
    out["mqtt.msg_decoded_as"] = rng.choice(["0", "1"], size=n)
    out["mqtt.proto_len"] = rng.integers(0, 10, size=n)
    out["mqtt.protoname"] = rng.choice(["0", "MQTT"], size=n)
    out["mqtt.topic_len"] = rng.integers(0, 20, size=n)
    out["mqtt.ver"] = rng.choice([0, 3, 4], size=n)
    out["mbtcp.len"] = rng.integers(0, 50, size=n)
    out["mbtcp.trans_id"] = rng.integers(0, 1000, size=n)
    out["mbtcp.unit_id"] = rng.integers(0, 5, size=n)
    out["Attack_type"] = df["label"]
    out["Attack_label"] = (df["label"] != "normal").astype(int)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"[edgeiiotset] wrote {len(out):,} rows -> {out_path}")


def _write_synthetic_marker(dataset_dir: Path, args: argparse.Namespace) -> None:
    """Sentinel file that tells every manuscript-facing script
    (src/utils/data_provenance.py) this directory holds generated
    smoke-test data, not the real dataset -- see that module's docstring.
    """
    import json
    from datetime import datetime, timezone

    marker = dataset_dir / ".SYNTHETIC_DATA_MARKER"
    marker.write_text(json.dumps({
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "generator": "scripts/generate_synthetic_data.py",
        "args": vars(args),
    }, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-assets", type=int, default=40)
    ap.add_argument("--attacked-frac", type=float, default=0.4)
    ap.add_argument("--hours", type=float, default=6.0)
    ap.add_argument("--rate-hz", type=float, default=0.05, help="mean records/sec per asset")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--clean-markers", action="store_true",
                     help="Only delete the .SYNTHETIC_DATA_MARKER files (e.g. after you've manually "
                          "replaced the CSVs with real data) -- does NOT generate/overwrite any data.")
    args = ap.parse_args()

    toniot_dir = PROJECT_ROOT / "data/raw/toniot"
    edge_dir = PROJECT_ROOT / "data/raw/edgeiiotset"

    if args.clean_markers:
        for d in (toniot_dir, edge_dir):
            marker = d / ".SYNTHETIC_DATA_MARKER"
            if marker.exists():
                marker.unlink()
                print(f"removed {marker}")
        return

    generate_toniot(
        toniot_dir / "Train_Test_Network.csv",
        args.n_assets, args.attacked_frac, args.seed, args.hours, args.rate_hz,
    )
    generate_edgeiiotset(
        edge_dir / "ML-EdgeIIoT-dataset.csv",
        args.n_assets, args.attacked_frac, args.seed, args.hours, args.rate_hz,
    )
    _write_synthetic_marker(toniot_dir, args)
    _write_synthetic_marker(edge_dir, args)
    print("Wrote .SYNTHETIC_DATA_MARKER in both raw data dirs -- manuscript-facing scripts "
          "will refuse to run against this data. Replace the CSVs with real data, which will "
          "leave the (now stale) marker behind -- run `--clean-markers` afterward, or the "
          "marker's own args (# rows etc.) will visibly mismatch the real file and you can just delete it by hand.")


if __name__ == "__main__":
    main()
