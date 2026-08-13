"""ToN-IoT (Network) dataset adapter.

ToN-IoT is the *main* dataset for this pipeline: it has clean `ts`
(unix epoch), `src_ip`/`dst_ip`, and a `type` attack-family column, so every
stage of the paper pipeline (stage detection, graph construction, impact
forecasting) is fully supported.
"""
from __future__ import annotations

from src.data.generic_adapter import GenericAdapter


class TonIotAdapter(GenericAdapter):
    """No dataset-specific overrides needed beyond the generic config-driven
    loader — ToN-IoT's `ts` column is handled by the 'epoch_seconds' branch
    of GenericAdapter.parse_timestamp."""

    pass
