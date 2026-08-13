"""Factory for picking the right adapter given a dataset name + config."""
from __future__ import annotations

from typing import Any, Dict

from src.data.edgeiiotset_adapter import EdgeIIoTAdapter
from src.data.generic_adapter import GenericAdapter
from src.data.schema import DatasetBundle
from src.data.toniot_adapter import TonIotAdapter

_ADAPTERS = {
    "toniot": TonIotAdapter,
    "edgeiiotset": EdgeIIoTAdapter,
}


def get_adapter(dataset_name: str, config: Dict[str, Any]) -> GenericAdapter:
    if dataset_name not in _ADAPTERS:
        raise ValueError(f"Unknown dataset '{dataset_name}'. Known: {list(_ADAPTERS)}")
    return _ADAPTERS[dataset_name](config)


def load_dataset(dataset_name: str, config: Dict[str, Any]) -> DatasetBundle:
    return get_adapter(dataset_name, config).load()
