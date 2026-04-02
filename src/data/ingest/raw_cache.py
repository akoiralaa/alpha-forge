from __future__ import annotations

import gzip
import hashlib
import json
import math
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd


def to_jsonable(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (datetime, date, pd.Timestamp)):
        return value.isoformat()
    if is_dataclass(value):
        return to_jsonable(asdict(value))
    if hasattr(value, "model_dump"):
        return to_jsonable(value.model_dump())
    if hasattr(value, "dict"):
        return to_jsonable(value.dict())
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(v) for v in value]
    if hasattr(value, "__dict__"):
        return {
            str(k): to_jsonable(v)
            for k, v in vars(value).items()
            if not str(k).startswith("_")
        }
    return str(value)


class RawDataCache:
    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def cache_path(self, namespace: str, key: str) -> Path:
        safe_namespace = namespace.strip("/").replace("..", "_")
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
        stem = "_".join(
            part
            for part in key.replace("/", "_").replace(",", "_").replace(" ", "_").split("_")
            if part
        )[:80]
        filename = f"{stem}_{digest}.json.gz" if stem else f"{digest}.json.gz"
        return self.root / safe_namespace / filename

    def read_json(self, namespace: str, key: str) -> Any | None:
        path = self.cache_path(namespace, key)
        if not path.exists():
            return None
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            return json.load(fh)

    def write_json(self, namespace: str, key: str, payload: Any) -> Path:
        path = self.cache_path(namespace, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with gzip.open(tmp, "wt", encoding="utf-8") as fh:
            json.dump(to_jsonable(payload), fh, sort_keys=True)
        tmp.replace(path)
        return path

    def get_or_fetch_json(
        self,
        namespace: str,
        key: str,
        fetcher: Callable[[], Any],
        *,
        refresh: bool = False,
        cache_only: bool = False,
    ) -> tuple[Any, bool]:
        if not refresh:
            cached = self.read_json(namespace, key)
            if cached is not None:
                return cached, True
        if cache_only:
            raise FileNotFoundError(f"raw cache miss for {namespace}:{key}")
        payload = fetcher()
        self.write_json(namespace, key, payload)
        return payload, False

__all__ = ["RawDataCache", "to_jsonable"]
