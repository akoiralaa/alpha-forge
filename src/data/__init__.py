
# Lazy imports — avoids pulling in arcticdb/sqlalchemy when only the ingest layer is needed
def __getattr__(name):
    _lazy = {
        "TickStore": "src.data.arctic_store",
        "SymbolMaster": "src.data.symbol_master",
        "DataQualityPipeline": "src.data.quality_pipeline",
        "UniverseManager": "src.data.universe",
        "PriceAdjuster": "src.data.adjustments",
        "FundamentalsStore": "src.data.fundamentals",
        "EventStore": "src.data.events",
    }
    if name in _lazy:
        import importlib
        mod = importlib.import_module(_lazy[name])
        return getattr(mod, name)
    raise AttributeError(f"module 'src.data' has no attribute {name!r}")

__all__ = [
    "TickStore",
    "SymbolMaster",
    "DataQualityPipeline",
    "UniverseManager",
    "PriceAdjuster",
    "FundamentalsStore",
    "EventStore",
]
