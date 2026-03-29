"""Python interface to the One Brain C++20 Feature Engine."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

# Try to import the compiled C++ module
try:
    from _onebrain_cpp import (
        Cancel,
        CircularBufferDouble,
        FeatureEngine,
        FeatureEngineConfig,
        Fill,
        MarketStateVector,
        Tick,
        WelfordAccumulator,
    )
except ImportError:
    # Check if the .so is in the build directory
    build_dir = Path(__file__).resolve().parent.parent / "cpp" / "build"
    if build_dir.exists():
        # Find the .so file
        so_files = list(build_dir.glob("_onebrain_cpp*.so")) + \
                   list(build_dir.glob("_onebrain_cpp*.dylib"))
        if so_files:
            sys.path.insert(0, str(so_files[0].parent))
            from _onebrain_cpp import (
                Cancel,
                CircularBufferDouble,
                FeatureEngine,
                FeatureEngineConfig,
                Fill,
                MarketStateVector,
                Tick,
                WelfordAccumulator,
            )
        else:
            raise ImportError(
                "C++ module not built. Run: cd src/cpp && mkdir -p build && cd build "
                "&& cmake .. && make -j"
            )
    else:
        raise ImportError(
            "C++ module not built. Run: cd src/cpp && mkdir -p build && cd build "
            "&& cmake .. && make -j"
        )

__all__ = [
    "Cancel",
    "CircularBufferDouble",
    "FeatureEngine",
    "FeatureEngineConfig",
    "Fill",
    "MarketStateVector",
    "Tick",
    "WelfordAccumulator",
]
