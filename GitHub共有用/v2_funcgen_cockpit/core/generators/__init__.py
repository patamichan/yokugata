"""
Airfoil Generator Plugin Registry
Provides flexible swapping between different airfoil generation algorithms:
- "warp": FuncGenFoil + C2 coordinate warp with C3 compact nose window
- "cst": CST fairing with fixed nose coefficient and exact peak constraints
"""

from core.generators.base import BaseAirfoilGenerator
from core.generators.funcgen_warp import FuncGenWarpGenerator
from core.generators.funcgen_cst import FuncGenCSTGenerator


_GENERATORS = {
    "warp": FuncGenWarpGenerator,
    "cst": FuncGenCSTGenerator,
}


def get_generator(method: str = "warp", **kwargs) -> BaseAirfoilGenerator:
    """Factory function to instantiate an airfoil generator."""
    if method not in _GENERATORS:
        raise ValueError(f"Unknown generator method '{method}'. Available: {list(_GENERATORS.keys())}")
    return _GENERATORS[method](**kwargs)
