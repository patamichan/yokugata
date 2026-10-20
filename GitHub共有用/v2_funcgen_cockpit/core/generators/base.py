"""
Base Abstract Airfoil Generator Interface
Allows plugging in different generation algorithms (FuncGen-Warp, CST-Fourier, B-spline, etc.)
seamlessly without breaking downstream analysis or optimization pipelines.
"""

from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np


class BaseAirfoilGenerator(ABC):
    """Abstract Base Class for Airfoil Generators in 3D parameter space [tc, xt, kle]."""

    @abstractmethod
    def generate(
        self,
        tc_pct: float,
        xt_pct: float,
        kle: float = 1.0,
        num_points: int = 161
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate unit-chord symmetric airfoil coordinates.
        
        Args:
            tc_pct: Thickness ratio in percent (e.g. 25.0 for 25% t/c)
            xt_pct: Maximum thickness position in percent (e.g. 42.0 for 42% c)
            kle: Leading Edge radius scaling ratio relative to standard NACA (1.0 = base)
            num_points: Total number of coordinates (default: 161)
            
        Returns:
            (x, y): 1D numpy arrays, ordered upper TE -> LE -> lower TE (Selig format)
        """
        pass
