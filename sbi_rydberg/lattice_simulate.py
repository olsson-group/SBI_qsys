"""Square-lattice geometry and simulation config for the Rydberg SBI pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RydbergSimConfig:
    """Rydberg quench on an nx×ny lattice (μm, μs, rad/μs)."""

    nx: int = 3
    ny: int = 3
    lattice_constant_um: float = 8.0
    omega_rad_per_us: float = 15.0
    detuning_rad_per_us: float = 0.0
    times_us: tuple[float, ...] = (0.4, 0.8, 1.2, 1.6)
    dt_us: float = 0.01
    c6_rad_um6_per_us: float = float(2.0 * np.pi * 862690.0)  # Rb 87S_{1/2}, n≈70
    interaction_scale: float = 1.0
    v_threshold: float = 1e-3

    @property
    def n_atoms(self) -> int:
        return self.nx * self.ny

    @property
    def n_obs_per_time(self) -> int:
        """Z_i, plus ZZ on NN and NNN bonds."""
        return (
            self.n_atoms
            + len(nearest_neighbor_pairs(self.nx, self.ny))
            + len(next_nearest_neighbor_pairs(self.nx, self.ny))
        )

    @property
    def n_obs(self) -> int:
        return len(self.times_us) * self.n_obs_per_time


def ideal_square_positions(nx: int, ny: int, a_um: float) -> np.ndarray:
    """(nx*ny, 2) site positions, x fastest."""
    pos = np.empty((nx * ny, 2), dtype=np.float64)
    k = 0
    for iy in range(ny):
        for ix in range(nx):
            pos[k] = (ix * a_um, iy * a_um)
            k += 1
    return pos


def nearest_neighbor_pairs(nx: int, ny: int) -> list[tuple[int, int]]:
    """Horizontal and vertical bonds (i, j), i < j."""
    pairs = []
    for iy in range(ny):
        for ix in range(nx):
            i = iy * nx + ix
            if ix + 1 < nx:
                pairs.append((i, i + 1))
            if iy + 1 < ny:
                pairs.append((i, i + nx))
    return pairs


def next_nearest_neighbor_pairs(nx: int, ny: int) -> list[tuple[int, int]]:
    """Diagonal bonds (i, j), i < j."""
    pairs = []
    for iy in range(ny - 1):
        for ix in range(nx - 1):
            i = iy * nx + ix
            pairs.append((i, i + nx + 1))
            pairs.append((i + 1, i + nx))
    return pairs
