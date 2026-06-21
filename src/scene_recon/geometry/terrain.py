from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from affine import Affine


@dataclass(frozen=True)
class TerrainModel:
    """Raster terrain with world<->pixel mapping and bilinear sampling.

    `array` row 0 is north/top (standard raster). `transform` maps
    (col, row) -> (easting, northing); its inverse maps world -> pixel.
    """

    array: np.ndarray
    transform: Affine
    nodata: float = -9999.0
    datum_offset_m: float = 0.0

    @classmethod
    def from_array(
        cls,
        array: np.ndarray,
        transform: Affine,
        *,
        nodata: float = -9999.0,
        datum_offset_m: float = 0.0,
    ) -> "TerrainModel":
        return cls(
            array=np.asarray(array, dtype=np.float32),
            transform=transform,
            nodata=float(nodata),
            datum_offset_m=float(datum_offset_m),
        )

    @classmethod
    def from_gpkg(
        cls,
        path,
        *,
        bbox_utm,
        margin_m: float = 500.0,
        datum_offset_m: float = 0.0,
    ) -> "TerrainModel":
        import rasterio  # lazy: importing this module must not require rasterio
        from rasterio.windows import from_bounds

        e_min, n_min, e_max, n_max = bbox_utm
        with rasterio.open(path) as ds:
            window = from_bounds(
                e_min - margin_m,
                n_min - margin_m,
                e_max + margin_m,
                n_max + margin_m,
                transform=ds.transform,
            )
            window = window.round_offsets().round_lengths()
            full = rasterio.windows.Window(0, 0, ds.width, ds.height)
            window = window.intersection(full)
            array = ds.read(1, window=window).astype(np.float32)
            transform = ds.window_transform(window)
            nodata = ds.nodata if ds.nodata is not None else -9999.0
        return cls(
            array=array,
            transform=transform,
            nodata=float(nodata),
            datum_offset_m=float(datum_offset_m),
        )

    def elevation_bounds(self) -> tuple[float, float]:
        """Min/max real elevation in this terrain window (datum-corrected).

        Used to bound ray-marching: a ray can only intersect terrain inside this
        band, so we march the [z_hi, z_lo] bracket instead of the full range.
        Returns (-inf, inf) if the window is all-nodata (disables the optimization).
        """
        a = self.array
        valid = np.isfinite(a) & (a != self.nodata)
        if not valid.any():
            return (float("-inf"), float("inf"))
        v = a[valid]
        return (float(v.min()) + self.datum_offset_m, float(v.max()) + self.datum_offset_m)

    def elevation_at(self, easting, northing) -> np.ndarray:
        e = np.asarray(easting, dtype=float)
        n = np.asarray(northing, dtype=float)
        shape = np.broadcast_shapes(e.shape, n.shape)
        ef, nf = (a.ravel() for a in np.broadcast_arrays(e, n))

        inv = ~self.transform
        col = inv.a * ef + inv.b * nf + inv.c
        row = inv.d * ef + inv.e * nf + inv.f

        h, w = self.array.shape
        in_bounds = (col >= 0) & (col <= w - 1) & (row >= 0) & (row <= h - 1)

        # ponytail: bilinear needs a 2x2 neighborhood, so arrays must be >=2x2.
        # Clamping col0/row0 to W-2/H-2 keeps the last row/col exact (frac->1).
        col0 = np.clip(np.floor(col).astype(int), 0, w - 2)
        row0 = np.clip(np.floor(row).astype(int), 0, h - 2)
        fx = col - col0
        fy = row - row0

        v00 = self.array[row0, col0]
        v01 = self.array[row0, col0 + 1]
        v10 = self.array[row0 + 1, col0]
        v11 = self.array[row0 + 1, col0 + 1]

        val = (
            v00 * (1 - fx) * (1 - fy)
            + v01 * fx * (1 - fy)
            + v10 * (1 - fx) * fy
            + v11 * fx * fy
        ).astype(float) + self.datum_offset_m

        bad = (
            ~in_bounds
            | (v00 == self.nodata)
            | (v01 == self.nodata)
            | (v10 == self.nodata)
            | (v11 == self.nodata)
        )
        val[bad] = np.nan
        return val.reshape(shape)
