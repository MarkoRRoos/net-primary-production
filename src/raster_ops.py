# src/raster_ops.py
from pathlib import Path
from typing import List, Optional
import geopandas as gpd
import xarray as xr
import rioxarray as rxr
from rioxarray.merge import merge_arrays
import logging
import numpy as np
from .utils import ensure_dir


logger = logging.getLogger(__name__)


def load_clip_reproject(
    raster_path: Path,
    boundary: gpd.GeoDataFrame,
    target_crs: str,
    scale_factor: float = 1.0,
    mask_nan: bool = True,
) -> xr.DataArray:
    """Load a raster, clip to the provided boundary, reproject to target CRS, and scale.

    Args:
        raster_path: Path to input raster.
        boundary: GeoDataFrame with geometry to clip.
        target_crs: CRS string to reproject to (e.g., "EPSG:32631").
        scale_factor: Factor by which to divide values (default 1.0).
        mask_nan: If True, keep mask applied.

    Returns:
        xarray.DataArray: clipped and reprojected raster.
    """
    raster_path = Path(raster_path)
    logger.info("Loading raster %s", raster_path)
    da = rxr.open_rasterio(raster_path, masked=True).squeeze()

    # Ensure boundary CRS matches raster CRS before clipping
    if boundary.crs != da.rio.crs:
        logger.debug("Reprojecting boundary from %s to %s", boundary.crs, da.rio.crs)
        boundary = boundary.to_crs(da.rio.crs)
        boundary = boundary.set_crs(da.rio.crs, allow_override=True)

    clipped = da.rio.clip(boundary.geometry, boundary.crs)
    logger.info("Reprojecting clipped raster to %s", target_crs)
    reprojected = clipped.rio.reproject(target_crs)

    if scale_factor != 1.0:
        reprojected = reprojected / scale_factor

    if mask_nan:
        # rioxarray keeps masks — ensure masked values are NaN for downstream arithmetic
        reprojected = reprojected.where(~reprojected.isnull(), other=xr.DataArray(xr.full_like(reprojected, float('nan'))))
    # print("Raster bounds:", da.rio.bounds())
    # print("Boundary bounds:", boundary.total_bounds)
    # print("Clipped min/max:", float(clipped.min().values), float(clipped.max().values))
    # print("Unique values before mask_nan:", np.unique(reprojected.values[~np.isnan(reprojected.values)][:100]))
    return reprojected


def merge_tiles(
    tile_paths: List[Path],
    boundary: gpd.GeoDataFrame,
    target_crs: str,
    scale_factor: float = 1.0,
    out_path: Optional[Path] = None,
) -> xr.DataArray:
    """Merge multiple raster tiles into a single raster over `boundary`.

    Args:
        tile_paths: List of file paths to tiles (order not important).
        boundary: GeoDataFrame to clip to.
        target_crs: Target CRS for final raster.
        scale_factor: Divide values by this factor (e.g., 10000).
        out_path: If provided, save merged raster to this path.

    Returns:
        xarray.DataArray: merged raster.
    """
    processed = [
        load_clip_reproject(Path(p), boundary, target_crs, scale_factor)
        for p in tile_paths
    ]

    merged = merge_arrays(
    processed,
    bounds=tuple(boundary.total_bounds)
    )
    # print("Merged bounds:", merged.rio.bounds())
    # print("Merged min/max:", float(merged.min().values), float(merged.max().values))
    merged = merged.fillna(0) #FIXME: fill NaNs with 0 > don't know if this is the correct solution so check. 

    if out_path:
        out_path = ensure_dir(Path(out_path).parent) / Path(out_path).name
        logger.info("Saving merged raster to %s", out_path)
        merged.rio.to_raster(str(out_path))

    # print("Merged stats:", float(merged.min()), float(merged.max()))
    # print("Unique sample:", np.unique(merged.values[~np.isnan(merged.values)])[0:20])
    # print("NoData check:", np.isnan(merged.values).sum(), "NaNs")
    # print("Zero count:", np.sum(merged.values == 0))

    return merged


def read_raster_match(raster_path: Path, match_da: xr.DataArray) -> xr.DataArray:
    """Open a raster and reproject/resample to match coordinates of `match_da`.

    Args:
        raster_path: Path to raster file.
        match_da: DataArray whose grid should be matched.

    Returns:
        DataArray aligned with `match_da`.
    """
    da = rxr.open_rasterio(raster_path, masked=True).squeeze()
    da_aligned = da.rio.reproject_match(match_da)
    return da_aligned
