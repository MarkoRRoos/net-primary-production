import os
import yaml
import rasterio
from pathlib import Path
import logging
import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)


def ensure_dir(path: Path) -> Path:
    """Ensure directory exists and return path.

    Args:
        path: Directory path to create.

    Returns:
        The same Path object (for chaining).
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def check_raster(path: Path, expected_crs: str = None) -> None:
    """Print basic raster metadata using logging instead of prints.

    Args:
        path: Path to raster.
        expected_crs: Optional expected CRS string. Logs a warning if mismatch.
    """
    try:
        with rasterio.open(path) as src:
            size_mb = path.stat().st_size / 1e6
            crs = src.crs.to_string() if src.crs else None
            bounds = src.bounds
            logger.info("%s — Size: %.1f MB | CRS: %s | Bounds: %s", path.name, size_mb, crs, bounds)
            if expected_crs and crs and crs != expected_crs:
                logger.warning("CRS mismatch for %s (expected %s)", path.name, expected_crs)
    except Exception as exc:
        logger.error("Failed to read %s: %s", path, exc)
def ensure_path_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")
    else:
        print(f"Directory already exists: {path}")


def load_config(config_path=None): 
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
        print(config_path)
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def summarize_dataarray(da: xr.DataArray) -> dict:
    """
    Summarize an xarray.DataArray with basic statistics.

    Args:
        da: xarray.DataArray

    Returns:
        Dictionary with min, max, mean, std, zero count, NaN count, total count.
    """
    values = da.values
    summary = {
        "min": float(np.nanmin(values)),
        "max": float(np.nanmax(values)),
        "mean": float(np.nanmean(values)),
        "std": float(np.nanstd(values)),
        "zero_count": int(np.sum(values == 0)),
        "nan_count": int(np.sum(np.isnan(values))),
        "total_count": values.size,
        "nonzero_count": int(np.sum(values != 0)),
    }
    return summary
