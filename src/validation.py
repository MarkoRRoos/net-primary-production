# src/validation.py
from pathlib import Path
import logging
import rasterio
import numpy as np
import xarray as xr
from typing import List, Optional

logger = logging.getLogger(__name__)


def validate_raster_crs(raster_path: Path, expected_crs: str) -> bool:
    """Check if raster CRS matches expected CRS.

    Args:
        raster_path: Path to raster file.
        expected_crs: CRS string to compare against (e.g., 'EPSG:32631').

    Returns:
        True if CRS matches, False otherwise.
    """
    try:
        with rasterio.open(raster_path) as src:
            raster_crs = src.crs.to_string() if src.crs else None
            if raster_crs != expected_crs:
                logger.warning("CRS mismatch for %s: found %s expected %s", raster_path.name, raster_crs, expected_crs)
                return False
            logger.info("CRS check passed for %s", raster_path.name)
            return True
    except Exception as e:
        logger.error("Failed to open raster %s for CRS check: %s", raster_path, e)
        return False


def validate_estk_classes(estk_path: Path, expected_classes: List[int]) -> bool:
    """Validate that ESTK raster contains all expected classes.

    Args:
        estk_path: Path to ESTK raster.
        expected_classes: List of integer classes expected to be present.

    Returns:
        True if all expected classes are found, False otherwise.
    """
    try:
        estk_da = xr.open_rasterio(estk_path).squeeze()
        present_classes = np.unique(estk_da.values[~np.isnan(estk_da.values)]).astype(int)
        missing_classes = [cls for cls in expected_classes if cls not in present_classes]
        if missing_classes:
            logger.warning("ESTK classes missing from %s: %s", estk_path.name, missing_classes)
            return False
        logger.info("All expected ESTK classes present in %s", estk_path.name)
        return True
    except Exception as e:
        logger.error("Error validating ESTK classes in %s: %s", estk_path, e)
        return False


def validate_data_range(raster_path: Path, min_val: Optional[float], max_val: Optional[float]) -> bool:
    """Check that raster values are within a given range.

    Args:
        raster_path: Path to raster file.
        min_val: Minimum expected value (inclusive) or None to skip.
        max_val: Maximum expected value (inclusive) or None to skip.

    Returns:
        True if all values are within range, False otherwise.
    """
    try:
        da = xr.open_rasterio(raster_path).squeeze()
        values = da.values
        if min_val is not None and np.nanmin(values) < min_val:
            logger.warning("Raster %s has values below minimum expected %s", raster_path.name, min_val)
            return False
        if max_val is not None and np.nanmax(values) > max_val:
            logger.warning("Raster %s has values above maximum expected %s", raster_path.name, max_val)
            return False
        logger.info("Data range validation passed for %s", raster_path.name)
        return True
    except Exception as e:
        logger.error("Failed to validate data range for %s: %s", raster_path, e)
        return False
