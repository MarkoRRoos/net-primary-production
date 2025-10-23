# src/estk_ops.py
from pathlib import Path
import logging
import rasterio
from rasterio.mask import mask
from shapely.geometry import mapping
import rioxarray as rxr
import geopandas as gpd
from .utils import ensure_dir

logger = logging.getLogger(__name__)


def clip_and_reproject_estk(
    estk_raster_path: Path,
    boundary_gdf: gpd.GeoDataFrame,
    clipped_raster_path: Path,
    reprojected_raster_path: Path,
    target_crs: str = "EPSG:32631",
) -> None:
    """Clip ESTK classification raster by a boundary polygon and reproject.

    Args:
        estk_raster_path: Path to the original ESTK raster file.
        boundary_gdf: GeoDataFrame containing polygon(s) for clipping.
        clipped_raster_path: Path where the clipped raster will be saved.
        reprojected_raster_path: Path where the reprojected raster will be saved.
        target_crs: Coordinate Reference System to reproject to (default EPSG:32631).

    Raises:
        FileNotFoundError: If ESTK raster does not exist.
        Exception: For rasterio or reprojection errors.
    """
    estk_raster_path = Path(estk_raster_path)
    clipped_raster_path = ensure_dir(Path(clipped_raster_path).parent) / clipped_raster_path.name
    reprojected_raster_path = ensure_dir(Path(reprojected_raster_path).parent) / reprojected_raster_path.name

    if not estk_raster_path.exists():
        raise FileNotFoundError(f"ESTK raster not found: {estk_raster_path}")

    logger.info("Opening ESTK raster: %s", estk_raster_path)
    with rasterio.open(estk_raster_path) as src:
        # Reproject boundary to raster CRS for masking
        boundary_in_raster_crs = boundary_gdf.to_crs(src.crs)
        geometries = [mapping(geom) for geom in boundary_in_raster_crs.geometry]

        logger.info("Masking ESTK raster with boundary polygon")
        clipped_data, clipped_transform = mask(src, geometries, crop=True)
        clipped_meta = src.meta.copy()
        clipped_meta.update({
            "height": clipped_data.shape[1],
            "width": clipped_data.shape[2],
            "transform": clipped_transform,
        })

        logger.info("Saving clipped ESTK raster to %s", clipped_raster_path)
        with rasterio.open(clipped_raster_path, "w", **clipped_meta) as dest:
            dest.write(clipped_data)

    # Reproject clipped raster to target CRS
    logger.info("Reprojecting clipped raster to %s", target_crs)
    estk_clipped_xr = rxr.open_rasterio(clipped_raster_path, masked=True)
    estk_reprojected = estk_clipped_xr.rio.reproject(target_crs)

    logger.info("Saving reprojected ESTK raster to %s", reprojected_raster_path)
    estk_reprojected.rio.to_raster(str(reprojected_raster_path))

    logger.info("Completed ESTK clipping and reprojection")
