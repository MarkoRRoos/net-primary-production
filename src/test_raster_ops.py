# tests/test_raster_ops.py
from src.raster_ops import load_clip_reproject
import geopandas as gpd
from pathlib import Path

def test_load_clip_reproject(tmp_path):
    gdf = gpd.read_file("tests/data/test_boundary.shp")
    result = load_clip_reproject(Path("tests/data/test_raster.tif"), gdf, "EPSG:32631", 100)
    assert result.rio.crs.to_string() == "EPSG:32631"
    assert result.shape[0] > 0
