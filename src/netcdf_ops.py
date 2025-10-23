# src/netcdf_ops.py
from pathlib import Path
from typing import Dict
import xarray as xr
import logging

logger = logging.getLogger(__name__)


def extract_monthly_mean(
    nc_path: Path,
    varname: str,
    bbox_coords: Dict[str, float],
    time_slice: Dict[str, str],
    out_crs: str,
    out_path: Path,
):
    """Extract a monthly mean (or mean over a time slice) and write a raster reprojected to `out_crs`.

    Args:
        nc_path: Path to netCDF file.
        varname: Variable name inside the netCDF.
        bbox_coords: Dict with lon_min/lon_max/lat_min/lat_max in WGS84.
        time_slice: Dict with 'start' and 'end' ISO dates.
        out_crs: Target CRS string (e.g., 'EPSG:32631').
        out_path: Output raster path.
    """
    ds = xr.open_dataset(nc_path)
    start = time_slice["start"]
    end = time_slice["end"]
    logger.info("Selecting time slice %s to %s from %s", start, end, nc_path)

    # some datasets use 'valid_time' coordinate
    if "valid_time" in ds.coords and "time" not in ds.coords:
        ds = ds.rename({"valid_time": "time"})

    ds_slice = ds.sel(time=slice(start, end))

    da = ds_slice[varname].mean(dim="time")

    # NOTE: many ERA5-like datasets have latitude descending — slicing must consider that
    da_clip = da.sel(
        latitude=slice(bbox_coords["lat_max"], bbox_coords["lat_min"]),
        longitude=slice(bbox_coords["lon_min"], bbox_coords["lon_max"]),
    )

    da_clip.rio.write_crs("EPSG:4326").rio.reproject(out_crs).rio.to_raster(str(out_path))
    logger.info("Wrote raster to %s", out_path)


def extract_ssrd_may_process(
    nc_path: Path,
    varname: str,
    bbox_coords: Dict[str, float],
    time_slice: Dict[str, str],
    out_crs: str,
    out_path: Path,
):
    """Specifically process SSRD: compute hourly diffs, daily sums, then mean over days, then crop.

    Kept as a separate function because of the diff/resample steps seen in original code.
    """
    ds = xr.open_dataset(nc_path)
    if "valid_time" in ds.coords and "time" not in ds.coords:
        ds = ds.rename({"valid_time": "time"})

    ssrd_raw = ds[varname]
    ssrd_sel = ssrd_raw.sel(time=slice(time_slice["start"], time_slice["end"]))
    ssrd_hourly = ssrd_sel.diff("time")
    ssrd_daily = ssrd_hourly.resample(time="1D").sum()
    ssrd_mean = ssrd_daily.mean(dim="time")

    ssrd_clip = ssrd_mean.sel(
        latitude=slice(bbox_coords["lat_max"], bbox_coords["lat_min"]),
        longitude=slice(bbox_coords["lon_min"], bbox_coords["lon_max"]),
    )
    ssrd_clip.rio.write_crs("EPSG:4326").rio.reproject(out_crs).rio.to_raster(str(out_path))
    logger.info("Processed and saved SSRD to %s", out_path)