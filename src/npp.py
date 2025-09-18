# src/raster_ops.py
from pathlib import Path
from typing import Optional
import numpy as np
import xarray as xr
import rioxarray as rxr
import pandas as pd
import logging
from .utils import ensure_dir
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def compute_npp(
    lai_path: Path,
    fapar_path: Path,
    temp_path: Path,
    ssrd_path: Path,
    estk_path: Path,
    conversion_df: pd.DataFrame,
    output_path: Path,
    plot_estk: bool = True,  
) -> xr.DataArray:
    """Calculate Net Primary Productivity (NPP) raster based on input datasets.

    This implementation follows your project's formula and steps:
    - Loads LAI, FAPAR, Temperature, SSRD, and ESTK rasters.
    - Converts temperature from Kelvin to Celsius if needed.
    - Aligns all rasters to LAI spatial grid.
    - For each valid ESTK class, computes class-specific NPP using:
        NPP = eps_max * FAPAR * SSRD * temp_factor
      where temp_factor depends on temperature thresholds (T_min, T_max).
    - Combines results into a single output raster.

    Args:
        lai_path: Path to LAI raster (already scaled physically).
        fapar_path: Path to FAPAR raster (already scaled physically).
        temp_path: Path to temperature raster (Kelvin expected).
        ssrd_path: Path to SSRD raster.
        estk_path: Path to ESTK classification raster.
        conversion_df: pandas DataFrame indexed by ESTK class values,
            with columns ["eps_max", "T_min", "T_max"].
        output_path: Path to save the computed NPP GeoTIFF.

    Returns:
        xarray.DataArray: The computed NPP raster.

    Raises:
        FileNotFoundError: If any input raster is missing.
        KeyError: If ESTK class missing in conversion table.
        Exception: For computation errors.
    """
    logger.info("Loading rasters for NPP computation")
    lai = rxr.open_rasterio(lai_path, masked=True).squeeze()
    fapar = rxr.open_rasterio(fapar_path, masked=True).squeeze()
    temp = rxr.open_rasterio(temp_path, masked=True).squeeze()
    ssrd = rxr.open_rasterio(ssrd_path, masked=True).squeeze()
    estk = rxr.open_rasterio(estk_path, masked=True).squeeze()

    # Convert temp from Kelvin to Celsius if median value > 100
    median_temp = float(np.nanmedian(temp.values))
    if median_temp > 100:
        logger.info("Converting temperature from Kelvin to Celsius")
        temp = temp - 273.15

    # Align all rasters to LAI grid
    logger.info("Aligning rasters to LAI grid")
    temp = temp.rio.reproject_match(lai)
    # print("Temperature min/max na align:", float(temp.min()), float(temp.max()))
    ssrd = ssrd.rio.reproject_match(lai)
    fapar = fapar.rio.reproject_match(lai)
    estk = estk.rio.reproject_match(lai)

     # --- NEW: Plot ESTK raster and class masks ---
    if plot_estk:
        # Identify ESTK classes actually present and valid
        estk_classes = np.unique(estk.values[~np.isnan(estk.values)]).astype(int)
        valid_classes = [cls for cls in estk_classes if cls in conversion_df.index]

        if not valid_classes:
            logger.warning("No valid ESTK classes found for plotting.")
        else:
            # Plot the ESTK raster showing only valid classes
            plt.figure(figsize=(8, 6))
            estk_masked = estk.where(np.isin(estk, valid_classes))
            estk_masked.plot(cmap="tab20")
            plt.title("ESTK Classes (Valid Only)")
            plt.show()

    # Prepare NPP output array with NaNs
    npp_result = xr.full_like(lai, np.nan)

    # Identify ESTK classes present and valid per conversion table
    estk_classes = np.unique(estk.values[~np.isnan(estk.values)]).astype(int)
    valid_classes = [cls for cls in estk_classes if cls in conversion_df.index]

    logger.info("Processing %d valid ESTK classes for NPP", len(valid_classes))

    for cls in valid_classes:
        logger.debug("Processing ESTK class %d", cls)
        mask = (estk == cls)

        # Skip if class absent in raster (just in case)
        if mask.sum() == 0:
            logger.warning("ESTK class %d present but masked out, skipping", cls)
            continue

        params = conversion_df.loc[cls]
        eps_max = float(params["eps_max"])
        t_min = float(params["T_min"])
        t_max = float(params["T_max"])
        fake_max = 50 #TODO: remove this, just for testing

        temp_masked = temp.where(mask)
        fapar_masked = fapar.where(mask)
        ssrd_masked = ssrd.where(mask)

        # Temperature factor computation (bounded by T_min and T_max)
        #temp_factor = ((temp_masked - t_min) * (t_max - temp_masked)) / ((t_max - t_min) ** 2) #The problem must occur here
        temp_factor = ((temp_masked - t_min) * (fake_max - temp_masked)) / ((t_max - t_min) ** 2) 
        # print("temp min & max",temp_masked.min(), temp_masked.max())
        temp_factor = temp_factor.clip(min=0)
        # print(f"Temp min/max: {float(temp.min())}, {float(temp.max())}")
        # print(f"T_min/T_max from conversion table: {t_min}, {t_max}")
        # print(f"Temp factor min/max: {float(temp_factor.min())}, {float(temp_factor.max())}")


        # Compute GPP and then NPP
        gpp = eps_max * fapar_masked * ssrd_masked #TODO: add LAI and full formula
        npp_class = gpp * temp_factor

        # Insert class-specific NPP values into result raster
        npp_result = npp_result.where(~mask, npp_class)

    # Write output raster with CRS
    output_path = ensure_dir(output_path.parent) / output_path.name
    npp_result.rio.write_crs(lai.rio.crs, inplace=True)
    npp_result.rio.to_raster(str(output_path))
    logger.info("NPP raster saved to %s", output_path)

    return npp_result