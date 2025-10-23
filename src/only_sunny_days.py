import os
import glob
import rasterio
import numpy as np
import re
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
#TODO: clean up and integrate

fapar_files = sorted(glob.glob("data/LAI_FAPAR/FAPAR_T32UMT_????-??-??.tiff"))
date_pattern = re.compile(r"FAPAR_T\d+[A-Z]+_(\d{4}-\d{2}-\d{2})\.tiff")
dates = []

arrays = []
meta = None

for f in tqdm(fapar_files, desc="Loading FAPAR stack"): #CHANGE
    with rasterio.open(f) as src:
        arr = src.read(1).astype(float)
        if meta is None:
            meta = src.meta.copy()
        nodata = src.nodata
        if nodata is not None:
            arr[arr == nodata] = np.nan
    arrays.append(arr)
    dates.append(date_pattern.search(os.path.basename(f)).group(1))

arrays = np.stack(arrays)   # shape: (time, H, W)
dates = np.array(dates)

pixel_means = np.nanmean(arrays, axis=0)

filled_arrays = arrays.copy()
for t in range(filled_arrays.shape[0]):
    mask = np.isnan(filled_arrays[t])
    filled_arrays[t][mask] = pixel_means[mask]


out_dir = "data/LAI_FAPAR_temporal_filled"
os.makedirs(out_dir, exist_ok=True)

for f, date, arr in zip(fapar_files, dates, filled_arrays):
    out_path = os.path.join(out_dir, f"FAPAR_T32UMT_{date}_filled_temporal.tiff") #CHANGE
    with rasterio.open(out_path, "w", **meta) as dst:
        dst.write(arr, 1)

stats = []
for t, date in enumerate(dates):
    orig = arrays[t]
    filled = filled_arrays[t]

    nan_count_orig = np.isnan(orig).sum()
    nan_count_filled = np.isnan(filled).sum()

    stats.append({
        "date": date,
        "nan_count_orig": nan_count_orig,
        "nan_count_filled": nan_count_filled,
        "mean_orig": np.nanmean(orig),
        "mean_filled": np.nanmean(filled)
    })

df_stats = pd.DataFrame(stats).sort_values("date")
df_stats.to_csv("fapar_temporal_fill_stats.csv", index=False)

