"""
Scan the MuST-C dataset to determine the max safe date count that has acceptable multi-spectral raster data,
since not all dates have the correct sensor data. This is then fed into the training code as "num_timesteps" config parameter.
"""
import os, glob
from datetime import datetime

def parse_date(folder_name):
    base = folder_name.split("-")[0][:6]
    try:
        return datetime.strptime(base, "%y%m%d")
    except:
        return datetime.min

plots_dir = "mustc_plots"
counts = []

for plot_folder in sorted(os.listdir(plots_dir)):
    pid = plot_folder.replace("plot_", "")
    inner = os.path.join(plots_dir, plot_folder, "plot-wise", f"plot{pid}")
    if not os.path.isdir(inner):
        continue
    date_folders = sorted(os.listdir(inner), key=parse_date)
    ms_dates = []
    for df in date_folders:
        ms_dir = os.path.join(inner, df, "raster_data", "UAV3-MS")
        if os.path.isdir(ms_dir) and glob.glob(os.path.join(ms_dir, "*.tif")):
            ms_dates.append(df)
    counts.append((pid, len(ms_dates), ms_dates))
    print(f"plot_{pid}: {len(ms_dates)} MS dates → {[d[:6] for d in ms_dates]}")

all_counts = [c[1] for c in counts]
print(f"\nMin: {min(all_counts)}  Max: {max(all_counts)}  "
      f"Mean: {sum(all_counts)/len(all_counts):.1f}")