import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import xarray as xr
from skimage.morphology import remove_small_objects, binary_closing, disk
from scipy.ndimage import binary_fill_holes

file = r"C:\Users\pmalesza\Documents\Python Codes\SWOT_L2_HR_PIXC_052_475_245R_20260706T065928_20260706T065939_PID0_01.nc"

data = xr.open_dataset(file, group="pixel_cloud")

mask = np.ones(data.longitude.values.shape, dtype=bool)

az_sub = data.azimuth_index.values[mask].astype(int)
rg_sub = data.range_index.values[mask].astype(int)

classification_sub = data.classification.values[mask]
classification_qual_sub = data.classification_qual.values[mask]
interferogram_qual_sub = data.interferogram_qual.values[mask]
sig0_qual_sub = data.sig0_qual.values[mask]

land_sub = (
    (classification_sub == 1)
    #& (classification_sub == 7)
    #& (classification_sub == 6)
    & (classification_qual_sub == 0)
    #& (interferogram_qual_sub == 0)
    #& (sig0_qual_sub == 0)
)

az_min, az_max = az_sub.min(), az_sub.max()
rg_min, rg_max = rg_sub.min(), rg_sub.max()
n_az = az_max - az_min + 1
n_rg = rg_max - rg_min + 1

land_grid = np.zeros((n_az, n_rg), dtype=bool)
land_grid[az_sub - az_min, rg_sub - rg_min] = land_sub

populated = np.zeros((n_az, n_rg), dtype=bool)
populated[az_sub - az_min, rg_sub - rg_min] = True

BW = land_grid | ~populated
se = disk(2)

BW1 = remove_small_objects(BW, min_size=50, connectivity=2)
BW2 = binary_closing(BW1, se)
BW3 = binary_fill_holes(BW2)

fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
titles = [
    "Initial Land Mask",
    "After bwareaopen",
    "After bwareaopen + imclose",
    "After bwareaopen + imclose + imfill",
]
for ax, img, title in zip(axes.ravel(), [BW, BW1, BW2, BW3], titles):
    ax.imshow(img, origin="lower", aspect="auto")
    ax.set_title(title)
plt.tight_layout()
plt.show()

land_mask_clean = BW3
water_grid = ~BW3
water_mask = water_grid[az_sub - az_min, rg_sub - rg_min]

lat_sub = data.latitude.values[mask]
lon_sub = data.longitude.values[mask]
height_sub = data.height.values[mask]
geolocation_qual_sub = data.geolocation_qual.values[mask]
coherent_power_sub = data.coherent_power.values[mask]  
sig0_sub = data.sig0.values[mask]

valid = (
    water_mask
    & np.isfinite(lat_sub)
    & np.isfinite(lon_sub)
    & np.isfinite(height_sub)
    & (np.abs(height_sub) < 1e4)
    & np.isfinite(classification_sub)
    & np.isfinite(coherent_power_sub)
    & (coherent_power_sub > 0)
    & np.isfinite(sig0_sub)
    & (sig0_sub > 0)
    & (geolocation_qual_sub == 0)
    & (classification_qual_sub == 0)
    & (interferogram_qual_sub == 0)
    & (sig0_qual_sub == 0)
    & (classification_sub != 7)
    & (classification_sub != 6)
)

classification_final = classification_sub[valid]
class_lon = lon_sub[valid]
class_lat = lat_sub[valid]

class_labels = [
    "Land",
    "Land near water",
    "Water near land",
    "Open water",
    "Dark water",
    "Low coherence water near land",
    "Open low coherence water",
]

norm = mcolors.BoundaryNorm(np.arange(0.5, 8.5, 1), 7)

plt.figure(figsize=(8, 8))

sc = plt.scatter(
    class_lon,
    class_lat,
    c=classification_final,
    s=2,
    cmap="tab10",
    norm=norm,
)

cbar = plt.colorbar(sc, ticks=np.arange(1, 8))
cbar.set_label("Classification")
cbar.set_ticklabels(class_labels)

plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("SWOT Classification (Mask)")

plt.show()

