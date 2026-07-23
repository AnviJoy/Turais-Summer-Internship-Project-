
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import netCDF4 as nc
from skimage.morphology import remove_small_objects, binary_closing, disk
from scipy.ndimage import binary_fill_holes
from netCDF4 import Dataset
import matplotlib.colors as mcolors

data = xr.open_dataset(
    r"\Users\pmalesza\Documents\Python Codes\SWOT_L2_HR_PIXC_052_475_245R_20260706T065928_20260706T065939_PID0_01.nc",
    group="pixel_cloud"
)

geom = data.geometry.iloc[0]

if geom.geom_type == "LineString":
    poly = Polygon(geom.coords)
else:
    poly = geom

lon = data.longitude.values
lat = data.latitude.values

mask = np.array([
    poly.covers(Point(lon, lat))
    for lon, lat in zip(
        data.longitude.values,
        data.latitude.values
    )
])

az = data.azimuth_index.values.astype(int)
rg = data.range_index.values.astype(int)

land_class = (
    mask
    & (data.classification.values == 1)
    & (data.classification_qual.values == 0)
    & (data.interferogram_qual.values == 0)
    & (data.sig0_qual.values == 0)
)

az_sub = az[mask]
rg_sub = rg[mask]
land_sub = land_class[mask]

az_min, az_max = az_sub.min(), az_sub.max()
rg_min, rg_max = rg_sub.min(), rg_sub.max()
n_az = az_max - az_min + 1
n_rg = rg_max - rg_min + 1

#land_grid = np.ones((az_max - az_min + 1, rg_max - rg_min + 1), dtype=int)
#land_grid[az_sub - az_min, rg_sub - rg_min] = 0

land_grid = np.zeros((n_az, n_rg), dtype=bool)
land_grid[az_sub - az_min, rg_sub - rg_min] = land_sub

populated = np.zeros((n_az, n_rg), dtype=bool)
populated[az_sub - az_min, rg_sub - rg_min] = True

#populated = np.zeros((n_az, n_rg), dtype=int)
#populated[az_sub - az_min, rg_sub - rg_min] = True

BW = land_grid | ~populated   
se = disk(2)

BW1 = remove_small_objects(BW, min_size=50, connectivity=2)
BW2 = binary_closing(BW1, se)
BW3 = binary_fill_holes(BW2)

land_mask_clean = BW3

water_grid = ~BW3
water_mask = water_grid[az_sub - az_min, rg_sub - rg_min]

file = r"\Users\pmalesza\Documents\Python Codes\SWOT_L2_HR_PIXC_052_475_245R_20260706T065928_20260706T065939_PID0_01.nc"

with Dataset(file, "r") as nc:

    pixc = nc.groups["pixel_cloud"]
    tvp = nc.groups["tvp"]

    lat = pixc.variables["latitude"][:][mask]
    lon = pixc.variables["longitude"][:][mask]
    height = pixc.variables["height"][:][mask]
    classification = pixc.variables["classification"][:][mask]
    geolocation_qual = pixc.variables["geolocation_qual"][:][mask]

classification = data.classification.values[mask]
longitude = data.longitude.values[mask]
latitude = data.latitude.values[mask]

valid = (
    water_mask
    & np.isfinite(lat)
    & np.isfinite(lon)
    & np.isfinite(height)
    & (np.abs(height) < 1e4)
    & np.isfinite(classification)
    & np.isfinite(coherent_power) 
    & (coherent_power > 0)
    & np.isfinite(sig0) 
    & (sig0 > 0)
    & (geolocation_qual == 0)   
)

classification = classification[valid]
class_lon = longitude[valid]
class_lat = latitude[valid]

class_labels = [
    "Land",
    "Land near water",
    "Water near land",
    "Open water",
    "Dark water",
    "Low coherence water near land",
    "Open low coherence water"
]

norm = mcolors.BoundaryNorm(
    np.arange(0.5, 8.5, 1),
    7
)

plt.figure(figsize=(8, 8))

sc = plt.scatter(
    class_lon,
    class_lat,
    c=classification,
    s=2,
    cmap="tab10",
    norm=norm
)

cbar = plt.colorbar(sc, ticks=np.arange(1, 8))
cbar.set_label("Classification")
cbar.set_ticklabels(class_labels)

plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("SWOT Classification (Mask)")
