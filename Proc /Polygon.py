import numpy as np
import xarray as xr
from skimage.morphology import remove_small_objects, binary_closing, disk
from scipy.ndimage import binary_fill_holes
from rasterio.features import shapes
from shapely.geometry import shape
import geopandas as gpd

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
    & (classification_qual_sub == 0)
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

land_mask_clean = BW3
water_grid = ~BW3
water_mask = water_grid[az_sub - az_min, rg_sub - rg_min]

lat_sub = data.latitude.values[mask]
lon_sub = data.longitude.values[mask]

class_labels = [
    "Land",
    "Land near water",
    "Water near land",
    "Open water",
    "Dark water",
    "Low coherence water near land",
    "Open low coherence water",
]

quality_ok_sub = (
    (classification_qual_sub == 0)
    & (interferogram_qual_sub == 0)
    & (sig0_qual_sub == 0)
)

classification_grid = np.zeros((n_az, n_rg), dtype=np.uint8)
classification_grid[az_sub - az_min, rg_sub - rg_min] = classification_sub

quality_grid = np.zeros((n_az, n_rg), dtype=bool)
quality_grid[az_sub - az_min, rg_sub - rg_min] = quality_ok_sub

land_classes = {1, 2}
water_classes = {3, 4, 5, 6, 7}

class_region_ok = np.where(
    np.isin(classification_grid, list(land_classes)), land_mask_clean,
    np.where(np.isin(classification_grid, list(water_classes)), water_grid, False),
)

final_mask = populated & quality_grid & class_region_ok
final_grid = np.where(final_mask, classification_grid, 0).astype(np.uint8)

lon_grid = np.full((n_az, n_rg), np.nan)
lat_grid = np.full((n_az, n_rg), np.nan)
lon_grid[az_sub - az_min, rg_sub - rg_min] = lon_sub
lat_grid[az_sub - az_min, rg_sub - rg_min] = lat_sub

def rc_to_lonlat(row, col):
    r = int(round(np.clip(row, 0, n_az - 1)))
    c = int(round(np.clip(col, 0, n_rg - 1)))
    return lon_grid[r, c], lat_grid[r, c]

records = []
for geom, val in shapes(final_grid, mask=final_grid > 0, connectivity=8):
    val = int(val)
    if val == 0:
        continue
    new_coords = []
    for ring in geom["coordinates"]:
        new_ring = [rc_to_lonlat(y, x) for x, y in ring]
        new_coords.append(new_ring)
    poly = shape({"type": geom["type"], "coordinates": new_coords})
    records.append({"classification": val, "geometry": poly})

label_map = dict(enumerate(class_labels, start=1))
category_map = {c: "land" for c in land_classes}
category_map.update({c: "water" for c in water_classes})

gdf_by_class = gpd.GeoDataFrame(records, crs="EPSG:4326")
gdf_by_class["class_label"] = gdf_by_class["classification"].map(label_map)
gdf_by_class["category"] = gdf_by_class["classification"].map(category_map)

gdf_by_category = gdf_by_class.dissolve(by="category", as_index=False)
gdf_by_category = gdf_by_category[["category", "geometry"]]

gdf_by_class.to_file("classified_polygons_by_class.geojson", driver="GeoJSON")
gdf_by_class.to_file("classified_polygons_by_class.shp")

gdf_by_category.to_file("classified_polygons_by_category.geojson", driver="GeoJSON")
gdf_by_category.to_file("classified_polygons_by_category.shp")

print(f"Wrote {len(gdf_by_class)} per-class polygons and {len(gdf_by_category)} category polygons.")
