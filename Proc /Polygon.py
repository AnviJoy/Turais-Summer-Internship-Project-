import numpy as np
import xarray as xr
from skimage.morphology import remove_small_objects, binary_closing, disk
from scipy.ndimage import binary_fill_holes
from rasterio.features import shapes
from shapely.geometry import shape
import geopandas as gpd

# path to the SWOT L2 HR PIXC (pixel cloud) NetCDF file
file = r"C:\Users\pmalesza\Documents\Python Codes\SWOT_L2_HR_PIXC_052_475_245R_20260706T065928_20260706T065939_PID0_01.nc"

# open just the "pixel_cloud" group, which holds one row per detected pixel
data = xr.open_dataset(file, group="pixel_cloud")
mask = np.ones(data.longitude.values.shape, dtype=bool)

# each pixel's position in SWOT's native along-track (azimuth) / cross-track (range) grid
az_sub = data.azimuth_index.values[mask].astype(int)
rg_sub = data.range_index.values[mask].astype(int)

# per pixel classification code and its associated quality flags
classification_sub = data.classification.values[mask]
classification_qual_sub = data.classification_qual.values[mask]
interferogram_qual_sub = data.interferogram_qual.values[mask]
sig0_qual_sub = data.sig0_qual.values[mask]
geolocation_qual_sub = data.geolocation_qual.values[mask]

# per pixel real world position
lat_sub = data.latitude.values[mask]
lon_sub = data.longitude.values[mask]

# which classification codes count as which category
water = {4, 5, 7}         
intertidal = {2, 3, 6}    
land = {1}                

# classification code and their respective labels 
class_label_map = {
    1: "Land",
    2: "Land near water",
    3: "Water near land",
    4: "Open water",
    5: "Dark water",
    6: "Low coherence water near land",
    7: "Open low coherence water",
}

# classification code assigned to broad category ("water" / "intertidal")
category_map = {c: "water" for c in water}
category_map.update({c: "intertidal" for c in intertidal})

# boolean array: True where a pixel is confidently classified as land
land_sub = (
    (classification_sub == 1)
    & (classification_qual_sub == 0)
)

# bounding box of the azimuth/range indices present in this data, used to size and offset the 2D grid below
az_min, az_max = az_sub.min(), az_sub.max()
rg_min, rg_max = rg_sub.min(), rg_sub.max()
n_az = az_max - az_min + 1
n_rg = rg_max - rg_min + 1

# scatter the flat land boolean array into its proper 2D azimuth/range grid position
land_grid = np.zeros((n_az, n_rg), dtype=bool)
land_grid[az_sub - az_min, rg_sub - rg_min] = land_sub

# marks every grid cell that actually has a pixel in it at all
populated = np.zeros((n_az, n_rg), dtype=bool)
populated[az_sub - az_min, rg_sub - rg_min] = True

# clean up the land mask 
BW = land_grid | ~populated
se = disk(2)
BW1 = remove_small_objects(BW, min_size=50, connectivity=2)  
BW2 = binary_closing(BW1, se)                                
BW3 = binary_fill_holes(BW2)                              

# inverse of the cleaned land mask
not_land_clean = ~BW3

# a pixel only counts as "good quality" if all four flags read 0
quality_ok_sub = (
    (classification_qual_sub == 0)
    & (interferogram_qual_sub == 0)
    & (sig0_qual_sub == 0)
    & (geolocation_qual_sub == 0)
)

# scatter the raw classification codes into the same 2D grid layout
classification_grid = np.zeros((n_az, n_rg), dtype=np.uint8)
classification_grid[az_sub - az_min, rg_sub - rg_min] = classification_sub

# scatter the quality ok flag into the same 2D grid layout
quality_grid = np.zeros((n_az, n_rg), dtype=bool)
quality_grid[az_sub - az_min, rg_sub - rg_min] = quality_ok_sub

# which grid cells hold a classification code we actually want to keep (water or intertidal)
keep_classes = water | intertidal
is_keep_class = np.isin(classification_grid, list(keep_classes))

# a cell survives only if; it has data, passed quality checks, isn't inside the cleaned solid land region, and is a class we want to keep
final_mask = populated & quality_grid & not_land_clean & is_keep_class

# zero out everything that didn't pass final_mask, keep the classification code elsewhere
final_grid = np.where(final_mask, classification_grid, 0).astype(np.uint8)

# grids that store each cell's real longitude/latitude (NaN where no pixel exists), used to translate pixel-grid coordinates back to real-world coordinates later
lon_grid = np.full((n_az, n_rg), np.nan)
lat_grid = np.full((n_az, n_rg), np.nan)
lon_grid[az_sub - az_min, rg_sub - rg_min] = lon_sub
lat_grid[az_sub - az_min, rg_sub - rg_min] = lat_sub


def rc_to_lonlat(row, col):
'''
clip to valid index range (shapes() can return a corner one past the last row/col) then round to the nearest actual cell before looking up its real coordinates
'''
    r = int(round(np.clip(row, 0, n_az - 1)))
    c = int(round(np.clip(col, 0, n_rg - 1)))
    return lon_grid[r, c], lat_grid[r, c]


records = []
# shapes() traces the outline of every contiguous same-valued region in final_grid, yielding (geometry, value) pairs; connectivity=8 counts diagonal neighbors as connected
for geom, val in shapes(final_grid, mask=final_grid > 0, connectivity=8):
    val = int(val)
    if val == 0:
        # skip the background region (everywhere final_mask was False)
        continue
    new_coords = []
    for ring in geom["coordinates"]:
        # geom's rings are lists of (x, y) = (col, row) vertex pairs (image convention);  rc_to_lonlat expects (row, col), hence the swap back to (y, x) here
        new_ring = [rc_to_lonlat(y, x) for x, y in ring]
        new_coords.append(new_ring)
    # rebuild a real shapely polygon, now in (lon, lat) coordinates instead of pixel space
    poly = shape({"type": geom["type"], "coordinates": new_coords})
    centroid = poly.centroid
    # one dict per polygon = one future row in the output shapefile's attribute table
    records.append(
        {
            "classif": val,
            "class_lbl": class_label_map[val],
            "category": category_map[val],
            "cent_lon": centroid.x,
            "cent_lat": centroid.y,
            "geometry": poly,
        }
    )

# turn the list of records into a geospatial table; EPSG:4326 = standard WGS84 lat/lon
gdf = gpd.GeoDataFrame(records, crs="EPSG:4326")

# write out as a shapefile (creates .shp, .shx, .dbf, .prj alongside it)
gdf.to_file("water_intertidal_polygons.shp")

print(f"Wrote {len(gdf)} polygons "
      f"({(gdf['category'] == 'water').sum()} water, "
      f"{(gdf['category'] == 'intertidal').sum()} intertidal).")


