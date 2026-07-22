
import os
import numpy as np
import xarray as xr
import geopandas as gpd
from shapely.geometry import MultiPoint
from skimage.morphology import remove_small_objects, binary_closing, disk
from scipy.ndimage import binary_fill_holes
from scipy.ndimage import label
import matplotlib.pyplot as plt

# path to the SWOT L2 HR PIXC (pixel cloud) NetCDF file
file = r"C:\Users\pmalesza\Documents\Python Codes\SWOT_L2_HR_PIXC_052_475_245R_20260706T065928_20260706T065939_PID0_01.nc"
output_base = r"C:\Users\pmalesza\Documents\SWOT_L2_HR_PIXC Output Polygons"

# base folder all outputs go into, plus a subfolder named after the numbers/tags
stem = os.path.splitext(os.path.basename(file))[0]
prefix = "SWOT_L2_HR_PIXC_"
name = stem[len(prefix):] if stem.startswith(prefix) else stem
output_dir = os.path.join(output_base, name)
os.makedirs(output_dir, exist_ok=True)

# open just the "pixel_cloud" group, which holds one row per detected pixel
data = xr.open_dataset(file, group="pixel_cloud")
mask = np.ones(data.longitude.values.shape, dtype=bool)

# each pixel's position in SWOT's native along-track (azimuth) / cross-track (range) grid
az = data.azimuth_index.values[mask].astype(int)
rg = data.range_index.values[mask].astype(int)

# per pixel classification code and its associated quality flags
classification = data.classification.values[mask]
classification_qual = data.classification_qual.values[mask]
interferogram_qual = data.interferogram_qual.values[mask]
sig0_qual = data.sig0_qual.values[mask]
geolocation_qual = data.geolocation_qual.values[mask]

# per pixel real world position
lat = data.latitude.values[mask]
lon = data.longitude.values[mask]

# which classification codes count as which category
water = {4,5,7}
intertidal = {2,3,6}

# boolean array: True where a pixel is confidently classified as land
land = (
    (classification == 1)
    & (classification_qual == 0)
)

# bounding box of the azimuth/range indices present in this data, used to size and offset the 2D grid below
az_min, az_max = az.min(), az.max()
rg_min, rg_max = rg.min(), rg.max()
n_az = az_max - az_min + 1
n_rg = rg_max - rg_min + 1

# scatter the flat land boolean array into its proper 2D azimuth/range grid position
land_grid = np.zeros((n_az,n_rg),bool)
land_grid[az-az_min,rg-rg_min]=land

# marks every grid cell that actually has a pixel in it at all
populated = np.zeros((n_az, n_rg), dtype=bool)
populated[az - az_min, rg - rg_min] = True

# clean up the land mask
BW = land_grid | ~populated
se = disk(2)
BW1 = remove_small_objects(BW, min_size=50, connectivity=2)
BW2 = binary_closing(BW1, se)
BW3 = binary_fill_holes(BW2)

# inverse of the cleaned land mask
not_land_clean = ~BW3

# a pixel only counts as "good quality" if all four flags read 0
quality = (
    (classification_qual == 0)
    & (interferogram_qual == 0)
    & (sig0_qual == 0)
    & (geolocation_qual == 0)
)

# scatter the raw classification codes into the same 2D grid layout
classification_grid = np.zeros((n_az, n_rg), dtype=np.uint8)
classification_grid[az - az_min, rg - rg_min] = classification

# scatter the quality ok flag into the same 2D grid layout
quality_grid = np.zeros((n_az, n_rg), dtype=bool)
quality_grid[az - az_min, rg - rg_min] = quality

# a cell survives only if: it has data, passed quality checks, isn't inside the cleaned solid land region, and is a class we want to keep
final_mask = populated & quality_grid & not_land_clean & np.isin(classification_grid,list(water|intertidal))

# grids that store each cell's real longitude/latitude (NaN where no pixel exists)
lon_grid = np.full((n_az, n_rg), np.nan)
lat_grid = np.full((n_az, n_rg), np.nan)
lon_grid[az - az_min, rg - rg_min] = lon
lat_grid[az - az_min, rg - rg_min] = lat


# create polygons from connected water/intertidal regions
def export(category, codes):

    # pixels belonging to this category
    category_mask = final_mask & np.isin(classification_grid, list(codes))

    if not np.any(category_mask):
        print(f"No {category} pixels found.")
        return

    # identify connected groups of pixels
    labelled_array, num_features = label(
        category_mask,
        structure=np.ones((3,3))
    )

    records = []

    print(f"{category}: found {num_features} connected regions")

    for region_id in range(1, num_features + 1):

        region_mask = labelled_array == region_id

        xs = lon_grid[region_mask]
        ys = lat_grid[region_mask]

        # remove NaN values
        valid = ~np.isnan(xs) & ~np.isnan(ys)
        xs = xs[valid]
        ys = ys[valid]

        # ignore tiny regions
        if len(xs) < 3:
            continue

        # create convex hull around this individual region
        polygon = MultiPoint(
            list(zip(xs, ys))
        ).convex_hull

        centroid = polygon.centroid

        records.append(
            {
                "category": category,
                "region_id": region_id,
                "num_points": len(xs),
                "area": polygon.area,
                "cent_lon": centroid.x,
                "cent_lat": centroid.y,
                "geometry": polygon,
            }
        )

    # convert polygons into GeoDataFrame
    polygon_gdf = gpd.GeoDataFrame(
        records,
        crs="EPSG:4326"
    )

    # save shapefile using your original naming style
    polygon_gdf.to_file(
        os.path.join(output_dir, f"{category}_polygon.shp")
    )

    print(
        f"Wrote {len(polygon_gdf)} {category} polygons to "
        f"{os.path.join(output_dir, f'{category}_polygon.shp')}"
    )

    return polygon_gdf


# create water and intertidal polygons
water_gdf = export("water", water)
intertidal_gdf = export("intertidal", intertidal)


# print polygon information
print("\nWater polygon shapes:")
print(water_gdf.geometry.apply(lambda x: x.geom_type))

print("\nIntertidal polygon shapes:")
print(intertidal_gdf.geometry.apply(lambda x: x.geom_type))


print(f"\nNumber of water polygons: {len(water_gdf)}")
print(f"Number of intertidal polygons: {len(intertidal_gdf)}")


# plot output polygons for checking

fig, ax = plt.subplots(figsize=(10, 10))

if water_gdf is not None:
    water_gdf.boundary.plot(
        ax=ax,
        color="blue",
        linewidth=1.5,
        label="Water"
    )

if intertidal_gdf is not None:
    intertidal_gdf.boundary.plot(
        ax=ax,
        color="red",
        linewidth=1.5,
        label="Intertidal"
    )


ax.set_title("SWOT PIXC Water and Intertidal Polygons")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.legend()

plt.show()

