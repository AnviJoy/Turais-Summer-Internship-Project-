import os
import numpy as np
from requests import codes
import xarray as xr
import geopandas as gpd
from shapely.geometry import MultiPoint
from skimage.morphology import remove_small_objects, binary_closing, disk
from scipy.ndimage import binary_fill_holes
from scipy.ndimage import label
import matplotlib.pyplot as plt
import simplekml
import matplotlib.colors as mcolors
from skimage.measure import find_contours
from shapely.geometry import Polygon

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
land = {1}
water = {4}
intertidal = {2, 3}

# boolean array: True where a pixel is confidently classified as land
land_mask = (
    (classification == 1)
    & (classification_qual == 0)
)

# bounding box of the azimuth/range indices present in this data, used to size and offset the 2D grid below
az_min, az_max = az.min(), az.max()
rg_min, rg_max = rg.min(), rg.max()
n_az = az_max - az_min + 1
n_rg = rg_max - rg_min + 1

# scatter the flat land boolean array into its proper 2D azimuth/range grid position
land_grid = np.zeros((n_az, n_rg), bool)
land_grid[az - az_min, rg - rg_min] = land_mask

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


land_quality = (classification_qual == 0)
land_quality_grid = np.zeros((n_az, n_rg), dtype=bool)
land_quality_grid[az - az_min, rg - rg_min] = land_quality

# a cell survives only if: it has data, passed quality checks, isn't inside the cleaned solid land region, and is a class we want to keep
final_mask = populated & quality_grid & not_land_clean & np.isin(classification_grid, list(water | intertidal))

# grids that store each cell's real longitude/latitude (NaN where no pixel exists)
lon_grid = np.full((n_az, n_rg), np.nan)
lat_grid = np.full((n_az, n_rg), np.nan)
lon_grid[az - az_min, rg - rg_min] = lon
lat_grid[az - az_min, rg - rg_min] = lat


def export_kml(gdf, category, output_dir):
    """Write a GeoDataFrame of Polygon/MultiPolygon geometries to a KML file."""
    if gdf is None or len(gdf) == 0:
        return

    kml = simplekml.Kml()

    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue

        # handle both single Polygons and MultiPolygons
        polys = [geom] if geom.geom_type == "Polygon" else list(geom.geoms)

        for i, poly in enumerate(polys):
            outer_coords = list(poly.exterior.coords)
            pol = kml.newpolygon(
                name=f"{category}_{row.region_id}"
                + (f"_{i}" if len(polys) > 1 else "")
            )
            pol.outerboundaryis = outer_coords

            # include any interior holes, if present
            if poly.interiors:
                pol.innerboundaryis = [list(ring.coords) for ring in poly.interiors]

            pol.style.linestyle.color = simplekml.Color.blue if category == "water" else simplekml.Color.red
            pol.style.linestyle.width = 2
            pol.style.polystyle.fill = 0

    kml_path = os.path.join(output_dir, f"{category}_polygon.kml")
    kml.save(kml_path)
    print(f"Wrote {len(gdf)} {category} polygons to {kml_path}")


# create polygons from connected water/intertidal/land regions
def export(category, codes):

    if category == "land":
        category_mask = (
            populated
            & land_quality_grid
            & np.isin(classification_grid, list(codes))
        )
    else:
        category_mask = (
            populated
            & quality_grid
            & not_land_clean
            & np.isin(classification_grid, list(codes))
        )


    if not np.any(category_mask):
        print(f"No {category} pixels found.")
        return gpd.GeoDataFrame(
            columns=[
                "category",
                "region_id",
                "num_points",
                "area",
                "cent_lon",
                "cent_lat",
                "geometry",
            ],
            geometry="geometry",
            crs="EPSG:4326",
        )


    # connected components
    labelled_array, num_features = label(
        category_mask,
        structure=np.ones((3,3))
    )


    print(f"{category}: found {num_features} connected regions")


    records = []


    for region_id in range(1, num_features+1):

        region = labelled_array == region_id


        # ignore tiny islands/noise
        if np.sum(region) < 10:
            continue


        # extract polygon boundary in pixel coordinates
        contours = find_contours(
            region.astype(float),
            0.5
        )


        for contour in contours:

            if len(contour) < 4:
                continue


            coords=[]

            for y,x in contour:

                # nearest pixel coordinate
                iy=int(round(y))
                ix=int(round(x))


                if (
                    0 <= iy < n_az
                    and 0 <= ix < n_rg
                    and not np.isnan(lon_grid[iy,ix])
                ):

                    coords.append(
                        (
                            lon_grid[iy,ix],
                            lat_grid[iy,ix]
                        )
                    )


            if len(coords)<4:
                continue


            poly = Polygon(coords)


            if not poly.is_valid:
                poly = poly.buffer(0)


            if poly.is_empty:
                continue


            centroid=poly.centroid


            records.append(
                {
                    "category":category,
                    "region_id":region_id,
                    "num_points":len(coords),
                    "area":poly.area,
                    "cent_lon":centroid.x,
                    "cent_lat":centroid.y,
                    "geometry":poly
                }
            )


    polygon_gdf=gpd.GeoDataFrame(
        records,
        crs="EPSG:4326"
    )


    outfile=os.path.join(
        output_dir,
        f"{category}_polygon.shp"
    )


    polygon_gdf.to_file(outfile)


    print(
        f"Wrote {len(polygon_gdf)} {category} polygons"
    )


    export_kml(
        polygon_gdf,
        category,
        output_dir
    )


    return polygon_gdf


# create water, intertidal, and land polygons
water_gdf = export("water", water)
intertidal_gdf = export("intertidal", intertidal)
land_gdf = export("land", land)


# print polygon information
print("\nWater polygon shapes:")
print(water_gdf.geometry.apply(lambda x: x.geom_type))

print("\nIntertidal polygon shapes:")
print(intertidal_gdf.geometry.apply(lambda x: x.geom_type))

print("\nLand polygon shapes:")
print(land_gdf.geometry.apply(lambda x: x.geom_type))


print(f"\nNumber of water polygons: {len(water_gdf)}")
print(f"Number of intertidal polygons: {len(intertidal_gdf)}")
print(f"Number of land polygons: {len(land_gdf)}")


# plot output polygons for checking

fig, ax = plt.subplots(figsize=(10, 10))

# NOTE: guard on len(...) > 0, not just "is not None" -- geopandas'
# .plot() tries to compute a mean-latitude aspect ratio internally, which
# raises "aspect must be finite and positive" if the GeoDataFrame is empty.
if water_gdf is not None and len(water_gdf) > 0:
    water_gdf.boundary.plot(
        ax=ax,
        color="blue",
        linewidth=1.5,
        label="Water"
    )

if intertidal_gdf is not None and len(intertidal_gdf) > 0:
    intertidal_gdf.boundary.plot(
        ax=ax,
        color="red",
        linewidth=1.5,
        label="Intertidal"
    )

if land_gdf is not None and len(land_gdf) > 0:
    land_gdf.boundary.plot(
        ax=ax,
        color="green",
        linewidth=1.5,
        label="Land"
    )


ax.set_title("SWOT PIXC Land,Water and Intertidal Polygons")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.legend()

plt.show()

# per-pixel land membership, pulled from the cleaned land mask (BW3) using each pixel's az/range grid position, used only to exclude land pixels
land_pixel = BW3[az - az_min, rg - rg_min]
not_land_pixel = ~land_pixel

water_pixel = quality & not_land_pixel & np.isin(classification, list(water))
intertidal_pixel = quality & not_land_pixel & np.isin(classification, list(intertidal))
land_pixel = quality & land_pixel & np.isin(classification, list(land))

# category code per pixel: 1 = water, 2 = intertidal, 3 = land, 0 = excluded
category = np.zeros(lon.shape, dtype=int)
category[water_pixel] = 1
category[intertidal_pixel] = 2
category[land_pixel] = 3

plot_valid = category > 0

class_labels = ["Water", "Intertidal", "Land"]
plot_colors = ["steelblue", "darkorange", "lightgreen"]

cmap = mcolors.ListedColormap(plot_colors)
norm = mcolors.BoundaryNorm([0.5, 1.5, 2.5, 3.5], cmap.N)

plt.figure(figsize=(8, 8))

sc = plt.scatter(
    lon[plot_valid],
    lat[plot_valid],
    c=category[plot_valid],
    s=2,
    cmap=cmap,
    norm=norm,
)

cbar = plt.colorbar(sc, ticks=[1, 2, 3])
cbar.set_label("Class")
cbar.set_ticklabels(class_labels)

plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("SWOT PIXC Land, Water and Intertidal Mask")

# correct aspect ratio for latitude distortion instead of plt.axis("equal")
if np.any(plot_valid):
    mean_lat = np.deg2rad(np.mean(lat[plot_valid]))
    aspect = 1 / np.cos(mean_lat)

    if np.isfinite(aspect) and aspect > 0:
        plt.gca().set_aspect(aspect)
else:
    print("No valid pixels to plot.")

plt.tight_layout()
plt.show()

