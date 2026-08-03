import os
import rasterio
import time
import numpy as np
import matplotlib.pyplot as plt
import shapely
from shapely.geometry import MultiPoint
import geopandas as gpd

from Piplineclass import SWOTIntertidalPipeline, SWOTPipelineConfig


file = r"C:\Users\pmalesza\Documents\Python Codes\SWOT_L2_HR_PIXC_052_475_245R_20260706T065928_20260706T065939_PID0_01.nc"
output_base = r"C:\Users\pmalesza\Documents\SWOT_L2_HR_PIXC Output Polygons"
cycle = 52


SHOW_PLOTS = False
concave_hull_ratio = 0.1  # 0 = tightest hull, 1 = convex hull


def _step_timer():
    """Return a function that prints elapsed time since this call."""
    t0 = time.time()
    return lambda label: print(f"    ({label} took {time.time() - t0:.1f}s)")

ref_lat = None
ref_lon = None

cfg = SWOTPipelineConfig()
pipe = SWOTIntertidalPipeline(cfg)

output_dir = pipe.make_output_directory(file, output_base)
print("Output directory:", output_dir)


# read pixel cloud as a flat DataFrame, draw the swath boundary, and subset to it
pixc_full = pipe.read_pixel_cloud(file, cycle)
print("Step 1 - read_pixel_cloud:", pixc_full.shape)
#pipe.plot_step(pixc_full, "step1_read_pixel_cloud", output_dir,
               #value_col="sigma_phase_noise", title="Step 1: Read Pixel Cloud")


kml_path = pipe.export_bbox_kml(pixc_full["longitude"], pixc_full["latitude"],
                                 file, output_base, name="SWOT PIXC Swath")
print(f"Step 1b - export_bbox_kml: wrote boundary KML to {kml_path}")


pixc = pipe.subset_by_kml(pixc_full, cfg.subset)
print(f"Step 1c - subset_by_kml: {len(pixc)} / {len(pixc_full)} pixels kept")
#pipe.plot_step(pixc, "step1c_subset_by_kml", output_dir,
               #value_col="sigma_phase_noise", title="Step 1c: Subset by Swath Boundary")

#print(pipe.estimate_phase_noise_threshold(pixc))  

# height anomaly

if ref_lat is None or ref_lon is None:
    ref_lat = float(pixc["latitude"].median())
    ref_lon = float(pixc["longitude"].median())
    print(
        "WARNING: ref_lat/ref_lon not set. Placeholder "
        f"used: ({ref_lat:.6f}, {ref_lon:.6f})"
    )

pixc = pipe.compute_height_anomaly(pixc, ref_lat, ref_lon)
print("Step 2 - compute_height_anomaly: added h_a column")
#pipe.plot_step(pixc, "step2_height_anomaly", output_dir,
               #value_col="h_a", title="Step 2: Height Anomaly (h_a)")

pipe.check_reference_point_classification(pixc)
if not pipe.cycle_has_reliable_xover(pixc):
    print(f"warning: cycle {cycle} does not have a reliable height_cor_xover")

# dark-water pixels elsewhere in open ocean can create a spurious third PDF peak
pixc = pipe.filter_dark_water(pixc)


# phase noise filter
filtered = pipe.filter_phase_noise(pixc)
print(f"Step 3 - filter_phase_noise: {len(filtered)} / {len(pixc)} pixels kept")
#pipe.plot_step(filtered, "step3_filter_phase_noise", output_dir,
               #value_col="h_a", title="Step 3: Phase-Noise Filtered Pixels")


# PDF-based open water filter to candidate intertidal+land pixels

_t = _step_timer()
candidates = pipe.filter_open_water(filtered)
print(f"Step 4 - filter_open_water: {len(candidates)} candidate pixels "
      f"(h_a_lower={candidates.attrs['h_a_lower']:.4f},"
      f"h_a_upper={candidates.attrs['h_a_upper']:.4f})")
_t("filter_open_water / KDE")
#pipe.plot_step(candidates, "step4_filter_open_water", output_dir,
               #value_col="h_a", title="Step 4: Open-Water Filtered Candidates")


# water extent mask, applied to the candidates, pixels that are the actual fully filtered intertidal set for this cycle

_t = _step_timer()
water_extent_mask = pipe.build_water_extent_mask([filtered])
print(f"Step 5a - build_water_extent_mask: polygon area="
      f"{water_extent_mask.area:.8f} deg^2, bounds={water_extent_mask.bounds}")
_t("build_water_extent_mask")
#pipe.plot_step(water_extent_mask, "step5a_water_extent_mask", output_dir,
              # title="Step 5a: Water Extent Mask")

_t = _step_timer()
print(f"    water_extent_mask.is_valid = {water_extent_mask.is_valid}, "
      f"n_exterior_coords = {len(water_extent_mask.exterior.coords)}")
intertidal_pixels = pipe.apply_water_extent_mask(candidates, water_extent_mask)
_t("apply_water_extent_mask")
print(f"Step 5b - apply_water_extent_mask: {len(intertidal_pixels)} final "
      f"intertidal pixels")
_t = _step_timer()
#pipe.plot_step(intertidal_pixels, "step5b_apply_water_extent_mask", output_dir,
               #value_col="h_a", title="Step 5b: Final Intertidal Pixels")
_t("plot_step (step5b)")


_t = _step_timer()
intertidal_pixels = pipe.estimate_pixel_uncertainty(intertidal_pixels)
_t("estimate_pixel_uncertainty")


# polygons

h_a_lower = candidates.attrs["h_a_lower"]
water_pixels = filtered[filtered["h_a"] < h_a_lower].reset_index(drop=True)

# Cluster points into spatially connected groups first 
_t = _step_timer()
water_gdf = pipe.cluster_points_to_polygons(
    water_pixels, "water", concave_hull_ratio=concave_hull_ratio
)
_t("cluster_points_to_polygons (water)")
print(f"Water polygons: {len(water_gdf)}")

_t = _step_timer()
intertidal_gdf = pipe.cluster_points_to_polygons(
    intertidal_pixels, "intertidal", concave_hull_ratio=concave_hull_ratio
)
_t("cluster_points_to_polygons (intertidal)")
print(f"Intertidal polygons (from filtered pixels only): {len(intertidal_gdf)}")


# export shapefiles + KMLs for each category that has polygons

shapefile_paths = {}
for category, category_gdf in (("water", water_gdf), ("intertidal", intertidal_gdf)):
    if category_gdf is None or len(category_gdf) == 0:
        print(f"No {category} pixels found.")
        continue
    shapefile_paths[category] = pipe.export_polygons_shapefile(category_gdf, category, output_dir)
    pipe.export_polygons_kml(category_gdf, category, output_dir)


# optional aggregate grid for parity

_t = _step_timer()
grid = pipe.aggregate_to_grid(intertidal_pixels)
grid_csv = os.path.join(output_dir, "intertidal_grid.csv")
grid.to_csv(grid_csv, index=False)
print(f"Wrote gridded output to {grid_csv}")
_t(f"aggregate_to_grid ({len(grid)} cells x "
   f"{cfg.mc_realizations} MC realizations)")


# plots

_t = _step_timer()
ax1 = pipe.plot_category_polygons({
    "water": water_gdf,
    "intertidal": intertidal_gdf,
})
polygons_png = os.path.join(output_dir, "category_polygons.png")
ax1.figure.savefig(polygons_png, dpi=150)
print(f"Saved category polygons plot to {polygons_png}")
if SHOW_PLOTS:
    plt.show()
else:
    plt.close(ax1.figure)
_t("plotting")

if "intertidal" in shapefile_paths:
    check_gdf = gpd.read_file(shapefile_paths["intertidal"])
    print(check_gdf.crs)                    # should be EPSG:4326
    print(check_gdf.geometry.is_valid)      # no self-intersections
    print(check_gdf.geometry.is_empty)      # not empty
    print(check_gdf.total_bounds)           # sanity-check lat/lon range makes sense for your swath
    print(check_gdf[["num_points", "area", "cent_lon", "cent_lat"]])

print("\nDone.")


