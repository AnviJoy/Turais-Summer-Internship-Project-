import os
import rasterio
import time
import numpy as np
import matplotlib.pyplot as plt
import shapely
from shapely.geometry import MultiPoint
import geopandas as gpd
from shapely.geometry import MultiPolygon

from Piplineclass import SWOTIntertidalPipeline, SWOTPipelineConfig

#kirby
#file = r"C:\Users\Lily Donaldson\Documents\Anvi\Python Codes\SWOT_L2_HR_PIXC_501_016_060L_20230425T064527_20230425T064532_PGC0_01.nc"

#weston
file = r"C:\Users\Lily Donaldson\Documents\Anvi\Python Codes\SWOT_L2_HR_PIXC_052_475_245R_20260706T065928_20260706T065939_PID0_01.nc"

#france
#file = r"C:\Users\Lily Donaldson\Documents\Anvi\Python Codes\SWOT_L2_HR_PIXC_053_348_073L_20260722T142201_20260722T142213_PID0_01.nc"

output_base = r"C:\Users\Lily Donaldson\Documents\Anvi\SWOT_L2_HR_PIXC Output Polygons"

#r"C:\Users\pmalesza\Documents\SWOT_L2_HR_PIXC Output Polygons"
cycle = 52


SHOW_PLOTS = False
build_plots = False
EXCLUDE_LAND = False   # turn land masking on/off


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


# read pixel cloud as a flat DataFrame
pixc_full = pipe.read_pixel_cloud(file, cycle)
print("Step 1 - read_pixel_cloud:", pixc_full.shape)
#pipe.plot_step(pixc_full, "step1_read_pixel_cloud", output_dir,
               #value_col="sigma_phase_noise", title="Step 1: Read Pixel Cloud")


# subset straight to the France KML polygon defined in cfg.subset
pixc, subset_mask = pipe.subset_by_kml(pixc_full, cfg.subset)
print(f"Step 1b - subset_by_kml (france_subset): {len(pixc)} / {len(pixc_full)} pixels kept")
print(type(subset_mask))
if build_plots:
    
   pipe.plot_mask_step(pixc_full, subset_mask, "step1b_subset_by_kml", output_dir,
                     title="Step 1b: Subset by France KML")

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

# land pixels can also pollute the h_a PDF / final polygons if left in
n_before_land = len(pixc)
if EXCLUDE_LAND:
    # filter_land() only trims the `pixc` DataFrame
    land_keep = (pixc_full["classification"] != cfg.land_class_code).to_numpy()
    subset_mask = subset_mask & land_keep
pixc = pipe.filter_land(pixc, enabled=EXCLUDE_LAND)
if EXCLUDE_LAND:
    print(f"Step 2b - filter_land: {len(pixc)} / {n_before_land} pixels kept")
else:
    print("Step 2b - filter_land: skipped (EXCLUDE_LAND=False)")


# phase noise filter
filtered, phase_noise_mask = pipe.filter_phase_noise(subset_mask, pixc)
print(f"Step 3 - filter_phase_noise: {len(filtered)} / {len(pixc)} pixels kept")
if build_plots:
   pipe.plot_mask_step(pixc_full, phase_noise_mask, "step3_filter_phase_noise", output_dir,
                     title="Step 3: Phase-Noise Filtered Pixels")


# PDF-based open water filter to candidate intertidal+land pixels

_t = _step_timer()
candidates, open_water_mask = pipe.filter_open_water(filtered, phase_noise_mask)
print(f"Step 4 - filter_open_water: {len(candidates)} candidate pixels "
      f"(h_a_lower={candidates.attrs['h_a_lower']:.4f},"
      f"h_a_upper={candidates.attrs['h_a_upper']:.4f})")
_t("filter_open_water / KDE")
if build_plots:
   pipe.plot_mask_step(pixc_full, open_water_mask, "step4_filter_open_water", output_dir,
                     title="Step 4: Open-Water Filtered Candidates")


# water extent mask, applied to the candidates, pixels that are the actual fully filtered intertidal set for this cycle
_t = _step_timer()
water_extent_mask = pipe.build_water_extent_mask(open_water_mask, [filtered])

if water_extent_mask.geom_type == "MultiPolygon":
    piece_areas = [poly.area for poly in water_extent_mask.geoms]
else:
    piece_areas = [water_extent_mask.area]
print(
    "    water_extent_mask piece area (deg^2) stats: "
    f"n_pieces={len(piece_areas)}, min={min(piece_areas):.3e}, "
    f"median={sorted(piece_areas)[len(piece_areas) // 2]:.3e}, "
    f"max={max(piece_areas):.3e}"
)

# Remove very small polygons. Threshold is set relative to the largest piece (rather than a fixed absolute number) so it scales correctly
water_extent_min_area = max(piece_areas) * cfg.water_extent_min_piece_fraction
print(f"    remove_small_polygons: using min_area={water_extent_min_area:.3e} deg^2 "
      f"({cfg.water_extent_min_piece_fraction:.1%} of largest piece)")

water_extent_mask = pipe.remove_small_polygons(
    water_extent_mask,
    min_area=water_extent_min_area
)

print(f"Step 5a - build_water_extent_mask: polygon area="
      f"{water_extent_mask.area:.8f} deg^2, bounds={water_extent_mask.bounds}")
_t("build_water_extent_mask")
if build_plots:
  pipe.plot_step(water_extent_mask, "step5a_water_extent_mask", output_dir,
              title="Step 5a: Water Extent Mask")

# _t = _step_timer()
# print(f"    water_extent_mask.is_valid = {water_extent_mask.is_valid}, "
#       f"n_exterior_coords = {len(water_extent_mask.exterior.coords)}")

_t = _step_timer()

if water_extent_mask.geom_type == "MultiPolygon":
    n_coords = sum(len(poly.exterior.coords) for poly in water_extent_mask.geoms)
else:
    n_coords = len(water_extent_mask.exterior.coords)

print(
    f"    water_extent_mask.is_valid = {water_extent_mask.is_valid}, "
    f"n_exterior_coords = {n_coords}"
)

#intertidal_pixels = pipe.apply_water_extent_mask(candidates, water_extent_mask)
intertidal_pixels, inside_mask = pipe.apply_water_extent_mask(
    candidates,
    water_extent_mask,
    base_mask=open_water_mask,
)
_t("apply_water_extent_mask")
print(f"Step 5b - apply_water_extent_mask: {len(intertidal_pixels)} final "
      f"intertidal pixels")
_t = _step_timer()
if build_plots:
  pipe.plot_step(intertidal_pixels, "step5b_apply_water_extent_mask", output_dir,
               value_col="h_a", title="Step 5b: Final Intertidal Pixels")
  _t("plot_step (step5b)")


_t = _step_timer()
intertidal_pixels = pipe.estimate_pixel_uncertainty(intertidal_pixels)
_t("estimate_pixel_uncertainty")

_t = _step_timer()
#grid = pipe.gridify(pixc_full["azimuth_index"], pixc_full["range_index"], inside_mask)
grid = pipe.gridify(pixc_full["azimuth_index"], pixc_full["range_index"], open_water_mask)
_t("gridify")
if build_plots:
    pipe.plot_step(grid, "step5c_intertidal_grid", output_dir,
                    title="Step 5c: Final Intertidal Mask (az/range grid)")


# polygons 

_t = _step_timer()
lon_grid = pipe.gridify(
    pixc_full["azimuth_index"], pixc_full["range_index"],
    pixc_full["longitude"], fill_value=np.nan, dtype=float,
)
lat_grid = pipe.gridify(
    pixc_full["azimuth_index"], pixc_full["range_index"],
    pixc_full["latitude"], fill_value=np.nan, dtype=float,
)
_t("gridify (lon/lat)")

_t = _step_timer()

intertidal_gdf = pipe.polygons_from_raster_mask(
    grid, lon_grid, lat_grid, category="intertidal",
    # fill_holes patches over *any* fully-enclosed gap in the mask
    fill_holes=not EXCLUDE_LAND,
)
_t("polygons_from_raster_mask")
print(f"Step 6 - polygons_from_raster_mask: {len(intertidal_gdf)} intertidal polygon(s)")

if len(intertidal_gdf) > 0:
    _areas_sorted = intertidal_gdf["area"].sort_values()
    _pcts = [10, 25, 50, 75, 90, 95, 99]
    _pct_str = ", ".join(
        f"p{p}={_areas_sorted.quantile(p / 100):.3e}" for p in _pcts
    )
    print(
        "    intertidal_gdf area (deg^2) stats: "
        f"min={intertidal_gdf['area'].min():.3e}, {_pct_str}, "
        f"max={intertidal_gdf['area'].max():.3e}"
    )

# drop final intertidal polygons smaller than cfg.min_intertidal_polygon_area
n_before = len(intertidal_gdf)
intertidal_gdf = pipe.remove_small_polygons(
    intertidal_gdf, min_area=cfg.min_intertidal_polygon_area, area_col="area",
)
print(f"Step 6b - remove_small_polygons: kept {len(intertidal_gdf)} / {n_before} "
      f"intertidal polygon(s) (min_area={cfg.min_intertidal_polygon_area} deg^2)")

# export shapefile + KML for the intertidal category, if any polygons were found

shapefile_paths = {}
if len(intertidal_gdf) == 0:
    print("No intertidal polygons found.")
else:
    shapefile_paths["intertidal"] = pipe.export_polygons_shapefile(intertidal_gdf, "intertidal", output_dir)
    pipe.export_polygons_kml(intertidal_gdf, "intertidal", output_dir)

# optional aggregate grid for parity

_t = _step_timer()
agg_grid = pipe.aggregate_to_grid(intertidal_pixels)
grid_csv = os.path.join(output_dir, "intertidal_grid.csv")
agg_grid.to_csv(grid_csv, index=False)
print(f"Wrote gridded output to {grid_csv}")
_t(f"aggregate_to_grid ({len(agg_grid)} cells x {cfg.mc_realizations} MC realizations)")

# plots

if len(intertidal_gdf) > 0:
    _t = _step_timer()
    ax1 = pipe.plot_category_polygons({"intertidal": intertidal_gdf})
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
