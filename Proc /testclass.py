from swot_intertidal_pipeline import SWOTIntertidalPipeline, SWOTPipelineConfig

file = r"C:\Users\pmalesza\Documents\Python Codes\SWOT_L2_HR_PIXC_052_475_245R_20260706T065928_20260706T065939_PID0_01.nc"
cycle = 52  

ref_lat = None
ref_lon = None

cfg = SWOTPipelineConfig()
pipe = SWOTIntertidalPipeline(cfg)

pixc = pipe.read_pixel_cloud(file, cycle)
print("Step 1 — read_pixel_cloud:", pixc.shape)
print(pixc.head())

if ref_lat is None or ref_lon is None:
    ref_lat = float(pixc["latitude"].median())
    ref_lon = float(pixc["longitude"].median())
    print(f"No ref_lat/ref_lon set, using region centroid as a placeholder: "
          f"({ref_lat:.6f}, {ref_lon:.6f})")
  
pixc = pipe.compute_height_anomaly(pixc, ref_lat, ref_lon)
print("\nStep 2 — compute_height_anomaly: added h_a column")
print(pixc[["height", "h_a"]].describe())

pipe.check_reference_point_classification(pixc)
pipe.cycle_has_reliable_xover(pixc)

filtered = pipe.filter_phase_noise(pixc)
print(f"\nStep 3 — filter_phase_noise: {len(filtered)} / {len(pixc)} pixels kept")

candidates = pipe.filter_open_water(filtered)
print(f"\nStep 4 — filter_open_water: {len(candidates)} candidate pixels "
      f"(h_a_lower={candidates.attrs['h_a_lower']:.4f}, "
      f"h_a_upper={candidates.attrs['h_a_upper']:.4f})")

mask = pipe.build_water_extent_mask([filtered])
print(f"\nStep 5a — build_water_extent_mask: polygon area={mask.area:.8f} deg^2, "
      f"bounds={mask.bounds}")

intertidal = pipe.apply_water_extent_mask(candidates, mask)
print(f"\nStep 5b — apply_water_extent_mask: {len(intertidal)} final intertidal pixels")

intertidal = pipe.estimate_pixel_uncertainty(intertidal)
print("\nStep 7 — estimate_pixel_uncertainty: added sigma_h column")
print(intertidal[["height", "sigma_h"]].describe())

grid = pipe.aggregate_to_grid(intertidal)
print(f"\nStep 8 — aggregate_to_grid: {len(grid)} output cells")
print(grid.head())

grid_stats = pipe.validate_against_dem(grid, r"path\to\reference_dem.tif")
print(grid_stats)

print("\nDone.")
