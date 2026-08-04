import matplotlib.pyplot as plt
from Piplineclass import SWOTIntertidalPipeline, SWOTPipelineConfig

file = r"C:\Users\Lily Donaldson\Documents\Anvi\Python Codes\SWOT_L2_HR_PIXC_052_475_245R_20260706T065928_20260706T065939_PID0_01.nc"

#r"C:\Users\pmalesza\Documents\Python Codes\SWOT_L2_HR_PIXC_052_475_245R_20260706T065928_20260706T065939_PID0_01.nc"

output_base = r"C:\Users\Lily Donaldson\Documents\Anvi\SWOT_L2_HR_PIXC Output Polygons"

#r"C:\Users\pmalesza\Documents\SWOT_L2_HR_PIXC Output Polygons"
cycle = 52

cfg = SWOTPipelineConfig()
pipe = SWOTIntertidalPipeline(cfg)

output_dir = pipe.make_output_directory(file, output_base)

# replicate the steps in PolygonProcesser.py
pixc_full = pipe.read_pixel_cloud(file, cycle)
kml_path = pipe.export_bbox_kml(pixc_full["longitude"], pixc_full["latitude"],
                                 file, output_base, name="SWOT PIXC Swath")
pixc = pipe.subset_by_kml(pixc_full, cfg.subset)

ref_lat = float(pixc["latitude"].median())
ref_lon = float(pixc["longitude"].median())
pixc = pipe.compute_height_anomaly(pixc, ref_lat, ref_lon)
pixc = pipe.filter_dark_water(pixc)

filtered = pipe.filter_phase_noise(pixc)
candidates = pipe.filter_open_water(filtered)

# pull the KDE diagnostics stashed on candidates.attrs by filter_open_water 
grid = candidates.attrs["grid"]
pdf = candidates.attrs["pdf"]
peak_idx = candidates.attrs["peak_idx"]
h_a_lower = candidates.attrs["h_a_lower"]
h_a_upper = candidates.attrs["h_a_upper"]

# every local max that cleared pdf_peak_min_density, not just the one that got used
all_peaks_idx = pipe.find_pdf_peaks(grid, pdf)

plt.figure(figsize=(9, 6))
plt.plot(grid, pdf, color="black", lw=1.2, label="KDE of h_a")

plt.axvline(h_a_lower, color="tab:green", ls="--", lw=1.5,
            label=f"h_a_lower = {h_a_lower:.4f}")
plt.axvline(h_a_upper, color="tab:red", ls="--", lw=1.5,
            label=f"h_a_upper = {h_a_upper:.4f}")
plt.axvspan(h_a_lower, h_a_upper, color="tab:blue", alpha=0.08,
            label="candidate band (Step 4 output)")

plt.scatter(grid[all_peaks_idx], pdf[all_peaks_idx], color="tab:orange",
            zorder=5, s=30, label="peaks found (>= pdf_peak_min_density)")
plt.scatter(grid[peak_idx], pdf[peak_idx], color="black", zorder=6, s=70,
            marker="*", label="peak_idx used (open water)")

plt.xlabel("Height anomaly h_a (m)")
plt.xlim(-25,50)
plt.ylabel("Density")
plt.title(
    f"Step 4 KDE diagnostic — cycle {cycle}\n"
    f"{len(candidates)} / {len(filtered)} pixels kept as candidates"
)
plt.legend(fontsize=8)
plt.tight_layout()
plt.show()

print(f"n peaks found: {len(all_peaks_idx)} at h_a = "
      f"{[round(float(v), 4) for v in grid[all_peaks_idx]]}")
print(f"peak_idx used: h_a = {grid[peak_idx]:.4f}")
print(f"h_a_lower = {h_a_lower:.4f}, h_a_upper = {h_a_upper:.4f}")
