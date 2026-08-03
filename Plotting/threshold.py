import xarray as xr
import matplotlib.pyplot as plt

data = xr.open_dataset(
    r"C:\Users\pmalesza\Documents\Python Codes\SWOT_L2_HR_PIXC_052_475_245R_20260706T065928_20260706T065939_PID0_01.nc",
    group="pixel_cloud"
)

phase = data.phase_noise_std.values

plt.figure(figsize=(8,8))
plt.scatter(
    data.longitude.values,
    data.latitude.values,
    c=phase,
    s=2,
    cmap="viridis",
    vmin=0,
    vmax=0.1
)
plt.colorbar(label="Phase Noise (rad)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("SWOT Phase Noise")
plt.show()
