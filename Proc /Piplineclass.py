import warnings
from dataclasses import dataclass, field
from typing import Optional, Sequence
import numpy as np
import pandas as pd
from scipy import stats
import xarray as xr
from rasterio.features import shapes as _rio_shapes
from affine import Affine
from shapely.geometry import shape as _shapely_shape
from scipy.signal import argrelextrema
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
from shapely.geometry import Point
from shapely import vectorized
from skimage.morphology import remove_small_objects, binary_closing, disk
from scipy.ndimage import binary_fill_holes
from shapely.geometry import MultiPoint
import geopandas as gpd
from scipy.ndimage import label
import os
import simplekml
import rasterio
from rasterio.transform import from_bounds
from rasterio.features import rasterize
from rasterio.transform import rowcol
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

@dataclass
class SWOTPipelineConfig:
    """Tunable thresholds for the pipeline, defaults from the paper's pseudocode."""

    sigma_phase_noise_threshold: float = 0.08
    ref_point_buffer_deg: float = 0.001
    pdf_bin_width_m: float = 0.05
    pdf_peak_min_density: float = 0.10
    pdf_max_bins: int = 2000
    kde_max_samples: Optional[int] = 50_000
    kde_random_seed: Optional[int] = 0
    eps_up_fraction: float = 0.01
    eps_low_divisor: float = 50.0
    mask_grid_res_deg: float = 0.0001
    mask_validity_quantile: float = 0.90
    output_grid_res_deg: float = 0.00025
    mc_realizations: int = 1000
    mc_ci_alpha: float = 0.025
    exclude_dark_water: bool = False
    min_hole_area_deg2: float = 1e-8
    smoothing_window: int = 5

    fill_value_threshold: float = 1e30

    # Placeholders — confirm against the PDD/ATBD classification table.
    open_water_class_codes: tuple = (3, 4)
    dark_water_class_codes: tuple = (5, 6)

    # Variable name aliases for reading L2_HR_PIXC granules.
    pixc_var_aliases: dict = field(default_factory=lambda: {
        "height": ["height"],
        "sigma_phase_noise": ["phase_noise_std", "sigma_phase_noise", "phase_noise_sigma"],
        "dh_dphi": ["dheight_dphase", "dh_dphi", "dhdphi"],
        "geolocation_qual": ["geolocation_qual", "geoloc_qual", "geo_qual"],
        "classification": ["classification", "classification_flag"],
        "latitude": ["latitude", "lat"],
        "longitude": ["longitude", "lon"],
        "azimuth_index": ["azimuth_index", "az_index"],
        "range_index": ["range_index", "rg_index"],
    })

    # Optional variables: read if present, otherwise just warn.
    pixc_optional_var_aliases: dict = field(default_factory=lambda: {
        "height_cor_xover": ["height_cor_xover"],
        "geoid": ["geoid"],
        "solid_earth_tide": ["solid_earth_tide"],
        "load_tide_fes": ["load_tide_fes"],
        "pole_tide": ["pole_tide"],
    })


class SWOTIntertidalPipeline:
    """SWOT L2_HR_PIXC to intertidal topography pipeline, following the paper's method."""

    def __init__(self, config: Optional[SWOTPipelineConfig] = None):
        """Store the config (or defaults) and reset the cached water extent mask."""
        self.cfg = config or SWOTPipelineConfig()
        self._water_extent_mask = None

    def read_pixel_cloud(self, filepath: str, cycle: Optional[int] = None,
                          bbox: Optional[tuple] = None,
                          group: Optional[str] = None,
                          extra_var_aliases: Optional[dict] = None) -> pd.DataFrame:
        """Read an L2_HR_PIXC granule into a flat DataFrame with standardized columns."""
        ds = self._open_pixc_group(xr, filepath, group)

        aliases = {k: list(v) for k, v in self.cfg.pixc_var_aliases.items()}
        if extra_var_aliases:
            for k, extra in extra_var_aliases.items():
                aliases.setdefault(k, []).extend(extra)

        data = {}
        missing_vars = []
        for standard_name, candidates in aliases.items():
            found = next((c for c in candidates if c in ds.variables), None)
            if found is None:
                missing_vars.append(standard_name)
                continue
            data[standard_name] = self._mask_fill_values(ds[found].values)

        if missing_vars:
            warnings.warn(
                f"Pixel cloud file is missing/unrecognized for: {missing_vars}. "
                "Pass `extra_var_aliases` to `read_pixel_cloud` if this "
                "product version uses different variable names."
            )

        for standard_name, candidates in self.cfg.pixc_optional_var_aliases.items():
            found = next((c for c in candidates if c in ds.variables), None)
            if found is not None:
                data[standard_name] = self._mask_fill_values(ds[found].values)

        df = pd.DataFrame(data)
        if cycle is not None:
            df["cycle"] = cycle

        n_before = len(df)
        required_cols = [c for c in aliases if c in df.columns]
        df = df.dropna(subset=required_cols).reset_index(drop=True)
        n_dropped = n_before - len(df)
        if n_dropped:
            warnings.warn(
                f"Dropped {n_dropped}/{n_before} pixels with fill-value/NaN "
                "entries in required fields."
            )

        if bbox is not None and "longitude" in df and "latitude" in df:
            lon_min, lon_max, lat_min, lat_max = bbox
            df = df[
                (df["longitude"] >= lon_min) & (df["longitude"] <= lon_max) &
                (df["latitude"] >= lat_min) & (df["latitude"] <= lat_max)
            ].reset_index(drop=True)

        return df

    def _mask_fill_values(self, arr: np.ndarray) -> np.ndarray:
        """Replace netCDF fill-value sentinels with NaN."""
        arr = np.asarray(arr)
        if not np.issubdtype(arr.dtype, np.floating):
            return arr
        arr = arr.astype(float, copy=True)
        arr[np.abs(arr) >= self.cfg.fill_value_threshold] = np.nan
        return arr

    @staticmethod
    def _open_pixc_group(xr, filepath: str, group: Optional[str]):
        """Open the granule's pixel_cloud group, auto-detecting if not given."""
        if group is not None:
            return xr.open_dataset(filepath, group=group)
        try:
            return xr.open_dataset(filepath, group="pixel_cloud")
        except (OSError, KeyError, ValueError):
            root = xr.open_dataset(filepath)
            for candidate in ("pixel_cloud", "PIXEL_CLOUD", "pixc"):
                try:
                    return xr.open_dataset(filepath, group=candidate)
                except (OSError, KeyError, ValueError):
                    continue
            return root

    def cycle_has_reliable_xover(self, pixc_df: pd.DataFrame,
                                  max_missing_frac: float = 0.5) -> bool:
        """Return False if height_cor_xover is absent or mostly missing for this cycle."""
        if "height_cor_xover" not in pixc_df.columns:
            warnings.warn("height_cor_xover not present in this granule; "
                           "cannot evaluate crossover-correction reliability.")
            return False
        missing_frac = pixc_df["height_cor_xover"].isna().mean()
        return missing_frac <= max_missing_frac

    def compute_height_anomaly(self, pixc_df: pd.DataFrame,
                                ref_lat: float, ref_lon: float,
                                buffer_deg: Optional[float] = None) -> pd.DataFrame:
        """Subtract the local open-water reference height, adding column 'h_a'."""
        buffer_deg = buffer_deg or self.cfg.ref_point_buffer_deg
        df = pixc_df.copy()

        good_geo = df[df["geolocation_qual"] == 0]
        if good_geo.empty:
            raise ValueError("No pixels with geolocation_qual == 0 found; "
                              "cannot locate a reference pixel.")

        dist2 = (good_geo["latitude"] - ref_lat) ** 2 + \
                (good_geo["longitude"] - ref_lon) ** 2
        ref_idx = dist2.idxmin()
        ref_pixel_lat = good_geo.loc[ref_idx, "latitude"]
        ref_pixel_lon = good_geo.loc[ref_idx, "longitude"]

        if good_geo.loc[ref_idx, "classification"] not in (
                self._open_water_class_codes()):
            warnings.warn(
                "Reference pixel's classification does not look like open "
                "water (see Step 6 QC note) — check ref_lat/ref_lon."
            )

        in_buffer = (
            (df["latitude"] - ref_pixel_lat).abs() <= buffer_deg
        ) & (
            (df["longitude"] - ref_pixel_lon).abs() <= buffer_deg
        )
        ref_median_height = df.loc[in_buffer, "height"].median()
        if np.isnan(ref_median_height):
            raise ValueError("No pixels found within the reference buffer; "
                              "widen buffer_deg or check ref point.")

        df["h_a"] = df["height"] - ref_median_height
        df.attrs["ref_median_height"] = ref_median_height
        df.attrs["ref_pixel_latlon"] = (ref_pixel_lat, ref_pixel_lon)
        return df

    def _open_water_class_codes(self) -> Sequence[int]:
        """Return the configured open-water classification codes (unverified default)."""
        return self.cfg.open_water_class_codes

    def _dark_water_class_codes(self) -> Sequence[int]:
        """Return the configured dark-water classification codes (unverified default)."""
        return self.cfg.dark_water_class_codes

    def filter_phase_noise(self, pixc_df: pd.DataFrame,
                            threshold: Optional[float] = None) -> pd.DataFrame:
        """Drop pixels with sigma_phase_noise above `threshold`."""
        threshold = threshold if threshold is not None \
            else self.cfg.sigma_phase_noise_threshold
        return pixc_df[pixc_df["sigma_phase_noise"] <= threshold].reset_index(drop=True)

    def estimate_phase_noise_threshold(self, pixc_df: pd.DataFrame) -> float:
        """Return the median sigma_phase_noise as a starting threshold estimate."""
        return float(pixc_df["sigma_phase_noise"].median())

    def _kde_pdf(self, h_a: np.ndarray, bin_width: Optional[float] = None):
        """Compute a Gaussian-KDE PDF of h_a over an evenly spaced grid."""
        bin_width = bin_width or self.cfg.pdf_bin_width_m
        h_a = h_a[~np.isnan(h_a)]
        if h_a.size < 2:
            raise ValueError("Not enough h_a samples to build a KDE.")

        max_samples = self.cfg.kde_max_samples
        if max_samples is not None and h_a.size > max_samples:
            # gaussian_kde evaluation is O(n_points * n_grid_bins); on full
            # pixel clouds (10^5-10^6 points) that's the dominant cost of
            # filter_open_water. A large random subsample gives a
            # statistically equivalent density estimate for peak/cutoff
            # detection at a fraction of the runtime.
            rng = np.random.default_rng(self.cfg.kde_random_seed)
            h_a = rng.choice(h_a, size=max_samples, replace=False)

        n_bins = max(int(np.ceil((h_a.max() - h_a.min()) / bin_width)), 10)
        n_bins = min(n_bins, self.cfg.pdf_max_bins)
        grid = np.linspace(h_a.min(), h_a.max(), n_bins)

        kde = stats.gaussian_kde(h_a)
        pdf = kde(grid)
        return grid, pdf, kde

    def find_pdf_peaks(self, grid: np.ndarray, pdf: np.ndarray,
                        min_density: Optional[float] = None) -> np.ndarray:
        """Return indices of PDF local maxima with density >= min_density."""
        min_density = min_density if min_density is not None \
            else self.cfg.pdf_peak_min_density

        maxima_idx = argrelextrema(pdf, np.greater_equal, order=1)[0]
        maxima_idx = maxima_idx[pdf[maxima_idx] >= min_density]
        return np.unique(maxima_idx)

    def compute_upper_cutoff(self, grid: np.ndarray, pdf: np.ndarray,
                              peak_idx: int, eps_up_fraction: Optional[float] = None) -> float:
        """Return h_a where the PDF first drops to eps_up_fraction of the peak density."""
        eps_up_fraction = eps_up_fraction if eps_up_fraction is not None \
            else self.cfg.eps_up_fraction
        peak_density = pdf[peak_idx]
        threshold = eps_up_fraction * peak_density

        for i in range(peak_idx, len(pdf)):
            if pdf[i] <= threshold:
                return float(grid[i])
        return float(grid[-1])

    def compute_lower_cutoff(self, grid: np.ndarray, pdf: np.ndarray,
                              peak_idx: int, upper_cutoff: float) -> float:
        """Return the h_a cutoff separating open water from non-open-water (Step 4 Case A/B)."""
        dpdf = np.gradient(pdf, grid)
        d2pdf = np.gradient(dpdf, grid)

        upper_idx = int(np.searchsorted(grid, upper_cutoff))
        window = slice(peak_idx, max(upper_idx, peak_idx + 1))

        minima_idx = argrelextrema(dpdf[window], np.less_equal, order=1)[0] + peak_idx

        eps_low_divisor = self.cfg.eps_low_divisor

        if minima_idx.size >= 2:
            mags = np.abs(dpdf[minima_idx])
            order = np.argsort(mags)[::-1]
            h_min_idx = minima_idx[order[0]]
            h_next_idx = minima_idx[order[1]] if len(order) > 1 else upper_idx

            lo, hi = sorted((h_min_idx, h_next_idx))
            eps_low = np.abs(dpdf[h_min_idx]) / eps_low_divisor

            other_peaks = self.find_pdf_peaks(grid, pdf)
            other_peaks = other_peaks[other_peaks != peak_idx]

            search_lo, search_hi = lo, hi
            h_a_lower = None
            attempts = 0
            while h_a_lower is None and attempts < 10:
                candidate_idx = None
                for i in range(search_lo, search_hi + 1):
                    if np.abs(dpdf[i]) <= eps_low and i not in other_peaks:
                        candidate_idx = i
                        break
                if candidate_idx is not None:
                    h_a_lower = grid[candidate_idx]
                else:
                    search_hi = max(search_lo + 1, search_hi - 1)
                    attempts += 1
            if h_a_lower is None:
                h_a_lower = grid[lo]

        else:
            h_min_idx = minima_idx[0] if minima_idx.size == 1 else peak_idx
            tail = slice(h_min_idx, max(upper_idx, h_min_idx + 1))
            if len(d2pdf[tail]) == 0:
                h_a_lower = grid[h_min_idx]
            else:
                peak_pdf2_idx = h_min_idx + int(np.argmax(d2pdf[tail]))
                beyond = slice(peak_pdf2_idx, max(upper_idx, peak_pdf2_idx + 1))
                if len(dpdf[beyond]) == 0:
                    eps_low = np.abs(dpdf[h_min_idx]) / eps_low_divisor
                else:
                    eps_low = np.min(dpdf[beyond]) / eps_low_divisor if np.min(dpdf[beyond]) != 0 \
                        else np.abs(dpdf[h_min_idx]) / eps_low_divisor
                    eps_low = abs(eps_low)

                candidate_idx = None
                for i in range(peak_pdf2_idx, max(upper_idx, peak_pdf2_idx + 1)):
                    if np.abs(dpdf[i]) <= eps_low:
                        candidate_idx = i
                        break
                h_a_lower = grid[candidate_idx] if candidate_idx is not None else grid[h_min_idx]

        doubling = 0
        while h_a_lower >= upper_cutoff and doubling < 10:
            eps_low = eps_low * 2 if 'eps_low' in dir() else 1e-6
            lo_idx = int(np.searchsorted(grid, h_a_lower))
            hi_idx = int(np.searchsorted(grid, upper_cutoff))
            candidate_idx = None
            for i in range(min(lo_idx, hi_idx), max(lo_idx, hi_idx) + 1):
                if np.abs(dpdf[i]) <= eps_low:
                    candidate_idx = i
                    break
            h_a_lower = grid[candidate_idx] if candidate_idx is not None else grid[max(hi_idx - 1, 0)]
            doubling += 1

        return float(h_a_lower)

    def filter_open_water(self, pixc_df: pd.DataFrame) -> pd.DataFrame:
        """Return candidate non-open-water pixels (h_a within the Step 4 cutoffs)."""
        h_a = pixc_df["h_a"].to_numpy()
        grid, pdf, _ = self._kde_pdf(h_a)

        peaks_idx = self.find_pdf_peaks(grid, pdf)
        if peaks_idx.size == 0:
            raise ValueError("No PDF peaks found >= min_density; "
                              "check pdf_peak_min_density / data quality.")
        peak_idx = int(peaks_idx[0])

        h_a_upper = self.compute_upper_cutoff(grid, pdf, peak_idx)
        h_a_lower = self.compute_lower_cutoff(grid, pdf, peak_idx, h_a_upper)

        candidates = pixc_df[(pixc_df["h_a"] >= h_a_lower) &
                              (pixc_df["h_a"] <= h_a_upper)].reset_index(drop=True)
        candidates.attrs.update({
            "grid": grid, "pdf": pdf, "peak_idx": peak_idx,
            "h_a_lower": h_a_lower, "h_a_upper": h_a_upper,
        })
        return candidates

    def build_water_extent_mask(self, per_cycle_filtered_pixc: Sequence[pd.DataFrame],
                                 grid_res_deg: Optional[float] = None,
                                 validity_quantile: Optional[float] = None):
        """Build and cache the region's water extent polygon from per-cycle pixel validity."""
        print('step 5-1')
        grid_res_deg = grid_res_deg or self.cfg.mask_grid_res_deg
        validity_quantile = validity_quantile if validity_quantile is not None \
            else self.cfg.mask_validity_quantile

        all_lat = np.concatenate([df["latitude"].to_numpy() for df in per_cycle_filtered_pixc])
        all_lon = np.concatenate([df["longitude"].to_numpy() for df in per_cycle_filtered_pixc])
        lat_min, lat_max = all_lat.min(), all_lat.max()
        lon_min, lon_max = all_lon.min(), all_lon.max()

        lat_bins = np.arange(lat_min, lat_max + grid_res_deg, grid_res_deg)
        lon_bins = np.arange(lon_min, lon_max + grid_res_deg, grid_res_deg)

        total_validity = np.zeros((len(lat_bins) - 1, len(lon_bins) - 1))
        print('step 5-2')
        for df in per_cycle_filtered_pixc:
            counts, _, _ = np.histogram2d(
                df["latitude"], df["longitude"], bins=[lat_bins, lon_bins]
            )
            total_validity += counts
        print('step 5-3')
        threshold = np.quantile(total_validity[total_validity > 0], validity_quantile) \
            if np.any(total_validity > 0) else 0
        high_validity_mask = total_validity >= threshold

        # Vectorize the boolean grid into a handful of merged polygons rather
        # than building one Polygon per True cell (which does not scale past
        # a few hundred thousand cells before unary_union grinds to a halt).
        print('step 5-4')
        transform = Affine.translation(lon_min, lat_min) * \
            Affine.scale(grid_res_deg, grid_res_deg)
        mask_u8 = high_validity_mask.astype(np.uint8)
        print('step 5-5')
        polygons = [
            _shapely_shape(geom)
            for geom, val in _rio_shapes(mask_u8, mask=high_validity_mask, transform=transform)
            if val == 1
        ]
        print('step 5-6')
        if not polygons:
            raise ValueError("No high-validity cells found; lower validity_quantile.")

        merged = unary_union(polygons).buffer(grid_res_deg).buffer(-grid_res_deg)
        print('step 5-7')
        if isinstance(merged, MultiPolygon):
            largest = max(merged.geoms, key=lambda p: p.area)
        else:
            largest = merged
        print('step 5-8')
        cleaned = self._clean_polygon(largest)
        self._water_extent_mask = cleaned
        return cleaned

    def _clean_polygon(self, polygon):
        """Drop small holes and lightly smooth a polygon's boundary."""
        min_hole_area = self.cfg.min_hole_area_deg2
        kept_interiors = [
            ring for ring in polygon.interiors
            if Polygon(ring).area >= min_hole_area
        ]
        cleaned = Polygon(polygon.exterior, kept_interiors)

        coords = np.array(cleaned.exterior.coords)
        w = self.cfg.smoothing_window
        if w > 1 and len(coords) > w:
            kernel = np.ones(w) / w
            smoothed = np.column_stack([
                np.convolve(coords[:, 0], kernel, mode="same"),
                np.convolve(coords[:, 1], kernel, mode="same"),
            ])
            cleaned = Polygon(smoothed, kept_interiors)

        return cleaned

    def apply_water_extent_mask(self, candidates_df: pd.DataFrame,
                                 mask_polygon=None) -> pd.DataFrame:
        """Keep only candidate pixels that fall inside the water extent mask."""
        mask_polygon = mask_polygon or self._water_extent_mask
        if mask_polygon is None:
            raise ValueError("No water_extent_mask provided or cached; "
                              "call build_water_extent_mask first.")

        df = candidates_df.copy()
        inside = vectorized.contains(mask_polygon, df["longitude"].to_numpy(),
                                      df["latitude"].to_numpy())

        return df[inside].reset_index(drop=True)

    def check_reference_point_classification(self, pixc_df: pd.DataFrame) -> bool:
        """Warn if the cached reference pixel looks like dark water."""
        ref_latlon = pixc_df.attrs.get("ref_pixel_latlon")
        if ref_latlon is None:
            warnings.warn("No reference pixel recorded on this DataFrame.")
            return False
        lat, lon = ref_latlon
        row = pixc_df.iloc[
            ((pixc_df["latitude"] - lat).abs() +
             (pixc_df["longitude"] - lon).abs()).idxmin()
        ]
        is_dark = row["classification"] in self._dark_water_class_codes()
        if is_dark:
            warnings.warn("Reference point classification looks like dark water; "
                           "the whole h_a PDF may be biased.")
        return not is_dark

    def filter_dark_water(self, pixc_df: pd.DataFrame,
                           enabled: Optional[bool] = None) -> pd.DataFrame:
        """Optionally drop pixels classified as dark water."""
        enabled = self.cfg.exclude_dark_water if enabled is None else enabled
        if not enabled:
            return pixc_df
        return pixc_df[~pixc_df["classification"].isin(
            self._dark_water_class_codes())].reset_index(drop=True)

    def remove_regional_gradient(self, pixc_df: pd.DataFrame,
                                  gradient_model=None) -> pd.DataFrame:
        """Subtract a fitted (or given) large-scale height gradient before computing anomalies."""
        df = pixc_df.copy()
        if gradient_model is None:
            A = np.column_stack([
                df["latitude"], df["longitude"], np.ones(len(df))
            ])
            coeffs, *_ = np.linalg.lstsq(A, df["height"], rcond=None)
            gradient_model = lambda lat, lon: (
                coeffs[0] * lat + coeffs[1] * lon + coeffs[2]
            )
        df["height"] = df["height"] - gradient_model(df["latitude"], df["longitude"])
        return df

    def estimate_pixel_uncertainty(self, intertidal_df: pd.DataFrame) -> pd.DataFrame:
        """Add sigma_h = |dh_dphi| * sigma_phase_noise to each pixel."""
        df = intertidal_df.copy()
        df["sigma_h"] = df["dh_dphi"].abs() * df["sigma_phase_noise"]
        return df

    def monte_carlo_median(self, heights: np.ndarray, sigmas: np.ndarray,
                            n_realizations: Optional[int] = None,
                            alpha: Optional[float] = None,
                            rng: Optional[np.random.Generator] = None):
        """Return a Monte Carlo median height and [alpha, 1-alpha] confidence interval."""
        n_realizations = n_realizations or self.cfg.mc_realizations
        alpha = alpha if alpha is not None else self.cfg.mc_ci_alpha
        rng = rng or np.random.default_rng()

        heights = np.asarray(heights)
        sigmas = np.asarray(sigmas)
        n = len(heights)
        if n == 0:
            return np.nan, (np.nan, np.nan)

        noise = rng.normal(loc=0.0, scale=sigmas, size=(n_realizations, n))
        perturbed = heights[None, :] + noise
        M = np.median(perturbed, axis=1)

        final_height = float(np.median(M))
        M_sorted = np.sort(M)
        lo = float(np.quantile(M_sorted, alpha))
        hi = float(np.quantile(M_sorted, 1 - alpha))
        return final_height, (lo, hi)

    def aggregate_to_grid(self, intertidal_df: pd.DataFrame,
                           grid_res_deg: Optional[float] = None,
                           rng: Optional[np.random.Generator] = None) -> pd.DataFrame:
        """Aggregate intertidal pixels into grid cells via Monte Carlo median."""
        grid_res_deg = grid_res_deg or self.cfg.output_grid_res_deg
        df = intertidal_df.copy()

        df["cell_lat"] = (df["latitude"] // grid_res_deg) * grid_res_deg
        df["cell_lon"] = (df["longitude"] // grid_res_deg) * grid_res_deg

        results = []
        for (clat, clon), group in df.groupby(["cell_lat", "cell_lon"]):
            height, (lo, hi) = self.monte_carlo_median(
                group["height"].to_numpy(), group["sigma_h"].to_numpy(), rng=rng
            )
            results.append({
                "cell_lat": clat + grid_res_deg / 2,
                "cell_lon": clon + grid_res_deg / 2,
                "n_pixels": len(group),
                "height": height,
                "ci_low": lo,
                "ci_high": hi,
            })

        return pd.DataFrame(results)

    def stack_multi_cycle(self, intertidal_dfs: Sequence[pd.DataFrame],
                           grid_res_deg: Optional[float] = None,
                           rng: Optional[np.random.Generator] = None) -> pd.DataFrame:
        """Pool pixels from multiple cycles and aggregate them into one grid."""
        combined = pd.concat(intertidal_dfs, ignore_index=True)
        return self.aggregate_to_grid(combined, grid_res_deg=grid_res_deg, rng=rng)

    def validate_against_dem(self, grid_df: pd.DataFrame, dem_path: str,
                              stratify_by: Optional[str] = None) -> pd.DataFrame:
        """Compare gridded heights to a reference DEM and report bias/std/RMSE."""
        with rasterio.open(dem_path) as src:
            rows, cols = rowcol(src.transform,
                                 grid_df["cell_lon"].to_numpy(),
                                 grid_df["cell_lat"].to_numpy())
            dem_band = src.read(1)
            valid = (
                (np.array(rows) >= 0) & (np.array(rows) < dem_band.shape[0]) &
                (np.array(cols) >= 0) & (np.array(cols) < dem_band.shape[1])
            )
            ref_height = np.full(len(grid_df), np.nan)
            ref_height[valid] = dem_band[
                np.array(rows)[valid], np.array(cols)[valid]
            ]

        df = grid_df.copy()
        df["ref_height"] = ref_height
        df["diff"] = df["height"] - df["ref_height"]
        df = df.dropna(subset=["diff"])

        def _summary(sub: pd.DataFrame) -> pd.Series:
            """Return bias/std/RMSE summary stats for a subset of cells."""
            return pd.Series({
                "mean_bias": sub["diff"].mean(),
                "median_bias": sub["diff"].median(),
                "std": sub["diff"].std(),
                "rmse": np.sqrt((sub["diff"] ** 2).mean()),
                "n": len(sub),
            })

        if stratify_by and stratify_by in df.columns:
            summary = df.groupby(stratify_by).apply(_summary).reset_index()
        else:
            summary = _summary(df).to_frame().T

        summary.attrs["per_cell_diffs"] = df
        return summary

    def run_pipeline(self, filepaths_by_cycle: dict, ref_lat: float, ref_lon: float,
                      bbox: Optional[tuple] = None,
                      dem_path: Optional[str] = None) -> dict:
        """Run the full pipeline across cycles and return intermediate and final products."""
        per_cycle_filtered = {}
        per_cycle_intertidal = {}

        for cycle, fp in filepaths_by_cycle.items():
            pixc = self.read_pixel_cloud(fp, cycle=cycle, bbox=bbox)
            pixc = self.compute_height_anomaly(pixc, ref_lat=ref_lat, ref_lon=ref_lon)
            self.check_reference_point_classification(pixc)
            pixc = self.filter_dark_water(pixc)
            filtered = self.filter_phase_noise(pixc)
            per_cycle_filtered[cycle] = filtered

            candidates = self.filter_open_water(filtered)
            per_cycle_intertidal[cycle] = candidates

        mask = self.build_water_extent_mask(list(per_cycle_filtered.values()))

        final_intertidal = {}
        for cycle, candidates in per_cycle_intertidal.items():
            intertidal = self.apply_water_extent_mask(candidates, mask)
            intertidal = self.estimate_pixel_uncertainty(intertidal)
            final_intertidal[cycle] = intertidal

        grids = {
            cycle: self.aggregate_to_grid(df)
            for cycle, df in final_intertidal.items()
        }

        result = {
            "water_extent_mask": mask,
            "per_cycle_filtered": per_cycle_filtered,
            "per_cycle_intertidal": final_intertidal,
            "per_cycle_grids": grids,
        }

        if dem_path:
            result["validation"] = {
                cycle: self.validate_against_dem(grid, dem_path)
                for cycle, grid in grids.items()
            }

        return result
    
    def read_pixel_cloud_arrays(self, filepath: str, group: Optional[str] = None) -> dict:
        """Read the raw per-pixel arrays needed for az/range grid masking."""
        ds = self._open_pixc_group(xr, filepath, group)

        arrays = {
            "az": ds.azimuth_index.values.astype(int),
            "rg": ds.range_index.values.astype(int),
            "classification": ds.classification.values,
            "lat": ds.latitude.values,
            "lon": ds.longitude.values,
        }
        for qual_name in self.cfg.quality_flag_names:
            arrays[qual_name] = ds[qual_name].values

        return arrays

    def build_land_water_intertidal_grids(self, arrays: dict) -> dict:
        """Scatter flat pixel arrays into the azimuth/range grid, clean the
        land mask, and derive the final water/intertidal keep-mask."""
        az, rg = arrays["az"], arrays["rg"]
        classification = arrays["classification"]

        az_min, az_max = az.min(), az.max()
        rg_min, rg_max = rg.min(), rg.max()
        n_az = az_max - az_min + 1
        n_rg = rg_max - rg_min + 1

        # confidently-classified land pixels
        land = (
            (classification == self.cfg.land_class_code)
            & (arrays["classification_qual"] == 0)
        )

        land_grid = np.zeros((n_az, n_rg), dtype=bool)
        land_grid[az - az_min, rg - rg_min] = land

        populated = np.zeros((n_az, n_rg), dtype=bool)
        populated[az - az_min, rg - rg_min] = True

        # clean up the land mask
        BW = land_grid | ~populated
        se = disk(self.cfg.land_closing_disk_radius)
        BW1 = remove_small_objects(
            BW, min_size=self.cfg.land_min_object_size,
            connectivity=self.cfg.land_connectivity,
        )
        BW2 = binary_closing(BW1, se)
        BW3 = binary_fill_holes(BW2)

        not_land_clean = ~BW3

        # a pixel only counts as "good quality" if every configured flag reads 0
        quality = np.ones(classification.shape, dtype=bool)
        for qual_name in self.cfg.quality_flag_names:
            quality &= (arrays[qual_name] == 0)

        classification_grid = np.zeros((n_az, n_rg), dtype=np.uint8)
        classification_grid[az - az_min, rg - rg_min] = classification

        quality_grid = np.zeros((n_az, n_rg), dtype=bool)
        quality_grid[az - az_min, rg - rg_min] = quality

        keep_codes = list(self.cfg.water_class_codes) + list(self.cfg.intertidal_class_codes)
        final_mask = (
            populated & quality_grid & not_land_clean
            & np.isin(classification_grid, keep_codes)
        )

        lon_grid = np.full((n_az, n_rg), np.nan)
        lat_grid = np.full((n_az, n_rg), np.nan)
        lon_grid[az - az_min, rg - rg_min] = arrays["lon"]
        lat_grid[az - az_min, rg - rg_min] = arrays["lat"]

        return {
            "az_min": az_min, "rg_min": rg_min, "n_az": n_az, "n_rg": n_rg,
            "populated": populated,
            "land_grid_cleaned": BW3,
            "not_land_clean": not_land_clean,
            "quality_grid": quality_grid,
            "classification_grid": classification_grid,
            "final_mask": final_mask,
            "lon_grid": lon_grid,
            "lat_grid": lat_grid,
        }

    def polygons_from_grid(self, category: str, codes: Sequence[int], grids: dict):
        """Convex-hull polygons for each connected region of `codes` pixels.

        Mirrors the `export()` function from the standalone script: label
        connected components in the final keep-mask restricted to `codes`,
        then build a convex-hull polygon (in EPSG:4326) per region.
        """
        final_mask = grids["final_mask"]
        classification_grid = grids["classification_grid"]
        lon_grid = grids["lon_grid"]
        lat_grid = grids["lat_grid"]

        category_mask = final_mask & np.isin(classification_grid, list(codes))
        if not np.any(category_mask):
            warnings.warn(f"No {category} pixels found.")
            return gpd.GeoDataFrame(
                columns=["category", "region_id", "num_points", "area",
                         "cent_lon", "cent_lat", "geometry"],
                geometry="geometry", crs="EPSG:4326",
            )

        labelled_array, num_features = label(category_mask, structure=np.ones((3, 3)))

        records = []
        for region_id in range(1, num_features + 1):
            region_mask = labelled_array == region_id

            xs = lon_grid[region_mask]
            ys = lat_grid[region_mask]
            valid = ~np.isnan(xs) & ~np.isnan(ys)
            xs, ys = xs[valid], ys[valid]

            if len(xs) < self.cfg.min_region_points:
                continue

            polygon = MultiPoint(list(zip(xs, ys))).convex_hull
            centroid = polygon.centroid

            records.append({
                "category": category,
                "region_id": region_id,
                "num_points": len(xs),
                "area": polygon.area,
                "cent_lon": centroid.x,
                "cent_lat": centroid.y,
                "geometry": polygon,
            })

        return gpd.GeoDataFrame(records, crs="EPSG:4326")

    def export_polygons_shapefile(self, gdf, category: str, output_dir: str) -> str:
        """Write a polygon GeoDataFrame to `<output_dir>/<category>_polygon.shp`."""
        os.makedirs(output_dir, exist_ok=True)
        shp_path = os.path.join(output_dir, f"{category}_polygon.shp")
        gdf.to_file(shp_path)
        print(f"Wrote {len(gdf)} {category} polygons to {shp_path}")
        return shp_path

    def export_polygons_kml(self, gdf, category: str, output_dir: str) -> str:
        """Write a polygon GeoDataFrame to `<output_dir>/<category>_polygon.kml`."""
        os.makedirs(output_dir, exist_ok=True)
        color_by_category = {"water": "steelblue", "intertidal": "darkorange"}
        line_color = color_by_category.get(category, "steelblue")

        kml = simplekml.Kml()
        for _, row in gdf.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue

            polys = [geom] if geom.geom_type == "Polygon" else list(geom.geoms)
            for i, poly in enumerate(polys):
                pol = kml.newpolygon(
                    name=f"{category}_{row.region_id}" + (f"_{i}" if len(polys) > 1 else "")
                )
                pol.outerboundaryis = list(poly.exterior.coords)
                if poly.interiors:
                    pol.innerboundaryis = [list(ring.coords) for ring in poly.interiors]
                pol.style.linestyle.color = getattr(simplekml.Color, line_color, None) or "ff1478b3"
                pol.style.linestyle.width = 2
                pol.style.polystyle.fill = 0

        kml_path = os.path.join(output_dir, f"{category}_polygon.kml")
        kml.save(kml_path)
        print(f"Wrote {len(gdf)} {category} polygons to {kml_path}")
        return kml_path

    def export_bbox_kml(self, lon: np.ndarray, lat: np.ndarray,
                         filepath: str, output_base: str,
                         name: str = "SWOT PIXC Swath") -> str:
        """Write a rectangular bounding-box KML around a cloud of lon/lat points.

        Reusable version of the standalone "KML of the PIXC swath using a
        Bounding Box" script: flattens the coordinates, drops NaNs/out-of-range/
        duplicate points, takes the min/max lon/lat, and writes the resulting
        rectangle as a single KML polygon. Handy for quickly drawing a rough
        AOI box in Google Earth that can then be hand-edited into a tighter
        polygon and fed back in via `subset_by_kml`.

        The output goes to `<output_base>/<granule_name>/swath_bbox.kml`,
        via `make_output_directory`, same convention as every other output
        this class writes.

        Parameters
        ----------
        lon, lat : array-like
            Longitude/latitude values, any shape (will be flattened), e.g.
            `arrays["lon"], arrays["lat"]` from `read_pixel_cloud_arrays`,
            or `pixc_df["longitude"], pixc_df["latitude"]`.
        filepath : str
            Path to the source PIXC granule (used to name the output folder).
        output_base : str
            Base output directory; the granule-named subfolder is created
            under this via `make_output_directory`.
        name : str
            Name label for the KML polygon placemark.
        """
        lon = np.asarray(lon).ravel()
        lat = np.asarray(lat).ravel()

        points = np.column_stack((lon, lat))
        points = points[np.isfinite(points).all(axis=1)]
        points = points[
            (points[:, 0] >= -180) & (points[:, 0] <= 180) &
            (points[:, 1] >= -90) & (points[:, 1] <= 90)
        ]
        points = np.unique(points, axis=0)

        if len(points) < 4:
            raise ValueError("Not enough valid points to build a swath boundary.")

        min_lon, max_lon = points[:, 0].min(), points[:, 0].max()
        min_lat, max_lat = points[:, 1].min(), points[:, 1].max()

        boundary = np.array([
            [min_lon, min_lat],
            [max_lon, min_lat],
            [max_lon, max_lat],
            [min_lon, max_lat],
            [min_lon, min_lat],
        ])

        kml = simplekml.Kml()
        polygon = kml.newpolygon(name=name)
        polygon.outerboundaryis = [(float(x), float(y)) for x, y in boundary]
        polygon.style.linestyle.width = 3
        polygon.style.polystyle.fill = 0

        output_dir = self.make_output_directory(filepath, output_base)
        output_path = os.path.join(output_dir, "swath_bbox.kml")
        kml.save(output_path)

        print(f"Boundary vertices: {len(boundary)}")
        print(f"Longitude: {min_lon} {max_lon}")
        print(f"Latitude: {min_lat} {max_lat}")
        print(f"Saved bounding-box KML ({len(points)} pts) to {output_path}")
        return output_path

    def subset_by_kml(self, pixc_df: pd.DataFrame, kml_path: str,
                       lon_col: str = "longitude",
                       lat_col: str = "latitude") -> pd.DataFrame:
        """Subset a pixel-cloud DataFrame to points that fall inside a KML polygon.

        Reusable version of the standalone "mask PIXC points against a KML
        boundary" script: reads the (first) geometry out of `kml_path` —
        typically a polygon you drew/edited in Google Earth, or one produced
        by `export_bbox_kml` — and keeps only the rows of `pixc_df` whose
        (lon, lat) fall inside (or on the boundary of) that polygon.

        Uses `shapely.vectorized` instead of a per-row `Point(...).covers()`
        loop, which is dramatically faster on full pixel clouds (10^5-10^6 rows).

        Parameters
        ----------
        pixc_df : pd.DataFrame
            Pixel-cloud DataFrame, e.g. from `read_pixel_cloud`. Must have
            `lon_col`/`lat_col` columns.
        kml_path : str
            Path to the boundary .kml (polygon or closed line string).
        lon_col, lat_col : str
            Column names holding longitude/latitude in `pixc_df`.

        Returns
        -------
        pd.DataFrame
            The subset of `pixc_df` inside the KML boundary (index reset).
        """
        kml_gdf = gpd.read_file(kml_path)
        if kml_gdf.empty:
            raise ValueError(f"No geometry found in {kml_path}.")

        geom = kml_gdf.geometry.iloc[0]
        if geom.geom_type == "LineString":
            poly = Polygon(geom.coords)
        elif geom.geom_type == "Polygon":
            poly = geom
        elif geom.geom_type in ("MultiPolygon", "GeometryCollection"):
            poly = unary_union(list(geom.geoms))
        else:
            raise ValueError(f"Unsupported KML geometry type: {geom.geom_type}")

        lon = pixc_df[lon_col].to_numpy(dtype=float)
        lat = pixc_df[lat_col].to_numpy(dtype=float)

        # contains + touches ~= shapely's covers() (interior OR boundary),
        # but vectorized over the whole array instead of a Python loop.
        mask = vectorized.contains(poly, lon, lat) | vectorized.touches(poly, lon, lat)

        subset = pixc_df[mask].reset_index(drop=True)
        print(f"Kept {int(mask.sum())}/{len(pixc_df)} points inside {kml_path}")
        return subset

    def rasterize_category_polygons(self, gdf, output_path: str,
                                     resolution_deg: Optional[float] = None,
                                     burn_value: int = 1) -> str:
        """Burn a polygon GeoDataFrame into a single-band GeoTIFF using rasterio.

        Useful for turning the water/intertidal polygons into a raster mask
        (e.g. for overlaying on a DEM, or for further raster analysis).
        """
        resolution_deg = resolution_deg or self.cfg.raster_default_resolution_deg

        if gdf is None or len(gdf) == 0:
            raise ValueError("Cannot rasterize an empty GeoDataFrame.")

        minx, miny, maxx, maxy = gdf.total_bounds
        width = max(int(np.ceil((maxx - minx) / resolution_deg)), 1)
        height = max(int(np.ceil((maxy - miny) / resolution_deg)), 1)
        transform = from_bounds(minx, miny, maxx, maxy, width, height)

        raster = rasterize(
            [(geom, burn_value) for geom in gdf.geometry if geom is not None and not geom.is_empty],
            out_shape=(height, width),
            transform=transform,
            fill=0,
            dtype="uint8",
        )

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with rasterio.open(
            output_path, "w",
            driver="GTiff",
            height=height, width=width, count=1,
            dtype="uint8", crs="EPSG:4326", transform=transform,
        ) as dst:
            dst.write(raster, 1)

        print(f"Wrote raster mask ({width}x{height}) to {output_path}")
        return output_path

    def sample_raster_at_points(self, raster_path: str,
                                 lons: np.ndarray, lats: np.ndarray) -> np.ndarray:
        """Sample a single-band raster (e.g. a DEM or the rasterized mask) at points."""
        with rasterio.open(raster_path) as src:
            values = np.array([v[0] for v in src.sample(zip(lons, lats))], dtype=float)
        return values

    def pixel_category_flat(self, arrays: dict, grids: dict) -> np.ndarray:
        """Per-pixel category code in flat (original) pixel order.

        0 = excluded (land or failed quality/classification), 1 = water,
        2 = intertidal. Matches the category scheme used by
        `plot_water_intertidal_mask`.
        """
        az, rg = arrays["az"], arrays["rg"]
        classification = arrays["classification"]
        az_min, rg_min = grids["az_min"], grids["rg_min"]

        land_pixel = grids["land_grid_cleaned"][az - az_min, rg - rg_min]
        not_land_pixel = ~land_pixel

        quality = np.ones(classification.shape, dtype=bool)
        for qual_name in self.cfg.quality_flag_names:
            quality &= (arrays[qual_name] == 0)

        water_pixel = quality & not_land_pixel & np.isin(classification, list(self.cfg.water_class_codes))
        intertidal_pixel = quality & not_land_pixel & np.isin(classification, list(self.cfg.intertidal_class_codes))

        category = np.zeros(classification.shape, dtype=int)
        category[water_pixel] = 1
        category[intertidal_pixel] = 2
        return category

    def plot_water_intertidal_mask(self, arrays: dict, grids: dict, ax=None):
        """Scatter-plot water/intertidal pixels in lon/lat, in the same style
        used for the classification scatter plots (discrete colormap +
        labeled colorbar, cos-latitude aspect correction)."""
        category = self.pixel_category_flat(arrays, grids)
        lon, lat = arrays["lon"], arrays["lat"]
        plot_valid = category > 0

        class_labels = ["Water", "Intertidal"]
        plot_colors = ["steelblue", "darkorange"]
        cmap = mcolors.ListedColormap(plot_colors)
        norm = mcolors.BoundaryNorm([0.5, 1.5, 2.5], cmap.N)

        if ax is None:
            plt.figure(figsize=(8, 8))
            ax = plt.gca()

        sc = ax.scatter(
            lon[plot_valid], lat[plot_valid],
            c=category[plot_valid], s=2, cmap=cmap, norm=norm,
        )

        cbar = plt.colorbar(sc, ax=ax, ticks=[1, 2])
        cbar.set_label("Class")
        cbar.set_ticklabels(class_labels)

        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title("SWOT PIXC Water / Intertidal Mask")

        mean_lat = np.deg2rad(np.mean(lat[plot_valid])) if np.any(plot_valid) else 0.0
        ax.set_aspect(1 / np.cos(mean_lat))

        plt.tight_layout()
        return ax
    
    def make_output_directory(self, filepath: str, output_base: str) -> str:
        """
        Create an output directory based on the PIXC filename.

        Example
        -------
        SWOT_L2_HR_PIXC_052_475_245R_20260706T065928_20260706T065939_PID0_01.nc

        becomes

        <output_base>/
            052_475_245R_20260706T065928_20260706T065939_PID0_01/
        """

        stem = os.path.splitext(os.path.basename(filepath))[0]
        prefix = "SWOT_L2_HR_PIXC_"
        name = stem[len(prefix):] if stem.startswith(prefix) else stem
        output_dir = os.path.join(output_base, name)
        os.makedirs(output_dir, exist_ok=True)

        return output_dir
    def create_kml_subset(
        self,
        filepath: str,
        kml_path: str,
        output_base: str,
        cycle: Optional[int] = None,
        group: Optional[str] = None
):
        """
        Read a PIXC file, subset it using a KML polygon,
        save the subset to NetCDF, and return both the
        subset DataFrame and the output filename.
        """
        output_dir = self.make_output_directory(filepath, output_base)

        subset_file = os.path.join(output_dir, "subset.nc")

        pixc = self.read_pixel_cloud(
            filepath,
            cycle=cycle,
            group=group
        )

        subset = self.subset_by_kml(
            pixc,
            kml_path
        )

        xr.Dataset.from_dataframe(subset).to_netcdf(subset_file)

        print(f"Subset written to {subset_file}")

        return subset, subset_file, output_dir

    def run_polygon_export_pipeline(self, filepath: str, output_base: str,
                                     group: Optional[str] = None,
                                     make_plot: bool = True,
                                     make_rasters: bool = False,
                                     raster_resolution_deg: Optional[float] = None) -> dict:
        """End-to-end: read PIXC -> build masks -> export water/intertidal
        polygons as shapefile + KML -> (optionally) rasterize -> (optionally) plot.

        This is the reusable-pipeline equivalent of the standalone
        "SWOT PIXC to polygons with KML" script. `output_dir` is derived from
        `filepath`/`output_base` via `make_output_directory`, the same
        directory convention used by `create_kml_subset`.
        """
        output_dir = self.make_output_directory(filepath, output_base)

        arrays = self.read_pixel_cloud_arrays(filepath, group=group)
        grids = self.build_land_water_intertidal_grids(arrays)

        water_gdf = self.polygons_from_grid("water", self.cfg.water_class_codes, grids)
        intertidal_gdf = self.polygons_from_grid("intertidal", self.cfg.intertidal_class_codes, grids)

        result = {"arrays": arrays, "grids": grids, "water_gdf": water_gdf, "intertidal_gdf": intertidal_gdf}

        for category, gdf in (("water", water_gdf), ("intertidal", intertidal_gdf)):
            if gdf is None or len(gdf) == 0:
                continue
            result[f"{category}_shp"] = self.export_polygons_shapefile(gdf, category, output_dir)
            result[f"{category}_kml"] = self.export_polygons_kml(gdf, category, output_dir)
            if make_rasters:
                raster_path = os.path.join(output_dir, f"{category}_mask.tif")
                result[f"{category}_tif"] = self.rasterize_category_polygons(
                    gdf, raster_path, resolution_deg=raster_resolution_deg
                )

        if make_plot:
            self.plot_water_intertidal_mask(arrays, grids)
            plt.show()

        return result
