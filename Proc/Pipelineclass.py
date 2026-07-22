import warnings
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats
import xarray as xr


@dataclass
class SWOTPipelineConfig:
    """Tunable thresholds for the pipeline, defaults from the paper's pseudocode."""

    sigma_phase_noise_threshold: float = 0.08
    ref_point_buffer_deg: float = 0.001
    pdf_bin_width_m: float = 0.05
    pdf_peak_min_density: float = 0.10
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
        #try:
            #import xarray as xr
        #except ImportError as e:
            #raise ImportError(
            #     "reading L2_HR_PIXC granules requires xarray + netCDF4/h5netcdf. "
            #     "Install with `pip install xarray netCDF4`."
            # ) from e

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

        n_bins = max(int(np.ceil((h_a.max() - h_a.min()) / bin_width)), 10)
        grid = np.linspace(h_a.min(), h_a.max(), n_bins)

        kde = stats.gaussian_kde(h_a)
        pdf = kde(grid)
        return grid, pdf, kde

    def find_pdf_peaks(self, grid: np.ndarray, pdf: np.ndarray,
                        min_density: Optional[float] = None) -> np.ndarray:
        """Return indices of PDF local maxima with density >= min_density."""
        from scipy.signal import argrelextrema
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

        from scipy.signal import argrelextrema
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
        try:
            from shapely.geometry import Polygon, MultiPolygon
            from shapely.ops import unary_union
        except ImportError as e:
            raise ImportError(
                "build_water_extent_mask requires shapely. "
                "Install with `pip install shapely`."
            ) from e

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

        for df in per_cycle_filtered_pixc:
            counts, _, _ = np.histogram2d(
                df["latitude"], df["longitude"], bins=[lat_bins, lon_bins]
            )
            total_validity += counts

        threshold = np.quantile(total_validity[total_validity > 0], validity_quantile) \
            if np.any(total_validity > 0) else 0
        high_validity_mask = total_validity >= threshold

        polygons = []
        for i in range(high_validity_mask.shape[0]):
            for j in range(high_validity_mask.shape[1]):
                if high_validity_mask[i, j]:
                    polygons.append(Polygon([
                        (lon_bins[j], lat_bins[i]),
                        (lon_bins[j + 1], lat_bins[i]),
                        (lon_bins[j + 1], lat_bins[i + 1]),
                        (lon_bins[j], lat_bins[i + 1]),
                    ]))

        if not polygons:
            raise ValueError("No high-validity cells found; lower validity_quantile.")

        merged = unary_union(polygons).buffer(grid_res_deg).buffer(-grid_res_deg)

        if isinstance(merged, MultiPolygon):
            largest = max(merged.geoms, key=lambda p: p.area)
        else:
            largest = merged

        cleaned = self._clean_polygon(largest)
        self._water_extent_mask = cleaned
        return cleaned

    def _clean_polygon(self, polygon):
        """Drop small holes and lightly smooth a polygon's boundary."""
        from shapely.geometry import Polygon

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
        try:
            from shapely.geometry import Point
            from shapely import vectorized
            has_vectorized = True
        except ImportError:
            from shapely.geometry import Point
            has_vectorized = False

        mask_polygon = mask_polygon or self._water_extent_mask
        if mask_polygon is None:
            raise ValueError("No water_extent_mask provided or cached; "
                              "call build_water_extent_mask first.")

        df = candidates_df.copy()
        if has_vectorized:
            inside = vectorized.contains(mask_polygon, df["longitude"].to_numpy(),
                                          df["latitude"].to_numpy())
        else:
            inside = df.apply(
                lambda r: mask_polygon.contains(Point(r["longitude"], r["latitude"])),
                axis=1
            ).to_numpy()

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
        try:
            import rasterio
            from rasterio.transform import rowcol
        except ImportError as e:
            raise ImportError(
                "validate_against_dem requires rasterio. "
                "Install with `pip install rasterio`."
            ) from e

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
