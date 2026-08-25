import warnings
from dataclasses import dataclass, field
from typing import Optional, Sequence
import numpy as np
import pandas as pd
import shapely
from scipy import stats
import xarray as xr
from rasterio.features import shapes as _rio_shapes
from affine import Affine
from shapely.geometry import shape as _shapely_shape
from scipy.signal import argrelextrema
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
try:
    from shapely import coverage_union_all as _coverage_union_all
except ImportError:
    _coverage_union_all = None
from shapely.geometry import Point
from shapely import vectorized
from skimage.morphology import remove_small_objects, closing as binary_closing, disk
from scipy.ndimage import binary_fill_holes
from scipy.ndimage import distance_transform_edt
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
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from config_loader import load_config

@dataclass
class SWOTIntertidalPipeline:
    """SWOT L2_HR_PIXC to intertidal topography pipeline, following the paper's method."""

    def __init__(self, config: Optional[load_config] = None):
        """Store the config (or defaults) and reset the cached water extent mask."""
        self.cfg = load_config("config.xml")
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

    def filter_quality_flags(self, pixc_df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
        """Keep only pixels where geolocation_qual, interferogram_qual, and
        sig0_qual are all 0. 
        """
        required = ["geolocation_qual", "interferogram_qual", "sig0_qual"]
        missing = [c for c in required if c not in pixc_df.columns]
        if missing:
            raise ValueError(
                f"filter_quality_flags: missing required column(s): {missing}. ")

        keep = (
            (pixc_df["geolocation_qual"] == 0) &
            (pixc_df["interferogram_qual"] == 0) &
            (pixc_df["sig0_qual"] == 0)
        )

        if verbose:
            for col in required:
                n_zero = int((pixc_df[col] == 0).sum())
                print(f"    {col} == 0: {n_zero} / {len(pixc_df)} pixels")

        filtered = pixc_df[keep].reset_index(drop=True)

        if verbose:
            print(
                f"Step 0 - filter_quality_flags: {len(filtered)} / {len(pixc_df)} "
                "pixels kept (geolocation_qual, interferogram_qual, sig0_qual "
                "all == 0; classification not checked)"
            )

        if filtered.empty:
            warnings.warn(
                "filter_quality_flags: 0 pixels passed. These qual fields "
                "are bitmasks, not booleans -- run "
                "pixc_df['geolocation_qual'].value_counts() (and the same "
                "for interferogram_qual / sig0_qual) to see which bits are "
                "actually set before assuming this is a bug."
            )

        return filtered

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

    def filter_phase_noise(self, mask, pixc_df: pd.DataFrame,
                            threshold: Optional[float] = None) -> pd.DataFrame:
        """Drop pixels with sigma_phase_noise above `threshold`, folding in
        the keep-mask from the previous pipeline step.
        """
        threshold = threshold if threshold is not None \
            else self.cfg.sigma_phase_noise_threshold

        mask_series = mask if isinstance(mask, pd.Series) \
            else pd.Series(np.asarray(mask), index=range(len(mask)))

        own_cond = pixc_df["sigma_phase_noise"] <= threshold
        combined = mask_series.loc[pixc_df.index] & own_cond

        phase_noise_mask = mask_series.copy()
        phase_noise_mask.loc[pixc_df.index] = combined

        #phase_noise_mask = pd.Series(False, index=mask_series.index)
        #phase_noise_mask.loc[pixc_df.index] = combined

        thres = pixc_df[combined]
        return thres, phase_noise_mask

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
            rng = np.random.default_rng(self.cfg.kde_random_seed)
            h_a = rng.choice(h_a, size=max_samples, replace=False)

        n_bins = max(int(np.ceil((h_a.max() - h_a.min()) / bin_width)), 10)
        n_bins = min(n_bins, self.cfg.pdf_max_bins)
        grid = np.linspace(h_a.min(), h_a.max(), n_bins)

        kde = stats.gaussian_kde(h_a, bw_method=self.cfg.kde_bw_scalar)
        pdf = kde(grid)
        return grid, pdf, kde, h_a

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

    def filter_open_water(self, pixc_df: pd.DataFrame, phase_noise_mask) -> pd.DataFrame: 
        """Return candidate non-open-water pixels (h_a within the Step 4 cutoffs)."""
        h_a = pixc_df["h_a"].to_numpy()
        grid, pdf, h_a, _ = self._kde_pdf(h_a)

        peaks_idx = self.find_pdf_peaks(grid, pdf)
        if peaks_idx.size == 0:
            raise ValueError("No PDF peaks found >= min_density; "
                              "check pdf_peak_min_density / data quality.")
        peak_idx = int(peaks_idx[0])

        h_a_upper = self.compute_upper_cutoff(grid, pdf, peak_idx)
        h_a_lower = self.compute_lower_cutoff(grid, pdf, peak_idx, h_a_upper)

        own_cond = (pixc_df["h_a"] >= h_a_lower) & (pixc_df["h_a"] <= h_a_upper)

        mask_series = phase_noise_mask if isinstance(phase_noise_mask, pd.Series) \
            else pd.Series(np.asarray(phase_noise_mask), index=range(len(phase_noise_mask)))

        combined = mask_series.loc[pixc_df.index] & own_cond

        open_water_mask = mask_series.copy()
        open_water_mask.loc[pixc_df.index] = combined

        #open_water_mask = pd.Series(False, index=mask_series.index)
        #open_water_mask.loc[pixc_df.index] = combined

        candidates = pixc_df[combined]

        candidates.attrs.update({
            "grid": grid, "pdf": pdf, "peak_idx": peak_idx,
            "h_a_lower": h_a_lower, "h_a_upper": h_a_upper,
        })
        return candidates, open_water_mask

    def build_water_extent_mask(self, open_water_mask, per_cycle_filtered_pixc: Sequence[pd.DataFrame],
                                 grid_res_deg: Optional[float] = None,
                                 validity_quantile: Optional[float] = None):
        """Build and cache the region's water extent polygon from per-cycle pixel validity."""
        grid_res_deg = grid_res_deg or self.cfg.mask_grid_res_deg
        validity_quantile = validity_quantile if validity_quantile is not None \
            else self.cfg.mask_validity_quantile

        if isinstance(open_water_mask, (list, tuple)):
            # one mask per cycle
            if len(open_water_mask) != len(per_cycle_filtered_pixc):
                raise ValueError(
                    f"build_water_extent_mask: got {len(open_water_mask)} masks for "
                    f"{len(per_cycle_filtered_pixc)} per-cycle dataframes; need one mask per cycle."
                )
            masks = open_water_mask
        else:
            masks = [open_water_mask] * len(per_cycle_filtered_pixc)

        per_cycle_filtered_pixc = [
            df[(mask if isinstance(mask, pd.Series)
                else pd.Series(np.asarray(mask), index=range(len(mask)))).reindex(df.index, fill_value=False)]
            for df, mask in zip(per_cycle_filtered_pixc, masks)
        ]

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
       

        se = disk(self.cfg.mask_closing_disk_radius)
        closed_mask = binary_closing(high_validity_mask, se)
        denoised_mask = remove_small_objects(
            closed_mask, min_size=self.cfg.mask_min_object_size,
            connectivity=self.cfg.mask_connectivity,
        )

        transform = Affine.translation(lon_min, lat_min) * \
            Affine.scale(grid_res_deg, grid_res_deg)
        mask_u8 = denoised_mask.astype(np.uint8)

        polygons = [
            _shapely_shape(geom)
            for geom, val in _rio_shapes(
                mask_u8, mask=denoised_mask, transform=transform, connectivity=8,
            )
            if val == 1
        ]
        
        if not polygons:
            raise ValueError("No high-validity cells found; lower validity_quantile.")

        polygons = gpd.GeoSeries(polygons).buffer(0)

        if _coverage_union_all is not None:
            try:
                merged = _coverage_union_all(polygons.values)
            except Exception:
                merged = unary_union(polygons.values)
        else:
            merged = unary_union(polygons.values)

        # Clean the merged geometry but keep all disconnected polygons
        if isinstance(merged, MultiPolygon):
            cleaned = MultiPolygon(
                [self._clean_polygon(poly) for poly in merged.geoms]
            )
        else:
            cleaned = self._clean_polygon(merged)

        self._water_extent_mask = cleaned
        return cleaned
        
        # if isinstance(merged, MultiPolygon):
        #     largest = max(merged.geoms, key=lambda p: p.area)
        # else:
        #     largest = merged
        
        # cleaned = self._clean_polygon(largest)
        # self._water_extent_mask = cleaned
        # return cleaned

    def polygons_from_binary_mask(
        self,
        binary_mask,
        lon_grid,
        lat_grid,
        category="intertidal",
    ):
        """Convert a binary mask into a GeoDataFrame of polygons."""

        labelled_array, num_features = label(binary_mask, structure=np.ones((3, 3)))

        records = []

        for region_id in range(1, num_features + 1):
            region_mask = labelled_array == region_id

            xs = lon_grid[region_mask]
            ys = lat_grid[region_mask]

            valid = ~np.isnan(xs) & ~np.isnan(ys)
            xs = xs[valid]
            ys = ys[valid]

            if len(xs) < self.cfg.min_region_points:
                continue

            polygon = MultiPoint(list(zip(xs, ys))).convex_hull

            if polygon.is_empty:
                continue

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

        if len(records) == 0:
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

        return gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:4326")

    def polygons_from_raster_mask(
        self,
        binary_mask,
        lon_grid,
        lat_grid,
        category="intertidal",
        fill_holes=True,
        corner_fill_max_radius=3,
        verbose=True,
        output_dir: Optional[str] = None,
        show: bool = False,
    ):
        """Vectorise a binary az/range mask into polygons that trace each
        region's actual pixel boundary, instead of convex-hulling its
        points like `polygons_from_binary_mask` does.
        """
        mask_bool = np.asarray(binary_mask, dtype=bool)
        if fill_holes:
            mask_bool = binary_fill_holes(mask_bool, structure=np.ones((3, 3)))

        if output_dir is not None:
            self.plot_step(
                mask_bool, f"{category}_mask_bool", output_dir,
                title=f"{category.title()}: mask_bool (post fill_holes={fill_holes})",
                show=show,
            )

        lon_filled, lat_filled, n_filled = self._fill_grid_nearest(
            lon_grid, lat_grid, max_radius=corner_fill_max_radius
        )
        if verbose and n_filled > 0:
            print(
                f"    polygons_from_raster_mask: filled {n_filled} gap cell(s) in "
                f"lon/lat grid using nearest populated cell (max_radius="
                f"{corner_fill_max_radius})"
            )

        labelled_array, num_features = label(
            mask_bool, structure=np.ones((3, 3))
        )

        if output_dir is not None:
                    self.plot_step(
                        mask_bool, f"{category}_mask_bool", output_dir,
                        title=f"{category.title()}: mask_bool (post fill_holes={fill_holes})",
                        show=show,
                    )

        records = []
        n_dropped_pieces = 0
        for region_id in range(1, num_features + 1):
            region_mask = labelled_array == region_id
            num_points = int(region_mask.sum())
            if num_points < self.cfg.min_region_points:
                continue

            region_u8 = region_mask.astype(np.uint8)
            pixel_polys = [
                _shapely_shape(geom)
                for geom, val in _rio_shapes(region_u8, mask=region_mask, connectivity=8)
                if val == 1
            ]
            if not pixel_polys:
                continue
            pixel_polygon = pixel_polys[0] if len(pixel_polys) == 1 else unary_union(pixel_polys)

            lonlat_polygon, dropped = self._pixel_polygon_to_lonlat(
                pixel_polygon, lon_filled, lat_filled
            )
            n_dropped_pieces += dropped
            if lonlat_polygon is None or lonlat_polygon.is_empty:
                continue

            centroid = lonlat_polygon.centroid
            records.append({
                "category": category,
                "region_id": region_id,
                "num_points": num_points,
                "area": lonlat_polygon.area,
                "cent_lon": centroid.x,
                "cent_lat": centroid.y,
                "geometry": lonlat_polygon,
            })

        if verbose and n_dropped_pieces > 0:
            print(
                f"    polygons_from_raster_mask: {n_dropped_pieces} polygon piece(s) "
                f"still dropped (vertex more than {corner_fill_max_radius} px from any "
                f"populated cell) -- consider raising corner_fill_max_radius"
            )

        if len(records) == 0:
            return gpd.GeoDataFrame(
                columns=[
                    "category", "region_id", "num_points",
                    "area", "cent_lon", "cent_lat", "geometry",
                ],
                geometry="geometry",
                crs="EPSG:4326",
            )

        return gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:4326")

    def _fill_grid_nearest(self, lon_grid, lat_grid, max_radius=3):
        """Fill NaN cells in lon_grid/lat_grid with the value of the
        nearest populated (non-NaN) cell, but only up to `max_radius`
        pixels away. Cells with no populated cell within that radius are
        left as NaN. Returns (lon_filled, lat_filled, n_filled).
        """
        populated = np.isfinite(lon_grid) & np.isfinite(lat_grid)

        if populated.all() or max_radius <= 0:
            return lon_grid, lat_grid, 0

        # distance (in pixels) and index of the nearest populated cell,
        # for every cell in the grid
        dist, (nearest_r, nearest_c) = distance_transform_edt(
            ~populated, return_distances=True, return_indices=True
        )

        fillable = (~populated) & (dist <= max_radius)
        n_filled = int(fillable.sum())
        if n_filled == 0:
            return lon_grid, lat_grid, 0

        lon_filled = lon_grid.copy()
        lat_filled = lat_grid.copy()
        lon_filled[fillable] = lon_grid[nearest_r[fillable], nearest_c[fillable]]
        lat_filled[fillable] = lat_grid[nearest_r[fillable], nearest_c[fillable]]
        return lon_filled, lat_filled, n_filled

    def _pixel_polygon_to_lonlat(self, pixel_polygon, lon_grid, lat_grid):
        """Remap a Polygon/MultiPolygon whose coordinates are (col, row)
        pixel-space vertices (as produced by `rasterio.features.shapes`
        with the default identity transform) into lon/lat space, using
        `lon_grid`/`lat_grid` to look up each vertex's real-world position.
        """
        def convert_ring(coords):
            return [self._pixel_corner_to_lonlat(x, y, lon_grid, lat_grid) for x, y in coords]

        polys = [pixel_polygon] if pixel_polygon.geom_type == "Polygon" else list(pixel_polygon.geoms)

        out_polys = []
        n_dropped = 0
        for poly in polys:
            exterior = convert_ring(poly.exterior.coords)
            if any(p is None for p in exterior):
                n_dropped += 1
                continue

            interiors, skip = [], False
            for ring in poly.interiors:
                conv = convert_ring(ring.coords)
                if any(p is None for p in conv):
                    skip = True
                    break
                interiors.append(conv)
            if skip:
                n_dropped += 1
                continue

            new_poly = Polygon(exterior, interiors)
            if not new_poly.is_valid:
                new_poly = new_poly.buffer(0)
            if not new_poly.is_empty:
                if isinstance(new_poly, MultiPolygon):
                    out_polys.extend(g for g in new_poly.geoms if not g.is_empty)
                else:
                    out_polys.append(new_poly)

        if not out_polys:
            return None, n_dropped
        if len(out_polys) == 1:
            return out_polys[0], n_dropped

        merged = unary_union(out_polys)
        return merged, n_dropped

    def _pixel_corner_to_lonlat(self, x, y, lon_grid, lat_grid):
        """Average the lon/lat of the (up to 4) grid cells touching
        pixel-space corner (x, y) = (col, row), skipping any that fall
        outside the populated swath (NaN in `lon_grid`/`lat_grid`).
        Returns None only if none of the touching cells are populated.
        """
        n_rows, n_cols = lon_grid.shape
        rows = sorted({int(np.floor(y)), int(np.ceil(y))})
        cols = sorted({int(np.floor(x)), int(np.ceil(x))})

        lons, lats = [], []
        for r in rows:
            for c in cols:
                rc = int(np.clip(r, 0, n_rows - 1))
                cc = int(np.clip(c, 0, n_cols - 1))
                lon_val, lat_val = lon_grid[rc, cc], lat_grid[rc, cc]
                if np.isfinite(lon_val) and np.isfinite(lat_val):
                    lons.append(lon_val)
                    lats.append(lat_val)

        if not lons:
            return None
        return (float(np.mean(lons)), float(np.mean(lats)))

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
            pad = w // 2
            open_coords = coords[:-1]  
            padded = np.vstack([open_coords[-pad:], open_coords, open_coords[:pad]]) \
                if pad > 0 else open_coords
            kernel = np.ones(w) / w
            smoothed_open = np.column_stack([
                np.convolve(padded[:, 0], kernel, mode="same")[pad:pad + len(open_coords)],
                np.convolve(padded[:, 1], kernel, mode="same")[pad:pad + len(open_coords)],
            ])
            smoothed = np.vstack([smoothed_open, smoothed_open[:1]])  # re-close ring
            candidate = Polygon(smoothed, kept_interiors)
            if candidate.is_valid and not candidate.is_empty:
                cleaned = candidate
            else:
                repaired = candidate.buffer(0)
                if not repaired.is_empty:
                    if isinstance(repaired, MultiPolygon):
                        repaired = max(repaired.geoms, key=lambda p: p.area)
                    cleaned = repaired

        if not cleaned.is_valid:
            cleaned = cleaned.buffer(0)
            if isinstance(cleaned, MultiPolygon):
                cleaned = max(cleaned.geoms, key=lambda p: p.area)

        return cleaned

    def apply_water_extent_mask(self, candidates_df: pd.DataFrame,
                                 mask_polygon=None, base_mask=None) -> pd.DataFrame:
        """Keep only candidate pixels that fall inside the water extent mask.
        """
        mask_polygon = mask_polygon or self._water_extent_mask
        if mask_polygon is None:
            raise ValueError("No water_extent_mask provided or cached; "
                              "call build_water_extent_mask first.")

        if not mask_polygon.is_valid:
            warnings.warn("water_extent_mask is invalid; repairing with buffer(0) "
                           "before running containment checks.")
            repaired = mask_polygon.buffer(0)
            if isinstance(repaired, MultiPolygon):
                repaired = max(repaired.geoms, key=lambda p: p.area)
            mask_polygon = repaired

        df = candidates_df.copy()
        inside = vectorized.contains(mask_polygon, df["longitude"].to_numpy(),
                                      df["latitude"].to_numpy())

        if base_mask is not None:
            base_series = base_mask if isinstance(base_mask, pd.Series) \
                else pd.Series(np.asarray(base_mask), index=range(len(base_mask)))
            inside_series = pd.Series(False, index=base_series.index)
            inside_series.loc[df.index] = inside
            return df[inside].reset_index(drop=True), inside_series

        return df[inside].reset_index(drop=True), inside

    def cluster_points_to_polygons(self, df: pd.DataFrame, category: str,
                                    grid_res_deg: Optional[float] = None,
                                    min_region_points: Optional[int] = None,
                                    concave_hull_ratio: Optional[float] = None,
                                    connectivity: int = 2,
                                    closing_disk_radius: Optional[int] = None) -> gpd.GeoDataFrame:
        """Split a scattered lon/lat point set into spatially connected
        clusters and return one (concave-hull) polygon per cluster.
        """
        grid_res_deg = grid_res_deg or self.cfg.mask_grid_res_deg
        min_region_points = (min_region_points if min_region_points is not None
                              else self.cfg.min_region_points)
        concave_hull_ratio = (concave_hull_ratio if concave_hull_ratio is not None
                               else self.cfg.concave_hull_ratio)
        closing_disk_radius = (closing_disk_radius if closing_disk_radius is not None
                                else self.cfg.cluster_closing_disk_radius)

        empty = gpd.GeoDataFrame(
            columns=["category", "region_id", "num_points", "area",
                     "cent_lon", "cent_lat", "geometry"],
            geometry="geometry", crs="EPSG:4326",
        )
        if len(df) == 0:
            return empty

        lon = df["longitude"].to_numpy()
        lat = df["latitude"].to_numpy()

        lat_min, lat_max = lat.min(), lat.max()
        lon_min, lon_max = lon.min(), lon.max()
        lat_bins = np.arange(lat_min, lat_max + grid_res_deg, grid_res_deg)
        lon_bins = np.arange(lon_min, lon_max + grid_res_deg, grid_res_deg)
        if len(lat_bins) < 2 or len(lon_bins) < 2:
            lat_bins = np.array([lat_min, lat_min + grid_res_deg])
            lon_bins = np.array([lon_min, lon_min + grid_res_deg])

        row = np.clip(np.digitize(lat, lat_bins) - 1, 0, len(lat_bins) - 2)
        col = np.clip(np.digitize(lon, lon_bins) - 1, 0, len(lon_bins) - 2)

        occ = np.zeros((len(lat_bins) - 1, len(lon_bins) - 1), dtype=bool)
        occ[row, col] = True

        if closing_disk_radius and closing_disk_radius > 0:
            occ_for_labeling = binary_closing(occ, disk(closing_disk_radius))
        else:
            occ_for_labeling = occ

        structure = np.ones((3, 3)) if connectivity == 2 else None
        labelled, num_features = label(occ_for_labeling, structure=structure)
        point_labels = labelled[row, col]
        order = np.argsort(point_labels, kind="stable")
        sorted_labels = point_labels[order]
        unique_labels, start_idx, counts = np.unique(
            sorted_labels, return_index=True, return_counts=True
        )

        records = []
        for region_id, start, n in zip(unique_labels, start_idx, counts):
            if region_id == 0:
                continue 
            n = int(n)
            if n < min_region_points:
                continue

            idx = order[start:start + n]
            xs, ys = lon[idx], lat[idx]
            if n >= 3:
                try:
                    hull = shapely.concave_hull(
                        MultiPoint(list(zip(xs, ys))), ratio=concave_hull_ratio
                    )
                except Exception:
                    hull = MultiPoint(list(zip(xs, ys))).convex_hull
            else:
                hull = MultiPoint(list(zip(xs, ys))).convex_hull

            if hull.is_empty:
                continue
            centroid = hull.centroid
            records.append({
                "category": category,
                "region_id": int(region_id),
                "num_points": n,
                "area": hull.area,
                "cent_lon": centroid.x,
                "cent_lat": centroid.y,
                "geometry": hull,
            })

        if not records:
            return empty
        return gpd.GeoDataFrame(records, crs="EPSG:4326")

    def check_reference_point_classification(self, pixc_df: pd.DataFrame) -> bool:
        """Warn if the cached reference pixel looks like dark water."""
        ref_latlon = pixc_df.attrs.get("ref_pixel_latlon")
        if ref_latlon is None:
            warnings.warn("No reference pixel recorded on this DataFrame.")
            return False
        lat, lon = ref_latlon
        row = pixc_df.loc[
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
            self._dark_water_class_codes())]

    def filter_land(self, pixc_df: pd.DataFrame,
                     enabled: Optional[bool] = None) -> pd.DataFrame:
        """Optionally drop pixels classified as land.
        """
        enabled = self.cfg.exclude_land if enabled is None else enabled
        if not enabled:
            return pixc_df
        return pixc_df[pixc_df["classification"] != self.cfg.land_class_code]

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
        per_cycle_open_water_mask = {}

        for cycle, fp in filepaths_by_cycle.items():
            pixc = self.read_pixel_cloud(fp, cycle=cycle, bbox=bbox)
            pixc = self.compute_height_anomaly(pixc, ref_lat=ref_lat, ref_lon=ref_lon)
            self.check_reference_point_classification(pixc)
            pixc = self.filter_dark_water(pixc)

            initial_mask = pd.Series(True, index=pixc.index)
            filtered, phase_noise_mask = self.filter_phase_noise(initial_mask, pixc)
            per_cycle_filtered[cycle] = filtered

            candidates, open_water_mask = self.filter_open_water(filtered, phase_noise_mask)
            per_cycle_intertidal[cycle] = candidates
            per_cycle_open_water_mask[cycle] = open_water_mask

        mask = self.build_water_extent_mask(
            list(per_cycle_open_water_mask.values()),
            list(per_cycle_filtered.values()),
        )

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

        land = (
            (classification == self.cfg.land_class_code)
            & (arrays["classification_qual"] == 0)
        )

        land_grid = np.zeros((n_az, n_rg), dtype=bool)
        land_grid[az - az_min, rg - rg_min] = land

        populated = np.zeros((n_az, n_rg), dtype=bool)
        populated[az - az_min, rg - rg_min] = True

        BW = land_grid | ~populated
        se = disk(self.cfg.land_closing_disk_radius)
        BW1 = remove_small_objects(
            BW, min_size=self.cfg.land_min_object_size,
            connectivity=self.cfg.land_connectivity,
        )
        BW2 = binary_closing(BW1, se)
        BW3 = binary_fill_holes(BW2)

        not_land_clean = ~BW3

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

    def gridify(self, az, rg, values, grids=None, fill_value=0, dtype=None) -> np.ndarray:
        """Scatter any per-pixel array — a boolean mask, h_a, sigma_phase_noise,
        whatever — into an azimuth/range grid.
        """
        az = np.asarray(az).astype(int)
        rg = np.asarray(rg).astype(int)
        values = values.to_numpy() if isinstance(values, pd.Series) else np.asarray(values)

        if not (len(az) == len(rg) == len(values)):
            raise ValueError(
                f"gridify: az ({len(az)}), rg ({len(rg)}), and values "
                f"({len(values)}) must be the same length and share row order."
            )

        if grids is not None:
            az_min, rg_min = grids["az_min"], grids["rg_min"]
            n_az, n_rg = grids["populated"].shape
        else:
            az_min, rg_min = int(az.min()), int(rg.min())
            n_az = int(az.max()) - az_min + 1
            n_rg = int(rg.max()) - rg_min + 1

        row = az - az_min
        col = rg - rg_min
        in_bounds = (row >= 0) & (row < n_az) & (col >= 0) & (col < n_rg)
        if not in_bounds.all():
            warnings.warn(
                f"gridify: {int((~in_bounds).sum())} pixel(s) fell outside "
                "the target grid and were dropped."
            )

        dtype = dtype if dtype is not None else values.dtype
        grid = np.full((n_az, n_rg), fill_value, dtype=dtype)
        grid[row[in_bounds], col[in_bounds]] = values[in_bounds]
        return grid

    def scatter_indices_to_grid(self, az: np.ndarray, rg: np.ndarray, grids: dict) -> np.ndarray:
        """Return a boolean grid (same shape as `grids`) marking every
        (az, rg) pixel-index pair present in the given arrays."""
        az = np.asarray(az).astype(int)
        rg = np.asarray(rg).astype(int)
        az_min, rg_min = grids["az_min"], grids["rg_min"]
        n_az, n_rg = grids["populated"].shape

        row = az - az_min
        col = rg - rg_min
        in_bounds = (row >= 0) & (row < n_az) & (col >= 0) & (col < n_rg)

        kept = np.zeros((n_az, n_rg), dtype=bool)
        kept[row[in_bounds], col[in_bounds]] = True
        return kept

    def restrict_grids_to_mask(self, grids: dict, keep_mask: np.ndarray) -> dict:
        """Return a copy of `grids` with `final_mask` ANDed against `keep_mask`
        (e.g. the output of scatter_indices_to_grid), leaving the original
        grids dict untouched."""
        restricted = dict(grids)
        restricted["final_mask"] = grids["final_mask"] & keep_mask
        return restricted

    def polygons_from_grid(self, category: str, codes: Sequence[int], grids: dict):
        """Convex-hull polygons for each connected region of `codes` pixels"""
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


    def remove_small_polygons(self, geometry, min_area, area_col: Optional[str] = None):
       """
       Remove polygons smaller than min_area.
       """
       if isinstance(geometry, gpd.GeoDataFrame):
           if len(geometry) == 0:
               return geometry
           if area_col is not None and area_col in geometry.columns:
               areas = geometry[area_col]
           else:
               areas = geometry.geometry.area
           return geometry[areas >= min_area].reset_index(drop=True)

       if isinstance(geometry, Polygon):
          return geometry if geometry.area >= min_area else Polygon()

       elif isinstance(geometry, MultiPolygon):
           kept = [poly for poly in geometry.geoms if poly.area >= min_area]

           if len(kept) == 0:
               return Polygon()
           elif len(kept) == 1:
               return kept[0]
           else:
               return MultiPolygon(kept)

       return geometry

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
        """Write a rectangular bounding-box KML around a cloud of lon/lat points"""
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
        """Subset a pixel-cloud DataFrame to points that fall inside a KML polygon"""
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

        mask = vectorized.contains(poly, lon, lat) | vectorized.touches(poly, lon, lat)

        subset = pixc_df[mask]
        print(f"Kept {int(mask.sum())}/{len(pixc_df)} points inside {kml_path}")
        return subset, mask

    def build_coastline_buffer_mask(
        self,
        coastline_kml_path: Optional[str] = None,
        buffer_km: Optional[float] = None,
        projected_crs: Optional[str] = None,
    ) -> gpd.GeoDataFrame:
        """Read a coastline KML (one or more lines) and build a polygon mask
        extending `buffer_km` kilometres either side of the coastline.
        """
        coastline_kml_path = coastline_kml_path or self.cfg.coastline_kml_path
        buffer_km = buffer_km if buffer_km is not None else self.cfg.coastline_buffer_km
        projected_crs = projected_crs or self.cfg.coastline_projected_crs

        if not coastline_kml_path:
            raise ValueError(
                "No coastline_kml_path given and none configured "
                "(cfg.coastline_kml_path)."
            )

        coast_gdf = gpd.read_file(coastline_kml_path)
        if coast_gdf.empty:
            raise ValueError(f"No geometry found in {coastline_kml_path}.")

        if coast_gdf.crs is None:
            coast_gdf = coast_gdf.set_crs("EPSG:4326")

        coast_proj = coast_gdf.to_crs(projected_crs)

        buffer_m = buffer_km * 1000.0
        buffered = coast_proj.geometry.buffer(buffer_m)
        merged = unary_union(buffered.values)

        mask_gdf = gpd.GeoDataFrame(geometry=[merged], crs=projected_crs).to_crs("EPSG:4326")

        self._coastline_mask = mask_gdf
        print(
            f"Built coastline buffer mask: {buffer_km:.2f} km either side of "
            f"{len(coast_gdf)} coastline feature(s) in {coastline_kml_path}"
        )
        return mask_gdf

    def subset_kml_to_swath(
        self,
        mask_gdf: gpd.GeoDataFrame,
        swath_kml_path: str,
        output_path: str,
        name: str = "Coastal subset",
    ) -> str:
        """Intersect a mask polygon (e.g. the buffered coastline) with the
        SWOT swath bounding-box KML, and save the resulting subset polygon
        to a new KML at `output_path`."""
        swath_gdf = gpd.read_file(swath_kml_path)
        if swath_gdf.empty:
            raise ValueError(f"No geometry found in {swath_kml_path}.")
        if swath_gdf.crs is None:
            swath_gdf = swath_gdf.set_crs("EPSG:4326")
        swath_geom = unary_union(swath_gdf.geometry.values)

        if mask_gdf.crs is None:
            mask_gdf = mask_gdf.set_crs("EPSG:4326")
        elif str(mask_gdf.crs) != "EPSG:4326":
            mask_gdf = mask_gdf.to_crs("EPSG:4326")
        mask_geom = unary_union(mask_gdf.geometry.values)

        subset_geom = mask_geom.intersection(swath_geom)
        if subset_geom.is_empty:
            raise ValueError(
                "Coastline mask does not intersect the SWOT swath bbox; "
                "check coastline_kml_path / buffer_km / swath location."
            )

        polys = [subset_geom] if subset_geom.geom_type == "Polygon" else [
            g for g in subset_geom.geoms if g.geom_type == "Polygon" and not g.is_empty
        ]
        if not polys:
            raise ValueError("Coastline/swath intersection produced no polygon area.")

        kml = simplekml.Kml()
        for i, poly in enumerate(polys):
            pol = kml.newpolygon(name=f"{name}_{i}" if len(polys) > 1 else name)
            pol.outerboundaryis = list(poly.exterior.coords)
            if poly.interiors:
                pol.innerboundaryis = [list(ring.coords) for ring in poly.interiors]
            pol.style.linestyle.width = 2
            pol.style.polystyle.fill = 0

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        kml.save(output_path)
        print(f"Wrote coastal subset polygon ({len(polys)} part(s)) to {output_path}")
        return output_path

    def make_coastal_subset_kml(
        self,
        filepath: str,
        output_base: str,
        swath_lon: np.ndarray,
        swath_lat: np.ndarray,
        coastline_kml_path: Optional[str] = None,
        buffer_km: Optional[float] = None,
    ) -> str:
        """End-to-end helper for the coastal-subsetting workflow
        """
        output_dir = self.make_output_directory(filepath, output_base)

        swath_kml_path = self.export_bbox_kml(
            swath_lon, swath_lat, filepath, output_base, name="SWOT PIXC Swath"
        )

        mask_gdf = self.build_coastline_buffer_mask(coastline_kml_path, buffer_km)

        subset_kml_path = os.path.join(output_dir, "coastal_subset.kml")
        self.subset_kml_to_swath(mask_gdf, swath_kml_path, subset_kml_path)

        return subset_kml_path

    def rasterize_category_polygons(self, gdf, output_path: str,
                                     resolution_deg: Optional[float] = None,
                                     burn_value: int = 1) -> str:
        """Burn a polygon GeoDataFrame into a single-band GeoTIFF using rasterio"""
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
        """Per-pixel category code in flat (original) pixel order"""
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

    def _scatter_mask_panel(self, ax, base_df: pd.DataFrame, mask, *,
                             title: Optional[str] = None,
                             xlim: Optional[tuple] = None,
                             ylim: Optional[tuple] = None,
                             aspect: Optional[float] = None):
        """Shared lon/lat scatter styling for a boolean pixel mask (kept =
        darkorange vs. dropped = lightgray). Used by both `plot_mask_step`
        (single-panel) and `plot_pipeline_summary_grid` (2x2 combined) so
        every mask panel in the pipeline looks exactly the same.
        """
        mask_arr = mask.to_numpy() if isinstance(mask, pd.Series) else np.asarray(mask)
        if isinstance(mask, pd.Series):
            mask_arr = mask.reindex(base_df.index, fill_value=False).to_numpy()
        elif len(mask_arr) != len(base_df):
            raise ValueError(
                f"mask length ({len(mask_arr)}) does not match base_df length "
                f"({len(base_df)}); pass mask as a pandas Series sharing "
                "base_df's index if lengths legitimately differ."
            )

        lon = base_df["longitude"].to_numpy()
        lat = base_df["latitude"].to_numpy()
        colors = np.where(mask_arr, "darkorange", "lightgray")
        ax.scatter(lon, lat, c=colors, s=2)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

        if aspect is None:
            mean_lat = np.deg2rad(np.nanmean(lat)) if lat.size else 0.0
            aspect = 1 / np.cos(mean_lat)
        if np.isfinite(aspect) and aspect > 0:
            ax.set_aspect(aspect)

        n_kept = int(np.sum(mask_arr))
        if title is not None:
            ax.set_title(f"{title}\n({n_kept}/{len(mask_arr)} kept)")
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        return n_kept

    def _scatter_value_panel(self, ax, data: pd.DataFrame, value_col: Optional[str], *,
                              lon_col: str = "longitude", lat_col: str = "latitude",
                              cmap: str = "viridis", vmin: Optional[float] = None,
                              vmax: Optional[float] = None, title: Optional[str] = None,
                              xlim: Optional[tuple] = None, ylim: Optional[tuple] = None,
                              aspect: Optional[float] = None,
                              inset_colorbar: bool = False):
        """Shared lon/lat scatter styling for pixels optionally colored by a
        value column (e.g. h_a), with a colorbar when a value_col is given.
        Used by both `plot_step`'s DataFrame branch (single-panel) and
        `plot_pipeline_summary_grid` (2x2 combined) so every value panel in
        the pipeline looks exactly the same.
        """
        lon = data[lon_col].to_numpy()
        lat = data[lat_col].to_numpy()
        c = data[value_col].to_numpy() if value_col and value_col in data.columns else None
        sc = ax.scatter(lon, lat, c=c, s=2, cmap=cmap, vmin=vmin, vmax=vmax)
        if c is not None:
            if inset_colorbar:
                cax = inset_axes(
                    ax, width="4%", height="100%", loc="center left",
                    bbox_to_anchor=(1.02, 0.0, 1, 1),
                    bbox_transform=ax.transAxes, borderpad=0,
                )
                cbar = plt.colorbar(sc, cax=cax)
            else:
                cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label(value_col)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

        if aspect is None:
            mean_lat = np.deg2rad(np.nanmean(lat))
            aspect = 1 / np.cos(mean_lat)
        if np.isfinite(aspect) and aspect > 0:
            ax.set_aspect(aspect)

        if title is not None:
            ax.set_title(title)
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        return sc
    
    def plot_step(self, data, step_name: str, output_dir: str, *,
                  value_col: Optional[str] = None, title: Optional[str] = None,
                  cmap: str = "viridis", dpi: int = 150, show: bool = False,
                  vmin: Optional[float] = None, vmax: Optional[float] = None,
                  xlim: Optional[tuple] = None,
                  ylim: Optional[tuple] = None):
        """Generic diagnostic plotting for a pipeline step. Inspects the type of data and picks a sensible default view, then
        saves a png to file directory.
        """
        fig, ax = plt.subplots(figsize=(8, 8))
        plot_title = title or step_name.replace("_", " ").title()
        made_plot = False

        try:
            if isinstance(data, gpd.GeoDataFrame):
                if len(data) > 0:
                    data.boundary.plot(ax=ax, linewidth=1.2, color="steelblue")
                    made_plot = True
                ax.set_xlabel("Longitude")
                ax.set_ylabel("Latitude")

            elif isinstance(data, (Polygon, MultiPolygon)):
                gpd.GeoSeries([data], crs="EPSG:4326").boundary.plot(
                    ax=ax, linewidth=1.5, color="black"
                )
                ax.set_xlabel("Longitude")
                ax.set_ylabel("Latitude")
                made_plot = True

            elif isinstance(data, dict) and data and all(
                v is None or isinstance(v, gpd.GeoDataFrame) for v in data.values()
            ):
                default_colors = {"water": "blue", "intertidal": "red", "land": "green"}
                for category, gdf in data.items():
                    if gdf is None or len(gdf) == 0:
                        continue
                    gdf.boundary.plot(
                        ax=ax, linewidth=1.5,
                        color=default_colors.get(category, None),
                        label=str(category),
                    )
                    made_plot = True
                ax.set_xlabel("Longitude")
                ax.set_ylabel("Latitude")
                if made_plot:
                    ax.legend()

            elif isinstance(data, pd.DataFrame):
                lon_aliases = ["longitude", "lon", "cell_lon", "cent_lon"]
                lat_aliases = ["latitude", "lat", "cell_lat", "cent_lat"]
                lon_col = next((c for c in lon_aliases if c in data.columns), None)
                lat_col = next((c for c in lat_aliases if c in data.columns), None)
                if lon_col and lat_col and len(data) > 0:
                    self._scatter_value_panel(
                        ax, data, value_col, lon_col=lon_col, lat_col=lat_col,
                        cmap=cmap, vmin=vmin, vmax=vmax,
                    )
                    made_plot = True
                elif value_col and value_col in data.columns and len(data) > 0:
                    ax.hist(data[value_col].to_numpy(), bins=3000, color="steelblue")
                    ax.set_xlabel(value_col)
                    ax.set_ylabel("Count")
                    made_plot = True

            elif isinstance(data, dict) and data:
                grid = None
                if value_col and value_col in data and isinstance(data[value_col], np.ndarray):
                    grid = data[value_col]
                else:
                    grid = next(
                        (v for v in data.values() if isinstance(v, np.ndarray) and v.ndim == 2),
                        None,
                    )
                if grid is not None:
                    im = ax.imshow(grid, origin="lower", cmap=cmap, aspect="auto")
                    plt.colorbar(im, ax=ax)
                    ax.set_xlabel("Range index")
                    ax.set_ylabel("Azimuth index")
                    made_plot = True

            elif isinstance(data, np.ndarray) and data.ndim == 2:
                im = ax.imshow(data, origin="lower", cmap=cmap, aspect="auto")
                plt.colorbar(im, ax=ax)
                ax.set_xlabel("Range index")
                ax.set_ylabel("Azimuth index")
                made_plot = True

            elif isinstance(data, np.ndarray) and data.ndim == 1:
                if data.size > 0:
                    ax.hist(data, bins=3000, color="steelblue")
                    made_plot = True

            if not made_plot:
                ax.text(0.5, 0.5, "No data to plot", ha="center", va="center",
                         transform=ax.transAxes)

            if xlim is not None:
                ax.set_xlim(xlim)
            if ylim is not None:
                ax.set_ylim(ylim)

            ax.set_title(plot_title)
            plt.tight_layout()

            path = os.path.join(output_dir, f"{step_name}.png")
            fig.savefig(path, dpi=dpi)
            print(f"Saved step plot: {path}")

            if show:
                plt.show()
            return path
        finally:
            plt.close(fig)

    def plot_mask_step(self, base_df: pd.DataFrame, mask, step_name: str,
                        output_dir: str, *, title: Optional[str] = None,
                        dpi: int = 150, show: bool = False,
                        xlim: Optional[tuple] = None, ylim: Optional[tuple] = None):
        """Scatter-plot a boolean pixel mask (kept vs. dropped) over the
        lon/lat of `base_df`.
        """
        fig, ax = plt.subplots(figsize=(8, 8))
        try:
            plot_title = title or step_name.replace("_", " ").title()
            self._scatter_mask_panel(ax, base_df, mask, title=plot_title, xlim=xlim, ylim=ylim)
            plt.tight_layout()

            path = os.path.join(output_dir, f"{step_name}.png")
            fig.savefig(path, dpi=dpi)
            print(f"Saved step plot: {path}")

            if show:
                plt.show()
            return path
        finally:
            plt.close(fig)

    def plot_pipeline_summary_grid(
        self,
        pixc_full: pd.DataFrame,
        subset_mask,
        pixc_ha: pd.DataFrame,
        phase_noise_mask,
        open_water_mask,
        step_name: str,
        output_dir: str,
        *,
        value_col: str = "h_a",
        cmap: str = "viridis",
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        xlim: Optional[tuple] = None,
        ylim: Optional[tuple] = None,
        dpi: int = 150,
        show: bool = False,
    ):
        """Combined 2x2 lon/lat diagnostic figure:

            top-left     = Step 1b subset-by-kml mask
            top-right    = Step 3 phase-noise filtered mask
            bottom-left  = Step 4 open-water filtered mask
            bottom-right = Step 2 height anomaly (h_a)
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 14), constrained_layout=True)
        ax_tl, ax_tr = axes[0, 0], axes[0, 1]
        ax_bl, ax_br = axes[1, 0], axes[1, 1]

        full_lat = pixc_full["latitude"].to_numpy()
        mean_lat = np.deg2rad(np.nanmean(full_lat)) if full_lat.size else 0.0
        shared_aspect = 1 / np.cos(mean_lat)
        if not (np.isfinite(shared_aspect) and shared_aspect > 0):
            shared_aspect = None

        try:
            self._scatter_mask_panel(
                ax_tl, pixc_full, subset_mask,
                title="(1) Subset by kml", xlim=xlim, ylim=ylim,
                aspect=shared_aspect,
            )
            self._scatter_mask_panel(
                ax_tr, pixc_full, phase_noise_mask,
                title="(2) Phase-Noise Filtered Pixels", xlim=xlim, ylim=ylim,
                aspect=shared_aspect,
            )
            self._scatter_mask_panel(
                ax_bl, pixc_full, open_water_mask,
                title="(4) Open-Water Filtered Candidates", xlim=xlim, ylim=ylim,
                aspect=shared_aspect,
            )
            self._scatter_value_panel(
                ax_br, pixc_ha, value_col,
                cmap=cmap, vmin=vmin, vmax=vmax,
                title="(3) Height Anomaly (h_a)", xlim=xlim, ylim=ylim,
                aspect=shared_aspect, inset_colorbar=True,
            )

            path = os.path.join(output_dir, f"{step_name}.png")
            fig.savefig(path, dpi=dpi)
            print(f"Saved step plot: {path}")

            if show:
                plt.show()
            return path
        finally:
            plt.close(fig)

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

    def plot_category_polygons(self, category_gdfs: dict, ax=None,
                                colors: Optional[dict] = None):
        """Plot polygon boundaries for one or more categories on the same axes."""
        default_colors = {"water": "blue", "intertidal": "red", "land": "green"}
        colors = colors or default_colors

        if ax is None:
            _, ax = plt.subplots(figsize=(10, 10))

        for category, gdf in category_gdfs.items():
            if gdf is None or len(gdf) == 0:
                continue
            gdf.boundary.plot(
                ax=ax,
                color=colors.get(category, "black"),
                linewidth=1.5,
                label=category.capitalize(),
            )

        ax.set_title("SWOT PIXC Category Polygons")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.legend()

        plt.tight_layout()
        return ax

    def make_output_directory(self, filepath: str, output_base: str) -> str:
        """
        Create an output directory based on the PIXC filename"""

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
        """End-to-end: read PIXC, build masks, export water/intertidal
        polygons as shapefile + KML, (optionally) rasterize, (optionally) plot."""
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
