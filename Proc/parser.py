import os
import argparse as ap

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np

from config_loader import load_config
from Piplineclass import SWOTIntertidalPipeline


def polygon_parser(prog: str, desc: str) -> ap.ArgumentParser:
    p = ap.ArgumentParser(prog, description=desc)
    p.add_argument(
        "-c", "--config", type=str, metavar="PATH", help="configuration file path")
    p.add_argument(
        "-i", "--input", type=str, metavar="PATH", help="input file path", nargs="+"
    )
    p.add_argument("-o", "--output", type=str, metavar="PATH", help="output directory")
    p.add_argument(
        "-k", "--kml", type=str, metavar="PATH", help="kml polygon to subset to (overrides cfg.subset)"
    )
    p.add_argument(
        "--exclude-land", action="store_true", default=None,
        help="filter out land-classified pixels (overrides cfg.exclude_land; default comes from config.xml)"
    )
    p.add_argument(
        "--build-plots", action="store_true", default=None,
        help="save per-step diagnostic plots (overrides cfg.build_plots; default comes from config.xml)"
    )
    p.add_argument(
        "--show-plots", action="store_true", default=None,
        help="display plots interactively, in addition to saving them "
             "(overrides cfg.show_plots; default comes from config.xml)"
    )
    p.add_argument(
        "--no-step5", dest="step5", action="store_false", default=None,
        help="skip step 5 (water-extent-mask build/clean/apply, i.e. polygon cleaning); "
             "use the raw open-water mask/candidates instead "
             "(overrides cfg.step5; default comes from config.xml)"
    )
    p.add_argument(
        "--peak-diagnostic-plot", dest="peak_diagnostic_plot", action="store_true", default=None,
        help="build the step 4 KDE peak diagnostic plot (h_a density, peaks found, "
             "candidate band) for each input file "
             "(overrides cfg.peak_diagnostic_plot; default comes from config.xml)"
    )

    g = p.add_mutually_exclusive_group()
    g.add_argument(
        "-v", "--verbose", action="store_true", help="print additional information"
    )
    g.add_argument(
        "-s", "--silent", action="store_true", default=None,
        help="run silently (overrides cfg.silent; default comes from config.xml)"
    )

    return p


def _step_timer():
    """Return a function that prints elapsed time since this call."""
    import time
    t0 = time.time()
    return lambda label: print(f"    ({label} took {time.time() - t0:.1f}s)")


def _plot_peak_diagnostic(pipe, candidates, filtered, output_dir, show_plots=False, silent=False,  xlim=None):
    """Step 4 KDE peak diagnostic plot (same logic as peak.py).

    Plots the h_a KDE, all peaks that cleared pdf_peak_min_density, the
    single peak_idx that filter_open_water actually used, and the
    [h_a_lower, h_a_upper] candidate band. Saved to output_dir; only shown
    interactively if show_plots is True.
    """
    grid = candidates.attrs["grid"]
    pdf = candidates.attrs["pdf"]
    peak_idx = candidates.attrs["peak_idx"]
    h_a_lower = candidates.attrs["h_a_lower"]
    h_a_upper = candidates.attrs["h_a_upper"]

    # every local max that cleared pdf_peak_min_density, not just the one that got used
    all_peaks_idx = pipe.find_pdf_peaks(grid, pdf)

    fig = plt.figure(figsize=(9, 6))
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
    if xlim is not None:
        plt.xlim(xlim)
    plt.ylabel("Density")
    plt.title(
        "Step 4 KDE diagnostic\n"
        f"{len(candidates)} / {len(filtered)} pixels kept as candidates"
    )
    plt.legend(fontsize=8)
    plt.tight_layout()

    peak_png = os.path.join(output_dir, "step4_peak_diagnostic.png")
    fig.savefig(peak_png, dpi=150)
    if not silent:
        print(f"Saved peak diagnostic plot to {peak_png}")
        print(f"    n peaks found: {len(all_peaks_idx)} at h_a = "
              f"{[round(float(v), 4) for v in grid[all_peaks_idx]]}")
        print(f"    peak_idx used: h_a = {grid[peak_idx]:.4f}")

    if show_plots:
        plt.show()
    else:
        plt.close(fig)


def run_polygon_pipeline(pipe: SWOTIntertidalPipeline, config, filepaths, output_base,
    exclude_land=None,
    build_plots=None,
    show_plots=None,
    silent=None,
    step5=None,
    peak_diagnostic_plot=None,
):
    """Run the full polygon pipeline for each input file
    """
    exclude_land = config.exclude_land if exclude_land is None else exclude_land
    build_plots = config.build_plots if build_plots is None else build_plots
    show_plots = config.show_plots if show_plots is None else show_plots
    silent = config.silent if silent is None else silent
    step5 = config.step5 if step5 is None else step5
    peak_diagnostic_plot = config.peak_diagnostic_plot if peak_diagnostic_plot is None else peak_diagnostic_plot

    results = {}

    for filepath in filepaths:
        if not silent:
            print(f"Processing {filepath}")

        ref_lat = None
        ref_lon = None

        output_dir = pipe.make_output_directory(filepath, output_base)
        if not silent:
            print("Output directory:", output_dir)

        pixc_full = pipe.read_pixel_cloud(filepath)
        if not silent:
            print("Step 1 - read_pixel_cloud:", pixc_full.shape)

        pixc, subset_mask = pipe.subset_by_kml(pixc_full, config.subset)
        if not silent:
            print(f"Step 1b - subset_by_kml: {len(pixc)} / {len(pixc_full)} pixels kept ({config.subset})")
            print(pixc["geolocation_qual"].value_counts())
            print(pixc["geolocation_qual"].attrs)
        if build_plots:
            pipe.plot_mask_step(pixc_full, subset_mask, "step1b_subset_by_kml", output_dir,
                                 title="Step 1b: Subset by kml")

        if ref_lat is None or ref_lon is None:
            ref_lat = float(pixc["latitude"].median())
            ref_lon = float(pixc["longitude"].median())
            print(
                "WARNING: ref_lat/ref_lon not set. Placeholder "
                f"used: ({ref_lat:.6f}, {ref_lon:.6f})"
            )

        pixc = pipe.compute_height_anomaly(pixc, ref_lat, ref_lon)
        if not silent:
            print("Step 2 - compute_height_anomaly: added h_a column")

            h_a_median = np.median(pixc["h_a"])
            h_a_std = np.std(pixc["h_a"])
            vmin = -10
            vmax = 10

            pipe.plot_step(pixc, "step2_height_anomaly", output_dir,
               value_col="h_a", title="Step 2: Height Anomaly (h_a)",
               vmin=vmin, vmax=vmax, show=show_plots)

            pipe.plot_step(pixc["h_a"].to_numpy(), "step2_height_anomaly_hist", output_dir,
               title="Step 2: Height Anomaly (h_a) Histogram", show=show_plots)

        pipe.check_reference_point_classification(pixc)
        if not pipe.cycle_has_reliable_xover(pixc):
            print("warning: this cycle does not have a reliable height_cor_xover")

        pixc = pipe.filter_dark_water(pixc)

        n_before_land = len(pixc)
        if exclude_land:
            land_keep = (pixc_full["classification"] != config.land_class_code).to_numpy()
            subset_mask = subset_mask & land_keep

        pixc = pipe.filter_land(pixc, enabled=exclude_land)
        if not silent:
            if exclude_land:
                print(f"Step 2b - filter_land: {len(pixc)} / {n_before_land} pixels kept")
            else:
                print("Step 2b - filter_land: skipped (exclude_land=False)")

        filtered, phase_noise_mask = pipe.filter_phase_noise(subset_mask, pixc)
        if not silent:
            print(f"Step 3 - filter_phase_noise: {len(filtered)} / {len(pixc)} pixels kept")
        if build_plots:
            pipe.plot_mask_step(pixc_full, phase_noise_mask, "step3_filter_phase_noise", output_dir,
                                 title="Step 3: Phase-Noise Filtered Pixels")

        _t = _step_timer()
        candidates, open_water_mask = pipe.filter_open_water(filtered, phase_noise_mask)
        if not silent:
            print(f"Step 4 - filter_open_water: {len(candidates)} candidate pixels "
                  f"(h_a_lower={candidates.attrs['h_a_lower']:.4f},"
                  f"h_a_upper={candidates.attrs['h_a_upper']:.4f})")
            _t("filter_open_water / KDE")
        if build_plots:
            pipe.plot_mask_step(pixc_full, open_water_mask, "step4_filter_open_water", output_dir,
                                 title="Step 4: Open-Water Filtered Candidates")
        if peak_diagnostic_plot:
            _plot_peak_diagnostic(pipe, candidates, filtered, output_dir,
                                   show_plots=show_plots, silent=silent)
            pipe.plot_step(candidates["h_a"].to_numpy(),
                "step4_height_anomaly_hist", output_dir,
                title="Step 4: Height Anomaly (h_a) Histogram", show=show_plots)


        _t = _step_timer()
        water_extent_mask = pipe.build_water_extent_mask(open_water_mask, [filtered])

        if water_extent_mask.geom_type == "MultiPolygon":
            piece_areas = [poly.area for poly in water_extent_mask.geoms]
        else:
            piece_areas = [water_extent_mask.area]
        if not silent:
            print(
                "    water_extent_mask piece area (deg^2) stats: "
                f"n_pieces={len(piece_areas)}, min={min(piece_areas):.3e}, "
                f"median={sorted(piece_areas)[len(piece_areas) // 2]:.3e}, "
                f"max={max(piece_areas):.3e}"
            )

        water_extent_min_area = max(piece_areas) * config.water_extent_min_piece_fraction
        if not silent:
            print(f"    remove_small_polygons: using min_area={water_extent_min_area:.3e} deg^2 "
                  f"({config.water_extent_min_piece_fraction:.1%} of largest piece)")

        water_extent_mask = pipe.remove_small_polygons(
            water_extent_mask,
            min_area=water_extent_min_area
        )

        if not silent:
            print(f"Step 5a - build_water_extent_mask: polygon area="
                  f"{water_extent_mask.area:.8f} deg^2, bounds={water_extent_mask.bounds}")
            _t("build_water_extent_mask")
        if build_plots:
            pipe.plot_step(water_extent_mask, "step5a_water_extent_mask", output_dir,
                            title="Step 5a: Water Extent Mask")

        _t = _step_timer()

        if water_extent_mask.geom_type == "MultiPolygon":
            n_coords = sum(len(poly.exterior.coords) for poly in water_extent_mask.geoms)
        else:
            n_coords = len(water_extent_mask.exterior.coords)

        if not silent:
            print(
                f"    water_extent_mask.is_valid = {water_extent_mask.is_valid}, "
                f"n_exterior_coords = {n_coords}"
            )

        intertidal_pixels, inside_mask = pipe.apply_water_extent_mask(
            candidates,
            water_extent_mask,
            base_mask=open_water_mask,
        )
        if not silent:
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
        if not silent:
            _t("estimate_pixel_uncertainty")

        _t = _step_timer()
        if step5:
            finalmask = inside_mask
        else:
            finalmask = open_water_mask
        grid = pipe.gridify(pixc_full["azimuth_index"], pixc_full["range_index"], finalmask)
        if not silent:
            _t("gridify")
        if build_plots:
            pipe.plot_step(grid, "step5c_intertidal_grid", output_dir,
                            title="Step 5c: Final Intertidal Mask (az/range grid)")

        _t = _step_timer()
        lon_grid = pipe.gridify(
            pixc_full["azimuth_index"], pixc_full["range_index"],
            pixc_full["longitude"], fill_value=np.nan, dtype=float,
        )
        lat_grid = pipe.gridify(
            pixc_full["azimuth_index"], pixc_full["range_index"],
            pixc_full["latitude"], fill_value=np.nan, dtype=float,
        )
        if not silent:
            _t("gridify (lon/lat)")

        _t = _step_timer()
        intertidal_gdf = pipe.polygons_from_raster_mask(
            grid, lon_grid, lat_grid, category="intertidal",
            fill_holes=not exclude_land,
        )
        if not silent:
            _t("polygons_from_raster_mask")
            print(f"Step 6 - polygons_from_raster_mask: {len(intertidal_gdf)} intertidal polygon(s)")

        if not silent and len(intertidal_gdf) > 0:
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

        n_before = len(intertidal_gdf)
        intertidal_gdf = pipe.remove_small_polygons(
            intertidal_gdf, min_area=config.min_intertidal_polygon_area, area_col="area",
        )
        if not silent:
            print(f"Step 6b - remove_small_polygons: kept {len(intertidal_gdf)} / {n_before} "
                  f"intertidal polygon(s) (min_area={config.min_intertidal_polygon_area} deg^2)")

        shapefile_paths = {}
        if len(intertidal_gdf) == 0:
            print("No intertidal polygons found.")
        else:
            shapefile_paths["intertidal"] = pipe.export_polygons_shapefile(intertidal_gdf, "intertidal", output_dir)
            pipe.export_polygons_kml(intertidal_gdf, "intertidal", output_dir)

        _t = _step_timer()
        water_mask_bool = subset_mask & pixc_full["classification"].isin(config.water_class_codes).to_numpy()
        water_grid = pipe.gridify(pixc_full["azimuth_index"], pixc_full["range_index"], water_mask_bool)
        if not silent:
            _t("gridify (water mask)")
        if build_plots:
            pipe.plot_step(water_grid, "step6w_water_grid", output_dir,
                            title="Step 6w: Water Mask (az/range grid)")

        _t = _step_timer()
        water_gdf = pipe.polygons_from_raster_mask(
            water_grid, lon_grid, lat_grid, category="water",
            fill_holes=not exclude_land,
        )
        if not silent:
            _t("polygons_from_raster_mask (water)")
            print(f"Step 6w - polygons_from_raster_mask: {len(water_gdf)} water polygon(s)")

        n_before_water = len(water_gdf)
        water_gdf = pipe.remove_small_polygons(
            water_gdf, min_area=config.min_intertidal_polygon_area, area_col="area",
        )
        if not silent:
            print(f"Step 6wb - remove_small_polygons: kept {len(water_gdf)} / {n_before_water} "
                  f"water polygon(s) (min_area={config.min_intertidal_polygon_area} deg^2)")

        if len(water_gdf) == 0:
            print("No water polygons found.")
        else:
            shapefile_paths["water"] = pipe.export_polygons_shapefile(water_gdf, "water", output_dir)
            pipe.export_polygons_kml(water_gdf, "water", output_dir)

        _t = _step_timer()
        agg_grid = pipe.aggregate_to_grid(intertidal_pixels)
        grid_csv = os.path.join(output_dir, "intertidal_grid.csv")
        agg_grid.to_csv(grid_csv, index=False)
        if not silent:
            print(f"Wrote gridded output to {grid_csv}")
            _t(f"aggregate_to_grid ({len(agg_grid)} cells x {config.mc_realizations} MC realizations)")

        if len(intertidal_gdf) > 0 or len(water_gdf) > 0:
            _t = _step_timer()
            ax1 = pipe.plot_category_polygons({"water": water_gdf, "intertidal": intertidal_gdf})
            polygons_png = os.path.join(output_dir, "category_polygons.png")
            ax1.figure.savefig(polygons_png, dpi=150)
            if not silent:
                print(f"Saved category polygons plot to {polygons_png}")
            if show_plots:
                plt.show()
            else:
                plt.close(ax1.figure)
            if not silent:
                _t("plotting")

        if not silent:
            for _cat in ("intertidal", "water"):
                if _cat not in shapefile_paths:
                    continue
                check_gdf = gpd.read_file(shapefile_paths[_cat])
                print(f"-- {_cat} shapefile sanity check --")
                print(check_gdf.crs)                    # should be EPSG:4326
                print(check_gdf.geometry.is_valid)      # no self-intersections
                print(check_gdf.geometry.is_empty)      # not empty
                print(check_gdf.total_bounds)           # sanity-check lat/lon range makes sense for your swath
                print(check_gdf[["num_points", "area", "cent_lon", "cent_lat"]])

        if not silent:
            print("Done.\n")

        results[filepath] = {
            "output_dir": output_dir,
            "intertidal_gdf": intertidal_gdf,
            "water_gdf": water_gdf,
            "shapefile_paths": shapefile_paths,
            "grid_csv": grid_csv,
        }

    return results


def polygon_main():
    """
    main entry point for polygon processing
    """
    # create the CLI parser object
    parser = polygon_parser(None, "Compute polygons from image")
    # get the CLI args
    args = parser.parse_args()

    # read the specified config file
    config = load_config(args.config)
    if args.kml:
        config.subset = args.kml

    if not args.input:
        parser.error("no input file(s) given (-i/--input)")

    exclude_land = config.exclude_land if args.exclude_land is None else args.exclude_land
    build_plots = config.build_plots if args.build_plots is None else args.build_plots
    show_plots = config.show_plots if args.show_plots is None else args.show_plots
    silent = config.silent if args.silent is None else args.silent
    step5 = config.step5 if args.step5 is None else args.step5
    peak_diagnostic_plot = (
        config.peak_diagnostic_plot if args.peak_diagnostic_plot is None else args.peak_diagnostic_plot
    )

    # create the pipeline object
    pipe = SWOTIntertidalPipeline(config)

    # run the processor
    run_polygon_pipeline(
        pipe,
        config,
        args.input,
        args.output,
        exclude_land=exclude_land,
        build_plots=build_plots,
        show_plots=show_plots,
        silent=silent,
        step5=step5,
        peak_diagnostic_plot=peak_diagnostic_plot,
    )


if __name__ == "__main__":
    polygon_main()