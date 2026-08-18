"""
Loader for config.xml -> Config object.

All values come from config.xml. Nothing is hardcoded here -- if a field
is missing from the XML, load_config() will raise an error rather than
silently falling back to some default.

Usage:
    from config_loader import load_config
    cfg = load_config("config.xml")
    print(cfg.sigma_phase_noise_threshold)
    print(cfg.intertidal_class_codes)
"""

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Optional


# --- type casting helpers -------------------------------------------------

_CASTERS = {
    "float": float,
    "int": int,
    "str": str,
    "bool": lambda s: s.strip().lower() in ("true", "1", "yes"),
}


def _cast(elem):
    """Cast a leaf <tag type="...">value</tag> element to its Python type."""
    type_attr = elem.get("type", "str")
    return _CASTERS[type_attr](elem.text.strip())


def _read_tuple(elem):
    """Read a <tag type='tuple'><value type='...'>...</value>...</tag> element."""
    return tuple(_CASTERS[v.get("type", "str")](v.text.strip()) for v in elem.findall("value"))


def _read_alias_dict(parent):
    """Read a block of <alias key="..."><value>...</value>...</alias> elements into a dict."""
    result = {}
    for alias in parent.findall("alias"):
        key = alias.get("key")
        result[key] = [v.text.strip() for v in alias.findall("value")]
    return result


# --- Config container -------------------------------------------------
# No default values here -- every field is required to come from the XML.

@dataclass
class Config:
    sigma_phase_noise_threshold: float
    ref_point_buffer_deg: float
    pdf_bin_width_m: float
    pdf_peak_min_density: float
    pdf_max_bins: int
    kde_max_samples: Optional[int]
    kde_random_seed: Optional[int]
    kde_bw_scalar: Optional[float]
    eps_up_fraction: float
    eps_low_divisor: float
    mask_grid_res_deg: float
    mask_validity_quantile: float
    mask_closing_disk_radius: int
    mask_min_object_size: int
    mask_connectivity: int
    cluster_closing_disk_radius: int
    output_grid_res_deg: float
    mc_realizations: int
    mc_ci_alpha: float
    exclude_dark_water: bool
    exclude_land: bool
    build_plots: bool
    show_plots: bool
    silent: bool
    step5: bool
    peak_diagnostic_plot: bool
    min_hole_area_deg2: float
    smoothing_window: int
    fill_value_threshold: float
    open_water_class_codes: tuple
    dark_water_class_codes: tuple
    land_class_code: int
    water_class_codes: tuple
    intertidal_class_codes: tuple
    quality_flag_names: tuple
    land_closing_disk_radius: int
    land_min_object_size: int
    land_connectivity: int
    min_region_points: int
    raster_default_resolution_deg: float
    concave_hull_ratio: float
    min_intertidal_polygon_area: float
    water_extent_min_piece_fraction: float
    pixc_var_aliases: dict
    pixc_optional_var_aliases: dict
    subset: str
    coastline_kml_path: str
    coastline_buffer_km: float
    coastline_projected_crs: str


# Field names, grouped by how they're parsed out of the XML tree.

_SCALAR_FIELDS = [
    "sigma_phase_noise_threshold", "fill_value_threshold",
    "ref_point_buffer_deg", "pdf_bin_width_m", "pdf_peak_min_density",
    "pdf_max_bins", "kde_max_samples", "kde_random_seed", "kde_bw_scalar",
    "eps_up_fraction", "eps_low_divisor",
    "mask_grid_res_deg", "mask_validity_quantile", "mask_closing_disk_radius",
    "mask_min_object_size", "mask_connectivity", "cluster_closing_disk_radius",
    "output_grid_res_deg", "mc_realizations", "mc_ci_alpha",
    "exclude_dark_water", "exclude_land", "build_plots", "show_plots",
    "silent", "step5", "peak_diagnostic_plot",
    "min_hole_area_deg2", "smoothing_window",
    "land_class_code", "land_closing_disk_radius", "land_min_object_size",
    "land_connectivity", "min_region_points", "raster_default_resolution_deg",
    "concave_hull_ratio", "min_intertidal_polygon_area", "water_extent_min_piece_fraction",
    "coastline_buffer_km", "coastline_projected_crs",
    "subset", "coastline_kml_path",
]

_TUPLE_FIELDS = [
    "open_water_class_codes", "dark_water_class_codes",
    "water_class_codes", "intertidal_class_codes", "quality_flag_names",
]


# --- main loader -------------------------------------------------------

def load_config(path: str) -> Config:
    """Parse config.xml and return a fully populated Config instance.

    Raises a ValueError if any expected field is missing from the XML,
    rather than silently substituting a default.
    """
    tree = ET.parse(path)
    root = tree.getroot()

    values = {}
    missing = []

    for name in _SCALAR_FIELDS:
        elem = root.find(f".//{name}")
        if elem is None:
            missing.append(name)
        else:
            values[name] = _cast(elem)

    for name in _TUPLE_FIELDS:
        elem = root.find(f".//{name}")
        if elem is None:
            missing.append(name)
        else:
            values[name] = _read_tuple(elem)

    aliases_elem = root.find(".//pixc_var_aliases")
    if aliases_elem is None:
        missing.append("pixc_var_aliases")
    else:
        values["pixc_var_aliases"] = _read_alias_dict(aliases_elem)

    optional_aliases_elem = root.find(".//pixc_optional_var_aliases")
    if optional_aliases_elem is None:
        missing.append("pixc_optional_var_aliases")
    else:
        values["pixc_optional_var_aliases"] = _read_alias_dict(optional_aliases_elem)

    if missing:
        raise ValueError(f"config.xml is missing required field(s): {', '.join(missing)}")

    return Config(**values)


if __name__ == "__main__":
    import sys

    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "config.xml"
    cfg = load_config(cfg_path)
    print(cfg)