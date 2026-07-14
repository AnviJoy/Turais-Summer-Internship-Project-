
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import netCDF4 as nc
from skimage.morphology import remove_small_objects, binary_closing, disk
from scipy.ndimage import binary_fill_holes
from netCDF4 import Dataset
import matplotlib.colors as mcolors
