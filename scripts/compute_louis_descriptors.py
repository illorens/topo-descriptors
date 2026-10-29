"""
Example script on how to compute spatial descriptors from a DEM file.
"""

import logging

import topo_descriptors.topo as tp
import topo_descriptors.helpers as hlp

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.captureWarnings(True)

    # Paths
    dems_path = "/store_new/mch/msclim/appclim/data/grids/topodem/v2/dem"
    dem_type = "radar_500"
    path_dem = f"{dems_path}/{dem_type}/dem.zarr"

    # get the DEM
    dem_ds = hlp.get_dem_netcdf(path_dem)
    ind_nans, dem_ds = hlp.fill_na(dem_ds)

    # define the target domain
    domain = {"x": slice(255000, 965000), "y": slice(-160000, 480000)}

    # Foehn
    tp.compute_foehn(dem_ds, ind_nans=ind_nans, crop=domain)

    # Valley
    tp.compute_valley_effect(dem_ds, ind_nans=ind_nans, crop=domain)
