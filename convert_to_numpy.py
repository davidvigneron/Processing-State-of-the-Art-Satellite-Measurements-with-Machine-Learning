#!./.venv/bin/python

#SBATCH --nodes=1
#SBATCH --mem=5000 # 4123930 # 501600
#SBATCH --ntasks=1
#SBATCH --mail-type=END
##SBATCH --mail-user=youremail@kit.edu

#SBATCH --time=1-18:00:00
#SBATCH --constraint=LSDF
#SBATCH --error=errorconvert_to_numpy.log
#SBATCH --output=output_convert_to_numpy.log
#SBATCH --job-name=conv_np

import numpy as np
import xarray as xr
import os
import multiprocessing as mp
from functools import partial
import sys

from path_constants import PATH_TO_WS
import logging

logger = logging.getLogger(__name__)

data_vars = set(["lon","lat","time","musica_spectra_simulations","musica_spectra_residuals","musica_wv_apriori","surface_emissivity",
                "musica_pressure_levels","musica_at_apriori","musica_st_apriori","platform_zenith_angle","musica_fit_quality_flag"])
label_vars = set(["musica_st","musica_at","musica_wv","musica_fit_quality_flag"])

def preprocess_data(dataset: xr.Dataset):
    data: xr.Dataset = dataset.drop_vars([x for x in dataset.keys() if x not in data_vars])
    # data = data.where(lambda x: x["musica_fit_quality_flag"] == 3, drop=True)
    # data = data.drop_vars(["musica_fit_quality_flag"])
    # convert simulation and residual to raw
    data["musica_spectra"] = data["musica_spectra_simulations"] - data["musica_spectra_residuals"]
    data = data.drop_vars(["musica_spectra_simulations","musica_spectra_residuals"])
    # forward fill atmospheric levels to remove all nan
    data = data.ffill(dim="atmospheric_levels")
    # remove HDO in watervapor
    data["musica_wv_apriori"] = data["musica_wv_apriori"][:,0]

    # Add time, lat, and lon coordinates
    data = data.assign_coords({
        'time': dataset.time,
        'lat': dataset.lat,
        'lon': dataset.lon
    })
    return data
def preprocess_labels(dataset: xr.Dataset):
    labels: xr.Dataset = dataset.drop_vars([x for x in dataset.keys() if x not in label_vars]).reset_coords(drop=True).drop_dims(["surface_emissivity_wn", "spectra_wn"])
    # labels = labels.where(lambda x: x["musica_fit_quality_flag"] == 3, drop=True)
    # labels = labels.drop_vars(["musica_fit_quality_flag"])
    labels = labels.ffill(dim="atmospheric_levels")
    labels["musica_wv"] = labels["musica_wv"][:,0]
    return labels

def get_ind_for_vars(dataset, include_coords=False):
    var_sizes = {var: 1 if len(dataset[var].sizes) == 1 else int(np.prod([size for dim, size in dataset[var].sizes.items() if dim != "observation_id"])) for var in dataset.data_vars}
    # Add coordinates to var_sizes
    if include_coords:
        for coord in ['time', 'lat', 'lon']:
            var_sizes[coord] = 1
    
    ind_for_vars = {}
    current = 0
    for var, s in var_sizes.items():
        ind_for_vars.update([(var, (current, current + s))])
        current += s
    del current
    return ind_for_vars

def convert_to_numpy(dataset: xr.Dataset, ind_for_vars=None, out=None):
    """
    Converts dataset to flattened numpy array including coordinates
    
    ind_for_vars should contain the start and end index in the flattened array, correctness is not checked
    if out is specified write to out and return how many rows have been written
    """
    if ind_for_vars is None:
        ind_for_vars = get_ind_for_vars(dataset)
    
    values_per_obs = sum([end - start for var, (start, end) in ind_for_vars.items()])
    all_vars_flat = np.empty(shape=(dataset.sizes["observation_id"], values_per_obs), dtype=np.float32)
    
    for var, (start, end) in ind_for_vars.items():
        if var in dataset.data_vars:
            if end - start > 1:
                all_vars_flat[:dataset.sizes["observation_id"], start:end] = dataset[var].values.reshape(-1, end-start)
            else:
                all_vars_flat[:dataset.sizes["observation_id"], start] = dataset[var].values
        elif var in ['time', 'lat', 'lon']:
            all_vars_flat[:dataset.sizes["observation_id"], start] = dataset[var].values
    
    if out is not None:
        out[:dataset.sizes["observation_id"]] = all_vars_flat
        return dataset.sizes["observation_id"]
    return all_vars_flat


def process_file(file, data_dir, X_ind_for_vars, y_ind_for_vars, x_shape, y_shape, x_filename, y_filename):
    dataset = xr.open_dataset(data_dir + file).load()
    obs_count = dataset.sizes["observation_id"]
    
    x_data = convert_to_numpy(preprocess_data(dataset), X_ind_for_vars)
    y_data = convert_to_numpy(preprocess_labels(dataset), y_ind_for_vars)
    
    # Write directly to the memory-mapped files
    x = np.memmap(x_filename, dtype=np.float32, mode='r+', shape=x_shape)
    y = np.memmap(y_filename, dtype=np.float32, mode='r+', shape=y_shape)
    
    with mp.Lock():
        # Find the next available index
        next_index = 0
        while next_index < x_shape[0] and np.any(x[next_index] != 0):
            next_index += 1
        
        if next_index + obs_count <= x_shape[0]:
            x[next_index:next_index+obs_count] = x_data
            y[next_index:next_index+obs_count] = y_data
    
    dataset.close()
    del dataset, x, y
    return obs_count

def _load_mfdataset_known(data_dir, files, num_obs):
    """
    Loads all files in "files" and writes them to a memory map, num obs is not checked and assumed to be the actual number of observations of all files in "files".
    """
    obs_id = 0

    x = np.memmap(f"{PATH_TO_WS}{sys.argv[1]}_Xtsu.npy", dtype=np.float32, mode='w+', shape=(num_obs, sum([end - start for var, (start, end) in X_ind_for_vars.items()])))
    y = np.memmap(f"{PATH_TO_WS}{sys.argv[1]}_ytsu.npy", dtype=np.float32, mode='w+', shape=(num_obs, sum([end - start for var, (start, end) in y_ind_for_vars.items()])))
    for file in files:
        dataset = xr.open_dataset(data_dir + file).load()
        convert_to_numpy(preprocess_data(dataset), X_ind_for_vars, out=x[obs_id:])
        obs_id += convert_to_numpy(preprocess_labels(dataset), y_ind_for_vars, out=y[obs_id:])
        dataset.close()
        del dataset
    print("obs id at end", obs_id)
    return x, y

def count_single_dataset(dataset):
    res = dataset.sizes["observation_id"]
    dataset.close()
    return res
def count_obs(data_dir, files) -> int:
    sums = []
    # executor = concurrent.futures.ThreadPoolExecutor(20)
    print("start for loop", flush=True)
    for i, file in enumerate(files):
        print(f"file {i}  ", end=",", flush=True)
        dataset = xr.open_dataset(data_dir + "/" + file)
        sums.append(count_single_dataset(dataset))
    return sum(sums)

if __name__ == "__main__":
    # data is being read from lsdf make sure you have access
    if len(sys.argv) > 1 and sys.argv[1] in ["2019","2020","2021"]:
        data_dir = {"2019":"/lsdf/kit/imk-asf/projects/MUSICA/external/read/MUSICA_IASI/Full_v030300/2019/",
                    "2020":"/lsdf/kit/imk-asf/projects/MUSICA/external/read/MUSICA_IASI/Full_v030300/2020/",
                    "2021":"/lsdf/kit/imk-asf/projects/MUSICA/external/read/MUSICA_IASI/Full_v030301/2021/"}[sys.argv[1]]
    else:
        raise ValueError("Dont know what year to convert.")

    file_list = [x for x in os.listdir(data_dir) if x[-3:] == ".nc"]
    total_num = len(file_list)
    sliced = file_list

    X_ind_for_vars = get_ind_for_vars(preprocess_data(dataset=xr.open_dataset(data_dir + file_list[0])), True)
    print(X_ind_for_vars)
    y_ind_for_vars = get_ind_for_vars(preprocess_labels(dataset=xr.open_dataset(data_dir + file_list[0])))
    print(y_ind_for_vars)

    num_obs = count_obs(data_dir, sliced) # if sys.argv[1] not in ["2019", "2020","2021"] else {"2019":123762657,"2020":306818090,"2021":135223109}[sys.argv[1]] # these were true for me idk if they still are
    logger.info(f"year {sys.argv[1]} has {num_obs} observations, this is needed to read the mmap file with correct shape")
    x,y = _load_mfdataset_known(data_dir, sliced, num_obs)
    logger.info(f"done {sys.argv[1]} shape: {x.shape} \nlast value: {x[-1,:]}")

