#!./.venv/bin/python

#SBATCH --nodes=1
#SBATCH --mail-type=END
##SBATCH --mail-user=youremail@kit.edu

#SBATCH --time=0-01:00:00
#SBATCH --constraint=LSDF
#SBATCH --error=error_eval.log
#SBATCH --output=output_eval.log
#SBATCH --job-name=eval_one


from datetime import datetime

print("start time:", datetime.now())
import pickle
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
import xarray as xr
import multiprocessing as mp
from joblib import parallel_backend


USE_COORDS = False
X_ind_for_vars = {'platform_zenith_angle': (0, 1), 'surface_emissivity': (1, 21), 'musica_pressure_levels': (21, 50), 'musica_st_apriori': (50, 51), 'musica_at_apriori': (51, 80), 'musica_wv_apriori': (80, 109), 'musica_spectra': (109, 950)}
if USE_COORDS:
    data_vars = set(["lon","lat","time","musica_spectra_simulations","musica_spectra_residuals","musica_wv_apriori","surface_emissivity",
                    "musica_pressure_levels","musica_at_apriori","musica_st_apriori","platform_zenith_angle","musica_fit_quality_flag"]) 
else:
    data_vars = set(["musica_spectra_simulations","musica_spectra_residuals","musica_wv_apriori","surface_emissivity",
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

def convert_to_numpy(dataset: xr.Dataset, ind_for_vars: dict[str, tuple[int,int]], out=None):
    """
    Converts dataset to flattened numpy array including coordinates
    
    ind_for_vars should contain the start and end index in the flattened array, correctness is not checked
    if out is specified write to out and return how many rows have been written
    """
    
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


data_dir = "/lsdf/kit/imk-asf/projects/MUSICA/external/read/MUSICA_IASI/Full_v030300/2020/"
orbit = "IASIA_MUSICA_030300_L2_AllTargetProducts_20200822175655_71831.nc"

dataset = xr.open_dataset(data_dir + orbit)

X = convert_to_numpy(preprocess_data(dataset), X_ind_for_vars)

MODEL = "GCN"

def predict_orbit(X, MODEL):
    if MODEL == "RandomForest":
        with open("random_forest_w5.pkl", "rb") as path:
            forest: list[RandomForestRegressor] = pickle.load(path)

        output = np.empty((X.shape[0], # type: ignore
                        len(forest)))
        for dim in range(len(forest)):
            output[:,dim] = forest[dim].predict(X)

    elif MODEL == "Ridge":
        with open(f"ridge_regressor_w6.pkl", "rb") as path:
            ridge: Pipeline = pickle.load(path)
        with parallel_backend('multiprocessing', n_jobs=-1):
            output = ridge.predict(X) # type: ignore

    elif MODEL == "Gaussian":
        import gpflow
        gpflow.config.set_default_float(np.float32)
        with open(f"gp_w4.pkl", "rb") as path:
            gaussian, x_scaler, mean_y, std_y = pickle.load(path)
        X_norm = x_scaler.transform(X)
        mean, _ = gaussian.predict_y(X_norm)
        output = mean * std_y + mean_y
        output = output.numpy()

    elif MODEL == "GCN":
        import torch
        from gcn import SatelliteGCN
            
        model = SatelliteGCN(950, 512, 59, k=4)
        model.load_state_dict(torch.load(f"gnn_w{4}_k{4}c.pt", map_location=torch.device('cpu')))
        with open("gcn_scaler.pkl", "rb") as path:
            x_scaler, mean_y, std_y = pickle.load(path)
        X_norm = x_scaler.transform(X)
        output = model(torch.tensor(X_norm), torch.tensor([[],[]], dtype=torch.int), torch.tensor([[],[]]).view(-1, 1))
        output = output.detach()
        output = output * std_y + mean_y

    return output

output = predict_orbit(X, MODEL)

print("shape of prediction:", output.shape) # type: ignore

print("end:", datetime.now())
