#!./.venv/bin/python

#SBATCH --nodes=1
###SBATCH --mem=501600 # 4123930
#SBATCH --ntasks=76
#SBATCH --mail-type=END
##SBATCH --mail-user=youremail@kit.edu

#SBATCH --time=0-6:00:00

#SBATCH --error=error_ridge.log
#SBATCH --output=output_ridge.log
#SBATCH --job-name=ridge_regr

from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from datetime import datetime
import logging
from path_constants import *

logger = logging.getLogger(__name__)

# variable names and their respective indices (start, end) in the array
X_ind_for_vars = {'platform_zenith_angle': (0, 1), 'surface_emissivity': (1, 21), 'musica_pressure_levels': (21, 50), 'musica_st_apriori': (50, 51), 'musica_at_apriori': (51, 80), 'musica_wv_apriori': (80, 109), 'musica_spectra': (109, 950)}
y_ind_for_vars = {'musica_st': (0, 1), 'musica_at': (1, 30), 'musica_wv': (30, 59)}


# alphas from gridsearch with filtered data
alpha = np.array([100, 0, 0, 0, 0, 0, 0, 0, 0.003, 0.003, 0.01, 0.01, 0.003, 0.003, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1e-08, 0, 0, 0, 0, 0, 0.1, 0.3, 0.3, 0.3,
                1, 1, 0.003, 1e-08, 3e-10, 1e-05, 0.01, 0.01, 0.003, 0, 1e-06, 1e-09, 1e-05, 3e-10, 0, 1e-08, 3e-10, 1e-06, 3e-10, 0.001, 0.003, 0.003, 0.003, 0.001, 0.003])

fit_intercept=True

# Create a pipeline
pipeline = Pipeline([
        ('scaler', StandardScaler(copy=True)),
        ('ridge', Ridge(alpha=alpha, fit_intercept=fit_intercept,max_iter=1000, random_state=100, solver="svd", copy_X=True))
    ])

start_time = datetime.now()
X = np.load(PATH_TO_2020_X, mmap_mode="r")[::100,:].copy()
Y = np.load(PATH_TO_2020_Y, mmap_mode="r")[::100,:].copy()
logger.info(f"time elapsed after loading files {datetime.now() - start_time}")
pipeline.fit(X,Y)
logger.info(f"time elapsed after training: {datetime.now() - start_time}")

# save trained model
import pickle
file_name = "ridge_regressor_w6.pkl"
with open(file_name, "wb+") as path:
    pickle.dump(pipeline, path)
logger.info(f"Model saved at {file_name}")

del X
del Y

# Test
X_validate = np.load(PATH_TO_2021_X, mmap_mode="r")[::100,:].copy()
y_validate = np.load(PATH_TO_2021_Y, mmap_mode="r")[::100,:].copy()

output = pipeline.predict(X_validate)
test_score = r2_score(y_validate, output, multioutput="raw_values")
for var, (start, end) in y_ind_for_vars.items():
    print("r2 on test data", var, np.mean(test_score[start:end])) # type: ignore
print("raw values:", test_score)
