#!./.venv/bin/python

#SBATCH --nodes=1
#SBATCH --ntasks=76
#SBATCH --mem=501600
#SBATCH --mail-type=END
##SBATCH --mail-user=youremail@kit.edu

#SBATCH --time=0-16:00:00
#SBATCH --constraint=LSDF
#SBATCH --error=error_forest_regr.log

#SBATCH --output=output_forest_regr.log
#SBATCH --job-name=forest_regr

from joblib import parallel_backend
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn.metrics import r2_score
import concurrent.futures

from path_constants import *
import logging

logger = logging.getLogger(__name__)

y_ind_for_vars = {'musica_st': (0, 1), 'musica_at': (1, 30), 'musica_wv': (30, 59)}

# best hyperparameters outputed by optuna
some_n_estimators = [20]*30 + [100]*29

best_max_features = [336, 948, 948, 945, 936, 895, 832, 850, 828, 733, 885, 839, 803, 782, 873, 943, 931, 872, 706, 941, 850, 814, 642, 634, 610, 727, 399,
                        608, 332, 330, 526, 488, 218, 792, 489, 760, 575, 570, 570, 621, 394, 828, 833, 747, 415, 704, 937, 451, 418, 692, 738, 475, 519, 819, 362, 560, 543, 625, 614]

best_min_samples_leaf = [3, 4, 1, 5, 2, 3, 5, 3, 2, 4, 2, 3, 3, 2, 4, 2, 4, 5, 3, 3, 5, 4, 6, 5, 4, 6, 4, 3, 3, 4, 4, 3, 6, 2, 2, 2, 2, 3, 4, 3, 2, 3, 1, 1,
                            2, 2, 1, 9, 1, 2, 2, 2, 1, 2, 1, 4, 2, 3, 2]

best_max_depth = [29, 13, 29, 27, 23, 23, 32, 24, 13, 29, 22, 22, 19, 32, 20, 19, 23, 25, 31, 20, 23, 24, 24, 22, 32, 23, 15, 17, 20, 16, 20, 23, 21, 23,
                    28, 30, 23, 29, 30, 32, 25, 19, 22, 32, 28, 26, 31, 29, 19, 20, 32, 19, 32, 28, 29, 20, 25, 23, 27]


X = np.load(PATH_TO_2020_X, mmap_mode="r")[::500].copy()
Y = np.load(PATH_TO_2020_Y, mmap_mode="r")[::500].copy()

def train_one_forest(some_n_estimators, best_max_features, best_max_depth, best_min_samples_leaf, X, Y, dim):
    n_estimators = some_n_estimators[dim] # 100
    max_features = best_max_features[dim]
    max_depth = best_max_depth[dim]
    min_samples_split = 2 
    min_samples_leaf = best_min_samples_leaf[dim]

    # Create and fit random forest model
    model = RandomForestRegressor(n_estimators=n_estimators,
                                    max_features=max_features,
                                    max_depth=max_depth,
                                    min_samples_split=min_samples_split,
                                    min_samples_leaf=min_samples_leaf,
                                    verbose=2)
    with parallel_backend('multiprocessing', n_jobs=-1):
        model.fit(X, Y[:, dim])
        # Make predictions and calculate R^2
        y_pred = model.predict(X)
    return model, r2_score(Y[:, dim], y_pred)

forest: list[RandomForestRegressor] = []
scores = []
# train trees for temperature
executor = concurrent.futures.ThreadPoolExecutor(7)
futures = [executor.submit(train_one_forest, some_n_estimators, best_max_features, best_max_depth, best_min_samples_leaf, X, Y, dim) for dim in range(30)]

for future in futures:
    f, r = future.result()
    forest.append(f)
    scores.append(r)
executor.shutdown()

# train trees for wv
executor = concurrent.futures.ThreadPoolExecutor(3)
futures = [executor.submit(train_one_forest, some_n_estimators, best_max_features, best_max_depth, best_min_samples_leaf, X, Y, dim) for dim in range(30,59)]

for future in futures:
    f, r = future.result()
    forest.append(f)
    scores.append(r)
executor.shutdown()

# save trained model
import pickle
file_name = f"random_forest_w5.pkl"
with open(file_name, "wb+") as path:
    pickle.dump(forest, path)
logger.info(f"Model saved at {file_name}")

# Evaluate the model on the test set
X_test = np.load(PATH_TO_2021_X, mmap_mode="r")[::100].copy()
Y_test = np.load(PATH_TO_2021_Y, mmap_mode="r")[::100].copy()
test_scores = []
for dim in range(Y_test.shape[1]):
    test_scores.append(forest[dim].score(X_test, Y_test[:,dim]))

for var, (start, end) in y_ind_for_vars.items():
    print("r2 on validation data", var, np.mean(test_scores[start:end]))

print("r2_scores raw:", test_scores, flush=True)