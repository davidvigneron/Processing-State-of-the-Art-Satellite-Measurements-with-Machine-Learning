#!./.venv/bin/python

#SBATCH --nodes=1
##SBATCH --ntasks=76
##SBATCH --mem=4123930
#SBATCH --mail-type=END
##SBATCH --mail-user=youremail@kit.edu

#SBATCH --time=0-02:00:00

#SBATCH --error=error_gaussian_regr.log
#SBATCH --output=output_gaussian_regr.log
#SBATCH --job-name=gaussian_regr

import os
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
import sys
import time
from multiprocessing import shared_memory, Queue, Manager
import logging
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from sklearn.preprocessing import StandardScaler
import gpflow
from gpflow.kernels import RBF, Matern52, White
from gpflow.models import SVGP, GPR, SGPR
from gpflow.optimizers import Scipy
import concurrent.futures
from sklearn.metrics import r2_score
import tensorflow as tf

from path_constants import *
import logging

logger = logging.getLogger(__name__)
tf.random.set_seed(7)

def transform_to_apriori_diff(X, Y):
    Y_tr = np.empty_like(Y)
    for v, (start, end) in y_ind_for_vars.items():
        Y_tr[:, start:end] = Y[:, start:end] - X[:, X_ind_for_vars[v+"_apriori"][0]:X_ind_for_vars[v+"_apriori"][1]]
    return Y_tr

def inverse_apriori_diff(X, Y_tr):
    Y = np.empty_like(Y_tr)
    for v, (start, end) in y_ind_for_vars.items():
        Y[:, start:end] = Y_tr[:, start:end] + X[:, X_ind_for_vars[v+"_apriori"][0]:X_ind_for_vars[v+"_apriori"][1]]
    return Y

REMOVE_APRIORI = False
USE_SPARSE = True

X_ind_for_vars = {'platform_zenith_angle': (0, 1), 'surface_emissivity': (1, 21), 'musica_pressure_levels': (21, 50), 'musica_st_apriori': (50, 51), 'musica_at_apriori': (51, 80), 'musica_wv_apriori': (80, 109), 'musica_spectra': (109, 950)}
y_ind_for_vars = {'musica_st': (0, 1), 'musica_at': (1, 30), 'musica_wv': (30, 59)}

for gpu in tf.config.list_physical_devices('GPU'):
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except:
        logger.error(f"Couldn't set memory growth for device {gpu}")
gpflow.config.set_default_float(np.float32)

def create_and_optimize_model(X_train_norm, Y_train_norm, X_test_norm, Y_test_norm, lengthscale: int | np.ndarray=30, n_restarts=5) -> tuple[GPR, float]:
    best_score = -float('inf')
    failed_count = 0
    for _ in range(n_restarts):
        if USE_SPARSE:
            kernel = Matern52(lengthscales=lengthscale) + White() # type: ignore # ndarray and int are compatible with TensorType
            Z = X_train_norm[np.random.choice(X_train.shape[0], 1000, replace=False), :]  # Using random inducing points for simplicity
            model = SGPR(kernel=kernel, inducing_variable=Z, data=(X_train_norm, Y_train_norm), noise_variance=np.random.Generator(np.random.PCG64()).choice([1,10]))
            # lengthscales are about as good as they get, setting this to true resulted in worse result (but feel free to try)
            gpflow.set_trainable(model.kernel.kernels[0].lengthscales, False) # type: ignore
            gpflow.set_trainable(model.kernel.kernels[1].variance, True) # type: ignore
        else:
            kernel = Matern52(lengthscales=lengthscale) + White() # type: ignore # ndarray and int are compatible with TensorType
            model = GPR(kernel=kernel, data=(X_train_norm, Y_train_norm))
        with tf.distribute.MirroredStrategy().scope():
            optimizer = Scipy()
            res = optimizer.minimize(model.training_loss, model.trainable_variables,  # type: ignore
                                    options={'maxiter': 200, 'disp': True})
        if not res.success:
            failed_count += 1
            try:
                logger.warning(f"Failed to train, end lengthscales are:\n{model.kernel.kernels[0].lengthscales}") # type: ignore
            except:
                logger.warning(f"Failed to train, end lengthscales are:\n{model.kernel.lengthscales}") # type: ignore
            continue

        current_score = r2_score(Y_test_norm, model.predict_y(X_test_norm)[0].numpy()) # type: ignore
        logger.info(f"current score: {current_score}")
        if current_score > best_score:
            best_score = current_score
            best_model = model

    if failed_count == n_restarts:
        logger.warning(f"Failed to train a single model with {n_restarts} restarts.")
        raise SystemExit
    logging.info("\n", failed_count, "attempts have failed to converge.")
    return best_model, best_score # type: ignore

if __name__ == "__main__":
    # Load data
    skip_step = 300 if USE_SPARSE else 20000
    X_train = np.load(PATH_TO_2020_X, mmap_mode="r")[::skip_step,:].copy()
    Y_train = np.load(PATH_TO_2020_Y, mmap_mode="r")[::skip_step,:].copy()
    X_test = np.load(PATH_TO_2019_X, mmap_mode="r")[::skip_step,:].copy()
    Y_test = np.load(PATH_TO_2019_Y, mmap_mode="r")[::skip_step,:].copy()
    X_val = np.load(PATH_TO_2021_X, mmap_mode="r")[::skip_step,:].copy()
    Y_val = np.load(PATH_TO_2021_Y, mmap_mode="r")[::skip_step,:].copy()

    if REMOVE_APRIORI:
        Y_train = transform_to_apriori_diff(X_train, Y_train)
        Y_test = transform_to_apriori_diff(X_test, Y_test)

    x_scaler = StandardScaler()
    x_scaler.fit(X_train)
    X_train_norm = x_scaler.transform(X_train)
    X_test_norm = x_scaler.transform(X_test)
    X_val_norm = x_scaler.transform(X_val)

    mean_y, std_y = Y_train.mean(axis=0), Y_train.std(axis=0)
    std_y[std_y == 0] = sys.float_info.epsilon
    Y_train_norm = (Y_train - mean_y) / std_y
    Y_test_norm = (Y_test - mean_y) / std_y

    lengthscales = np.ones(X_train.shape[1]) * 40 if USE_SPARSE else np.ones(X_train.shape[1]) * (30 if REMOVE_APRIORI else 7)
    if USE_SPARSE:
        # lengthscales found in training without sparse, did not manage to improve from these...
        lengthscales = np.array([436.25018 , 241.22423 , 237.48013 , 235.17767 , 232.40341 ,
       223.96826 , 218.56277 , 214.84958 , 232.41759 , 242.67429 , 246.55513 , 247.65996 , 251.73949 , 251.63626 , 263.32513 ,
       253.72087 , 251.61711 , 244.838   , 244.35046 , 240.45953 , 251.26884 , 126.06316 , 125.50885 , 122.64513 , 116.06909 ,
       105.87919 ,  97.403   ,  87.94001 ,  63.425842,  41.23408 , 33.254837,  36.468025,  43.099976,  52.48183 ,  59.98421 ,
        67.67422 ,  76.09287 ,  85.20677 ,  95.34364 , 108.00135 , 118.42404 , 101.4987  ,  85.23212 ,  97.43062 ,  96.862816,
        92.852356, 104.309944, 117.807106,  91.7464  ,  89.918884,  8.13575 ,  49.96737 ,  57.99308 ,  85.89952 ,  93.50539 ,
       122.489876, 110.97426 ,  89.12656 ,  54.24943 ,  39.42101 , 64.209755,  77.20709 , 195.65456 , 274.51416 ,   5.335849,
        55.568623, 117.8472  ,  60.338722,   8.715869,   8.432716, 11.334876,  12.456564,  40.85227 , 136.35777 ,  64.695435,
       259.41556 , 275.76495 ,  79.1181  ,   9.854483,   9.983164, 76.674934,  89.48763 , 147.69511 , 123.74648 , 108.957596,
        66.0886  ,  90.655014, 198.31686 , 238.50653 ,  73.54307 , 116.131905, 121.73336 , 133.87547 , 143.38477 , 153.373   ,
       162.42212 , 168.5354  , 173.9176  , 173.44257 , 167.49516 , 157.35814 , 145.70691 , 138.20354 , 126.36497 ,  84.43139 ,
        79.48277 ,  72.144196,  65.13844 ,  65.44702 ,  37.203194, 37.78538 ,  37.37737 ,  37.294952,  37.34242 ,  34.872375,
        37.34045 ,  37.54279 ,  37.00988 ,  36.19596 ,  36.976906, 37.203964,  36.50611 ,  37.60391 ,  37.84969 ,  38.091038,
        36.44061 ,  36.057076,  36.278725,  36.291843,  36.971985, 38.511204,  38.279182,  37.03871 ,  37.49388 ,  37.374176,
        36.851376,  36.508263,  36.511894,  34.84401 ,  28.332315, 22.667343,  29.633587,  34.657513,  24.789417,  24.51372 ,
        31.867693,  36.098133,  36.481277,  36.64812 ,  36.784523, 36.815712,  36.93359 ,  37.08914 ,  36.328545,  36.65295 ,
        37.419926,  37.034473,  36.15232 ,  35.422195,  35.094822, 35.307682,  35.519733,  35.983505,  36.06547 ,  35.887848,
        35.77221 ,  36.03163 ,  36.387806,  37.240868,  37.378242, 37.159866,  37.076717,  37.81305 ,  34.21612 ,  33.131123,
        37.2922  ,  37.438812,  37.584137,  37.54241 ,  37.468616, 36.414978,  36.326344,  36.939762,  37.353565,  35.90784 ,
        35.46341 ,  35.038895,  35.735237,  35.67111 ,  34.97194 , 33.45685 ,  30.636253,  25.477362,  24.880056,  32.28435 ,
        25.266909,  23.690298,  33.33496 ,  43.110832,  31.835247, 20.878817,  16.890709,  22.34411 ,  29.802586,  33.55032 ,
        34.972023,  35.53278 ,  35.7017  ,  33.287586,  29.2253  , 31.500492,  31.793804,  35.82193 ,  35.36659 ,  34.415634,
        35.20165 ,  35.97653 ,  36.195545,  35.81841 ,  34.63262 , 30.037533,  23.510408,  25.770641,  34.981045,  25.916872,
        18.650997,  25.782326,  32.781044,  34.144524,  34.777893, 31.743763,  30.890642,  31.494143,  29.840456,  33.330795,
        35.08225 ,  33.193   ,  33.883854,  35.13754 ,  32.820675, 32.355583,  34.635017,  34.346794,  32.544548,  29.631178,
        24.143764,  17.728662,  20.716686,  30.23092 ,  45.432144, 42.03582 ,  33.994076,  23.4104  ,  16.842371,  18.92409 ,
        20.204483,  21.73798 ,  27.733173,  35.2008  ,  37.33148 , 38.155582,  38.414364,  38.196392,  36.98763 ,  35.122997,
        35.24362 ,  33.080753,  33.612846,  35.645443,  32.19738 , 32.277924,  36.238552,  38.734917,  39.355698,  39.17024 ,
        39.172756,  39.23544 ,  39.15254 ,  39.40346 ,  39.533222, 38.768757,  38.758236,  36.945625,  36.385338,  37.93371 ,
        39.144207,  39.10438 ,  39.297096,  39.450848,  39.54905 ,  39.653862,  39.057632,  34.88297 ,  32.965237,  32.28112 ,
        32.784843,  31.554209,  28.418142,  28.12918 ,  32.64972 , 36.05542 ,  35.587   ,  36.912216,  35.840107,  32.144653,
        25.18588 ,  24.101198,  24.256212,  24.8508  ,  26.395943, 31.258966,  30.432953,  30.528477,  32.581425,  34.087772,
        35.485657,  33.770138,  32.41364 ,  28.297697,  26.215288, 25.187277,  20.52845 ,  21.074995,  23.263462,  30.181847,
        42.757404,  44.810936,  32.01036 ,  26.613962,  25.17789 , 28.096493,  29.407139,  31.294031,  33.215363,  33.25163 ,
        33.026157,  33.772873,  36.379776,  36.92947 ,  32.09103 , 28.821255,  31.35973 ,  29.301762,  27.819574,  30.138264,
        31.24759 ,  31.765863,  33.198708,  28.394825,  31.747643, 36.119453,  36.168133,  35.061333,  30.283983,  27.147852,
        28.727182,  31.518867,  31.243677,  28.335447,  29.982592, 32.776684,  32.508793,  31.723988,  46.395096,  44.97133 ,
        26.69874 ,  26.063581,  27.99228 ,  28.037514,  31.642962, 31.974377,  32.460815,  30.088503,  28.640278,  32.202034,
        34.085354,  25.188011,  19.033405,  23.661446,  19.456196, 14.067252,  16.478659,  26.400307,  30.31111 ,  29.723694,
        25.557543,  31.14642 ,  40.477543,  52.235245,  55.52767 , 68.10963 ,  60.56193 ,  45.194386,  38.85802 ,  27.230589,
        25.476068,  32.08776 ,  40.70045 ,  34.465893,  25.158705, 27.561785,  34.8298  ,  36.003033,  32.79843 ,  32.61091 ,
        29.268227,  20.688496,  22.278728,  30.805609,  29.776623, 33.056034,  44.64186 ,  38.285275,  32.8599  ,  29.169634,
        29.844381,  28.072388,  21.155653,  21.693756,  39.73761 , 50.998665,  41.020683,  42.152996,  41.76429 ,  36.927933,
        43.105827,  44.346043,  35.15128 ,  39.306316,  49.17688 , 39.8079  ,  34.785423,  44.331448,  57.217777,  57.376488,
        64.878456,  75.513756,  47.454464,  39.040596,  34.669567, 34.369034,  31.861837,  28.143423,  30.749516,  36.033333,
        31.105816,  28.082872,  33.0801  ,  39.91255 ,  41.707268, 49.250137,  44.919655,  35.42278 ,  32.741364,  34.364933,
        38.34568 ,  40.653896,  42.3525  ,  49.37195 ,  43.949688, 34.399227,  35.054947,  33.27777 ,  31.601007,  31.014862,
        30.233244,  24.189445,  23.543886,  31.785645,  38.00831 , 31.115566,  28.7135  ,  27.937796,  25.507488,  29.376696,
        36.941223,  31.033127,  28.944365,  35.77005 ,  45.978157, 49.109722,  54.334946,  59.840878,  46.66784 ,  33.575745,
        31.57297 ,  36.202446,  29.44309 ,  19.677433,  12.732739, 9.725129,  11.784901,  22.451305,  28.346334,  24.899775,
        26.160091,  25.408924,  29.269674,  37.403606,  47.047577, 49.763527,  49.90684 ,  53.88703 ,  44.21815 ,  38.38866 ,
        37.634422,  23.540955,  21.025558,  32.627937,  28.248083, 25.35444 ,  29.823545,  25.694887,  17.308964,  18.760082,
        28.627201,  22.466946,  22.264023,  36.089867,  44.311913, 39.878998,  39.505184,  36.732098,  33.8089  ,  39.277756,
        42.09803 ,  42.65785 ,  51.9666  ,  48.272213,  39.989716, 49.504505,  56.02411 ,  37.448753,  32.182537,  40.988693,
        46.089203,  47.102566,  47.08733 ,  52.703754,  64.35042 , 72.97541 ,  53.169003,  43.464874,  41.5619  ,  34.640778,
        30.766829,  35.05101 ,  53.58228 ,  62.48818 ,  65.947426, 58.519733,  39.379787,  35.71903 ,  40.415085,  48.423435,
        76.97368 ,  81.83647 ,  59.70582 ,  48.241196,  51.877087, 63.24173 ,  70.29576 ,  82.325035, 102.81732 ,  96.0375  ,
        81.44135 ,  86.79461 , 107.048965, 119.6666  , 108.9465  , 103.88226 , 124.6048  , 124.45777 , 123.45231 , 101.61806 ,
        56.33075 ,  48.466354,  35.188377,  26.224459,  29.973955, 37.083813,  45.939312,  51.350243,  34.872517,  29.425964,
        26.690182,  24.056553,  23.485321,  27.75668 ,  25.204613, 22.872723,  26.218252,  22.740137,  20.900583,  29.743792,
        34.286907,  31.607143,  37.625195,  47.405247,  63.34778 , 52.105083,  46.442444,  51.75754 ,  65.235695,  49.671566,
        40.247272,  35.513035,  39.38677 ,  44.371113,  38.51966 , 31.947857,  28.938044,  31.191624,  36.165348,  37.985725,
        38.934734,  43.417336,  57.16045 ,  48.587963,  43.865112, 43.452316,  43.9382  ,  46.705788,  53.88786 ,  76.618774,
        91.41244 ,  69.29466 ,  55.87521 ,  59.579533,  77.44925 , 62.19082 ,  48.664013,  45.197952,  41.796535,  34.47647 ,
        28.453476,  28.962418,  37.061226,  33.43199 ,  25.642565, 24.559244,  30.638605,  41.04796 ,  37.160583,  26.740076,
        24.05622 ,  26.331635,  24.390522,  23.256096,  17.394766, 10.730845,  14.371659,  20.997505,  27.808168,  27.64634 ,
        22.303701,  24.982134,  44.627567,  86.18404 ,  45.524925, 35.6754  ,  35.34455 ,  28.614368,  16.316496,  12.70967 ,
        19.250212,  29.48648 ,  33.48588 ,  41.397373,  45.32923 , 33.31361 ,  21.38402 ,  12.097995,  10.137129,  10.335035,
        13.018883,  21.104233,  36.69607 ,  61.26894 ,  84.86454 , 55.416225,  31.399712,  29.824533,  35.24645 ,  31.944098,
        27.127789,  30.8251  ,  28.555727,  24.120195,  26.012268, 34.75119 ,  41.741657,  40.741524,  42.972256,  46.237072,
        56.889668,  73.47949 ,  55.27118 ,  41.182037,  44.1463  , 58.424046,  52.135586,  71.57789 ,  91.10878 ,  77.99134 ,
        63.942688,  76.75324 ,  84.60745 ,  74.470535,  88.08786 , 29.977692,  83.36356 ,  76.29518 ,  53.09744 ,  41.78441 ,
        42.517464,  69.7279  ,  46.211826,  37.032223,  39.008137, 55.704338,  37.829517,  29.953314,  30.425045,  30.948261,
        31.8035  ,  32.56645 ,  30.041044,  32.09395 ,  31.640474, 26.862173,  29.272007,  29.953676,  30.162783,  34.348305,
        50.578995,  56.13943 ,  46.23947 ,  35.186176,  33.27393 , 32.445225,  37.04469 ,  32.12618 ,  37.658974,  47.070103,
        56.654312,  77.048035,  77.14783 ,  55.72418 ,  47.516888, 43.175365,  35.81591 ,  32.81003 ,  36.49926 ,  35.599342,
        36.04449 ,  33.429356,  28.779959,  33.018036,  32.34117 , 27.731554,  30.573769,  29.844769,  29.076899,  32.575436,
        37.11813 ,  44.18413 ,  54.272484,  77.39755 ,  73.29553 , 52.016567,  48.379974,  39.089485,  35.688057,  33.142433,
        34.7871  ,  34.705494,  36.472282,  35.57135 ,  31.84003 , 32.489063,  34.40037 ,  38.039497,  44.068405,  46.29484 ,
        46.938435,  42.61422 ,  37.69307 ,  36.524372,  38.14129 , 40.83177 ,  45.68931 ,  61.955555,  85.65652 ,  74.267525,
        57.44689 ,  55.225903,  59.79404 ,  75.71107 , 103.947914, 78.729935, 111.01705 , 100.896416,  90.79331 ,  67.28455 ,
        52.459026,  44.10317 ,  42.23052 ,  42.590816,  41.362602, 40.594498,  40.6163  ,  40.006603,  39.160534,  36.232372,
        36.368355,  37.30652 ,  38.316467,  41.97752 ,  46.836105, 48.879696,  49.478508,  60.288074,  88.99242 ,  92.744965,
        67.90432 ,  56.124023,  51.501698,  50.10186 ,  50.012604, 43.873127,  41.649883,  43.70375 ,  47.688717,  48.5566  ,
        48.28582 ,  50.079468,  56.423214,  67.05104 ,  62.004692, 55.293446,  56.972485,  71.060165,  95.54612 , 106.26998 ,
       105.43361 ,  80.60942 ,  75.19639 ,  98.43722 ,  51.56015 , 100.13876 ,  84.327866,  68.50382 ,  60.121387,  55.06501 ,
        51.521553,  48.534504,  47.078583,  47.09566 ,  46.197376, 48.01948 ,  57.020626,  61.408318,  62.623985,  51.087116,
        46.577232,  48.93192 ,  64.74711 ,  76.42093 ,  63.922504, 57.466213,  46.94094 ,  40.567688,  37.629528,  37.518337,
        39.271946,  43.478493,  48.211914,  46.259613,  41.425117, 39.725513,  41.251644,  48.19086 ,  51.25024 ,  46.79519 ,
        43.760212,  44.56827 ,  44.711796,  45.61135 ,  47.591896, 50.48844 ,  50.919273,  52.148678,  59.47872 ,  78.40849 ,
       100.91289 ,  85.12075 ,  81.51612 ,  90.48231 ,  81.65129 , 100.79676 ,  84.82662 ,  72.72774 ,  70.52577 ,  59.20709 ,
        53.38469 ,  51.74798 ,  50.647076,  54.336975,  57.74716 , 59.013382,  69.628395,  68.46029 ,  62.42571 ,  56.96711 ,
        52.411186,  50.939583,  52.738914,  52.65713 ,  51.934196, 52.41216 ,  55.408436,  60.51149 ,  68.71183 ,  83.64898 ,
        37.778984,  97.51357 ,  18.285828,  54.830677,  52.240437, 72.149826,  95.35557 , 116.91686 , 111.201744,  84.845604,
        67.470184,  61.69114 ,  60.12675 ,  65.27534 ,  80.772865, 82.58375 ,  87.33978 ,  72.37824 ,  80.32466 ,  45.198704,
        94.39145 , 448.81595 ,  76.67041 ,  63.59559 ,  77.33424 ])
    model, score_test = create_and_optimize_model(X_train_norm=X_train_norm, Y_train_norm=Y_train_norm, X_test_norm=X_test_norm, Y_test_norm=Y_test_norm, lengthscale=lengthscales, n_restarts=5)
    # Collect results

    try:
        logger.debug(f"lengthscales after training: {model.kernel.kernels[0].lengthscales}") # type: ignore
    except:
        ...

    model_number = 3 if REMOVE_APRIORI else (4 if USE_SPARSE else 2)
    import pickle
    model_path = f"gp_w{model_number}.pkl"
    with open(model_path, "wb+") as path:
        pickle.dump((model, x_scaler, mean_y, std_y), path)
    logger.info(f"Model saved to {model_path}")

    mean, _ = model.predict_y(X_val_norm) # type: ignore
    output = mean * std_y + mean_y
    output = output.numpy()
    # values below zero are not possible as we have kelvin and particles per ...
    logger.info(f"Removed {np.count_nonzero(output < 0)} values smaller than zero, this is  {np.count_nonzero(output < 0) * 100 / output.size}% of the entries" )
    output[output < 0] = 0

    if REMOVE_APRIORI:
        output = inverse_apriori_diff(X_val, output)
    score = r2_score(Y_val, output, multioutput="raw_values")
    for var, (start, end) in y_ind_for_vars.items():
        print(var, np.mean(score[start:end], dtype=np.float64)) # type: ignore
    print("raw values:", score, flush=True)

    import matplotlib.pyplot as plt
    import mpl_scatter_density # type: ignore # adds projection='scatter_density'
    # "Viridis-like" colormap with white background
    white_viridis = LinearSegmentedColormap.from_list('white_viridis', [
        (0, '#ffffff'),
        (sys.float_info.epsilon, '#440053'),
        (0.2, '#404388'),
        (0.4, '#2a788e'),
        (0.6, '#21a784'),
        (0.8, '#78d151'),
        (1, '#fde624'),
    ], N=512)
    def plot_scatter(predicted, truth, var_name, dimensions):
        fig, axes = plt.subplots(5, 6, figsize=(15, 12), subplot_kw={'projection': 'scatter_density'})  # 5 rows and 6 columns of subplots
        display_str = f'Comparison of predicted and true values for {var_name}'
        if False:
            display_str += f"\n{'Displayed':<9} {removed_outliers_percent:>4.4f}"
        fig.suptitle(display_str)
        for i, dim in zip(range(dimensions), range( y_ind_for_vars["musica_wv"][0]
                            , y_ind_for_vars["musica_wv"][1])):
            row, col = divmod(i, 6)
            ax = axes[row, col]
            cb = ax.scatter_density(predicted[:,dim], truth[:,dim], cmap=white_viridis, dpi=600)
            plt.colorbar(cb, ax=ax)
            mean = np.mean(truth[:,dim])
            ax.axline((mean, mean), slope=1, c="black", linestyle="--", linewidth=.5, alpha=.5)
            ax.set_title(f'{var_name} lvl {i}', fontsize='x-small')
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Full physics")
        # Hide any unused subplots
        for i in range(dimensions, 30):
            fig.delaxes(axes.flatten()[i])

        plt.tight_layout(rect=(0, 0, 0.98, 1))
        plt.savefig(f"python/images/result_gaussian_w{model_number}_2021.png", dpi=600)

    plot_scatter(output, Y_val, "wv", 29)
