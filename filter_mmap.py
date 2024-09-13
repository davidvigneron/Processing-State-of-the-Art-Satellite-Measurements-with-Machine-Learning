import numpy as np
import sys

from path_constants import PATH_TO_WS

WITH_TIME_SPACE = True

year = sys.argv[1] if len(sys.argv) > 1 else "2019"

# these might change over time, if you reconverted the files in the output there should be information about how many observations where converted.
obs_p_year = {"2019": 123762657, "2020": 306818090, "2021": 135223109}

X = np.memmap(f"{PATH_TO_WS}{year}_Xtsu.npy",dtype=np.float32,mode="r",shape=(obs_p_year[year], 954))
Y = np.memmap(f"{PATH_TO_WS}{year}_ytsu.npy",dtype=np.float32,mode="r",shape=(obs_p_year[year], 60))

# index 0 of y is the fit quality flag, as well as index 21 of X
inds = np.argwhere(Y[:,0] == 3)

if WITH_TIME_SPACE:
    X_ind_for_vars={'platform_zenith_angle': (0, 1), 'surface_emissivity': (1, 21), 'musica_pressure_levels': (21, 50), 'musica_st_apriori': (50, 51), 'musica_at_apriori': (51, 80), 'musica_wv_apriori': (80, 109), 'musica_spectra': (109, 950), 'time': (950, 951), 'lat': (951, 952), 'lon': (952, 953)}
else:
    X_ind_for_vars = {'platform_zenith_angle': (0, 1), 'surface_emissivity': (1, 21), 'musica_pressure_levels': (21, 50), 'musica_st_apriori': (50, 51), 'musica_at_apriori': (51, 80), 'musica_wv_apriori': (80, 109), 'musica_spectra': (109, 950)}
y_ind_for_vars = {'musica_st': (0, 1), 'musica_at': (1, 30), 'musica_wv': (30, 59)}

# only with fitquality flag 3 and remove fitquality flag from array
Xnf = X[inds,:]
Ynf = Y[inds,1:]
Xnf = Xnf.reshape(Xnf.shape[0], Xnf.shape[2])
Ynf = Ynf.reshape(Ynf.shape[0], Ynf.shape[2])
Xnf = np.delete(Xnf,21, axis=1)
# calculate relative humidity
wv = Ynf[:, y_ind_for_vars['musica_wv'][1]-1]
at = Ynf[:, y_ind_for_vars['musica_at'][1]-1]
pressure = Xnf[:, X_ind_for_vars['musica_pressure_levels'][1]-1]
RH = np.empty(Ynf.shape[0])
for i in range(Ynf.shape[0]-1):
    if at[i] - 273.15 >= 0:
        RH[i] = wv[i] / (1e2*1e6*(6.112*np.exp(17.62*(at[i]-273.15)/(243.12+(at[i]-273.15))))/pressure[i])  # over liquid
    else:
        RH[i] = wv[i] / (1e2*1e6*(6.112*np.exp(22.46*(at[i]-273.15)/(272.62+(at[i]-273.15))))/pressure[i]) # over ice

inds = np.argwhere(RH < 2)
Xf = Xnf[inds, :]
np.save(f"{PATH_TO_WS}Xnfts2_{year}.npy", Xf.reshape(Xf.shape[0],Xf.shape[2]))
Yf = Ynf[inds, :]
np.save(f"{PATH_TO_WS}Ynfts2_{year}.npy", Yf.reshape(Yf.shape[0], Yf.shape[2]))

