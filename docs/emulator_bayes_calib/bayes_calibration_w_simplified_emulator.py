





import pandas as pd

rgi_id = 'RGI60-11.01450'

pd_spec_mb = pd.read_csv(f'test_mb_{rgi_id}_fixed_pf.csv', index_col=[0])

pd_spec_mb_test = pd.read_csv(f'test_mb_{rgi_id}_fixed_pf_test.csv', index_col=[0])

import sklearn
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

X_train = pd_spec_mb[['melt_f', 'temp_bias']].values
X_test = pd_spec_mb_test[['melt_f', 'temp_bias']].values

y_train = pd_spec_mb[['spec_mb_mean']].values
y_test = pd_spec_mb_test[['spec_mb_mean']].values


# Instantiate a Gaussian Process model
#k1 = C(1.0, (1e-10, 1e10)) * RBF(10, (1e-10, 1e10))
# if we use the default settings, get the following error message:
# estimated length scale close to the length_scale bounds
k1 = C(1.0, (1e-5, 1e5)) * RBF(100, (1e-5, 1e5))
kernel = k1 #+ k2
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
kernel = DotProduct() + DotProduct() +  WhiteKernel()
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)

# Fit to data using Maximum Likelihood Estimation of the parameters
gp.fit(X_train, y_train)















######

#import seaborn as sns
import matplotlib.pyplot as plt
plt.rc('font', size=20)
import warnings
warnings.filterwarnings("once", category=DeprecationWarning)  # noqa: E402
import scipy
# imports from OGGM
import oggm
from oggm import utils, workflow, tasks, cfg, entity_task
import numpy as np
import pandas as pd
from MBsandbox.mbmod_daily_oneflowline import (TIModel_Sfc_Type, TIModel, process_w5e5_data)
from MBsandbox.wip.projections_bayescalibration import process_isimip_data, process_isimip_data_no_corr
from MBsandbox.help_func import (minimize_winter_mb_brentq_geod_via_pf,
                                 minimize_bias_geodetic,
                                 calibrate_to_geodetic_bias_winter_mb)
from MBsandbox.mbmod_daily_oneflowline import compile_fixed_geometry_mass_balance_TIModel
import time
import logging

log = logging.getLogger(__name__)

base_url = ('https://cluster.klima.uni-bremen.de/~oggm/gdirs/oggm_v1.4/'
            'L1-L2_files/elev_bands')
climate_type = 'W5E5'

# get the geodetic calibration data
pd_geodetic_all = utils.get_geodetic_mb_dataframe()
# pd_geodetic_all = pd.read_hdf(path_geodetic, index_col='rgiid')
pd_geodetic = pd_geodetic_all.loc[pd_geodetic_all.period == '2000-01-01_2020-01-01']


#pd_wgms_ref_glac_analysis = pd.read_csv('/home/lilianschuster/Schreibtisch/PhD/wgms_data_analysis/wgms_data_analysis.csv', index_col=[0])
#rgis_w_mb_profiles = pd_wgms_ref_glac_analysis[pd_wgms_ref_glac_analysis.MB_profile.dropna()].index

import MBsandbox
fp = utils.file_downloader('https://cluster.klima.uni-bremen.de/~lschuster/ref_glaciers' +
                           '/data/mb_overview_seasonal_mb_time_periods_20220301.csv')
pd_mb_overview = pd.read_csv(fp, index_col='Unnamed: 0')
fp_stats = utils.file_downloader('https://cluster.klima.uni-bremen.de/~lschuster/ref_glaciers' +
                           '/data/wgms_data_stats_20220301.csv')
pd_wgms_data_stats = pd.read_csv(fp_stats, index_col='Unnamed: 0')
# should have at least 5 annual MB estimates in the time period 1980-2019
# (otherwise can also not have MB profiles or winter MB!)
pd_wgms_data_stats = pd_wgms_data_stats.loc[pd_wgms_data_stats.len_annual_balance>=5]
ref_candidates = pd_wgms_data_stats.rgi_id.unique()

# for tests
#testing = testing
#if testing:
#    ref_candidates = ['RGI60-11.01450'] #rgis_w_mb_profiles #oggm.utils.get_ref_mb_glaciers_candidates()
#    working_dir = utils.gettempdir(dirname='OGGM_seasonal_mb_calib', reset=True)
#else:
#working_dir = '/home/lilianschuster/Schreibtisch/PhD/bayes_2022/oct_2022_emulator/'
working_dir = utils.gettempdir(dirname='test_emu', reset=False)

cfg.initialize()
cfg.PARAMS['use_multiprocessing'] = True #True
cfg.PATHS['working_dir'] = working_dir
cfg.PARAMS['hydro_month_nh'] = 1
cfg.PARAMS['hydro_month_sh'] = 1
cfg.PARAMS['continue_on_error'] = False
warnings.filterwarnings("ignore", category=DeprecationWarning)

correction = False


gdirs = workflow.init_glacier_directories(
                ['RGI60-11.01450'],
                from_prepro_level=2,
                prepro_border=10,
                prepro_base_url=base_url,
                prepro_rgi_version='62')

import pymc

gdirs


