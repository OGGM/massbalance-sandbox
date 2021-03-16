# this has been called "pieps" at the beginning
# this could be input arguments when executing the script
mb_type = 'mb_monthly'
grad_type = 'cte'
uniform_firstprior = True
start_ind = 0
end_ind = 100
cluster = True
melt_f_prior = 'freq_bayesian'
uniform = False
cluster = True
reset_gdir = False
compute_missing = True
historical_dataset = 'WFDE5_CRU'
dataset = historical_dataset
path = '/home/users/lschuster/bayesian_calibration/WFDE5_ISIMIP/'
import numpy as np
import sys


if cluster == True:
    start_ind = int(np.absolute(int(sys.argv[1])))
    end_ind = int(np.absolute(int(sys.argv[2])))
    mb_type = str(sys.argv[3])
    grad_type = str(sys.argv[4])
    historical_dataset = str(sys.argv[5])

print(start_ind, end_ind, mb_type, grad_type, historical_dataset)
###############################
import pymc3 as pm
# 	conda install -c conda-forge python-graphviza
import pandas as pd

# %matplotlib inline
import os
from oggm import cfg, utils, workflow, tasks

# import aesara.tensor as aet
# import aesara

# from drounce_analyze_mcmc import effective_n, mcse_batchmeans
# plotting bayesian stuff
import arviz as az

az.rcParams['stats.hdi_prob'] = 0.95

cfg.initialize()
cfg.PARAMS['use_multiprocessing'] = True
cfg.PARAMS['continue_on_error'] = True
if cluster:
    working_dir = '/home/users/lschuster/oggm_files/all'
else:
    working_dir = '/home/lilianschuster/Schreibtisch/PhD/oggm_files/all'

cfg.PATHS['working_dir'] = working_dir
# use Huss flowlines
base_url = ('https://cluster.klima.uni-bremen.de/~oggm/gdirs/oggm_v1.4/'
            'L1-L2_files/elev_bands')

# import the MSsandbox modules
from MBsandbox.mbmod_daily_oneflowline import process_era5_daily_data, TIModel
from MBsandbox.wip.bayes_calib_geod_direct import bayes_dummy_model_better
from MBsandbox.wip.projections_bayescalibration import bayes_mbcalibration

cfg.PARAMS['hydro_month_nh'] = 1
typ = '{}_{}'.format(mb_type, grad_type)

if uniform_firstprior == True:
    pd_params = pd.read_csv(path +
                            'only_perfect_precalibration_alps_refglaciers_{}_{}_{}_uniformfirstpriors.csv'.format(
            historical_dataset, mb_type, grad_type),
        index_col='Unnamed: 0')
else:
    pd_params = pd.read_csv(path +
                            'only_perfect_precalibration_alps_refglaciers_{}_{}_{}.csv'.format(
            historical_dataset, mb_type, grad_type),
        index_col='Unnamed: 0')
    # sys.exit('need to compute precalibration first...')

pd_geodetic_comp_alps = pd.read_csv(path + 'alps_geodetic_solid_prcp.csv')
pd_geodetic_comp_alps.index = pd_geodetic_comp_alps.rgiid

pd_geodetic_comp_alps = pd_geodetic_comp_alps.loc[pd_geodetic_comp_alps[
                                                      'solid_prcp_mean_nopf_weighted_{}_{}'.format(
                                                          dataset, typ)] >= 0]

pd_geodetic_comp_alps = pd_geodetic_comp_alps[
    ['dmdtda', 'err_dmdtda', 'dmdtda_2000_2010',
     'err_dmdtda_2000_2010', 'dmdtda_2010_2020', 'err_dmdtda_2010_2020',
     'reg', 'area', 'solid_prcp_mean_nopf_weighted_{}_{}'.format(dataset, typ),
     'max_allowed_specificMB']].astype(float)
# pd_geodetic_comp_alps = pd_geodetic_comp.loc[pd_geodetic_comp.reg==11.0]
if reset_gdir:
    gdirs = workflow.init_glacier_directories(
        pd_geodetic_comp_alps.dropna().index[start_ind:end_ind].values,
        from_prepro_level=2,
        prepro_border=10,
        prepro_base_url=base_url,
        prepro_rgi_version='62')

    if mb_type != 'mb_real_daily':
        cfg.PARAMS['baseline_climate'] = 'ERA5dr'
        workflow.execute_entity_task(tasks.process_ecmwf_data,
                                     gdirs, dataset='ERA5dr')
    else:
        cfg.PARAMS['baseline_climate'] = 'ERA5_daily'
        for gd in pd_geodetic_comp_alps.index[start_ind:end_ind]:
            process_era5_daily_data(gd)

elif compute_missing:
    #raise NotImplementedError('this has to be adapted')
    missing = []
    # only those where there are NO geodetic measurements !!!
    for rgi_id in pd_geodetic_comp_alps.dropna().index[
                  start_ind:end_ind].values:
        try:
            # ds_historical_j = xr.open_dataset(gdir.get_filepath('model_run')[:-12] +'model_run_historical_199.nc', cache=False)
            burned_trace = az.from_netcdf(path+ 'burned_trace_plus200samples/'+
                                          '/{}_burned_trace_plus200samples_{}_{}_{}_meltfprior{}.nc'.format(
                    rgi_id, dataset,
                    mb_type, grad_type, melt_f_prior))
        except:
            missing.append(rgi_id)
    print('amount of missing glaciers that are computed again: {}'.format(
        len(missing)))
    gdirs = workflow.init_glacier_directories(missing)
else:
    gdirs = workflow.init_glacier_directories(
        pd_geodetic_comp_alps.dropna().index[start_ind:end_ind])

burned_trace = az.from_netcdf(path + 'burned_trace_alps_regression_pf_{}_{}.nc'.format(
        dataset, typ))

# predict_data = xr.open_dataset('predict_alps_regression_pf_{}.nc'.format(dataset))
predict_data = burned_trace.predictions
# melt_f_prior == 'bayesian'
if melt_f_prior == 'frequentist':
    # this is actually not used anymore, beacuse we only need it for freqeuentist
    filepath = "/home/users/lschuster/bayesian_calibration/all/dict_calib_opt_pf_allrefglaciers_difftypes.pkl"
    dict_pd_calib_opt = pd.read_pickle(filepath)
    typ = '{}_{}'.format(mb_type, grad_type)
    pd_nonan = dict_pd_calib_opt[typ].loc[
        dict_pd_calib_opt[typ].pf_opt.dropna().index]
    pd_nonan_alps = pd_nonan[pd_nonan.O1Region == '11'][
        ['pf_opt', 'solid prcp mean nopf weighted', 'melt_f_opt_pf',
         'amount_glacmsm']].astype(float)
    pd_nonan_alps['solid_prcp_mean_nopf_weighted'] = pd_nonan_alps[
        'solid prcp mean nopf weighted']
    pd_nonan_alps_calib = pd_nonan_alps
else:
    pd_nonan_alps_calib = None



workflow.execute_entity_task(bayes_mbcalibration, gdirs, mb_type=mb_type, grad_type=grad_type,
                            melt_f_prior=melt_f_prior, path=path, uniform=uniform, cores=1,
                            dataset=dataset, 
                            pd_geodetic_comp_alps=pd_geodetic_comp_alps, predict_data=predict_data,
                            pd_nonan_alps_calib=pd_nonan_alps_calib, pd_params_calib=pd_params)