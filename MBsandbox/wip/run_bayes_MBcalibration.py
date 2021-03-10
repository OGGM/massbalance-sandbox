# this has been called "pieps" at the beginning
# this could be input arguments when executing the script
mb_type = 'mb_monthly'
grad_type = 'cte'
uniform_firstprior = True
start_ind = 0
end_ind = 3
cluster = True
melt_f_prior = 'freq_bayesian'
uniform = False
cluster = True
reset_gdir = False
compute_missing = True
import numpy as np
import sys

if cluster == True:
    start_ind = int(np.absolute(int(sys.argv[1])))
    end_ind = int(np.absolute(int(sys.argv[2])))
    mb_type = str(sys.argv[3])
    grad_type = str(sys.argv[4])
print(type(start_ind), type(end_ind), mb_type, grad_type)
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

if cluster:
    working_dir = '/home/users/lschuster/oggm_files/oneFlowline'
else:
    working_dir = '/home/lilianschuster/Schreibtisch/PhD/oggm_files/oneFlowline'
# this needs to be changed if working on another computer
if not os.path.exists(working_dir):
    working_dir = utils.gettempdir(dirname='OGGM_mb_type_calibration',
                                   reset=False)

cfg.PATHS['working_dir'] = working_dir
# use Huss flowlines
base_url = ('https://cluster.klima.uni-bremen.de/~oggm/gdirs/oggm_v1.4/'
            'L1-L2_files/elev_bands')

# import the MSsandbox modules
from MBsandbox.mbmod_daily_oneflowline import process_era5_daily_data, TIModel
from wip.bayes_calib_geod_direct import bayes_dummy_model_better

cfg.PARAMS['hydro_month_nh'] = 1
typ = '{}_{}'.format(mb_type, grad_type)

if typ == 'mb_monthly_cte':
    if uniform_firstprior == True:
        pd_params = pd.read_csv(
            '/home/users/lschuster/bayesian_calibration/only_perfect_precalibration_alps_refglaciers_uniformfirstpriors.csv',
            index_col='Unnamed: 0')
    else:
        pd_params = pd.read_csv(
            '/home/users/lschuster/bayesian_calibration/only_perfect_precalibration_alps_refglaciers.csv',
            index_col='Unnamed: 0')
else:
    if uniform_firstprior == True:
        pd_params = pd.read_csv(
            '/home/users/lschuster/bayesian_calibration/only_perfect_precalibration_alps_refglaciers_{}_{}_uniformfirstpriors.csv'.format(
                mb_type, grad_type),
            index_col='Unnamed: 0')
    else:
        pd_params = pd.read_csv(
            '/home/users/lschuster/bayesian_calibration/only_perfect_precalibration_alps_refglaciers_{}_{}.csv'.format(
                mb_type, grad_type),
            index_col='Unnamed: 0')
    # sys.exit('need to compute precalibration first...')

pd_geodetic_comp_alps = pd.read_csv(
    '/home/users/lschuster/bayesian_calibration/alps_geodetic_solid_prcp.csv',
    )
pd_geodetic_comp_alps.index = pd_geodetic_comp_alps.rgiid

dataset = 'ERA5'
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
        pd_geodetic_comp_alps.index[start_ind:end_ind].values,
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
    missing = []
    # only those where there are NO geodetic measurements !!!
    for rgi_id in pd_geodetic_comp_alps.dropna().index[
                  start_ind:end_ind].values:
        try:
            # ds_historical_j = xr.open_dataset(gdir.get_filepath('model_run')[:-12] +'model_run_historical_199.nc', cache=False)
            burned_trace = az.from_netcdf(
                'alps/{}_burned_trace_plus200samples_{}_{}_{}_meltfprior{}.nc'.format(
                    rgi_id, dataset,
                    mb_type, grad_type, melt_f_prior))
        except:
            missing.append(rgi_id)
    print('amount of missing glaciers that are computed again: {}'.format(
        len(missing)))
    gdirs = workflow.init_glacier_directories(missing)
else:
    gdirs = workflow.init_glacier_directories(
        pd_geodetic_comp_alps.index[start_ind:
                                    end_ind])
    if mb_type != 'mb_real_daily':
        cfg.PARAMS['baseline_climate'] = 'ERA5dr'
    else:
        cfg.PARAMS['baseline_climate'] = 'ERA5_daily'
if typ == 'mb_monthly_cte':

    burned_trace = az.from_netcdf(
        '/home/users/lschuster/bayesian_calibration/burned_trace_alps_regression_pf_{}.nc'.format(
            dataset))
else:
    burned_trace = az.from_netcdf(
        '/home/users/lschuster/bayesian_calibration/burned_trace_alps_regression_pf_{}_{}.nc'.format(
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
    pd_nonan_alps['1/solid prcp mean nopf weighted'] = 1 / pd_nonan_alps[
        'solid prcp mean nopf weighted']
    pd_nonan_alps['solid_prcp_mean_nopf_weighted'] = pd_nonan_alps[
        'solid prcp mean nopf weighted']
    pd_nonan_alps_calib = pd_nonan_alps
pd_params_calib = pd_params
for gd in gdirs:
    try:
        print(gd.rgi_id)
        max_allowed_specificMB = pd_geodetic_comp_alps.loc[
            gd.rgi_id, 'max_allowed_specificMB']

        h, w = gd.get_inversion_flowline_hw()

        # at instantiation use prcp_fac = 2.5, change this in def_get_mb later on
        gd_mb = TIModel(gd, 150, mb_type=mb_type, N=100, prcp_fac=2.5,
                        grad_type=grad_type)
        gd_mb.historical_climate_qc_mod(gd)

        with pm.Model() as model_new:
            pf = pm.TruncatedNormal('pf',
                                    mu=predict_data.sel(
                                        rgi_id=gd.rgi_id).pf_interp.mean(),
                                    sigma=predict_data.sel(
                                        rgi_id=gd.rgi_id).pf_interp.std(),
                                    lower=0.001, upper=10)
            # maybe take also predict_data interpolated mu somehow ??? for that prcp_solid has to predict pf and melt_f at once ...
            # or use instead a uniform distribution ??? or interpolate a descriptor for both pf and melt_f ???
            if melt_f_prior == 'frequentist':
                melt_f = pm.TruncatedNormal('melt_f',
                                            mu=pd_nonan_alps_calib[
                                                'melt_f_opt_pf'].mean(),
                                            sigma=pd_nonan_alps_calib[
                                                'melt_f_opt_pf'].std(),
                                            lower=1, upper=1000)
            elif melt_f_prior == 'freq_bayesian':
                melt_f = pm.TruncatedNormal('melt_f',
                                            mu=pd_params_calib[
                                                'melt_f_mean_calib'].mean(),
                                            sigma=pd_params_calib[
                                                'melt_f_mean_calib'].std(),
                                            lower=1, upper=1000)
            elif melt_f_prior == 'bayesian':
                # this does NOT work ...
                # mu_melt_f_bayes = pm.Data('mu_melt_f_bayes', pd_params_calib['melt_f_mean_calib'])
                # sigma_melt_f_bayes = pm.Data('sigma_melt_f_bayes', pd_params_calib['melt_f_std_calib'])
                # could also use from_posterior(...) to be more precise: https://docs.pymc.io/notebooks/updating_priors.html
                mu_melt_f_bayes = pm.TruncatedNormal('mu_melt_f_bayes',
                                                     mu=pd_params_calib[
                                                         'melt_f_mean_calib'].mean(),
                                                     sigma=pd_params_calib[
                                                         'melt_f_mean_calib'].std())
                sigma_melt_f_bayes = pm.TruncatedNormal('sigma_melt_f_bayes',
                                                        mu=pd_params_calib[
                                                            'melt_f_std_calib'].mean(),
                                                        sigma=pd_params_calib[
                                                            'melt_f_std_calib'].std())
                melt_f = pm.TruncatedNormal('melt_f',
                                            mu=mu_melt_f_bayes,
                                            sigma=sigma_melt_f_bayes,
                                            lower=1, upper=1000)
        # print(pd_geodetic_comp_alps.loc[gd.rgi_id])
        burned_trace_valid, model_T_valid, _ = bayes_dummy_model_better(uniform,
                                                                        max_allowed_specificMB=max_allowed_specificMB,
                                                                        gd=gd,
                                                                        sampler='nuts',
                                                                        ys=np.arange(
                                                                            2000,
                                                                            2019,
                                                                            1),
                                                                        gd_mb=gd_mb,
                                                                        h=h,
                                                                        w=w,
                                                                        use_two_msm=True,
                                                                        nosigma=False,
                                                                        # predict_data = predict_data,
                                                                        model=model_new,
                                                                        # pd_calib_opt=pd_calib_opt,
                                                                        first_ppc=False,
                                                                        first_ppc_200=True,
                                                                        predict_historic=False,
                                                                        pd_geodetic_comp=pd_geodetic_comp_alps)
        # burned_trace_valid.posterior_predictive = burned_trace_valid.posterior_predictive.sel(chain=0).drop('chain')
        burned_trace_valid.to_netcdf(
            '/home/users/lschuster/bayesian_calibration/alps/{}_burned_trace_plus200samples_{}_{}_{}_meltfprior{}.nc'.format(
                gd.rgi_id, dataset, mb_type, grad_type, melt_f_prior))
    except:
        print('failed: {}'.format(gd.rgi_id))
        pass