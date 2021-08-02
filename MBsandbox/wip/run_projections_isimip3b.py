start_ind = 0
end_ind = 10

ensemble = 'mri-esm2-0_r1i1p1f1'
ssps = ['ssp126', 'ssp370'] #, 'ssp585']

# this could be input arguments when executing the script
uniform_firstprior = True

cluster = True
melt_f_prior = 'freq_bayesian'
uniform = False
cluster = True

dataset = 'WFDE5_CRU'
step = 'run_proj'

import numpy as np 
import sys
#start_ind = int(np.absolute(int(sys.argv[1])))
#end_ind = int(np.absolute(int(sys.argv[2])))
step = str(sys.argv[1])
# glen_a = 'single'
glen_a = 'per_glacier_and_draw'
    
import os


if step == 'run_proj':
    jobid = int(os.environ.get('JOBID'))
    if jobid >27:
        sys.exit('do not need more arrays in alps...')
    #jobid = int(sys.argv[2])
    print('this is job id {}, '.format(jobid))
    start_ind = int(jobid*127)
    end_ind = int(jobid*127 + 127)
    print('start_ind: {}, end_ind: {}'.format(start_ind, end_ind))
    print(ensemble, dataset, step)
#    mb_type = str(sys.argv[3])
#    grad_type = str(sys.argv[4])
#print(type(start_ind), type(end_ind), mb_type, grad_type)
###############################
import pymc3 as pm
# 	conda install -c conda-forge python-graphviza
import pandas as pd
import xarray as xr
import seaborn as sns
import pickle
import ast

import matplotlib.pyplot as plt
import matplotlib

# %matplotlib inline
import statsmodels as stats
import scipy
import scipy.stats as stats
from IPython.core.pylabtools import figsize
import os
import oggm
from oggm import cfg, utils, workflow, tasks, graphics
from oggm.core import massbalance, flowline, climate
import logging
log = logging.getLogger(__name__)
#import aesara.tensor as aet
#import aesara

# from drounce_analyze_mcmc import effective_n, mcse_batchmeans
# plotting bayesian stuff
import arviz as az
az.rcParams['stats.hdi_prob'] = 0.95


cfg.initialize(logging_level='WARNING') # logging_level='WARNING'
cfg.PARAMS['use_multiprocessing'] = True
cfg.PARAMS['continue_on_error'] = True

working_dir = os.environ.get('OGGM_WORKDIR')
# working_dir = '/home/users/lschuster/oggm_files/all'
cfg.PATHS['working_dir'] = working_dir
# use Huss flowlines
base_url = ('https://cluster.klima.uni-bremen.de/~oggm/gdirs/oggm_v1.4/'
            'L1-L2_files/elev_bands')

import theano
import theano.tensor as aet


from oggm.shop import gcm_climate
from oggm import entity_task

# import the MBsandbox modules
from MBsandbox.mbmod_daily_oneflowline import process_w5e5_data
from MBsandbox.mbmod_daily_oneflowline import process_era5_daily_data, TIModel, BASENAMES
from MBsandbox.help_func import compute_stat, minimize_bias, optimize_std_quot_brentq
from MBsandbox.wip.help_func_geodetic import minimize_bias_geodetic, optimize_std_quot_brentq_geod, get_opt_pf_melt_f
from MBsandbox.wip.bayes_calib_geod_direct import get_TIModel_clim_model_type, get_slope_pf_melt_f, bayes_dummy_model_better, bayes_dummy_model_ref_std, bayes_dummy_model_ref

from MBsandbox.wip.projections_bayescalibration import (process_isimip_data,
                                                        inversion_and_run_from_climate_with_bayes_mb_params)


cfg.PARAMS['hydro_month_nh']=1

pd_geodetic_comp_alps = pd.read_csv(
    '/home/users/lschuster/bayesian_calibration/WFDE5_ISIMIP/alps_geodetic_solid_prcp.csv',
    )
pd_geodetic_comp_alps.index = pd_geodetic_comp_alps.rgiid


# TODO: check if this is right!!!
y0 = 1979
ye = 2018+1

if step=='ice_thickness_calibration':
    print(len(pd_geodetic_comp_alps.dropna().index.values))

    gdirs = workflow.init_glacier_directories( pd_geodetic_comp_alps.dropna().index.values)
    print(len(gdirs))
    # cfg.set_logging_config(logging_level='WARNING')
    # Get end date. The first gdir might have blown up, try some others


    
    #if start_ind != 0 and end_ind < 3400:
    #    sys.exit(
    print('we want to calibrate for all Alpine glaciers at once, so all glaciers are selected, even if start_ind or end_ind are given')
    for mb_type in ['mb_monthly', 'mb_pseudo_daily', 'mb_real_daily']:
        for grad_type in ['cte', 'var_an_cycle']:
            # compute apparent mb from any mb ... 
            print(mb_type, grad_type)
            
            if glen_a == 'single':
                for gdir in gdirs:
                    try:
                        # in this case a-factor calibrated individually for each glacier ... 
                        sample_path = '/home/users/lschuster/bayesian_calibration/WFDE5_ISIMIP/burned_trace_plus200samples/'
                        burned_trace = az.from_netcdf(sample_path + '{}_burned_trace_plus200samples_WFDE5_CRU_{}_{}_meltfpriorfreq_bayesian.nc'.format(gdir.rgi_id, mb_type, grad_type))

                        melt_f_point_estimate = az.plots.plot_utils.calculate_point_estimate(
                            'mean', burned_trace.posterior.melt_f.stack(
                                draws=("chain", "draw"))).values
                        pf_point_estimate = az.plots.plot_utils.calculate_point_estimate(
                            'mean',
                            burned_trace.posterior.pf.stack(draws=("chain", "draw"))).values

                        mb = TIModel(gdir, melt_f_point_estimate, mb_type=mb_type,
                                     grad_type=grad_type, baseline_climate=dataset,
                                     residual=0, prcp_fac=pf_point_estimate)
                        mb.historical_climate_qc_mod(gdir)

                        climate.apparent_mb_from_any_mb(gdir, mb_model=mb,
                                                        mb_years=np.arange(y0, ye, 1))
                    except:
                        print('burned_trace is not working for glacier: {} with {} {}'.format(gdir.rgi_id, mb_type, grad_type))
                # Inversion: we match the consensus
                #TODO: check this filtering approach with Fabien!
                border = 80
                filter = border >= 20
                # here I calibrate on glacier per glacier basis!
                df = oggm.workflow.calibrate_inversion_from_consensus(gdirs,
                                                                      apply_fs_on_mismatch=False,
                                                                      error_on_mismatch=False,
                                                                      filter_inversion_output=filter)


                # check if calibration worked: total volume of OGGM selected glaciers should ratio to ITMIX should be closer than one percent
                np.testing.assert_allclose(
                    df.sum()['vol_itmix_m3'] / df.sum()['vol_oggm_m3'], 1, rtol=1e-2)

                a_factor = gdirs[0].get_diagnostics()['inversion_glen_a'] / cfg.PARAMS['inversion_glen_a']     
                np.testing.assert_allclose(a_factor, gdirs[-1].get_diagnostics()['inversion_glen_a'] / cfg.PARAMS['inversion_glen_a'])      

                # ToDO: need to include the a-factor computation in a task ... and somehow make sure that the right a-factor is used for the right mb type ... 
                #workflow.execute_entity_task(
                print(a_factor)

                df['glen_a_factor_calib'] = a_factor
                df.to_csv(working_dir + '/ice_thickness_inversion_farinotti_calib_{}_{}_{}_meltfprior{}.csv'.format(
                    dataset, mb_type, grad_type, melt_f_prior))
 
            elif glen_a =='per_glacier_and_draw_with_uncertainties':
                sys.exit('not yet implemented')
elif step == 'run_proj':
    print(type(start_ind))
    run_init = True
    if run_init:
        gdirs = workflow.init_glacier_directories(pd_geodetic_comp_alps.dropna().index.values[start_ind:end_ind],
                                                  from_prepro_level=2,
                                                  prepro_border=80,
                                                  prepro_base_url=base_url,
                                                  prepro_rgi_version='62')

        workflow.execute_entity_task(tasks.compute_downstream_line, gdirs)
        workflow.execute_entity_task(tasks.compute_downstream_bedshape, gdirs)

        #workflow.execute_entity_task(oggm.shop.ecmwf.process_ecmwf_data, gdirs, dataset='ERA5dr', output_filesuffix='_monthly_ERA5dr')
        #workflow.execute_entity_task(process_era5_daily_data, gdirs, output_filesuffix='_daily_ERA5dr')

        workflow.execute_entity_task(process_w5e5_data, gdirs, output_filesuffix='_daily_WFDE5_CRU', temporal_resol='daily',
                            climate_path='/home/www/lschuster/', cluster=True)
        workflow.execute_entity_task(process_w5e5_data, gdirs, output_filesuffix='_monthly_WFDE5_CRU', temporal_resol='monthly',
                            climate_path='/home/www/lschuster/', cluster=True)

        print('start_monthly')
        for ssp in ['ssp126', 'ssp370']: #, 'ssp585']:
            print(ssp)
            workflow.execute_entity_task(process_isimip_data, gdirs, 
                                     ensemble = ensemble,
                                     ssp = ssp,
                                     temporal_resol ='monthly',
                                     climate_historical_filesuffix='_monthly_WFDE5_CRU',
                                        cluster=True);
        print('start daily')
        for ssp in ['ssp126', 'ssp370']: #, 'ssp585']:
            print(ssp)
            workflow.execute_entity_task(process_isimip_data, gdirs, 
                                     ensemble = ensemble,
                                     ssp = ssp,
                                     temporal_resol ='daily',
                                     climate_historical_filesuffix='_daily_WFDE5_CRU',
                                        cluster=True);
    else:
        gdirs = workflow.init_glacier_directories(pd_geodetic_comp_alps.dropna().index.values[start_ind:end_ind])
    
    for mb_type in ['mb_monthly', 'mb_pseudo_daily', 'mb_real_daily']: # 'mb_monthly',
        for grad_type in ['cte', 'var_an_cycle']:
            log.workflow(print(len(gdirs)))
            log.workflow(print(mb_type, grad_type))
            if glen_a == 'single':
                a_factor = pd.read_csv(working_dir + '/ice_thickness_inversion_farinotti_calib_{}_{}_{}_meltfprior{}.csv'.format(dataset, mb_type, grad_type, melt_f_prior))['glen_a_factor_calib'].mean()
            elif glen_a == 'per_glacier_and_draw':
                # no uncertainties in Farinotti ice thickness estimate assumed
                a_factor = glen_a
            log.workflow(print(a_factor))
            
            workflow.execute_entity_task(inversion_and_run_from_climate_with_bayes_mb_params, gdirs, ssps = ssps,
                    a_factor=a_factor, y0=y0, ye_h=2014, mb_type=mb_type,
                    grad_type=grad_type, ensemble=ensemble, #burned_trace=burned_trace,
                    melt_f_prior=melt_f_prior, dataset=dataset,
                                        path_proj=working_dir + '/proj/')
            
            
