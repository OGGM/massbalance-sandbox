# this could be input arguments when executing the script
mb_type = 'mb_monthly'
grad_type = 'cte'
uniform_firstprior = True
# start_ind = 0
# end_ind = 0
cluster = True
melt_f_prior = 'freq_bayesian'
uniform = False
cluster = True
reset_gdir = False
dataset = 'ERA5'
run_proj = False
ice_thickness_calibration = False

import numpy as np
import sys

if cluster == True:
    start_ind = int(np.absolute(int(sys.argv[1])))
    end_ind = int(np.absolute(int(sys.argv[2])))
    mb_type = str(sys.argv[3])
    grad_type = str(sys.argv[4])
print(type(start_ind), type(end_ind), mb_type, grad_type)
###############################
# 	conda install -c conda-forge python-graphviza
import pandas as pd
import xarray as xr

# %matplotlib inline
import os
import oggm
from oggm import cfg, utils, workflow, tasks
from oggm.core import climate
import logging

log = logging.getLogger(__name__)
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
    working_dir = '/home/users/lschuster/oggm_files/projections'
else:
    working_dir = '/home/lilianschuster/Schreibtisch/PhD/oggm_files/projections'


cfg.PATHS['working_dir'] = working_dir
# use Huss flowlines
base_url = ('https://cluster.klima.uni-bremen.de/~oggm/gdirs/oggm_v1.4/'
            'L1-L2_files/elev_bands')

# import the MSsandbox modules

from oggm.shop import gcm_climate

from MBsandbox.mbmod_daily_oneflowline import TIModel

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
    gdirs = workflow.init_glacier_directories(pd_geodetic_comp_alps.index,
                                              from_prepro_level=2,
                                              prepro_border=80,
                                              prepro_base_url=base_url,
                                              prepro_rgi_version='62')

    workflow.execute_entity_task(tasks.compute_downstream_line, gdirs)
    workflow.execute_entity_task(tasks.compute_downstream_bedshape, gdirs)

    # this is at least independent of the mass balance, but not independent of climate:
    cfg.PARAMS['baseline_climate'] = 'ERA5dr'
    workflow.execute_entity_task(oggm.shop.ecmwf.process_ecmwf_data, gdirs,
                                 dataset='ERA5dr')

    # for gdir in gdirs:
    #    try:
    #        # can do workflow.execute_entity_task
    # oggm.shop.ecmwf.process_ecmwf_data(gdir, dataset='ERA5dr')
    #    except:
    #        print(gdir.rgi_id)

    # download the CMIP5 files and bias correct them to the calibration data
    try:
        bp = 'https://cluster.klima.uni-bremen.de/~oggm/cmip5-ng/pr/pr_mon_CCSM4_{}_r1i1p1_g025.nc'
        bt = 'https://cluster.klima.uni-bremen.de/~oggm/cmip5-ng/tas/tas_mon_CCSM4_{}_r1i1p1_g025.nc'
        for rcp in ['rcp26', 'rcp45', 'rcp60', 'rcp85']:
            # Download the files
            ft = utils.file_downloader(bt.format(rcp))
            fp = utils.file_downloader(bp.format(rcp))
            # bias correct them
            workflow.execute_entity_task(gcm_climate.process_cmip_data, gdirs,
                                         filesuffix='_CCSM4_{}'.format(rcp),
                                         # recognize the climate file for later
                                         fpath_temp=ft,
                                         # temperature projections
                                         fpath_precip=fp,  # precip projections
                                         );
    except:
        pass
else:
    gdirs = workflow.init_glacier_directories(
        pd_geodetic_comp_alps.index[start_ind:end_ind])

# df = ['RGI60-11.00183']


# cfg.set_logging_config(logging_level='WARNING')
# Get end date. The first gdir might have blown up, try some others
i = 0
while True:
    if i >= len(gdirs):
        raise RuntimeError('Found no valid glaciers!')
    try:
        y0 = gdirs[i].get_climate_info()[
            'baseline_hydro_yr_0']  # why is this 1979 and not 1980, I set hydro_month_nh to 1 !!!
        # One adds 1 because the run ends at the end of the year
        ye = gdirs[i].get_climate_info()['baseline_hydro_yr_1'] + 1
        break
    except BaseException:
        i += 1
print(gdirs[0].rgi_id)
# only first 5 yet ...
# if first_run:

# make an execute_entity_task out of this for the logs?
if ice_thickness_calibration:
    if start_ind != 0 and end_ind < 3400:
        sys.exit(
            'we want to calibrate for all Alpine glaciers at once, so all glaciers have to be selected!')
    # first need to compute the apparent_mb_from_any_mb using the medium best pf/melt_f combination ...
    for gdir in gdirs:
        try:
            burned_trace = az.from_netcdf(
                'alps/{}_burned_trace_plus200samples_{}_{}_{}_meltfprior{}.nc'.format(
                    gdir.rgi_id, dataset, mb_type, grad_type, melt_f_prior))
            melt_f_point_estimate = az.plots.plot_utils.calculate_point_estimate(
                'mean', burned_trace.posterior.melt_f.stack(
                    draws=("chain", "draw"))).values
            pf_point_estimate = az.plots.plot_utils.calculate_point_estimate(
                'mean',
                burned_trace.posterior.pf.stack(draws=("chain", "draw"))).values

            mb = TIModel(gdir, melt_f_point_estimate, mb_type=mb_type,
                         grad_type=grad_type,
                         residual=0, prcp_fac=pf_point_estimate)
            mb.historical_climate_qc_mod(gdir)

            climate.apparent_mb_from_any_mb(gdir, mb_model=mb,
                                            mb_years=np.arange(y0, ye, 1))
        except:
            print(gdir.rgi_id)

    # Inversion: we match the consensus
    border = 80
    filter = border >= 20
    df = oggm.workflow.calibrate_inversion_from_consensus(gdirs,
                                                          apply_fs_on_mismatch=True,
                                                          error_on_mismatch=False,
                                                          filter_inversion_output=filter)

    # just check if they are the same
    np.testing.assert_allclose(gdirs[0].get_diagnostics()['inversion_glen_a'],
                               gdirs[100].get_diagnostics()['inversion_glen_a'])
    # check if calibration worked: total volume of OGGM selected glaciers should ratio to ITMIX should be closer than one percent
    np.testing.assert_allclose(
        df.sum()['vol_itmix_m3'] / df.sum()['vol_oggm_m3'], 1, rtol=1e-2)

    a_factor = gdirs[0].get_diagnostics()['inversion_glen_a'] / cfg.PARAMS[
        'inversion_glen_a']

    df['glen_a_factor_calib'] = a_factor
    df.to_csv(
        working_dir + '/ice_thickness_inversion_farinotti_calib_{}_{}_{}_meltfprior{}.nc'.format(
            dataset, mb_type, grad_type, melt_f_prior))
else:
    a_factor = \
    pd.read_csv(working_dir + '/ice_thickness_inversion_farinotti_calib.csv')[
        'glen_a_factor_calib'].mean()

if run_proj:
    workflow.execute_entity_task(
        inversion_and_run_from_climate_with_bayesian_mb_params, gdirs,
        a_factor=a_factor, y0=y0, ye=ye, mb_type=mb_type,
        grad_type=grad_type, rcps=['rcp26', 'rcp45', 'rcp60', 'rcp85'],
        melt_f_prior=melt_f_prior, dataset=dataset)
    # execute entity task ....

# take sample j and do inversion, best glenA calibration ..., but why should I merge draw j of glacier i with draw j of glacier i+1,
# do I need to repeat all of that for every calibration parameter combination ???
# need first climate data ... do I need this and what is it doing???
# can I convert it to a execute_entity_task and put inside the correct mb_model?


for gdir in gdirs:
    try:
        rgi_id = gdir.rgi_id
        try:
            xr.open_dataset(
                'projections/merged/{}_gcm_{}_{}_{}_{}'.format(rgi_id, dataset,
                                                               mb_type,
                                                               grad_type,
                                                               melt_f_prior))
        except:
            burned_trace = az.from_netcdf(
                'alps/{}_burned_trace_plus200samples_{}_{}_{}_meltfprior{}.nc'.format(
                    rgi_id, dataset, mb_type, grad_type, melt_f_prior))
            burned_trace.posterior_predictive = burned_trace.posterior_predictive.sel(
                chain=0).drop('chain')

            ds_historical = xr.open_dataset(gdir.get_filepath('model_run')[
                                            :-12] + 'model_run_historical_0.nc',
                                            cache=False)
            ds_historical.coords['draw'] = 0
            ds_gcm_rcp26 = xr.open_dataset(gdir.get_filepath('model_run')[
                                           :-12] + 'model_run_CCSM4_rcp26_0.nc',
                                           cache=False)
            ds_gcm_rcp26.coords['draw'] = 0
            ds_gcm_rcp26.coords['rcp'] = 'rcp26'

            ds_gcm_rcp45 = xr.open_dataset(gdir.get_filepath('model_run')[
                                           :-12] + 'model_run_CCSM4_rcp45_0.nc',
                                           cache=False)
            ds_gcm_rcp45.coords['draw'] = 0
            ds_gcm_rcp45.coords['rcp'] = 'rcp45'

            ds_gcm_rcp60 = xr.open_dataset(gdir.get_filepath('model_run')[
                                           :-12] + 'model_run_CCSM4_rcp60_0.nc',
                                           cache=False)
            ds_gcm_rcp60.coords['draw'] = 0
            ds_gcm_rcp60.coords['rcp'] = 'rcp60'

            ds_gcm_rcp85 = xr.open_dataset(gdir.get_filepath('model_run')[
                                           :-12] + 'model_run_CCSM4_rcp85_0.nc',
                                           cache=False)
            ds_gcm_rcp85.coords['draw'] = 0
            ds_gcm_rcp85.coords['rcp'] = 'rcp85'
            ds_gcm_0 = xr.concat(
                [ds_gcm_rcp26, ds_gcm_rcp45, ds_gcm_rcp60, ds_gcm_rcp85], 'rcp')

            ds_gcm_0['melt_f'] = burned_trace.posterior_predictive.sel(
                draw=0).melt_f
            ds_gcm_0['pf'] = burned_trace.posterior_predictive.sel(draw=0).pf
            ds_historical['melt_f'] = burned_trace.posterior_predictive.sel(
                draw=0).melt_f
            ds_historical['pf'] = burned_trace.posterior_predictive.sel(
                draw=0).pf

            for jj in np.arange(1, 200):
                try:
                    ds_historical_j = xr.open_dataset(
                        gdir.get_filepath('model_run')[
                        :-12] + 'model_run_historical_{}.nc'.format(jj),
                        cache=False)
                    ds_historical_j.coords['draw'] = jj
                    ds_historical_j[
                        'melt_f'] = burned_trace.posterior_predictive.sel(
                        draw=jj).melt_f
                    ds_historical_j[
                        'pf'] = burned_trace.posterior_predictive.sel(
                        draw=jj).pf
                    ds_historical = xr.concat([ds_historical, ds_historical_j],
                                              'draw')
                except:
                    print('draw {} did NOT work'.format(jj))

            ds_gcm_rr = []
            for rcp in ['rcp26', 'rcp45', 'rcp60', 'rcp85']:
                ds_gcm_r = ds_gcm_0.sel(rcp=rcp)
                for jj in np.arange(1, 200):
                    ds_gcm_j = xr.open_dataset(gdir.get_filepath('model_run')[
                                               :-12] + 'model_run_CCSM4_{}_{}.nc'.format(
                        rcp, jj), cache=False)
                    ds_gcm_j.coords['draw'] = jj
                    ds_gcm_j.coords['rcp'] = rcp
                    ds_gcm_j['melt_f'] = burned_trace.posterior_predictive.sel(
                        draw=jj).melt_f
                    ds_gcm_j['pf'] = burned_trace.posterior_predictive.sel(
                        draw=jj).pf
                    ds_gcm_r = xr.concat([ds_gcm_r, ds_gcm_j], 'draw')

                ds_gcm_rr.append(ds_gcm_r)
            ds_gcm = xr.concat(ds_gcm_rr, 'rcp')

            # if repeat_inversion:
            ds_historical.to_netcdf(
                'projections/merged/{}_historical_{}_{}_{}_{}'.format(rgi_id,
                                                                      dataset,
                                                                      mb_type,
                                                                      grad_type,
                                                                      melt_f_prior))
            ds_gcm.to_netcdf(
                'projections/merged/{}_gcm_{}_{}_{}_{}'.format(rgi_id, dataset,
                                                               mb_type,
                                                               grad_type,
                                                               melt_f_prior))
            # else:
            #    ds_historical.to_netcdf('{}_historical_noinvers'.format(rgi_id))
            #    ds_gcm.to_netcdf('{}_gcm_noinvers'.format(rgi_id))

            # remove the old _j files ...
    except:
        pass

