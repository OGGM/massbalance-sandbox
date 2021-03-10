import pymc3 as pm
# 	conda install -c conda-forge python-graphviza
import numpy as np
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
from oggm.core import massbalance, flowline
from oggm import entity_task

from oggm.core.flowline import FileModel
# import aesara.tensor as aet
# import aesara

from oggm.exceptions import InvalidParamsError, InvalidWorkflowError
from oggm.cfg import SEC_IN_YEAR, SEC_IN_MONTH, SEC_IN_DAY
import warnings
# from drounce_analyze_mcmc import effective_n, mcse_batchmeans
# plotting bayesian stuff
import arviz as az

az.rcParams['stats.hdi_prob'] = 0.95

# import the MSsandbox modules
from MBsandbox.mbmod_daily_oneflowline import process_era5_daily_data, TIModel, \
    BASENAMES
from MBsandbox.help_func import compute_stat, minimize_bias, \
    optimize_std_quot_brentq

import theano
import theano.tensor as aet

from MBsandbox.mbmod_daily_oneflowline import MultipleFlowlineMassBalance_TIModel
from oggm.shop.gcm_climate import process_gcm_data
from oggm.core.flowline import flowline_model_run
from oggm.core.massbalance import MultipleFlowlineMassBalance, MassBalanceModel
from oggm.core import climate

import logging

log = logging.getLogger(__name__)



###############################################
# # do it similar as in run_from_climate_data()
@entity_task(log)
def run_from_climate_data_TIModel(gdir, ys=None, ye=None, min_ys=None,
                                  max_ys=None,
                                  store_monthly_step=False,
                                  climate_filename='climate_historical',
                                  climate_input_filesuffix='',
                                  output_filesuffix='',
                                  init_model_filesuffix=None,
                                  init_model_yr=None,
                                  init_model_fls=None,
                                  zero_initial_glacier=False,
                                  bias=None, temperature_bias=None, melt_f=None,
                                  precipitation_factor=None,
                                  check_calib_params=False,
                                  mb_type='mb_monthly', grad_type='cte',
                                  **kwargs):
    """
    same as in run_from_climate_data but compatible with TIModel
    """
    # same as run_from_climate_data but compatible with TIModel ...
    if init_model_filesuffix is not None:
        fp = gdir.get_filepath('model_run', filesuffix=init_model_filesuffix)
        with FileModel(fp) as fmod:
            if init_model_yr is None:
                init_model_yr = fmod.last_yr
            fmod.run_until(init_model_yr)
            init_model_fls = fmod.fls
            if ys is None:
                ys = init_model_yr

    # Take from rgi date if not set yet
    if ys is None:
        try:
            ys = gdir.rgi_date.year
        except AttributeError:
            ys = gdir.rgi_date
        # The RGI timestamp is in calendar date - we convert to hydro date,
        # i.e. 2003 becomes 2004 (so that we don't count the MB year 2003
        # in the simulation)
        # See also: https://github.com/OGGM/oggm/issues/1020
        ys += 1

    # Final crop
    if min_ys is not None:
        ys = ys if ys > min_ys else min_ys
    if max_ys is not None:
        ys = ys if ys < max_ys else max_ys

    mb = MultipleFlowlineMassBalance_TIModel(gdir, mb_model_class=TIModel,
                                             prcp_fac=precipitation_factor,
                                             melt_f=melt_f,
                                             filename=climate_filename,
                                             bias=bias,
                                             input_filesuffix=climate_input_filesuffix,
                                             mb_type=mb_type,
                                             grad_type=grad_type,
                                             # check_calib_params=check_calib_params,
                                             )

    # if temperature_bias is not None:
    #    mb.temp_bias = temperature_bias
    if precipitation_factor is not None:
        mb.prcp_fac = precipitation_factor

    # do the quality check!
    mb.flowline_mb_models[-1].historical_climate_qc_mod(gdir)

    if ye is None:
        # Decide from climate (we can run the last year with data as well)
        ye = mb.flowline_mb_models[0].ye + 1

    return flowline_model_run(gdir, output_filesuffix=output_filesuffix,
                              mb_model=mb, ys=ys, ye=ye,
                              store_monthly_step=store_monthly_step,
                              init_model_fls=init_model_fls,
                              zero_initial_glacier=zero_initial_glacier,
                              **kwargs)


# make an execute_entity_task out of this to make it parallelisable!
@entity_task(log)
def inversion_and_run_from_climate_with_bayesian_mb_params(gdir, a_factor=1,
                                                           y0=None, ye=None,
                                                           mb_type='mb_monthly',
                                                           grad_type='cte',
                                                           rcps=['rcp26',
                                                                 'rcp45',
                                                                 'rcp60',
                                                                 'rcp85'],
                                                           melt_f_prior=None,
                                                           dataset=None):
    """ Does inversion tasks for predefined a_factor and then runs a glacier first with historical then with projections

    TODO: add ensemble type as option, add historical and projection climate as options ... maybe similar to run_from_climate_data
    """
    # for gdir in gdirs:
    # print(gdir.rgi_id)
    # instead: create a file with only the meltf_pf combinations for each glacier merges, this is faster than opening it always again ...
    # try:
    burned_trace = az.from_netcdf(
        'alps/{}_burned_trace_plus200samples_{}_{}_{}_meltfprior{}.nc'.format(
            gdir.rgi_id, dataset, mb_type, grad_type, melt_f_prior))
    burned_trace.posterior_predictive = burned_trace.posterior_predictive.sel(
        chain=0).drop('chain')

    # instantiatoin can happen independent of pf and melt_f:
    mb = TIModel(gdir, 200, mb_type=mb_type, grad_type=grad_type,
                 residual=0)
    mb.historical_climate_qc_mod(gdir)
    for j in np.arange(0, 200):
        # get the parameter of draw j:
        # and put them on TIModel
        # try:
        melt_f = burned_trace.posterior_predictive.sel(draw=j).melt_f.values
        pf = burned_trace.posterior_predictive.sel(draw=j).pf.values
        # print(type(pf), melt_f, type(y0), ye)
        # ye=2018
        mb.melt_f = melt_f
        mb.prcp_fac = pf

        # get the apparent_mb
        climate.apparent_mb_from_any_mb(gdir, mb_model=mb,
                                        mb_years=np.arange(y0, ye, 1))
        # let's do this without calibrating to the ice thickness consensus first:
        # a_factor = 1
        tasks.prepare_for_inversion(gdir)
        tasks.mass_conservation_inversion(gdir, glen_a=cfg.PARAMS[
                                                           'glen_a'] * a_factor)
        tasks.filter_inversion_output(gdir)
        tasks.init_present_time_glacier(gdir)
        # workflow.execute_entity_task(tasks.prepare_for_inversion, gdirs)
        # workflow.execute_entity_task(tasks.mass_conservation_inversion, gdirs, glen_a=cfg.PARAMS['glen_a']*a_factor)
        # workflow.execute_entity_task(tasks.filter_inversion_output, gdirs)
        # workflow.execute_entity_task(tasks.init_present_time_glacier, gdirs);

        # should I somehow write that differently to allow mb to put directly inside???
        run_from_climate_data_TIModel(gdir, bias=0, min_ys=y0, ye=ye,
                                      precipitation_factor=pf, melt_f=melt_f,
                                      output_filesuffix='_historical_{}'.format(
                                          j))
        # ds = utils.compile_run_output(gdirs, input_filesuffix='_historical_{}'.format(j))
        # ds.volume.plot(hue='rgi_id', ax = ax);

        for rcp in rcps:
            rid = '_CCSM4_{}'.format(rcp)
            run_from_climate_data_TIModel(gdir, bias=0,
                                          precipitation_factor=pf,
                                          melt_f=melt_f,
                                          output_filesuffix=rid + '_{}'.format(
                                              j),
                                          climate_input_filesuffix=rid,
                                          init_model_filesuffix='_historical_{}'.format(
                                              j), climate_filename='gcm_data',
                                          ys=2019)
        # maybe merge them here directly and remove the old _j files ???
        # except:
        #    print(gdir.rgi_id)
        # except:
        # print(gdir.rgi_id)

    # this could go into tests later on...
    np.testing.assert_allclose(gdir.get_diagnostics()['inversion_glen_a'],
                               a_factor)
    print('succesfully projected: {}'.format(gdir.rgi_id))


@entity_task(log, writes=['gcm_data'])
def process_isimip_data(gdir, filesuffix='', fpath_temp=None,
                        fpath_precip=None, **kwargs):
    """Read, process and store the isimip climate data for this glacier.

    It stores the data in a format that can be used by the OGGM mass balance
    model and in the glacier directory.

    Currently, this function is built for the ISIMIP
    simulations that are on the OGGM servers.

    Parameters
    ----------
    filesuffix : str
        append a suffix to the filename (useful for ensemble experiments).
    fpath_temp : str
        path to the temp file (default: cfg.PATHS['isimip3b_temp_file'])
    fpath_precip : str
        path to the precip file (default: cfg.PATHS['isimip3b_precip_file'])
    **kwargs: any kwarg to be passed to ref:`process_gcm_data`
    """

    # Get the path of GCM temperature & precipitation data
    if fpath_temp is None:
        if not ('cmip5_temp_file' in cfg.PATHS):
            raise ValueError("Need to set cfg.PATHS['isimip3b_temp_file']")
        fpath_temp = cfg.PATHS['isimip3b_temp_file']
    if fpath_precip is None:
        if not ('cmip5_precip_file' in cfg.PATHS):
            raise ValueError("Need to set cfg.PATHS['isimip3b_precip_file']")
        fpath_precip = cfg.PATHS['isimip3b_precip_file']

    # Glacier location
    glon = gdir.cenlon
    glat = gdir.cenlat

    # Read the GCM files
    with xr.open_dataset(fpath_temp, use_cftime=True) as tempds, \
            xr.open_dataset(fpath_precip, use_cftime=True) as precipds:

        # Check longitude conventions
        if tempds.lon.min() >= 0 and glon <= 0:
            glon += 360

        # Take the closest to the glacier
        # Should we consider GCM interpolation?
        temp = tempds.tas.sel(lat=glat, lon=glon, method='nearest')
        precip = precipds.pr.sel(lat=glat, lon=glon, method='nearest')

        # Back to [-180, 180] for OGGM
        temp.lon.values = temp.lon if temp.lon <= 180 else temp.lon - 360
        precip.lon.values = precip.lon if precip.lon <= 180 else precip.lon - 360

        # Convert kg m-2 s-1 to mm mth-1 => 1 kg m-2 = 1 mm !!!
        assert 'kg m-2 s-1' in precip.units, 'Precip units not understood'

        ny, r = divmod(len(temp), 12)
        assert r == 0
        dimo = [cfg.DAYS_IN_MONTH[m - 1] for m in temp['time.month']]
        precip = precip * dimo * (60 * 60 * 24)

    process_gcm_data(gdir, filesuffix=filesuffix, prcp=precip, temp=temp,
                     source=filesuffix, **kwargs)

