import pymc3 as pm
# 	conda install -c conda-forge python-graphviza
import numpy as np
import pandas as pd
import xarray as xr
#import seaborn as sns
#import pickle
#import ast

import matplotlib.pyplot as plt
import matplotlib

# %matplotlib inline
#import statsmodels as stats
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

from MBsandbox.mbmod_daily_oneflowline import \
    MultipleFlowlineMassBalance_TIModel
from oggm.shop.gcm_climate import process_gcm_data
from oggm.core.flowline import flowline_model_run
from oggm.core.massbalance import MultipleFlowlineMassBalance, MassBalanceModel
from oggm.core import climate
from MBsandbox.mbmod_daily_oneflowline import write_climate_file
from MBsandbox.flowline_TIModel import run_from_climate_data_TIModel
from MBsandbox.wip.bayes_calib_geod_direct import bayes_dummy_model_better

import logging

log = logging.getLogger(__name__)


###############################################


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
    
    tasks.compute_downstream_line(gdir)
    tasks.compute_downstream_bedshape(gdir)

    
    
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
        try:
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
                                          precipitation_factor=pf,
                                          melt_f=melt_f,
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
                                                  j),
                                              climate_filename='gcm_data',
                                              ys=2019)
        except:
            # if this occurs, it should save the non-working draw and gdir in a list?s
            print('draw {} did not work for {}: {} {}'.format(j, gdir.rgi_id,
                                                              mb_type,
                                                              grad_type))
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
def process_gcm_data_adv_monthly(gdir, output_filesuffix='', prcp=None,
                                 temp=None,
                                 temp_std=None,
                                 year_range=('1979', '2014'), scale_stddev=True,
                                 time_unit=None, calendar=None, source='',
                                 climate_historical_filesuffix='',
                                 correct=True):
    """ TODO: adapt ...Applies the anomaly method to GCM climate data

    This function can be applied to any GCM data, if it is provided in a
    suitable :py:class:`xarray.DataArray`. See Parameter description for
    format details.

    For CESM-LME a specific function :py:func:`tasks.process_cesm_data` is
    available which does the preprocessing of the data and subsequently calls
    this function.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        where to write the data
    output_filesuffix : str
        append a suffix to the output gcm
        filename (useful for ensemble experiments).
    prcp : :py:class:`xarray.DataArray`
        | monthly total precipitation [mm month-1]
        | Coordinates:
        | lat float64
        | lon float64
        | time: cftime object
    temp : :py:class:`xarray.DataArray`
        | monthly temperature [K]
        | Coordinates:
        | lat float64
        | lon float64
        | time cftime object
    year_range : tuple of str
        the year range for which you want to compute the anomalies. Default
        is `('1961', '1990')`
    scale_stddev : bool
        whether or not to scale the temperature standard deviation as well
    time_unit : str
        The unit conversion for NetCDF files. It must be adapted to the
        length of the time series. The default is to choose
        it ourselves based on the starting year.
        For example: 'days since 0850-01-01 00:00:00'
    calendar : str
        If you use an exotic calendar (e.g. 'noleap')
    source : str
        For metadata: the source of the climate data
    climate_historical_filesuffix : str
        filesuffix of historical climate dataset that should be used to
        apply the anomaly method
    """

    # Standard sanity checks
    months = temp['time.month']
    if months[0] != 1:
        raise ValueError('We expect the files to start in January!')
    if months[-1] < 10:
        raise ValueError('We expect the files to end in December!')

    if (np.abs(temp['lon']) > 180) or (np.abs(prcp['lon']) > 180):
        raise ValueError('We expect the longitude coordinates to be within '
                         '[-180, 180].')

    # from normal years to hydrological years
    sm = cfg.PARAMS['hydro_month_' + gdir.hemisphere]
    # only if sm!=1
    if sm != 1:
        prcp = prcp[sm - 1:sm - 13].load()
        temp = temp[sm - 1:sm - 13].load()
        temp_std = temp_std[sm - 1:sm - 13].load()

    assert len(prcp) // 12 == len(
        prcp) / 12, 'Somehow we didn\'t get full years'
    assert len(temp) // 12 == len(
        temp) / 12, 'Somehow we didn\'t get full years'

    assert np.all(temp_std>=0)
    assert np.all(prcp>=0)

    # Get historical_climate to apply the anomaly to
    fpath = gdir.get_filepath('climate_historical',
                              filesuffix=climate_historical_filesuffix)
    ds_climobs = xr.open_dataset(fpath, use_cftime=True)

    # Add climobs (or the defined climate from above...) clim
    dsclimobs = ds_climobs.sel(time=slice(*year_range))

    # compute monthly anomalies
    # of temp
    if correct:
        if scale_stddev:
            # This is a bit more arithmetic
            ts_tmp_sel = temp.sel(time=slice(*year_range))
            ts_tmp_std = ts_tmp_sel.groupby('time.month').std(dim='time')
            # observed/gcm
            std_fac = dsclimobs.temp.groupby('time.month').std(
                dim='time') / ts_tmp_std
            assert np.all(std_fac == std_fac.roll(month=13 - sm,
                                                  roll_coords=True))
            # if sm =1, this just changes nothing as it should
            if sm != 1:
                # just to avoid useless roll (same as in process_gcm_data now)
                std_fac = std_fac.roll(month=13 - sm, roll_coords=True)
            std_fac = np.tile(std_fac.data, len(temp) // 12)
            # We need an even number of years for this to work
            if ((len(ts_tmp_sel) // 12) % 2) == 1:
                raise InvalidParamsError('We need an even number of years '
                                         'for this to work')
            win_size = len(ts_tmp_sel) + 1

            def roll_func(x, axis=None):
                x = x[:, ::12]
                n = len(x[0, :]) // 2
                # print(n)
                xm = np.nanmean(x, axis=axis)
                return xm + (x[:, n] - xm) * std_fac

            # a little bit slower but more comprehensible
            temp_test = temp.copy()
            aoy_sel = len(ts_tmp_sel.groupby('time.year').mean())
            for m in np.arange(0, 12):
                # at the first / last years less observation over which mean is
                # made but I guess this does not matter (otherwise it would be nan)
                x = temp[temp.time.dt.month == m + 1].copy()
                xm = x.rolling(time=aoy_sel, center=True, min_periods=1).mean()
                # same as in Zekollari 2019, eq. 2, but mean over aoy_sel years
                # instead
                temp_test[temp.time.dt.month == m + 1] = xm + (x - xm) * \
                                                         std_fac[m]

            temp = temp.rolling(time=win_size, center=True,
                                min_periods=1).reduce(roll_func)

            np.testing.assert_allclose(temp, temp_test, rtol=1e-2)  # -3
            ### do the same for daily temp std
            ts_tmpdstd_sel = temp_std.sel(time=slice(*year_range))
            ts_tmpdstd_std = ts_tmpdstd_sel.groupby('time.month').std(
                dim='time')
            tmpdstd_std_fac = dsclimobs.temp_std.groupby('time.month').std(
                dim='time') / ts_tmpdstd_std
            tmpdstd_std_fac = tmpdstd_std_fac.roll(month=13 - sm,
                                                   roll_coords=True)
            tmpdstd_std_fac = np.tile(tmpdstd_std_fac.data, len(temp_std) // 12)
            # We need an even number of years for this to work
            if ((len(ts_tmp_sel) // 12) % 2) == 1:
                raise InvalidParamsError('We need an even number of years '
                                         'for this to work')
            win_size = len(ts_tmpdstd_sel) + 1

            def roll_func_std(x, axis=None):
                x = x[:, ::12]
                n = len(x[0, :]) // 2
                xm = np.nanmean(x, axis=axis)
                return xm + (x[:, n] - xm) * tmpdstd_std_fac

            temp_std = temp_std.rolling(time=win_size, center=True,
                                        min_periods=1).reduce(roll_func_std)
            # this can result in very rare cases too negative temp_std values
            # Clip them
            #TODO: think about a better solution!!!
            #TODO: print how often this occurs
            #TODO: think about whether this should be done here and also later
            #or just once at the end
            #assert len(temp_std.where(temp_std < 0, drop=True)) < 10
            #assert np.min(temp_std) > -1
            #temp_std.values = utils.clip_min(temp_std.values, 1e-3)


        ts_tmp_sel = temp.sel(time=slice(*year_range))
        assert len(ts_tmp_sel) == len(dsclimobs.time)
        ts_tmp_avg = ts_tmp_sel.groupby('time.month').mean(dim='time')
        ts_tmp = temp.groupby('time.month') - ts_tmp_avg
        # of precip -- scaled anomalies
        ts_pre_avg = prcp.sel(time=slice(*year_range))
        ts_pre_avg = ts_pre_avg.groupby('time.month').mean(dim='time')
        ts_pre_ano = prcp.groupby('time.month') - ts_pre_avg
        # scaled anomalies is the default. Standard anomalies above
        # are used later for where ts_pre_avg == 0
        ts_pre = prcp.groupby('time.month') / ts_pre_avg

        # daily temp std
        ts_dstdtmp_avg = temp_std.sel(time=slice(*year_range))
        ts_dstdtmp_avg = ts_dstdtmp_avg.groupby('time.month').mean(dim='time')
        ts_dstdtmp = temp_std.groupby('time.month') - ts_dstdtmp_avg

        # for temp
        loc_tmp = dsclimobs.temp.groupby('time.month').mean()
        ts_tmp = ts_tmp.groupby('time.month') + loc_tmp

        # for temp daily std
        loc_dstdtmp = dsclimobs.temp_std.groupby('time.month').mean()
        ts_dstdtmp = ts_dstdtmp.groupby('time.month') + loc_dstdtmp

        # for prcp
        loc_pre = dsclimobs.prcp.groupby('time.month').mean()
        # scaled anomalies
        ts_pre = ts_pre.groupby('time.month') * loc_pre
        # standard anomalies
        ts_pre_ano = ts_pre_ano.groupby('time.month') + loc_pre
        # Correct infinite values with standard anomalies
        ts_pre.values = np.where(np.isfinite(ts_pre.values),
                                 ts_pre.values,
                                 ts_pre_ano.values)
        # The previous step might create negative values (unlikely). Clip them
        ts_pre.values = utils.clip_min(ts_pre.values, 0)

        assert np.all(np.isfinite(ts_pre.values))
        assert np.all(np.isfinite(ts_tmp.values))
        assert np.all(np.isfinite(ts_dstdtmp.values))
        #if it occurs not too negative
        amount_neg_std = len(ts_dstdtmp.where(ts_dstdtmp<0, drop=True))
        if amount_neg_std >0:
            print('time points with negative temp std that are clipped'
                  'to 1e-3: {}'.format(amount_neg_std))
        # CHECK that this occurs only very rarely (<10 times )
        assert amount_neg_std < 10
        assert np.min(ts_dstdtmp) > -0.5

        ts_dstdtmp.values = utils.clip_min(ts_dstdtmp.values, 1e-3)

        assert np.all(ts_dstdtmp.values >= 0)
        # there are some glaciers where apparently ts_dstdtmp gets negative (I guees I need to do the same as done for the prcp. values! )
    else:
        # do no correction at all (!!! only for testing)
        ts_tmp = temp - 273.15
        ts_pre = prcp
        ts_dstdtmp = temp_std
        output_filesuffix = output_filesuffix + '_no_correction'
        source = output_filesuffix + '_no_correction'

    # for gradient
    # try:
    hist_gradient = dsclimobs.gradient.groupby('time.month').mean().values
    # use the same gradient for every year
    # may be this could be done differently to save data...
    # print(hist_gradient)
    # print(len(temp.time.values)/12)
    gradient = np.tile(hist_gradient, int(round(len(temp.time.values) / 12)))
    # except:
    # gradient = None
    # elif temporal_resol =='daily':
    #    temp_std = None
    # print(len(temp.time.values), len(ts_pre.values), len(gradient), len(temp_std))

    # time_unit = temp.time.units
    # calendar = temp.time.calendar
    # print(time_unit, calendar)
    write_climate_file(gdir, temp.time.values,
                       ts_pre.values, ts_tmp.values,
                       float(dsclimobs.ref_hgt),
                       prcp.lon.values, prcp.lat.values,
                       time_unit=time_unit,
                       calendar=calendar,
                       file_name='gcm_data',
                       source=source + '_historical{}'.format(
                           climate_historical_filesuffix),
                       filesuffix=output_filesuffix,
                       gradient=gradient,
                       temp_std=ts_dstdtmp.values,
                       temporal_resol='monthly')

    ds_climobs.close()


@entity_task(log, writes=['gcm_data'])
def process_gcm_data_adv_daily(gdir, output_filesuffix='', prcp=None,
                               temp=None,
                               year_range=('1979', '2014'),
                               scale_stddev=True,
                               time_unit=None, calendar=None, source='',
                               climate_historical_filesuffix='',
                               correct=True):
    """ Applies the anomaly method to daily GCM climate data

    This function can be applied to any GCM data, if it is provided in a
    suitable :py:class:`xarray.DataArray` daily format. See Parameter description for
    format details.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        where to write the data
    output_filesuffix : str
        append a suffix to the output gcm
        filename (useful for ensemble experiments).
    prcp : :py:class:`xarray.DataArray`
        | daily total precipitation [mm day-1]
        | Coordinates:
        | lat float64
        | lon float64
        | time: cftime object
    temp : :py:class:`xarray.DataArray`
        | daily mean temperature [K]
        | Coordinates:
        | lat float64
        | lon float64
        | time cftime object
    year_range : tuple of str
        the year range for which you want to compute the anomalies. Default
        is `('1979', '2014')` because this is the common period between historical
        with WFDE5 and because ssp-dependence starts in 2015
    scale_stddev : bool
        whether or not to scale the temperature standard deviation as well
    time_unit : str
        The unit conversion for NetCDF files. It must be adapted to the
        length of the time series. The default is to choose
        it ourselves based on the starting year.
        For example: 'days since 0850-01-01 00:00:00'
    calendar : str
        If you use an exotic calendar (e.g. 'noleap')
    source : str
        For metadata: the source of the climate data
    climate_historical_filesuffix : str
        filesuffix of historical climate dataset that should be used to
        apply the anomaly method
    """

    # Standard sanity checks that this is really daily
    months = temp['time.month']
    if months[0] != 1:
        raise ValueError('We expect the files to start in January!')
    days = temp['time.dayofyear']
    if days[0] != 1:
        raise ValueError('We expect the files to start on first day of year!')
    if days[-1] < 365:
        raise ValueError('We expect the files to end on last day of year!')
    if months[-1] < 10:
        raise ValueError('We expect the files to end in December!')

    if (np.abs(temp['lon']) > 180) or (np.abs(prcp['lon']) > 180):
        raise ValueError('We expect the longitude coordinates to be within '
                         '[-180, 180].')

    # from normal years to hydrological years
    sm = cfg.PARAMS['hydro_month_' + gdir.hemisphere]
    # only if sm!=1
    if sm != 1:
        raise NotImplementedError('this works at the moment only for '
                                  'hydro_month=1, but the function should be'
                                  'easy to adapt as done in '
                                  'process_gcm_data_adv_monthly')
    # check that len(prcp) and len(temp) /365 or 366???
    # find a way to check if we have full years !!!
    # assert len(prcp) // 12 == len(
    #    prcp) / 12, 'Somehow we didn\'t get full years'
    # assert len(temp) // 12 == len(
    #    temp) / 12, 'Somehow we didn\'t get full years'

    # Get historical_climate to apply the anomaly to
    fpath = gdir.get_filepath('climate_historical',
                              filesuffix=climate_historical_filesuffix)
    ds_climobs = xr.open_dataset(fpath, use_cftime=True)

    # select the time range where from where we want to apply the anomaly correction
    # dsclimobs: historical climate dataset
    dsclimobs = ds_climobs.sel(time=slice(*year_range))

    # compute daily anomalies
    # of temp
    if correct:
        if scale_stddev:
            # scale the temperature standard deviation
            # that means historic gcm should have equal std as the climate dataset

            # select the time range where from where we want to apply the anomaly correction
            ts_tmp_sel = temp.sel(time=slice(*year_range))
            # get annual cycle standard deviation of temperature
            # from GCM over same time period
            ts_tmp_std = ts_tmp_sel.groupby('time.dayofyear').std(dim='time')

            # get the std factor (observed_reanalysis / gcm ) for same time period
            # corresponds to phi_daily in Zekollari (2019, eq. 2)
            std_fac = dsclimobs.temp.groupby('time.dayofyear').std(
                dim='time') / ts_tmp_std
            # this needs to be adapted if sm!=1

            # Trial to incorporate OLD METHOD of Fabien process_gcm_data to daily
            # but actually I decided that it gets too complicated ...
            # amount of years
            # aoy = len(temp.groupby('time.year').mean())
            # We need an even number of years for this to work
            # if ((len(ts_tmp_sel.groupby('time.year').mean())) % 2) == 1:
            #    raise InvalidParamsError('We need an even number of years '
            #                             'for this to work')
            # win_size = len(ts_tmp_sel) + 1
            # for y in temp.groupby('time.year').time.mean():
            #    xm + (x[:, n])
            # def roll_func(x, axis=None):
            #    # x = x[:, ::12]# temp for each month
            #    x = x[:, ::365]  # temp for each day
            #    n = len(x[0, :]) // 2 # either 0 or 1
            #    xm = np.nanmean(x, axis=axis)
            #    return xm + (x[:, n] - xm) * std_fac
            # xm = temp.groupby('time.dayofyear').mean()
            # x = temp.where(temp.time.dt.dayofyear == 1)
            # temp = temp.rolling(time=win_size, center=True,
            #                    min_periods=1).reduce(roll_func)

            # a little bit slower with the loop but more comprehensible
            # amount of years for the selected time period
            aoy_sel = len(ts_tmp_sel.groupby('time.year').mean())
            for doy in np.arange(0, 366):
                # repeat this for each day of the year
                # T_d,y,corrected = mean(T$_{d,36}$) + (T_${d,y}$ - mean (T$_{d,36}$)) *std_fac_d
                # we don't mind about lap years ...
                # select all temp. values for this day (+1 because we start doy with 0)
                # corresponds to T_${d,y}$:
                x = temp[temp.time.dt.dayofyear == doy + 1].copy()
                # do the moving average with the amount of years that were selected
                # corresponds to mean(T$_{d,36}$)
                xm = x.rolling(time=aoy_sel, center=True, min_periods=1).mean()
                # due to rolling: at first / last years less observation over which mean is
                # made but I guess this does not matter (otherwise it would be nan)

                # next line is exactly the same as in Zekollari 2019, eq. 2,
                # but mean over aoy_sel years instead
                # overwrite the temperature value (get T_d,y,corrected)
                temp[temp.time.dt.dayofyear == doy + 1] = xm + (x - xm) * std_fac[doy]

        # here we additionally correct to match the mean values
        # first get scaled anomalies of temperature
        ts_tmp_sel = temp.sel(time=slice(*year_range))
        assert len(temp.sel(time=slice(*year_range)).time) == len(
            dsclimobs.time)
        ts_tmp_avg = ts_tmp_sel.groupby('time.dayofyear').mean(dim='time')
        ts_tmp = temp.groupby('time.dayofyear') - ts_tmp_avg
        # then of precip -- scaled anomalies
        # has to be corrected differently because prcp has to be above or equal zero!
        # first compute the standard anomaly compared to the average time period
        ts_pre_avg = prcp.sel(time=slice(*year_range))
        ts_pre_avg = ts_pre_avg.groupby('time.dayofyear').mean(dim='time')
        ts_pre_ano = prcp.groupby('time.dayofyear') - ts_pre_avg
        # scaled anomalies is the default. Standard anomalies above
        # are used later for where ts_pre_avg == 0
        ts_pre = prcp.groupby('time.dayofyear') / ts_pre_avg

        # here the actual anomaly correction to match the mean begins:
        # for temperature
        # loc_tmp: mean temperature for each day of year of historical climate period
        # (from the "observed"/reanalysis climate [not GCM])
        loc_tmp = dsclimobs.temp.groupby('time.dayofyear').mean()
        # correct the scaled temperature anomalies
        ts_tmp = ts_tmp.groupby('time.dayofyear') + loc_tmp

        # for prcp
        loc_pre = dsclimobs.prcp.groupby('time.dayofyear').mean()
        # scaled anomalies
        # loc_tmp: what is multiplied as factor to match the mean
        ts_pre = ts_pre.groupby('time.dayofyear') * loc_pre
        # standard anomalies
        ts_pre_ano = ts_pre_ano.groupby('time.dayofyear') + loc_pre
        # Correct infinite values with standard anomalies
        ts_pre.values = np.where(np.isfinite(ts_pre.values),
                                 ts_pre.values,
                                 ts_pre_ano.values)
        # The previous step might create negative values (unlikely). Clip them
        ts_pre.values = utils.clip_min(ts_pre.values, 0)

        # check again that the correction went well
        assert np.all(np.isfinite(ts_pre.values))
        assert np.all(np.isfinite(ts_tmp.values))
    else:
        # do no correction at all (!!! only for testing)
        ts_tmp = temp - 273.15
        ts_pre = prcp
        output_filesuffix = output_filesuffix + '_no_correction'
        source = output_filesuffix + '_no_correction'

    # for the temperature gradient
    # no correction: we assume that the temperature gradient
    # does not change over the years
    # we use the same mean annual cyle from the climate dataset for the GCMs
    try:
        # amount of days for each year
        aod = temp.groupby('time.year').count()
        # aod[aod == 365].year
        hist_gradient = dsclimobs.gradient.groupby(
            'time.dayofyear').mean().values
        # use the same gradient for every year
        # may be this could be done differently to save data...
        # amount of years: len(temp.time.values) / 365
        gradient_ls = []
        for y in aod.year:
            if aod.sel(year=y) == 365:
                # more correct would be to delete Feb 29th, however,
                # we prefer to be consistent with the climate dataset
                # (e.g. WFDE5_CRU)
                gradient_ls.append(np.delete(hist_gradient, 366 - 1))
            elif aod.sel(year=y) == 366:
                gradient_ls.append(hist_gradient)
            else:
                raise InvalidParamsError('sth went wrong')
        gradient = np.concatenate(gradient_ls)

        assert len(temp) == len(gradient), 'problem with length of gradient'
        # gradient = np.repeat(hist_gradient,
        #                     len(dsclimobs.groupby('time.year').mean().year))
    except:
        gradient = None

    # write the data into a netCDF file
    write_climate_file(gdir, temp.time.values,
                       ts_pre.values, ts_tmp.values,
                       float(dsclimobs.ref_hgt),
                       prcp.lon.values, prcp.lat.values,
                       time_unit=time_unit,
                       calendar=calendar,
                       file_name='gcm_data',
                       source=source + '_historical{}'.format(
                           climate_historical_filesuffix),
                       filesuffix=output_filesuffix,
                       gradient=gradient,
                       temporal_resol='daily')

    ds_climobs.close()


@entity_task(log, writes=['gcm_data'])
def process_isimip_data(gdir, output_filesuffix='', fpath_temp=None,
                        fpath_temp_std=None,
                        fpath_precip=None,
                        fpath_temp_h=None,
                        fpath_temp_std_h=None,
                        fpath_precip_h=None,
                        climate_historical_filesuffix='',
                        ensemble='mri-esm2-0_r1i1p1f1',
                        # from temperature tie series the "median" ensemble
                        ssp='ssp126', flat=True,
                        temporal_resol='monthly',
                        cluster=False,
                        year_range=('1979', '2014'),
                        correct=True,
                        **kwargs):
    """Read, process and store the isimip climate data for this glacier.

    It stores the data in a format that can be used by the OGGM mass balance
    model and in the glacier directory.

    Currently, this function is built for the ISIMIP3b
    simulations that are on the OGGM servers.

    Parameters
    ----------
    filesuffix : str
        append a suffix to the filename (useful for ensemble experiments).
    fpath_temp : str
        path to the temp file (default: cfg.PATHS['isimip3b_temp_file'])
    fpath_precip : str
        path to the precip file (default: cfg.PATHS['isimip3b_precip_file'])
    climate_historical_filesuffix : str
        filesuffix of historical climate dataset that should be used to
        apply the anomaly method
    correct : bool
        whether the bias correction is applied (default is True) or not. As
        we use already internally bias-corrected GCMs, we can also set this
        to correct=False!
    **kwargs: any kwarg to be passed to ref:`process_gcm_data`
    """

    if output_filesuffix == '':
        # recognize the gcm climate file for later
        if temporal_resol == 'monthly':
            output_filesuffix = '_monthly_ISIMIP3b_{}_{}'.format(ensemble, ssp)
        elif temporal_resol == 'daily':
            output_filesuffix = '_daily_ISIMIP3b_{}_{}'.format(ensemble, ssp)

    if temporal_resol == 'monthly':
        assert 'monthly' in climate_historical_filesuffix
    elif temporal_resol == 'daily':
        assert 'daily' in climate_historical_filesuffix
    # Get the path of GCM temperature & precipitation data
    # if fpath_temp is None:
    #    if not ('cmip5_temp_file' in cfg.PATHS):
    #       raise ValueError("Need to set cfg.PATHS['isimip3b_temp_file']")
    #    fpath_temp = cfg.PATHS['isimip3b_temp_file']
    # if fpath_precip is None:
    #    if not ('cmip5_precip_file' in cfg.PATHS):
    #        raise ValueError("Need to set cfg.PATHS['isimip3b_precip_file']")
    #    fpath_precip = cfg.PATHS['isimip3b_precip_file']

    # Glacier location
    glon = gdir.cenlon
    glat = gdir.cenlat
    if None in [fpath_temp, fpath_temp_h, fpath_temp_std, fpath_temp_std_h,
                fpath_precip, fpath_precip_h]:
        if temporal_resol == 'monthly':
            if cluster:
                path = '/home/www/lschuster/isimip3b_flat/flat/monthly/'
            else:
                path = 'https://cluster.klima.uni-bremen.de/~lschuster/isimip3b_flat/flat/monthly/'
            add = '_global_monthly_flat_glaciers.nc'
        elif temporal_resol == 'daily':
            if cluster:
                path = '/home/www/lschuster/isimip3b_flat/flat/daily/'
            else:
                path = 'https://cluster.klima.uni-bremen.de/~lschuster/isimip3b_flat/flat/daily/'
            add = '_global_daily_flat_glaciers.nc'

        fpath_spec = path + '{}_w5e5_'.format(ensemble) + '{ssp}_{var}' + add
        fpath_temp = fpath_spec.format(var='tasAdjust', ssp=ssp)
        fpath_temp_h = fpath_spec.format(var='tasAdjust', ssp='historical')

        if temporal_resol == 'monthly':
            fpath_temp_std = fpath_spec.format(var='tasAdjust_std', ssp=ssp)
            fpath_temp_std_h = fpath_spec.format(var='tasAdjust_std',
                                                 ssp='historical')

        fpath_precip = fpath_spec.format(var='prAdjust', ssp=ssp)
        fpath_precip_h = fpath_spec.format(var='prAdjust', ssp='historical')
        if not cluster:
            fpath_temp = utils.file_downloader(fpath_temp)
            fpath_temp_h = utils.file_downloader(fpath_temp_h)
            if temporal_resol == 'monthly':
                fpath_temp_std = utils.file_downloader(fpath_temp_std)
                fpath_temp_std_h = utils.file_downloader(fpath_temp_std_h)

            fpath_precip = utils.file_downloader(fpath_precip)
            fpath_precip_h = utils.file_downloader(fpath_precip_h)
    # # need to aggregate first both gcm types !!!!
    # if flat:
    #     add = '_global_monthly_flat_glaciers.nc'
    #     fpath_temp_gcm = fpath_temp+ '{}_w5e5_{}_tasAdjust_{}'.format(ensemble, ssp, add)
    #     fpath_temp_historical = fpath_temp+ '{}_w5e5_historical_tasAdjust_{}'.format(ensemble, add)
    #
    #     fpath_temp_std_gcm = fpath_temp_std+ '{}_w5e5_{}_tasAdjust_std_{}'.format(ensemble, ssp, add)
    #     fpath_temp_std_historical = fpath_temp_std+ '{}_w5e5_historical_tasAdjust_std_{}'.format(ensemble, add)
    #
    #     fpath_prcp_gcm = fpath_precip+ '{}_w5e5_{}_prAdjust_{}'.format(ensemble, ssp, add)
    #     fpath_prcp_historical = fpath_precip+ '{}_w5e5_historical_prAdjust_{}'.format(ensemble, add)
    # else:
    #     fpath_temp_gcm = fpath_temp+ '{}_w5e5_{}_tasAdjust_global_monthly_2015_2100.nc'.format(ensemble, ssp)
    #     fpath_temp_historical = fpath_temp+ '{}_w5e5_historical_tasAdjust_global_monthly_1850_2014.nc'.format(ensemble)
    #
    #     fpath_temp_std_gcm = fpath_temp_std+ '{}_w5e5_{}_tasAdjust_std_global_monthly_2015_2100.nc'.format(ensemble, ssp)
    #     fpath_temp_std_historical = fpath_temp_std+ '{}_w5e5_historical_tasAdjust_std_global_monthly_1850_2014.nc'.format(ensemble)
    #
    #     fpath_prcp_gcm = fpath_precip+ '{}_w5e5_{}_prAdjust_global_monthly_2015_2100.nc'.format(ensemble, ssp)
    #     fpath_prcp_historical = fpath_precip+ '{}_w5e5_historical_prAdjust_global_monthly_1850_2014.nc'.format(ensemble)

    # Read the GCM files
    with xr.open_dataset(fpath_temp_h, use_cftime=True) as tempds_hist, \
            xr.open_dataset(fpath_temp, use_cftime=True) as tempds_gcm:

        # Check longitude conventions
        if tempds_gcm.longitude.min() >= 0 and glon <= 0:
            glon += 360
        assert tempds_gcm.attrs['experiment'] == ssp
        # Take the closest to the glacier
        # Should we consider GCM interpolation?
        # try:
        # computing all the distances and choose the nearest gridpoint
        c = (tempds_gcm.longitude - glon) ** 2 + (
                    tempds_gcm.latitude - glat) ** 2
        # first select gridpoint, then merge, should be faster!!!
        temp_a_gcm = tempds_gcm.isel(points=c.argmin())
        temp_a_hist = tempds_hist.isel(points=c.argmin())
        # merge historical with gcm together
        # TODO: change to drop_conflicts when xarray version v0.17.0 can
        # be used with salem
        temp_a = xr.merge([temp_a_gcm, temp_a_hist],
                          combine_attrs='override')
        temp = temp_a.tasAdjust
        temp['lon'] = temp_a.longitude
        temp['lat'] = temp_a.latitude
        # except ValueError:
        #    temp = tempds.tasAdjust.sel(latitude=glat, longitude=glon,
        #                                method='nearest')
        temp.lon.values = temp.lon if temp.lon <= 180 else temp.lon - 360
        # tempds.close()

    with xr.open_dataset(fpath_precip_h, use_cftime=True) as precipds_hist, \
            xr.open_dataset(fpath_precip, use_cftime=True) as precipds_gcm:

        if temporal_resol == 'monthly':
            tempds_std_hist = xr.open_dataset(fpath_temp_std_h,
                                              use_cftime=True)
            tempds_std_gcm = xr.open_dataset(fpath_temp_std,
                                             use_cftime=True)
            tempds_std = xr.merge([tempds_std_gcm, tempds_std_hist],
                                  combine_attrs='override')
            try:
                c = (tempds_std.longitude - glon) ** 2 + \
                    (tempds_std.latitude - glat) ** 2
                temp_std_a = tempds_std.isel(points=c.argmin())
                temp_std = temp_std_a.tasAdjust_std
                temp_std['lon'] = temp_std_a.longitude
                temp_std['lat'] = temp_std_a.latitude
            except ValueError:
                temp_std = tempds_std.tasAdjust_std.sel(latitude=glat,
                                                        longitude=glon,
                                                        method='nearest')
            temp_std.lon.values = temp_std.lon if temp_std.lon <= 180 \
                else temp.lon - 360

        # precipds = xr.merge([precipds_gcm, precipds_hist],
        #                    combine_attrs='override')
        # try:
        c = (precipds_gcm.longitude - glon) ** 2 + (
                    precipds_gcm.latitude - glat) ** 2
        precip_a_gcm = precipds_gcm.isel(points=c.argmin())
        precip_a_hist = precipds_hist.isel(points=c.argmin())
        precip_a = xr.merge([precip_a_gcm, precip_a_hist],
                            combine_attrs='override')

        if temporal_resol == 'monthly':
            precip = precip_a.prAdjust
        elif temporal_resol == 'daily':
            precip = precip_a.tp
        precip['lon'] = precip_a.longitude
        precip['lat'] = precip_a.latitude
        # except ValueError:
        #    precip = precipds.prAdjust.sel(latitude=glat,
        #                                    longitude=glon, method='nearest')
        # precipds.close()
        # Back to [-180, 180] for OGGM
        precip.lon.values = precip.lon if precip.lon <= 180 \
            else precip.lon - 360

        # Convert kg m-2 s-1 to mm mth-1 => 1 kg m-2 = 1 mm !!!

        if temporal_resol == 'monthly':
            assert 'kg m-2 s-1' in precip.units, \
                'Precip units not understood'
            ny, r = divmod(len(temp), 12)
            assert r == 0
            dimo = [cfg.DAYS_IN_MONTH[m - 1] for m in temp['time.month']]
            precip = precip * dimo * (60 * 60 * 24)
            tempds_std_gcm.close()
            tempds_std_hist.close()

        elif temporal_resol == 'daily':
            # we want here to have daily precipitation: mm/day
            # check if it is really daily: amount of days should be either
            # 365 or 366
            aod = temp.groupby('time.year').count()
            assert np.all(aod[aod != 365] == 366)
            # for daily: precip is already converted to mm/day during flattening
            # check this
            assert 'mm/day' in precip.units
            # precip = precip * (60*60*24)
        # print(len(precip)

    if temporal_resol == 'monthly':
        process_gcm_data_adv_monthly(gdir,
                                     output_filesuffix=output_filesuffix,
                                     prcp=precip, temp=temp,
                                     temp_std=temp_std,
                                     source=output_filesuffix,
                                     year_range=year_range,
                                     climate_historical_filesuffix=climate_historical_filesuffix,
                                     correct=correct,
                                     **kwargs)
    elif temporal_resol == 'daily':
        process_gcm_data_adv_daily(gdir,
                                   output_filesuffix=output_filesuffix,
                                   prcp=precip, temp=temp,
                                   source=output_filesuffix,
                                   year_range=year_range,
                                   climate_historical_filesuffix=climate_historical_filesuffix,
                                   correct=correct,
                                   **kwargs)

@entity_task(log, writes=['gcm_data'])
def process_isimip_data_no_corr(gdir, output_filesuffix='', fpath_temp=None,
                        fpath_temp_std=None,
                        fpath_precip=None,
                        climate_historical_filesuffix='',
                        ensemble='mri-esm2-0_r1i1p1f1',
                        # from temperature tie series the "median" ensemble
                        ssp='ssp126', flat=True,
                        temporal_resol='monthly',
                        cluster=False,
                        year_range=('1979', '2014'), correct = False,
                        **kwargs):
    """Read, process and store the isimip climate data for this glacier.

    It stores the data in a format that can be used by the OGGM mass balance
    model and in the glacier directory.

    Currently, this function is built for the ISIMIP3b
    simulations that are on the OGGM servers.

    Parameters
    ----------
    filesuffix : str
        append a suffix to the filename (useful for ensemble experiments).
    fpath_temp : str
        path to the temp file (default: cfg.PATHS['isimip3b_temp_file'])
    fpath_precip : str
        path to the precip file (default: cfg.PATHS['isimip3b_precip_file'])
    climate_historical_filesuffix : str
        filesuffix of historical climate dataset that should be used to
        apply the anomaly method
    **kwargs: any kwarg to be passed to ref:`process_gcm_data`
    """
    if correct != False:
        raise InvalidWorkflowError('for processing isimip data with correction'
                                   'please use process_isimip_data function')

    if output_filesuffix == '':
        # recognize the gcm climate file for later
        if temporal_resol == 'monthly':
            output_filesuffix = '_monthly_ISIMIP3b_{}_{}'.format(ensemble, ssp)
        elif temporal_resol == 'daily':
            output_filesuffix = '_daily_ISIMIP3b_{}_{}'.format(ensemble, ssp)

    if temporal_resol == 'monthly':
        assert 'monthly' in climate_historical_filesuffix
    elif temporal_resol == 'daily':
        assert 'daily' in climate_historical_filesuffix
    # Get the path of GCM temperature & precipitation data
    # if fpath_temp is None:
    #    if not ('cmip5_temp_file' in cfg.PATHS):
    #       raise ValueError("Need to set cfg.PATHS['isimip3b_temp_file']")
    #    fpath_temp = cfg.PATHS['isimip3b_temp_file']
    # if fpath_precip is None:
    #    if not ('cmip5_precip_file' in cfg.PATHS):
    #        raise ValueError("Need to set cfg.PATHS['isimip3b_precip_file']")
    #    fpath_precip = cfg.PATHS['isimip3b_precip_file']

    # Glacier location
    glon = gdir.cenlon
    glat = gdir.cenlat
    if None in [fpath_temp, fpath_temp_std,
                fpath_precip]:
        if temporal_resol == 'monthly':
            if cluster:
                path = '/home/www/lschuster/isimip3b_flat/flat/monthly/'
            else:
                path = 'https://cluster.klima.uni-bremen.de/~lschuster/isimip3b_flat/flat/monthly/'
            add = '_global_monthly_flat_glaciers.nc'
        elif temporal_resol == 'daily':
            if climate_historical_filesuffix == "_daily_W5E5_dw":
                path = 'https://cluster.klima.uni-bremen.de/~shanus/ISIMIP3b/flattened/daily/'
                add = '_global_daily_flat_glaciers_2015_2100.nc'
            else:
                if cluster:
                    path = '/home/www/lschuster/isimip3b_flat/flat/daily/'
                else:
                    path = 'https://cluster.klima.uni-bremen.de/~lschuster/isimip3b_flat/flat/daily/'
                add = '_global_daily_flat_glaciers.nc'

        fpath_spec = path + '{}_w5e5_'.format(ensemble) + '{ssp}_{var}' + add
        fpath_temp = fpath_spec.format(var='tasAdjust', ssp=ssp)

        if temporal_resol == 'monthly':
            fpath_temp_std = fpath_spec.format(var='tasAdjust_std', ssp=ssp)

        fpath_precip = fpath_spec.format(var='prAdjust', ssp=ssp)
        if not cluster:
            fpath_temp = utils.file_downloader(fpath_temp)
            if temporal_resol == 'monthly':
                fpath_temp_std = utils.file_downloader(fpath_temp_std)

            fpath_precip = utils.file_downloader(fpath_precip)

    with xr.open_dataset(fpath_temp, use_cftime=True) as tempds_gcm:

        # Check longitude conventions
        if tempds_gcm.longitude.min() >= 0 and glon <= 0:
            glon += 360
        #assert tempds_gcm.attrs['experiment'] == ssp
        # Take the closest to the glacier
        # Should we consider GCM interpolation?
        # try:
        # computing all the distances and choose the nearest gridpoint
        c = (tempds_gcm.longitude - glon) ** 2 + (
                    tempds_gcm.latitude - glat) ** 2
        # first select gridpoint, then merge, should be faster!!!
        temp_a = tempds_gcm.isel(points=c.argmin())
        # merge historical with gcm together
        # TODO: change to drop_conflicts when xarray version v0.17.0 can
        # be used with salem
        # it should take the first variable of the dataset
        temp = temp_a[list(temp_a.keys())[0]]
        temp['lon'] = temp_a.longitude
        temp['lat'] = temp_a.latitude
        # except ValueError:
        #    temp = tempds.tasAdjust.sel(latitude=glat, longitude=glon,
        #                                method='nearest')
        temp.lon.values = temp.lon if temp.lon <= 180 else temp.lon - 360
        # tempds.close()

    with xr.open_dataset(fpath_precip, use_cftime=True) as precipds_gcm:

        if temporal_resol == 'monthly':
            tempds_std = xr.open_dataset(fpath_temp_std,
                                             use_cftime=True)
            try:
                c = (tempds_std.longitude - glon) ** 2 + \
                    (tempds_std.latitude - glat) ** 2
                temp_std_a = tempds_std.isel(points=c.argmin())
                temp_std = temp_std_a.tasAdjust_std
                temp_std['lon'] = temp_std_a.longitude
                temp_std['lat'] = temp_std_a.latitude
            except ValueError:
                temp_std = tempds_std.tasAdjust_std.sel(latitude=glat,
                                                        longitude=glon,
                                                        method='nearest')
            temp_std.lon.values = temp_std.lon if temp_std.lon <= 180 \
                else temp.lon - 360

        # precipds = xr.merge([precipds_gcm, precipds_hist],
        #                    combine_attrs='override')
        # try:
        c = (precipds_gcm.longitude - glon) ** 2 + (
                    precipds_gcm.latitude - glat) ** 2
        precip_a = precipds_gcm.isel(points=c.argmin())

        # it should just take the first variable in the given data set, so it can be used with different datasets
        precip = precip_a[list(precip_a.keys())[0]]
        precip['lon'] = precip_a.longitude
        precip['lat'] = precip_a.latitude
        # except ValueError:
        #    precip = precipds.prAdjust.sel(latitude=glat,
        #                                    longitude=glon, method='nearest')
        # precipds.close()
        # Back to [-180, 180] for OGGM
        precip.lon.values = precip.lon if precip.lon <= 180 \
            else precip.lon - 360

        # Convert kg m-2 s-1 to mm mth-1 => 1 kg m-2 = 1 mm !!!

        if temporal_resol == 'monthly':
            assert 'kg m-2 s-1' in precip.units, \
                'Precip units not understood'
            ny, r = divmod(len(temp), 12)
            assert r == 0
            dimo = [cfg.DAYS_IN_MONTH[m - 1] for m in temp['time.month']]
            precip = precip * dimo * (60 * 60 * 24)
            tempds_std.close()

        elif temporal_resol == 'daily':
            # we want here to have daily precipitation: mm/day
            # check if it is really daily: amount of days should be either
            # 365 or 366
            aod = temp.groupby('time.year').count()
            assert np.all(aod[aod != 365] == 366)
            # for daily: precip is already converted to mm/day during flattening
            # check this
            # assert 'mm/day' in precip.units
            if 'kg m-2 s-1' in precip.units:
                precip = precip * (60*60*24)
            else:
                assert 'mm/day' in precip.units

    if temporal_resol == 'monthly':
        process_gcm_data_adv_monthly(gdir,
                                     output_filesuffix=output_filesuffix,
                                     prcp=precip, temp=temp,
                                     temp_std=temp_std,
                                     source=output_filesuffix,
                                     year_range=year_range,
                                     climate_historical_filesuffix=climate_historical_filesuffix,
                                     correct = correct,
                                     **kwargs)
    elif temporal_resol == 'daily':
        process_gcm_data_adv_daily(gdir,
                                   output_filesuffix=output_filesuffix,
                                   prcp=precip, temp=temp,
                                   source=output_filesuffix,
                                   year_range=year_range,
                                   climate_historical_filesuffix=climate_historical_filesuffix,
                                   correct = correct,
                                   **kwargs)



@entity_task(log)
def bayes_mbcalibration(gd, mb_type='mb_monthly', cores=4,
                        grad_type='cte', melt_f_prior=None,
                        dataset=None, path=None, uniform=None,
                        pd_geodetic_comp_alps=None, predict_data=None,
                        pd_nonan_alps_calib=None, pd_params_calib=None,
                        nosigma=False, use_two_msm = True, 
                        ):
    """ bayesian mass balance calibration """
    max_allowed_specificMB = pd_geodetic_comp_alps.loc[
        gd.rgi_id, 'max_allowed_specificMB']

    h, w = gd.get_inversion_flowline_hw()

    # at instantiation use prcp_fac = 2.5, change this in def_get_mb later on
    gd_mb = TIModel(gd, 150, mb_type=mb_type, N=100, prcp_fac=2.5,
                    grad_type=grad_type, baseline_climate=dataset)
    #gd_mb.historical_climate_qc_mod(gd)

    with pm.Model() as model_new:
        pf = pm.TruncatedNormal('pf',
                                mu=predict_data.sel(
                                    rgi_id=gd.rgi_id).pf_interp.mean(),
                                sigma=predict_data.sel(
                                    rgi_id=gd.rgi_id).pf_interp.std(),
                                lower=0.1, upper=10)
        # maybe take also predict_data interpolated mu somehow ??? for that prcp_solid has to predict pf and melt_f at once ...
        # or use instead a uniform distribution ??? or interpolate a descriptor for both pf and melt_f ???
        if melt_f_prior == 'frequentist':
            melt_f = pm.TruncatedNormal('melt_f',
                                        mu=pd_nonan_alps_calib[
                                            'melt_f_opt_pf'].mean(),
                                        sigma=pd_nonan_alps_calib[
                                            'melt_f_opt_pf'].std(),
                                        lower=10, upper=1000)
        elif melt_f_prior == 'freq_bayesian':
            # reconstruct the distribution
            pd_normi = pd.DataFrame(columns=np.arange(5000))
            for rgi in pd_params_calib.index:
                pd_normi.loc[rgi] = np.random.normal(loc = pd_params_calib.loc[rgi]['melt_f_mean_calib'],
                                     scale = pd_params_calib.loc[rgi]['melt_f_std_calib'],
                                        size=5000) 
            melt_f = pm.TruncatedNormal('melt_f',
                                        mu= pd_normi.stack().mean(),
                                        sigma = pd_normi.stack().std(), lower=10, upper=1000)
            #melt_f = pm.TruncatedNormal('melt_f',
            #                            mu=pd_params_calib[
            #                                'melt_f_mean_calib'].mean(),
            #                            sigma=pd_params_calib[
            #                                'melt_f_mean_calib'].std(),
            #                            lower=10, upper=1000)
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
                                        lower=10, upper=1000)
    # print(pd_geodetic_comp_alps.loc[gd.rgi_id])
    print(nosigma)
    burned_trace_valid, model_T_valid, _ = bayes_dummy_model_better(uniform,
                                                                    max_allowed_specificMB=max_allowed_specificMB,
                                                                    gd=gd,
                                                                    sampler='nuts',
                                                                    cores=cores,
                                                                    ys=np.arange(
                                                                        2000,
                                                                        2019,
                                                                        1),
                                                                    gd_mb=gd_mb,
                                                                    h=h,
                                                                    w=w,
                                                                    use_two_msm=use_two_msm,
                                                                    nosigma=nosigma,
                                                                    # predict_data = predict_data,
                                                                    model=model_new,
                                                                    # pd_calib_opt=pd_calib_opt,
                                                                    first_ppc=False,
                                                                    first_ppc_200=True,
                                                                    predict_historic=False,
                                                                    pd_geodetic_comp=pd_geodetic_comp_alps)
    # burned_trace_valid.posterior_predictive = burned_trace_valid.posterior_predictive.sel(chain=0).drop('chain')
    if use_two_msm:
        addi = ''
    else:
        addi = '_use_one_msm'
    if nosigma:
        print(nosigma)
        burned_trace_valid.to_netcdf(path + 'burned_trace_plus200samples/' +
                                 '{}_burned_trace_plus200samples_{}_{}_{}_meltfprior{}_nosigma{}.nc'.format(
                                     gd.rgi_id, dataset, mb_type, grad_type,
                                     melt_f_prior, addi))
    else:
        burned_trace_valid.to_netcdf(path + 'burned_trace_plus200samples/' +
                                 '{}_burned_trace_plus200samples_{}_{}_{}_meltfprior{}{}.nc'.format(
                                     gd.rgi_id, dataset, mb_type, grad_type,
                                     melt_f_prior, addi))


# make an execute_entity_task out of this to make it ointarallelisable!
@entity_task(log)
def inversion_and_run_from_climate_with_bayes_mb_params(gdir, a_factor=1,
                                                        y0=None, ye_h=2014,
                                                        mb_type='mb_monthly',
                                                        grad_type='cte',
                                                        rcps=None,
                                                        ssps=['ssp126',
                                                              'ssp370'],
                                                        melt_f_prior=None,
                                                        burned_trace=None,
                                                        dataset=None,
                                                        nosigma=False,
                                                        ensemble=None,
                                                        use_two_msm=True,
                                                        path_proj = '/home/users/lschuster/bayesian_calibration/WFDE5_ISIMIP/projections/'):
    """ Does inversion tasks for predefined a_factor and then runs a glacier first with historical then with projections

    TODO: add ensemble type as option, add historical and projection climate as options ... maybe similar to run_from_climate_data
    """
    # for gdir in gdirs:
    # print(gdir.rgi_id)
    # instead: create a file with only the meltf_pf combinations for each glacier merges, this is faster than opening it always again ...
    # try:
    if nosigma:
        nosigmaadd = '_nosigma'
    else:
        nosigmaadd = ''
    if use_two_msm:
        add_msm = ''
    else:
        add_msm = '_use_one_msm'
    if burned_trace == None:
        sample_path = '/home/users/lschuster/bayesian_calibration/WFDE5_ISIMIP/burned_trace_plus200samples/'
        burned_trace = az.from_netcdf(
            sample_path + '{}_burned_trace_plus200samples_WFDE5_CRU_{}_{}_meltfpriorfreq_bayesian{}{}.nc'.format(
                gdir.rgi_id, mb_type, grad_type, nosigmaadd, add_msm))
    try:
        burned_trace.posterior_predictive = burned_trace.posterior_predictive.sel(
            chain=0).drop('chain')
    except:
        pass

    # instantiatoin can happen independent of pf and melt_f:
    mb = TIModel(gdir, 200, mb_type=mb_type, grad_type=grad_type,
                 residual=0, baseline_climate=dataset)
    mb.historical_climate_qc_mod(gdir)

    mb_grad_draws_h = {}
    a_factors_dict = {}
    fs_dict = {}

    # mb_grad_draws_gcm ={'ssp126':{}, 'ssp370':{}}
    for j in np.arange(0, 201):
        # try:
        # get the parameter of draw j:
        # and put them on TIModel
        # try:
        if j < 200:
            melt_f = burned_trace.posterior_predictive.sel(draw=j).melt_f.values
            pf = burned_trace.posterior_predictive.sel(draw=j).pf.values
        # print(type(pf), melt_f, type(y0), ye)
        # ye=2018
        else:
            # compute aswell best "point" estimate:

            melt_f = az.plots.plot_utils.calculate_point_estimate(
                'mean', burned_trace.posterior.melt_f.stack(
                    draws=("chain", "draw"))).values
            pf = az.plots.plot_utils.calculate_point_estimate(
                'mean',
                burned_trace.posterior.pf.stack(draws=("chain", "draw"))).values
            j = 'point_estimate'
        mb.melt_f = melt_f
        mb.prcp_fac = pf

        # get the apparent_mb
        climate.apparent_mb_from_any_mb(gdir, mb_model=mb,
                                        mb_years=np.arange(y0, 2018 + 1, 1))
        
        if a_factor == 'per_glacier_and_draw':
            add = '_{}'.format(a_factor)
            # here I calibrate on glacier per glacier and draw per draw basis!            
            border = 80
            filter = border >= 20
            _df = oggm.workflow.calibrate_inversion_from_consensus([gdir],
                                                                   apply_fs_on_mismatch=True,
                                                                   error_on_mismatch=False,
                                                                   filter_inversion_output=filter)
            # np.testing.assert_allclose(_df.sum()['vol_itmix_m3'] / _df.sum()['vol_oggm_m3'], 1, rtol=1e-2)

            a_factor_calib = gdir.get_diagnostics()['inversion_glen_a'] / cfg.PARAMS['inversion_glen_a']   
            #print(add, gdir.rgi_id, j, a_factor_calib)
            a_factors_dict[j] = a_factor_calib
            fs_dict[j] = gdir.get_diagnostics()['inversion_fs']
            
        else:
            # ice thickness calibration done on regional level... 
            add = ''
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
        run_from_climate_data_TIModel(gdir, bias=0, min_ys=y0, ye=ye_h,
                                      mb_type=mb_type,
                                      grad_type=grad_type,
                                      precipitation_factor=pf, melt_f=melt_f,
                                      climate_input_filesuffix=dataset,
                                      output_filesuffix='{}{}_ISIMIP3b_{}_{}_{}_{}{}_historical_{}'.format(nosigmaadd,add_msm,
                                          dataset, ensemble, mb_type, grad_type, add,
                                          j))
        # only mass balance below ELA:
        h, w = gdir.get_inversion_flowline_hw()
        mb_gradient = []
        for y in np.arange(2004, 2015, 1):
            mbz = mb.get_annual_mb(h, year=y) * cfg.SEC_IN_YEAR * cfg.PARAMS[
                'ice_density']
            pab = np.where(mbz < 0)
            try:
                # Try to get the slope
                mb_gradient_y, _, _, _, _ = stats.linregress(h[pab], mbz[pab])
            except:
                mb_gradient_y = np.NaN
            mb_gradient.append(mb_gradient_y)
        mb_grad_draws_h[j] = np.array(mb_gradient).mean()
        # ds = utils.compile_run_output(gdirs, input_filesuffix='_historical_{}'.format(j))
        # ds.volume.plot(hue='rgi_id', ax = ax);
        if rcps != None:
            print('not implemented')
        else:
            for ssp in ssps:
                rid = '{}_{}_{}'.format('ISIMIP3b', ensemble, ssp)
                run_from_climate_data_TIModel(gdir, bias=0, mb_type=mb_type,
                                              grad_type=grad_type,
                                              precipitation_factor=pf,
                                              melt_f=melt_f,
                                              output_filesuffix='{}{}_ISIMIP3b_{}_{}_{}_{}{}_{}_{}'.format(nosigmaadd,add_msm,
                                                  dataset, ensemble, mb_type,
                                                  grad_type, add,
                                                  ssp, j),
                                              climate_input_filesuffix=rid,
                                              init_model_filesuffix='{}{}_ISIMIP3b_{}_{}_{}_{}{}_historical_{}'.format(nosigmaadd,add_msm,
                                                  dataset, ensemble, mb_type,
                                                  grad_type, add, j),
                                              climate_filename='gcm_data',
                                              ys=2015, ye=2100)
        # maybe merge them here directly and remove the old _j files ???
        # except:
        #    print(gdir.rgi_id)
        # except:
        # print(j)

    # print(mb_grad_draws)
    # pd.DataFrame(mb_grad_draws_h,
    #            index=['{}_{}_{}_{}_{}'.format(gdir.rgi_id, dataset,
    #                                           ensemble, mb_type, grad_type)]).to_csv('{}_{}_{}_{}_{}_mb_gradient.csv'.format(gdir.rgi_id, dataset, ensemble, mb_type, grad_type))

    # this could go into tests later on...
    if a_factor != 'per_glacier_and_draw':
        np.testing.assert_allclose(gdir.get_diagnostics()['inversion_glen_a'],
                                   cfg.PARAMS['glen_a'] * a_factor)
    if nosigma:
        nosigmaaddi = 'nosigma_'
    else:
        nosigmaaddi = ''
    if not use_two_msm:
        addi_msm = 'use_one_msm_'
    else:
        addi_msm = ''
    diag_path = '{}{}ISIMIP3b_{}_{}_{}_{}{}'.format(nosigmaaddi, addi_msm, dataset, ensemble, mb_type,
                                                grad_type, add)
    #else:
    #    diag_path = 'ISIMIP3b_{}_{}_{}_{}_{}'.format(dataset, ensemble, mb_type,
    #                                          grad_type, 'a_factor_per_draw')
    print('succesfully projected: {}'.format(gdir.rgi_id))

    # now merge the files together ...
    rgi_id = gdir.rgi_id
    


    ##burned_trace = az.from_netcdf('/home/users/lschuster/bayesian_calibration/WFDE5_ISIMIP/burned_trace_plus200samples/{}_burned_trace_plus200samples_{}_{}_{}_meltfprior{}.nc'.format(rgi_id, dataset, mb_type, grad_type, melt_f_prior))

    # burned_trace.posterior_predictive = burned_trace.posterior_predictive.sel(
    #    chain=0).drop('chain')

    # choose draw 200 (point estimate as this is most likely to work)
    ds_historical = xr.open_dataset(gdir.get_filepath('model_diagnostics')[
                                    :-20] + 'model_diagnostics_{}_historical_point_estimate.nc'.format(
        diag_path))
    ds_historical.coords[
        'draw'] = 200  # a string is not working with numbers ...

    ds_gcm_ssps = []
    for ssp in ssps:
        try:
            ds_gcm_ssp = xr.open_dataset(gdir.get_filepath('model_diagnostics')[
                                         :-20] + 'model_diagnostics_{}_{}_point_estimate.nc'.format(
                diag_path, ssp))
            ds_gcm_ssp.coords['draw'] = 200
            ds_gcm_ssp.coords['ssp'] = ssp
            # ds_gcm_ssp['mb_gradient_mean'] = mb_grad_draws_gcm[ssp][0]

            ds_gcm_ssps.append(ds_gcm_ssp)
        except:
            '{} did not work'.format(ssp)
    ds_gcm_0 = xr.concat(ds_gcm_ssps, 'ssp')

    melt_f = az.plots.plot_utils.calculate_point_estimate(
        'mean', burned_trace.posterior.melt_f.stack(
            draws=("chain", "draw"))).values
    pf = az.plots.plot_utils.calculate_point_estimate('mean',
                                                      burned_trace.posterior.pf.stack(
                                                          draws=(
                                                          "chain", "draw")))

    ds_gcm_0['melt_f'] = melt_f
    ds_gcm_0['pf'] = pf
    ds_historical['melt_f'] = melt_f
    ds_historical['pf'] = pf
    ds_historical['mb_gradient_mean'] = mb_grad_draws_h['point_estimate']
    if a_factor == 'per_glacier_and_draw':
        ds_historical['a_factor'] = a_factors_dict['point_estimate']
        ds_gcm_0['a_factor'] = a_factors_dict['point_estimate']
        ds_historical['fs'] = fs_dict['point_estimate']
        ds_gcm_0['fs'] = fs_dict['point_estimate']

    
    j_works = []
    for jj in np.arange(0, 200):
        try:
            pf = burned_trace.posterior_predictive.sel(draw=jj).pf
            melt_f = burned_trace.posterior_predictive.sel(draw=jj).melt_f

            ds_historical_j = xr.open_dataset(
                gdir.get_filepath('model_diagnostics')[
                :-20] + 'model_diagnostics_{}_historical_{}.nc'.format(
                    diag_path, jj))

            ds_historical_j.coords['draw'] = jj
            ds_historical_j['melt_f'] = melt_f
            ds_historical_j['pf'] = pf
            ds_historical_j['mb_gradient_mean'] = mb_grad_draws_h[jj]
            if a_factor == 'per_glacier_and_draw':
                ds_historical_j['a_factor'] = a_factors_dict[jj]
                ds_historical_j['fs'] = fs_dict[jj]

            ds_historical = xr.concat([ds_historical, ds_historical_j],
                                      'draw')
            j_works.append(jj)
        except:
            print('draw {} did NOT work'.format(jj))

    ds_historical['mb_type_grad'] = mb_type + '_' + grad_type
    if a_factor != 'per_glacier_and_draw':
        ds_historical['a_factor'] = a_factor

    ds_gcm_rr = []
    for ssp in ssps:
        try:
            ds_gcm_r = ds_gcm_0.sel(ssp=ssp)
            for jj in j_works:  # np.arange(1, 201):
                try:
                    pf = burned_trace.posterior_predictive.sel(draw=jj).pf
                    melt_f = burned_trace.posterior_predictive.sel(
                        draw=jj).melt_f

                    ds_gcm_j = xr.open_dataset(
                        gdir.get_filepath('model_diagnostics')[
                        :-20] + 'model_diagnostics_{}_{}_{}.nc'.format(
                            diag_path, ssp, jj))

                    ds_gcm_j.coords[
                        'draw'] = jj # if jj='point_estimate' 200 else jj
                    ds_gcm_j.coords['ssp'] = ssp
                    ds_gcm_j['melt_f'] = melt_f
                    ds_gcm_j['pf'] = pf
                    if a_factor == 'per_glacier_and_draw':
                        ds_gcm_j['a_factor'] = a_factors_dict[jj]
                        ds_gcm_j['fs'] = fs_dict[jj]

                    # ds_gcm_j['mb_gradient_mean'] = mb_grad_draws_gcm[ssp][jj]

                    ds_gcm_r = xr.concat([ds_gcm_r, ds_gcm_j], 'draw')
                except:
                    print('in addition: draw {} did NOT work'.format(jj))

            ds_gcm_rr.append(ds_gcm_r)
        except:
            print('{} did not work'.format(ssp))
    ds_gcm = xr.concat(ds_gcm_rr, 'ssp')

    ds_gcm['mb_type_grad'] = mb_type + '_' + grad_type
    
    if a_factor != 'per_glacier_and_draw':
        ds_gcm['a_factor'] = a_factor

    # need to include also the "best" point estimate combination
    # and also include here inside the saved mass balance gradient (but use the right definition)
    if a_factor != 'per_glacier_and_draw':
        ds_historical.to_netcdf(
            path_proj + '{}_{}_{}_{}_{}_{}_{}_historical'.format(
                rgi_id, 'ISIMIP3b', ensemble, dataset,
                mb_type, grad_type, melt_f_prior))
        ds_gcm.to_netcdf(
            path_proj + '{}_{}_{}_{}_{}_{}_{}_ssps'.format(
                rgi_id, 'ISIMIP3b', ensemble, dataset,
                mb_type, grad_type,
                melt_f_prior))
    else:
        ds_historical.to_netcdf(
            path_proj + '{}{}a_factor_per_draw_{}_{}_{}_{}_{}_{}_{}_historical'.format(nosigmaaddi, addi_msm,
                rgi_id, 'ISIMIP3b', ensemble, dataset,
                mb_type, grad_type, melt_f_prior))
        ds_gcm.to_netcdf(
            path_proj + '{}{}a_factor_per_draw_{}_{}_{}_{}_{}_{}_{}_ssps'.format(nosigmaaddi, addi_msm,
                rgi_id, 'ISIMIP3b', ensemble, dataset,
                mb_type, grad_type,
                melt_f_prior))
        