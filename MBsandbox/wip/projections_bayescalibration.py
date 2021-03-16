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
from MBsandbox.mbmod_daily_oneflowline import write_climate_file
from MBsandbox.wip.bayes_calib_geod_direct import bayes_dummy_model_better

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
        fp = gdir.get_filepath('model_geometry', filesuffix=init_model_filesuffix)
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
def process_gcm_data_adv(gdir, filesuffix='', prcp=None, temp=None,
                         temp_std=None, temporal_resol='monthly',
                     year_range=('1961', '1990'), scale_stddev=True,
                     time_unit=None, calendar=None, source='',
                     climate_historical_filesuffix=''):
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
    filesuffix : str
        append a suffix to the filename (useful for ensemble experiments).
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
    if sm !=1:
        prcp = prcp[sm-1:sm-13].load()
        temp = temp[sm-1:sm-13].load()
        temp_std = temp_std[sm-1:sm-13].load()

    assert len(prcp) // 12 == len(prcp) / 12, 'Somehow we didn\'t get full years'
    assert len(temp) // 12 == len(temp) / 12, 'Somehow we didn\'t get full years'

    # Get historical_climate to apply the anomaly to 
    fpath = gdir.get_filepath('climate_historical',
                              filesuffix = climate_historical_filesuffix)
    ds_cru = xr.open_dataset(fpath, use_cftime=True)

    # Add CRU (or the defined climate from above...) clim
    dscru = ds_cru.sel(time=slice(*year_range))

    # compute monthly anomalies
    # of temp
    if scale_stddev:
        # This is a bit more arithmetic
        ts_tmp_sel = temp.sel(time=slice(*year_range))
        ts_tmp_std = ts_tmp_sel.groupby('time.month').std(dim='time')
        std_fac = dscru.temp.groupby('time.month').std(dim='time') / ts_tmp_std
        std_fac = std_fac.roll(month=13-sm, roll_coords=True)
        std_fac = np.tile(std_fac.data, len(temp) // 12)
        # We need an even number of years for this to work
        if ((len(ts_tmp_sel) // 12) % 2) == 1:
            raise InvalidParamsError('We need an even number of years '
                                     'for this to work')
        win_size = len(ts_tmp_sel) + 1

        def roll_func(x, axis=None):
            x = x[:, ::12]
            n = len(x[0, :]) // 2
            xm = np.nanmean(x, axis=axis)
            return xm + (x[:, n] - xm) * std_fac

        temp = temp.rolling(time=win_size, center=True,
                            min_periods=1).reduce(roll_func)

    ts_tmp_sel = temp.sel(time=slice(*year_range))
    assert len(temp.sel(time=slice(*year_range)).time) == len(dscru.time)
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
    loc_tmp = dscru.temp.groupby('time.month').mean()
    ts_tmp = ts_tmp.groupby('time.month') + loc_tmp
    
    # for temp daily std
    loc_dstdtmp = dscru.temp_std.groupby('time.month').mean()
    ts_dstdtmp = ts_dstdtmp.groupby('time.month') + loc_dstdtmp

    # for prcp
    loc_pre = dscru.prcp.groupby('time.month').mean()
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
    
    # for gradient
    if temporal_resol == 'monthly':
        try:
            hist_gradient = dscru.gradient.groupby('time.month').mean().values
            # use the same gradient for every year
            # may be this could be done differently to save data...
            # print(hist_gradient)
            gradient = np.repeat(hist_gradient, len(temp.time.values)/12)
        except:
            gradient = None
    elif temporal_resol =='daily':
        temp_std = None
    # print(len(temp.time.values), len(ts_pre.values), len(gradient), len(temp_std))
    
    #time_unit = temp.time.units
    #calendar = temp.time.calendar
    #print(time_unit, calendar)
    write_climate_file(gdir, temp.time.values,
                                    ts_pre.values, ts_tmp.values,
                                    float(dscru.ref_hgt),
                                    prcp.lon.values, prcp.lat.values,
                                    time_unit=time_unit,
                                    calendar=calendar,
                                    file_name='gcm_data',
                                    source=source+'_historical{}'.format(climate_historical_filesuffix),
                                    filesuffix=filesuffix,
                            gradient=gradient,
                            temp_std =ts_dstdtmp.values,
                           temporal_resol=temporal_resol)

    ds_cru.close()
    
@entity_task(log, writes=['gcm_data'])
def process_isimip_data(gdir, filesuffix='', fpath_temp=None,
                        fpath_temp_std=None,
                        fpath_precip=None, 
                        climate_historical_filesuffix='',
                        ensemble = 'mri-esm2-0_r1i1p1f1',  # from temperature tie series the "median" ensemble
                        ssp = 'ssp126',
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

    if filesuffix == '':
        # recognize the gcm climate file for later
        filesuffix = '_ISIMIP3b_{}_{}'.format(ensemble, ssp)
    
    # Get the path of GCM temperature & precipitation data
    #if fpath_temp is None:
    #    if not ('cmip5_temp_file' in cfg.PATHS):
     #       raise ValueError("Need to set cfg.PATHS['isimip3b_temp_file']")
    #    fpath_temp = cfg.PATHS['isimip3b_temp_file']
    #if fpath_precip is None:
    #    if not ('cmip5_precip_file' in cfg.PATHS):
    #        raise ValueError("Need to set cfg.PATHS['isimip3b_precip_file']")
    #    fpath_precip = cfg.PATHS['isimip3b_precip_file']

    # Glacier location
    glon = gdir.cenlon
    glat = gdir.cenlat

    # need to aggregate first both gcm types !!!!
    fpath_temp_gcm = fpath_temp+ '{}_w5e5_{}_tasAdjust_global_monthly_2015_2100.nc'.format(ensemble, ssp)
    fpath_temp_historical = fpath_temp+ '{}_w5e5_historical_tasAdjust_global_monthly_1850_2014.nc'.format(ensemble)
    
    fpath_temp_std_gcm = fpath_temp_std+ '{}_w5e5_{}_tasAdjust_std_global_monthly_2015_2100.nc'.format(ensemble, ssp)
    fpath_temp_std_historical = fpath_temp_std+ '{}_w5e5_historical_tasAdjust_std_global_monthly_1850_2014.nc'.format(ensemble)
    
    fpath_prcp_gcm = fpath_precip+ '{}_w5e5_{}_prAdjust_global_monthly_2015_2100.nc'.format(ensemble, ssp)
    fpath_prcp_historical = fpath_precip+ '{}_w5e5_historical_prAdjust_global_monthly_1850_2014.nc'.format(ensemble)

    # Read the GCM files
    with xr.open_dataset(fpath_temp_historical, use_cftime=True) as tempds_hist,\
        xr.open_dataset(fpath_temp_gcm, use_cftime=True) as tempds_gcm,\
        xr.open_dataset(fpath_temp_std_historical, use_cftime=True) as tempds_std_hist,\
        xr.open_dataset(fpath_temp_std_gcm, use_cftime=True) as tempds_std_gcm,\
        xr.open_dataset(fpath_prcp_historical, use_cftime=True) as precipds_hist,\
        xr.open_dataset(fpath_prcp_gcm, use_cftime=True) as precipds_gcm:

        # first merge historical with gcm together
        tempds = xr.merge([tempds_hist, tempds_gcm])
        tempds_std = xr.merge([tempds_std_hist, tempds_std_gcm])
        precipds = xr.merge([precipds_hist, precipds_gcm])
        
        # Check longitude conventions
        if tempds.lon.min() >= 0 and glon <= 0:
            glon += 360

        # Take the closest to the glacier
        # Should we consider GCM interpolation?
        # print(tempds_std)
        temp = tempds.tasAdjust.sel(lat=glat, lon=glon, method='nearest')
        temp_std = tempds_std.tasAdjust_std.sel(lat=glat, lon=glon, method='nearest')
        precip = precipds.prAdjust.sel(lat=glat, lon=glon, method='nearest')

        # Back to [-180, 180] for OGGM
        temp.lon.values = temp.lon if temp.lon <= 180 else temp.lon - 360
        temp_std.lon.values = temp_std.lon if temp_std.lon <= 180 else temp.lon - 360
        precip.lon.values = precip.lon if precip.lon <= 180 else precip.lon - 360

        # Convert kg m-2 s-1 to mm mth-1 => 1 kg m-2 = 1 mm !!!
        assert 'kg m-2 s-1' in precip.units, 'Precip units not understood'

        ny, r = divmod(len(temp), 12)
        assert r == 0
        dimo = [cfg.DAYS_IN_MONTH[m - 1] for m in temp['time.month']]
        precip = precip * dimo * (60 * 60 * 24)
        # print(len(precip))
    process_gcm_data_adv(gdir, filesuffix=filesuffix, prcp=precip, temp=temp, temp_std=temp_std,
                         source=filesuffix, year_range=('1979', '2018'), 
                         climate_historical_filesuffix=climate_historical_filesuffix,
                         **kwargs)



@entity_task(log)
def bayes_mbcalibration(gd, mb_type='mb_monthly', cores = 4,
                         grad_type='cte', melt_f_prior=None,
                         dataset=None, path=None, uniform=None,
                         pd_geodetic_comp_alps=None, predict_data=None,
                         pd_nonan_alps_calib=None, pd_params_calib=None,
                         ):
    """ bayesian mass balance calibration """
    max_allowed_specificMB = pd_geodetic_comp_alps.loc[
        gd.rgi_id, 'max_allowed_specificMB']

    h, w = gd.get_inversion_flowline_hw()

    # at instantiation use prcp_fac = 2.5, change this in def_get_mb later on
    gd_mb = TIModel(gd, 150, mb_type=mb_type, N=100, prcp_fac=2.5,
                    grad_type=grad_type, baseline_climate=dataset)
    gd_mb.historical_climate_qc_mod(gd)

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
            melt_f = pm.TruncatedNormal('melt_f',
                                        mu=pd_params_calib[
                                            'melt_f_mean_calib'].mean(),
                                        sigma=pd_params_calib[
                                            'melt_f_mean_calib'].std(),
                                        lower=10, upper=1000)
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
    burned_trace_valid, model_T_valid, _ = bayes_dummy_model_better(uniform,
                                                                    max_allowed_specificMB=max_allowed_specificMB,
                                                                    gd=gd,
                                                                    sampler='nuts',
                                                                    cores = cores,
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
    burned_trace_valid.to_netcdf(path + 'burned_trace_plus200samples/' +
                                 '{}_burned_trace_plus200samples_{}_{}_{}_meltfprior{}.nc'.format(
            gd.rgi_id, dataset, mb_type, grad_type, melt_f_prior))