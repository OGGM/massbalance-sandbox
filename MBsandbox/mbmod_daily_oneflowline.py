#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 24 12:28:37 2020

@author: lilianschuster

different temperature index mass balance types added that are working with the Huss flowlines
"""
# jax_true = True
# if jax_true:
#     import jax.numpy as np
#     import numpy as onp
# else: problem: nan values, where stuff ...
import numpy as np

import pandas as pd
import xarray as xr
import os
import netCDF4
import datetime
import warnings
import scipy.stats as stats
import logging
# import oggm

# imports from oggm
from oggm import entity_task
from oggm import cfg, utils
from oggm.cfg import SEC_IN_YEAR, SEC_IN_MONTH
from oggm.utils import (floatyear_to_date, ncDataset,
                        clip_min, clip_array)
from oggm.utils._funcs import haversine
from oggm.exceptions import InvalidParamsError
from oggm.shop.ecmwf import get_ecmwf_file, BASENAMES
from oggm.core.massbalance import MassBalanceModel

# Module logger
log = logging.getLogger(__name__)

ECMWF_SERVER = 'https://cluster.klima.uni-bremen.de/~oggm/climate/'


# %%

# add era5_daily dataset, this only works with process_era5_daily_data
BASENAMES['ERA5_daily'] = {
        'inv': 'era5/daily/v1.0/era5_glacier_invariant_flat.nc',
        'tmp': 'era5/daily/v1.0/era5_daily_t2m_1979-2018_flat.nc'
        # only glacier-relevant gridpoints included!
        }


# this could be used in general
def write_climate_file(gdir, time, prcp, temp,
                       ref_pix_hgt, ref_pix_lon, ref_pix_lat,
                       gradient=None, temp_std=None,
                       time_unit=None, calendar=None,
                       source=None, file_name='climate_historical',
                       filesuffix=''):
    """Creates a netCDF4 file with climate data timeseries.

    Parameters
    ----------
    gdir:
        glacier directory
    time : ndarray
        the time array, in a format understood by netCDF4
    prcp : ndarray
        the precipitation array (unit: 'kg m-2 month-1')
    temp : ndarray
        the temperature array (unit: 'degC')
    ref_pix_hgt : float
        the elevation of the dataset's reference altitude
        (for correction). In practice it is the same altitude as the
        baseline climate.
    ref_pix_lon : float
        the location of the gridded data's grid point
    ref_pix_lat : float
        the location of the gridded data's grid point
    gradient : ndarray, optional
        whether to use a time varying gradient
    temp_std : ndarray, optional
        the daily standard deviation of temperature (useful for PyGEM)
    time_unit : str
        the reference time unit for your time array. This should be chosen
        depending on the length of your data. The default is to choose
        it ourselves based on the starting year.
    calendar : str
        If you use an exotic calendar (e.g. 'noleap')
    source : str
        the climate data source (required)
    file_name : str
        How to name the file
    filesuffix : str
        Apply a suffix to the file
    """

    if source == 'ERA5_daily' and filesuffix == '':
        raise InvalidParamsError('filesuffix should be "_daily" as only \
                                 file_name climate_historical is normally \
                                     monthly data')
    # overwrite is default
    fpath = gdir.get_filepath(file_name, filesuffix=filesuffix)
    if os.path.exists(fpath):
        os.remove(fpath)

    if source is None:
        raise InvalidParamsError('`source` kwarg is required')

    zlib = cfg.PARAMS['compress_climate_netcdf']

    try:
        y0 = time[0].year
        y1 = time[-1].year
    except AttributeError:
        time = pd.DatetimeIndex(time)
        y0 = time[0].year
        y1 = time[-1].year

    if time_unit is None:
        # http://pandas.pydata.org/pandas-docs/stable/timeseries.html
        # #timestamp-limitations
        if y0 > 1800:
            time_unit = 'days since 1801-01-01 00:00:00'
        elif y0 >= 0:
            time_unit = ('days since {:04d}-01-01 '
                         '00:00:00'.format(time[0].year))
        else:
            raise InvalidParamsError('Time format not supported')

    with ncDataset(fpath, 'w', format='NETCDF4') as nc:
        nc.ref_hgt = ref_pix_hgt
        nc.ref_pix_lon = ref_pix_lon
        nc.ref_pix_lat = ref_pix_lat
        nc.ref_pix_dis = haversine(gdir.cenlon, gdir.cenlat,
                                   ref_pix_lon, ref_pix_lat)
        nc.climate_source = source
        if time[0].month == 1:
            nc.hydro_yr_0 = y0
        else:
            nc.hydro_yr_0 = y0 + 1
        nc.hydro_yr_1 = y1

        nc.createDimension('time', None)

        nc.author = 'OGGM'
        nc.author_info = 'Open Global Glacier Model'

        timev = nc.createVariable('time', 'i4', ('time',))

        tatts = {'units': time_unit}
        if calendar is None:
            calendar = 'standard'

        tatts['calendar'] = calendar
        try:
            numdate = netCDF4.date2num([t for t in time], time_unit,
                                       calendar=calendar)
        except TypeError:
            # numpy's broken datetime only works for us precision
            time = time.astype('M8[us]').astype(datetime.datetime)
            numdate = netCDF4.date2num(time, time_unit, calendar=calendar)

        timev.setncatts(tatts)
        timev[:] = numdate

        v = nc.createVariable('prcp', 'f4', ('time',), zlib=zlib)
        v.units = 'kg m-2'
        # this could be made more beautriful
        # just rough estimate
        if len(prcp) > (nc.hydro_yr_1 - nc.hydro_yr_0 + 1) * 30 * 12:
            v.long_name = ("total daily precipitation amount, "
                           "assumed same for each day of month")
        elif len(prcp) == (nc.hydro_yr_1 - nc.hydro_yr_0 + 1) * 12:
            v.long_name = 'total monthly precipitation amount'
        else:
            v.long_name = 'total monthly precipitation amount'
            warnings.warn("there might be a conflict in the prcp timeseries,"
                          "please check!")

        v[:] = prcp

        v = nc.createVariable('temp', 'f4', ('time',), zlib=zlib)
        v.units = 'degC'
        if source == 'ERA5_daily' and len(temp) > (y1 - y0) * 30 * 12:
            v.long_name = '2m daily temperature at height ref_hgt'
        elif source == 'ERA5_daily' and len(temp) <= (y1 - y0) * 30 * 12:
            raise InvalidParamsError('if the climate dataset (here source)'
                                     'is ERA5_daily, temperatures should be in'
                                     'daily resolution, please check or set'
                                     'set source to another climate dataset')
        else:
            v.long_name = '2m monthly temperature at height ref_hgt'

        v[:] = temp

        if gradient is not None:
            v = nc.createVariable('gradient', 'f4', ('time',), zlib=zlib)
            v.units = 'degC m-1'
            v.long_name = ('temperature gradient from local regression or'
                           'lapserates')
            v[:] = gradient

        if temp_std is not None:
            v = nc.createVariable('temp_std', 'f4', ('time',), zlib=zlib)
            v.units = 'degC'
            v.long_name = 'standard deviation of daily temperatures'
            v[:] = temp_std


@entity_task(log, writes=['climate_historical_daily'])
def process_era5_daily_data(gdir, y0=None, y1=None, output_filesuffix='_daily',
                            cluster=False):
    """Processes and writes the era5 daily baseline climate data for a glacier.
    into climate_historical_daily.nc

    Extracts the nearest timeseries and writes everything to a NetCDF file.
    This uses only the ERA5 daily temperatures. The precipitation, lapse
    rate and standard deviations are used from ERA5dr.

    TODO: see _verified_download_helper no known hash for
    era5_daily_t2m_1979-2018_flat.nc and era5_glacier_invariant_flat
    ----------
    y0 : int
        the starting year of the timeseries to write. The default is to take
        the entire time period available in the file, but with this kwarg
        you can shorten it (to save space or to crop bad data)
    y1 : int
        the starting year of the timeseries to write. The default is to take
        the entire time period available in the file, but with this kwarg
        you can shorten it (to save space or to crop bad data)
    output_filesuffix : str
        this add a suffix to the output file (useful to avoid overwriting
        previous experiments)
    cluster : bool
        default is False, if this is run on the cluster, set it to True,
        because we do not need to download the files

    """

    # era5daily only for temperature
    dataset = 'ERA5_daily'
    # for the other variables use the data of ERA5dr
    dataset_othervars = 'ERA5dr'

    # get the central longitude/latidudes of the glacier
    lon = gdir.cenlon + 360 if gdir.cenlon < 0 else gdir.cenlon
    lat = gdir.cenlat

    cluster_path = '/home/www/oggm/climate/'

    if cluster:
        path = cluster_path + BASENAMES[dataset]['tmp']
    else:
        path = get_ecmwf_file(dataset, 'tmp')

    # Use xarray to read the data
    # would go faster with netCDF -.-
    with xr.open_dataset(path) as ds:
        assert ds.longitude.min() >= 0

        # set temporal subset for the ts data (hydro years)
        if gdir.hemisphere == 'nh':
            sm = cfg.PARAMS['hydro_month_nh']
        elif gdir.hemisphere == 'sh':
            sm = cfg.PARAMS['hydro_month_sh']

        em = sm - 1 if (sm > 1) else 12

        yrs = ds['time.year'].data
        y0 = yrs[0] if y0 is None else y0
        y1 = yrs[-1] if y1 is None else y1

        if y1 > 2018 or y0 < 1979:
            text = 'The climate files only go from 1979--2018,\
                choose another y0 and y1'
            raise InvalidParamsError(text)
        # if default settings: this is the last day in March or September
        time_f = '{}-{:02d}'.format(y1, em)
        end_day = int(ds.sel(time=time_f).time.dt.daysinmonth[-1].values)

        #  this was tested also for hydro_month = 1
        ds = ds.sel(time=slice('{}-{:02d}-01'.format(y0, sm),
                               '{}-{:02d}-{}'.format(y1, em, end_day)))

        try:
            # computing all the distances and choose the nearest gridpoint
            c = (ds.longitude - lon)**2 + (ds.latitude - lat)**2
            ds = ds.isel(points=c.argmin())
        # I turned this around
        except ValueError:
            ds = ds.sel(longitude=lon, latitude=lat, method='nearest')
            # normally if I do the flattening, this here should not occur

        # temperature should be in degree Celsius for the glacier climate files
        temp = ds['t2m'].data - 273.15
        time = ds.time.data

        ref_lon = float(ds['longitude'])
        ref_lat = float(ds['latitude'])

        ref_lon = ref_lon - 360 if ref_lon > 180 else ref_lon

    # pre should be done as in ERA5dr datasets
    with xr.open_dataset(get_ecmwf_file(dataset_othervars, 'pre')) as ds:
        assert ds.longitude.min() >= 0

        yrs = ds['time.year'].data
        y0 = yrs[0] if y0 is None else y0
        y1 = yrs[-1] if y1 is None else y1
        # Attention here we take the same y0 and y1 as given from the
        # daily tmp dataset (goes till end of 2018)

        ds = ds.sel(time=slice('{}-{:02d}-01'.format(y0, sm),
                               '{}-{:02d}-01'.format(y1, em)))
        try:
            # prcp is not flattened, so this here should work normally
            ds = ds.sel(longitude=lon, latitude=lat, method='nearest')
        except ValueError:
            # if Flattened ERA5_precipitation?
            c = (ds.longitude - lon)**2 + (ds.latitude - lat)**2
            ds = ds.isel(points=c.argmin())

        # the prcp dataset needs to be restructured to have values for each day
        prcp = ds['tp'].data * 1000
        # just assume that precipitation is every day the same:
        prcp = np.repeat(prcp, ds['time.daysinmonth'])
        # Attention the unit is now prcp per day
        # (not per month as in OGGM default:
        # prcp = ds['tp'].data * 1000 * ds['time.daysinmonth']

    if cluster:
        path_inv = cluster_path + BASENAMES[dataset]['inv']
    else:
        path_inv = get_ecmwf_file(dataset, 'inv')
    with xr.open_dataset(path_inv) as ds:
        assert ds.longitude.min() >= 0
        ds = ds.isel(time=0)
        try:
            # Flattened ERA5_invariant (only possibility at the moment)
            c = (ds.longitude - lon)**2 + (ds.latitude - lat)**2
            ds = ds.isel(points=c.argmin())
        except ValueError:
            # this should not occur
            ds = ds.sel(longitude=lon, latitude=lat, method='nearest')

        G = cfg.G  # 9.80665
        hgt = ds['z'].data / G

    gradient = None
    temp_std = None
    path_lapserates = get_ecmwf_file(dataset_othervars, 'lapserates')
    with xr.open_dataset(path_lapserates) as ds:
        assert ds.longitude.min() >= 0

        yrs = ds['time.year'].data
        y0 = yrs[0] if y0 is None else y0
        y1 = yrs[-1] if y1 is None else y1
        # Attention here we take the same y0 and y1 as given from the
        # daily tmp dataset (goes till end of 2018)

        ds = ds.sel(time=slice('{}-{:02d}-01'.format(y0, sm),
                               '{}-{:02d}-01'.format(y1, em)))

        # no flattening done for the ERA5dr gradient dataset
        ds = ds.sel(longitude=lon, latitude=lat, method='nearest')

        # get the monthly gradient values
        gradient = ds['lapserate'].data

        # gradient needs to be restructured to have values for each day
        gradient = np.repeat(gradient, ds['time.daysinmonth'])
        # assume same gradient for each day

    # OK, ready to write
    write_climate_file(gdir, time, prcp, temp, hgt, ref_lon, ref_lat,
                       filesuffix=output_filesuffix,
                       gradient=gradient,
                       temp_std=temp_std,
                       source=dataset,
                       file_name='climate_historical')
    # This is now a new function, which could also work for other climates
    # but is used, so far, only for ERA5_daily as source dataset ..


# TODO:
# - name: TIModel? + DDFModel?
class TIModel(MassBalanceModel):
    """Different mass balance modules compatible to OGGM with one flowline

    so far this is only tested for the Huss flowlines
    """

    def __init__(self, gdir, melt_f, residual=0,
                 mb_type='mb_daily', N=100, loop=False,
                 grad_type='cte', filename='climate_historical',
                 input_filesuffix='',
                 repeat=False, ys=None, ye=None,
                 t_solid=0, t_liq=2, t_melt=0, prcp_fac=2.5,
                 default_grad=-0.0065,
                 temp_local_gradient_bounds=[-0.009, -0.003],
                 # check_climate=True,
                 SEC_IN_YEAR=SEC_IN_YEAR,
                 SEC_IN_MONTH=SEC_IN_MONTH
                 ):
        """ Initialize.
        Parameters
        ----------
        gdir : GlacierDirectory
            the glacier directory
        melt_f : float
            melt temperature sensitivity factor per month (kg /m² /mth /K),
            need to be prescribed, e.g. such that
            |mean(MODEL_MB)-mean(REF_MB)|--> 0
        residual : float, optional
            default is to use a residual of zero [mm we yr-1]
            Note that this residual is *substracted* from the computed MB.
            Indeed: residual = MODEL_MB - REFERENCE_MB.
            ToDO: maybe change the sign?,
            opposite to OGGM "MB terms + residual"
        mb_type: str
            three types: 'mb_daily' (default: use temp_std and N percentiles),
            'mb_monthly' (same as default OGGM mass balance),
            'mb_real_daily' (use daily temperature values).
            OGGM "MB terms + residual"
            the mb_type only work if the baseline_climate of gdir is right
        N : int
            number of percentiles used to generate gaussian-like daily
            temperatures from daily std and mean monthly temp
        loop : bool
            the way how the matrix multiplication is done,
            using np.matmul or a loop(default: False)
            only applied if mb_type is 'mb_daily'
            which one is faster?
        grad_type : str
            three types of applying the temperature gradient:
            'cte' (default, constant lapse rate, set to default_grad,
                   same as in default OGGM)
            'var_an_cycle' (varies spatially and over annual cycle,
                            but constant over the years)
            'var' (varies spatially & temporally as in the climate files)
        filename : str, optional
            set to a different BASENAME if you want to use alternative climate
            data, default is climate_historical
        input_filesuffix : str,
            the file suffix of the input climate file, default is '',
            if ERA5_daily with daily temperatures, it is set to _daily
        repeat : bool
            Whether the climate period given by [ys, ye] should be repeated
            indefinitely in a circular way
        ys : int
            The start of the climate period where the MB model is valid
            (default: the period with available data)
        ye : int
            The end of the climate period where the MB model is valid
            (default: the period with available data)
        t_solid : float
            temperature threshold for solid precipitation
            (degree Celsius, default 0)
        t_liq: float
            temperature threshold for liquid precipitation
            (degree Celsius, default 2)
        t_melt : float
            temperature threshold where snow/ice melts
            (degree Celsius, default 0)
        prcp_fac : float, >0
            multiplicative precipitation correction factor (default 2.5)
        default_grad : float,
            constant lapse rate (temperature gradient, default: -0.0065 m/K)
            if grad_type != cte, then this value is not used
            but instead the changing lapse rate from the climate datasets
        temp_local_gradient_bounds : [float, float],
            if grad_type != cte and the lapse rate does not lie in this range,
            set it instead to these minimum, maximum gradients
            (default: [-0.009, -0.003] m/K)
        SEC_IN_YEAR: float
            seconds in a year (default: 31536000s),
            maybe this could be changed
        SEC_IN_MONTH: float
            seconds in a month (default: 2628000s),
            maybe this could be changed as not each
            month has the same amount of seconds,
            in February can be a difference of 8%

        Attributes
        ----------
        temp_bias : float, default 0
            Add a temperature bias to the time series
        prcp_bias : float, default 1
            Precipitation factor to the time series (called bias for
            consistency with `temp_bias`)
        """
        # melt_f is only initiated here, and not used in __init__.py
        # so it does not matter if it is changed
        self.melt_f = melt_f
        if self.melt_f != None and self.melt_f <= 0:
            raise InvalidParamsError('melt_f has to be above zero!')
        # but there is a problem with prcp_fac,
        # as self.prcp is produced by changing prcp_fac
        # so there is no self.prcp_fac here
        # and we need to update the prcp via prcp_fac by a property
        if prcp_fac <= 0:
            raise InvalidParamsError('prcp_fac has to be above zero!')
        # private attribute
        self._prcp_fac = prcp_fac
        self.residual = residual

        # Parameters (from cfg.PARAMS in OGGM default)
        self.t_solid = t_solid
        self.t_liq = t_liq
        self.t_melt = t_melt
        self.N = N
        self.mb_type = mb_type
        self.loop = loop
        self.grad_type = grad_type
        # default rho is 900  kg/m3
        self.rho = cfg.PARAMS['ice_density']

        # Public attrs
        self.hemisphere = gdir.hemisphere
        self.temp_bias = 0.
        self.prcp_bias = 1.
        self.repeat = repeat

        self.SEC_IN_YEAR = SEC_IN_YEAR
        self.SEC_IN_MONTH = SEC_IN_MONTH

        # check if the right climate is used for the right mb_type
        # these checks might be changed if there are more climate datasets
        # available!!!
        # only have daily temperatures for 'ERA5_daily'
        baseline_climate = gdir.get_climate_info()['baseline_climate_source']
        if (self.mb_type == 'mb_real_daily' and
                baseline_climate != 'ERA5_daily'):
            text = ('wrong climate for mb_real_daily, need to do e.g. '
                    'process_era5_daily_data(gd) to enable ERA5_daily')
            raise InvalidParamsError(text)
        # mb_monthly does not work when daily temperatures are used
        if self.mb_type == 'mb_monthly' and baseline_climate == 'ERA5_daily':
            text = ('wrong climate for mb_monthly, need to do e.g.'
                    'oggm.shop.ecmwf.process_ecmwf_data(gd, dataset="ERA5dr")')
            raise InvalidParamsError(text)
        # mb_daily needs temp_std
        if self.mb_type == 'mb_daily' and baseline_climate == 'ERA5_daily':
            text = 'wrong climate for mb_daily, need to do e.g. \
            oggm.shop.ecmwf.process_ecmwf_data(gd, dataset = "ERA5dr")'
            raise InvalidParamsError(text)

        if baseline_climate == 'ERA5_daily':
            input_filesuffix = '_daily'

        # Read climate file
        fpath = gdir.get_filepath(filename, filesuffix=input_filesuffix)

        # used xarray instead of netCDF4, is this slower?
        with xr.open_dataset(fpath) as xr_nc:
            if self.mb_type == 'mb_real_daily' or self.mb_type == 'mb_monthly':
                # even if there is temp_std inside the dataset, we won't use
                # it for these mb_types
                self.temp_std = np.NaN
            else:
                try:
                    self.temp_std = xr_nc['temp_std'].values
                except KeyError:
                    text = ('The applied climate has no temp std, do e.g.'
                            'oggm.shop.ecmwf.process_ecmwf_data'
                            '(gd, dataset="ERA5dr")')

                    raise InvalidParamsError(text)

            # goal is to get self.years/self.months in hydro_years
            if self.mb_type != 'mb_real_daily':
                time = xr_nc.time
                ny, r = divmod(len(time), 12)
                if r != 0:
                    raise ValueError('Climate data should be N full years')
                # This is where we switch to hydro float year format
                # Last year gives the tone of the hydro year
                self.years = np.repeat(np.arange(xr_nc.time[-1].dt.year-ny+1,
                                                 xr_nc.time[-1].dt.year+1), 12)
                self.months = np.tile(np.arange(1, 13), ny)

            elif self.mb_type == 'mb_real_daily':
                # use pandas to convert month/year to hydro_years
                # this has to be done differently than above because not
                # every month, year has the same amount of days
                pd_test = pd.DataFrame(xr_nc.time.to_series().dt.year.values,
                                       columns=['year'])
                pd_test.index = xr_nc.time.to_series().values
                pd_test['month'] = xr_nc.time.to_series().dt.month.values
                pd_test['hydro_year'] = np.NaN
                # get the month where the hydrological month starts
                # as chosen from the gdir climate file
                # default 10 for 'nh', 4 for 'sh'
                hydro_month_start = int(xr_nc.time[0].dt.month.values)
                if hydro_month_start == 1:
                    # hydro_year corresponds to normal year
                    pd_test.loc[pd_test.index.month >= hydro_month_start,
                                'hydro_year'] = pd_test['year']
                else:
                    pd_test.loc[pd_test.index.month < hydro_month_start,
                                'hydro_year'] = pd_test['year']
                    # otherwise, those days with a month>=hydro_month_start
                    # belong to the next hydro_year
                    pd_test.loc[pd_test.index.month >= hydro_month_start,
                                'hydro_year'] = pd_test['year']+1
                # month_hydro is 1 if it is hydro_month_start
                month_hydro = pd_test['month'].values+(12-hydro_month_start+1)
                month_hydro[month_hydro > 12] += -12
                pd_test['hydro_month'] = month_hydro
                pd_test = pd_test.astype('int')
                self.years = pd_test['hydro_year'].values
                ny = self.years[-1] - self.years[0]+1
                self.months = pd_test['hydro_month'].values
            # Read timeseries
            self.temp = xr_nc['temp'].values
            # this is prcp computed by instantiation
            # this changes if prcp_fac is updated (see @property)
            self.prcp = xr_nc['prcp'].values * self._prcp_fac

            # lapse rate (temperature gradient)
            if self.grad_type == 'var' or self.grad_type == 'var_an_cycle':
                try:
                    grad = xr_nc['gradient'].values
                    # Security for stuff that can happen with local gradients
                    g_minmax = temp_local_gradient_bounds

                    # if gradient is not a number, or positive/negative
                    # infinity, use the default gradient
                    grad = np.where(~np.isfinite(grad), default_grad, grad)

                    # if outside boundaries of default -0.009 and above
                    # -0.003 -> use the boundaries instead
                    grad = clip_array(grad, g_minmax[0], g_minmax[1])

                    if self.grad_type == 'var_an_cycle':
                        # if we want constant lapse rates over the years
                        # that change over the annual cycle, but not over time
                        if self.mb_type == 'mb_real_daily':
                            grad_gb = xr_nc['gradient'].groupby('time.month')
                            grad = grad_gb.mean().values
                            g_minmax = temp_local_gradient_bounds

                            # if gradient is not a number, or positive/negative
                            # infinity, use the default gradient
                            grad = np.where(~np.isfinite(grad), default_grad,
                                            grad)

                            # if outside boundaries of default -0.009 and above
                            # -0.003 -> use the boundaries instead
                            grad = clip_array(grad, g_minmax[0], g_minmax[1])

                            stack_grad = grad.reshape(-1, 12)
                            grad = np.tile(stack_grad.mean(axis=0), ny)
                            reps_day1 = xr_nc.time[xr_nc.time.dt.day == 1]
                            reps = reps_day1.dt.daysinmonth
                            grad = np.repeat(grad, reps)

                        else:
                            stack_grad = grad.reshape(-1, 12)
                            grad = np.tile(stack_grad.mean(axis=0), ny)
                except KeyError:
                    text = ('there is no gradient available in chosen climate'
                            'file, try instead e.g. ERA5_daily or ERA5dr e.g.'
                            'oggm.shop.ecmwf.process_ecmwf_data'
                            '(gd, dataset="ERA5dr")')

                    raise InvalidParamsError(text)

            elif self.grad_type == 'cte':
                # if grad_type is chosen cte, we use the default_grad!
                grad = self.prcp * 0 + default_grad
            else:
                raise InvalidParamsError('grad_type can be either cte,'
                                         'var or var_an_cycle')
            self.grad = grad
            self.ref_hgt = xr_nc.ref_hgt  # xr_nc.uncorrected_ref_hgt
            # ref_hgt
            # if climate dataset has been corrected once again
            # or non corrected reference height!
            try:
                self.uncorrected_ref_hgt = xr_nc.uncorrected_ref_hgt
            except:
                self.uncorrected_ref_hgt = xr_nc.ref_hgt
            # xr_nc.ref_hgt

            self.ys = self.years[0] if ys is None else ys
            self.ye = self.years[-1] if ye is None else ye

        self.fpath = fpath

    # copying Cat class idea ;-)
    # https://fabienmaussion.info/scientific_programming/week_10/01-OOP-Part-1.html
    @property
    def prcp_fac(self):
        ''' prints the _prcp_fac
        '''
        return self._prcp_fac

    #def _recompute_prcp(self, new_prcp_fac):



    @prcp_fac.setter
    def prcp_fac(self, new_prcp_fac):
        '''
        '''
        if new_prcp_fac <= 0:
            raise InvalidParamsError('prcp_fac has to be above zero!')
        # attention, prcp_fac should not be called here
        # otherwise there is recursion occurring forever...
        # use new_prcp_fac to not get maximum recusion depth error
        self.prcp = self.prcp * new_prcp_fac / self._prcp_fac

        # update old prcp_fac in order that it can be updated
        # again ...
        self._prcp_fac = new_prcp_fac


    def historical_climate_qc_mod(self, gdir,
                                  # t_solid=0, t_liq=2, t_melt=0, prcp_fac=2.5,
                                  # default_grad=-0.0065,
                                  # templocal_gradient_bounds=[-0.009, -0.003],
                                  climate_qc_months=3,
                                  use_cfg_params=False):
        """"Check the "quality" of climate data and correct it if needed.

        Similar to historical_climate_qc from oggm.core.climate but checks
        that climate that is used in TIModels directly

        This forces the climate data to have at least one month of melt
        per year at the terminus of the glacier (i.e. simply shifting
        temperatures up
        when necessary), and at least one month where accumulation is possible
        at the glacier top (i.e. shifting the temperatures down).

        TODO: why ist here not the temp_bias inside???
        """

        # # Parameters
        # if use_cfg_params:
        #     temp_s = (cfg.PARAMS['temp_all_liq'] +
        # cfg.PARAMS['temp_all_solid'])/2
        #     temp_m = cfg.PARAMS['temp_melt']
        #     default_grad = cfg.PARAMS['temp_default_gradient']
        #     g_minmax = cfg.PARAMS['temp_local_gradient_bounds']
        #     qc_months = cfg.PARAMS['climate_qc_months']
        # else:
        #     temp_s = (t_liq + t_solid) / 2
        #     temp_m = t_melt
        #     default_grad = default_grad
        #     g_minmax = temp_local_gradient_bounds
        # if qc_months == 0:
        #     return

        # Read file
        # if cfg.PARAMS['baseline_climate'] == 'ERA5_daily':
        #     filesuffix = '_daily'
        # fpath = gdir.get_filepath('climate_historical'+filesuffix)
        # igrad = None
        # with utils.ncDataset(fpath) as nc:
        #     # time
        #     # Read timeseries
        #     itemp = nc.variables['temp'][:]
        #     if 'gradient' in nc.variables:
        #         igrad = nc.variables['gradient'][:]
        #         # Security for stuff that can happen with local gradients
        #         igrad = np.where(~np.isfinite(igrad), default_grad, igrad)
        #         igrad = utils.clip_array(igrad, g_minmax[0], g_minmax[1])
        #     ref_hgt = nc.ref_hgt

        # # Default gradient?
        # if igrad is None:
        #     igrad = itemp * 0 + default_grad

        # Parameters (from cfg.PARAMS in OGGM defaul
        fpath = self.fpath
        grad = self.grad
        # get non-corrected quality check
        ref_hgt = self.uncorrected_ref_hgt
        itemp = self.temp
        temp_m = self.t_melt
        temp_s = (self.t_liq + self.t_solid) / 2
        if cfg.PARAMS['baseline_climate'] == 'ERA5_daily':
            # different amount of days per year ...
            d_m = 30
            pass
        else:
            d_m = 1
            ny = len(grad) // 12
            assert ny == len(grad) / 12

        # Geometry data
        fls = gdir.read_pickle('inversion_flowlines')
        heights = np.array([])
        for fl in fls:
            heights = np.append(heights, fl.surface_h)
        top_h = np.max(heights)
        bot_h = np.min(heights)

        # First check - there should be at least "climate_qc_months"
        # month of melt every year
        prev_ref_hgt = ref_hgt
        while True:
            # removed default_grad and uses instead grad!
            ts_bot = itemp + grad * (bot_h - ref_hgt)
            # reshape does not work , because of different amount of days
            # per year ...
            pd_ts = pd.DataFrame({'ts_threshold': ts_bot > temp_m,
                                  'year': self.years})
            ts_bot = pd_ts.groupby('year').sum()['ts_threshold'].values
            # ts_bot = (ts_bot.reshape((ny, 12)) > temp_m).sum(axis=1)
            if np.all(ts_bot >= climate_qc_months * d_m):
                # Ok all good
                break
            # put ref hgt a bit higher so that we warm things a bit
            ref_hgt += 10

        # If we changed this it makes no sense to lower it down again,
        # so resume here:
        if ref_hgt != prev_ref_hgt:
            with utils.ncDataset(fpath, 'a') as nc:
                nc.ref_hgt = ref_hgt
                nc.uncorrected_ref_hgt = prev_ref_hgt
            gdir.add_to_diagnostics('ref_hgt_qc_diff',
                                    int(ref_hgt - prev_ref_hgt))
            # need to save the new ref_hgt
            self.ref_hgt = ref_hgt
            return

        # Second check - there should be at least "climate_qc_months"
        # month of acc every year
        while True:
            # grad instead of default_grad
            ts_top = itemp + grad * (top_h - ref_hgt)
            # reshape does not work , because of different amount of days
            # per year ...
            pd_ts = pd.DataFrame({'ts_threshold': ts_top < temp_s,
                                  'year': self.years})
            ts_top = pd_ts.groupby('year').sum()['ts_threshold'].values
            # ts_top = (ts_top.reshape((ny, 12)) < temp_s).sum(axis=1)
            if np.all(ts_top >= climate_qc_months * d_m):
                # Ok all good
                break
            # put ref hgt a bit lower so that we cold things a bit
            ref_hgt -= 10

        if ref_hgt != prev_ref_hgt:
            with utils.ncDataset(fpath, 'a') as nc:
                nc.ref_hgt = ref_hgt
                nc.uncorrected_ref_hgt = prev_ref_hgt
            gdir.add_to_diagnostics('ref_hgt_qc_diff',
                                    int(ref_hgt - prev_ref_hgt))
            # need to save the new ref_hgt
            self.ref_hgt = ref_hgt
            return

    def _get_climate(self, heights, climate_type, year=None):
        """Climate information at given heights.
        year has to be given as float hydro year from what the month is taken,
        hence year 2000 -> y=2000, m = 1, & year = 2000.09, y=2000, m=2 ...
        which corresponds to the real year 1999 an months October or November
        if hydro year starts in October

        Note that prcp is corrected with the precipitation factor and that
        all other model biases (temp and prcp) are applied.

        same as in OGGM default except that tempformelt is computed by
        self._get_tempformelt

        Parameters
        -------
        heights : np.array or list
            heights along flowline
        climate_type : str
            either 'monthly' or 'annual', if annual floor of year is used,
            if monthly float year is converted into month and year

        Returns
        -------
        (temp, tempformelt, prcp, prcpsol)
        """

        y, m = floatyear_to_date(year)
        if self.repeat:
            y = self.ys + (y - self.ys) % (self.ye - self.ys + 1)
        if y < self.ys or y > self.ye:
            raise ValueError('year {} out of the valid time bounds: '
                             '[{}, {}]'.format(y, self.ys, self.ye))

        if self.mb_type == 'mb_real_daily' or climate_type == 'annual':
            if climate_type == 'annual':
                pok = np.where(self.years == year)[0]
                if len(pok) < 1:
                    raise ValueError('Year {} not in record'.format(int(year)))
            else:
                pok = np.where((self.years == y) & (self.months == m))[0]
                if len(pok) < 28:
                    warnings.warn('something goes wrong with amount of entries\
                                  per month for mb_real_daily')
        else:
            pok = np.where((self.years == y) & (self.months == m))[0][0]
        # Read timeseries
        itemp = self.temp[pok] + self.temp_bias
        iprcp = self.prcp[pok] * self.prcp_bias
        igrad = self.grad[pok]

        # For each height pixel:
        # Compute temp and tempformelt (temperature above melting threshold)
        heights = np.asarray(heights)
        npix = len(heights)
        if self.mb_type == 'mb_real_daily' or climate_type == 'annual':
            grad_temp = np.atleast_2d(igrad).repeat(npix, 0)
            if len(pok) != 12 and self.mb_type != 'mb_real_daily':
                warnings.warn('something goes wrong with amount of entries'
                              'per year')
            grad_temp *= (heights.repeat(len(pok)).reshape(grad_temp.shape) -
                          self.ref_hgt)
            temp2d = np.atleast_2d(itemp).repeat(npix, 0) + grad_temp

            # temp_for_melt is computed separately depending on mb_type
            temp2dformelt = self._get_tempformelt(temp2d, pok)

            # Compute solid precipitation from total precipitation
            prcp = np.atleast_2d(iprcp).repeat(npix, 0)
            fac = 1 - (temp2d - self.t_solid) / (self.t_liq - self.t_solid)
            prcpsol = prcp * clip_array(fac, 0, 1)
            return temp2d, temp2dformelt, prcp, prcpsol

        else:
            temp = np.ones(npix) * itemp + igrad * (heights - self.ref_hgt)

            # temp_for_melt is computed separately depending on mb_type
            tempformelt = self._get_tempformelt(temp, pok)
            prcp = np.ones(npix) * iprcp
            fac = 1 - (temp - self.t_solid) / (self.t_liq - self.t_solid)
            prcpsol = prcp * clip_array(fac, 0, 1)

            return temp, tempformelt, prcp, prcpsol

    def _get_2d_monthly_climate(self, heights, year=None):
        # first get the climate data
        Warning('Attention: this has not been tested enough to be sure that '
                'it works')
        if self.mb_type == 'mb_real_daily':
            return self._get_climate(heights, 'monthly', year=year)
        else:
            raise InvalidParamsError('_get_2d_monthly_climate works only\
                                     with mb_real_daily as mb_type!!!')

    def get_monthly_climate(self, heights, year=None):
        # first get the climate data
        Warning('Attention: this has not been tested enough to be sure that \
                it works')
        if self.mb_type == 'mb_real_daily':
            t, tfmelt, prcp, prcpsol = self._get_climate(heights, 'monthly',
                                                         year=year)
            return (t.mean(axis=1), tfmelt.sum(axis=1),
                    prcp.sum(axis=1), prcpsol.sum(axis=1))
        else:
            return self._get_climate(heights, 'monthly', year=year)
            # if it is mb_real_daily, the data has daily resolution

    def _get_2d_annual_climate(self, heights, year):
        return self._get_climate(heights, 'annual', year=year)

    # If I also want to use this outside of the class because
    # (e.g. in climate.py), I have to change this again and remove the self...
    # and somehow there is aproblem if I put not self in
    # _get_tempformelt when it is inside the class

    def _get_tempformelt(self, temp, pok):
        """ Helper function to compute tempformelt to avoid code duplication
        in get_monthly_climate() and _get2d_annual_climate()

        If using this again outside of this class, need to remove the "self",
        such as for 'mb_climate_on_height' in climate.py, that has no self....
        (would need to change temp, t_melt ,temp_std, mb_type, N, loop)

        Input: stuff that is different for the different methods
            temp: temperature time series
            pok: indices of time series

        Returns
        -------
        (tempformelt)
        """

        tempformelt_without_std = temp - self.t_melt

        # computations change only if 'mb_daily' as mb_type!
        if self.mb_type == 'mb_monthly' or self.mb_type == 'mb_real_daily':
            tempformelt = tempformelt_without_std
        elif self.mb_type == 'mb_daily':

            itemp_std = self.temp_std[pok]

            tempformelt_with_std = np.full(np.shape(tempformelt_without_std),
                                           np.NaN)
            # matrix with N values that are distributed around 0
            # showing how much fake 'daily' values vary from the mean
            z_scores_mean = stats.norm.ppf(np.arange(1/self.N-1/(2*self.N),
                                                     1, 1/self.N))

            z_std = np.matmul(np.atleast_2d(z_scores_mean).T,
                              np.atleast_2d(itemp_std))

            # there are two possibilities,
            # not using the loop is most of the times faster
            if self.loop is False:
                # without the loop: but not much faster ..
                tempformelt_daily = np.atleast_3d(tempformelt_without_std).T + \
                                    np.atleast_3d(z_std)
                clip_min(tempformelt_daily, 0, out=tempformelt_daily)
                tempformelt_with_std = tempformelt_daily.mean(axis=0).T
            else:
                shape_tfm = np.shape(tempformelt_without_std)
                tempformelt_with_std = np.full(shape_tfm, np.NaN)
                for h in np.arange(0, np.shape(tempformelt_without_std)[0]):
                    h_tfm_daily_ = np.atleast_2d(tempformelt_without_std[h, :])
                    h_tempformelt_daily = h_tfm_daily_ + z_std
                    clip_min(h_tempformelt_daily, 0, out=h_tempformelt_daily)
                    h_tempformelt_monthly = h_tempformelt_daily.mean(axis=0)
                    tempformelt_with_std[h, :] = h_tempformelt_monthly
            tempformelt = tempformelt_with_std

        else:
            raise InvalidParamsError('mb_type can only be "mb_monthly,\
                                     mb_daily or mb_real_daily" ')
        #  replace all values below zero to zero
        clip_min(tempformelt, 0, out=tempformelt)

        return tempformelt

    # same as in OGGM default
    def get_annual_climate(self, heights, year=None):
        """Annual climate information at given heights.

        Note that prcp is corrected with the precipitation factor and that
        all other model biases (temp and prcp) are applied.

        Returns
        -------
        (temp, tempformelt, prcp, prcpsol)
        """
        t, tfmelt, prcp, prcpsol = self._get_2d_annual_climate(heights, year)
        return (t.mean(axis=1), tfmelt.sum(axis=1),
                prcp.sum(axis=1), prcpsol.sum(axis=1))

    def get_monthly_mb(self, heights, year=None, **kwargs):
        """ computes annual mass balance in kg /m² /month

        Attention year is here in hydro float year

        """
        # get_monthly_mb and get_annual_mb are only different
        # to OGGM default for mb_real_daily

        if self.mb_type == 'mb_real_daily':
            # get 2D values, dependencies on height and time (days)
            out = self._get_2d_monthly_climate(heights, year)
            _, temp2dformelt, _, prcpsol = out
            # 1 / (days per month)
            fact = 1/len(prcpsol.T)
            # to have the same unit of melt_f, which is
            # the monthly temperature sensitivity (kg /m² /mth /K),
            mb_daily = prcpsol - self.melt_f * temp2dformelt * fact

            mb_month = np.sum(mb_daily, axis=1)
            # more correct than using a mean value for days in a month
            warnings.warn('get_monthly_mb has not been tested enough,'
                          ' there might be a problem with SEC_IN_MONTH'
                          '.., see test_monthly_glacier_massbalance()')

        else:
            # get 1D values for each height, no dependency on days
            _, tmelt, _, prcpsol = self.get_monthly_climate(heights, year=year)
            mb_month = prcpsol - self.melt_f * tmelt

        # residual is in general, so SEC_IN_MONTH .. can be used
        mb_month -= self.residual * self.SEC_IN_MONTH / self.SEC_IN_YEAR
        # this is for mb_daily otherwise it gives the wrong shape
        mb_month = mb_month.flatten()
        # instead of SEC_IN_MONTH, use instead len(prcpsol.T)==daysinmonth
        return mb_month / self.SEC_IN_MONTH / self.rho

    def get_annual_mb(self, heights, year=None, **kwargs):
        """ computes annual mass balance in kg /m² /second """
        # get_monthly_mb and get_annual_mb are only different
        # to OGGM default for mb_real_daily

        _, temp2dformelt, _, prcpsol = self._get_2d_annual_climate(heights,
                                                                   year)
        # *12/daysofthisyear in order to have the same unit of melt_f, which
        # is the monthly temperature sensitivity (kg /m² /mth /K),
        if self.mb_type == 'mb_real_daily':
            # in this case we have the temp2dformelt for each day
            # but self.melt_f is in per month -> divide trough days/month
            # more correct than using a mean value for days in a year
            fact = 12/len(prcpsol.T)
        else:
            fact = 1
        mb_annual = np.sum(prcpsol - self.melt_f * temp2dformelt*fact,
                           axis=1)
        return (mb_annual - self.residual) / self.SEC_IN_YEAR / self.rho
