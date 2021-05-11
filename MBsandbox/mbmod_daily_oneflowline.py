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
from oggm.cfg import SEC_IN_YEAR, SEC_IN_MONTH, SEC_IN_DAY
from oggm.utils import (floatyear_to_date, date_to_floatyear, ncDataset,
                        clip_min, clip_array)
from oggm.utils._funcs import haversine
from oggm.exceptions import InvalidParamsError, InvalidWorkflowError
from oggm.shop.ecmwf import get_ecmwf_file, BASENAMES
from oggm.core.massbalance import MassBalanceModel

# Module logger
log = logging.getLogger(__name__)

ECMWF_SERVER = 'https://cluster.klima.uni-bremen.de/~oggm/climate/'
# ECMWF_SERVER1 = 'https://cluster.klima.uni-bremen.de/~lschuster/'



# %%

# add era5_daily dataset, this only works with process_era5_daily_data
BASENAMES['ERA5_daily'] = {
        'inv': 'era5/daily/v1.0/era5_glacier_invariant_flat.nc',
        'tmp': 'era5/daily/v1.0/era5_daily_t2m_1979-2018_flat.nc'
        # only glacier-relevant gridpoints included!
        }

BASENAMES['WFDE5_daily_cru'] = {
    'inv': 'wfde5_cru/daily/v1.1/wfde5_cru_glacier_invariant_flat.nc',
    'tmp': 'wfde5_cru/daily/v1.1/wfde5_cru_tmp_1979-2018_flat.nc',
    'prcp': 'wfde5_cru/daily/v1.1/wfde5_cru_prcp_1979-2018_flat.nc',
    }



# this could be used in general
def write_climate_file(gdir, time, prcp, temp,
                       ref_pix_hgt, ref_pix_lon, ref_pix_lat,
                       gradient=None, temp_std=None,
                       time_unit=None, calendar=None,
                       source=None, file_name='climate_historical',
                       filesuffix='',
                       temporal_resol='monthly'):
    """Creates a netCDF4 file with climate data timeseries.

    Parameters
    ----------
    gdir:
        glacier directory
    time : ndarray
        the time array, in a format understood by netCDF4
    prcp : ndarray
        the precipitation array (unit: 'kg m-2')
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
    temporal_resol : str
        temporal resolution of climate file, either monthly (default) or
        daily
    """

    if source == 'ERA5_daily' and filesuffix == '':
        raise InvalidParamsError("filesuffix should be '_daily' for ERA5_daily"
                                 "file_name climate_historical is normally"
                                 "monthly data")
    elif (source == 'WFDE5_daily_cru' and filesuffix == ''
          and temporal_resol == 'daily'):
        raise InvalidParamsError("filesuffix should be '_daily' for WFDE5_daily_cru"
                                 "if daily chosen as temporal_resol"
                                 "file_name climate_historical is normally"
                                 "monthly data")
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
        if (len(prcp) > (nc.hydro_yr_1 - nc.hydro_yr_0 + 1) * 28 * 12 and
            temporal_resol == 'daily'):
            if source == 'ERA5_daily':
                v.long_name = ("total daily precipitation amount, "
                               "assumed same for each day of month")
            elif source == 'WFDE5_daily_cru':
                v.long_name = ("total daily precipitation amount"
                               "sum of snowfall and rainfall")
        elif (len(prcp) == (nc.hydro_yr_1 - nc.hydro_yr_0 + 1) * 12
              and temporal_resol == 'monthly'):
            v.long_name = 'total monthly precipitation amount'
        else:
            # v.long_name = 'total monthly precipitation amount'
            raise InvalidParamsError('there is a conflict in the'
                                     'prcp timeseries, '
                                     'please check temporal_resol')
            # warnings.warn("there might be a conflict in the prcp timeseries,"
            #              "please check!")

        v[:] = prcp

        v = nc.createVariable('temp', 'f4', ('time',), zlib=zlib)
        v.units = 'degC'
        if ((source == 'ERA5_daily' or source == 'WFDE5_daily_cru') and
            len(temp) > (y1 - y0) * 28 * 12 and temporal_resol == 'daily'):
            v.long_name = '2m daily temperature at height ref_hgt'
        elif source == 'ERA5_daily' and len(temp) <= (y1 - y0) * 30 * 12:
            raise InvalidParamsError('if the climate dataset (here source)'
                                     'is ERA5_daily, temperatures should be in'
                                     'daily resolution, please check or set'
                                     'set source to another climate dataset')
        elif (source == 'WFDE5_daily_cru' and temporal_resol == 'monthly' and
              len(temp) > (y1 - y0) * 28 * 12):
            raise InvalidParamsError('something wrong in the implementation')
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
def process_wfde5_data(gdir, y0=None, y1=None, temporal_resol='daily',
                       output_filesuffix='_daily_WFDE5_CRU',
                       cluster = True,
                       climate_path='/home/lilianschuster/Schreibtisch/PhD/WP0_bayesian/WPx_WFDE5/'):
    """ TODO: let it work on the cluster first by giving there the right path...

    Processes and writes the WFDE5 daily baseline climate data for a glacier.
    into climate_historical_daily.nc

    Extracts the nearest timeseries and writes everything to a NetCDF file.
    This uses only the WFDE5 daily temperatures. The temperature lapse
    rate are used from ERA5dr.

    TODO: see _verified_download_helper no known hash for
    wfde5_daily_t2m_1979-2018_flat.nc and wfde5_glacier_invariant_flat
    ----------
    y0 : int
        the starting year of the timeseries to write. The default is to take
        the entire time period available in the file, but with this kwarg
        you can shorten it (to save space or to crop bad data)
    y1 : int
        the starting year of the timeseries to write. The default is to take
        the entire time period available in the file, but with this kwarg
        you can shorten it (to save space or to crop bad data)
    temporal_resol : str
        uses either daily (default) or monthly data
    output_filesuffix : str
        this add a suffix to the output file (useful to avoid overwriting
        previous experiments)
    cluster : bool
        default is False, if this is run on the cluster, set it to True,
        because we do not need to download the files

    """

    # wfde5_daily for temperature and precipitation
    dataset = 'WFDE5_daily_cru'
    # but need temperature lapse rates from ERA5
    dataset_othervars = 'ERA5dr'

    # get the central longitude/latitudes of the glacier
    lon = gdir.cenlon + 360 if gdir.cenlon < 0 else gdir.cenlon
    lat = gdir.cenlat

    # cluster_path = '/home/www/oggm/climate/'
    # cluster_path = '/home/users/lschuster/'
    if cluster:
        path_tmp = climate_path + BASENAMES[dataset]['tmp']
        path_prcp = climate_path + BASENAMES[dataset]['prcp']
        path_inv = climate_path + BASENAMES[dataset]['inv']

    else:
        raise InvalidParamsError('not yet implemented...')
        path_tmp = get_ecmwf_file(dataset, 'tmp')
        path_prcp = get_ecmwf_file(dataset, 'pre')
        path_inv = get_ecmwf_file(dataset, 'inv')



    # Use xarray to read the data
    # would go faster with netCDF -.-
    # first temperature dataset
    with xr.open_dataset(path_tmp) as ds:
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

        # if we want to use monthly mean tempeatures of wfde5 and
        # standard deviation of daily temperature:
        if temporal_resol == 'monthly':
            Tair_std = ds.resample(time='MS').std().Tair
            temp_std = Tair_std.data
            ds = ds.resample(time='MS').mean()
            ds['longitude'] = ds.longitude.isel(time=0)
            ds['latitude'] = ds.latitude.isel(time=0)
        elif temporal_resol == 'daily':
            temp_std = None
        else:
            raise InvalidParamsError('temporal_resol can only be monthly'
                                     'or daily!')


        # temperature should be in degree Celsius for the glacier climate files
        temp = ds['Tair'].data - 273.15
        time = ds.time.data

        ref_lon = float(ds['longitude'])
        ref_lat = float(ds['latitude'])

        ref_lon = ref_lon - 360 if ref_lon > 180 else ref_lon

    # precipitation: similar ar temperature
    with xr.open_dataset(path_prcp) as ds:
        assert ds.longitude.min() >= 0

        yrs = ds['time.year'].data
        y0 = yrs[0] if y0 is None else y0
        y1 = yrs[-1] if y1 is None else y1
        # Attention here we take the same y0 and y1 as given from the
        # daily tmp dataset (goes till end of 2018)

        # attention if daily data, need endday!!!
        ds = ds.sel(time=slice('{}-{:02d}-01'.format(y0, sm),
                               '{}-{:02d}-{}'.format(y1, em, end_day)))
        try:
            # wfde5 prcp is also flattened
            c = (ds.longitude - lon)**2 + (ds.latitude - lat)**2
            ds = ds.isel(points=c.argmin())
        except ValueError:
            # this should not occur
            ds = ds.sel(longitude=lon, latitude=lat, method='nearest')

        # if we want to use monthly summed up wfde5 precipitation:
        if temporal_resol == 'monthly':
            ds = ds.resample(time='MS').sum()
        elif temporal_resol == 'daily':
            pass
        # the prcp data of wfde5 is in kg m-2 day-1 ~ mm/day
        # or in kg m-2 month-1 ~ mm/month
        prcp = ds['tp'].data  # * 1000
        # just assume that precipitation is every day the same:
        # prcp in daily reso prcp = np.repeat(prcp, ds['time.daysinmonth'])
        # Attention the unit is now prcp per day
        # (not per month as in OGGM default:
        # prcp = ds['tp'].data * 1000 * ds['time.daysinmonth']

    # wfde5 invariant file
    with xr.open_dataset(path_inv) as ds:
        assert ds.longitude.min() >= 0
        ds = ds.isel(time=0)
        try:
            # Flattened wfde5_inv (only possibility at the moment)
            c = (ds.longitude - lon)**2 + (ds.latitude - lat)**2
            ds = ds.isel(points=c.argmin())
        except ValueError:
            # this should not occur
            ds = ds.sel(longitude=lon, latitude=lat, method='nearest')

        # wfde5 inv ASurf/hgt is already in hgt coordinates
        # G = cfg.G  # 9.80665
        hgt = ds['ASurf'].data  # / G


    # here we need to use the ERA5dr data ...
    # there are no lapse rates from wfde5 !!!
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
        if temporal_resol == 'monthly':
            pass
        elif temporal_resol == 'daily':
            # gradient needs to be restructured to have values for each day
            # when wfde5_daily is applied
            gradient = np.repeat(gradient, ds['time.daysinmonth'])
            # assume same gradient for each day

    if temporal_resol == 'monthly':
        if output_filesuffix == '_daily':
            output_filesuffix = ''
        dataset = 'WFDE5_monthly_cru'
    elif temporal_resol == 'daily' and output_filesuffix == '':
        output_filesuffix = '_daily'
    # OK, ready to write
    write_climate_file(gdir, time, prcp, temp, hgt, ref_lon, ref_lat,
                       filesuffix=output_filesuffix,
                       temporal_resol=temporal_resol,
                       gradient=gradient,
                       temp_std=temp_std,
                       source=dataset,
                       file_name='climate_historical')
    # This is now a new function, maybe it would better to make a general
    # process_daily_data function where ERA5_daily and WFDE5_daily 
    # but is used, so far, only for ERA5_daily as source dataset ..


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
                       temporal_resol='daily',
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
                 repeat=False, ys=None, ye=None,
                 t_solid=0, t_liq=2, t_melt=0, prcp_fac=2.5,
                 default_grad=-0.0065,
                 temp_local_gradient_bounds=[-0.009, -0.003],
                 # check_climate=True,
                 SEC_IN_YEAR=SEC_IN_YEAR,
                 SEC_IN_MONTH=SEC_IN_MONTH,
                 SEC_IN_DAY=SEC_IN_DAY,
                 baseline_climate=None,
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
        prcp_fac : float, >0
            multiplicative precipitation correction factor (default 2.5)
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
        #  to allow prcp_fac to be changed after instantiation
        #  prescribe the prcp_fac as it is instantiated
        self._prcp_fac = prcp_fac
        # same for temp bias
        self._temp_bias = 0.

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
        self.repeat = repeat

        self.SEC_IN_YEAR = SEC_IN_YEAR
        self.SEC_IN_MONTH = SEC_IN_MONTH
        self.SEC_IN_DAY = SEC_IN_DAY
        # what is this???
        self.valid_bounds = [-1e4, 2e4]  # in m


        # check if the right climate is used for the right mb_type
        # these checks might be changed if there are more climate datasets
        # available!!!
        # only have daily temperatures for 'ERA5_daily'
        if baseline_climate == None:
            baseline_climate = gdir.get_climate_info()['baseline_climate_source']

        if mb_type != 'mb_real_daily':
            # cfg.PARAMS['baseline_climate'] = 'ERA5dr'
            input_filesuffix = '_monthly_{}'.format(baseline_climate)
        else:
            # this is just the climate "type"
            # cfg.PARAMS['baseline_climate'] = 'ERA5_daily'
            input_filesuffix = '_daily_{}'.format(baseline_climate)

        self._input_filesuffix = input_filesuffix
        #if (self.mb_type == 'mb_real_daily' and
        #    (baseline_climate != 'ERA5dr' and
         #    baseline_climate != 'WFDE5_CRU')):
         #   text = ('wrong climate for mb_real_daily, need to do e.g. '
        #            'process_era5_daily_data(gd) to produce daily_ERA5dr'
        #            'or process_wfde5_data(gd) for daily_WFDE5_CRU')
        #    raise InvalidParamsError(text)
        # mb_monthly does not work when daily temperatures are used
        if self.mb_type == 'mb_monthly' and input_filesuffix == 'daily_ERA5dr':
            text = ('wrong climate for mb_monthly, need to do e.g.'
                    'oggm.shop.ecmwf.process_ecmwf_data(gd, dataset="ERA5dr")')
            raise InvalidParamsError(text)
        # mb_daily needs temp_std
        if self.mb_type == 'mb_daily' and input_filesuffix == 'daily_ERA5dr':
            text = 'wrong climate for mb_daily, need to do e.g. \
            oggm.shop.ecmwf.process_ecmwf_data(gd, dataset = "ERA5dr")'
            raise InvalidParamsError(text)

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
                    self.temp_std = xr_nc['temp_std'].values.astype(np.float64)
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
            # Read timeseries and correct it
            self.temp = xr_nc['temp'].values.astype(np.float64) + self._temp_bias
            # this is prcp computed by instantiation
            # this changes if prcp_fac is updated (see @property)
            self.prcp = xr_nc['prcp'].values.astype(np.float64) * self._prcp_fac


            # lapse rate (temperature gradient)
            if self.grad_type == 'var' or self.grad_type == 'var_an_cycle':
                try:
                    grad = xr_nc['gradient'].values.astype(np.float64)
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

    @prcp_fac.setter
    def prcp_fac(self, new_prcp_fac):
        '''
        '''
        if new_prcp_fac <= 0:
            raise InvalidParamsError('prcp_fac has to be above zero!')
        # attention, prcp_fac should not be called here
        # otherwise there is recursion occurring forever...
        # use new_prcp_fac to not get maximum recusion depth error
        self.prcp *= new_prcp_fac / self._prcp_fac
        # update old prcp_fac in order that it can be updated
        # again ...
        self._prcp_fac = new_prcp_fac

    # same for temp_bias:
    @property
    def temp_bias(self):
        return self._temp_bias

    @temp_bias.setter
    def temp_bias(self, new_temp_bias):
        self.temp += new_temp_bias - self._temp_bias
        # update old temp_bias in order that it can be updated again ...
        self._temp_bias = new_temp_bias

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

        This has a similar effect as introducing a temperature bias
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
        if self.temp_bias != 0:
            raise InvalidParamsError('either use no temp_bias or do no quality'
                                     'check corrections, as they have the '
                                     'same effects!')
        fpath = self.fpath
        grad = self.grad
        # get non-corrected quality check
        ref_hgt = self.uncorrected_ref_hgt
        itemp = self.temp
        temp_m = self.t_melt
        temp_s = (self.t_liq + self.t_solid) / 2
        if ('daily' in self._input_filesuffix):
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
                #if type(year) == float:
                #    raise InvalidParamsError('')
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
        # (already temperature bias and precipitation factor corrected!)
        itemp = self.temp[pok]
        iprcp = self.prcp[pok]
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

    def get_daily_climate(self, heights, year = None):
        raise NotImplementedError('look at _get_2d_daily_climate instead')

    def _get_2d_annual_climate(self, heights, year):
        return self._get_climate(heights, 'annual', year=year)

    def _get_2d_daily_climate(self, heights, year = None):
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
        """ computes annual mass balance in kg /m² /second

        Attention year is here in hydro float year

        year has to be given as float hydro year from what the month is taken,
        hence year 2000 -> y=2000, m = 1, & year = 2000.09, y=2000, m=2 ...
        which corresponds to the real year 1999 an months October or November
        if hydro year starts in October
        """
        # get_monthly_mb and get_annual_mb are only different
        # to OGGM default for mb_real_daily

        if self.mb_type == 'mb_real_daily':
            # get 2D values, dependencies on height and time (days)
            out = self._get_2d_monthly_climate(heights, year)
            _, temp2dformelt, _, prcpsol = out
            #(days per month)
            dom = 365.25/12  # len(prcpsol.T)
            # attention, I should not use the days of years as the melt_f is
            # per month ~mean days of that year 12/daysofyear
            # to have the same unit of melt_f, which is
            # the monthly temperature sensitivity (kg /m² /mth /K),
            mb_daily = prcpsol - (self.melt_f/dom) * temp2dformelt

            mb_month = np.sum(mb_daily, axis=1)
            # more correct than using a mean value for days in a month
            warnings.warn('get_monthly_mb has not been tested enough,'
                          ' there might be a problem with SEC_IN_MONTH'
                          '.., see test_monthly_glacier_massbalance()')

        else:
            # get 1D values for each height, no dependency on days
            _, tmelt, _, prcpsol = self.get_monthly_climate(heights, year=year)
            mb_month = prcpsol - self.melt_f * tmelt

        # residual is in mm w.e per year, so SEC_IN_MONTH .. but mb_month
        # shoud be per month!
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
            fact = 12/365.25
            # len(prcpsol.T): make it more consistent as melt_f is described
            # per month independent of which month it is ...
        else:
            fact = 1  # eventually correct here with 365.25
        mb_annual = np.sum(prcpsol - self.melt_f * temp2dformelt*fact,
                           axis=1)
        return (mb_annual - self.residual) / self.SEC_IN_YEAR / self.rho

    def get_daily_mb(self, heights, year = None):
                     #, m = None,
                     #float_year=None, **kwargs):
        """computes daily mass balance in kg/m2/second

        year has to be given as float hydro year from what the month is taken,
        hence year 2000 -> y=2000, m = 1, & year = 2000.09, y=2000, m=2 ...
        which corresponds to the real year 1999 an months October or November
        if hydro year starts in October

        """

        # todo: make this more user friendly
        if type(year)==float:
            raise InvalidParamsError('here year has to be the integer year')
        else:
            pass
        #if y==None:
        #    year = date_to_floatyear(y, m)
        ##elif y==None and m!=None:
        #    raise InvalidParamsError('give y or only give float_year'
        #                             'and no month m')
        #else:
        #    year = float_year
        if self.mb_type == 'mb_real_daily':
            # get 2D values, dependencies on height and time (days)
            out = self._get_2d_daily_climate(heights, year)
            _, temp2dformelt, _, prcpsol = out
            # days of year
            doy = 365.25 #len(prcpsol.T)
            # assert doy > 360
            # to have the same unit of melt_f, which is
            # the monthly temperature sensitivity (kg /m² /mth /K),
            melt_f_daily = self.melt_f * 12/doy
            mb_daily = prcpsol - melt_f_daily * temp2dformelt

            # mb_month = np.sum(mb_daily, axis=1)
            # more correct than using a mean value for days in a month
            warnings.warn('get_daily_mb has not been tested enough,')

            # residual is in mm w.e per year, so SEC_IN_MONTH .. but mb_daily
            # is per day!
            mb_daily -= self.residual * self.SEC_IN_DAY / self.SEC_IN_YEAR
            # this is for mb_daily otherwise it gives the wrong shape
            # mb_daily = mb_month.flatten()
            # instead of SEC_IN_MONTH, use instead len(prcpsol.T)==daysinmonth
            return mb_daily / self.SEC_IN_DAY / self.rho
        else:
            raise InvalidParamsError('get_daily_mb works only with'
                                     'mb_real_daily as mb_type!')

    def get_specific_daily_mb(self, heights=None, widths=None, year=None):
        " returns specific daily mass balance in kg m-2 day "
        if len(np.atleast_1d(year)) > 1:
            out = [self.get_specific_daily_mb(heights=heights, widths=widths,
                                        year=yr) for yr in year]
            return np.asarray(out)

        mb = self.get_daily_mb(heights, year=year)
        spec_mb = np.average(mb * self.rho * SEC_IN_DAY, weights=widths, axis=0)
        assert len(spec_mb) > 360
        return spec_mb

# copy of MultipleFlowlineMassBalance that works with TIModel
class MultipleFlowlineMassBalance_TIModel(MassBalanceModel):
    """ TODO: adapt this: main changes are mu_star -> melt_f
    Handle mass-balance at the glacier level instead of flowline level.

    Convenience class doing not much more than wrapping a list of mass-balance
    models, one for each flowline.

    This is useful for real-case studies, where each flowline might have a
    different mu*.

    Attributes
    ----------
    fls : list
        list of flowline objects
    mb_models : list
        list of mass-balance objects
    """

    def __init__(self, gdir, fls=None, melt_f=None, prcp_fac=None,
                 mb_model_class=TIModel, use_inversion_flowlines=False,
                 input_filesuffix='', bias=0,
                 **kwargs):
        """Initialize.

        Parameters
        ----------
        gdir : GlacierDirectory
            the glacier directory
        mu_star : float or list of floats, optional
            set to the alternative value of mu* you want to use
            (the default is to use the calibrated value). Give a list of values
            for flowline-specific mu*
        fls : list, optional
            list of flowline objects to use (defaults to 'model_flowlines',
            and if not available, to 'inversion_flowlines')
        mb_model_class : class, optional
            the mass-balance model to use (e.g. PastMassBalance,
            ConstantMassBalance...)
        use_inversion_flowlines: bool, optional
            if True 'inversion_flowlines' instead of 'model_flowlines' will be
            used.
        input_filesuffix : str
            the file suffix of the input climate file
        bias : float, optional
            set to the alternative value of the calibration bias [mm we yr-1]
            you want to use (the default is to use the calibrated value)
            Note that this bias is *substracted* from the computed MB. Indeed:
            BIAS = MODEL_MB - REFERENCE_MB.
        kwargs : kwargs to pass to mb_model_class
        """

        # Read in the flowlines
        if use_inversion_flowlines:
            fls = gdir.read_pickle('inversion_flowlines')

        if fls is None:
            try:
                fls = gdir.read_pickle('model_flowlines')
            except FileNotFoundError:
                raise InvalidWorkflowError('Need a valid `model_flowlines` '
                                           'file. If you explicitly want to '
                                           'use `inversion_flowlines`, set '
                                           'use_inversion_flowlines=True.')

        self.fls = fls
        _y0 = kwargs.get('y0', None)


        # Initialise the mb models
        self.flowline_mb_models = []
        for fl in self.fls:
            # Merged glaciers will need different climate files, use filesuffix
            if (fl.rgi_id is not None) and (fl.rgi_id != gdir.rgi_id):
                rgi_filesuffix = '_' + fl.rgi_id + input_filesuffix
            else:
                rgi_filesuffix = input_filesuffix

            # merged glaciers also have a different MB bias from calibration
            if ((bias is None) and cfg.PARAMS['use_bias_for_run'] and
                    (fl.rgi_id != gdir.rgi_id)):
                df = gdir.read_json('local_mustar', filesuffix='_' + fl.rgi_id)
                fl_bias = df['bias']
            else:
                fl_bias = bias

            # Constant and RandomMassBalance need y0 if not provided
            #if (issubclass(mb_model_class, RandomMassBalance) or
            #    issubclass(mb_model_class, ConstantMassBalance)) and (
            #        fl.rgi_id != gdir.rgi_id) and (_y0 is None):#

            #    df = gdir.read_json('local_mustar', filesuffix='_' + fl.rgi_id)
            #    kwargs['y0'] = df['t_star']

            if (issubclass(mb_model_class, TIModel)):
                self.flowline_mb_models.append(
                    mb_model_class(gdir, melt_f, prcp_fac = prcp_fac,
                                   residual=fl_bias, baseline_climate=rgi_filesuffix,
                                    **kwargs))
            else:
                self.flowline_mb_models.append(
                    mb_model_class(gdir, mu_star=fl.mu_star, bias=fl_bias,
                                   input_filesuffix=rgi_filesuffix, **kwargs))

        self.valid_bounds = self.flowline_mb_models[-1].valid_bounds
        self.hemisphere = gdir.hemisphere

    @property
    def temp_bias(self):
        """Temperature bias to add to the original series."""
        return self.flowline_mb_models[0].temp_bias

    @temp_bias.setter
    def temp_bias(self, value):
        """Temperature bias to add to the original series."""
        for mbmod in self.flowline_mb_models:
            mbmod.temp_bias = value

    @property
    def prcp_fac(self):
        """Precipitation factor to apply to the original series."""
        return self.flowline_mb_models[0].prcp_fac

    @prcp_fac.setter
    def prcp_fac(self, value):
        """Precipitation factor to apply to the original series."""
        for mbmod in self.flowline_mb_models:
            mbmod.prcp_fac = value

    @property
    def bias(self):
        """Residual bias to apply to the original series."""
        return self.flowline_mb_models[0].bias

    @bias.setter
    def bias(self, value):
        """Residual bias to apply to the original series."""
        for mbmod in self.flowline_mb_models:
            mbmod.bias = value

    def get_monthly_mb(self, heights, year=None, fl_id=None, **kwargs):

        if fl_id is None:
            raise ValueError('`fl_id` is required for '
                             'MultipleFlowlineMassBalance!')

        return self.flowline_mb_models[fl_id].get_monthly_mb(heights,
                                                             year=year)

    def get_annual_mb(self, heights, year=None, fl_id=None, **kwargs):

        if fl_id is None:
            raise ValueError('`fl_id` is required for '
                             'MultipleFlowlineMassBalance!')

        return self.flowline_mb_models[fl_id].get_annual_mb(heights,
                                                            year=year)

    def get_annual_mb_on_flowlines(self, fls=None, year=None):
        """Get the MB on all points of the glacier at once.

        Parameters
        ----------
        fls: list, optional
            the list of flowlines to get the mass-balance from. Defaults
            to self.fls
        year: float, optional
            the time (in the "floating year" convention)
        Returns
        -------
        Tuple of (heights, widths, mass_balance) 1D arrays
        """

        if fls is None:
            fls = self.fls

        heights = []
        widths = []
        mbs = []
        for i, fl in enumerate(fls):
            h = fl.surface_h
            heights = np.append(heights, h)
            widths = np.append(widths, fl.widths)
            mbs = np.append(mbs, self.get_annual_mb(h, year=year, fl_id=i))

        return heights, widths, mbs

    def get_specific_mb(self, heights=None, widths=None, fls=None,
                        year=None):

        if heights is not None or widths is not None:
            raise ValueError('`heights` and `widths` kwargs do not work with '
                             'MultipleFlowlineMassBalance!')

        if fls is None:
            fls = self.fls

        if len(np.atleast_1d(year)) > 1:
            out = [self.get_specific_mb(fls=fls, year=yr) for yr in year]
            return np.asarray(out)

        mbs = []
        widths = []
        for i, (fl, mb_mod) in enumerate(zip(self.fls, self.flowline_mb_models)):
            _widths = fl.widths
            try:
                # For rect and parabola don't compute spec mb
                _widths = np.where(fl.thick > 0, _widths, 0)
            except AttributeError:
                pass
            widths = np.append(widths, _widths)
            mb = mb_mod.get_annual_mb(fl.surface_h, year=year, fls=fls, fl_id=i)
            mbs = np.append(mbs, mb * SEC_IN_YEAR * mb_mod.rho)

        return np.average(mbs, weights=widths)

    def get_ela(self, year=None, **kwargs):

        # ELA here is not without ambiguity.
        # We compute a mean weighted by area.

        if len(np.atleast_1d(year)) > 1:
            return np.asarray([self.get_ela(year=yr) for yr in year])

        elas = []
        areas = []
        for fl_id, (fl, mb_mod) in enumerate(zip(self.fls,
                                                 self.flowline_mb_models)):
            elas = np.append(elas, mb_mod.get_ela(year=year, fl_id=fl_id,
                                                  fls=self.fls))
            areas = np.append(areas, np.sum(fl.widths))

        return np.average(elas, weights=areas)


@entity_task(log)
def fixed_geometry_mass_balance_TIModel(gdir, ys=None, ye=None, years=None,
                                monthly_step=False,
                                use_inversion_flowlines=True,
                                climate_filename='climate_historical',
                                climate_input_filesuffix='',
                                ds_gcm = None,
                                **kwargs):
    """Computes the mass-balance with climate input from e.g. CRU or a GCM.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory to process
    ys : int
        start year of the model run (default: from the climate file)
        date)
    ye : int
        end year of the model run (default: from the climate file)
    years : array of ints
        override ys and ye with the years of your choice
    monthly_step : bool
        whether to store the diagnostic data at a monthly time step or not
        (default is yearly)
    use_inversion_flowlines : bool
        whether to use the inversion flowlines or the model flowlines
    climate_filename : str
        name of the climate file, e.g. 'climate_historical' (default) or
        'gcm_data'
    climate_input_filesuffix: str
        filesuffix for the input climate file
    **kwargs:
        added to MultipleFlowlineMassBalance_TIModel
    """

    if monthly_step:
        raise NotImplementedError('monthly_step not implemented yet')
    if ds_gcm != None:
        melt_f = ds_gcm.sel(rgi_id=gdir.rgi_id).melt_f.values
        pf = ds_gcm.sel(rgi_id=gdir.rgi_id).pf.values

        mb = MultipleFlowlineMassBalance_TIModel(gdir, mb_model_class=TIModel,
                                                 filename=climate_filename,
                                                 use_inversion_flowlines=use_inversion_flowlines,
                                                 input_filesuffix=climate_input_filesuffix,
                                                 melt_f=melt_f, prcp_fac=pf,
                                                 **kwargs)
    else:
        mb = MultipleFlowlineMassBalance_TIModel(gdir, mb_model_class=TIModel,
                                     filename=climate_filename,
                                     use_inversion_flowlines=use_inversion_flowlines,
                                     input_filesuffix=climate_input_filesuffix,
                                     **kwargs)

    if years is None:
        if ys is None:
            ys = mb.flowline_mb_models[0].ys
        if ye is None:
            ye = mb.flowline_mb_models[0].ye
        years = np.arange(ys, ye + 1)


    odf = pd.Series(data=mb.get_specific_mb(year=years),
                    index=years)
    return odf


from oggm.utils._workflow import global_task
@global_task(log)
def compile_fixed_geometry_mass_balance_TIModel(gdirs, filesuffix='',
                                        path=True, csv=False,
                                        use_inversion_flowlines=True,
                                        ys=None, ye=None, years=None,
                                        ds_gcm=None,
                                        **kwargs):
    """Compiles a table of specific mass-balance timeseries for all glaciers.

    The file is stored in a hdf file (not csv) per default. Use pd.read_hdf
    to open it.

    Parameters
    ----------
    gdirs : list of :py:class:`oggm.GlacierDirectory` objects
        the glacier directories to process
    filesuffix : str
        add suffix to output file
    path : str, bool
        Set to "True" in order  to store the info in the working directory
        Set to a path to store the file to your chosen location (file
        extension matters)
    csv: bool
        Set to store the data in csv instead of hdf.
    use_inversion_flowlines : bool
        whether to use the inversion flowlines or the model flowlines
    ys : int
        start year of the model run (default: from the climate file)
        date)
    ye : int
        end year of the model run (default: from the climate file)
    years : array of ints
        override ys and ye with the years of your choice
    """
    from oggm.workflow import execute_entity_task
    #from oggm.core.massbalance import fixed_geometry_mass_balance

    out_df = execute_entity_task(fixed_geometry_mass_balance_TIModel, gdirs,
                                 use_inversion_flowlines=use_inversion_flowlines,
                                 ys=ys, ye=ye, years=years,
                                 ds_gcm=ds_gcm, **kwargs)

    for idx, s in enumerate(out_df):
        if s is None:
            out_df[idx] = pd.Series(np.NaN)

    out = pd.concat(out_df, axis=1, keys=[gd.rgi_id for gd in gdirs])
    out = out.dropna(axis=0, how='all')

    if path:
        if path is True:
            fpath = os.path.join(cfg.PATHS['working_dir'],
                                 'fixed_geometry_mass_balance' + filesuffix)
            if csv:
                out.to_csv(fpath + '.csv')
            else:
                out.to_hdf(fpath + '.hdf', key='df')
        else:
            ext = os.path.splitext(path)[-1]
            if ext.lower() == '.csv':
                out.to_csv(path)
            elif ext.lower() == '.hdf':
                out.to_hdf(path, key='df')
    return out


def extend_past_climate_run_TIModel(past_run_file=None,
                            fixed_geometry_mb_file=None,
                            glacier_statistics_file=None,
                            path=False,
                            use_compression=True):
    """Utility function to extend past MB runs prior to the RGI date.

    We use a fixed geometry (and a fixed calving rate) for all dates prior
    to the RGI date.

    This is not parallelized, i.e a bit slow.

    Parameters
    ----------
    past_run_file : str
        path to the historical run (nc)
    fixed_geometry_mb_file : str
        path to the MB file (csv)
    glacier_statistics_file : str
        path to the glacier stats file (csv)
    path : str
        where to store the file
    use_compression : bool

    Returns
    -------
    the extended dataset
    """

    log.workflow('Applying extend_past_climate_run on '
                 '{}'.format(past_run_file))

    fixed_geometry_mb_df = pd.read_csv(fixed_geometry_mb_file, index_col=0,
                                       low_memory=False)
    stats_df = pd.read_csv(glacier_statistics_file, index_col=0,
                           low_memory=False)

    with xr.open_dataset(past_run_file) as past_ds:

        # We need at least area and vol to do something
        if 'volume' not in past_ds.data_vars or 'area' not in past_ds.data_vars:
            raise InvalidWorkflowError('Need both volume and area to proceed')

        y0_run = int(past_ds.time[0])
        y1_run = int(past_ds.time[-1])
        if (y1_run - y0_run + 1) != len(past_ds.time):
            raise NotImplementedError('Currently only supports annual outputs')
        y0_clim = int(fixed_geometry_mb_df.index[0])
        y1_clim = int(fixed_geometry_mb_df.index[-1])
        if y0_clim > y0_run or y1_clim < y0_run:
            raise InvalidWorkflowError('Dates do not match.')
        if y1_clim != y1_run - 1:
            raise InvalidWorkflowError('Dates do not match.')
        if len(past_ds.rgi_id) != len(fixed_geometry_mb_df.columns):
            raise InvalidWorkflowError('Nb of glaciers do not match.')
        if len(past_ds.rgi_id) != len(stats_df.index):
            raise InvalidWorkflowError('Nb of glaciers do not match.')

        # Make sure we agree on order
        df = fixed_geometry_mb_df[past_ds.rgi_id]

        # Output data
        years = np.arange(y0_clim, y1_run+1)
        ods = past_ds.reindex({'time': years})

        # Time
        ods['hydro_year'].data[:] = years
        ods['hydro_month'].data[:] = ods['hydro_month'][-1]
        if ods['hydro_month'][-1] == 1:
            ods['calendar_year'].data[:] = years
        else:
            ods['calendar_year'].data[:] = years - 1
        ods['calendar_month'].data[:] = ods['calendar_month'][-1]
        for vn in ['hydro_year', 'hydro_month',
                   'calendar_year', 'calendar_month']:
            ods[vn] = ods[vn].astype(int)

        # New vars
        for vn in ['volume', 'volume_bsl', 'volume_bwl',
                   'area', 'length', 'calving', 'calving_rate']:
            if vn in ods.data_vars:
                ods[vn + '_ext'] = ods[vn].copy(deep=True)
                ods[vn + '_ext'].attrs['description'] += ' (extended with MB data)'

        vn = 'volume_fixed_geom_ext'
        ods[vn] = ods['volume'].copy(deep=True)
        ods[vn].attrs['description'] += ' (replaced with fixed geom data)'

        rho = cfg.PARAMS['ice_density']
        # Loop over the ids
        for i, rid in enumerate(ods.rgi_id.data):
            # Both do not need to be same length but they need to start same
            mb_ts = df.values[:, i]
            orig_vol_ts = ods.volume_ext.data[:, i]
            if not (np.isfinite(mb_ts[-1]) and np.isfinite(orig_vol_ts[-1])):
                # Not a valid glacier
                continue
            if np.isfinite(orig_vol_ts[0]):
                # Nothing to extend, really
                continue

            # First valid id
            fid = np.argmax(np.isfinite(orig_vol_ts))

            # Add calving to the mix
            try:
                calv_flux = stats_df.loc[rid, 'calving_flux'] * 1e9
                calv_rate = stats_df.loc[rid, 'calving_rate_myr']
            except KeyError:
                calv_flux = 0
                calv_rate = 0
            if not np.isfinite(calv_flux):
                calv_flux = 0
            if not np.isfinite(calv_rate):
                calv_rate = 0

            # Fill area and length which stays constant before date
            orig_area_ts = ods.area_ext.data[:, i]
            orig_area_ts[:fid] = orig_area_ts[fid]

            # We convert SMB to volume
            mb_vol_ts = (mb_ts / rho * orig_area_ts[fid] - calv_flux).cumsum()
            calv_ts = (mb_ts * 0 + calv_flux).cumsum()

            # The -1 is because the volume change is known at end of year
            mb_vol_ts = mb_vol_ts + orig_vol_ts[fid] - mb_vol_ts[fid-1]

            # Now back to netcdf
            ods.volume_fixed_geom_ext.data[1:, i] = mb_vol_ts
            ods.volume_ext.data[1:fid, i] = mb_vol_ts[0:fid-1]
            ods.area_ext.data[:, i] = orig_area_ts

            # Optional variables
            if 'length' in ods.data_vars:
                orig_length_ts = ods.length_ext.data[:, i]
                orig_length_ts[:fid] = orig_length_ts[fid]
                ods.length_ext.data[:, i] = orig_length_ts

            if 'calving' in ods.data_vars:
                orig_calv_ts = ods.calving_ext.data[:, i]
                # The -1 is because the volume change is known at end of year
                calv_ts = calv_ts + orig_calv_ts[fid] - calv_ts[fid-1]
                ods.calving_ext.data[1:fid, i] = calv_ts[0:fid-1]

            if 'calving_rate' in ods.data_vars:
                orig_calv_rate_ts = ods.calving_rate_ext.data[:, i]
                # +1 because calving rate at year 0 is unkown from the dyns model
                orig_calv_rate_ts[:fid+1] = calv_rate
                ods.calving_rate_ext.data[:, i] = orig_calv_rate_ts

            # Extend vol bsl by assuming that % stays constant
            if 'volume_bsl' in ods.data_vars:
                bsl = ods.volume_bsl.data[fid, i] / ods.volume.data[fid, i]
                ods.volume_bsl_ext.data[:fid, i] = bsl * ods.volume_ext.data[:fid, i]
            if 'volume_bwl' in ods.data_vars:
                bwl = ods.volume_bwl.data[fid, i] / ods.volume.data[fid, i]
                ods.volume_bwl_ext.data[:fid, i] = bwl * ods.volume_ext.data[:fid, i]

        # Remove old vars
        for vn in list(ods.data_vars):
            if '_ext' not in vn and 'time' in ods[vn].dims:
                del ods[vn]

        # Rename vars to their old names
        ods = ods.rename(dict((o, o.replace('_ext', ''))
                              for o in ods.data_vars))

        # Remove t0 (which is NaN)
        ods = ods.isel(time=slice(1, None))

        # To file?
        if path:
            enc_var = {'dtype': 'float32'}
            if use_compression:
                enc_var['complevel'] = 5
                enc_var['zlib'] = True
            encoding = {v: enc_var for v in ods.data_vars}
            ods.to_netcdf(path, encoding=encoding)

    return ods
