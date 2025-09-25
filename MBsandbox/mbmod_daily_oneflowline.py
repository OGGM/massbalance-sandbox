#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 24 12:28:37 2020

@author: lilianschuster

different temperature index mass balance types added that are working with the Huss flowlines

"""

import numpy as np
from scipy.interpolate import interp1d
import pandas as pd
import xarray as xr
import os
import netCDF4
import datetime
import warnings
import scipy.stats as stats
import logging
import copy
from scipy import optimize as optimization

# imports from oggm
from oggm import entity_task
from oggm import cfg, utils
from oggm.cfg import SEC_IN_YEAR, SEC_IN_MONTH, SEC_IN_DAY
from oggm.utils import (floatyear_to_date, date_to_floatyear, ncDataset,
                        lazy_property, monthly_timeseries,
                        clip_min, clip_array)
from oggm.utils._funcs import haversine
from oggm.utils._workflow import global_task

from oggm.exceptions import InvalidParamsError, InvalidWorkflowError
from oggm.shop.ecmwf import get_ecmwf_file, BASENAMES
from oggm.core.massbalance import MassBalanceModel

import MBsandbox

# Module logger
log = logging.getLogger(__name__)
ECMWF_SERVER = 'https://cluster.klima.uni-bremen.de/~oggm/climate/'

# only relevant when using
# server='https://cluster.klima.uni-bremen.de/~lschuster/'
# only glacier-relevant gridpoints included!
BASENAMES['ERA5_daily'] = {
        'inv': 'era5/daily/v1.0/era5_glacier_invariant_flat.nc',
        'tmp': 'era5/daily/v1.0/era5_daily_t2m_1979-2018_flat.nc'
        }

BASENAMES['WFDE5_CRU_daily'] = {
    'inv': 'wfde5_cru/daily/v1.1/wfde5_cru_glacier_invariant_flat.nc',
    'tmp': 'wfde5_cru/daily/v1.1/wfde5_cru_tmp_1979-2018_flat.nc',
    'prcp': 'wfde5_cru/daily/v1.1/wfde5_cru_prcp_1979-2018_flat.nc',
    }

BASENAMES['W5E5_daily'] = {
    'inv': 'w5e5v2.0/flattened/2023.2/daily/w5e5v2.0_glacier_invariant_flat.nc',
    'tmp': 'w5e5v2.0/flattened/2023.2/daily/w5e5v2.0_tas_global_daily_flat_glaciers_1979_2019.nc',
    'prcp': 'w5e5v2.0/flattened/2023.2/daily/w5e5v2.0_pr_global_daily_flat_glaciers_1979_2019.nc',
    }

BASENAMES['MSWEP_daily'] = {
    'prcp': 'mswepv2.8/flattened/daily/mswep_pr_global_daily_flat_glaciers_1979_2019.nc'
    # there is no orography file for MSWEP!!! (and also no temperature file)
    }

# this is for Sarah Hanus workflow
# this only works if
# ECMWF_SERVER = 'https://cluster.klima.uni-bremen.de/~oggm/shanus/'
BASENAMES['W5E5_daily_dw'] = {
    'inv': 'ISIMIP3a/flattened/daily/gswp3-w5e5_obsclim_glacier_invariant_flat.nc',
    'tmp': 'ISIMIP3a/flattened/daily/gswp3-w5e5_obsclim_tas_global_daily_flat_glaciers_1979_2019.nc',
    'prcp': 'ISIMIP3a/flattened/daily/gswp3-w5e5_obsclim_pr_global_daily_flat_glaciers_1979_2019.nc',
    }


def get_w5e5_file(dataset='W5E5_daily', var=None,
                  server='https://cluster.klima.uni-bremen.de/~lschuster/'):
    """returns a path to desired WFDE5_CRU or W5E5 or MSWEP
     baseline climate file.

    If the file is not present, downloads it

    ... copy of get_ecmwf_file (the only difference is that this function is more variable with a
     server keyword-argument instead of using only ECMWF_SERVER),
     if wished could be easily adapted to work in OGGM proper

    dataset : str
        e.g. 'W5E5_daily', 'WFDE5_CRU_daily', 'MSWEP_daily', 'W5E5_daily_dw', could define more BASENAMES
    var : str
        'inv' for invariant
        'tmp' for temperature
        'pre' for precipitation
    server : str
        path to climate files on the cluster
    """
    if dataset == 'WFDE5_CRU_daily' or dataset == 'ERA5_daily':
        print('Attention: these datasets are basically deprecated. There '
                          'is also an issue with the flattened files, i.e., some'
                          'glacier gridpoints are missing. If you really want to use'
                          'them, you need to do the flattening yourself. ')
    # check if input makes sense
    if dataset not in BASENAMES.keys():
        raise InvalidParamsError('ECMWF dataset {} not '
                                 'in {}'.format(dataset, BASENAMES.keys()))
    if var not in BASENAMES[dataset].keys():
        raise InvalidParamsError('ECMWF variable {} not '
                                 'in {}'.format(var,
                                                BASENAMES[dataset].keys()))

    # File to look for
    return utils.file_downloader(server + BASENAMES[dataset][var])


def write_climate_file(gdir, time, prcp, temp,
                       ref_pix_hgt, ref_pix_lon, ref_pix_lat,
                       ref_pix_lon_pr=None, ref_pix_lat_pr=None,
                       gradient=None, temp_std=None,
                       time_unit=None, calendar=None,
                       source=None, long_source=None,
                       file_name='climate_historical',
                       filesuffix='',
                       temporal_resol='monthly'):
    """Creates a netCDF4 file with climate data timeseries.
    this could be used in general also in OGGM proper

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
        baseline climate (if MSWEP prcp used, only of temp. climate file).
    ref_pix_lon : float
        the location of the gridded data's grid point
        (if MSWEP prcp used, only of temp. climate file)
    ref_pix_lat : float
        the location of the gridded data's grid point
        (if MSWEP prcp used, only of temp. climate file)
    ref_pix_lon_pr : float
        default is None, only if MSWEP prcp used, it is the
        location of the gridded prcp data grid point
    ref_pix_lat_pr : float
        default is None, only if MSWEP prcp used, it is the
        location of the gridded prcp data grid point
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
    long_source : str
        the climate data source describing origin of
        temp, prpc and lapse rate in detail
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
    elif (source == 'WFDE5_CRU_daily' and filesuffix == ''
          and temporal_resol == 'daily'):
        raise InvalidParamsError("filesuffix should be '_daily' for WFDE5_CRU_daily"
                                 "if daily chosen as temporal_resol"
                                 "file_name climate_historical is normally"
                                 "monthly data")
    elif (source == 'W5E5_daily' and filesuffix == ''
          and temporal_resol == 'daily'):
        raise InvalidParamsError("filesuffix should be '_daily' for W5E5_daily"
                                 "if daily chosen as temporal_resol"
                                 "file_name climate_historical is normally"
                                 "monthly data")
    if long_source is None:
        long_source = source
    if 'MSWEP' in long_source:
        prcp_from_mswep = True
    else:
        prcp_from_mswep = False
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
        # these are only valid for temperature if MSWEP prcp is used!!!
        nc.ref_hgt = ref_pix_hgt
        nc.ref_pix_lon = ref_pix_lon
        nc.ref_pix_lat = ref_pix_lat
        nc.ref_pix_dis = haversine(gdir.cenlon, gdir.cenlat,
                                   ref_pix_lon, ref_pix_lat)
        if prcp_from_mswep:
            # there is no reference height given!!!
            if ref_pix_lon_pr is None or ref_pix_lat_pr is None:
                raise InvalidParamsError('if MSWEP is used for prcp, need to add'
                                         'precipitation lon/lat gridpoints')
            nc.ref_pix_lon_pr = np.round(ref_pix_lon_pr,3)
            nc.ref_pix_lat_pr = np.round(ref_pix_lat_pr,3)

        nc.climate_source = long_source
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
        # this could be made more beautiful
        # just rough estimate
        if (len(prcp) > (nc.hydro_yr_1 - nc.hydro_yr_0 + 1) * 28 * 12 and
            temporal_resol == 'daily'):
            if source == 'ERA5_daily':
                v.long_name = ("total daily precipitation amount, "
                               "assumed same for each day of month")
            elif source == 'WFDE5_daily_cru':
                v.long_name = ("total daily precipitation amount"
                               "sum of snowfall and rainfall")
            elif source == 'W5E5_daily' and not prcp_from_mswep:
                v.long_name = ("total daily precipitation amount")
            elif source == 'W5E5_daily' and prcp_from_mswep:
                v.long_name = ("total daily precipitation amount, "
                               "1979-01-01 prcp assumed to be equal to 1979-01-02"
                               "due to missing data, MSWEP prcp with 0.1deg resolution"
                               "(finer than temp. data), so refpixhgt, "
                               "refpixlon and refpixlat not valid for prcp data!!!")

        elif (len(prcp) == (nc.hydro_yr_1 - nc.hydro_yr_0 + 1) * 12
              and temporal_resol == 'monthly' and not prcp_from_mswep):
            v.long_name = 'total monthly precipitation amount'
        elif (len(prcp) == (nc.hydro_yr_1 - nc.hydro_yr_0 + 1) * 12
              and temporal_resol == 'monthly' and prcp_from_mswep):
            v.long_name = ("total monthly precipitation amount, MSWEP prcp with 0.1deg resolution"
                          "(finer than temp. data), so refpixhgt, " 
                          "refpixlon and refpixlat not valid for prcp data!!!")
        else:
            raise InvalidParamsError('there is a conflict in the'
                                     'prcp timeseries, '
                                     'please check temporal_resol')
        # just to check that it is in kg m-2 per day or per month and not in per second
        assert prcp.max() > 1
        v[:] = prcp
        # check that there are no filling values inside
        assert np.all(v[:].data < 1e5)

        v = nc.createVariable('temp', 'f4', ('time',), zlib=zlib)
        v.units = 'degC'
        if ((source == 'ERA5_daily' or source == 'WFDE5_daily_cru' or source =='W5E5_daily') and
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
        # no filling values!
        assert np.all(v[:].data < 1e5)

        if gradient is not None:
            v = nc.createVariable('gradient', 'f4', ('time',), zlib=zlib)
            v.units = 'degC m-1'
            v.long_name = ('temperature gradient from local regression or'
                           'lapserates')
            v[:] = gradient
            # no filling values
            assert np.all(v[:].data < 1e5)

        if temp_std is not None:
            v = nc.createVariable('temp_std', 'f4', ('time',), zlib=zlib)
            v.units = 'degC'
            v.long_name = 'standard deviation of daily temperatures'
            v[:] = temp_std
            # no filling values
            assert np.all(v[:].data < 1e5)

@entity_task(log, writes=['climate_historical_daily'])
def process_w5e5_data(gdir, y0=None, y1=None, temporal_resol='daily',
                      climate_type=None,
                      output_filesuffix=None,
                      cluster=False):
    """
    Processes and writes the WFDE5_CRU & W5E5 daily baseline climate data for a glacier.
    Either on daily or on monthly basis. Can also use W5E5_MSWEP, where precipitation is taken
    from MSWEP (with higher resolution, 0.1°) and temperature comes from W5E5 (so actually from ERA5).
    In this case, the mean gridpoint altitude is only valid for the temperature gridpoint
    (there is no orography for MSWEP, and it is different because of a finer resolution in MSWEP)

    Extracts the nearest timeseries and writes everything to a NetCDF file.
    This uses only the WFDE5_CRU / W5E5 daily temperatures. The temperature lapse
    rate are used from ERA5dr.

    comment: This is similar to process_era5_daily, maybe a general function would be better
    TODO: see _verified_download_helper no known hash for ...
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
    climate_type: str
        either WFDE5_CRU (default, v1.1 only till end of 2018) or W5E5 (end of 2019)
        or W5E5_MSWEP (precipitation from MSWEP, temp. from W5E5)
    output_filesuffix : optional
         None by default, as the output_filesuffix is automatically chosen
         from the temporal_resol and climate_type. But you can change the filesuffix here,
         just make sure that you use then later the right climate file
    cluster : bool
        default is False, if this is run on the cluster, set it to True,
        because we do not need to download the files
        todo: this logic does not make anymore sense as there are other ways to prevent the cluster from downloading
        stuff -> probably I can remove cluster = True entirely ?!

    """
    if cfg.PARAMS['hydro_month_nh'] != 1 and climate_type != 'WFDE5_CRU':
        raise InvalidParamsError('Hydro months different than 1 are not tested, there is some'
                                 'issue with the lapse rates, as they only go until 2019-05'
                                 'if you want other hydro months, need to adapt the code!!!')

    if climate_type == 'WFDE5_CRU':
        if temporal_resol == 'monthly':
            output_filesuffix_def = '_monthly_WFDE5_CRU'
        elif temporal_resol == 'daily':
            output_filesuffix_def = '_daily_WFDE5_CRU'
        # basename of climate
        # (we use for both the daily dataset and resample to monthly)
        dataset = 'WFDE5_CRU_daily'
        dataset_prcp = dataset
    elif climate_type == 'W5E5':
        if temporal_resol == 'monthly':
            output_filesuffix_def = '_monthly_W5E5'
        elif temporal_resol == 'daily':
            output_filesuffix_def = '_daily_W5E5'
        # basename of climate
        # (for both the daily dataset and resample to monthly)
        dataset = 'W5E5_daily'
        dataset_prcp = dataset
    elif climate_type == 'W5E5_dw':
        output_filesuffix_def = '_daily_W5E5_dw'
        dataset = 'W5E5_daily_dw'
        dataset_prcp = dataset
    elif climate_type =='W5E5_MSWEP':
        if temporal_resol == 'monthly':
            output_filesuffix_def = '_monthly_W5E5_MSWEP'
        elif temporal_resol == 'daily':
            output_filesuffix_def = '_daily_W5E5_MSWEP'
        # basename of climate
        # (for both the daily dataset and resample to monthly)
        dataset = 'W5E5_daily'
        # precipitation from MSWEP!!!
        dataset_prcp = 'MSWEP_daily'
    else:
        raise NotImplementedError('climate_type can either be WFDE5_CRU or W5E5 and '
                                  'temporal_resol either monthly or daily!')

    if output_filesuffix is None:
        # set the default output_filesuffix
        output_filesuffix = output_filesuffix_def
    else:
        # use the user-given output-filesufix
        pass

    # wfde5_daily for temperature and precipitation
    # but need temperature lapse rates from ERA5
    dataset_othervars = 'ERA5dr'

    # get the central longitude/latitudes of the glacier
    lon = gdir.cenlon + 360 if gdir.cenlon < 0 else gdir.cenlon
    lat = gdir.cenlat

    # todo: this logic should be removed
    if cluster:
        cluster_path = '/home/www/lschuster/'
        path_tmp = cluster_path + BASENAMES[dataset]['tmp']
        path_prcp = cluster_path + BASENAMES[dataset_prcp]['prcp']
        path_inv = cluster_path + BASENAMES[dataset]['inv']

    else:
        if climate_type != 'W5E5_dw':
            path_tmp = get_w5e5_file(dataset, 'tmp')
            path_prcp = get_w5e5_file(dataset_prcp, 'prcp')
            path_inv = get_w5e5_file(dataset, 'inv')
        elif climate_type == 'W5E5_dw':
            path_tmp = get_w5e5_file(dataset, 'tmp', server='https://cluster.klima.uni-bremen.de/~shanus/')
            path_prcp = get_w5e5_file(dataset_prcp, 'prcp', server='https://cluster.klima.uni-bremen.de/~shanus/')
            path_inv = get_w5e5_file(dataset, 'inv', server='https://cluster.klima.uni-bremen.de/~shanus/')

    # Use xarray to read the data
    # todo: would go faster with only netCDF -.-, but easier with xarray
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
        if climate_type == 'WFDE5_CRU':
            # old version of WFDE5_CRU that only goes till 2018
            if y1 > 2018 or y0 < 1979:
                text = 'The climate files only go from 1979--2018,\
                    choose another y0 and y1'
                raise InvalidParamsError(text)
        elif climate_type[:4] == 'W5E5':
            if y1 > 2019 or y0 < 1979:
                text = 'The climate files only go from 1979 --2019, something is wrong'
                raise InvalidParamsError(text)
        # if default settings: this is the last day in March or September
        time_f = '{}-{:02d}'.format(y1, em)
        end_day = int(ds.sel(time=time_f).time.dt.daysinmonth[-1].values)

        #  this was tested also for hydro_month = 1
        ds = ds.sel(time=slice('{}-{:02d}-01'.format(y0, sm),
                               '{}-{:02d}-{}'.format(y1, em, end_day)))
        if sm == 1 and y1 == 2019 and (climate_type[:4] == 'W5E5'):
            days_in_month = ds['time.daysinmonth'].copy()
        try:
            # computing all the distances and choose the nearest gridpoint
            c = (ds.longitude - lon)**2 + (ds.latitude - lat)**2
            ds = ds.isel(points=c.argmin())
        # I turned this around
        except ValueError:
            ds = ds.sel(longitude=lon, latitude=lat, method='nearest')
            # normally if I do the flattening, this here should not occur

        # if we want to use monthly mean temperatures and
        # standard deviation of daily temperature:
        Tvar = 'Tair'
        Pvar = 'tp'
        if climate_type[:4] == 'W5E5':
            Tvar = 'tas'
            Pvar = 'pr'
        if temporal_resol == 'monthly':
            Tair_std = ds.resample(time='MS').std()[Tvar]
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
        temp = ds[Tvar].data - 273.15
        time = ds.time.data

        ref_lon = float(ds['longitude'])
        ref_lat = float(ds['latitude'])

        ref_lon = ref_lon - 360 if ref_lon > 180 else ref_lon

    # precipitation: similar as temperature
    with xr.open_dataset(path_prcp) as ds:
        assert ds.longitude.min() >= 0

        yrs = ds['time.year'].data
        y0 = yrs[0] if y0 is None else y0
        y1 = yrs[-1] if y1 is None else y1
        # Attention here we take the same y0 and y1 as given from the
        # daily tmp dataset (goes till end of 2018, or end of 2019)

        # attention if daily data, need endday!!!
        if climate_type == 'W5E5_MSWEP' and y0 == 1979 and sm == 1:
            # first day of 1979 is missing!!! (will assume later that it is equal to the
            # median of January daily prcp ...)
            ds = ds.sel(time=slice('{}-{:02d}-02'.format(y0, sm),
                                   '{}-{:02d}-{}'.format(y1, em, end_day)))
        else:
            ds = ds.sel(time=slice('{}-{:02d}-01'.format(y0, sm),
                                   '{}-{:02d}-{}'.format(y1, em, end_day)))
        try:
            # ... prcp is also flattened
            # in case of W5E5_MSWEP this will be another gridpoint than for temperature
            # but normally it should work
            c = (ds.longitude - lon)**2 + (ds.latitude - lat)**2
            ds = ds.isel(points=c.argmin())
        except ValueError:
            # this should not occur
            ds = ds.sel(longitude=lon, latitude=lat, method='nearest')

        if y0 == 1979 and sm == 1 and climate_type == 'W5E5_MSWEP':
            median_jan_1979 = np.median(ds.sel(time='1979-01').pr.values) * SEC_IN_DAY
        # if we want to use monthly summed up precipitation:
        if temporal_resol == 'monthly':
            ds = ds.resample(time='MS').sum()
            ds['longitude'] = ds.longitude.isel(time=0)
            ds['latitude'] = ds.latitude.isel(time=0)
        elif temporal_resol == 'daily':
            pass
        if climate_type == 'WFDE5_CRU':
        # the prcp data of wfde5_CRU has been converted already into
        # kg m-2 day-1 ~ mm/day or into kg m-2 month-1 ~ mm/month
            prcp = ds[Pvar].data  # * 1000
        elif climate_type[:4] == 'W5E5':
            # if daily: convert kg m-2 s-1 into kg m-2 day-1
            # if monthly: convert monthly sum of kg m-2 s-1 into kg m-2 month-1
            prcp = ds[Pvar].data * SEC_IN_DAY
            if climate_type == 'W5E5_MSWEP':
                # 1st day of January 1979 is missing for MSWEP prcp
                # assume that precipitation is equal to
                # median of January daily prcp
                if y0 == 1979 and sm == 1:
                    if temporal_resol == 'daily':
                        prcp = np.append(np.array(median_jan_1979), prcp)
                    elif temporal_resol == 'monthly':
                        # in the monthly sum, 1st of January 1979 is missing
                        # add this value by assuming mean over the other days
                        prcp[0] = prcp[0] + median_jan_1979
                    if y1 == 2019:
                        if temporal_resol == 'daily':
                            assert len(prcp) == 14975  # 41 years * amount of days
                        elif temporal_resol == 'monthly':
                            assert len(prcp) == 492  # 41 years * amount of months
        ref_lon_pr = float(ds['longitude'])
        ref_lat_pr = float(ds['latitude'])

        ref_lon_pr = ref_lon_pr - 360 if ref_lon_pr > 180 else ref_lon_pr

    # wfde5/w5e5 invariant file
    # (gridpoint altitude only valid for temperature
    # in case of 'W5E5_MSWEP')
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
    # there are no lapse rates from wfde5/W5E5 !!!
    # TODO: use updated ERA5dr files that go until end of 2019 and update the code accordingly !!!
    path_lapserates = get_ecmwf_file(dataset_othervars, 'lapserates')
    with xr.open_dataset(path_lapserates) as ds:
        assert ds.longitude.min() >= 0

        yrs = ds['time.year'].data
        y0 = yrs[0] if y0 is None else y0
        y1 = yrs[-1] if y1 is None else y1
        # Attention here we take the same y0 and y1 as given from the
        # daily tmp dataset (goes till end of 2018/2019)

        ds = ds.sel(time=slice('{}-{:02d}-01'.format(y0, sm),
                               '{}-{:02d}-01'.format(y1, em)))

        # no flattening done for the ERA5dr gradient dataset
        ds = ds.sel(longitude=lon, latitude=lat, method='nearest')
        if sm == 1 and y1 == 2019 and (climate_type[:4] == 'W5E5'):
            # missing some months of ERA5dr (which only goes till middle of 2019)
            # otherwise it will fill it with large numbers ...
            ds = ds.sel(time=slice('{}-{:02d}-01'.format(y0, sm), '2018-{:02d}-01'.format(em)))
            mean_grad = ds.groupby('time.month').mean().lapserate
            # fill the last year with mean gradients
            gradient = np.concatenate((ds['lapserate'].data, mean_grad.values), axis=None)
        else:
            # get the monthly gradient values
            gradient = ds['lapserate'].data
        if temporal_resol == 'monthly':
            pass
        elif temporal_resol == 'daily':
            # gradient needs to be restructured to have values for each day
            # when daily resolution  is applied
            # assume same gradient for each day
            if sm == 1 and y1 == 2019 and (climate_type[:4] == 'W5E5'):
                gradient = np.repeat(gradient, days_in_month.resample(time='MS').mean())
                assert len(gradient) == len(days_in_month)
            else:
                gradient = np.repeat(gradient, ds['time.daysinmonth'])

        long_source = 'temp: {}, prcp: {}, lapse rate: {}'.format(dataset,
                                                                  dataset_prcp,
                                                                  dataset_othervars)
        if climate_type == 'W5E5_MSWEP':
            ref_pix_lon_pr = ref_lon_pr
            ref_pix_lat_pr = ref_lat_pr
        else:
            # if not MSWEP prcp and temp gridpoints should be the same ones!!!
            np.testing.assert_allclose(ref_lon_pr, ref_lon)
            np.testing.assert_allclose(ref_lat_pr, ref_lat)
            ref_pix_lon_pr = None
            ref_pix_lat_pr = None

    # OK, ready to write
    write_climate_file(gdir, time, prcp, temp, hgt, ref_lon, ref_lat,
                       ref_pix_lon_pr=ref_pix_lon_pr, ref_pix_lat_pr=ref_pix_lat_pr,
                       filesuffix=output_filesuffix,
                       temporal_resol=temporal_resol,
                       gradient=gradient,
                       temp_std=temp_std,
                       source=dataset,
                       long_source=long_source,
                       file_name='climate_historical')


@entity_task(log, writes=['climate_historical_daily'])
def process_era5_daily_data(gdir, y0=None, y1=None, output_filesuffix='_daily_ERA5',
                            cluster=False):
    """Processes and writes the era5 daily baseline climate data for a glacier.
    into climate_historical_daily.nc

    Extracts the nearest timeseries and writes everything to a NetCDF file.
    This uses only the ERA5 daily temperatures. The precipitation, lapse
    rate and standard deviations are used from ERA5dr.

    comment: this is now a new function, which could also be adapted for other climates, but
    it seemed to be easier to make a new function for a new climate dataset (such as process_w5e5_data)
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
        todo: this logic does not make anymore sense as there are other
         ways to prevent the cluster from downloading stuff
         -> probably I can remove cluster = True entirely ?!

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

    temp_std = None
    path_lapserates = get_ecmwf_file(dataset_othervars, 'lapserates')
    with xr.open_dataset(path_lapserates) as ds:
        assert ds.longitude.min() >= 0

        yrs = ds['time.year'].data
        y0 = yrs[0] if y0 is None else y0
        y1 = yrs[-1] if y1 is None else y1
        # Attention here we take the same y0 and y1 as given from the
        # daily tmp dataset

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


class TIModel_Parent(MassBalanceModel):
    """ Parent class that works for different temperature-index models, this is only instanciated
    via the child classes TIModel or TIModel_Sfc_Type. It is just a container with shared code
    to get annual, monthly and daily climate. The actual mass balance can only be computed in child classes as
    there the methods between using surface type and not using sfc type differ.

    Different mass balance modules compatible to OGGM with one flowline,
    so far this is only tested for the elevation-band flowlines

    """

    def __init__(self, gdir, melt_f, prcp_fac=2.5, residual=0,
                 mb_type='mb_pseudo_daily', N=100, loop=False,
                 temp_std_const_from_hist = False,
                 grad_type='cte', filename='climate_historical',
                 repeat=False, ys=None, ye=None,
                 t_solid=0, t_liq=2, t_melt=0,
                 default_grad=-0.0065,
                 temp_local_gradient_bounds=[-0.009, -0.003],
                 SEC_IN_YEAR=SEC_IN_YEAR,
                 SEC_IN_MONTH=SEC_IN_MONTH,
                 SEC_IN_DAY=SEC_IN_DAY,
                 baseline_climate=None,
                 input_filesuffix='default',
                 input_filesuffix_h_for_temp_std = '_monthly_W5E5'
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
        prcp_fac : float
            multiplicative precipitation factor, has to be calibrated for each option and
            each climate dataset, default is 2.5,
        residual : float, optional
            default is to use a residual of zero [mm we yr-1 ~ kg/m2/mth]
            Note that this residual is *substracted* from the computed MB.
            do not change this unless you know what you do!
            Indeed: residual = MODEL_MB - REFERENCE_MB.
            ToDO: maybe change the sign?,
            opposite to OGGM "MB terms + residual"
        mb_type: str
            three types: 'mb_pseudo_daily' (default: use temp_std and N percentiles),
            'mb_monthly' (same as default OGGM PastMassBalance),
            'mb_real_daily' (use daily temperature values).
            the mb_type only work if the baseline_climate of gdir is right
        N : int
            number of percentiles used to generate gaussian-like daily
            temperatures from daily std and mean monthly temp (default is N=100)
        loop : bool
            the way how the matrix multiplication is done,
            using np.matmul or a loop(default: False)
            only applied if mb_type is 'mb_pseudo_daily'
            todo: can actually remove loop=True, because this is not anymore used!
             (loop=True is deprecated)
        grad_type : str
            three types of applying the temperature gradient:
            'cte' (default, constant lapse rate, set to default_grad,
                   same as in default OGGM)
            'var_an_cycle' (varies spatially and over annual cycle,
                            but constant over the years)
            'var' (varies spatially & temporally as in the climate files, deprecated!)
        temp_std_const_from_hist : boolean, optional
            whether a variable temp_std is used or one that is constant but just change
            for each month of the year (avg. over calib period 2000-2019). Only is used
            in combination with mb_type='mb_pseudo_daily'. If set to True, mb_pseudo_daily
            is more similar to the approach inside of GloGEM, GloGEMFlow ...
        filename : str, optional
            set it to a different BASENAME if you want to use alternative climate
            data, default is 'climate_historical'
        repeat : bool
            Whether the climate period given by [ys, ye] should be repeated
            indefinitely in a circular way (default is False)
            todo: add when this is used
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
            if grad_type != cte and the estimated lapse rate (from ERA5)
            does not lie in this range,
            set gradient instead to these minimum, maximum gradients
            (default: [-0.009, -0.003] m/K)
        SEC_IN_YEAR: float
            seconds in a year (default: 31536000s),
            comment: maybe this could be changed
        SEC_IN_MONTH: float
            seconds in a month (default: 2628000s),
            comment: maybe this could be changed as not each
            month has the same amount of seconds,
            in February can be a difference of 8%
        baseline_climate: str
            climate that should be applied, e.g. ERA5dr, WFDE5_CRU, W5E5
        input_filesuffix: str
            if set to 'default', it is set depending on mb_type and
            baseline_climate, but can change this here,
            e.g. change it to '' to work without filesuffix as
            default in oggm PastMassBalance
        input_filesuffix_h_for_temp_std : str
            historical climate_historical filesuffix which was used for calibration,
            only used if  filename is not
            climate_historical and if mb_type = 'mb_pseudo_daily' and if
            temp_std_const_from_hist is True. It is then necessary to get the 12 monthly
            daily std from the historical climate file.

        Attributes
        ----------
        temp_bias : float,
            Add a temperature bias to the time series (default is 0)
            todo: maybe also add it as keyword argument?
        prcp_fac : float, >0
            multiplicative precipitation correction factor (default 2.5)
        melt_f : float, >0
            melt temperature sensitivity factor per month (kg /m² /mth /K),
            need to be prescribed, e.g. such that
            |mean(MODEL_MB)-mean(REF_MB)|--> 0
        """

        if mb_type == 'mb_pseudo_daily_fake':
            temp_std_const_from_hist = True
            mb_type = 'mb_pseudo_daily'
        self.mb_type = mb_type

        # melt_f is only initiated here, and not used in __init__
        # so it does not matter if it is changed
        # just enforce it then it is easier for run_from_climate_data ...
        self._melt_f = melt_f
        if self._melt_f != None and self._melt_f <= 0:
            raise InvalidParamsError('melt_f has to be above zero!')
        if prcp_fac <= 0:
            raise InvalidParamsError('prcp_fac has to be above zero!')
        # as self.prcp is produced by changing prcp_fac
        # we need to update the prcp via prcp_fac by a property (see code after __init__)
        # (reason why there is here only a self._prcp_fac and no self.prcp_fac)
        #  to allow prcp_fac to be changed after instantiation
        #  prescribe the prcp_fac as it is instantiated
        self._prcp_fac = prcp_fac
        # same is true for temp bias
        self._temp_bias = 0.
        self.residual = residual

        # Parameters (that stay cte)
        # comment: those ones are not allowed to be changed after initiation !!!
        # (for my use case, I don't need to change them,
        # but if necessary would need to set properties as well)
        self.t_solid = t_solid
        self.t_liq = t_liq
        self.t_melt = t_melt
        self.N = N

        self.loop = loop
        self.grad_type = grad_type
        # default rho is 900  kg/m3
        # (to convert from kg/m2 into m ice per second=
        self.rho = cfg.PARAMS['ice_density']

        # Public attrs
        self.hemisphere = gdir.hemisphere
        self.repeat = repeat

        self.SEC_IN_YEAR = SEC_IN_YEAR
        self.SEC_IN_MONTH = SEC_IN_MONTH
        self.SEC_IN_DAY = SEC_IN_DAY
        # valid_bounds: The altitudinal bounds where the MassBalanceModel is valid.
        # This is necessary for automated ELA search.
        self.valid_bounds = [-1e4, 2e4]

        # only have one flowline when using elevation bands
        self.fl = gdir.read_pickle('inversion_flowlines')[-1]

        # check if the right climate is used for the right mb_type
        if baseline_climate == None:
            try:
                baseline_climate = gdir.get_climate_info()['baseline_climate_source']
            except:
                baseline_climate = cfg.PARAMS['baseline_climate']
            if baseline_climate != cfg.PARAMS['baseline_climate']:
                raise InvalidParamsError('need to use filesuffixes to define the right climate!')

        if input_filesuffix == 'default':
            if mb_type != 'mb_real_daily':
                input_filesuffix = '_monthly_{}'.format(baseline_climate)
            else:
                input_filesuffix = '_daily_{}'.format(baseline_climate)
        else:
            warnings.warn('you changed the default input_filesuffix of the climate,'
                           'make sure that the default climate (without filesuffix)'
                           'is what you want and is compatible to the chosen temporal resolution!')

        self._input_filesuffix = input_filesuffix
        monthly_climates = ['CRU', 'ERA5dr', 'HISTALP', 'CERA']
        if (self.mb_type == 'mb_real_daily' and
            (baseline_climate in monthly_climates)):
            text = ('wrong climate for mb_real_daily, need to do e.g. '
                    'process_era5_daily_data(gd) to produce daily_ERA5dr'
                    'or process_w5e5_data(gd) for daily_WFDE5_CRU')
            raise InvalidParamsError(text)
        # mb_monthly does not work when daily temperatures are used
        if self.mb_type == 'mb_monthly' and \
                (baseline_climate == 'ERA5_daily' or 'daily' in input_filesuffix):
            text = ('wrong climate for mb_monthly, need to do e.g.'
                    'oggm.shop.ecmwf.process_ecmwf_data(gd, dataset="ERA5dr")')
            raise InvalidParamsError(text)
        # mb_pseudo_daily needs temp_std
        if self.mb_type == 'mb_pseudo_daily' and baseline_climate == 'ERA5_daily':
            text = 'wrong climate for mb_pseudo_daily, need to do e.g. \
            oggm.shop.ecmwf.process_ecmwf_data(gd, dataset = "ERA5dr")'
            raise InvalidParamsError(text)

        # Read climate file
        fpath = gdir.get_filepath(filename, filesuffix=input_filesuffix)

        # todo exectime:
        #  I am using xarray instead of netCDF4, which is slower and I should change this at some point to netCDF4
        with xr.open_dataset(fpath) as xr_nc:
            if self.mb_type == 'mb_real_daily' or self.mb_type == 'mb_monthly':
                # even if there is temp_std inside the dataset, we won't use
                # it for these mb_types
                self.temp_std = np.nan
            else:
                try:
                    if temp_std_const_from_hist:
                        # really take the temp_std from historical calibration time period!!!
                        # this has to take the historical data even if we look here into gcm data
                        if int(xr_nc.time[0].dt.month.values) != 1 or cfg.PARAMS[f'hydro_month_{self.hemisphere}'] != 1:
                            raise InvalidWorkflowError('temp_std_const_from_hist is only implemented and tested'
                                                       'with hydro_month = 1')
                        if filename == 'climate_historical':
                            xr_nc_h = xr_nc
                        else:
                            fpath_h = gdir.get_filepath('climate_historical',
                                                        filesuffix=input_filesuffix_h_for_temp_std)
                            xr_nc_h = xr.open_dataset(fpath_h)
                        # here we need to have the lenth of used climate file (which can be the gcm file)
                        # so, we need xr_nc and not xr_nc_h !!!
                        n_yrs = len(xr_nc.time.groupby('time.year').mean())
                        temp_std_cte = xr_nc_h['temp_std'].sel(time=slice('2000',
                                                                          '2020')).groupby('time.month').mean()
                        self.temp_std = np.concatenate([temp_std_cte.values.astype(np.float64)] * n_yrs)
                        xr_nc_h.close()
                    else:
                        self.temp_std = xr_nc['temp_std'].values.astype(np.float64)

                except KeyError:
                    text = ('The applied climate has no temp std, do e.g.'
                            'oggm.shop.ecmwf.process_ecmwf_data'
                            '(gd, dataset="ERA5dr")')
                    raise InvalidParamsError(text)

            # goal is to get self.years/self.months in hydro_years
            # (only important for TIModel if we do not use calendar years,
            # for TIModel_Sfc_Type, we can only use calendar years anyways!)

            # get the month where the hydrological month starts
            # as chosen from the gdir climate file
            hydro_month_start = int(xr_nc.time[0].dt.month.values)
            # if we actually use TIModel_Sfc_Type -> hydro_month has to be 1
            if type(self) == TIModel_Sfc_Type:
                if hydro_month_start != 1 or cfg.PARAMS[f'hydro_month_{self.hemisphere}'] != 1:
                    raise InvalidWorkflowError('TIModel_Sfc_Type works only with calendar years, set '
                                               'cfg.PARAMS["hydro_month_nh"] (or sh)'
                                               ' to 1 and process the climate data again')

            if self.mb_type != 'mb_real_daily':
                ny, r = divmod(len(xr_nc.time), 12)
                if r != 0:
                    raise ValueError('Climate data should be N full years')
                # This is where we switch to hydro float year format
                # Last year gives the tone of the hydro year
                self.years = np.repeat(np.arange(xr_nc.time[-1].dt.year-ny+1,
                                                 xr_nc.time[-1].dt.year+1),
                                       12)
                self.months = np.tile(np.arange(1, 13), ny)

            elif self.mb_type == 'mb_real_daily':
                # use pandas to convert month/year to hydro_years
                # this has to be done differently than above because not
                # every month, year has the same amount of days
                pd_test = pd.DataFrame(xr_nc.time.to_series().dt.year.values,
                                       columns=['year'])
                pd_test.index = xr_nc.time.to_series().values
                pd_test['month'] = xr_nc.time.to_series().dt.month.values
                pd_test['hydro_year'] = np.nan

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
                    # need this to ensure that gradients are not fill-values
                    xr_nc['gradient'] = xr_nc['gradient'].where(xr_nc['gradient'] < 1e12)
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
                            assert np.all(grad < 1e12)
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
            self.ref_hgt = xr_nc.ref_hgt
            # if climate dataset has been corrected once again
            # or non corrected reference height!
            try:
                self.uncorrected_ref_hgt = xr_nc.uncorrected_ref_hgt
            except:
                self.uncorrected_ref_hgt = xr_nc.ref_hgt

            self.ys = self.years[0] if ys is None else ys
            self.ye = self.years[-1] if ye is None else ye

        self.fpath = fpath

    @property
    def prcp_fac(self):
        """ prints the _prcp_fac as initiated or changed"""
        return self._prcp_fac

    @prcp_fac.setter
    def prcp_fac(self, new_prcp_fac):
        """ sets new prcp_fac, changes prcp time series
         and if TIModel_Sfc_Type resets the pd_mb and buckets
        """
        if new_prcp_fac <= 0:
            raise InvalidParamsError('prcp_fac has to be above zero!')
        # attention, prcp_fac should not be called here
        # otherwise there is recursion occurring forever...
        # use new_prcp_fac to not get maximum recusion depth error
        self.prcp *= new_prcp_fac / self._prcp_fac

        if type(self) == TIModel_Sfc_Type:
            # if the prcp_fac is set to another value,
            # need to reset pd_mb_annual, pd_mb_monthly
            # and also reset pd_bucket
            self.reset_pd_mb_bucket()

        # update old prcp_fac in order that it can be updated
        # again ...
        self._prcp_fac = new_prcp_fac

    @property
    def melt_f(self):
        """ prints the _melt_f """
        return self._melt_f

    @melt_f.setter
    def melt_f(self, new_melt_f):
        """ sets new melt_f and if TIModel_Sfc_Type resets the pd_mb and buckets
        """
        # first update self._melt_f
        self._melt_f = new_melt_f

        if type(self) == TIModel_Sfc_Type:
            # if the prcp_fac is set to another value,
            # need to reset pd_mb_annual, pd_mb_monthly
            # and also reset pd_bucket
            self.reset_pd_mb_bucket()
            # In addition, we need to recompute here once the melt_f buckets
            # (as they depend on self._melt_f)
            # IMPORTANT: this has to be done AFTER self._melt_f got updated!!!
            self.recompute_melt_f_buckets()

    # same for temp_bias:
    @property
    def temp_bias(self):
        return self._temp_bias

    @temp_bias.setter
    def temp_bias(self, new_temp_bias):
        self.temp += new_temp_bias - self._temp_bias

        if type(self) == TIModel_Sfc_Type:
            # if the prcp_fac is set to another value,
            # need to reset pd_mb_annual, pd_mb_monthly
            # and also reset pd_bucket
            self.reset_pd_mb_bucket()

        # update old temp_bias in order that it can be updated again ...
        self._temp_bias = new_temp_bias

    def historical_climate_qc_mod(self, gdir,
                                  climate_qc_months=3):
        """ Check the "quality" of climate data and correct it if needed.

        Similar to historical_climate_qc from oggm.core.climate but checks
        that climate that is used in TIModels directly

        This forces the climate data to have at least one month of melt
        per year at the terminus of the glacier (i.e. simply shifting
        temperatures up
        when necessary), and at least one month where accumulation is possible
        at the glacier top (i.e. shifting the temperatures down).

        This has a similar effect as introducing a temperature bias

        (deprecated at the moment, i.e. we don't apply it for any workflow)
        """
        if type(self) == TIModel_Sfc_Type:
            # if the prcp_fac is set to another value,
            # need to reset pd_mb_annual, pd_mb_monthly
            # and also reset pd_bucket
            self.reset_pd_mb_bucket()
        # Parameters
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
        """ Climate information at given heights.

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
        year : float
            float hydro year from what both, year and month, are taken if climate_type is monthly.
            hence year 2000 -> y=2000, m = 1, & year = 2000.09, y=2000, m=2 ...
            which corresponds to the real year 1999 and months October or November
            if hydro year starts in October

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
                # todo: exectime the line code below is "a bit" expensive in TIModel
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
        # Read time series
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
            # todo: exectime the line code below is quite expensive in TIModel
            grad_temp *= (heights.repeat(len(pok)).reshape(grad_temp.shape) -
                          self.ref_hgt)
            temp2d = np.atleast_2d(itemp).repeat(npix, 0) + grad_temp

            # temp_for_melt is computed separately depending on mb_type
            # todo: exectime the line code below is quite expensive in TIModel
            temp2dformelt = self._get_tempformelt(temp2d, pok)

            # Compute solid precipitation from total precipitation
            prcp = np.atleast_2d(iprcp).repeat(npix, 0)
            fac = 1 - (temp2d - self.t_solid) / (self.t_liq - self.t_solid)
            # line code below also quite expensive!
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

    def get_2d_avg_annual_air_hydro_temp(self, heights, year=None,
                                         avg_climate = False, hs = 15,
                                         hydro_m = 10):
        raise NotImplementedError('not tested and only useful'
                                  ' for estimating refreezing')
        # WIP
        # (I won't work on this for the next half a year, but leave
        # it inside as a template)
        # only computes avg annual air temperature in hydrological years,
        # this is necessary for the refreezing estimate of Woodward et al.1997
        # only tested for NH here
        # IMPORTANT: this does not take into account that months
        # have different amount of days (should I include the weighting as
        # it is inside of PyGEM??? [see: https://github.com/drounce/PyGEM/blob/ac619bd1fd862b93a01068a0887efa2a97478b99/pygem/utils/_funcs.py#L10 )
        # pok = np.where(self.years == year)[0]
        if avg_climate:
            year0 = year - hs
            year1 = year + hs
        else:
            year0 = year
            year1 = year
        if self.mb_type != 'mb_real_daily':
            pok_begin = np.where((self.years == year0-1) & (self.months == hydro_m))[0]
            pok_end = np.where((self.years == year1) & (self.months == hydro_m-1))[0]
            pok_hydro = np.arange(pok_begin, pok_end + 1, 1)
        else:
            pok_begin = np.where((self.years == year0-1) & (self.months == hydro_m))[0][0]
            pok_end = np.where((self.years == year1) & (self.months == hydro_m-1))[0][-1]
            pok_hydro = np.arange(pok_begin, pok_end + 1, 1)
        assert self.months[pok_hydro[0]] == hydro_m
        assert self.months[pok_hydro[-1]] == hydro_m - 1
        if len(pok_hydro) < 1:
            raise ValueError('Year {} not in record'.format(int(year)))
        # Read time series
        # (already temperature bias and precipitation factor corrected!)
        itemp = self.temp[pok_hydro]
        igrad = self.grad[pok_hydro]

        heights = np.asarray(heights)
        npix = len(heights)

        grad_temp = np.atleast_2d(igrad).repeat(npix, 0)
        if self.mb_type != 'mb_real_daily' and hydro_m == 10 and not avg_climate:
            assert np.all(self.months[pok_hydro] == np.array([10, 11, 12,  1,  2,  3,  4,  5,  6,  7,  8,  9]))
        if len(pok_hydro) != 12 and self.mb_type != 'mb_real_daily' and not avg_climate:
            warnings.warn('something goes wrong with amount of entries'
                          'per year')
        grad_temp *= (heights.repeat(len(pok_hydro)).reshape(grad_temp.shape) -
                      self.ref_hgt)
        temp2d = np.atleast_2d(itemp).repeat(npix, 0) + grad_temp

        return temp2d

    def _get_2d_monthly_climate(self, heights, year=None):
        # only works with real daily climate data!
        # (is used by get_monthly_mb of TIModel)
        # comment: as it is not used in TIModel_Sfc_type it could also directly go inside of TIModel ?!
        if self.mb_type == 'mb_real_daily':
            return self._get_climate(heights, 'monthly', year=year)
        else:
            raise InvalidParamsError('_get_2d_monthly_climate works only\
                                     with mb_real_daily as mb_type!!!')

    def get_monthly_climate(self, heights, year=None):
        if self.mb_type == 'mb_real_daily':
            # first get the climate data
            t, tfmelt, prcp, prcpsol = self._get_climate(heights, 'monthly',
                                                         year=year)
            return (t.mean(axis=1), tfmelt.sum(axis=1),
                    prcp.sum(axis=1), prcpsol.sum(axis=1))
        else:
            return self._get_climate(heights, 'monthly', year=year)
            # if it is mb_real_daily, the data has daily resolution (2d array then)

    def get_daily_climate(self, heights, year=None):
        raise NotImplementedError('look at _get_2d_daily_climate instead')

    def _get_2d_annual_climate(self, heights, year):
        return self._get_climate(heights, 'annual', year=year)

    def _get_2d_daily_climate(self, heights, year = None):
        return self._get_climate(heights, 'annual', year=year)

    def _get_tempformelt(self, temp, pok):
        """ Helper function to compute tempformelt to avoid code duplication
        in get_monthly_climate() and _get2d_annual_climate()

        comment: I can't use  _get_tempformelt outside the class, but sometimes this could be useful.
        If using this again outside of this class, need to remove the "self",
        such as for 'mb_climate_on_height' in climate.py, that has no self....
        (would need to change temp, t_melt ,temp_std, mb_type, N, loop)

        Parameters
        -------
            temp: ndarray
                temperature time series
            pok: ndarray
                indices of time series

        Returns
        -------
        (tempformelt)
        """

        tempformelt_without_std = temp - self.t_melt

        # computations change only if 'mb_pseudo_daily' as mb_type!
        if self.mb_type == 'mb_monthly' or self.mb_type == 'mb_real_daily':
            tempformelt = tempformelt_without_std
        elif self.mb_type == 'mb_pseudo_daily':
            itemp_std = self.temp_std[pok]

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
                tempformelt_with_std = np.full(shape_tfm, np.nan)
                for h in np.arange(0, np.shape(tempformelt_without_std)[0]):
                    h_tfm_daily_ = np.atleast_2d(tempformelt_without_std[h, :])
                    h_tempformelt_daily = h_tfm_daily_ + z_std
                    clip_min(h_tempformelt_daily, 0, out=h_tempformelt_daily)
                    h_tempformelt_monthly = h_tempformelt_daily.mean(axis=0)
                    tempformelt_with_std[h, :] = h_tempformelt_monthly
            tempformelt = tempformelt_with_std

        else:
            raise InvalidParamsError('mb_type can only be "mb_monthly,\
                                     mb_pseudo_daily or mb_real_daily" ')
        # replace all values below zero to zero
        # todo: exectime this is also quite expensive
        clip_min(tempformelt, 0, out=tempformelt)

        return tempformelt

    def get_annual_climate(self, heights, year=None):
        """Annual climate information at given heights.

        Note that prcp is corrected with the precipitation factor and that
        all other model biases (temp and prcp) are applied.

        same as in OGGM default but only for a single elevation band flowline

        Attention: temperature for melt is either in sum over monthly
        mean or in annual sum (if real_daily)!

        Parameters
        -------
        heights: ndarray
            the altitudes at which the mass-balance will be computed (has to be set)
        year: int
            hydro integer year (has to be set)

        Returns
        -------
        (temp, tempformelt, prcp, prcpsol)
        """
        t, tfmelt, prcp, prcpsol = self._get_2d_annual_climate(heights, year)
        return (t.mean(axis=1), tfmelt.sum(axis=1),
                prcp.sum(axis=1), prcpsol.sum(axis=1))

    def get_specific_winter_mb(self, heights, year=None, #begin_winter_mb=None, end_winter_mb=None,
                               widths=None, add_climate=False, period_from_wgms=False,
                               **kwargs_monthly_mb):
        """outputs specific winter MB in kg/m2. The actual winter time period can be (at the moment) either
        default October 1st to April 30th  or the one as observed by the WGMS
        (different for each glacier and observation year)

        todo: copy this and create a similar get_specific_summer_mb() -
         -> and maybe a get_specific_hydro_mb() (which would be the doing first get_specific_winter_mb
         and then get_specific_summer_mb

        Parameters
        -------
        heights: ndarray
            the altitudes at which the mass-balance will be computed (has to be set)
        year: int
            hydro integer year, (e.g. 2020 means winter 2019/2020),  (has to be set)
        #begin_winter_mb : float or np.array
        #    deprecated at the moment
        #    if several years, has to be an array. Corresponds to yearmonthday of the starting day of the winter MB.
        #    If perfect, should be YYYY1001 .
        #end_winter_mb : float or np.array
        #    deprecated at the moment
        #    if several years, has to be an array. Corresponds to yearmonthday of the end day of the winter MB.
        #    If perfect, should be YYYY0430 .
        widths : ndarray
            widths that correspond to the given heights
            (if not given specific MB is estimated without weighting over the widths which is actually not wanted,
            -> that's why you have to set the widths!)
        add_climate : bool
            default is False. If True, climate (temperature, temp_for_melt, prcp, prcp_solid) are also given as output.
            Prcp and temp_for_melt as sum over winter months, temperature as mean.
        period_from_wgms : bool
            if we compute MB by using the same observed time periods as the WGMS does (hence: time period is different
            for each year and glacier). Default is False (computing instead fixed date winter MB
            on northern Hemisphere, always from October 1st - April 30th).
        **kwargs_monthly_mb:
            todo: do I need that?

        Returns
        -------
        either just specific winter mb , or (spec_winter_mb, temp, tempformelt, prcp, prcpsol). Note that temp. is the
        mean temperature and all other variables are the sum over the winter months. Attention: temperature for melt
        is either in sum over monthly mean winter months or in annual sum (if real_daily)!
        """
        # replace this to oggm-sample-data path and use utils.file_downloader
        if period_from_wgms:
            _, path = utils.get_wgms_files()
            oggm_updated = False
            if oggm_updated:
                _, path = utils.get_wgms_files()
                pd_mb_overview = pd.read_csv(
                    path[:-len('/mbdata')] + '/mb_overview_seasonal_mb_time_periods_20220301.csv',
                    index_col='Unnamed: 0')
            else:
                #path_mbsandbox = MBsandbox.__file__[:-len('/__init__.py')]
                #pd_mb_overview = pd.read_csv(path_mbsandbox + '/data/mb_overview_seasonal_mb_time_periods_20220301.csv',
                #                            index_col='Unnamed: 0')
                #fp = utils.file_downloader('https://cluster.klima.uni-bremen.de/~lschuster/ref_glaciers'+
                #                                '/data/mb_overview_seasonal_mb_time_periods_20220301.csv')
                fp = 'https://cluster.klima.uni-bremen.de/~lschuster/ref_glaciers/data/mb_overview_seasonal_mb_time_periods_20220301.csv'
                pd_mb_overview = pd.read_csv(fp, index_col='Unnamed: 0')
            pd_mb_overview = pd_mb_overview[pd_mb_overview['at_least_5_winter_mb']]

            pd_mb_overview_sel_gdir = pd_mb_overview.loc[pd_mb_overview.rgi_id == self.fl.rgi_id]
            pd_mb_overview_sel_gdir_yr = pd_mb_overview_sel_gdir.loc[pd_mb_overview_sel_gdir.Year == year]
            ### starting period
            m_start = pd_mb_overview_sel_gdir_yr['BEGIN_PERIOD'].astype(np.datetime64).iloc[0].month
            d_start = pd_mb_overview_sel_gdir_yr['BEGIN_PERIOD'].astype(np.datetime64).iloc[0].day
            m_start_days_in_month = pd_mb_overview_sel_gdir_yr['BEGIN_PERIOD'].astype(np.datetime64).iloc[0].days_in_month
            # ratio of 1st month that we want to estimate?
            # if d_start is 1 -> ratio should be 1 --> the entire month should be added to the winter MB
            ratio_m_start = 1 - (d_start-1)/m_start_days_in_month
            ### end period
            m_end = pd_mb_overview_sel_gdir_yr['END_WINTER'].astype(np.datetime64).iloc[0].month + 1
            m_end_days_in_month = pd_mb_overview_sel_gdir_yr['END_WINTER'].astype(np.datetime64).iloc[0].days_in_month
            d_end = pd_mb_overview_sel_gdir_yr['END_WINTER'].astype(np.datetime64).iloc[0].day
            # ratio of last month that we want to estimate?
            # if d_end == m_end_days_in_month, then the entire month should be used
            ratio_m_end = d_end/m_end_days_in_month

        else:
            if self.hemisphere == 'nh':
                m_start = 10
                m_end = 4 + 1  # until end of April
            else:
                # just a bit arbitrarily
                m_start = 4
                m_end = 10 + 1
            ratio_m_start = 1
            ratio_m_end = 1

        if widths is None:
            raise InvalidParamsError('need to set widths to get correct specific winter MB')

        if len(np.atleast_1d(year)) > 1:
            if isinstance(self, TIModel_Sfc_Type):
                year_spinup = np.append(year[0] - 1, year)
                # first pre-compute all MB, then take only those months that are wished!
                self.get_specific_mb(heights, year=year_spinup, widths=widths)
                # they are saved, so computational time should not increase much

            if add_climate:
                out = []
                t_list = []
                tfm_list = []
                prcp_list = []
                prcp_sol_list = []
                for k in np.arange(0, len(year), 1):
                    out_k = self.get_specific_winter_mb(heights, year=year[k],
                                                        widths=widths,
                                                        add_climate=add_climate,
                                                        period_from_wgms=period_from_wgms,
                                                        **kwargs_monthly_mb)
                    out.append(out_k[0])
                    for numi, listi, in enumerate([t_list, tfm_list, prcp_list, prcp_sol_list]):
                        listi.append(out_k[numi+1])
                return (np.asarray(out), np.asarray(t_list), np.asarray(tfm_list),
                        np.asarray(prcp_list), np.asarray(prcp_sol_list))


            else:
                out = [self.get_specific_winter_mb(heights, year=year[k],
                                                   widths=widths,
                                                   add_climate=add_climate,
                                                   period_from_wgms=period_from_wgms,
                                                   **kwargs_monthly_mb) for k in np.arange(0, len(year), 1)]
            return np.asarray(out)

        #if year == 1980 and isinstance(self, TIModel_Sfc_Type):
        #    # need to do a "special" spinup!
        #    for m in np.arange(1, m_start, 1):
        #        floatyr = utils.date_to_floatyear(year-1, m)
        #        self.get_monthly_mb(heights, year=floatyr)
        mbs_winter = 0
        if add_climate:
            t_winter_sum = 0
            tfm_winter_sum = 0
            prcp_winter_sum = 0
            prcp_sol_winter_sum = 0
        yr_changes = m_end < m_start
        if yr_changes:
            m_winter_mb = np.concatenate([np.arange(m_start, 13, 1), np.arange(1, m_end, 1)])
        else:
            m_winter_mb = np.arange(m_start, m_end, 1)
        for m in m_winter_mb:
            if (m in np.arange(m_start, 13, 1)) and (yr_changes):
                floatyr = utils.date_to_floatyear(year-1, m)
            else:
                floatyr = utils.date_to_floatyear(year, m)
            out = self.get_monthly_mb(heights, year=floatyr, add_climate=add_climate)
            if m == m_winter_mb[0]:
                ratio = ratio_m_start
            elif m == m_winter_mb[-1]:
                ratio = ratio_m_end
            else:
                # take the entire months if these
                # are not starting or ending months
                ratio = 1
            if add_climate:
                out, t, tfm, prcp, prcp_sol = out
                t_winter_sum += t * ratio
                tfm_winter_sum += tfm * ratio
                prcp_winter_sum += prcp * ratio
                prcp_sol_winter_sum += prcp_sol * ratio
            mbs_winter += out * ratio
            #mbs_winter += mb_winter_m
        mbs_winter = mbs_winter * self.SEC_IN_MONTH * self.rho
        if add_climate:
            # we took the sum of the temperature of the winter months
            # so now we need to divide by the amount of months
            m_length_corrected = len(m_winter_mb) - 2 + ratio + ratio # use the ratio of the winter MBs
            t_winter_mean = np.array(t_winter_sum)/m_length_corrected
            # need to correct the winter temp. mean
            return (np.average(mbs_winter, weights=widths), t_winter_mean,
                    np.array(tfm_winter_sum), np.array(prcp_winter_sum), np.array(prcp_sol_winter_sum))
        else:
            return np.average(mbs_winter, weights=widths)

   # def get_specific_summer_mb(self, heights, year=None, widths=None, add_climate=False, **kwargs_monthly_mb):
   #     """outputs specific summer MB in kg/m2.
   #     This is not yet tested and only works for northern Hemisphere at the moment.
   #     year corresponds to hydro "integer" year
   #
   #     WORK IN PROCESS -> adapt it if get_specific_winter_mb is corrected the right way!
   #     """
   #     raise NotImplementedError('todo: has to be copied / adapted from get_specific_winter_mb')
   #     m_start = 5 # start in May
   #     m_end = 10  # until end of September
   #     mbs_summer = 0
   #     if widths is None:
   #         raise InvalidParamsError('need to set widths to get correct specific winter MB')
   #     for m in np.arange(m_start, m_end , 1):
   #         floatyr = utils.date_to_floatyear(year, m)
   #         mbs_summer += self.get_monthly_mb(heights, year=floatyr, **kwargs_monthly_mb)
   #     mbs_summer = mbs_summer * self.SEC_IN_MONTH * self.rho
   #     if add_climate:
   #         raise NotImplementedError('TODO: add_climate has to be implemented!')
   #     return np.average(mbs_summer, weights=widths)

    def get_specific_summer_mb(self, heights, year=None,  # begin_winter_mb=None, end_winter_mb=None,
                               widths=None, add_climate=False, period_from_wgms=False,
                               **kwargs_monthly_mb):
        """outputs specific summer MB in kg/m2. The actual summer time period can be (at the moment) either
        default May 1st to end of September  or the one as observed by the WGMS
        (different for each glacier and observation year)

        todo: maybe also create a get_specific_hydro_mb() (which would be the doing first get_specific_winter_mb
         and then get_specific_summer_mb

        Parameters
        -------
        heights: ndarray
            the altitudes at which the mass-balance will be computed (has to be set)
        year: int
            hydro integer year,  (has to be set)
        widths : ndarray
            widths that correspond to the given heights
            (if not given specific MB is estimated without weighting over the widths which is actually not wanted,
            -> that's why you have to set the widths!)
        add_climate : bool
            default is False. If True, climate (temperature, temp_for_melt, prcp, prcp_solid) are also given as output.
            Prcp and temp_for_melt as sum over winter months, temperature as mean.
        period_from_wgms : bool
            if we compute MB by using the same observed time periods as the WGMS does (hence: time period is different
            for each year and glacier). Default is False (computing instead fixed date winter MB
            on northern Hemisphere, always from May 1st - end of September).
        **kwargs_monthly_mb:
            todo: do I need that?

        Returns
        -------
        either just specific summer mb , or (spec_summmer_mb, temp, tempformelt, prcp, prcpsol). Note that temp. is the
        mean temperature and all other variables are the sum over the winter months. Attention: temperature for melt
        is either in sum over monthly mean winter months or in annual sum (if real_daily)!
        """
        # replace this to oggm-sample-data path and use utils.file_downloader
        if period_from_wgms:
            _, path = utils.get_wgms_files()
            oggm_updated = False
            if oggm_updated:
                _, path = utils.get_wgms_files()
                pd_mb_overview = pd.read_csv(
                    path[:-len('/mbdata')] + '/mb_overview_seasonal_mb_time_periods_20220301.csv',
                    index_col='Unnamed: 0')
            else:
                # path_mbsandbox = MBsandbox.__file__[:-len('/__init__.py')]
                # pd_mb_overview = pd.read_csv(path_mbsandbox + '/data/mb_overview_seasonal_mb_time_periods_20220301.csv',
                #                            index_col='Unnamed: 0')
                #fp = utils.file_downloader('https://cluster.klima.uni-bremen.de/~lschuster/ref_glaciers' +
                #                           '/data/mb_overview_seasonal_mb_time_periods_20220301.csv')
                fp = 'https://cluster.klima.uni-bremen.de/~lschuster/ref_glaciers/data/mb_overview_seasonal_mb_time_periods_20220301.csv'
                pd_mb_overview = pd.read_csv(fp, index_col='Unnamed: 0')
            pd_mb_overview = pd_mb_overview[pd_mb_overview['at_least_5_winter_mb']]
            pd_mb_overview_sel_gdir = pd_mb_overview.loc[pd_mb_overview.rgi_id == self.fl.rgi_id]
            pd_mb_overview_sel_gdir_yr = pd_mb_overview_sel_gdir.loc[pd_mb_overview_sel_gdir.Year == year]
            ### starting period
            m_start = pd_mb_overview_sel_gdir_yr['END_WINTER'].astype(np.datetime64).iloc[0].month
            d_start = pd_mb_overview_sel_gdir_yr['END_WINTER'].astype(np.datetime64).iloc[0].day
            m_start_days_in_month = pd_mb_overview_sel_gdir_yr['END_WINTER'].astype(np.datetime64).iloc[0].days_in_month
            # ratio of 1st month that we want to estimate?
            # if d_start is 1 -> ratio should be 1 --> the entire month should be added to the winter MB
            ratio_m_start = 1 - (d_start - 1) / m_start_days_in_month
            ### end period
            m_end = pd_mb_overview_sel_gdir_yr['END_PERIOD'].astype(np.datetime64).iloc[0].month + 1
            m_end_days_in_month = pd_mb_overview_sel_gdir_yr['END_PERIOD'].astype(np.datetime64).iloc[0].days_in_month
            d_end = pd_mb_overview_sel_gdir_yr['END_PERIOD'].astype(np.datetime64).iloc[0].day
            # ratio of last month that we want to estimate?
            # if d_end == m_end_days_in_month, then the entire month should be used
            ratio_m_end = d_end / m_end_days_in_month

        else:
            m_start = 5
            m_end = 9 + 1  # until end of April
            ratio_m_start = 1
            ratio_m_end = 1

        if widths is None:
            raise InvalidParamsError('need to set widths to get correct specific winter MB')

        if len(np.atleast_1d(year)) > 1:
            if isinstance(self, TIModel_Sfc_Type):
                year_spinup = np.append(year[0] - 1, year)
                # first pre-compute all MB, then take only those months that are wished!
                self.get_specific_mb(heights, year=year_spinup, widths=widths)
                # they are saved, so computational time should not increase much

            if add_climate:
                out = []
                t_list = []
                tfm_list = []
                prcp_list = []
                prcp_sol_list = []
                for k in np.arange(0, len(year), 1):
                    out_k = self.get_specific_summer_mb(heights, year=year[k],
                                                        widths=widths,
                                                        add_climate=add_climate,
                                                        period_from_wgms=period_from_wgms,
                                                        **kwargs_monthly_mb)
                    out.append(out_k[0])
                    for numi, listi, in enumerate([t_list, tfm_list, prcp_list, prcp_sol_list]):
                        listi.append(out_k[numi + 1])
                return (np.asarray(out), np.asarray(t_list), np.asarray(tfm_list),
                        np.asarray(prcp_list), np.asarray(prcp_sol_list))


            else:
                out = [self.get_specific_summer_mb(heights, year=year[k],
                                                   widths=widths,
                                                   add_climate=add_climate,
                                                   period_from_wgms=period_from_wgms,
                                                   **kwargs_monthly_mb) for k in np.arange(0, len(year), 1)]
            return np.asarray(out)

        # if year == 1980 and isinstance(self, TIModel_Sfc_Type):
        #    # need to do a "special" spinup!
        #    for m in np.arange(1, m_start, 1):
        #        floatyr = utils.date_to_floatyear(year-1, m)
        #        self.get_monthly_mb(heights, year=floatyr)
        mbs_summer = 0
        if add_climate:
            t_summer_sum = 0
            tfm_summer_sum = 0
            prcp_summer_sum = 0
            prcp_sol_summer_sum = 0
        yr_changes = m_end < m_start
        if yr_changes:
            m_summer_mb = np.concatenate([np.arange(m_start, 13, 1), np.arange(1, m_end, 1)])
        else:
            m_summer_mb = np.arange(m_start, m_end, 1)
        for m in m_summer_mb:
            if (m in np.arange(m_start, 13, 1)) and (yr_changes):
                floatyr = utils.date_to_floatyear(year - 1, m)
            else:
                floatyr = utils.date_to_floatyear(year, m)
            out = self.get_monthly_mb(heights, year=floatyr, add_climate=add_climate)
            if m == m_summer_mb[0]:
                ratio = ratio_m_start
            elif m == m_summer_mb[-1]:
                ratio = ratio_m_end
            else:
                # take the entire months if these
                # are not starting or ending months
                ratio = 1
            if add_climate:
                out, t, tfm, prcp, prcp_sol = out
                t_summer_sum += t * ratio
                tfm_summer_sum += tfm * ratio
                prcp_summer_sum += prcp * ratio
                prcp_sol_summer_sum += prcp_sol * ratio
            mbs_summer += out * ratio
            # mbs_winter += mb_winter_m
        mbs_summer = mbs_summer * self.SEC_IN_MONTH * self.rho
        if add_climate:
            # we took the sum of the temperature of the winter months
            # so now we need to divide by the amount of months
            m_length_corrected = len(m_summer_mb) - 2 + ratio + ratio  # use the ratio of the winter MBs
            t_winter_mean = np.array(t_summer_sum) / m_length_corrected
            # need to correct the winter temp. mean
            return (np.average(mbs_summer, weights=widths), t_winter_mean,
                    np.array(tfm_summer_sum), np.array(prcp_summer_sum), np.array(prcp_sol_summer_sum))
        else:
            return np.average(mbs_summer, weights=widths)


class TIModel(TIModel_Parent):
    """ Temperature-Index model without surface type distinction (but different climate resolution
    and temperature lapse rate options (child class of TIModel_Parent)"""

    def __init__(self, *args, **kwargs):
        # same as those from TIModel_Parent
        super().__init__(*args, **kwargs)

    def get_monthly_mb(self, heights, year=None, add_climate=False,
                       **kwargs):
        """ computes annual mass balance in m of ice per second!

        Attention year is here in hydro float year

        year has to be given as float hydro year from what the month is taken,
        hence year 2000 -> y=2000, m = 1, & year = 2000.09, y=2000, m=2 ...
        which corresponds to the real year 1999 and months October or November
        if hydro year starts in October
        """
        # todo: can actually remove **kwargs???
        # comment: get_monthly_mb and get_annual_mb are only different
        #  to OGGM default for mb_real_daily

        if self.mb_type == 'mb_real_daily':
            # get 2D values, dependencies on height and time (days)
            out = self._get_2d_monthly_climate(heights, year)
            t, temp2dformelt, prcp, prcpsol = out
            # (days per month)
            # dom = 365.25/12  # len(prcpsol.T)
            fact = 12/365.25
            # attention, I should not use the days of years as the melt_f is
            # per month ~mean days of that year 12/daysofyear
            # to have the same unit of melt_f, which is
            # the monthly temperature sensitivity (kg /m² /mth /K),
            mb_daily = prcpsol - self.melt_f * temp2dformelt * fact

            mb_month = np.sum(mb_daily, axis=1)
            # more correct than using a mean value for days in a month
            #warnings.warn('there might be a problem with SEC_IN_MONTH'
            #              'as February changes amount of days inbetween the years'
            #              ' see test_monthly_glacier_massbalance()')
        else:
            # get 1D values for each height, no dependency on days
            t, temp2dformelt, prcp, prcpsol = self.get_monthly_climate(heights, year=year)
            mb_month = prcpsol - self.melt_f * temp2dformelt

        # residual is in mm w.e per year, so SEC_IN_MONTH ... but mb_month
        # should be per month!
        mb_month -= self.residual * self.SEC_IN_MONTH / self.SEC_IN_YEAR
        # this is for mb_pseudo_daily otherwise it gives the wrong shape
        mb_month = mb_month.flatten()
        if add_climate:
            if self.mb_type == 'mb_real_daily':
                # for run_with_hydro want to get monthly output (sum of daily),
                # if we want daily output in run_with_hydro need to directly use get_daily_mb()
                prcp = prcp.sum(axis=1)
                prcpsol = prcpsol.sum(axis=1)
                t = t.mean(axis=1)
                temp2dformelt = temp2dformelt.sum(axis=1)
            if self.mb_type == 'mb_pseudo_daily':
                temp2dformelt = temp2dformelt.flatten()
            return (mb_month / SEC_IN_MONTH / self.rho, t, temp2dformelt,
                    prcp, prcpsol)
        # instead of SEC_IN_MONTH, use instead len(prcpsol.T)==daysinmonth
        return mb_month / self.SEC_IN_MONTH / self.rho

    def get_annual_mb(self, heights, year=None, add_climate=False,
                      **kwargs):
        """ computes annual mass balance in m of ice per second !
        Attention: temperature for melt of add_climate
        is either in sum over monthly mean or in annual sum (if real_daily)!
        """
        # todo: can actually remove **kwargs???
        # comment: get_monthly_mb and get_annual_mb are only different
        # to OGGM default for mb_real_daily

        t, temp2dformelt, prcp, prcpsol = self._get_2d_annual_climate(heights,
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
        # todo: exectime -> this line code is quite expensive in TIModel (e.g. ~12% of computation time)
        mb_annual = np.sum(prcpsol - self.melt_f * temp2dformelt*fact,
                           axis=1)
        # and also this one is expensive
        mb_annual = (mb_annual - self.residual) / self.SEC_IN_YEAR / self.rho
        if add_climate:
            # for run_with_hydro, want climate as sum over year (for prcp...)
            return (mb_annual, t.mean(axis=1), temp2dformelt.sum(axis=1),
                    prcp.sum(axis=1), prcpsol.sum(axis=1))
        return mb_annual

    def get_daily_mb(self, heights, year=None,
                     add_climate=False):
        """computes daily mass balance in m of ice per second

        attention this accounts as well for leap years, hence
        doy are not 365.25 as in get_annual_mb but the amount of days the year
        has in reality!!! (needed for hydro model of Sarah Hanus)

        year has to be given as float hydro year from what the month is taken,
        hence year 2000 -> y=2000, m = 1, & year = 2000.09, y=2000, m=2 ...
        which corresponds to the real year 1999 and months October or November
        if hydro year starts in October

        (implemented in order that Sarah Hanus can use daily input and also gets daily output)

        """

        # todo: make this more user friendly
        if type(year) == float:
            raise InvalidParamsError('here year has to be the integer year')
        else:
            pass

        if self.mb_type == 'mb_real_daily':
            # get 2D values, dependencies on height and time (days)
            out = self._get_2d_daily_climate(heights, year)
            t, temp2dformelt, prcp, prcpsol = out
            # days of year
            doy = len(prcpsol.T)  # 365.25
            # assert doy > 360
            # to have the same unit of melt_f, which is
            # the monthly temperature sensitivity (kg /m² /mth /K),
            melt_f_daily = self.melt_f * 12/doy
            mb_daily = prcpsol - melt_f_daily * temp2dformelt

            # mb_month = np.sum(mb_daily, axis=1)
            # more correct than using a mean value for days in a month
            warnings.warn('be cautiuous when using get_daily_mb and test yourself if it does '
                          'what you expect')

            # residual is in mm w.e per year, so SEC_IN_MONTH .. but mb_daily
            # is per day!
            mb_daily -= self.residual * doy
            # this is for mb_daily otherwise it gives the wrong shape
            # mb_daily = mb_month.flatten()
            # instead of SEC_IN_MONTH, use instead len(prcpsol.T)==daysinmonth
            if add_climate:
                # these are here daily values as output for the entire year
                # might need to be changed a bit to be used for run_with_hydro
                return (mb_daily / self.SEC_IN_DAY / self.rho,
                        t, temp2dformelt, prcp, prcpsol)
            return mb_daily / self.SEC_IN_DAY / self.rho
        else:
            raise InvalidParamsError('get_daily_mb works only with'
                                     'mb_real_daily as mb_type!')

    def get_specific_daily_mb(self, heights=None, widths=None, year=None):
        """ returns specific daily mass balance in kg m-2 day

        (implemented in order that Sarah Hanus can use daily input and also gets daily output)
        """
        if len(np.atleast_1d(year)) > 1:
            out = [self.get_specific_daily_mb(heights=heights, widths=widths,
                                              year=yr) for yr in year]
            return np.asarray(out)

        mb = self.get_daily_mb(heights, year=year)
        spec_mb = np.average(mb * self.rho * SEC_IN_DAY, weights=widths, axis=0)
        assert len(spec_mb) > 360
        return spec_mb

class TIModel_Sfc_Type(TIModel_Parent):
    """ Temperature-Index model with surface type distinction using a bucket system
    (child class of TIModel_Parent)

    this works only with calendar years (i.e. hydro_month = 1)

    """

    def __init__(self, gdir, melt_f,
                 melt_f_ratio_snow_to_ice=0.5,
                 melt_f_update='annual',
                 spinup_yrs=6,
                 tau_e_fold_yr=1,  #0.5,
                 melt_f_change='linear',
                 check_availability=True,
                 interpolation_optim=False,
                 hbins=np.nan,
                 **kwargs):

        # doc_TIModel_Sfc_Type =
        """
        Other terms are equal to TIModel_Parent!
        The following parameters are initialized specifically only for TIModel_Sfc_Type

        Parameters
        ----------
        melt_f_ratio_snow_to_ice : float
            ratio of snow melt factor to ice melt factor (from 0 to 1),
            if set to 1, it is equal to no surface type distinction
            (default is 0.5 same as in GloGEM)
        melt_f_update : str, default: 'annual'
            'annual' or 'monthly' , how
            often the melt factor should be updated and how
            many buckets exist. If annual it uses 1 snow
            and 5 firn buckets with yearly melt factor updates. If monthly,
            each month the snow is ageing over the amount of spinup years
            (default 6yrs, i.e., 72 months).
        spinup_yrs : int
            amount of years to compute as spinup (if spinup=True in
            get_annual_mb or get_monthly_mb) before computing the actual year.
            Default is 6 years. After these 6 years every bucket has the opportunity
            to be filled (as we have 1 snow bucket and 5 firn buckets, or
            72 monthly buckets)
        tau_e_fold_yr : float
            default is 1, (before it was set to 0.5!!!)
            only used if melt_f_change is 'neg_exp',
            it describes how fast the snow melt
            factor approximates to the ice melt factor via
            melt_f = melt_f_ice + (melt_f_snow - melt_f_ice)* np.exp(-time_yr/tau_e_fold_yr) # s: in months
            do not set to 0, otherwise the melt_f of the first bucket is np.nan
        melt_f_change : str
            default is linear,
            how the snow melt factor changes to the ice melt factor, either 'linear'
            or 'neg_exp' (see `tau_e_fold_yr` for the equation)
        check_availability: boolean
            default is True, checks if year / month has already been computed, if true it gives that output
            (which comes from either pd_mb_monthly or pd_mb_annual). This ensures to give always the same
            values for a specific year/month and is also faster.
            However, in case of random_climate or constant_climate, we have to set this to False!!!
        interpolation_optim:
            default is False. If it is true, a cte bucket profile (via spinup) is estimated and then kept cte,
            so we just do once the computation of the mass balance for different heights
             & reuse it then again without considering that the surface type changes over time
             ... as we need to use an emulation anyway this is deprecated (but was once used for equilibration runs)
             todo: maybe remove?
        hbins:
            default is np.nan. Here you can set different height bins for the sfc type distinction method.
            Best is to keep it at the default. Only necessary when using e.g. ConstantMBModel
            todo: which other specific things?
        kwargs:
            those keyword arguments that are equal to TIModel_Parent
            todo: or should I just brutally copy the docs from TIModel_Parent?
        """
        # todo Fabi: how can I best inherit the init docs from TIModel_Parent to be visible as well in TIModel_Sfc_Type
        #self.__init__.__doc__ = super().__init__.__doc__ + doc_TIModel_Sfc_Type

        super().__init__(gdir, melt_f, **kwargs)

        self.hbins = hbins
        self.interpolation_optim = interpolation_optim

        self.tau_e_fold_yr = tau_e_fold_yr
        if melt_f_change == 'neg_exp':
            assert tau_e_fold_yr > 0, "tau_e_fold_yr has to be above zero!"
        self.melt_f_change = melt_f_change
        assert melt_f_change in ['linear', 'neg_exp'], "melt_f_change has to be either 'linear' or 'neg_exp'"
        # ratio of snow melt_f to ice melt_f
        self.melt_f_ratio_snow_to_ice = melt_f_ratio_snow_to_ice
        self.melt_f_update = melt_f_update
        self.spinup_yrs = spinup_yrs
        # amount of bucket and bucket naming depends on melt_f_update:
        if self.melt_f_update == 'annual':
            self.buckets = ['snow', 'firn_yr_1', 'firn_yr_2', 'firn_yr_3',
                            'firn_yr_4', 'firn_yr_5']
        elif self.melt_f_update == 'monthly':
            # for each month over 6 years a bucket-> in total 72!
            self.buckets = np.arange(0, 12 * 6, 1).tolist()
        else:
            raise InvalidParamsError('melt_f_update can either be annual or monthly!')
        # first bucket: if annual it is 'snow', if monthly update it is 0
        self.first_snow_bucket = self.buckets[0]

        self.columns = self.buckets + ['delta_kg/m2']
        # TODO: maybe also include snow_delta_kg/m2, firn_yr_1_delta_kg/m2...
        #  (only important if I add refreezing or a densification scheme)
        # comment: I don't need an ice bucket because this is assumed to be "infinite"
        # (instead just have a 'delta_kg/m2' bucket)

        # save the inversion height to later check if the same height is applied!!!
        self.inv_heights = self.fl.surface_h
        self.check_availability = check_availability

        # container template (has to be updatable -> pd_mb_template is property/setter thing)
        # use the distance_along_flowline as index
        self._pd_mb_template = pd.DataFrame(0, index=self.fl.dx_meter * np.arange(self.fl.nx),
                                            columns=[]) # exectime-> directly addind columns here should be faster
        self._pd_mb_template.index.name = 'distance_along_flowline'

        # bucket template:
        # make a different template for the buckets, because we want directly the right columns inside
        self._pd_mb_template_bucket = pd.DataFrame(0, index=self.fl.dx_meter * np.arange(self.fl.nx),
                                            columns=self.columns)  # exectime-> directly addind columns here should be faster
        self._pd_mb_template_bucket.index.name = 'distance_along_flowline'

        # storage containers for monthly and annual mb
        # columns are the months or years respectively
        # IMPORTANT: those need other columns
        self.pd_mb_monthly = self._pd_mb_template.copy()
        self.pd_mb_annual = self._pd_mb_template.copy()

        # bucket containers with buckets as columns
        pd_bucket = self._pd_mb_template_bucket.copy()
        # exectime comment:  this line is quite expensive -> actually this line can be removed as
        # self._pd_mb_template was defined more clever !!!
        # pd_bucket[self.columns] = 0
        # I don't need a total_kg/m2 because I don't know it anyway
        # as we do this before inversion!



        # storage container for buckets (in kg/m2)
        # columns are the buckets
        # (6*12 or 6 buckets depending if melt_f_update is monthly or annual)
        self.pd_bucket = pd_bucket

    @property
    def melt_f_buckets(self):
        # interpolated from snow melt_f to ice melt_f by using melt_f_ratio_snow_to_ice and tau_e_fold_yr
        # to get the bucket melt_f. Need to do this here and not in init, because otherwise it does not get updated.
        # Or I would need to set melt_f as property / setter function that updates self.melt_f_buckets....

        # exectime too long --> need to make that more efficient:
        #  only "update" melt_f_buckets and recompute them if melt_f
        #  is changed (melt_f_ratio_snow_to_ice, tau_e_fold_yr and amount of buckets do not change after instantiation!)
        # Only if _melt_f_buckets does not exist, compute it out of self.melt_f ... and the other stuff!!
        try:
            return self._melt_f_buckets
        except:
            if self.melt_f_change == 'linear':
                self._melt_f_buckets = dict(zip(self.buckets + ['ice'],
                                                np.linspace(self.melt_f * self.melt_f_ratio_snow_to_ice,
                                                            self.melt_f, len(self.buckets)+1)
                                                ))
            elif self.melt_f_change == 'neg_exp':
                time = np.linspace(0, 6, len(self.buckets + ['ice']))
                melt_f_snow = self.melt_f_ratio_snow_to_ice * self.melt_f
                self._melt_f_buckets = dict(zip(self.buckets + ['ice'],
                                                self.melt_f + (melt_f_snow - self.melt_f) * np.exp(-time/self.tau_e_fold_yr)
                                                ))

            # at the moment we don't allow to set externally the melt_f_buckets but this could be added later ...
            return self._melt_f_buckets

    def recompute_melt_f_buckets(self):
        # This is called always when self.melt_f is updated  (the other self.PARAMS are not changed after instantiation)
        # IMPORTANT:
        # - we need to use it after having updated self._melt_f inside of melt_f setter  (is in TIModel_Sfc_Type)!!!
        # - we need to use self._melt_f -> as this is what is before changed inside of self.melt_f!!!

        # interpolate from snow melt_f to ice melt_f by using melt_f_ratio_snow_to_ice and tau_e_fold_yr
        # to get the bucket melt_f. Need to do this here and not in init, because otherwise it does not get updated.

        # comment exectime: made that more efficient. Now it only "updates" melt_f_buckets and recomputes them if melt_f
        # is changed (melt_f_ratio_snow<_to_ice, tau_e_fold_yr and amount of buckets do not change after instantiation)!
        if self.melt_f_change == 'linear':
            self._melt_f_buckets = dict(zip(self.buckets + ['ice'],
                                            np.linspace(self._melt_f * self.melt_f_ratio_snow_to_ice,
                                                        self._melt_f, len(self.buckets)+1)
                                            ))
        elif self.melt_f_change == 'neg_exp':
            time = np.linspace(0, 6, len(self.buckets + ['ice']))
            melt_f_snow = self.melt_f_ratio_snow_to_ice * self._melt_f
            self._melt_f_buckets = dict(zip(self.buckets + ['ice'],
                                            self._melt_f +
                                            (melt_f_snow - self._melt_f) * np.exp(-time/self.tau_e_fold_yr)
                                            ))

    def reset_pd_mb_bucket(self,
                           init_model_fls='use_inversion_flowline'):
        """ resets pandas mass balance bucket monthly and annual dataframe as well as the bucket dataframe
        (basically removes all years/months and empties the buckets so that everything is as if freshly instantiated)
        It is called when setting new melt_f, prcp_fac, temp_bias, ...
        """
        # comment: do I need to reset sometimes only the buckets,
        #  or only mb_monthly??? -> for the moment I don't think so
        if self.interpolation_optim:
            self._pd_mb_template_bucket = pd.DataFrame(0, index=self.hbins[::-1],
                                                columns=self.columns)
            self._pd_mb_template_bucket.index.name = 'hbins_height'
            self._pd_mb_template = pd.DataFrame(0, index=self.hbins[::-1],
                                                columns=[])
            self._pd_mb_template.index.name = 'distance_along_flowline'

            self.pd_mb_monthly = self._pd_mb_template.copy()
            self.pd_mb_annual = self._pd_mb_template.copy()

            pd_bucket = self._pd_mb_template_bucket.copy()
            # pd_bucket[self.columns] = 0
            self.pd_bucket = pd_bucket
        else:
            if init_model_fls != 'use_inversion_flowline':
                #fls = gdir.read_pickle('model_flowlines')
                self.fl = copy.deepcopy(init_model_fls)[-1]
                self.mod_heights = self.fl.surface_h

                # container templates
                # use the distance_along_flowline as index
                self._pd_mb_template = pd.DataFrame(0, index=self.fl.dx_meter * np.arange(self.fl.nx),
                                                           columns=[])
                self._pd_mb_template.index.name = 'distance_along_flowline'
                self._pd_mb_template_bucket = pd.DataFrame(0, index=self.fl.dx_meter * np.arange(self.fl.nx),
                                                    columns=self.columns)
                self._pd_mb_template_bucket.index.name = 'distance_along_flowline'


            self.pd_mb_monthly = self._pd_mb_template.copy()
            self.pd_mb_annual = self._pd_mb_template.copy()

            pd_bucket = self._pd_mb_template_bucket.copy()
            # this expensive code line below  is not anymore necessary if we define columns directly when creating
            # self._pd_mb_template during instantiation!!! (took up to 15% of
            # pd_bucket[self.columns] = 0
            self.pd_bucket = pd_bucket

    def _add_delta_mb_vary_melt_f(self, heights, year=None,
                                  climate_resol='annual',
                                  add_climate=False):
        """ get the mass balance of all buckets and add/remove the
         new mass change for each bucket

         this is probably where most computational time is needed

         Parameters
         ----
         heights: np.array()
            heights of elevation flowline, at the monent only inversion height works!!!
            Only put heights inside that fit to "distance_along_flowline.
         year: int or float
            if melt_f_update='annual', integer calendar float year,
            if melt_f_update=='monthly', year should be a calendar float year,
            so it corresponds to 1 month of a specific year, i
         climate_resol: 'annual' or 'monthly
            if melt_f_update -> climate_resol has to be always monthly
            but if annual melt_f_update can have either annual climate_resol (if get_annual_mb is used)
            or monthly climate_resol (if get_monthly_mb is used)
         add_climate : bool
            default is False. If True, climate (temperature, temp_for_melt, prcp, prcp_solid) are also given as output
            practical for get_monthly_mb with add_climate=True, as then we do not need to compute climate twice!!!

         """

        # new: get directly at the beginning the pd_bucket and convert it to np.array, at the end it is reconverted and
        # saved again under self.pd_bucket:
        np_pd_bucket = self.pd_bucket.values  # last column is delta_kg/m2

        if self.melt_f_update == 'monthly':
            if climate_resol != 'monthly':
                raise InvalidWorkflowError('Need monthly climate_resol if melt_f_update is monthly')
            if not isinstance(year, float):
                raise InvalidParamsError('Year has to be the calendar float year '
                                         '_add_delta_mb_vary_melt_f with monthly melt_f_update,'
                                         'year needs to be a float')

        #########
        # todo: if I put heights inside that are not fitting to
        #  distance_along_flowline, it can get problematic, I can only check it by testing if
        #  length of the heights are the same as distance along
        #  flowline of pd_bucket dataframe
        #  but the commented check is not feasible as it does not work for all circumstances
        #  however, we check at other points if the height array has at least the right order!
        # if len(heights) != len(self.fl.dis_on_line):
        #    raise InvalidParamsError('length of the heights should be the same as '
        #                             'distance along flowline of pd_bucket dataframe,'
        #                             'use for heights e.g. ...fl.surface_h()')
        ##########

        # check if the first bucket is be empty if:
        condi1 = climate_resol == 'annual' and self.melt_f_update == 'annual'
        # from the last year, all potential snow should be no firn, and from this year, the
        # new snow is not yet added, so snow buckets should be empty
        # or
        condi2 = self.melt_f_update == 'monthly'
        # from the last month, all potential snow should have been update and should be now
        # in the next older month bucket
        # or 1st month with annual update and get_monthly_mb
        # !!! in case of climate_resol =='monthly' but annual update,
        # first_snow_bucket is not empty if mc != 1!!!
        _, mc = floatyear_to_date(float(year))
        condi3 = climate_resol == 'monthly' and self.melt_f_update == 'annual' and mc == 1
        if condi1 or condi2 or condi3:
            # if not np.any(self.pd_bucket[self.first_snow_bucket] == 0):
            # comment exectime: this code here below is faster than the one from above
            np_pd_values_first_snow_bucket = np_pd_bucket[:, 0]  # -> corresponds to first_snow_bucket
            # todo exectime : this still takes long to evaluate !!! is there a betterway ???
            if len(np_pd_values_first_snow_bucket[np_pd_values_first_snow_bucket != 0]) > 0:
                raise InvalidWorkflowError('the first snow buckets should be empty in this use case '
                                           'but it is not, try e.g. to do '
                                           'reset_pd_mb_buckets() before and then rerun the task')

        # Let's do the same as in get_annual_mb of TIModel but with varying melt_f:
        # first get the climate
        if climate_resol == 'annual':
            t, temp2dformelt, prcp, prcpsol = self._get_2d_annual_climate(heights, year)
            if add_climate:
                raise NotImplementedError('Not implemented. Need to check here in the code'
                                      'that the dimensions are right. ')
            # t = t.mean(axis=1)
            # temp2dformelt = temp2dformelt.sum(axis=1)
            # prcp = prcp.sum(axis=1)
            # prcpsol = prcpsol.sum(axis=1)
        elif climate_resol == 'monthly':
            if isinstance(type(year), int):
                raise InvalidWorkflowError('year should be a float with monthly climate resolution')
            # year is here float year, so it corresponds to 1 month of a specific year
            t, temp2dformelt, prcp, prcpsol = self.get_monthly_climate(heights, year)

        # put first all solid precipitation into first snow bucket  (actually corresponds to first part of the loop)
        if climate_resol == 'annual':
            # if melt_f_update annual: treat snow as the amount of solid prcp over that year
            # first add all solid prcp amount to the bucket
            # (the amount of snow that has melted in the same year is taken away in the for loop)
            np_pd_bucket[:, 0] = prcpsol.sum(axis=1)  # first snow bucket should be filled with solid prcp
            # at first, remaining temp for melt energy is all temp for melt (tfm)
            # when looping over the buckets this term will get gradually smaller until the remaining tfm corresponds
            # to the potential ice melt
            remaining_tfm = temp2dformelt.sum(axis=1)
            # delta has to be set to the solid prcp (the melting part comes in during the for loop)
            # self.pd_bucket['delta_kg/m2']
            np_pd_bucket[:, -1] = prcpsol.sum(axis=1)

        elif climate_resol == 'monthly':
            # if self.melt_f_update is  'monthly': snow is the amount
            # of solid prcp over that month, ...
            # SLOW
            # self.pd_bucket[self.first_snow_bucket] = self.pd_bucket[self.first_snow_bucket].values.copy() + prcpsol.flatten() # .sum(axis=1)
            # faster
            np_pd_bucket[:, 0] = np_pd_bucket[:, 0] + prcpsol.flatten()
            remaining_tfm = temp2dformelt.flatten()
            # the last column corresponds to self.pd_bucket['delta_kg/m2'] -> np. is faster
            np_pd_bucket[:, -1] = prcpsol.flatten()

        # need the right unit
        if self.mb_type == 'mb_real_daily':
            fact = 12/365.25
        else:
            fact = 1

        # now do the melting processes for each bucket in a loop:
        # how much tempformelt (tfm) would we need to remove all snow, firn of each bucket
        # in the case of snow it corresponds to tfm to melt all solid prcp.
        # To convert from kg/m2 in the buckets to tfm [K], we use the melt_f values
        # of each bucket accordingly. No need of .copy(), because directly changed by divisor (/)
        # this can actually be done outside the for-loop!!!

        # just get the melt_f_buckets once
        melt_f_buckets = self.melt_f_buckets
        # do not include delta_kg/m2 in pd_buckets, and also do not include melt_f of 'ice'
        tfm_to_melt_b = np_pd_bucket[:, :-1] / (np.array(list(melt_f_buckets.values()))[:-1]*fact)  # in K
        # need to transpose it to have the right shape
        tfm_to_melt_b = tfm_to_melt_b.T

        ### loop:
        # todo Fabi: exectime , can I use another kind of loop -> e.g. "apply" or sth.?
        #  maybe I can convert this first into a np.array then do the loop there and reconvert it at the end
        #  again into a pd.DataFrame for better handling?

        # faster: (old pandas code commented below)
        np_delta_kg_m2 = np_pd_bucket[:, -1]
        for e, b in enumerate(self.buckets):
            # there is no ice bucket !!!

            # now recompute buckets mass:
            # kg/m2 of that bucket that is not lost (that has not melted) in that year / month (i.e. what will
            # stay in that bucket after that month/year -> and gets later on eventually updated -> older)
            # if all lost -> set it to 0
            # -> to get this need to reconvert the tfm energy unit into kg/m2 by using the right melt_factor
            # e.g. at the uppest layers there is new snow added ...
            # todo Fabi: exectime now one of slowest lines (e.g. ~8% of computational time)
            not_lost_bucket = utils.clip_min(tfm_to_melt_b[e] - remaining_tfm, 0) * melt_f_buckets[b] * fact

            # if all of the bucket is removed (not_lost_bucket=0), how much energy (i.e. tfm) is left
            # to remove mass from other buckets?
            # remaining tfm to melt older firn layers -> for the next loop ...
            remaining_tfm = utils.clip_min(remaining_tfm - tfm_to_melt_b[e], 0)
            # exectime: is this faster?
            # find sth. faster

            # in case of ice, the remaining_tfm is only used to update once again delta_kg/m2
            # todo : exectime time Fabi check if I can merge not_lost_bucket and remaining_tfm computation
            #  (so that we don't do the clip_min twice)?
            #  or maybe I should only compute that if not_lost_bucket == 0:

            # amount of kg/m2 lost in this bucket -> this will be added to delta_kg/m2
            # corresponds to: not yet updated total bucket - amount of not lost mass of that bucket
            # comment: the line below was very expensive (Number 1, e.g. 22% of computing time)-> rewrite to numpy?
            # .copy(), not necessary because minus
            # self.pd_bucket['delta_kg/m2'] += not_lost_bucket - self.pd_bucket[b].values
            # comment:  now faster
            # np_delta_kg_m2 = (np_delta_kg_m2 + not_lost_bucket - self.pd_bucket[b].values)
            # but we want it even faster: removed all panda stuff
            np_delta_kg_m2 = (np_delta_kg_m2 + not_lost_bucket - np_pd_bucket[:, e])

            # update pd_bucket with what has not melted from the bucket
            # how can we make this faster?
            # self.pd_bucket[b] = not_lost_bucket
            # new: removed all panda stuff
            # exectime: not so slow anymore (e.g. 3% of computational time)
            np_pd_bucket[:, e] = not_lost_bucket

            # new since autumn 2021: if all the remaining_tfm is zero, then go out of the loop,
            # to make the code faster!
            # e.g. in winter this should mean that we don't loop over all buckets (as tfm is very small)
            # if np.all(remaining_tfm == 0):
            # is this faster?
            if len(remaining_tfm[remaining_tfm != 0]) == 0:
                break

        # we assume that the ice bucket is infinite,
        # so everything that could be melted is included inside of delta_kg/m2
        # that means all the remaining tfm energy is used to melt the infinite ice bucket
        # self.pd_bucket['delta_kg/m2'] = np_delta_kg_m2 -remaining_tfm * self.melt_f_buckets['ice'] * fact
        # use the np.array and recompute delta_kg_m2
        np_pd_bucket[:, -1] = np_delta_kg_m2 -remaining_tfm * melt_f_buckets['ice'] * fact
        # create pd_bucket again at the end
        # todo Fabi: exectime -> this is also now quite time expensive (~8-13%)
        #  but on the other hand, it takes a lot of time to
        #  create a bucket system without pandas at all !
        #  if we would always stay in the self.pd_bucket.values system instead of np_pd_bucket,
        #  we would not need to recreate the pd_bucket, maybe this would be then faster?
        self.pd_bucket = pd.DataFrame(np_pd_bucket, columns=self.pd_bucket.columns.values,
                                      index=self.pd_bucket.index)
        if add_climate:
            return self.pd_bucket, t, temp2dformelt, prcp, prcpsol
        else:
            return self.pd_bucket

        # old:
        # loop_via_pd = False
        # if loop_via_pd:
        #     for e, b in enumerate(self.buckets):
        #         # there is no ice bucket !!!
        #
        #         # now recompute buckets mass:
        #         # kg/m2 of that bucket that is not lost (that has not melted) in that year / month (i.e. what will
        #         # stay in that bucket after that month/year -> and gets later on eventually updated ->older)
        #         # if all lost -> set it to 0
        #         # -> to get this need to reconvert the tfm energy unit into kg/m2 by using the right melt_factor
        #         # e.g. at the uppest layers there is new snow added ...
        #         not_lost_bucket = utils.clip_min(tfm_to_melt_b[e] - remaining_tfm, 0) * self.melt_f_buckets[b] * fact
        #
        #         # if all of the bucket is removed (not_lost_bucket=0), how much energy (i.e. tfm) is left
        #         # to remove mass from other buckets?
        #         # remaining tfm to melt older firn layers -> for the next loop ...
        #         remaining_tfm = utils.clip_min(remaining_tfm - tfm_to_melt_b[e], 0)
        #         # exectime: is this faster?
        #         # find sth. faster
        #
        #         # in case of ice, the remaining_tfm is only used to update once again delta_kg/m2
        #
        #         # amount of kg/m2 lost in this bucket -> this will be added to delta_kg/m2
        #         # corresponds to: not yet updated total bucket - amount of not lost mass of that bucket
        #         # comment: this line was very expensive (Number 1, e.g. 22% of computing time)-> rewrite to numpy?
        #         # self.pd_bucket['delta_kg/m2'] += not_lost_bucket - self.pd_bucket[b].values # .copy(), not necessary because minus
        #         self.pd_bucket['delta_kg/m2'] = (self.pd_bucket['delta_kg/m2'].values
        #                                          + not_lost_bucket - self.pd_bucket[
        #                                              b].values)  # .copy(), not necessary because minus
        #
        #         # is this faster:
        #         # self.pd_bucket['delta_kg/m2'] = not_lost_bucket - self.pd_bucket[b].values # .copy(), not necessary because minus
        #
        #         # update pd_bucket with what has not melted from the bucket
        #         # comment: exectime this line was very expensive (Number 2, e.g. 11% of computing time)->
        #         # how can we make this faster?
        #         self.pd_bucket[b] = not_lost_bucket
        #
        #         # new since autumn 2021: if all the remaining_tfm is zero, then go out of the loop,
        #         # to make the code faster!
        #         # e.g. in winter this should mean that we don't loop over all buckets (as tfm is very small)
        #         if np.all(remaining_tfm == 0):
        #             break
        #     # we assume that the ice bucket is infinite,
        #     # so everything that could be melted is included inside of delta_kg/m2
        #     # that means all the remaining tfm energy is used to melt the infinite ice bucket
        #     self.pd_bucket['delta_kg/m2'] += - remaining_tfm * self.melt_f_buckets['ice'] * fact




    # comment: should this be a setter ??? because no argument ...
    # @update_buckets.setter
    def _update(self):
        """ this is called by get_annual_mb or get_monthly_mb after one year/one month to update
        the buckets as they got older

        if it is called on monthly or annual basis depends if we set melt_f_update to monthly or to annual!


        new: I removed a loop here because I can just rename the bucket column names in order that they are
        one bucket older, then set the first_snow_bucket to 0, and the set pd_bucket[kg/m2] to np.nan
        just want to shift all buckets from one column to another one .> can do all at once via self.buckets[::-1].iloc[1:] ???
        """
        # first convert pd_bucket to np dataframe -> at the end reconvert to pd.DataFrame
        np_pd_bucket = self.pd_bucket.values

        # this here took sometimes 1.1%
        # if np.any(np.isnan(self.pd_bucket['delta_kg/m2'])):
        # now faster (~0.2%)
        if np.any(np.isnan(np_pd_bucket[:, -1])): # delta_kg/m2 is in the last column !!!
            raise InvalidWorkflowError('the buckets have been updated already, need'
                                       'to add_delta_mb first')

        # exectime: this here took sometimes 7.5% of total computing time !!!
        # if np.any(self.pd_bucket[self.buckets] < 0):
        #    raise ValueError('the buckets should only have positive values')
        # a bit faster now: but can be still ~5%
        # if self.pd_bucket[self.buckets].values.min() < 0:
        # np_pd_values = self.pd_bucket[self.buckets].values
        # buckets are in all columns except the last column!!!
        if len(np_pd_bucket[:, :-1][np_pd_bucket[:, :-1] < 0]) > 0:
            raise ValueError('the buckets should only have positive values')

        # old snow without last bucket and without delta_kg/ms
        # all buckets to update: np_pd_bucket[1:-1]
        # kg/m2 in those buckets before the update: self.bucket[0:-2]
        pd_bucket_val_old = np_pd_bucket[:, :-2]
        len_h = len(self.pd_bucket.index)
        # exectime: <0.1%
        np_updated_bucket = np.concatenate([np.zeros(len_h).reshape(len_h, 1),  # first bucket should be zero
                                            pd_bucket_val_old,
                                            np.full((len_h, 1), np.nan)],  # kg/m2 bucket should be np.nan
                                            axis=1)  # , np.nan(len(self.pd_bucket.index))])
        # as we add a zero array in the front, we don't have any snow more, what has been in the youngest bucket
        # went into the next older bucket and so on!!!

        # recreate pd_bucket:
        # todo Fabi exectime: if we want to save more time would need to restructure all -> slowest line
        #  (9-12% computational time)
        #  if we would always stay in the self.pd_bucket.values system instead of np_pd_bucket,
        #  we would not need to recreate the pd_bucket, maybe this would be then faster?
        self.pd_bucket = pd.DataFrame(np_updated_bucket, columns=self.pd_bucket.columns.values,
                                      index=self.pd_bucket.index)
        return self.pd_bucket

        # OLD stuff
        # other idea -> but did not work!
        # just rename the columns : and add afterwards the snow bucket inside
        # self.pd_bucket = self.pd_bucket.drop(columns=['delta_kg/m2', self.buckets[-1]])
        # problem here: need to have it in the right order (first snow, then firn buckets -> otherwise tests fail)
        # self.pd_bucket.columns = self.buckets[1:]

        # pd_version of above -> was still too slow!
        # exectime: this line was very expensive (sometimes even number 1 with 40% of computing time)
        # self.pd_bucket[self.buckets[1:]] = self.pd_bucket[self.buckets[:-1]].values

        # self.pd_bucket[self.first_snow_bucket] = 0
        # self.pd_bucket['delta_kg/m2'] = np.nan

        # the new version should work as the old version (e.g. this test here should check it: test_sfc_type_update)
        # the other loop version will be removed
        # else:
        #     for e, b in enumerate(self.buckets[::-1]):
        #         # start with updating oldest snow bucket (in reversed order!= ...
        #         if b != self.first_snow_bucket:
        #             # e.g. update ice by using old firn_yr_5 ...
        #             # @Fabi: do I need the copy here?
        #             self.pd_bucket[b] = self.pd_bucket[self.buckets[::-1][e + 1]].copy()
        #             # we just overwrite it so we don't need to reset it to zero
        #             # self.pd_bucket[self.buckets[::-1][e+1]] = 0 #pd_bucket['firn_yr_4']
        #         elif b == self.first_snow_bucket:
        #             # the first_snow bucket is set to 0 after the update
        #             # (if annual update there is only one snow bucket)
        #             self.pd_bucket[b] = 0
        #     # reset delta_kg/m2 to make clear that it is updated
        #     self.pd_bucket['delta_kg/m2'] = np.nan

    def get_annual_mb(self, heights, year=None, unit='m_of_ice',
                      bucket_output=False, spinup=True,
                      add_climate=False,
                      auto_spinup=True,
                      **kwargs):
        """
        computes annual mass balance in m of ice per second

        Parameters
        ----------
        heights : np.array
            at the moment works only with inversion heights!
        year: int
            integer CALENDAR year! if melt_f_update='monthly', it will loop over each month.
        unit : str
            default is 'm of ice', nothing else implemented at the moment!
            comment: include option of metre of glacier where the different densities
             are taken into account but in this case would need to add further columns in pd_buckets
             like: snow_delta_kg/m2 ... and so on (only important if I add refreezing or a densification scheme)
        bucket_output: bool (default is False)
            if True, returns as second output the pd.Dataframe with details about
            amount of kg/m2 for each bucket and height grid point,
            set it to True to visualize how the buckets change over time or for testing
            (these are the buckets before they got updated for the next year!)
        spinup : bool (default is True)
            if a spinup is applied to fill up sfc type buckets beforehand
            (default are 6 years, check under self.spinup_yrs)
        add_climate: bool (default is False)
            for run_with_hydro (not yet implemented!)
            todo: implement and test it
        auto_spinup: bool (default is true)
            if True, it automatically computes the spinup years (default is 6) beforehand (however in this case,
            these 6 years at the beginning, although saved in pd_mb_annual, had no spinup...)
            todo: maybe need to add a '_no_spinup' to those that had no spinup?
             or save them in a separate pd.DataFrame?
        **kwargs:
            other stuff passed to get_monthly_mb or to _add_delta_mb_vary_melt_f
            **kwargs necessary to take stuff we don't use (like fls...)

        """

        # when we set spinup_yrs to zero, then there should be no spinup occurring, even if spinup
        # is set to True
        if self.spinup_yrs == 0:
            spinup = False

        if len(self.pd_mb_annual.columns) > 0:
            if self.pd_mb_annual.columns[0] > self.pd_mb_annual.columns[-1]:
                raise InvalidWorkflowError('need to run years in ascending order! Maybe reset first!')
        # dirty check that the given heights are in the right order
        assert heights[0] > heights[-1], "heights should be in descending order!"
        # otherwise pd_buckets does not work ...
        # below would be a more robust test, but this is not compatible to all use cases:
        # np.testing.assert_allclose(heights, self.inv_heights,
        #                           err_msg='the heights should correspond to the inversion heights',
        #                           )
        # np.testing.assert_allclose(heights, self.mod_heights)

        # we just convert to integer without checking ...
        year = int(year)
        # comment: there was some reason why I commented that code again
        # if not isinstance(year, int):
        #    raise InvalidParamsError('Year has to be the full year for get_annual_mb,'
        #                             'year needs to be an integer')
        if year < 1979+self.spinup_yrs:
            # most climatic data starts in 1979, so if we want to
            # get the first 6 years can not use a spinup!!!
            # (or would need to think about sth. else, but not so
            # important right now!!!)
            spinup = False
        if year in self.pd_mb_annual.columns and self.check_availability:
            # print('takes existing annual mb')
            # if that year has already been computed, and we did not change any parameter settings
            # just get the annual_mb without redoing all the computations
            mb_annual = self.pd_mb_annual[year].values
            if bucket_output:
                raise InvalidWorkflowError('if you want to output the buckets, you need to do'
                                           'reset_pd_mb_bucket() and rerun')

            if add_climate:
                t, temp2dformelt, prcp, prcpsol = self._get_2d_annual_climate(heights,
                                                                              year)
                return (mb_annual, t.mean(axis=1), temp2dformelt.sum(axis=1),
                        prcp.sum(axis=1), prcpsol.sum(axis=1))
                # raise NotImplementedError('TODO: add_climate has to be implemented!')
            else:
                return mb_annual
        else:
            # do we need to do the spinup beforehand
            # if any of the default 6 spinup years before was not computed,
            # we need to reset all and do the spinup properly -> (i.e. condi = True)
            if self.melt_f_update == 'annual':
                # so we really need to check if every year exists!!!
                condis = []
                for bef in np.arange(1, self.spinup_yrs, 1):
                    condis.append(int(year - bef) not in self.pd_mb_annual.columns)
                condi = np.any(np.array(condis))
            elif self.melt_f_update == 'monthly':
                try:
                    # check if first and last month of each spinup years before exists
                    condis = []
                    for bef in np.arange(1, self.spinup_yrs, 1):
                        condis.append(date_to_floatyear(year - bef, 1) not in self.pd_mb_monthly.columns)
                        condis.append(date_to_floatyear(year - bef, 12) not in self.pd_mb_monthly.columns)
                    condi = np.any(np.array(condis))
                except:
                    condi = True

            if condi and spinup and auto_spinup:
                # reset and do the spinup:
                # comment: I think we should always reset when
                # doing the spinup and not having computed year==2000
                # problem: the years before year should not be saved up (in get_annual_mb)!
                # (because they are not computed right!!! (they don't have a spinup)
                self.reset_pd_mb_bucket()
                for yr in np.arange(year-self.spinup_yrs, year):
                    self.get_annual_mb(heights, year=yr, unit=unit,
                                       bucket_output=False,
                                       spinup=False, add_climate=False,
                                       auto_spinup=False,
                                       **kwargs)

            if spinup:
                # check if the spinup years had been computed (should be inside of pd_mb_annual)
                # (it should have been computed automatically if auto_spinup=True)
                for bef in np.arange(1, 6, 1):
                    if int(year-bef) not in self.pd_mb_annual.columns.values:
                        raise InvalidWorkflowError('need to do get_annual_mb of all spinup years'
                                                   '(default is 6) beforehand')
            if self.melt_f_update == 'annual':
                self.pd_bucket = self._add_delta_mb_vary_melt_f(heights, year=year,
                                                                climate_resol='annual')
                mb_annual = ((self.pd_bucket['delta_kg/m2'].values
                              - self.residual) / self.SEC_IN_YEAR / self.rho)
                if bucket_output:
                    # copy because we want to output the bucket that is not yet updated!!!
                    pd_bucket = self.pd_bucket.copy()
                # update to one year later ... (i.e. put the snow / firn into the next older bucket)
                self._update()
                # save the annual mb
                # todo Fabi exectime: --> would need to restructure code to remove pd stuff
                self.pd_mb_annual[year] = mb_annual
                # this is done already if melt_f_update is monthly (see get_monthly_mb for m == 12)
            elif self.melt_f_update == 'monthly':
                # will be summed up over each month by getting pd_mb_annual
                # that is set in december (month=12)
                for m in np.arange(1, 13, 1):
                    floatyear = date_to_floatyear(year, m)
                    out = self.get_monthly_mb(heights, year=floatyear, unit=unit,
                                              bucket_output=bucket_output, spinup=spinup,
                                              add_climate=add_climate,
                                              auto_spinup=auto_spinup,
                                              **kwargs)
                    if bucket_output and m == 12:
                        pd_bucket = out[1]
                # get mb_annual that is produced by
                mb_annual = self.pd_mb_annual[year].values
            if bucket_output:
                return mb_annual, pd_bucket
            else:
                if add_climate:
                    t, temp2dformelt, prcp, prcpsol = self._get_2d_annual_climate(heights,
                                                                                  year)
                    return (mb_annual, t.mean(axis=1), temp2dformelt.sum(axis=1),
                            prcp.sum(axis=1), prcpsol.sum(axis=1))
                    #raise NotImplementedError('TODO: add_climate has to be implemented!')
                else:
                    return mb_annual

            #todo
            # if add_climate:
            #    return (mb_annual, t.mean(axis=1), tmelt.sum(axis=1),
            #            prcp.sum(axis=1), prcpsol.sum(axis=1))

    def get_monthly_mb(self, heights, year=None, unit='m_of_ice',
                       bucket_output=False, spinup=True,
                       add_climate=False,
                       auto_spinup=True,
                       **kwargs):
        """
        computes monthly mass balance in m of ice per second!

        year should be the calendar float year,
        so it corresponds to 1 month of a specific year

        Parameters
        ----------
        heights : np.array
            at the moment only works with inversion heights!
        year: int
            year has to be given as CALENDAR float year from what the (integer) year and month is taken,
            hence year 2000 -> y=2000, m = 1, & year = 2000.09, y=2000, m=2 ...
        unit : str
            default is 'm of ice', nothing else implemented at the moment!
            TODO: include option of metre of glacier where the different densities
             are taken into account but in this case would need to add further columns in pd_buckets
             like: snow_delta_kg/m2 ... (only important if I add refreezing or a densification scheme)
        bucket_output: bool (default is False)
            if True, returns as second output the pd.Dataframe with details about
            amount of kg/m2 for each bucket and height grid point,
            set it to True to visualize how the buckets change over time or for testing
            (these are the buckets before they got updated for the next year!)
        spinup : bool (default is True)
            if a spinup is applied to fill up sfc type buckets beforehand
            (default are 6 years, check under self.spinup_yrs), if month is not January,
            also checks if the preceding monts of thar year have been computed
        add_climate: bool (default is False)
            for run_with_hydro or for get_specific_winter_mb to get winter precipitation
            If True, climate (temperature, temp_for_melt, prcp, prcp_solid) are also given as output.
            prcp and temp_for_melt as monthly sum, temp. as mean.
        auto_spinup: bool (default is true)
            if True, it automatically computes the spinup years (default is 6) beforehand and
            all months before the existing year
            (however in this case, these spinup years, although saved in pd_mb_annual, had no spinup...)
            todo: maybe need to add a '_no_spinup' to those that had no spinup?
             or save them in a separate pd.DataFrame?
        **kwargs:
            other stuff passed to get_annual_mb or to _add_delta_mb_vary_melt_f
            todo: not yet necessary, maybe later? **kwargs necessary to take stuff we don't use (like fls...)
        """

        if self.spinup_yrs == 0:
            spinup = False
        if year < 1979+self.spinup_yrs:
            # comment: most climatic data starts in 1979, so if we want to
            # get the first 6 years can not use a spinup!!!
            # (or would need to think about sth. else, but not so
            # important right now!!!)
            spinup = False

        if len(self.pd_mb_monthly.columns) > 1:
            if self.pd_mb_monthly.columns[0] > self.pd_mb_monthly.columns[-1]:
                raise InvalidWorkflowError('need to run months in ascending order! Maybe reset first!')

        assert heights[0] > heights[-1], "heights should be in descending order!"
        # below would be a more robust test, but this is not compatible to all use cases:
        # np.testing.assert_allclose(heights, self.inv_heights,
        #                           err_msg='the heights should correspond to the inversion heights',
        #                           )

        if year in self.pd_mb_monthly.columns and self.check_availability:
            # if that float year has already been computed
            # just get the saved monthly_mb
            # (but don't need to compute something or update pd_buckets)
            # print('takes existing monthly mb')
            mb_month = self.pd_mb_monthly[year].values
            if bucket_output:
                raise InvalidWorkflowError('if you want to output the buckets, you need to do'
                                           'reset_pd_mb_bucket() and rerun')
            if add_climate:
                if isinstance(type(year), int):
                    raise InvalidWorkflowError('year should be a float with monthly climate resolution')
                # year is here float year, so it corresponds to 1 month of a specific year
                t, temp2dformelt, prcp, prcpsol = self.get_monthly_climate(heights, year)
                if self.mb_type == 'mb_pseudo_daily':
                    temp2dformelt = temp2dformelt.flatten()
                return mb_month, t, temp2dformelt, prcp, prcpsol
            else:
                return mb_month
        else:
            # only allow float years
            if not isinstance(year, float):
                raise InvalidParamsError('Year has to be the calendar float year '
                                         'for get_monthly_mb,'
                                         'year needs to be a float')
            # need to somehow check if the months before that were computed as well
            # otherwise monthly mass-balance does not make sense when using sfc type distinction ...
            y, m = floatyear_to_date(year)

            ### spinup
            # find out if the spinup has to be computed
            # this is independent of melt_f_update, always need to reset if the spinup years before
            # were not computed
            try:
                # it is sufficient to check if the first month of that year is there
                # if there is any other problem it will raise an error later anyways
                condi = date_to_floatyear(y - self.spinup_yrs, 1) not in self.pd_mb_monthly.columns
            except:
                condi = True
            # if annual melt_f_update and annual_mb exist from previous years
            # (by doing , get_annual_mb), and there exist not yet the mass-balance for the next year
            # then no reset & new spinup is necessary
            # comment (Feb2022): why did I have here before:
            # condi_add = int(y + 1) in self.pd_mb_annual.columns
            # NEW: don't reset if the years before already exist (by checking pd_mb_annual)
            if condi:
                try:
                    if np.any(self.pd_mb_annual.columns[-self.spinup_yrs:] == np.arange(y - self.spinup_yrs, y, 1)):
                        no_spinup_necessary = True
                    else:
                        no_spinup_necessary = False
                except:
                    no_spinup_necessary = False
                if self.melt_f_update == 'annual' and no_spinup_necessary:
                    condi = False

            if condi and spinup and auto_spinup:
                # reset and do the spinup:
                self.reset_pd_mb_bucket()
                for yr in np.arange(y-self.spinup_yrs, y):
                    self.get_annual_mb(heights, year=yr, unit=unit, bucket_output=False,
                                       spinup=False, add_climate=False,
                                       auto_spinup=False, **kwargs)
                # need to also run the months before our actual wanted month m
                if m > 1:
                    for mb in np.arange(1, m, 1):
                        floatyear_before = date_to_floatyear(y, mb)
                        self.get_monthly_mb(heights, year=floatyear_before,
                                            bucket_output=False,
                                            spinup=False, add_climate=False,
                                            auto_spinup=False, **kwargs)

            if spinup:
                # check if the years before that had been computed
                # (if 2000, it should have been computed above)
                for bef in np.arange(1, 6, 1):
                    if int(y-bef) not in self.pd_mb_annual.columns:
                        raise InvalidWorkflowError('need to do get_monthly_mb of all spinup years '
                                                   '(default is 6) beforehand')
                # check if the months before had been computed for that year
                for mb in np.arange(1, m, 1):
                   floatyear_before = date_to_floatyear(y, mb)
                   if floatyear_before not in self.pd_mb_monthly.columns:
                       raise InvalidWorkflowError('need to get_monthly_mb of each month '
                                                  'of that year beforehand')

            # year is here in float year!
            if add_climate:
                self.pd_bucket, t, temp2dformelt, prcp, prcpsol = self._add_delta_mb_vary_melt_f(heights,
                                                                year=year,
                                                                climate_resol='monthly',
                                                                add_climate=add_climate)
            else:
                self.pd_bucket = self._add_delta_mb_vary_melt_f(heights,
                                                                year=year,
                                                                climate_resol='monthly',
                                                                add_climate=add_climate)

            # ?Fabi: if I want to have it in two lines, can I then  still put the .copy() away?
            #mb_month = self.pd_bucket['delta_kg/m2'].copy().values
            #mb_month -= self.residual * self.SEC_IN_MONTH / self.SEC_IN_YEAR
            mb_month = self.pd_bucket['delta_kg/m2'].values - \
                       self.residual * self.SEC_IN_MONTH / self.SEC_IN_YEAR
            # need to flatten for mb_pseudo_daily otherwise it gives the wrong shape
            mb_month = mb_month.flatten() / self.SEC_IN_MONTH / self.rho
            # save the mass-balance to pd_mb_monthly
            # todo Fabi exectime: -> this line is also quite expensive -> to make it faster would need
            #  to entirely remove the pd_structure!
            self.pd_mb_monthly[year] = mb_month
            self.pd_mb_monthly[year] = mb_month

            # the update happens differently fot the two cases -> so we need to differentiate here:
            if self.melt_f_update == 'annual':
                warnings.warn('get_monthly_mb with annual melt_f_update results in a different summed up annual MB'
                              'than using get_annual_mb. This is because we use bulk estimates in get_annual_mb.'
                              'Only use it when you know what you do!!! ')
                # todo: maybe better to set a NotImplementedError
                # raise NotImplementedError('get_monthly_mb works at the moment '
                #                          'only when melt_f is updated monthly')

                # if annual, only want to update at the end of the year
                if m == 12:
                    # sum up the monthly mb and update the annual pd_mb
                    # but as the mb is  m of ice per second -> use the mean !!!
                    # only works if same height ... and if
                    if bucket_output:
                        pd_bucket = self.pd_bucket.copy()
                    self._update()
                    if int(year) not in self.pd_mb_annual.columns:
                        condi = [int(c) == int(year) for c in self.pd_mb_monthly.columns]
                        # first check if we have all 12 months of the year together
                        if len(self.pd_mb_monthly.loc[:, condi].columns) != 12:
                            raise InvalidWorkflowError('Not all months were computed beforehand,'
                                                       'need to do get_monthly_mb for all months before')
                        # get all columns that correspond to that year and do the mean to get the annual estimate
                        # mean because unit is in m of ice per second
                        self.pd_mb_annual[int(year)] = self.pd_mb_monthly.loc[:, condi].mean(axis=1).values

                if bucket_output and m != 12:
                    pd_bucket = self.pd_bucket.copy()

            elif self.melt_f_update == 'monthly':

                # get the output before the update
                if bucket_output:
                    pd_bucket = self.pd_bucket.copy()
                # need to update it after each month
                self._update()
                # if December, also want to save it to the annual mb
                if m == 12:
                    # sum up the monthly mb and update the annual pd_mb
                    # run this if this year not yet inside or if we don't want to check availability
                    # e.g. for random / constant runs !!!
                    if int(year) not in self.pd_mb_annual.columns or not self.check_availability:
                        condi = [int(c) == int(year) for c in self.pd_mb_monthly.columns]
                        # first check if we have all 12 months of the year together
                        if len(self.pd_mb_monthly.loc[:, condi].columns) != 12:
                            raise InvalidWorkflowError('Not all months were computed beforehand,'
                                                       'need to do get_monthly_mb for all months before')
                        # get all columns that correspond to that year and do the mean to get the annual estimate
                        # mean because unit is in m of ice per second
                        self.pd_mb_annual[int(year)] = self.pd_mb_monthly.loc[:, condi].mean(axis=1).values

            if add_climate and not bucket_output:
                if self.mb_type == 'mb_pseudo_daily':
                    temp2dformelt = temp2dformelt.flatten()
                return mb_month, t, temp2dformelt, prcp, prcpsol
            elif bucket_output:
                return mb_month, pd_bucket
            elif add_climate and bucket_output:
                raise InvalidWorkflowError('either set bucket_output or add_climate to True, not both. '
                                           'Otherwise you have to change sth. in the code!')
            else:
                return mb_month

    def get_daily_mb(self):
        raise NotImplementedError('this has not been implemented with surface type distinction')

    def get_ela(self, year=None, **kwargs):
        """Compute the equilibrium line altitude for a given year.

        copied and adapted from OGGM main -> had to remove the invalid ELA check, as it won't work
        together with sfc type distinction. It still does not work, as the buckets are not made to be applied just
        that


        Parameters
        ----------
        year: float, optional
            the time (in the "hydrological floating year" convention)
        **kwargs: any other keyword argument accepted by self.get_annual_mb
        Returns
        -------
        the equilibrium line altitude (ELA, units: m)
        """
        if len(np.atleast_1d(year)) > 1:
            return np.asarray([self.get_ela(year=yr, **kwargs) for yr in year])

        if self.valid_bounds is None:
            raise ValueError('attribute `valid_bounds` needs to be '
                             'set for the ELA computation.')

        # Check for invalid ELAs
        #b0, b1 = self.valid_bounds
        #if (np.any(~np.isfinite(
        #        self.get_annual_mb([b0, b1], year=year, **kwargs))) or
        #        (self.get_annual_mb([b0], year=year, **kwargs)[0] > 0) or
        #        (self.get_annual_mb([b1], year=year, **kwargs)[0] < 0)):
        #    return np.nan

        def to_minimize(x):
            return (self.get_annual_mb([x], year=year, **kwargs)[0] *
                    SEC_IN_YEAR * self.rho)

        return optimization.brentq(to_minimize, *self.valid_bounds, xtol=0.1)


# copy of MultipleFlowlineMassBalance that works with TIModel
class MultipleFlowlineMassBalance_TIModel(MassBalanceModel):
    """ Adapted MultipleFlowlineMassBalance that is compatible for all TIModel classes

    TODO: do better documentation

    Handle mass-balance at the glacier level instead of flowline level.

    Convenience class doing not much more than wrapping a list of mass-balance
    models, one for each flowline.

    This was useful for real-case studies, where each flowline had a
    different MB parameters. TIModel and TIModel_Sfc_Type only works with a single flowline (elevation band flowline).
    Here we just use MultipleFlowlineMassBalance_TIModel to make it easier for "coupling" it to default OGGM stuff
    from the dynamics flowline stuff.

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
        fls :
        melt_f : float
            melt factor
            (has to be set to a value!)
        prcp-fac : float
            multiplicative precipitation factor
            (has to be set to a value)
        mb_model_class : class, optional
            the mass-balance model to use (e.g. PastMassBalance,
            ConstantMassBalance...)
        use_inversion_flowlines: bool, optional
            if True 'inversion_flowlines' instead of 'model_flowlines' will be
            used.
        input_filesuffix : str
            the file suffix of the input climate file
        bias :
            default is 0
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

            if ((issubclass(mb_model_class, TIModel_Parent))
                    or (issubclass(mb_model_class, RandomMassBalance_TIModel))
                    or (issubclass(mb_model_class, ConstantMassBalance_TIModel))
                    or (issubclass(mb_model_class, AvgClimateMassBalance_TIModel))):
                self.flowline_mb_models.append(
                    mb_model_class(gdir, melt_f, prcp_fac=prcp_fac,
                                   residual=fl_bias,
                                   baseline_climate=rgi_filesuffix,
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
        return self.flowline_mb_models[0].residual

    @bias.setter
    def bias(self, value):
        """Residual bias to apply to the original series."""
        for mbmod in self.flowline_mb_models:
            mbmod.residual = value

    def get_daily_mb(self, heights, year=None, fl_id=None, **kwargs):

        if fl_id is None:
            raise ValueError('`fl_id` is required for '
                             'MultipleFlowlineMassBalance!')

        return self.flowline_mb_models[fl_id].get_daily_mb(heights,
                                                             year=year,
                                                             **kwargs)

    def get_monthly_mb(self, heights, year=None, fl_id=None, **kwargs):

        if fl_id is None:
            raise ValueError('`fl_id` is required for '
                             'MultipleFlowlineMassBalance!')

        return self.flowline_mb_models[fl_id].get_monthly_mb(heights,
                                                             year=year,
                                                             **kwargs)

    def get_annual_mb(self, heights, year=None, fl_id=None, **kwargs):

        if fl_id is None:
            raise ValueError('`fl_id` is required for '
                             'MultipleFlowlineMassBalance!')

        return self.flowline_mb_models[fl_id].get_annual_mb(heights,
                                                            year=year,
                                                            **kwargs)

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
                        year=None, **kwargs):

        """ computes specific mass-balance for each year in [kg /m2]"""

        if heights is not None or widths is not None:
            raise ValueError('`heights` and `widths` kwargs do not work with '
                             'MultipleFlowlineMassBalance!')

        if fls is None:
            fls = self.fls

        if len(np.atleast_1d(year)) > 1:
            out = [self.get_specific_mb(fls=fls, year=yr, **kwargs) for yr in year]
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
            mb = mb_mod.get_annual_mb(fl.surface_h, year=year, fls=fls,
                                      fl_id=i, **kwargs)
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



class ConstantMassBalance_TIModel(MassBalanceModel):
    """Constant mass-balance during a chosen period.

    if interpolation_optim=True and mb_model_sub_class
    the goal is actually to create once a cte bucket profile (via spinup),
     and then leave this and do the interpolation
     -> we assume that the buckets are constant over the whole running period,
      so we just do once the computation of the mass balance for different heights
      & reuse it then again without considering that the surface type changes
      over time ... at the end we might need to use an emulation anyway so it
     is not such a problem

    This is useful for equilibrium experiments.
    """

    def __init__(self, gdir, melt_f=None, prcp_fac = None,
                 mb_model_sub_class = TIModel,
                 residual = 0,
                 baseline_climate=None,
                 input_filesuffix='default',
                 filename='climate_historical',
                 y0=None, halfsize=15,
                 interpolation_optim=False,
                 **kwargs):
        """Initialize

        Parameters
        ----------
        gdir : GlacierDirectory
            the glacier directory
        melt_f : float
            melt factor (has to be set to a value!)
        prcp-fac :
            multiplicative precipitation factor (has to be set to a value)
        mb_model_sub_class : class, optional
            the mass-balance model to use: either TIModel (default)
            or TIModel_Sfc_Type
        residual :
            default for TIModel's is 0 ! (best is to not change!)
        input_filesuffix : str
            defult is '', filesuffix of climate file that should be used
        filename : str
            climate filename, default is 'climate_historical',
        y0 : int
            the year at the center of the period of interest, needs to be set!!!
        halfsize : int, optional
            the half-size of the time window (window size = 2 * halfsize + 1),
            default are 15 years
        **kwargs :
            stuff passed to the TIModel instance!
        """
        #filename = 'climate_historical',
        #input_filesuffix = '',
        super(ConstantMassBalance_TIModel, self).__init__()

        self.interpolation_optim = interpolation_optim

        if y0 is None:
            raise InvalidWorkflowError('need to set y0 as we do not '
                                       'use tstar in this case')
        # This is a quick'n dirty optimisation
        try:
            fls = gdir.read_pickle('model_flowlines')
            h = []
            for fl in fls:
                # We use bed because of overdeepenings
                h = np.append(h, fl.bed_h)
                h = np.append(h, fl.surface_h)
            zminmax = np.round([np.min(h)-50, np.max(h)+2000])
        except FileNotFoundError:
            # in case we don't have them
            with ncDataset(gdir.get_filepath('gridded_data')) as nc:
                if np.isfinite(nc.min_h_dem):
                    # a bug sometimes led to non-finite
                    zminmax = [nc.min_h_dem-250, nc.max_h_dem+1500]
                else:
                    zminmax = [nc.min_h_glacier-1250, nc.max_h_glacier+1500]
        self.hbins = np.arange(*zminmax, step=10)
        self.valid_bounds = self.hbins[[0, -1]]
        self.y0 = y0
        self.halfsize = halfsize
        self.years = np.arange(y0-halfsize, y0+halfsize+1)
        self.hemisphere = gdir.hemisphere

        if mb_model_sub_class == TIModel_Sfc_Type:
            # was not sure how to add something to **kwargs
            kwargs2 = {'check_availability': False,
                       'interpolation_optim': interpolation_optim,
                       'hbins': self.hbins}

        else:
            interpolation_optim = True
            kwargs2 = {}

        self.mbmod = mb_model_sub_class(gdir, melt_f=melt_f,
                                         prcp_fac=prcp_fac, residual=residual,
                                         input_filesuffix=input_filesuffix,
                                        baseline_climate=baseline_climate,
                                         filename=filename,
                                         **kwargs, **kwargs2)

        self._mb_debug_container = pd.DataFrame({'yr': [], #'ryr': [],
                                                 'heights': [], 'mb': []})

    @property
    def temp_bias(self):
        """Temperature bias to add to the original series."""
        return self.mbmod.temp_bias

    @temp_bias.setter
    def temp_bias(self, value):
        """Temperature bias to add to the original series."""
        for attr_name in ['_lazy_interp_yr', '_lazy_interp_m']:
            if hasattr(self, attr_name):
                delattr(self, attr_name)
        self.mbmod.temp_bias = value

    @property
    def prcp_fac(self):
        """Precipitation factor to apply to the original series."""
        return self.mbmod.prcp_fac

    @prcp_fac.setter
    def prcp_fac(self, value):
        """Precipitation factor to apply to the original series."""
        for attr_name in ['_lazy_interp_yr', '_lazy_interp_m']:
            if hasattr(self, attr_name):
                delattr(self, attr_name)
        self.mbmod.prcp_fac = value

    def historical_climate_qc_mod(self, gdir):
        return self.mbmod.historical_climate_qc_mod(gdir)

    @property
    def residual(self):
        """Residual bias to apply to the original series."""
        return self.mbmod.residual

    @residual.setter
    def residual(self, value):
        """Residual bias to apply to the original series."""
        self.mbmod.residual = value

    def reset_pd_mb_bucket(self, init_model_fls='use_inversion_flowline'):
        if self.interpolation_optim:
            self.mbmod._pd_mb_template_bucket = pd.DataFrame(0,
                                                      index=self.hbins[::-1],
                                                      columns=self.mbmod.columns)

            self.mbmod._pd_mb_template_bucket.index.name = 'hbins_height'

            self.mbmod.pd_mb_monthly = self.mbmod._pd_mb_template.copy()
            self.mbmod.pd_mb_annual = self.mbmod._pd_mb_template.copy()

            pd_bucket = self.mbmod._pd_mb_template_bucket.copy()
            # pd_bucket[self.mbmod.columns] = 0
            self.mbmod.pd_bucket = pd_bucket
        else:
            self.mbmod.reset_pd_mb_bucket(init_model_fls=init_model_fls)
    @lazy_property
    def interp_yr(self, **kwargs):
        mb_on_h = self.hbins*0.
        for yr in self.years:
            if self.mbmod.__class__ == TIModel_Sfc_Type:
                # just compute once the bucket distribution, then assume t
                if self.mbmod.spinup_yrs == 0:
                #    # because we can only indirectly "transmit" it from lazy_property
                    kwargs['spinup'] = False
                else:
                    kwargs['spinup'] = True
                mb_on_h += self.mbmod.get_annual_mb(self.hbins[::-1],
                                                    year=yr, **kwargs)
            else:
                mb_on_h += self.mbmod.get_annual_mb(self.hbins, year=yr)
        if self.mbmod.__class__ == TIModel_Sfc_Type:
            return interp1d(self.hbins, mb_on_h[::-1]/ len(self.years))
        else:
            return interp1d(self.hbins, mb_on_h/len(self.years))


    @lazy_property
    def interp_m(self):
        # monthly MB
        if self.mbmod.__class__ == TIModel_Sfc_Type:
            raise NotImplementedError('need to implement it for TIModel_Sfc_Type')
        else:
            months = np.arange(12)+1
            interp_m = []
            for m in months:
                mb_on_h = self.hbins*0.
                for yr in self.years:
                    yr = date_to_floatyear(yr, m)
                    mb_on_h += self.mbmod.get_monthly_mb(self.hbins, year=yr)
                interp_m.append(interp1d(self.hbins, mb_on_h / len(self.years)))
            return interp_m

    def get_monthly_climate(self, heights, year=None):
        """Average climate information at given heights.

        Note that prcp is corrected with the precipitation factor and that
        all other biases (precipitation, temp) are applied

        Returns
        -------
        (temp, tempformelt, prcp, prcpsol)
        """
        _, m = floatyear_to_date(year)
        yrs = [date_to_floatyear(y, m) for y in self.years]
        heights = np.atleast_1d(heights)
        nh = len(heights)
        shape = (len(yrs), nh)
        temp = np.zeros(shape)
        tempformelt = np.zeros(shape)
        prcp = np.zeros(shape)
        prcpsol = np.zeros(shape)
        for i, yr in enumerate(yrs):
            t, tm, p, ps = self.mbmod.get_monthly_climate(heights, year=yr)
            temp[i, :] = t
            tempformelt[i, :] = tm
            prcp[i, :] = p
            prcpsol[i, :] = ps
        return (np.mean(temp, axis=0),
                np.mean(tempformelt, axis=0),
                np.mean(prcp, axis=0),
                np.mean(prcpsol, axis=0))

    def get_annual_climate(self, heights, year=None):
        """Average climate information at given heights.

        Note that prcp is corrected with the precipitation factor and that
        all other biases (precipitation, temp) are applied

        Attention: temperature for melt of
        is either in sum over monthly mean or in annual sum (if real_daily)!

        Returns
        -------
        (temp, tempformelt, prcp, prcpsol)
        """
        yrs = monthly_timeseries(self.years[0], self.years[-1],
                                 include_last_year=True)
        heights = np.atleast_1d(heights)
        nh = len(heights)
        shape = (len(yrs), nh)
        temp = np.zeros(shape)
        tempformelt = np.zeros(shape)
        prcp = np.zeros(shape)
        prcpsol = np.zeros(shape)
        for i, yr in enumerate(yrs):
            t, tm, p, ps = self.mbmod.get_monthly_climate(heights, year=yr)
            temp[i, :] = t
            tempformelt[i, :] = tm
            prcp[i, :] = p
            prcpsol[i, :] = ps
        # Note that we do not weight for number of days per month:
        # this is consistent with OGGM's calendar
        return (np.mean(temp, axis=0),
                np.mean(tempformelt, axis=0) * 12,
                np.mean(prcp, axis=0) * 12,
                np.mean(prcpsol, axis=0) * 12)

    def get_monthly_mb(self, heights, year=None,
                       add_climate=False, **kwargs):
        if self.mbmod.__class__ == TIModel_Sfc_Type and not self.interpolation_optim:
            raise NotImplementedError('need to implement it for TIModel_Sfc_Type')
            yr, m = floatyear_to_date(year)

            months = np.arange(12) + 1
            interp_m = []
            # askFabi shouldnt we first loop over years and then over months
            for m in months:
                mb_on_h = heights * 0.
                for yr in self.years:
                    yr = date_to_floatyear(yr, m)
                    mb_on_h += self.mbmod.get_monthly_mb(self.hbins,
                                                         year=yr,
                                                         **kwargs)
                interp_m.append(interp1d(self.hbins, mb_on_h / len(self.years)))

            if add_climate:
                t, tmelt, prcp, prcpsol = self.get_monthly_climate(heights,
                                                                   year=year)
                return mb, t, tmelt, prcp, prcpsol
            return mb
        else:
            yr, m = floatyear_to_date(year)
            if add_climate:
                t, tmelt, prcp, prcpsol = self.get_monthly_climate(heights,
                                                                   year=year)
                return self.interp_m[m-1](heights), t, tmelt, prcp, prcpsol
            return self.interp_m[m-1](heights)

    def get_annual_mb(self, heights, year=None,
                      add_climate=False, **kwargs):
        if self.mbmod.__class__ == TIModel_Sfc_Type and not self.interpolation_optim:
            # slow version ... (without interp_yr) ... if interp_yr version works,
            # can say that this should run only for comparison ???
            mb_on_h = heights * 0. # self.hbins
            for yr in self.years:
                # in the first yr it computes the spinup, but afterwards not anymore
                try:
                    if kwargs['bucket_output']:
                        mb_on_h_yr, pd_bucket = self.mbmod.get_annual_mb(heights,
                                                            year=yr,
                                                            **kwargs)
                        mb_on_h += mb_on_h_yr
                except:
                    mb_on_h += self.mbmod.get_annual_mb(heights,
                                                        year=yr,
                                                        **kwargs)
            mb = mb_on_h / len(self.years)
            #mb = self.interp_yr(heights)
            # #raise NotImplementedError('need to implement it for TIModel_Sfc_Type')
        else:
            use_new = True
            if use_new:
                if self.mbmod.__class__ == TIModel_Sfc_Type and 'spinup' in kwargs:
                    #try:
                    #    kwargs_2 = kwargs['spinup']
                    # remove the fls kwargs if inside
                    #if 'fls' in kwargs:
                    #    kwargs.pop('fls')
                    if not kwargs['spinup']:
                        # if spinup_yrs is = 0, it is the same as saying spinup=False
                        # but it can be "transmitted" to the lazy-property self.interp_yr
                        # (just transmitting the kwargs gave an error such as:
                        # TypeError: __call__() got an unexpected keyword argument 'spinup'
                        # mb = self.interp_yr(heights, **kwargs)

                        self.mbmod.spinup_yrs = 0
                #else:
                #    #todo test if this works at tleast for TIModel
                mb = self.interp_yr(heights)
            else:
                if self.mbmod.__class__ == TIModel_Sfc_Type:
                    # try:
                    #    kwargs_2 = kwargs['spinup']
                    # remove the fls kwargs
                    kwargs.pop('fls')
                    mb = self.interp_yr(heights, **kwargs)
                else:
                    # todo test if this works at tleast for TIModel
                    mb = self.interp_yr(heights)

        pd_out = pd.DataFrame({'yr': year, 'heights': heights, 'mb': mb})
        self._mb_debug_container = self._mb_debug_container.append(pd_out)

        if add_climate:
            t, tmelt, prcp, prcpsol = self.get_annual_climate(heights)
            try:
                if kwargs['bucket_output']:
                    return mb, t, tmelt, prcp, prcpsol, pd_bucket
            except:
                return mb, t, tmelt, prcp, prcpsol
        else:
            try:
                if kwargs['bucket_output']:
                    return mb, pd_bucket
            except:
                return mb



class AvgClimateMassBalance_TIModel(ConstantMassBalance_TIModel):
    """Mass balance with the average climate of a selected period compatible with TIModel
    only works for TIModel (without sfc type distinction)

    !!!Careful! This is conceptually wrong!!! This is here only to make
    a point.

    See https://oggm.org/2021/08/05/mean-forcing/
    """

    def __init__(self, gdir, melt_f=None, prcp_fac = None,
                 mb_model_sub_class = TIModel,
                 residual = 0,
                 baseline_climate=None,
                 input_filesuffix='default',
                 filename='climate_historical',
                 y0=None, halfsize=15,
                 interpolation_optim=False,
                 **kwargs):

        """Initialize.

        Parameters
        TODO
        """



        super(AvgClimateMassBalance_TIModel, self).__init__(gdir, melt_f=melt_f,
                                                    residual=residual,
                                                    filename=filename,
                                                    input_filesuffix=input_filesuffix,
                                                    y0=y0, halfsize=halfsize,
                                                    baseline_climate=baseline_climate,
                                                    prcp_fac=prcp_fac,
                                                    interpolation_optim=interpolation_optim,
                                                    **kwargs)


        self.mbmod = mb_model_sub_class(gdir, melt_f=melt_f, residual=residual,
                                        prcp_fac=prcp_fac,baseline_climate=baseline_climate,
                                     filename=filename,
                                     input_filesuffix=input_filesuffix,
                                     ys=y0-halfsize, ye=y0+halfsize,
                                     **kwargs)

        if self.mbmod.mb_type == 'mb_real_daily':
            raise NotImplementedError('real_daily needs to be implemented!!!')
        #years_pok = np.arange(y0-halfsize, y0+halfsize+1)
        #if self.repeat:
        #    y = self.ys + (y - self.ys) % (self.ye - self.ys + 1)
        #if y < self.ys or y > self.ye:
        #    raise ValueError('year {} out of the valid time bounds: '
        #                     '[{}, {}]'.format(y, self.ys, self.ye))

        # if self.mb_type == 'mb_real_daily' or climate_type == 'annual':
        #     if climate_type == 'annual':
        #         #if type(year) == float:
        #         #    raise InvalidParamsError('')
        #         pok = np.where(self.mbmod.years == year)[0]
        #         if len(pok) < 1:
        #             raise ValueError('Year {} not in record'.format(int(year)))
        #     else:
        #         pok = np.where((self.years == y) & (self.months == m))[0]
        #         if len(pok) < 28:
        #             warnings.warn('something goes wrong with amount of entries\
        #                           per month for mb_real_daily')
        # else:
        #     pok = np.where((self.years == y) & (self.months == m))[0][0]

        if self.mbmod.mb_type == 'mb_monthly' or 'mb_pseudo_daily':
            pok = (self.mbmod.years >= y0 - halfsize)
            pok = pok & (self.mbmod.years <= y0 + halfsize)
            #if y0 - halfsize is not None:
            #    pok = self.mbmod.years >= y0 - halfsize
            #if y0 + halfsize is not None:
            #    try:
            #        pok = pok & (self.mbmod.years <= y0 + halfsize)
            #    except TypeError:
            #        pok = self.mbmod.years <= y0 + halfsize
        tmp = self.mbmod.temp[pok]

        assert (len(tmp) // 12) == (halfsize * 2 + 1)

        self.mbmod.temp = tmp.reshape((len(tmp) // 12, 12)).mean(axis=0)
        # problematic to do that ....!!!! mean of std
        if self.mbmod.mb_type =='mb_pseudo_daily':
            tmp = self.mbmod.temp_std[pok]
            self.mbmod.temp_std = tmp.reshape((len(tmp) // 12, 12)).mean(axis=0)

        tmp = self.mbmod.prcp[pok]
        self.mbmod.prcp = tmp.reshape((len(tmp) // 12, 12)).mean(axis=0)
        tmp = self.mbmod.grad[pok]
        self.mbmod.grad = tmp.reshape((len(tmp) // 12, 12)).mean(axis=0)

        self.mbmod.ys = y0
        self.mbmod.ye = y0
        self.mbmod.months = np.arange(1, 13, dtype=int)
        self.mbmod.years = np.asarray([y0]*12)
        self.years = np.asarray([y0]*12)


@entity_task(log)
def fixed_geometry_mass_balance_TIModel(gdir, ys=None, ye=None,
                                        years=None,
                                monthly_step=False,
                                use_inversion_flowlines=True,
                                climate_filename='climate_historical',
                                climate_input_filesuffix='',
                                ds_gcm = None,
                                        from_json = False,
                                        json_filename='',
                                        sfc_type=False,
                                **kwargs):
    """Computes the mass-balance with climate input
    from e.g. CRU or a GCM.

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
    ds_gcm : xarray dataset
        netCDF dataset output of the gdir 
    **kwargs:
        added to MultipleFlowlineMassBalance_TIModel
    """
    temp_bias = 0
    if sfc_type == False or sfc_type == 'False':
        mb_model_sub_class = TIModel
        kwargs_for_TIModel_Sfc_Type = {}
    else:
        mb_model_sub_class = TIModel_Sfc_Type
        kwargs_for_TIModel_Sfc_Type = {}
        # try:
        # melt_f_update =
        # except:
        #    melt_f_update = 'monthly'
        kwargs_for_TIModel_Sfc_Type['melt_f_update'] = kwargs['melt_f_update']
        kwargs_for_TIModel_Sfc_Type['melt_f_change'] = sfc_type
        kwargs_for_TIModel_Sfc_Type['tau_e_fold_yr'] = kwargs['tau_e_fold_yr']
    if monthly_step:
        raise NotImplementedError('monthly_step not implemented yet')
    if ds_gcm != None or from_json:
        if ds_gcm!=None:
            melt_f = ds_gcm.sel(rgi_id=gdir.rgi_id).melt_f.values
            pf = ds_gcm.sel(rgi_id=gdir.rgi_id).pf.values
        elif from_json:
            if sfc_type is not False:
                if kwargs_for_TIModel_Sfc_Type['melt_f_update'] == 'annual':
                    fs_new = '_{}_sfc_type_{}_annual_{}_{}'.format('W5E5', sfc_type, kwargs['mb_type'],
                                                            kwargs['grad_type'])
                else:
                    fs_new = '_{}_sfc_type_{}_{}_{}'.format('W5E5', sfc_type, kwargs['mb_type'],
                                                            kwargs['grad_type'])
            else:
                fs_new = '_{}_sfc_type_{}_{}_{}'.format('W5E5', sfc_type, kwargs['mb_type'],
                                                    kwargs['grad_type'])
            # json_filename = 'melt_f_geod_opt_winter_mb_approx_std'
            # get the calibrated melt_f that suits to the prcp factor
            try:
                d = gdir.read_json(filename=json_filename,
                                   filesuffix=fs_new)
                # get the corrected ref_hgt so that we can apply this again on the mb model
                # if otherwise not melt_f could be found!
                pf = d['pf']
                melt_f = d['melt_f']
                temp_bias = d['temp_bias']
            except:
                raise InvalidWorkflowError(
                    'there is no calibrated melt_f for this precipitation factor, glacier, climate'
                    'mb_type and grad_type, need to do the calibration first!')

        mb = MultipleFlowlineMassBalance_TIModel(gdir, mb_model_class=mb_model_sub_class,
                                                 melt_f=melt_f, prcp_fac=pf,
                                                 filename=climate_filename,
                                                 use_inversion_flowlines=use_inversion_flowlines,
                                                 bias=0,
                                                 input_filesuffix=climate_input_filesuffix,
                                                 mb_type=kwargs['mb_type'],
                                                 grad_type=kwargs['grad_type'],
                                                 # check_calib_params=check_calib_params,
                                                 **kwargs_for_TIModel_Sfc_Type)
        mb.temp_bias = temp_bias
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


@global_task(log)
def compile_fixed_geometry_mass_balance_TIModel(gdirs, filesuffix='',
                                        path=True, csv=False,
                                        climate_filename='climate_historical',
                                        use_inversion_flowlines=True,
                                        ys=None, ye=None, years=None,
                                        climate_input_filesuffix='',
                                        ds_gcm=None,
                                                from_json=False,
                                                json_filename='',
                                                sfc_type=False,
                                        **kwargs):
    """
    same as `compile_fixed_geometry_mass_balance` but compatible to TIModel

    Compiles a table of specific mass-balance timeseries for all glaciers.

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
    kwargs :
        passed to fixed_geometry_mass_balance_TIModel
    todo: other docs!
    """
    from oggm.workflow import execute_entity_task
    #from oggm.core.massbalance import fixed_geometry_mass_balance

    out_df = execute_entity_task(fixed_geometry_mass_balance_TIModel, gdirs,
                                 climate_filename=climate_filename,
                                 use_inversion_flowlines=use_inversion_flowlines,
                                 ys=ys, ye=ye, years=years,
                                 ds_gcm=ds_gcm, from_json=from_json,
                                 climate_input_filesuffix=climate_input_filesuffix,
                                 json_filename=json_filename,
                                 sfc_type=sfc_type,
                                 **kwargs)

    for idx, s in enumerate(out_df):
        if s is None:
            out_df[idx] = pd.Series(np.nan)

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



class RandomMassBalance_TIModel(MassBalanceModel):
    """Random shuffle of all MB years within a given time period.

    (copy of RandomMassBalance adapted for TIModel
    TODO: not yet tested at all!!!

    This is useful for finding a possible past glacier state or for sensitivity
    experiments.

    Note that this is going to be sensitive to extreme years in certain
    periods, but it is by far more physically reasonable than other
    approaches based on gaussian assumptions.
    """

    def __init__(self, gdir, melt_f=None, residual=0,
                 y0=None, halfsize=15, seed=None,
                 mb_model_sub_class = TIModel, baseline_climate=None,
                 filename='climate_historical', input_filesuffix='default',
                 all_years=False, unique_samples=False, **kwargs):
        """Initialize.

        Parameters
        ----------
        gdir : GlacierDirectory
            the glacier directory
        mu_star : float, optional
            set to the alternative value of mu* you want to use
            (the default is to use the calibrated value)
        bias : float, optional
            set to the alternative value of the calibration bias [mm we yr-1]
            you want to use (the default is to use the calibrated value)
            Note that this bias is *substracted* from the computed MB. Indeed:
            BIAS = MODEL_MB - REFERENCE_MB.
        y0 : int, optional, default: tstar
            the year at the center of the period of interest. The default
            is to use tstar as center.
        halfsize : int, optional
            the half-size of the time window (window size = 2 * halfsize + 1)
        seed : int, optional
            Random seed used to initialize the pseudo-random number generator.
        filename : str, optional
            set to a different BASENAME if you want to use alternative climate
            data.
        input_filesuffix : str
            the file suffix of the input climate file
        all_years : bool
            if True, overwrites ``y0`` and ``halfsize`` to use all available
            years.
        unique_samples: bool
            if true, chosen random mass-balance years will only be available
            once per random climate period-length
            if false, every model year will be chosen from the random climate
            period with the same probability
        **kwargs:
            kyeword arguments to pass to the PastMassBalance model
        """

        super(RandomMassBalance_TIModel, self).__init__()
        self.valid_bounds = [-1e4, 2e4]  # in m
        #if mb_model_sub_class == TIModel_Sfc_Type:
        #    **kwargs['check_availability'] = False
        if mb_model_sub_class == TIModel_Sfc_Type:
            # was not sure how to add something to **kwargs
            kwargs2 = {'check_availability':False}
        else:
            kwargs2 = {}

        self.mbmod = mb_model_sub_class(gdir, melt_f=melt_f, residual=residual,
                                     filename=filename,
                                     input_filesuffix=input_filesuffix,
                                     baseline_climate=baseline_climate,
                                     **kwargs, **kwargs2)

        # Climate period
        if all_years:
            self.years = self.mbmod.years
        else:
            self.years = np.arange(y0-halfsize, y0+halfsize+1)
        self.yr_range = (self.years[0], self.years[-1]+1)
        self.ny = len(self.years)
        self.hemisphere = gdir.hemisphere

        # RandomState
        self.rng = np.random.RandomState(seed)
        self._state_yr = dict()

        # Sampling without replacement
        self.unique_samples = unique_samples
        if self.unique_samples:
            self.sampling_years = self.years

        self._mb_debug_container = pd.DataFrame({'yr':[], 'ryr':[],
                                                 'heights':[], 'mb':[]})


    def historical_climate_qc_mod(self, gdir):
        return self.mbmod.historical_climate_qc_mod(gdir)

    @property
    def temp_bias(self):
        """Temperature bias to add to the original series."""
        return self.mbmod.temp_bias

    @temp_bias.setter
    def temp_bias(self, value):
        """Temperature bias to add to the original series."""
        for attr_name in ['_lazy_interp_yr', '_lazy_interp_m']:
            if hasattr(self, attr_name):
                delattr(self, attr_name)
        self.mbmod.temp_bias = value

    @property
    def prcp_fac(self):
        """Precipitation factor to apply to the original series."""
        return self.mbmod.prcp_fac

    @prcp_fac.setter
    def prcp_fac(self, value):
        """Precipitation factor to apply to the original series."""
        for attr_name in ['_lazy_interp_yr', '_lazy_interp_m']:
            if hasattr(self, attr_name):
                delattr(self, attr_name)
        self.mbmod.prcp_fac = value

    @property
    def residual(self):
        """Residual bias to apply to the original series."""
        return self.mbmod.residual

    @residual.setter
    def residual(self, value):
        """Residual bias to apply to the original series."""
        self.mbmod.residual = value

    def get_state_yr(self, year=None):
        """For a given year, get the random year associated to it."""
        year = int(year)
        if year not in self._state_yr:
            if self.unique_samples:
                # --- Sampling without replacement ---
                if self.sampling_years.size == 0:
                    # refill sample pool when all years were picked once
                    self.sampling_years = self.years
                # choose one year which was not used in the current period
                _sample = self.rng.choice(self.sampling_years)
                # write chosen year to dictionary
                self._state_yr[year] = _sample
                # update sample pool: remove the chosen year from it
                self.sampling_years = np.delete(
                    self.sampling_years,
                    np.where(self.sampling_years == _sample))
            else:
                # --- Sampling with replacement ---
                self._state_yr[year] = self.rng.randint(*self.yr_range)
        return self._state_yr[year]

    def get_monthly_mb(self, heights, year=None, **kwargs):
        ryr, m = floatyear_to_date(year)
        ryr = date_to_floatyear(self.get_state_yr(ryr), m)
        return self.mbmod.get_monthly_mb(heights, year=ryr, **kwargs)

    def get_daily_mb(self, heights, year=None, **kwargs):
        ryr, m = floatyear_to_date(year)
        ryr = date_to_floatyear(self.get_state_yr(ryr), m)
        return self.mbmod.get_daily_mb(heights, year=ryr, **kwargs)

    def get_annual_mb(self, heights, year=None, **kwargs):
        ryr = self.get_state_yr(int(year))
        #return self.mbmod.get_annual_mb(heights, year=ryr, **kwargs)
        # debug stuff to find out more about this problem
        if isinstance(self.mbmod, TIModel_Sfc_Type):
            if self.mbmod.check_availability:
                raise InvalidWorkflowError('check availability should be false for get_annual_mb in '
                                       'RandomMassBalance_TIModel!!! sth. goes wrong!!!')
        _mb = self.mbmod.get_annual_mb(heights, year=ryr, **kwargs)
        # if add_climate is True we only want to save the mb inside of the debug container!
        if 'add_climate' in kwargs.keys():
            if kwargs['add_climate']:
                _mb_spec = _mb[0]
        else:
            _mb_spec = _mb
        pd_out = pd.DataFrame({'yr': year, 'ryr': ryr, 'heights': heights, 'mb': _mb_spec})
        self._mb_debug_container = self._mb_debug_container.append(pd_out)
        return _mb
