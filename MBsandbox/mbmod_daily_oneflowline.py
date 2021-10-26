#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 24 12:28:37 2020

@author: lilianschuster

different temperature index mass balance types added that are working with the Huss flowlines

this is the faster version
"""
# jax_true = True
# if jax_true:
#     import jax.numpy as np
#     import numpy as onp
# else: problem: nan values, where stuff ...
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

# import oggm

# imports from oggm
from oggm import entity_task
from oggm import cfg, utils
from oggm.cfg import SEC_IN_YEAR, SEC_IN_MONTH, SEC_IN_DAY
from oggm.utils import (floatyear_to_date, date_to_floatyear, ncDataset,
                        lazy_property, monthly_timeseries,
                        clip_min, clip_array)
from oggm.utils._funcs import haversine
from oggm.exceptions import InvalidParamsError, InvalidWorkflowError
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

BASENAMES['WFDE5_CRU_daily'] = {
    'inv': 'wfde5_cru/daily/v1.1/wfde5_cru_glacier_invariant_flat.nc',
    'tmp': 'wfde5_cru/daily/v1.1/wfde5_cru_tmp_1979-2018_flat.nc',
    'prcp': 'wfde5_cru/daily/v1.1/wfde5_cru_prcp_1979-2018_flat.nc',
    }

BASENAMES['W5E5_daily'] = {
    'inv': 'w5e5v2.0/flattened/daily/w5e5v2.0_glacier_invariant_flat.nc',
    'tmp': 'w5e5v2.0/flattened/daily/w5e5v2.0_tas_global_daily_flat_glaciers_1979_2019.nc',
    'prcp': 'w5e5v2.0/flattened/daily/w5e5v2.0_pr_global_daily_flat_glaciers_1979_2019.nc',
    }
BASENAMES['MSWEP_daily'] = {
    'prcp': 'mswepv2.8/flattened/daily/mswep_pr_global_daily_flat_glaciers_1979_2019.nc'
    # there is no orography file for MSWEP!!! (and also no temperature file)
    }


def get_w5e5_file(dataset='W5E5_daily', var=None,
                  server='https://cluster.klima.uni-bremen.de/~lschuster/'):
    """returns a path to desired WFDE5_CRU or W5E5 or MSWEP baseline climate file.

    If the file is not present, downloads it

    ... copy of get_ecmwf_file but with different ECMWF_SERVER ...
    """


    # Be sure input makes sense
    if dataset not in BASENAMES.keys():
        raise InvalidParamsError('ECMWF dataset {} not '
                                 'in {}'.format(dataset, BASENAMES.keys()))
    if var not in BASENAMES[dataset].keys():
        raise InvalidParamsError('ECMWF variable {} not '
                                 'in {}'.format(var,
                                                BASENAMES[dataset].keys()))

    # File to look for
    return utils.file_downloader(server + BASENAMES[dataset][var])
# this could be used in general
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
    climate_type: str
        either WFDE5_CRU (default, v1.1 only till end of 2018) or W5E5
        or W5E5_MSWEP (precipitation from MSWEP, temp. from W5E5)
    output_filesuffix : optional
         None by default, as the output_filesuffix is automatically chosen
         from the temporal_resol and climate_type. But you can change the filesuffix here,
         just make sure that you use then later the right climate file
    cluster : bool
        default is False, if this is run on the cluster, set it to True,
        because we do not need to download the files

    """
    if cfg.PARAMS['hydro_month_nh'] != 1 and climate_type != 'WFDE5_CRU':
        raise InvalidParamsError('Hydro months different than 1 are not tested, there is some'
                                 'issue with the lapse rates, as they only go until 2019-05'
                                 'if you want other hydro months, need to adapt the code!!!')

    if climate_type == 'WFDE5_CRU':
        if temporal_resol=='monthly':
            output_filesuffix_def = '_monthly_WFDE5_CRU'
        elif temporal_resol == 'daily':
            output_filesuffix_def = '_daily_WFDE5_CRU'
        # basename of climate
        # (we use for both the daily dataset and resample to monthly)
        dataset = 'WFDE5_CRU_daily'
        dataset_prcp = dataset

    elif climate_type =='W5E5':
        if temporal_resol == 'monthly':
            output_filesuffix_def = '_monthly_W5E5'
        elif temporal_resol == 'daily':
            output_filesuffix_def = '_daily_W5E5'
        # basename of climate
        # (for both the daily dataset and resample to monthly)
        dataset = 'W5E5_daily'
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

    if cluster:
        cluster_path = '/home/users/lschuster/'
        path_tmp = cluster_path + BASENAMES[dataset]['tmp']
        path_prcp = cluster_path + BASENAMES[dataset_prcp]['prcp']
        path_inv = cluster_path + BASENAMES[dataset]['inv']

    else:
        path_tmp = get_w5e5_file(dataset, 'tmp')
        path_prcp = get_w5e5_file(dataset_prcp, 'prcp')
        path_inv = get_w5e5_file(dataset, 'inv')

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
        if climate_type == 'WFDE5_CRU':
            # old version of WFDE5_CRU that only goes till 2018
            if y1 > 2018 or y0 < 1979:
                text = 'The climate files only go from 1979--2018,\
                    choose another y0 and y1'
                raise InvalidParamsError(text)
        elif climate_type == 'W5E5' or climate_type == 'W5E5_MSWEP':
            if y1 > 2019 or y0 < 1979:
                text = 'The climate files only go from 1979 --2019, something is wrong'
        # if default settings: this is the last day in March or September
        time_f = '{}-{:02d}'.format(y1, em)
        end_day = int(ds.sel(time=time_f).time.dt.daysinmonth[-1].values)

        #  this was tested also for hydro_month = 1
        ds = ds.sel(time=slice('{}-{:02d}-01'.format(y0, sm),
                               '{}-{:02d}-{}'.format(y1, em, end_day)))
        if sm == 1 and y1 == 2019 and (climate_type == 'W5E5' or climate_type == 'W5E5_MSWEP'):
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
        if climate_type == 'W5E5' or climate_type == 'W5E5_MSWEP':
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

    # precipitation: similar ar temperature
    with xr.open_dataset(path_prcp) as ds:
        assert ds.longitude.min() >= 0

        yrs = ds['time.year'].data
        y0 = yrs[0] if y0 is None else y0
        y1 = yrs[-1] if y1 is None else y1
        # Attention here we take the same y0 and y1 as given from the
        # daily tmp dataset (goes till end of 2018)

        # attention if daily data, need endday!!!
        if climate_type == 'W5E5_MSWEP' and y0 == 1979 and sm == 1:
            # first day of 1979 is missing!!! (will assume later that is equal to the
            # second day prcp ...
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

        if y0 == 1979 and sm ==1 and climate_type == 'W5E5_MSWEP':
            median_jan_1979 = np.median(ds.sel(time='1979-01').pr.values) * SEC_IN_DAY
        # if we want to use monthly summed up precipitation:
        if temporal_resol == 'monthly':
            ds = ds.resample(time='MS').sum()
            ds['longitude'] = ds.longitude.isel(time=0)
            ds['latitude'] = ds.latitude.isel(time=0)
        elif temporal_resol == 'daily':
            pass
        if climate_type == 'WFDE5_CRU':
        # the prcp data of wfde5_CRU  has been converted already into
        # kg m-2 day-1 ~ mm/day or into kg m-2 month-1 ~ mm/month
            prcp = ds[Pvar].data  # * 1000
        elif climate_type == 'W5E5' or climate_type == 'W5E5_MSWEP':
            # if daily convert kg m-2 s-1 into kg m-2 day-1
            # if monthly convert monthly sum of kg m-2 s-1 into kg m-2 month-1
            prcp = ds[Pvar].data * SEC_IN_DAY
            if climate_type == 'W5E5_MSWEP':
                # 1st day of January 1979 is missing for MSWEP prcp
                # assume that precipitation is equal to
                # precipitation of the next day (1979-01-02)
                if y0 == 1979 and sm == 1:
                    if temporal_resol == 'daily':
                        prcp = np.append(np.array(median_jan_1979), prcp)
                    elif temporal_resol == 'monthly':
                        # in the monthly sum, 1st of January 1979 is missing
                        # add this value by assuming mean over the other days
                        prcp[0] = prcp[0] + median_jan_1979
                    if y1 == 2019:
                        if temporal_resol == 'daily':
                            assert len(prcp) == 14975
                        elif temporal_resol == 'monthly':
                            assert len(prcp) == 492
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
    # TODO: use updated ERA5dr files that go until end of 2019
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
        if sm==1 and y1 == 2019 and (climate_type == 'W5E5' or climate_type =='W5E5_MSWEP'):
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
            # when wfde5_daily is applied
            # assume same gradient for each day
            if sm == 1 and y1 == 2019 and (climate_type == 'W5E5' or climate_type == 'W5E5_MSWEP'):
                gradient = np.repeat(gradient, days_in_month.resample(time='MS').mean())
                assert len(gradient) == len(days_in_month)
            else:
                gradient = np.repeat(gradient, ds['time.daysinmonth'])

        #if climate_type == 'W5E5_MSWEP':
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
    # This is now a new function, maybe it would be better to make a general
    # process_daily_data function where ERA5_daily and WFDE5_daily
    # but is used, so far, only for ERA5_daily as source dataset ..


@entity_task(log, writes=['climate_historical_daily'])
def process_era5_daily_data(gdir, y0=None, y1=None, output_filesuffix='_daily_ERA5',
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
class TIModel_Parent(MassBalanceModel):
    """ Parent class that works for different temperature-index models, this is only instanciated
    via the child classes TIModel or TIModel_Sfc_Type, just container with shared code
    to get annual, monthly and daily climate ... the actual mass balance can only be computed in child classes
    TODO: change documentation
    Different mass balance modules compatible to OGGM with one flowline

    so far this is only tested for the Huss flowlines
    """

    def __init__(self, gdir, melt_f, prcp_fac=2.5, residual=0,
                 mb_type='mb_pseudo_daily', N=100, loop=False,
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
            Indeed: residual = MODEL_MB - REFERENCE_MB.
            ToDO: maybe change the sign?,
            opposite to OGGM "MB terms + residual"
        mb_type: str
            three types: 'mb_pseudo_daily' (default: use temp_std and N percentiles),
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
            only applied if mb_type is 'mb_pseudo_daily'
        grad_type : str
            three types of applying the temperature gradient:
            'cte' (default, constant lapse rate, set to default_grad,
                   same as in default OGGM)
            'var_an_cycle' (varies spatially and over annual cycle,
                            but constant over the years)
            'var' (varies spatially & temporally as in the climate files, deprecated!)
        filename : str, optional
            set to a different BASENAME if you want to use alternative climate
            data, default is climate_historical
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
        baseline_climate: str
            climate that should be applied, e.g. ERA5dr, WFDE5_CRU, W5E5
        input_filesuffix: str
            if set to 'default', it is set depending on mb_type and
            baseline_climate, but can change this here,
            e.g. change it to '' to work without filesuffix as
            default in oggm PastMassBalance

        Attributes
        ----------
        temp_bias : float, default 0
            Add a temperature bias to the time series
        prcp_fac : float, >0
            multiplicative precipitation correction factor (default 2.5)
        melt_f : float, >0
            melt temperature sensitivity factor per month (kg /m² /mth /K),
            need to be prescribed, e.g. such that
            |mean(MODEL_MB)-mean(REF_MB)|--> 0
        """
        # melt_f is only initiated here, and not used in __init__.py
        # !!!
        # so it does not matter if it is changed
        self._melt_f = melt_f
        if self._melt_f != None and self._melt_f <= 0:
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
        # (to convert from kg/m2 into m ice per second=
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
        if baseline_climate == None:
            try:
                baseline_climate = gdir.get_climate_info()['baseline_climate_source']
            except:
                baseline_climate = cfg.PARAMS['baseline_climate']
            if baseline_climate != cfg.PARAMS['baseline_climate']:
                raise InvalidParamsError('need to use filesuffixes to define the right climate!')

        if input_filesuffix == 'default':
            if mb_type != 'mb_real_daily':
                # cfg.PARAMS['baseline_climate'] = 'ERA5dr'
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

        # used xarray instead of netCDF4, is slower and needs to be changed at some point
        # to netCDF4
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
                    # need this to ensure that gradients are not fill-values
                    xr_nc['gradient'] = xr_nc['gradient'].where(xr_nc['gradient']<1e12)
                    ###
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

    @property
    def prcp_fac(self):
        ''' prints the _prcp_fac
        '''
        return self._prcp_fac

    @prcp_fac.setter
    def prcp_fac(self, new_prcp_fac):
        ''' sets new prcp_fac, changes prcp time series
         and if TIModel_Sfc_Type resets the pd_mb and buckets
        '''
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
        ''' prints the _melt_f '''
        return self._melt_f

    @melt_f.setter
    def melt_f(self, new_melt_f):
        ''' sets new melt_f and if TIModel_Sfc_Type resets the pd_mb and buckets
        '''
        if type(self) == TIModel_Sfc_Type:
            # if the prcp_fac is set to another value,
            # need to reset pd_mb_annual, pd_mb_monthly
            # and also reset pd_bucket
            self.reset_pd_mb_bucket()
        self._melt_f = new_melt_f


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
                                  climate_qc_months=3,
                                  ):
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
        if type(self) == TIModel_Sfc_Type:
            # if the prcp_fac is set to another value,
            # need to reset pd_mb_annual, pd_mb_monthly
            # and also reset pd_bucket
            self.reset_pd_mb_bucket()
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

    def get_2d_avg_annual_air_hydro_temp(self, heights, year=None,
                                         avg_climate = False, hs = 15,
                                         hydro_m = 10):
        raise NotImplementedError('not tested and only useful'
                                  ' for estimating refreezing')
        # WIP
        # (I won't work on this for the next half a year, but leave
        # it inside as atemplate)
        # only computes avg annual air temperature in hydrological years,
        # this is necessary for the refreezing estimate of Woodward et al.1997
        # only tested for NH here
        # IMPORTANT: this does not take into account that months
        # have different amount of days (should I include the weighting as
        # it is inside of PyGEM??? [see: https://github.com/drounce/PyGEM/blob/ac619bd1fd862b93a01068a0887efa2a97478b99/pygem/utils/_funcs.py#L10 )
        #pok = np.where(self.years == year)[0]
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
        # first get the climate data
        #warnings.warn('Attention: this has not been tested enough to be sure that '
        #        'it works')
        if self.mb_type == 'mb_real_daily':
            return self._get_climate(heights, 'monthly', year=year)
        else:
            raise InvalidParamsError('_get_2d_monthly_climate works only\
                                     with mb_real_daily as mb_type!!!')

    def get_monthly_climate(self, heights, year=None):
        # first get the climate data
        #warnings.warn('Attention: this has not been tested enough to be sure that \
        #        it works')
        if self.mb_type == 'mb_real_daily':
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
    # If I also want to use this outside of the class because
    # (e.g. in climate.py), I have to change this again and remove the self...
    # and somehow there is a problem if I put not self in
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
                                     mb_pseudo_daily or mb_real_daily" ')
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


class TIModel(TIModel_Parent):

    """ child class of TIMOdel_Parent that does not use surface type distinction! """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_monthly_mb(self, heights, year=None, add_climate=False,
                       **kwargs):
        """ computes annual mass balance in m of ice per second!

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
            t, temp2dformelt, prcp, prcpsol = out
            #(days per month)
            #dom = 365.25/12  # len(prcpsol.T)
            fact = 12/365.25
            # attention, I should not use the days of years as the melt_f is
            # per month ~mean days of that year 12/daysofyear
            # to have the same unit of melt_f, which is
            # the monthly temperature sensitivity (kg /m² /mth /K),
            mb_daily = prcpsol - self.melt_f * temp2dformelt * fact

            mb_month = np.sum(mb_daily, axis=1)
            # more correct than using a mean value for days in a month
            warnings.warn('there might be a problem with SEC_IN_MONTH'
                          'as February changes amount of days inbetween the years'
                          ' see test_monthly_glacier_massbalance()')

        else:
            # get 1D values for each height, no dependency on days
            t, temp2dformelt, prcp, prcpsol = self.get_monthly_climate(heights, year=year)
            mb_month = prcpsol - self.melt_f * temp2dformelt

        # residual is in mm w.e per year, so SEC_IN_MONTH .. but mb_month
        # shoud be per month!
        mb_month -= self.residual * self.SEC_IN_MONTH / self.SEC_IN_YEAR
        # this is for mb_pseudo_daily otherwise it gives the wrong shape
        mb_month = mb_month.flatten()
        if add_climate:
            if self.mb_type == 'mb_real_daily':
                # for run_with_hydro want to get monthly output (sum of daily),
                # if we want daily output in run_with_hydro, then we need another option here
                # or rather directly use get_daily_mb()
                prcp = prcp.sum(axis=1)
                prcpsol = prcpsol.sum(axis=1)
            return (mb_month / SEC_IN_MONTH / self.rho, t, temp2dformelt,
                    prcp, prcpsol)
        # instead of SEC_IN_MONTH, use instead len(prcpsol.T)==daysinmonth
        return mb_month / self.SEC_IN_MONTH / self.rho

    def get_annual_mb(self, heights, year=None, add_climate=False,
                      **kwargs):
        """ computes annual mass balance in m of ice per second !"""
        # get_monthly_mb and get_annual_mb are only different
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
        mb_annual = np.sum(prcpsol - self.melt_f * temp2dformelt*fact,
                           axis=1)
        mb_annual = (mb_annual - self.residual) / self.SEC_IN_YEAR / self.rho
        if add_climate:
            # for run_with_hydro, want climate as sum over year (for prcp...)
            return (mb_annual, t.mean(axis=1), temp2dformelt.sum(axis=1),
                    prcp.sum(axis=1), prcpsol.sum(axis=1))
        return mb_annual

    def get_daily_mb(self, heights, year=None,
                     add_climate=False):
        """computes daily mass balance in m of ice per second

        year has to be given as float hydro year from what the month is taken,
        hence year 2000 -> y=2000, m = 1, & year = 2000.09, y=2000, m=2 ...
        which corresponds to the real year 1999 an months October or November
        if hydro year starts in October

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
        " returns specific daily mass balance in kg m-2 day "
        if len(np.atleast_1d(year)) > 1:
            out = [self.get_specific_daily_mb(heights=heights, widths=widths,
                                        year=yr) for yr in year]
            return np.asarray(out)

        mb = self.get_daily_mb(heights, year=year)
        spec_mb = np.average(mb * self.rho * SEC_IN_DAY, weights=widths, axis=0)
        assert len(spec_mb) > 360
        return spec_mb

class TIModel_Sfc_Type(TIModel_Parent):

    def __init__(self, gdir, melt_f,
                 melt_f_ratio_snow_to_ice=0.5,
                 melt_f_update='annual',
                 spinup_yrs=6,
                 tau_e_fold_yr = 0.5,
                 melt_f_change = 'linear',
                 check_availability = True,
                 interpolation_optim=False,
                 hbins = np.NaN,
                 **kwargs):

        '''
        TIModel with surface type distinction using a bucket system

        ... work in process ...
        Parameters
        ----------
        melt_f_ratio_snow_to_ice
            ratio of snow melt factor to ice melt factor,
            default is 0.5 same as in GloGEM, PyGEM ...
        melt_f_update : str, default: 'annual'
            'annual' or 'monthly' (daily in future?), how
            often the melt factor should be updated and how
            many buckets exist. If annual, then it uses 1 snow
            and 5 firn buckets with yearly melt factor updates. If monthly,
            each month the snow is ageing over the amount of spinup years
            (default 6yrs, i.e., 72 months).
            Melt factors are interpolated linearly inbetween the buckets.
            TODO: include non-linear melt factor change!
        spinup_yrs : int
            amount of years to compute as spinup (if spinup=True in
            get_annual_mb or get_monthly_mb) before computing the actual year.
            Default is 6 years. After these 6 years every bucket has the opportunity
            to be filled (as we have 1 snow bucket and 5 firn buckets, or
            72 monthly buckets)
        tau_e_fold_yr : float
            default is 0.5, only used if melt_f_change is 'neg_exp', it describes how fast the snow melt
            factor approximates to the ice melt factor via
            melt_f = melt_f_ice + (melt_f_snow - melt_f_ice)* np.exp(-time_yr/tau_e_fold_yr) # s: in months
            do not set to 0, otherwise the melt_f of the first bucket is np.NaN
        melt_f_change : str
            default is linear, how the snow melt factor changes to the ice melt factor, either 'linear'
            or 'neg_exp' (see `tau_e_fold_yr` for the equation)
        check_availability: boolean
            default is True, checks if year / month has already been computed, if true it gives that output.
            This ensures to give always the same values for a specific year/month and is also faster.
            However, in case of random_climate or constant_climate, we have to set this to False!!!
        kwargs
        '''
        super().__init__(gdir, melt_f, **kwargs)

        self.hbins = hbins
        self.interpolation_optim=interpolation_optim

        # ratio of snow melt_f to ice melt_f
        self.tau_e_fold_yr = tau_e_fold_yr  # t_albedo
        assert tau_e_fold_yr > 0, "tau_e_fold_yr has to be above zero!"
        self.melt_f_change = melt_f_change
        self.melt_f_ratio_snow_to_ice = melt_f_ratio_snow_to_ice
        self.melt_f_update = melt_f_update
        self.spinup_yrs = spinup_yrs
        if self.melt_f_update == 'annual':
            self.buckets = ['snow', 'firn_yr_1', 'firn_yr_2', 'firn_yr_3',
                            'firn_yr_4', 'firn_yr_5']
        elif self.melt_f_update == 'monthly':
            # for each month over 6 years a bucket
            self.buckets = np.arange(0, 12 * 6, 1).tolist()
        else:
            raise InvalidParamsError('melt_f_update can either be annual or monthly!')
        # first bucket: if annual it is 'snow', if monthly update it is 0
        self.first_snow_bucket = self.buckets[0]

        self.columns = self.buckets + ['delta_kg/m2']
        # TODO: maybe also include snow_delta_kg/m2, firn_yr_1_delta_kg/m2...
        # !!!: I don't need an ice bucket because this is assumed to be "infinite"

        # only have one flowline when using elevation bands
        self.fl = gdir.read_pickle('inversion_flowlines')[-1]
        # save the inversion height to later check if the same height is applied!!!
        self.inv_heights = self.fl.surface_h

        self.check_availability = check_availability

        # container template
        # use the distance_along_flowline as index
        self._pd_mb_template = pd.DataFrame(0, index=self.fl.dx_meter * np.arange(self.fl.nx),
                             columns=[])
        self._pd_mb_template.index.name = 'distance_along_flowline'

        # bucket containers with buckets as columns
        pd_bucket = self._pd_mb_template.copy()
        pd_bucket[self.columns] = 0
        # I don't need a total_kg/m2 because I don't know it anyway
        # we do this before inversion!
        # to make it "updatable", do we need this ?
        # self.pd_bucket_init = pd_bucket

        # storage containers for monthly and annual mb
        # columns are the months or years respectively
        self.pd_mb_monthly = self._pd_mb_template.copy()
        self.pd_mb_annual = self._pd_mb_template.copy()

        # storage container for buckets (in kg/m2)
        # columns are the buckets
        # (6*12 or 6 buckets depending if melt_f_update is monthly or annual)
        self.pd_bucket = pd_bucket

    @property
    def melt_f_buckets(self):
        # interpolate linearly from snow melt_f to ice melt_f, use melt_f_ratio_snow_to_ice
        # to get snow melt_f ()
        # need to do this here and not in init, because otherwise it does not get updated ..
        # or I would need to set melt_f as property / setter function that updates self.melt_f_buckets....
        # self.melt_f  : melt_f of ice !!!
        if self.melt_f_change == 'linear':
            self._melt_f_buckets = dict(zip(self.buckets + ['ice'],
                                      np.linspace(self.melt_f * self.melt_f_ratio_snow_to_ice,
                                                  self.melt_f,
                                                  len(self.buckets)+1)
                                      ))
        elif self.melt_f_change == 'neg_exp':
            time = np.linspace(0, 6, len(self.buckets + ['ice']))
            melt_f_snow = self.melt_f_ratio_snow_to_ice * self.melt_f
            self._melt_f_buckets = dict(zip(self.buckets + ['ice'],
                                      self.melt_f + (melt_f_snow - self.melt_f) * np.exp(-time / self.tau_e_fold_yr)
                                      ))
            # s: in months

        # at the moment we don't allow to set externally the melt_f_buckets but this could be added later ...
        return self._melt_f_buckets


    def reset_pd_mb_bucket(self,
                           init_model_fls='use_inversion_flowline'):
        """ resets pandas mass balance monthly and annual dataframe as well as the bucket dataframe
        is called when setting new melt_f, prcp_fac, temp_bias, ...
        """
        #TODO: do I need to reset sometimes only the buckets,
        # or only mb_monthly???
        if self.interpolation_optim:
            self._pd_mb_template = pd.DataFrame(0,
                                                      index=self.hbins[::-1],
                                                      columns=[])

            self._pd_mb_template.index.name = 'hbins_height'

            self.pd_mb_monthly = self._pd_mb_template.copy()
            self.pd_mb_annual = self._pd_mb_template.copy()

            pd_bucket = self._pd_mb_template.copy()
            pd_bucket[self.columns] = 0
            self.pd_bucket = pd_bucket
        else:
            if init_model_fls != 'use_inversion_flowline':
                #if init_model_fls == 'use_model_flowline':
                #fls = gdir.read_pickle('model_flowlines')
                self.fl = copy.deepcopy(init_model_fls)[-1]
                self.mod_heights = self.fl.surface_h

                # container template
                # use the distance_along_flowline as index
                self._pd_mb_template = pd.DataFrame(0, index=self.fl.dx_meter * np.arange(self.fl.nx),
                                                    columns=[])
                self._pd_mb_template.index.name = 'distance_along_flowline'


            self.pd_mb_monthly = self._pd_mb_template.copy()
            self.pd_mb_annual = self._pd_mb_template.copy()

            pd_bucket = self._pd_mb_template.copy()
            pd_bucket[self.columns] = 0
            self.pd_bucket = pd_bucket

    #def reset_pd_bucket(self):
    #    # deprecated ...
    #    pd_bucket = self._pd_mb_template.copy()
    #    pd_bucket.columns = self.columns
    #    self.pd_bucket = pd_bucket


    def _add_delta_mb_vary_melt_f(self, heights, year=None,
                                  climate_resol = 'annual'):
        """get the mass balance of all buckets and add/remove the
         new mass change for each bucket

         Parameters
         ----
         heights: np.array()
            heights of elevation flowline, at the monent only inversion height works!!!
         year: int or float
            if melt_f_update=='monthly', year should be a hydro float year,
            so it corresponds to 1 month of a specific year
         climate_resol: 'annual' or 'monthly
            if melt_f_update -> climate_resol has to be always monthly
            but if annual melt_f_update can have either annual climate resol (if get_annual_mb is used)
            or monthly climate resol (if get_monthly_mb is used)

         """
        # problem: @Fabi if I put heights inside that are not fitting to
        # distance_along_flowline, it can get problematic
        # how can I check this ???
        # lenght of the heights should be the same as distance along
        # flowline of pd_bucket dataframe

        #if self.melt_f_update == 'annual':
            #year = int(year) # abrunden here (want integer year)
            # wip: to let get_monthly_mb work with annual melt_f_updata ...
            # if monthly_mb:
            #         if self.mb_type == 'mb_real_daily':
            #             # get 2D values, dependencies on height and time (days)
            #             out = self._get_2d_monthly_climate(heights, year)
            #             t, temp2dformelt, prcp, prcpsol = out
            #             #(days per month)
            #             #dom = 365.25/12  # len(prcpsol.T)
            #             fact = 12/365.25
            #             # attention, I should not use the days of years as the melt_f is
            #             # per month ~mean days of that year 12/daysofyear
            #             # to have the same unit of melt_f, which is
            #             # the monthly temperature sensitivity (kg /m² /mth /K),
            #             mb_daily = prcpsol - self.melt_f * temp2dformelt * fact
            #
            #             mb_month = np.sum(mb_daily, axis=1)
            #             # more correct than using a mean value for days in a month
            #             warnings.warn('there might be a problem with SEC_IN_MONTH'
            #                           'as February changes amount of days inbetween the years'
            #                           ' see test_monthly_glacier_massbalance()')
            #         else:
            #             # get 1D values for each height, no dependency on days
            #             t, temp2dformelt, prcp, prcpsol = self.get_monthly_climate(heights, year=year)
            #             mb_month = prcpsol - self.melt_f * temp2dformelt
        if self.melt_f_update == 'monthly':
            if climate_resol != 'monthly':
                raise InvalidWorkflowError('Need monthly climate_resol if melt_f_update is monthly')
            if not isinstance(year, float):
                raise InvalidParamsError('Year has to be the hydro float year '
                                         '_add_delta_mb_vary_melt_f with monthly melt_f_update,'
                                         'year needs to be a float')

        #if len(heights) != len(self.fl.dis_on_line):
        #    raise InvalidParamsError('length of the heights should be the same as '
        #                             'distance along flowline of pd_bucket dataframe,'
        #                             'use for heights e.g. ...fl.surface_h()')

        # if self.melt_f_update == 'annual' and climate_resol == 'annual'
        # from the last year, all potential snow should be no firn, and from this year, the
        # new snow is not yet added, so snow buckets should be empty
        # if self.melt_f_update == 'monthly'
        # from the last month, all potential snow should have been update and should be now
        # in the next older month bucket
        # the first bucket should be empty if:
        condi1 = climate_resol == 'annual' and self.melt_f_update == 'annual'
        # or
        condi2 = self.melt_f_update == 'monthly'
        # or 1st month with annual update and get_monthly_mb
        _, mc = floatyear_to_date(float(year))
        condi3 = climate_resol == 'monthly' and self.melt_f_update == 'annual' and mc == 1
        if condi1 or condi2 or condi3:
            if not np.any(self.pd_bucket[self.first_snow_bucket] == 0):
                raise InvalidWorkflowError('the first snow buckets should be empty in this use case '
                                           'but it is not, try e.g. to do '
                                           'reset_pd_mb_buckets() before and then rerun the task')

        # if climate_resol  is annual and monthly melt_f_update,
        # the snow buckets are not empty if we don't look at 1st year

        # let's do the same as in get_annual_mb of TIModel but with varying melt_f ...
        # heights = self.fl.surface_h

        if climate_resol == 'annual':
            _, temp2dformelt, _, prcpsol = self._get_2d_annual_climate(heights, year)
        elif climate_resol == 'monthly':
            if isinstance(type(year), int):
                raise InvalidWorkflowError('year should be a float with monthly climate resolution')
            # year is here float year, so it corresponds to 1 month of a specific year
            _, temp2dformelt, _, prcpsol = self.get_monthly_climate(heights, year)

        # len(buckets) +1 because also need ice melt_f that has no bucket
        if climate_resol == 'annual':
            #if self.melt_f_update == 'annual':
            # if melt_f_update annual: treat snow as the amount of solid prcp over that year
            # first add all solid prcp amount to the bucket
            # (the amount of snow that has melted in the same year is taken away in the for loop)
            #assert np.all(self.pd_bucket[self.first_snow_bucket].copy().values == 0)
            assert np.all(self.pd_bucket[self.first_snow_bucket].values == 0)
            self.pd_bucket[self.first_snow_bucket] = prcpsol.sum(axis=1)
            # at first, remaining temp for melt energy is all temp for melt
            # over the bucket loop this term will get gradually smaller until the remaining corresponds
            # to the potential ice melt
            remaining_tfm = temp2dformelt.sum(axis=1)
            # delta has to be set to the solid prcp (the melting part comes in the for loop)
            self.pd_bucket['delta_kg/m2'] = prcpsol.sum(axis=1)

        elif climate_resol == 'monthly':
            #elif self.melt_f_update == 'monthly':
            # is already summed up for get_monthly_mb
            # snow is here the amount of solid prcp over that month, ...
            # checked the unit -> ok
            # !!! in case of climate_resol =='monthly' but annual update,
            # first_snow_bucket is not empty!!!
            if self.melt_f_update == 'monthly':
                assert np.all(self.pd_bucket[self.first_snow_bucket].values == 0)
            # self.pd_bucket[self.first_snow_bucket] = self.pd_bucket[self.first_snow_bucket].values.copy() + prcpsol.flatten() # .sum(axis=1)
            # do NOT need the .copy()
            self.pd_bucket[self.first_snow_bucket] = self.pd_bucket[self.first_snow_bucket].values + prcpsol.flatten() # .sum(axis=1)
            remaining_tfm = temp2dformelt.flatten()  # .sum(axis=1)
            self.pd_bucket['delta_kg/m2'] = prcpsol.flatten()

        # now do the melting processes for each bucket in a loop
        if self.mb_type == 'mb_real_daily':  # and self.melt_f_update == 'annual':
            fact = 12/365.25
        else:
            fact = 1

        # how much tempformelt (tfm) would we need to remove all snow, firn of each bucket
        # in the case of snow it corresponds to tfm to melt all solid prcp
        # to convert from kg/m2 in the buckets to tfm [K], we use the melt_f values
        # of each bucket accordingly
        # no need of .copy(), because directly changed by divisor (/)
        # tfm_to_melt_b = self.pd_bucket[b].values / (self.melt_f_buckets[b]*fact)  # in K
        # ok, this can be done outside of th for-loop!!!

        # do not include delta_kg/m2 in pd_buckets, and also do not include melt_f of 'ice'
        tfm_to_melt_b = self.pd_bucket.values[:,:-1] / (np.array(list(self.melt_f_buckets.values()))[:-1]*fact)  # in K
        # need to transpose it to have the right shape
        tfm_to_melt_b = tfm_to_melt_b.T
        for e, b in enumerate(self.buckets):
            # there is no ice bucket !!!

            # now recompute buckests mass:
            # first mass inside of each bucket that has not melted
            # (i.e., it remains in the bucket)
            # -> to get this need to reconvert the tfm energy unit into kg/m2 by using the right melt_factor
            # e.g. at the uppest layers there is new snow added ...
            not_lost_bucket = utils.clip_min(tfm_to_melt_b[e] - remaining_tfm, 0) * self.melt_f_buckets[b] * fact

            # remaining tfm to melt older firn layers -> for the next loop ...
            remaining_tfm = utils.clip_min(remaining_tfm - tfm_to_melt_b[e], 0)
            # in case of ice, the remaining_tfm is only used to update once again delta_kg/m2


            # amount of kg/m2 lost in this bucket will be added to delta_kg/m2
            # not yet updated total bucket - amount of not lost mass of that bucket
            self.pd_bucket['delta_kg/m2'] += not_lost_bucket - self.pd_bucket[b].values # .copy(), not necessary because minus
            # update pd_bucket with what has not melted from the bucket
            self.pd_bucket[b] = not_lost_bucket

            # if all the remaining_tfm is zero, then go out of the loop,
            # to make the code faster!
            # e.g. in winter this should mean that we don't loop over all buckets (as tfm is very small)
            if np.all(remaining_tfm == 0):
                break

        # we assume that the ice bucket is infinite,
        # so everything that could be melted is included inside of delta_kg/m2
        # that means all the remaining tfm energy is used to melt the infinite ice bucket
        self.pd_bucket['delta_kg/m2'] += -remaining_tfm * self.melt_f_buckets['ice'] * fact

        return self.pd_bucket

    # @update_buckets.setter ### should this be a setter ??? because no argument ...
    def _update(self):
        """ this is called by get_annual_mb or get_monthly_mb after one year/one month to update
        the buckets as they got older """

        if np.any(np.isnan(self.pd_bucket['delta_kg/m2'])):
            raise InvalidWorkflowError('the buckets have been updated already, need'
                                       'to add_delta_mb first')
        if np.any(self.pd_bucket[self.buckets] < 0):
            raise ValueError('the buckets should only have positive values')
        # after 5 years of firn -> add!!! it to ice -> but we don't have ice bucket
        # now the same for the other buckets ...
        for e, b in enumerate(self.buckets[::-1]):
            # start with updating oldest snow pack ...
            if b != self.first_snow_bucket:
                # e.g. update ice with old firn_yr_5 ...
                # @Fabi: do I need the copy here?
                self.pd_bucket[b] = self.pd_bucket[self.buckets[::-1][e + 1]].copy()
                # we just overwrite it so we don't need to reset it to zero
                # self.pd_bucket[self.buckets[::-1][e+1]] = 0 #pd_bucket['firn_yr_4']
            elif b == self.first_snow_bucket:
                # the first_snow bucket is set to 0 after the update
                # (if annual update there is only one snow bucket)
                self.pd_bucket[b] = 0
        # reset delta_kg/m2 to make clear that it is updated
        self.pd_bucket['delta_kg/m2'] = np.NaN
        return self.pd_bucket

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
            at the moment only works with inversion heights!
        year: int
            integer year ! if melt_f_update='monthly', will loop over each month
        unit : str
            default is 'm of ice', nothing else implemented at the moment!
            TODO: include option of metre of glacier where the different densities
            are taken into account but in this case would need to add further columns in pd_buckets
            like: snow_delta_kg/m2 ... and so on
        bucket_output: bool (default is False)
            if True, also returns pd.Dataframe with the buckets
            which are not yet updated for the next year!
        add_climate: bool (default is False)
            for run_with_hydro (not yet implemented!)
            todo: implement and test it
        auto_spinup: bool (default is true)
            if True, it automatically computes the 6 years beforehand (however in this case,
            these 5 years, although saved in pd_mb_annual, had no spinup...)
            todo: maybe need to add a '_no_spinup' to those that had no spinup?
            or save them in a separate pd.DataFrame?
        **kwargs:
            other stuff passed to get_monthly_mb or to _add_delta_mb_vary_melt_f
            **kwargs necessary to take stuff we don't use (like fls...)

        """
        # when we set spinup_yrs to zero, then there should be no spinup occuring, even if spinup
        # is set to True
        if self.spinup_yrs == 0:
            spinup=False

        # we just convert to integer without checking ...
        # if not isinstance(year, int):
        #    raise InvalidParamsError('Year has to be the full year for get_annual_mb,'
        #                             'year needs to be an integer')
        #np.testing.assert_allclose(heights, self.inv_heights,
        #                           err_msg='the heights should correspond to the inversion heights',
        #                           )
        # np.testing.assert_allclose(heights, self.mod_heights)
        # only allow that heights
        if len(self.pd_mb_annual.columns) >0:
            if self.pd_mb_annual.columns[0] > self.pd_mb_annual.columns[-1]:
                raise InvalidWorkflowError('need to run years in ascending order! Maybe reset first!')
        assert heights[0] > heights[-1], "heights should be in descending order!"
        # otherwise pd_buckets does not work ...
        year = int(year)
        if year < 1979+6:
            # most climatic data starts in 1979, so if we want to
            # get the first 6 years can not use a spinup!!!
            # (or would need to think about sth. else, but not so
            # important right now!!!
            spinup = False
        if year in self.pd_mb_annual.columns and self.check_availability:
            #print(self.check_availability)
            #print(year)
            #print('takes existing annual mb')
            # if that year has already been computed
            # just get the annual_mb
            # (but don't need to compute something or update pd_buckets)
            mb_annual = self.pd_mb_annual[year].values
            if bucket_output:
                raise InvalidWorkflowError('if you want to output the buckets, you need to do'
                                           'reset_pd_mb_bucket() and rerun')
            return mb_annual
        else:
            if self.melt_f_update == 'annual':
                condi = int(year - self.spinup_yrs) not in self.pd_mb_annual.columns
            elif self.melt_f_update == 'monthly':
                try:
                    #condi = ([date_to_floatyear(year-5, m) for m in np.arange(1, 13, 1)] not in
                    #        self.pd_mb_monthly.columns)
                    # check if first month exists
                    condi = date_to_floatyear(year - self.spinup_yrs, 1) not in self.pd_mb_monthly.columns
                except:
                    condi = True
            if condi and spinup and auto_spinup:
                # if the spinup is run already, year would be in pd_mb_annual ...
                # I think we should always reset when
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
                # check if the years before that had been computed
                # (it should have been computed above if auto_spinup=True)
                for bef in np.arange(1, 6, 1):
                    if int(year-bef) not in self.pd_mb_annual.columns:
                        raise InvalidWorkflowError('need to do get_annual_mb of all spinup years'
                                                   '(default is 6) beforehand')
            if self.melt_f_update == 'annual':
                self.pd_bucket = self._add_delta_mb_vary_melt_f(heights, year=year,
                                                                climate_resol='annual')
                #mb_annual = self.pd_bucket['delta_kg/m2'].copy().values
                # I don't need the .copy() here if it is changed anyway
                # but then the line gets too long ...
                mb_annual = (self.pd_bucket['delta_kg/m2'].values - self.residual) /\
                            self.SEC_IN_YEAR / self.rho
                # update to one year later ...
                if bucket_output:
                    # copy because we want to output the bucket that is not yet updated!!!
                    pd_bucket = self.pd_bucket.copy()
                self._update()
                # save the annual mb
                self.pd_mb_annual[year] = mb_annual
                # this is done already if melt_f_update is monthly ...
            elif self.melt_f_update == 'monthly':
                # will be summed up over each month by getting pd_mb_annual
                # that is set in december (month=12)
                for m in np.arange(1, 13, 1):
                    floatyear = date_to_floatyear(year, m)
                    # OLD:
                    # self.pd_bucket = self._add_delta_mb_vary_melt_f(heights,
                    #                                                year=floatyear,
                    #                                                climate_resol='monthly')
                    # mb_annual += self.pd_bucket['delta_kg/m2'].copy().values
                    # let the buckets "age" one month
                    # self._update()
                    # instead use get_monthly_mb()
                    out = self.get_monthly_mb(heights, year=floatyear, unit=unit,
                                        bucket_output=bucket_output, spinup=spinup,
                                        add_climate=add_climate,
                                        auto_spinup = auto_spinup,
                                        **kwargs)
                    if bucket_output and m == 12:
                        pd_bucket = out[1]
                # get mb_annual that is produced by
                mb_annual = self.pd_mb_annual[year].values
            if bucket_output:
                return mb_annual, pd_bucket
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

        year should be the hydro float year,
        so it corresponds to 1 month of a specific year

        Parameters
        ----------
        heights : np.array
            at the moment only works with inversion heights!
        year: int
            year has to be given as hydro float year from what the (integer) year and month is taken,
            hence year 2000 -> y=2000, m = 1, & year = 2000.09, y=2000, m=2 ...
            which corresponds to the real year 1999 and months October or November
            if hydro year starts in October
        unit : str
            default is 'm of ice', nothing else implemented at the moment!
            TODO: include option of metre of glacier where the different densities
            are taken into account but in this case would need to add further columns in pd_buckets
            like: snow_delta_kg/m2 ... and so on
        bucket_output: bool (default is False)
            if True, returns as second output the pd.Dataframe with the buckets
            which are not yet updated for the next month!
        add_climate: bool (default is False)
            for run_with_hydro (not yet implemented!)
            todo: implement and test it
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
        #np.testing.assert_allclose(heights, self.inv_heights,
        #                           err_msg='the heights should correspond to the inversion heights',
        #                           )

        if self.spinup_yrs == 0:
            spinup=False

        if len(self.pd_mb_monthly.columns) >1:
            if self.pd_mb_monthly.columns[0] > self.pd_mb_monthly.columns[-1]:
                raise InvalidWorkflowError('need to run months in ascending order! Maybe reset first!')

        assert heights[0] > heights[-1], "heights should be in descending order!"


        if year in self.pd_mb_monthly.columns and self.check_availability:
            # if that float year has already been computed
            # just get the monthly_mb
            # (but don't need to compute something or update pd_buckets)
            print('takes existing monthly mb')
            mb_month = self.pd_mb_monthly[year].values
            if bucket_output:
                raise InvalidWorkflowError('if you want to output the buckets, you need to do'
                                           'reset_pd_mb_bucket() and rerun')
            return mb_month
        else:
            # only allow float years
            if not isinstance(year, float):
                raise InvalidParamsError('Year has to be the hydro float year '
                                         'for get_monthly_mb,'
                                         'year needs to be a float')
            # need to somehow check if the months before that were computed as well
            # otherwise monthly mass-balance does not make sense when using sfc type distinction ...
            y, m = floatyear_to_date(year)

            # spinup
            # this is independent of melt_f_update, always need to reset if the spinup years before
            # were not computed
            try:
                # it is sufficient to check if the first month is there
                # if there is any other problem it will raise an error later anyways
                condi = date_to_floatyear(y - self.spinup_yrs, 1) not in self.pd_mb_monthly.columns
            except:
                condi = True
            # if annual melt_f_update and annual_mb exists from previous years
            # (by doing , get_annual_mb), and there exist not yet the mass-balance for the next year
            # then no reset & new spinup is necessary
            condi_add = int(y + 1) in self.pd_mb_annual.columns
            if self.melt_f_update == 'annual' and condi_add:
                condi = condi_add

            if condi and spinup and auto_spinup:
                # if the spinup is run already, year would be in pd_mb_annual ...
                # I think we should always reset when
                # doing the spinup and not having computed year==2000
                self.reset_pd_mb_bucket()
                for yr in np.arange(y-self.spinup_yrs, y):
                    self.get_annual_mb(heights, year=yr, unit=unit, bucket_output=False,
                                       spinup=False, add_climate=False,
                                       auto_spinup=False, **kwargs)
                # need to also run the months before m
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
                # check if the months before had been computed for that year?
                for mb in np.arange(1, m, 1):
                   floatyear_before = date_to_floatyear(y, mb)
                   if floatyear_before not in self.pd_mb_monthly.columns:
                       raise InvalidWorkflowError('need to get_monthly_mb of each month '
                                                  'of that year beforehand')

            if self.melt_f_update == 'annual':
                # raise NotImplementedError('get_monthly_mb works at the moment '
                #                          'only when melt_f is updated monthly')
                # for annual melt_f_update, this is a bit tricky because the
                # _update should only be done after a full year

                self.pd_bucket = self._add_delta_mb_vary_melt_f(heights,
                                                                year=year,
                                                                climate_resol='monthly')
                # ?Fabi: if I want to have it in two lines, can I then  still put the .copy() away?
                #mb_month = self.pd_bucket['delta_kg/m2'].copy().values
                #mb_month -= self.residual * self.SEC_IN_MONTH / self.SEC_IN_YEAR
                mb_month = self.pd_bucket['delta_kg/m2'].values - \
                           self.residual * self.SEC_IN_MONTH / self.SEC_IN_YEAR
                # need to flatten for mb_pseudo_daily otherwise it gives the wrong shape
                mb_month = mb_month.flatten() / self.SEC_IN_MONTH / self.rho
                # save the mass-balance to pd_mb_monthly
                self.pd_mb_monthly[year] = mb_month
                # if annual, only want to update at the end of the year
                if m == 12:
                    # sum up the monthly mb and update the annual pd_mb
                    # but as the mb is  m of ice per second -> use the mean !!!
                    # only works if same height ... and if
                    if int(year) not in self.pd_mb_annual.columns:
                        condi = [int(c) == int(year) for c in self.pd_mb_monthly.columns]
                        if len(self.pd_mb_monthly.loc[:, condi].columns) != 12:
                            raise InvalidWorkflowError('Not all months were computed beforehand,'
                                                       'need to do get_monthly_mb for all months before')
                        # sum of all columns of that year ...
                        self.pd_mb_annual[int(year)] = self.pd_mb_monthly.loc[:, condi].mean(axis=1).values
                    # do the update
                    if bucket_output:
                        pd_bucket = self.pd_bucket.copy()
                    self._update()
                if bucket_output and m != 12:
                    pd_bucket = self.pd_bucket.copy()
                if bucket_output:
                    return mb_month, pd_bucket
                else:
                    return mb_month

            elif self.melt_f_update == 'monthly':
                # year is in float year here
                self.pd_bucket = self._add_delta_mb_vary_melt_f(heights, year=year,
                                                                climate_resol='monthly')
                #fabi copy problem same here
                #mb_month = self.pd_bucket['delta_kg/m2'].copy().values

                # residual is in mm w.e per year, so SEC_IN_MONTH .. but mb_month
                # should be per month!
                #mb_month -= self.residual * self.SEC_IN_MONTH / self.SEC_IN_YEAR
                mb_month = self.pd_bucket['delta_kg/m2'].values - \
                           self.residual * self.SEC_IN_MONTH / self.SEC_IN_YEAR
                # flattening for mb_pseudo_daily otherwise it gives the wrong shape
                mb_month = mb_month.flatten() / self.SEC_IN_MONTH / self.rho
                #todo: if add_climate:
                #    if self.mb_type == 'mb_real_daily':
                #        # for run_with_hydro want to get monthly output (sum of daily),
                #        # if we want daily output in run_with_hydro, then we need another option here
                #        # or rather directly use get_daily_mb()
                #        prcp = prcp.sum(axis=1)
                #        prcpsol = prcpsol.sum(axis=1)
                #    return (mb_month / SEC_IN_MONTH / self.rho, t, temp2dformelt,
                #            prcp, prcpsol)
                # instead of SEC_IN_MONTH, use instead len(prcpsol.T)==daysinmonth ???

                # get the ouput before the update
                if bucket_output:
                    pd_bucket = self.pd_bucket.copy()
                # need to update it after each month
                self._update()
                # save the mass-balance to pd_mb_monthly
                self.pd_mb_monthly[year] = mb_month

                # if annual, also want to save the annual mb
                if m == 12:
                    # sum up the monthly mb and update the annual pd_mb
                    # run this if this year not yet inside or if we don't want to check availability
                    # e.g. for random / constant runs !!!
                    if int(year) not in self.pd_mb_annual.columns or not self.check_availability:
                        condi = [int(c) == int(year) for c in self.pd_mb_monthly.columns]
                        # sum of all columns of that year ...
                        if len(self.pd_mb_monthly.loc[:, condi].columns) != 12:
                            raise InvalidWorkflowError('Not all months were computed beforehand,'
                                                       'need to do get_monthly_mb for all months before')
                        self.pd_mb_annual[int(year)] = self.pd_mb_monthly.loc[:, condi].mean(axis=1).values
                if bucket_output:
                    return mb_month, pd_bucket
                else:
                    return mb_month

    def get_daily_mb(self):
        raise NotImplementedError('this has to be implemented ... ')


# copy of MultipleFlowlineMassBalance that works with TIModel
class MultipleFlowlineMassBalance_TIModel(MassBalanceModel):
    """ Adapted MultipleFlowlineMassBalance that is compatible for all TIModel classes

    TODO: do better documentation

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
        fls :
        melt_f :
        prcp-fac :
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

    if interpolation_optim=True ad mb_model_sub_class
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
            raise InvalidWorkflowError('need to set y0 as we do not use tstar in this case')
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
            kwargs2 = {'check_availability' : False,
                       'interpolation_optim':interpolation_optim,
                       'hbins':self.hbins}

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
            self.mbmod._pd_mb_template = pd.DataFrame(0,
                                                      index=self.hbins[::-1],
                                                      columns=[] )

            self.mbmod._pd_mb_template.index.name = 'hbins_height'

            self.mbmod.pd_mb_monthly = self.mbmod._pd_mb_template.copy()
            self.mbmod.pd_mb_annual = self.mbmod._pd_mb_template.copy()

            pd_bucket = self.mbmod._pd_mb_template.copy()
            pd_bucket[self.mbmod.columns] = 0
            self.mbmod.pd_bucket = pd_bucket
        else:
            self.mbmod.reset_pd_mb_bucket(init_model_fls=init_model_fls)
    @lazy_property
    def interp_yr(self, **kwargs):
        #todo: there is a problem with **kwargs and fls, but need kwargs for spinup params
        # when using TIModel_Sfc_Type
        # annual MB
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
        #print(mb_on_h, len(self.years))
        #print(interp1d(self.hbins[::-1], mb_on_h / len(self.years)))
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
            #mb = self.interp_yr(heights)            #raise NotImplementedError('need to implement it for TIModel_Sfc_Type')
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
        #print(len(pok))
        tmp = self.mbmod.temp[pok]
        #print(self.mbmod.ys, self.mbmod.ye)
        #print((len(tmp) // 12))
        #print((halfsize * 2 + 1))
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
def fixed_geometry_mass_balance_TIModel(gdir, ys=None, ye=None, years=None,
                                monthly_step=False,
                                use_inversion_flowlines=True,
                                climate_filename='climate_historical',
                                climate_input_filesuffix='',
                                ds_gcm = None,
                                **kwargs):
    """Computes the mass-balance with climate input from e.g. CRU or a GCM.

    TODO: do documentation
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
