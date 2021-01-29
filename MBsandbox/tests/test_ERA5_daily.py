#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 18:34:35 2020

@author: lilianschuster
"""
import warnings
import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_allclose
import oggm

# imports from oggm
from oggm.exceptions import InvalidParamsError
from oggm.shop.ecmwf import get_ecmwf_file

# imports from MBsandbox package modules
from MBsandbox.mbmod_daily_oneflowline import process_era5_daily_data

warnings.filterwarnings("once", category=DeprecationWarning)
# %%


def test_process_era5_daily_data(gdir):

    process_era5_daily_data(gdir, y0=1979, y1=2018)

    filename = 'climate_historical_daily'
    fpath = gdir.get_filepath(filename)

    # check the climate files of an individual glacier (Hintereisferner)
    xr_nc = xr.open_dataset(fpath)
    assert np.all(xr_nc.prcp) > 0
    # to be sure that there are no erroneaous filling values inside
    assert np.all(xr_nc.prcp) < 10000
    # temperature values are in °C and in the right range
    assert np.all(xr_nc.temp) > -100
    assert np.all(xr_nc.temp) < 100

    # temperature gradient should be in the following range
    assert np.all(xr_nc.gradient > -0.015)
    assert np.all(xr_nc.gradient < -0.002)

    # all lapse rates/ precipitation values in one month should be equal
    # because only temperature is on daily basis
    np.testing.assert_allclose(xr_nc.resample(time='1M').std().prcp,
                               0, atol=1e-3)
    np.testing.assert_allclose(xr_nc.resample(time='1M').std().gradient,
                               0, atol=1e-3)

    # summed up monthly precipitation from daily dataset
    xr_nc_prcp_m = xr_nc.prcp.resample(time='1M').sum()

    oggm.shop.ecmwf.process_ecmwf_data(gdir, dataset="ERA5",
                                       y0=1979, y1=2018)
    filename = 'climate_historical'

    fpath = gdir.get_filepath(filename)
    xr_nc_monthly = xr.open_dataset(fpath)

    # check if summed up monthly precipitation from daily
    # dataset equals approx. to the ERA5 monthly prcp
    np.testing.assert_allclose(xr_nc_prcp_m.values,
                               xr_nc_monthly.prcp.values, rtol=1e-4)

    xr_nc_temp_m = xr_nc.temp.resample(time='1M').mean()
    # check if mean temperature from daily dataset equals
    # approx. to the ERA5 monthly temp.
    np.testing.assert_allclose(xr_nc_temp_m.values,
                               xr_nc_monthly.temp.values, atol=0.05)

    with pytest.raises(InvalidParamsError):
        # dataset only goes from 1979--2018
        process_era5_daily_data(gdir, y0=1979, y1=2019)

        # in cfg.PARAMS that is initiated during testing,
        # cfg.PARAMS[hydro_month_nh = 10], this is in conflict with 8
        process_era5_daily_data(gdir, y0=1979, y1=2018, hydro_month_nh=8)


def test_ERA5_daily_dataset():

    dataset = 'ERA5_daily'

    ds_ERA5_daily = xr.open_dataset(get_ecmwf_file(dataset, 'tmp'))
    ds_ERA5_daily['time.month'][0] == 1
    ds_ERA5_daily['time.month'][-1] == 12

    # checks if it is in Kelvin
    assert np.all(ds_ERA5_daily.t2m > 0)
    # not too high temperatures
    assert np.max(ds_ERA5_daily.t2m) < 350

    # ERA5 daily should start in 1979 and end in 2018
    assert ds_ERA5_daily['time.year'][0] == 1979
    assert ds_ERA5_daily['time.year'][-1] == 2018
    assert ds_ERA5_daily['time.month'][0] == 1
    assert ds_ERA5_daily['time.month'][-1] == 12

    # compare the daily dataset to the monthly:
    ds_ERA5 = xr.open_dataset(get_ecmwf_file('ERA5', 'tmp'))

    # check if for Hintereisferner, both datasets produce similar monthly
    # temperature time series if ERA5_daily is resampled over month
    lon = 10.7584
    lat = 46.8003

    # compute all the distances and choose nearest gridpoint
    # this also checks if the flattened version is used!
    c = (ds_ERA5_daily.longitude - lon)**2 + (ds_ERA5_daily.latitude - lat)**2
    ds_ERA5_daily_g = ds_ERA5_daily.isel(points=c.argmin())

    ds_ERA5_g = ds_ERA5.sel(time=slice('1979-01-01',
                                       '2018-12-01')).sel(longitude=lon,
                                                          latitude=lat,
                                                          method='nearest')
    # do we use the same longitude/latitudes
    assert ds_ERA5_daily_g.longitude.values == ds_ERA5_g.longitude.values
    assert ds_ERA5_daily_g.latitude.values == ds_ERA5_g.latitude.values

    # do the two datasets have the same monthly temperatures?
    # at the neares gridpoint to HEF (some rounding errors are allowed)
    assert_allclose(ds_ERA5_daily_g.resample(time='1M').mean().t2m.values,
                    ds_ERA5_g.t2m.values, rtol=1e-4)
