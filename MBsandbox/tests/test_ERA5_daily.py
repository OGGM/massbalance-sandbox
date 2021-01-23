#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 18:34:35 2020

@author: lilianschuster
"""

# tests for mass balances 
import warnings
import os
from functools import partial
import numpy as np
import pandas as pd
import pytest

# Local imports
import oggm
import xarray as xr
from oggm import utils, workflow, cfg
from oggm.core import gcm_climate, climate, inversion, centerlines

from oggm.tests.funcs import get_test_dir


import matplotlib.pyplot as plt
from oggm.core.flowline import (FluxBasedModel, FlowlineModel,
                                init_present_time_glacier, glacier_from_netcdf,
                                RectangularBedFlowline, TrapezoidalBedFlowline,
                                ParabolicBedFlowline, MixedBedFlowline,
                                flowline_from_dataset, FileModel,
                                run_constant_climate, run_random_climate,
                                run_from_climate_data)
# from oggm.utils._workflow import *

from oggm.exceptions import InvalidWorkflowError, InvalidParamsError
# from oggm.cfg import cfg
from oggm.shop.ecmwf import set_ecmwf_url, get_ecmwf_file


# import the new models

warnings.filterwarnings("once", category=DeprecationWarning)  # noqa: E402




# %%
# add era5_daily dataset, this only works with process_era5_daily_data
#BASENAMES = {}
#BASENAMES['ERA5_daily'] =   { 
#        'inv':'era5/daily/v1.0/era5_glacier_invariant_flat.nc',
#        'tmp':'era5/daily/v1.0/era5_daily_t2m_1979-2018_flat.nc'
#        # only glacier-relevant gridpoints included!
#        }

from MBsandbox.mbmod_daily_oneflowline import process_era5_daily_data

# %%
def test_process_era5_daily_data():
    
    cfg.initialize()
    
    test_dir = '/home/lilianschuster/Schreibtisch/PhD/oggm_files/tests'
    if not os.path.exists(test_dir):
        test_dir = utils.gettempdir(dirname='OGGM_era5_daily_test',
                                    reset=True)
    cfg.PATHS['working_dir'] = test_dir

    # base_url = 'https://cluster.klima.uni-bremen.de/~fmaussion/gdirs/prepro_l2_202010/elevbands_fl'
    base_url = 'https://cluster.klima.uni-bremen.de/~fmaussion/gdirs/prepro_l2_202010/elevbands_fl_with_consensus'

    
    df = utils.get_rgi_glacier_entities(['RGI60-11.00897'])
    gdirs = workflow.init_glacier_directories(df, from_prepro_level=2,
                                              prepro_border=40, 
                                      prepro_base_url=base_url,
                                      prepro_rgi_version='62')
    gdir = gdirs[0]
    
    

    process_era5_daily_data(gdir, y0 =1979, y1 = 2018, 
                            )

    filename = 'climate_historical_daily'
    fpath = gdir.get_filepath(filename) # , filesuffix='_daily')

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
                               0, atol = 1e-3)
    np.testing.assert_allclose(xr_nc.resample(time='1M').std().gradient,
                               0, atol = 1e-3)
    
    # summed up monthly precipitation from daily dataset
    xr_nc_prcp_m = xr_nc.prcp.resample(time='1M').sum()
    
    oggm.shop.ecmwf.process_ecmwf_data(gdir, dataset = "ERA5",
                                       y0 =1979, y1 = 2018, 
                                       ) #output_filesuffix = '_monthly')
    filename = 'climate_historical'

    fpath = gdir.get_filepath(filename) #, filesuffix='_monthly')
    xr_nc_monthly = xr.open_dataset(fpath)
    
    # check if summed up monthly precipitation from daily
    # dataset equals approx. to the ERA5 monthly prcp 
    np.testing.assert_allclose(xr_nc_prcp_m.values,
                               xr_nc_monthly.prcp.values, rtol= 1e-4)
    
    xr_nc_temp_m = xr_nc.temp.resample(time='1M').mean()
    # check if mean temperature from daily dataset equals
    # approx. to the ERA5 monthly temp.
    np.testing.assert_allclose(xr_nc_temp_m.values, 
                               xr_nc_monthly.temp.values, atol= 0.05)
    
    
    # 
    with pytest.raises(InvalidParamsError):
        # dataset only goes from 1979--2018
        process_era5_daily_data(gdir, y0 =1979, y1 = 2019)
                                      # output_filesuffix = '_daily')
        
        # in cfg.PARAMS that is initiated during testing, 
        # cfg.PARAMS[hydro_month_nh = 10], this is in conflict with 8        
        process_era5_daily_data(gdir, y0 =1979, y1 = 2018, hydro_month_nh=8)
                                 #      output_filesuffix = '_daily')

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
                                       '2018-12-01')).sel(longitude = lon,
                                                          latitude = lat,
                                                          method = 'nearest')
    # do we use the same longitude/latitudes
    assert ds_ERA5_daily_g.longitude.values == ds_ERA5_g.longitude.values
    assert ds_ERA5_daily_g.latitude.values == ds_ERA5_g.latitude.values
    
    # do the two datasets have the same monthly temperatures?
    # at the neares gridpoint to HEF (some rounding errors are allowed)
    np.testing.assert_allclose(ds_ERA5_daily_g.resample(time='1M').mean().t2m.values, 
                               ds_ERA5_g.t2m.values, rtol=1e-4)