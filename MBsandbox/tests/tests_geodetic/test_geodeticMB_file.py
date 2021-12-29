#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 07:17:03 2021

@author: lilianschuster
"""

# tests for mass balances 
import warnings
import os
import numpy as np
import pandas as pd
import pytest
import oggm 
from oggm import cfg, utils, workflow
from MBsandbox.mbmod_daily_oneflowline import TIModel, process_era5_daily_data
warnings.filterwarnings("once", category=DeprecationWarning)  # noqa: E402
from numpy.testing import assert_allclose
from oggm.exceptions import InvalidParamsError

#############
import unittest
import os
import shutil
from distutils.version import LooseVersion
import pytest
import warnings

import shapely.geometry as shpg
import numpy as np
import pandas as pd
import xarray as xr

salem = pytest.importorskip('salem')
rasterio = pytest.importorskip('rasterio')
gpd = pytest.importorskip('geopandas')

# Local imports
import oggm
from oggm.core import (gis, inversion, climate, centerlines,
                       flowline, massbalance)
from oggm.shop import gcm_climate
import oggm.cfg as cfg
from oggm import utils, tasks
from oggm.utils import get_demo_file, tuple2int
from oggm.tests.funcs import (get_test_dir, init_columbia, init_columbia_eb,
                              apply_test_ref_tstars)
from oggm import workflow
from oggm.exceptions import InvalidWorkflowError, MassBalanceCalibrationError
##########





# %%
# # import the new models
# from mbmod_daily_oneflowline import process_era5_daily_data, mb_modules, BASENAMES
# from help_func import compute_stat, minimize_bias, optimize_std_quot_brentq
# import xarray as xr


# def test_hydro_years():
#     cfg.initialize()
#     working_dir = '/home/lilianschuster/Schreibtisch/PhD/oggm_files/oneFlowline'
#     # this needs to be changed if working on another computer
#     if not os.path.exists(working_dir):
#         working_dir = utils.gettempdir(dirname='OGGM_mb_type_intercomparison', reset=True)
        
#     cfg.PATHS['working_dir'] = working_dir
#     # use Huss flowlines
#     base_url = 'https://cluster.klima.uni-bremen.de/~fmaussion/gdirs/prepro_l2_202010/elevbands_fl'
    
#     # get HEF glacier
#     df = utils.get_rgi_glacier_entities(['RGI60-11.00897'])
#     gdirs = workflow.init_glacier_directories(df, from_prepro_level=2,
#                                               prepro_border=40, 
#                                       prepro_base_url=base_url,
#                                       prepro_rgi_version='62')
    
    
#     cfg.PARAMS['hydro_month_nh'] = 1
#     gd = gdirs[0]
    
#     h, w = gd.get_inversion_flowline_hw()
#     fls = gd.read_pickle('inversion_flowlines')
    
#     cfg.PARAMS['baseline_climate'] = 'ERA5dr'
#     oggm.shop.ecmwf.process_ecmwf_data(gd, dataset = 'ERA5dr')
#     f = gd.get_filepath('climate_historical', filesuffix='')
#     test_climate = xr.open_dataset(f)
#     assert test_climate.time[0] == np.datetime64('1979-01-01')
#     assert test_climate.time[-1] == np.datetime64('2018-12-01')
    
#     # now test it for ERA5_daily 
#     cfg.PARAMS['baseline_climate'] = 'ERA5_daily'
#     process_era5_daily_data(gd)
#     f = gd.get_filepath('climate_historical', filesuffix='_daily')
#     test_climate = xr.open_dataset(f)
#     assert test_climate.time[0] == np.datetime64('1979-01-01')
#     assert test_climate.time[-1] == np.datetime64('2018-12-31')
# %%
@pytest.mark.skip(reason="have to work on that to include pd_geodetic...")
def test_size():
    # amount of total glaciers depends on minmum-area threshold, DEM ... 
    oggm_world_amount = 216502 # according to glacier explorer
    np.testing.assert_allclose(len(pd_geodetic_loc), oggm_world_amount,
                                rtol = 0.02)
    # testing if area is around the same
    oggm_world_area = 746093 # in km2
    np.testing.assert_allclose(pd_geodetic_loc['area'].sum(),
                           oggm_world_area,
                           rtol = 0.1)
                               
    rho_geodetic = 850 # approximate value that is used kg/m3
    # there are some glaciers where there are no msm in period 2000-2020 (~7000)
    # and for some there are dhdt values but not dmdtda values (why?)
    # in total: 210068 glaciers with noNaNs!!!
    pd_geodetic_noNaNs = pd_geodetic_loc[~(np.isnan(pd_geodetic_loc.dmdtda) | np.isnan(pd_geodetic_loc.dhdt))]
    np.testing.assert_allclose(pd_geodetic_noNaNs.dhdt*rho_geodetic/1000,
                               pd_geodetic_noNaNs.dmdtda)
    
    # check: amount of rows corresponds to amount of glaciers in Alps
    # (according to RGI from OGGM)
    assert len(pd_geodetic_alps)==3927 
    
    # 2193 km2 is for the year 2000 according to 
    # https://www.nature.com/articles/s41467-020-16818-0
    # not so sure if area corresponds to 2000, to 2020 or a mean of both?,
    # as area is assumed constant 
    np.testing.assert_allclose(pd_geodetic_alps['area'].sum(), 2193, 100) 
# %%

@pytest.mark.skip(reason="have to work on that to include pd_geodetic...")
def test_geodetic_glaciologicalMB_HEF():

    cfg.initialize()
    cfg.PARAMS['use_multiprocessing'] = False

    working_dir = '/home/lilianschuster/Schreibtisch/PhD/oggm_files/oneFlowline'
    # this needs to be changed if working on another computer
    if not os.path.exists(working_dir):
        working_dir = utils.gettempdir(dirname='OGGM_mb_type_intercomparison', reset=True)
        
    cfg.PATHS['working_dir'] = working_dir
    # use Huss flowlines
    base_url = ('https://cluster.klima.uni-bremen.de/~oggm/gdirs/oggm_v1.4/'
                'L1-L2_files/elev_bands')
    
    # get HEF glacier
    df = utils.get_rgi_glacier_entities(['RGI60-11.00897'])
    gdirs = workflow.init_glacier_directories(df, from_prepro_level=2,
                                              prepro_border=10,
                                            prepro_base_url=base_url,
                                      prepro_rgi_version='62')
    gd = gdirs[0]
    
    h, w = gd.get_inversion_flowline_hw()
    fls = gd.read_pickle('inversion_flowlines')
    
    cfg.PARAMS['baseline_climate'] = 'ERA5dr'
    oggm.shop.ecmwf.process_ecmwf_data(gd, dataset='ERA5dr')
    ######################
    
    # test area is similar to OGGM gdir HEF area
    np.testing.assert_allclose(pd_geodetic_alps.loc['RGI60-11.00897'].area, gd.rgi_area_km2, rtol=0.01)
    
    glaciological_mb_data_HEF_mean = gd.get_ref_mb_data()['ANNUAL_BALANCE'].loc[2000:].mean() #.sel(2000,2020)
    
    # test if mass balance in period 2000 and 2020 of glaciological method is similar to the geodetic one 
    np.testing.assert_allclose(pd_geodetic_alps.loc['RGI60-11.00897'].dmdtda*1000,
                               glaciological_mb_data_HEF_mean, rtol = 0.05 )
