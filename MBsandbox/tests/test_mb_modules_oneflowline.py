#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 18:34:35 2020

@author: lilianschuster
"""

# tests for mass balances 
import warnings
warnings.filterwarnings("once", category=DeprecationWarning)  # noqa: E402

from functools import partial
import time
import numpy as np
import pandas as pd
import shapely.geometry as shpg
from numpy.testing import assert_allclose
import pytest
import scipy

# Local imports
import oggm
from oggm.core import massbalance
from oggm.core.massbalance import LinearMassBalance
import xarray as xr
from oggm import utils, workflow, tasks, cfg
from oggm.core import gcm_climate, climate, inversion, centerlines
from oggm.cfg import SEC_IN_DAY, SEC_IN_YEAR, SEC_IN_MONTH
from oggm.utils import get_demo_file

from oggm.tests.funcs import get_test_dir
from oggm.tests.funcs import (dummy_bumpy_bed, dummy_constant_bed,
                              dummy_constant_bed_cliff,
                              dummy_mixed_bed, bu_tidewater_bed,
                              dummy_noisy_bed, dummy_parabolic_bed,
                              dummy_trapezoidal_bed, dummy_width_bed,
                              dummy_width_bed_tributary)

import matplotlib.pyplot as plt
from oggm.core.flowline import (FluxBasedModel, FlowlineModel,
                                init_present_time_glacier, glacier_from_netcdf,
                                RectangularBedFlowline, TrapezoidalBedFlowline,
                                ParabolicBedFlowline, MixedBedFlowline,
                                flowline_from_dataset, FileModel,
                                run_constant_climate, run_random_climate,
                                run_from_climate_data)
#from oggm.utils._workflow import *
import os
from oggm.exceptions import InvalidParamsError

# above just the same input as in test_models

# import the new models
from MBsandbox.help_func import (compute_stat, minimize_bias,
                                 optimize_std_quot_brentq)
# add era5_daily dataset, this only works with process_era5_daily_data
# BASENAMES = {}
#BASENAMES['ERA5_daily'] =   { 
#        'inv':'era5/daily/v1.0/era5_glacier_invariant_flat.nc',
#        'tmp':'era5/daily/v1.0/era5_daily_t2m_1979-2018_flat.nc'
#        # only glacier-relevant gridpoints included!
#        }

from MBsandbox.mbmod_daily_oneflowline import (process_era5_daily_data,
                                               mb_modules)


FluxBasedModel = partial(FluxBasedModel, inplace=True)
FlowlineModel = partial(FlowlineModel, inplace=True)
# %%
# I can't use the test directory because of different flowlines
# # some stuff from hef_gdir...
# cfg.initialize(logging_level='INFO')
# border = 40
# testdir = os.path.join(get_test_dir(), 'tmp_border{}'.format(border))
# if not os.path.exists(testdir):
#     os.makedirs(testdir)
#     reset = True
    
# cfg.PATHS['working_dir'] = testdir
cfg.initialize()
# border = 40
# testdir = os.path.join(get_test_dir(), 'tmp_border{}'.format(border))
# if not os.path.exists(testdir):
#     os.makedirs(testdir)
    # reset = True
#test_dir = '/home/lilianschuster/OGGM/tests/f2cb553755951450761f5ee64e4a4c433ef61521/tmp_border40/per_glacier/RGI60-11/RGI60-11.00/RGI60-11.00897/'
#cfg.PATHS['working_dir'] = test_dir
test_dir = '/home/lilianschuster/Schreibtisch/PhD/oggm_files/tests'
if not os.path.exists(test_dir):
    test_dir = utils.gettempdir(dirname='OGGM_mb_type_intercomparison_test',
                                reset=True)
# %%
cfg.PATHS['working_dir'] = test_dir
base_url = 'https://cluster.klima.uni-bremen.de/~fmaussion/gdirs/prepro_l2_202010/elevbands_fl_with_consensus'

# %%
# maybe put this into a function??? 
df = utils.get_rgi_glacier_entities(['RGI60-11.00897'])
gdirs = workflow.init_glacier_directories(df, from_prepro_level=2,
                                          prepro_border=40,
                                  prepro_base_url=base_url,
                                  prepro_rgi_version='62')
gdir = gdirs[0]


# optimal values for HEF of mu_star for cte lapse rates
mu_star_opt_cte = {'mb_monthly': 213.540352,
               'mb_daily':181.054877,
               'mb_real_daily':180.404965}
# optimal values of mu_star when using variable lapse rates:
# 'var_an_cycle'
mu_star_opt_var = {'mb_monthly': 195.304843, 
                            'mb_daily':167.214101, 
                            'mb_real_daily':159.901181}
pf = 2.5
# %%
def test_hydro_years_HEF():  
    # only very basic test, the other stuff is done in oggm man basis
    # test if it also works for hydro_month ==1, necessary for geodetic mb
    # if hydro_month ==1, and msm start in 1979, then hydro_year should also be 1979??
    # this works only if changes in _workflow.py are made in write_monthly_climate...
    cfg.PARAMS['hydro_month_nh'] = 1
    
    h, w = gdir.get_inversion_flowline_hw()
    fls = gdir.read_pickle('inversion_flowlines')
    
    cfg.PARAMS['baseline_climate'] = 'ERA5dr'
    oggm.shop.ecmwf.process_ecmwf_data(gdir, dataset = 'ERA5dr')
    f = gdir.get_filepath('climate_historical', filesuffix='')
    test_climate = xr.open_dataset(f)
    assert test_climate.time[0] == np.datetime64('1979-01-01')
    assert test_climate.time[-1] == np.datetime64('2018-12-01')
    
    # now test it for ERA5_daily 
    cfg.PARAMS['baseline_climate'] = 'ERA5_daily'
    process_era5_daily_data(gdir)
    f = gdir.get_filepath('climate_historical', filesuffix='_daily')
    test_climate = xr.open_dataset(f)
    assert test_climate.time[0] == np.datetime64('1979-01-01')
    assert test_climate.time[-1] == np.datetime64('2018-12-31')
    # 

# %%
def test_minimize_bias():
    
    # important to initialize again, otherwise hydro_month_nh =1 
    # from test_hydro_years_HEF...
    cfg.initialize()

    # just checks if minimisation gives always same results 
    grad_type = 'cte'
    N = 2000
    loop = False

    for mb_type in ['mb_real_daily', 'mb_monthly','mb_daily']:

        if mb_type !='mb_real_daily':
            cfg.PARAMS['baseline_climate'] = 'ERA5dr'
            oggm.shop.ecmwf.process_ecmwf_data(gdir, dataset = 'ERA5dr') 
        else:
            cfg.PARAMS['baseline_climate'] = 'ERA5_daily'
            process_era5_daily_data(gdir)

            
        DDF_opt = scipy.optimize.brentq(minimize_bias,1,10000,
                                        args=(mb_type, grad_type, gdir, N,
                                        pf, loop, False), disp=True,
                                        xtol=0.1) 
        hgts, widths = gdir.get_inversion_flowline_hw()
        mbdf = gdir.get_ref_mb_data()
        # check if they give the same optimal DDF
        # print(mu_star_opt_cte[mb_type], DDF_opt)
        assert np.round(mu_star_opt_cte[mb_type]/DDF_opt, 3) == 1
        
        gd_mb = mb_modules(gdir, DDF_opt,   mb_type=mb_type,
                                           grad_type=grad_type)
        mb_specific = gd_mb.get_specific_mb(heights = hgts, widths = widths,
                                            year = mbdf.index.values)
        
        RMSD, bias , rcor, quot_std = compute_stat(mb_specific=mb_specific,
                                                   mbdf=mbdf)
        
        # check if the bias is optimised
        assert bias.round() == 0
# %%

# %%
def test_optimize_std_quot_brentq():
    
    cfg.initialize()

    # check if double optimisation of bias and std_quotient works
    
    grad_type = 'cte'
    N = 2000
    loop = False
    cfg.PARAMS['baseline_climate'] = 'ERA5dr'
    oggm.shop.ecmwf.process_ecmwf_data(gdir, dataset = 'ERA5dr')
    for mb_type in [ 'mb_monthly','mb_daily', 'mb_real_daily']:

        if mb_type !='mb_real_daily':
            cfg.PARAMS['baseline_climate'] = 'ERA5dr'
            oggm.shop.ecmwf.process_ecmwf_data(gdir, dataset = 'ERA5dr') 
        else:
            cfg.PARAMS['baseline_climate'] = 'ERA5_daily'
            process_era5_daily_data(gdir)
            
        hgts, widths = gdir.get_inversion_flowline_hw()
        mbdf = gdir.get_ref_mb_data()
        pf_opt = scipy.optimize.brentq(optimize_std_quot_brentq,
                                       0.01, 20, # min/max pf
                          args = (mb_type, grad_type, gdir, N, loop), xtol = 0.01)
        
        DDF_opt_pf = scipy.optimize.brentq(minimize_bias,1,10000,
                                        args=(mb_type, grad_type,
                                              gdir,N,pf_opt,
                                              loop, False), disp=True,
                                        xtol=0.1) 
        gd_mb = mb_modules(gdir, DDF_opt_pf, prcp_fac = pf_opt,   mb_type=mb_type,
                                           grad_type=grad_type)
        mb_specific = gd_mb.get_specific_mb(heights = hgts, widths = widths,
                                            year = mbdf.index.values)
        
        RMSD, bias , rcor, quot_std = compute_stat(mb_specific=mb_specific,
                                                   mbdf=mbdf)
        

        # check if the bias is optimised
        assert bias.round() == 0
        
        # check if the std_quotient is optimised
        assert quot_std.round(1) == 1

    
# %%
def test_mb_modules_monthly():
    
    cfg.initialize()

    mu_opt = mu_star_opt_cte['mb_monthly']
    cfg.PARAMS['baseline_climate'] = 'ERA5dr'
    oggm.shop.ecmwf.process_ecmwf_data(gdir, dataset = "ERA5")
    mb_mod = mb_modules(gdir, mu_opt,
                        mb_type = 'mb_monthly',
                        prcp_fac = 2.5,
                        t_solid = 0, t_liq =2, t_melt = 0, 
                        default_grad = -0.0065, bias = 0, 
                        grad_type = 'cte')
    
    hgts, widths = gdir.get_inversion_flowline_hw()
    mbdf = gdir.get_ref_mb_data()
    tot_mb = mb_mod.get_specific_mb(heights = hgts, widths = widths,
                                    year = mbdf.index.values)
   
    cfg.PARAMS['temp_default_gradient'] = -0.0065
    cfg.PARAMS['prcp_scaling_factor'] = 2.5
    cfg.PARAMS['temp_all_solid'] = 0
    cfg.PARAMS['temp_all_liq'] = 2
    cfg.PARAMS['temp_melt'] = 0

    # check if the default OGGM monthly mass balance with cte gradient
    # gives the same result as the new mb_modules with the options
    # mb_monthly and constant lapse rate gradient!
    mb_mod_default = massbalance.PastMassBalance(gdir, mu_star = mu_opt,
                                                 bias = 0,
                                                 check_calib_params=False)
    
    tot_mb_default = mb_mod_default.get_specific_mb(heights = hgts,
                                                    widths = widths, 
                                                    year = mbdf.index.values)

    
    np.testing.assert_allclose(tot_mb,tot_mb_default, rtol =1e-4)





# %%

# I use here the exact same names as in test_models from OGGM. 
# class TestInitPresentDayFlowline:
# but somehow the test for ref_mb_profile() is not equal


def test_present_time_glacier_massbalance(): # self
        cfg.initialize()
                
        # init_present_time_glacier(gdir) # if I use this, I have the wrong flowlines

        # check if area of  HUSS flowlines corresponds to the rgi area
        h, w = gdir.get_inversion_flowline_hw()
        fls = gdir.read_pickle('inversion_flowlines')
        np.testing.assert_allclose(gdir.rgi_area_m2,
                                   np.sum(w * gdir.grid.dx * fls[0].dx))

        # do this for all model types
        for climate in [ 'ERA5dr','ERA5_daily']: # ONLY TEST it for ERA5dr or ERA5_daily!!!
            for mb_type in ['mb_monthly','mb_daily','mb_real_daily']:
                for grad_type in ['cte', 'var_an_cycle']:
                    
                    if grad_type =='var_an_cycle':
                        fail_err_4 = (mb_type =='mb_monthly') and (climate == 'CRU')
                        mu_star_opt = mu_star_opt_var
                    else:
                        fail_err_4 = False
                        mu_star_opt = mu_star_opt_cte
                    if climate =='ERA5dr':
                        cfg.PARAMS['baseline_climate'] = 'ERA5dr'
                        oggm.shop.ecmwf.process_ecmwf_data(gdir, dataset = "ERA5dr")
                    elif climate == 'ERA5_daily':
                        cfg.PARAMS['baseline_climate'] = 'ERA5_daily'
                        process_era5_daily_data(gdir)
                    else:
                        tasks.process_climate_data(gdir)
                        pass
                    fail_err_1 = (mb_type =='mb_daily') and (climate !='ERA5dr')
                    fail_err_2 = (mb_type =='mb_monthly') and (climate =='ERA5_daily')
                    fail_err_3 = (mb_type =='mb_real_daily') and (climate !='ERA5_daily')
    
                    
                    if fail_err_1 or fail_err_2 or fail_err_3 or fail_err_4:
                        with pytest.raises(InvalidParamsError):
                            mb_mod = mb_modules(gdir, mu_star_opt[mb_type],
                                                mb_type = mb_type,
                                        prcp_fac = pf,
                                        t_solid = 0, t_liq =2, t_melt = 0, 
                                        default_grad = -0.0065, bias = 0, 
                                        grad_type = grad_type)
                    else:
                        # this is just a test for reproducibility!
                        mb_mod = mb_modules(gdir, mu_star_opt[mb_type],
                                            mb_type = mb_type,
                        prcp_fac = pf,
                        t_solid = 0, t_liq =2, t_melt = 0, 
                        default_grad = -0.0065, bias = 0, 
                        grad_type = grad_type)
            
                        #fls = gdir.read_pickle('inversion_flowlines') 
                        # model_flowlines does not work
                        #glacier = FlowlineModel(fls)
                
                        mbdf = gdir.get_ref_mb_data()
                
                        #hgts = np.array([])
                        #widths = np.array([])
                        #for fl in glacier.fls:
                        #    hgts = np.concatenate((hgts, fl.surface_h))
                        #    widths = np.concatenate((widths, fl.widths_m))
                        hgts, widths = gdir.get_inversion_flowline_hw()
        
                        tot_mb = []
                        refmb = []
                        grads = hgts * 0
                        for yr, mb in mbdf.iterrows():
                             refmb.append(mb['ANNUAL_BALANCE'])
                             mbh = (mb_mod.get_annual_mb(hgts, yr) * SEC_IN_YEAR *
                                    cfg.PARAMS['ice_density'])
                             grads += mbh
                             tot_mb.append(np.average(mbh, weights=widths))
                        grads /= len(tot_mb)
    
                        #tot_mb = mb_mod.get_specific_mb(heights = hgts, 
                        #                                widths = widths,
                
                
                                
                        # check if calibrated total mass balance similar
                        # to observe mass balance time series                        
                        assert np.abs(utils.md(tot_mb, refmb)) < 50
                
                        # Gradient THIS GIVES an error!!!
                        # possibly because I use the HUSS flowlines ...
                        # or is it because I use another calibration?
                        #dfg = gdir.get_ref_mb_profile().mean()
                
                        # Take the altitudes below 3100 and fit a line
                        #dfg = dfg[dfg.index < 3100]
                        #pok = np.where(hgts < 3100)
                        #from scipy.stats import linregress
                        #slope_obs, _, _, _, _ = linregress(dfg.index,
                        #                                   dfg.values)
                        #slope_our, _, _, _, _ = linregress(hgts[pok],
                        #                                   grads[pok])
                        #np.testing.assert_allclose(slope_obs, slope_our,
                        #                           rtol=0.15)
                        # 0.15 does not work
    
                        
# %%

def test_monthly_glacier_massbalance(): 
    # TODO: problem with that, monthly and annual MB not exactly the same!!!
    # I think there is a problem with SEC_IN_MONTH/SEC_IN_YEAR ...
    
    # this could be optimised and included in the above tests
    cfg.initialize()

    # do this for all model types
    for climate in [ 'ERA5dr','ERA5_daily']: # ONLY TEST it for ERA5dr or ERA5_daily!!!
        for mb_type in ['mb_monthly','mb_daily','mb_real_daily']:
        #for mb_type in ['mb_daily']:
    
            for grad_type in ['cte', 'var_an_cycle']:
                
                if grad_type =='var_an_cycle':
                    fail_err_4 = (mb_type =='mb_monthly') and (climate == 'CRU')
                    mu_star_opt = mu_star_opt_var
                else:
                    fail_err_4 = False
                    mu_star_opt = mu_star_opt_cte
                if climate =='ERA5dr':
                    cfg.PARAMS['baseline_climate'] = 'ERA5dr'
                    oggm.shop.ecmwf.process_ecmwf_data(gdir, dataset = "ERA5dr")
                elif climate == 'ERA5_daily':
                    process_era5_daily_data(gdir)
                    cfg.PARAMS['baseline_climate'] = 'ERA5_daily'
                else:
                    tasks.process_climate_data(gdir)
                    pass
                # mb_type ='mb_daily'
                fail_err_1 = (mb_type =='mb_daily') and (climate !='ERA5dr')
                fail_err_2 = (mb_type =='mb_monthly') and (climate =='ERA5_daily')
                fail_err_3 = (mb_type =='mb_real_daily') and (climate !='ERA5_daily')
    
                
                if fail_err_1 or fail_err_2 or fail_err_3 or fail_err_4:
                    with pytest.raises(InvalidParamsError):
                        mb_mod = mb_modules(gdir, mu_star_opt[mb_type],
                                            mb_type = mb_type,
                                    prcp_fac = pf,
                                    t_solid = 0, t_liq =2, t_melt = 0, 
                                    default_grad = -0.0065, bias = 0, 
                                    grad_type = grad_type)
                else:
                    # but this is just a test for reproducibility!
                    mb_mod = mb_modules(gdir, mu_star_opt[mb_type],
                                        mb_type = mb_type,
                    prcp_fac = pf,
                    t_solid = 0, t_liq =2, t_melt = 0, 
                    default_grad = -0.0065, bias = 0, 
                    grad_type = grad_type)
        
                    #fls = gdir.read_pickle('inversion_flowlines') # model_flowlines does not work
                    #glacier = FlowlineModel(fls)
            
                    mbdf = gdir.get_ref_mb_data()
            
                    #hgts = np.array([])
                    #widths = np.array([])
                    #for fl in glacier.fls:
                    #    hgts = np.concatenate((hgts, fl.surface_h))
                    #    widths = np.concatenate((widths, fl.widths_m))
                    hgts, widths = gdir.get_inversion_flowline_hw()
                    
                    grads = hgts * 0
                    rho = 900 # ice density
    
                    
                    yrp = [1980,2018]
                    from calendar import monthrange
    
                    for i, yr in enumerate(np.arange(yrp[0], yrp[1]+1)):
    
    
                        my_mon_mb_on_h = 0.
                        dayofyear = 0
                        for m in np.arange(12):
                            yrm = utils.date_to_floatyear(yr, m + 1)
                            _, dayofmonth = monthrange(yr, m+1)
                            dayofyear +=dayofmonth
                            tmp = mb_mod.get_monthly_mb(hgts, yrm) * dayofmonth* SEC_IN_DAY *rho
                            my_mon_mb_on_h += tmp
                        my_an_mb_on_h = mb_mod.get_annual_mb(hgts, yr) *dayofyear*SEC_IN_DAY*rho
                        
                        # these large errors might come from the problematic of 
                        # different amount of days in a year ... usage of SEC_IN_MONTH/SEC_IN_YEAR
                        # ask Fabien on that
                        # print(yr, np.mean(my_an_mb_on_h-my_mon_mb_on_h))
                        np.testing.assert_allclose(np.mean(my_an_mb_on_h-my_mon_mb_on_h),
                                                   0, atol = 100)
                        
                        # I am not sure if I can do such a test, because 
                        # the climate.py
                        # file only works with the default oggm stuff ...
                        # ref_mb_on_h = p[:, i] - mu_star_opt[mb_type] * t[:, i]
    
    
# %%
def test_loop(): 
    # tests whether ERA5dr works better with or without the loop in mb_daily
    # tests that both option give same results and in case that default option 
    # (no loop) is 30% slower, it raises an error
    
    # this could be optimised and included in the above tests
    cfg.initialize()

    climate = 'ERA5dr' 
    mb_type = 'mb_daily'
    cfg.PARAMS['baseline_climate'] = 'ERA5dr'
    oggm.shop.ecmwf.process_ecmwf_data(gdir, dataset = "ERA5dr")

    for grad_type in ['cte', 'var_an_cycle']:
                
        if grad_type =='var_an_cycle':
            fail_err_4 = (mb_type =='mb_monthly') and (climate == 'CRU')
            mu_star_opt = mu_star_opt_var
        else:
            fail_err_4 = False
            mu_star_opt = mu_star_opt_cte

        
        if fail_err_4:
            with pytest.raises(InvalidParamsError):
                mb_mod = mb_modules(gdir, mu_star_opt[mb_type],
                                    mb_type = mb_type,
                            prcp_fac = pf,
                            t_solid = 0, t_liq =2, t_melt = 0, 
                            default_grad = -0.0065, bias = 0, 
                            grad_type = grad_type)
        else:
            
            mbdf = gdir.get_ref_mb_data()
            hgts, widths = gdir.get_inversion_flowline_hw()
            
            
            
            ex_t = time.time()
            for t in np.arange(10):
                mb_mod_noloop = mb_modules(gdir, mu_star_opt[mb_type],
                                    mb_type = mb_type,
                prcp_fac = pf, loop = False, 
                t_solid = 0, t_liq =2, t_melt = 0, 
                default_grad = -0.0065, bias = 0, 
                grad_type = grad_type)
                tot_mb_noloop = mb_mod_noloop.get_specific_mb(heights = hgts, 
                 widths = widths,
                 year = mbdf.index.values)  
            ex_noloop = time.time() - ex_t
            
            ex_t = time.time()
            for t in np.arange(10):
                mb_mod_loop = mb_modules(gdir, mu_star_opt[mb_type],
                                    mb_type = mb_type,
                prcp_fac = pf, loop = True, 
                t_solid = 0, t_liq =2, t_melt = 0, 
                default_grad = -0.0065, bias = 0, 
                grad_type = grad_type)
                
                tot_mb_loop = mb_mod_loop.get_specific_mb(heights = hgts, 
                                                    widths = widths,
                                                    year =mbdf.index.values)  
            ex_loop = time.time() - ex_t
            
            


            #fls = gdir.read_pickle('inversion_flowlines') # model_flowlines does not work
            #glacier = FlowlineModel(fls)
    
            # both should give the same results!!!
            np.testing.assert_allclose(tot_mb_loop, tot_mb_noloop,
                                       atol = 1e-2)
            
            # if the loop would be at least 30% faster than not using the loop
            # raise an error, 
            assert (ex_loop-ex_noloop)/ex_noloop > -0.3
            # but in the moment loop is sometimes faster, why? !!!
            # actually it depends which one I test first, the one that runs first
            # is actually fast, so when running noloop first 
            # it is around 5% faster
                        
                    
                
                        
# %%

def test_N(): 
    # tests whether modelled mb_daily massbalances of different values of N
    # is similar to observed mass balances
    
    # this could be optimised and included in the above tests

    cfg.initialize()

    climate = 'ERA5dr' 
    mb_type = 'mb_daily'
    cfg.PARAMS['baseline_climate'] = 'ERA5dr'
    oggm.shop.ecmwf.process_ecmwf_data(gdir, dataset = "ERA5dr")

    for grad_type in ['cte', 'var_an_cycle']:
                
        if grad_type =='var_an_cycle':
            fail_err_4 = (mb_type =='mb_monthly') and (climate == 'CRU')
            mu_star_opt = mu_star_opt_var
        else:
            fail_err_4 = False
            mu_star_opt = mu_star_opt_cte

        
        if fail_err_4:
            with pytest.raises(InvalidParamsError):
                mb_mod = mb_modules(gdir, mu_star_opt[mb_type],
                                    mb_type = mb_type,
                            prcp_fac = pf,
                            t_solid = 0, t_liq =2, t_melt = 0, 
                            default_grad = -0.0065, bias = 0, 
                            grad_type = grad_type)
        else:
            
            mbdf = gdir.get_ref_mb_data()
            hgts, widths = gdir.get_inversion_flowline_hw()
            
            time_N = {}
            tot_mb_N = {}
            for N in [10000,5000,1000,500,100]:
                #ex_t = time.time()

                mb_mod = mb_modules(gdir, mu_star_opt[mb_type],
                                    mb_type = mb_type,
                prcp_fac = pf, N=N,
                t_solid = 0, t_liq =2, t_melt = 0, 
                default_grad = -0.0065, bias = 0, 
                grad_type = grad_type)
                
                tot_mb_N[N] = mb_mod.get_specific_mb(heights = hgts, 
                                                    widths = widths,
                                                    year =mbdf.index.values)  
                #ex_N = time.time() - ex_t
                #time_N[N] = ex_N
            
                
                assert np.abs(utils.md(tot_mb_N[N],
                                       mbdf['ANNUAL_BALANCE'])) < 10


    

   

            
# %% 
# something like that could also be included later on
# bbut only if I somehow get the data from the climate files ....
# or can I use these things with the ELA... without the climate files...
# in the moment oggm.core.climate works only with default OGGM mass balance

    # def test_mb_modules(self, hef_gdir):

    #     rho = cfg.PARAMS['ice_density']

    #     F = SEC_IN_YEAR * rho

    #     gdir = hef_gdir
    #     init_present_time_glacier(gdir)

    #     df = gdir.read_json('local_mustar')
    #     mu_star = df['mu_star_glacierwide']
    #     bias = df['bias']

    #     # Climate period
    #     yrp = [1851, 2000]

    #     # Flowlines height
    #     h, w = gdir.get_inversion_flowline_hw()

    #     mb_mod = massbalance.PastMassBalance(gdir, bias=0)
    #     for i, yr in enumerate(np.arange(yrp[0], yrp[1]+1)):
    #         ref_mb_on_h = p[:, i] - mu_star * t[:, i]
    #         my_mb_on_h = mb_mod.get_annual_mb(h, yr) * F
    #         np.testing.assert_allclose(ref_mb_on_h, my_mb_on_h,
    #                                    atol=1e-2)
    #         ela_z = mb_mod.get_ela(year=yr)
    #         totest = mb_mod.get_annual_mb([ela_z], year=yr) * F
    #         assert_allclose(totest[0], 0, atol=1)

    #     mb_mod = massbalance.PastMassBalance(gdir)
    #     for i, yr in enumerate(np.arange(yrp[0], yrp[1]+1)):
    #         ref_mb_on_h = p[:, i] - mu_star * t[:, i]
    #         my_mb_on_h = mb_mod.get_annual_mb(h, yr) * F
    #         np.testing.assert_allclose(ref_mb_on_h, my_mb_on_h + bias,
    #                                    atol=1e-2)
    #         ela_z = mb_mod.get_ela(year=yr)
    #         totest = mb_mod.get_annual_mb([ela_z], year=yr) * F
    #         assert_allclose(totest[0], 0, atol=1)

    #     for i, yr in enumerate(np.arange(yrp[0], yrp[1]+1)):

    #         ref_mb_on_h = p[:, i] - mu_star * t[:, i]
    #         my_mb_on_h = ref_mb_on_h*0.
    #         for m in np.arange(12):
    #             yrm = utils.date_to_floatyear(yr, m + 1)
    #             tmp = mb_mod.get_monthly_mb(h, yrm) * SEC_IN_MONTH * rho
    #             my_mb_on_h += tmp

    #         np.testing.assert_allclose(ref_mb_on_h,
    #                                    my_mb_on_h + bias,
    #                                    atol=1e-2)

    #     # real data
    #     h, w = gdir.get_inversion_flowline_hw()
    #     mbdf = gdir.get_ref_mb_data()
    #     mbdf.loc[yr, 'MY_MB'] = np.NaN
    #     mb_mod = massbalance.PastMassBalance(gdir)
    #     for yr in mbdf.index.values:
    #         my_mb_on_h = mb_mod.get_annual_mb(h, yr) * SEC_IN_YEAR * rho
    #         mbdf.loc[yr, 'MY_MB'] = np.average(my_mb_on_h, weights=w)

    #     np.testing.assert_allclose(mbdf['ANNUAL_BALANCE'].mean(),
    #                                mbdf['MY_MB'].mean(),
    #                                atol=1e-2)
    #     mbdf['MY_ELA'] = mb_mod.get_ela(year=mbdf.index.values)
    #     assert mbdf[['MY_ELA', 'MY_MB']].corr().values[0, 1] < -0.9
    #     assert mbdf[['MY_ELA', 'ANNUAL_BALANCE']].corr().values[0, 1] < -0.7

    #     mb_mod = massbalance.PastMassBalance(gdir, bias=0)
    #     for yr in mbdf.index.values:
    #         my_mb_on_h = mb_mod.get_annual_mb(h, yr) * SEC_IN_YEAR * rho
    #         mbdf.loc[yr, 'MY_MB'] = np.average(my_mb_on_h, weights=w)

    #     np.testing.assert_allclose(mbdf['ANNUAL_BALANCE'].mean() + bias,
    #                                mbdf['MY_MB'].mean(),
    #                                atol=1e-2)

    #     mb_mod = massbalance.PastMassBalance(gdir)
    #     for yr in mbdf.index.values:
    #         my_mb_on_h = mb_mod.get_annual_mb(h, yr) * SEC_IN_YEAR * rho
    #         mbdf.loc[yr, 'MY_MB'] = np.average(my_mb_on_h, weights=w)
    #         mb_mod.temp_bias = 1
    #         my_mb_on_h = mb_mod.get_annual_mb(h, yr) * SEC_IN_YEAR * rho
    #         mbdf.loc[yr, 'BIASED_MB'] = np.average(my_mb_on_h, weights=w)
    #         mb_mod.temp_bias = 0

    #     np.testing.assert_allclose(mbdf['ANNUAL_BALANCE'].mean(),
    #                                mbdf['MY_MB'].mean(),
    #                                atol=1e-2)
    #     assert mbdf.ANNUAL_BALANCE.mean() > mbdf.BIASED_MB.mean()

    #     # Repeat
    #     mb_mod = massbalance.PastMassBalance(gdir, repeat=True,
    #                                          ys=1901, ye=1950)
    #     yrs = np.arange(100) + 1901
    #     mb = mb_mod.get_specific_mb(h, w, year=yrs)
    #     assert_allclose(mb[50], mb[-50])