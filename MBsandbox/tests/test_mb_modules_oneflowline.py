#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 18:34:35 2020

@author: lilianschuster
"""

# tests for mass balances:
# tests for methods of mb_modules_oneflowline.py
# and for help_func.py
import warnings
warnings.filterwarnings("once", category=DeprecationWarning)  # noqa: E402

import time
import numpy as np
from numpy.testing import assert_allclose
import pytest
import scipy
import xarray as xr
import pandas as pd
from calendar import monthrange
import os

# imports from OGGM
import oggm
from oggm.core import massbalance
from oggm import utils, workflow, tasks, cfg
from oggm.cfg import SEC_IN_DAY, SEC_IN_YEAR
from oggm.exceptions import InvalidParamsError, InvalidWorkflowError
from oggm.utils import date_to_floatyear

import MBsandbox
# imports from MBsandbox package modules
from MBsandbox.wip.help_func_geodetic import minimize_bias_geodetic
from MBsandbox.help_func import (compute_stat, minimize_bias,
                                 optimize_std_quot_brentq,
                                 melt_f_calib_geod_prep_inversion,
                                 minimize_winter_mb_brentq_geod_via_pf,
                                 calibrate_to_geodetic_bias_winter_mb,
                                 optimize_std_quot_brentq_via_temp_b_w_min_winter_geod_bias)

from MBsandbox.mbmod_daily_oneflowline import (process_era5_daily_data,
                                               TIModel, TIModel_Sfc_Type)
from MBsandbox.mbmod_daily_oneflowline import (process_w5e5_data,
                                               AvgClimateMassBalance_TIModel,
                                               ConstantMassBalance_TIModel,
                                               )

# optimal values for HEF of mu_star for cte lapse rates (for wgms direct MB)
mu_star_opt_cte = {'mb_monthly': 213.561413,
                   'mb_pseudo_daily': 181.383413,
                   'mb_real_daily': 180.419554}
# optimal values of mu_star when using variable lapse rates ('var_an_cycle')
mu_star_opt_var = {'mb_monthly': 195.322804,
                   'mb_pseudo_daily': 167.506525,
                   'mb_real_daily': 159.912743}
# random precipitation factor
pf = 2.5

class Test_diff_models:
    @pytest.mark.parametrize('mb_type', ['mb_monthly', 'mb_pseudo_daily'])
    def test_avgclimate_mb_model_timodel(self, gdir, mb_type):
        #todo: do this for real_daily!!!
        pd_geodetic = utils.get_geodetic_mb_dataframe()
        pd_geodetic = pd_geodetic.loc[pd_geodetic.period == '2000-01-01_2020-01-01']
        gdir_geodetic = pd_geodetic.loc[gdir.rgi_id]['dmdtda'] * 1000
        cfg.PARAMS['hydro_month_nh'] = 1
        pf = 1
        if mb_type == 'mb_monthly':
            melt_f_calib = 217.8
            #glen_a_fac = 5.958421001753519
        elif mb_type == 'mb_pseudo_daily':
            melt_f_calib = 159.6
        grad_type = 'cte'
        climate_type='W5E5'
        temporal_resol = 'monthly'
        y0 = 2009
        hs = 10
        workflow.execute_entity_task(process_w5e5_data, [gdir],
                                     temporal_resol=temporal_resol,
                                     climate_type=climate_type)
        rho = cfg.PARAMS['ice_density']
        h, w = gdir.get_inversion_flowline_hw()

        cmb_mod = ConstantMassBalance_TIModel(gdir, melt_f_calib, prcp_fac=pf,
                                              mb_type=mb_type,
                                              grad_type=grad_type,
                                              baseline_climate=climate_type,
                                              y0=y0, halfsize=hs)
        ombh = cmb_mod.get_annual_mb(h) * SEC_IN_YEAR * rho
        otmb = np.average(ombh, weights=w)
        np.testing.assert_allclose(gdir_geodetic, otmb, rtol=0.02)
        np.testing.assert_allclose(gdir_geodetic, otmb, atol=17)

        avg_mod = AvgClimateMassBalance_TIModel(gdir, melt_f_calib, prcp_fac=pf,
                                                mb_type= mb_type,
                                                grad_type=grad_type,
                                                baseline_climate=climate_type,
                                                y0=y0, halfsize=hs)
        ombh = avg_mod.get_annual_mb(h) * SEC_IN_YEAR * rho
        otmb = np.average(ombh, weights=w)
        # This is now wrong -> but not too far we hope...
        # np.testing.assert_allclose(gdir_geodetic, otmb, rtol=0.2)
        np.testing.assert_allclose(gdir_geodetic, otmb, atol=150)

        # Another simulation
        cmb_mod = ConstantMassBalance_TIModel(gdir, melt_f_calib, prcp_fac=pf,
                                              mb_type= mb_type,
                                              grad_type = grad_type,
                                              baseline_climate = climate_type,
                                              y0=1991, halfsize=hs)
        ombh = cmb_mod.get_annual_mb(h) * SEC_IN_YEAR * rho
        otmb = np.average(ombh, weights=w)

        avg_mod = AvgClimateMassBalance_TIModel(gdir, melt_f_calib, prcp_fac=pf,
                                                mb_type= mb_type,
                                                grad_type = grad_type,
                                                baseline_climate = climate_type,
                                                y0=1991, halfsize=hs)

        _ombh = avg_mod.get_annual_mb(h) * SEC_IN_YEAR * rho
        _otmb = np.average(_ombh, weights=w)
        # This is now wrong -> but not too far we hope...
        np.testing.assert_allclose(otmb, _otmb, atol=200)


class Test_geodetic_sfc_type:
    def test_sfc_type_linear_vs_neg_exp(self, gdir, mb_type='mb_monthly'):
        cfg.PARAMS['hydro_month_nh'] = 1
        # just choose any random melt_f
        melt_f_ratio = 0.5
        tau_e_fold_yr = 0.5  # yr (6 months, e-folding change of melt_f)
        melt_f_ice = 300
        pf = 2  # precipitation factor

        baseline_climate = 'W5E5'
        temporal_resol = 'monthly'
        process_w5e5_data(gdir, temporal_resol=temporal_resol,
                          climate_type=baseline_climate)

        # with normal spinup for 6 years
        mb_mod_monthly_0_5_m_neg_exp = TIModel_Sfc_Type(gdir, melt_f_ice, mb_type=mb_type,
                                                        melt_f_ratio_snow_to_ice=melt_f_ratio, prcp_fac=pf,
                                                        melt_f_update='monthly',
                                                        melt_f_change='neg_exp',
                                                        tau_e_fold_yr=tau_e_fold_yr,  # default
                                                        baseline_climate=baseline_climate)
        melt_f_snow = melt_f_ice * melt_f_ratio
        time = np.linspace(0, 6, len(mb_mod_monthly_0_5_m_neg_exp.buckets + ['ice']))  # in years
        melt_f_buckets = melt_f_ice + (melt_f_snow - melt_f_ice) * np.exp(- time / tau_e_fold_yr)  # s: in months

        assert_allclose(np.fromiter(mb_mod_monthly_0_5_m_neg_exp.melt_f_buckets.values(),
                                    dtype=float),
                        melt_f_buckets)

        # if melt_f_ratio_snow_to_ice is 1, the melt_f should be everywhere the same
        mb_mod_monthly_1_m_neg_exp_tau_0_5 = TIModel_Sfc_Type(gdir, melt_f_ice, mb_type=mb_type,
                                                              melt_f_ratio_snow_to_ice=1, prcp_fac=pf,
                                                              melt_f_update='monthly',
                                                              melt_f_change='neg_exp',
                                                              tau_e_fold_yr=tau_e_fold_yr,  # default
                                                              baseline_climate=baseline_climate)
        assert_allclose(np.fromiter(mb_mod_monthly_1_m_neg_exp_tau_0_5.melt_f_buckets.values(),
                                    dtype=float),
                        melt_f_ice)

        # if very large tau_e_fold_yr, melt_f should be everywhere the melt_f_snow
        mb_mod_monthly_0_5_m_neg_exp_tau_large = TIModel_Sfc_Type(gdir, melt_f_ice, mb_type=mb_type,
                                                                  melt_f_ratio_snow_to_ice=melt_f_ratio,
                                                                  prcp_fac=pf,
                                                                  melt_f_update='monthly',
                                                                  melt_f_change='neg_exp',
                                                                  tau_e_fold_yr=1e9,  # default
                                                                  baseline_climate=baseline_climate)
        assert_allclose(np.fromiter(mb_mod_monthly_0_5_m_neg_exp_tau_large.melt_f_buckets.values(),
                                    dtype=float),
                        melt_f_ice*melt_f_ratio)

        # tau_e_fold_yr should be above zero ...
        with pytest.raises(AssertionError):
            TIModel_Sfc_Type(gdir, melt_f_ice, mb_type=mb_type, melt_f_ratio_snow_to_ice=melt_f_ratio,
                             prcp_fac=pf, melt_f_update='monthly', melt_f_change='neg_exp',
                             tau_e_fold_yr=0,  # default
                             baseline_climate=baseline_climate)

    def test_sfc_type_update(self, gdir, mb_type='mb_monthly'):
        cfg.PARAMS['hydro_month_nh'] = 1
        # just choose any random melt_f
        melt_f = 300
        pf = 2  # precipitation factor

        baseline_climate = 'W5E5'
        temporal_resol = 'monthly'
        process_w5e5_data(gdir, temporal_resol=temporal_resol,
                          climate_type=baseline_climate)
        h, w = gdir.get_inversion_flowline_hw()
        mb_mod_annual_0_5_a = TIModel_Sfc_Type(gdir, melt_f, mb_type=mb_type,
                                               melt_f_ratio_snow_to_ice=0.5, prcp_fac=pf,
                                               melt_f_update='annual',
                                               baseline_climate=baseline_climate
                                               )
        mb_mod_monthly_0_5_m = TIModel_Sfc_Type(gdir, melt_f, mb_type=mb_type,
                                                melt_f_ratio_snow_to_ice=0.5, prcp_fac=pf,
                                                melt_f_update='monthly',
                                                baseline_climate=baseline_climate)

        year = 2000
        for mb_mod, year1 in zip([mb_mod_annual_0_5_a, mb_mod_monthly_0_5_m],
                                 [2001, date_to_floatyear(2001, 1)]):
            mb_mod.get_annual_mb(h, year=year, spinup=True)
            # let's get manually the mb
            # as we want to test update the melt_f_update needs to be equal to climate_resol output!
            mb_mod.pd_bucket = mb_mod._add_delta_mb_vary_melt_f(h, year=year1,
                                                                climate_resol=mb_mod.melt_f_update
                                                                )
            bucket_bef_update = mb_mod.pd_bucket.copy()
            # let's do manually the update
            mb_mod._update()
            bucket_aft_update = mb_mod.pd_bucket.copy()
            # the first column should be zero after the update
            assert np.any(bucket_aft_update[mb_mod.first_snow_bucket] == 0)
            # the first column from the dataframe before the update should be the second one
            # of the next column
            for j, b in enumerate(mb_mod.buckets[:-1]):
                assert_allclose(bucket_bef_update[b],
                                bucket_aft_update[bucket_aft_update.columns[j + 1]])


    @pytest.mark.parametrize('mb_type', ['mb_monthly', 'mb_pseudo_daily',
                                         'mb_real_daily'])
    def test_geodetic_fixed_var_melt_f(self, gdir, mb_type):
        cfg.PARAMS['hydro_month_nh'] = 1
        # just choose any random melt_f
        melt_f = 200
        pf = 2  # precipitation factor
        df = ['RGI60-11.00897']

        baseline_climate = 'W5E5'
        if mb_type != 'mb_real_daily':
            temporal_resol = 'monthly'
        else:
            temporal_resol = 'daily'
        process_w5e5_data(gdir, temporal_resol=temporal_resol,
                          climate_type=baseline_climate)

        mb_mod_1 = TIModel_Sfc_Type(gdir, melt_f, mb_type=mb_type,
                                    melt_f_ratio_snow_to_ice=1, prcp_fac=pf,
                                    baseline_climate=baseline_climate)

        mb_mod_no_sfc_type = TIModel(gdir, melt_f, mb_type=mb_type,
                                     prcp_fac=pf,
                                     baseline_climate=baseline_climate)

        h, w = gdir.get_inversion_flowline_hw()
        year = 2000
        _, temp2dformelt_1, _, prcpsol_1 = mb_mod_1._get_2d_annual_climate(h, year)
        _, temp2dformelt_no_sfc_type, _, prcpsol_no_sfc_type = mb_mod_no_sfc_type._get_2d_annual_climate(h, year)

        assert_allclose(temp2dformelt_1, temp2dformelt_no_sfc_type)
        assert_allclose(prcpsol_1, prcpsol_no_sfc_type)

        # check if the pd_mb dataframe and bucket are empty
        assert np.shape(mb_mod_1.pd_mb_annual) == (len(mb_mod_1.fl.dis_on_line), 0)
        assert np.shape(mb_mod_1.pd_mb_monthly) == (len(mb_mod_1.fl.dis_on_line), 0)
        assert np.all(mb_mod_1.pd_bucket == 0)
        mb_annual_1 = mb_mod_1.get_annual_mb(h, year=2000, spinup=True)
        # check if spinup was done and if it worked
        assert np.shape(mb_mod_1.pd_mb_annual) == (len(mb_mod_1.fl.dis_on_line), 1+6)
        assert np.all(mb_mod_1.pd_mb_annual.columns.values == np.arange(1994, 2001))
        # this should not change as only get_annual_mb was done
        assert np.shape(mb_mod_1.pd_mb_monthly) == (len(mb_mod_1.fl.dis_on_line), 0)
        assert np.any(mb_mod_1.pd_bucket >= 0)

        mb_annual_no_sfc_type = mb_mod_no_sfc_type.get_annual_mb(h, year=2000)

        assert_allclose(mb_annual_1, mb_annual_no_sfc_type)

        # the next year should run
        mb_mod_1.get_annual_mb(h, year=2001, spinup=True)
        # but 2004 not, because 2002 and 2003 are missing
        with pytest.raises(InvalidWorkflowError):
            # mass balance of 5 years beforehand not fully computed so should raise an error
            mb_mod_1.get_annual_mb(h, year=2004, spinup=True, auto_spinup=False)

        # check if get_monthly_mb works
        m = 1
        floatyear = date_to_floatyear(year, m)
        mb_monthly_1 = mb_mod_1.get_monthly_mb(h, year=floatyear, spinup=True)
        # with spinup, want that the pd_mb_annual and bucket is reset, repeat the spinup
        # and then use this pd_bucket_state to get the monthly_mb
        # (so we don't need to have float years between 1994 and 1999 in pd_monthly_mb if melt_f_update is annual)
        assert np.shape(mb_mod_1.pd_mb_annual) == (len(mb_mod_1.fl.dis_on_line), 6)
        #assert np.shape(mb_mod_1.pd_mb_monthly) == (len(mb_mod_1.fl.dis_on_line), 72)

        assert np.all(mb_mod_1.pd_mb_annual.columns.values == np.arange(1994, 2000))
        assert np.all(mb_mod_1.pd_mb_monthly.columns == floatyear)
        assert np.any(mb_mod_1.pd_bucket >= 0)

        mb_monthly_no_sfc_type = mb_mod_no_sfc_type.get_monthly_mb(h, year=floatyear)
        assert_allclose(mb_monthly_1, mb_monthly_no_sfc_type)

        # the next month should work
        for m in np.arange(2, 13, 1):
            mb_mod_1.get_monthly_mb(h, year=date_to_floatyear(year, m),
                                    spinup=True)
        # the first month of the next year should work as well
        mb_mod_1.get_monthly_mb(h, year=date_to_floatyear(year+1, 1),
                                spinup=True)
        # do we get the same monthly_mb when we want to have mb_monthly_of January?

        # try it with another month of the next year that doesn't follow the month from before
        # this should raise an error!!!
        with pytest.raises(InvalidWorkflowError):
            mb_mod_1.get_monthly_mb(h, year=date_to_floatyear(year+1, 6), spinup=True,
                                    auto_spinup=False)

        # check if specific mass balance equal of TIModel equals TIModelSfcType if ratio=1:
        # use the years from geodetic data
        years = np.arange(2000, 2020)
        fls = gdir.read_pickle('inversion_flowlines')
        # it raises an error (as above Jan 2001 mb was computed, so we don't have the
        # right pd_buckets...(pd_bucket evolution is not saved)
        with pytest.raises(InvalidWorkflowError):
            # there is snow
            mb_mod_1.get_specific_mb(year=years, fls=fls)
        # so, we need to reset the buckets here
        mb_mod_1.reset_pd_mb_bucket()
        spec_1 = mb_mod_1.get_specific_mb(year=years, fls=fls)
        spec_no_sfc_type = mb_mod_no_sfc_type.get_specific_mb(year=years, fls=fls)
        # here is the actual check:
        assert_allclose(spec_1, spec_no_sfc_type)

        # next: check if optimizer works for both !
        pd_geodetic = utils.get_geodetic_mb_dataframe()
        pd_geodetic = pd_geodetic.loc[pd_geodetic.period == '2000-01-01_2020-01-01']
        mb_geodetic = pd_geodetic.loc[df].dmdtda.values * 1000

        melt_f_opt_1 = scipy.optimize.brentq(minimize_bias_geodetic, 1, 1000,
                                             xtol=0.01, args=(mb_mod_1, mb_geodetic,
                                                              h, w, pf, False,
                                                              np.arange(2000, 2020, 1),
                                                              False, True
                                                              ), disp=True)
        mb_mod_1.melt_f = melt_f_opt_1

        melt_f_opt_no_sfc_type = scipy.optimize.brentq(minimize_bias_geodetic, 1, 1000,
                                                       xtol=0.01,
                                                       args=(mb_mod_no_sfc_type,
                                                             mb_geodetic,
                                                             h, w, pf, False,
                                                             np.arange(2000, 2020, 1),
                                                             False, True), disp=True)
        mb_mod_no_sfc_type.melt_f = melt_f_opt_no_sfc_type
        # they should optimize to the same melt_f
        assert_allclose(melt_f_opt_1, melt_f_opt_no_sfc_type)

        # check reproducibility
        melt_f_check_1 = {'mb_monthly': 292.172224, 'mb_pseudo_daily': 214.232751,
                          'mb_real_daily': 218.836968}
        assert_allclose(melt_f_opt_1, melt_f_check_1[mb_type])

        # as the melt_f was set to another value,
        # check if the pd_mb_monthly, pd_mb_annual and pd_bucket is resetted
        assert np.shape(mb_mod_1.pd_mb_annual) == (len(mb_mod_1.fl.dis_on_line), 0)
        assert np.shape(mb_mod_1.pd_mb_monthly) == (len(mb_mod_1.fl.dis_on_line), 0)
        assert np.all(mb_mod_1.pd_bucket == 0)

        spec_1 = mb_mod_1.get_specific_mb(year=years, fls=fls)
        # now check if the pd_mb buckets are filled up
        # if melt_f_update == annual (here), have len(years) + spinup_yrs s as amount of columns
        assert np.shape(mb_mod_1.pd_mb_annual) == (len(mb_mod_1.fl.dis_on_line),
                                                   len(years) + mb_mod_1.spinup_yrs)
        # no monthly update done (because only get_annual_mb here)
        assert np.shape(mb_mod_1.pd_mb_monthly) == (len(mb_mod_1.fl.dis_on_line), 0)
        assert np.any(mb_mod_1.pd_bucket >= 0)
        spec_no_sfc_type = mb_mod_no_sfc_type.get_specific_mb(year=years, fls=fls)
        # actual check if TIModel and TIModel_Sfc_Type with ratio=1 and monthly update are equal
        assert_allclose(spec_1, spec_no_sfc_type)

        # now include the surface type distinction and choose the
        # melt factor of snow to be 0.5* smaller than the melt factor of ice
        mb_mod_0_5 = TIModel_Sfc_Type(gdir, melt_f, mb_type=mb_type,
                                      melt_f_ratio_snow_to_ice=0.5, prcp_fac=pf,
                                      baseline_climate=baseline_climate
                                      )
        melt_f_opt_0_5 = scipy.optimize.brentq(minimize_bias_geodetic, 1, 1000,
                                               xtol=0.01, args=(mb_mod_0_5,
                                                                mb_geodetic,
                                                                h, w, pf, False,
                                                                np.arange(2000, 2020, 1),
                                                                False, True),
                                               disp=True)
        # check reproducibility
        melt_f_check_0_5 = {'mb_monthly': 403.897452,
                            'mb_pseudo_daily': 306.490551,
                            'mb_real_daily': 317.096054}
        assert_allclose(melt_f_opt_0_5, melt_f_check_0_5[mb_type])

        # the melt factor of only ice with surface type distinction should be
        # higher than the "mixed" melt factor of ice (with snow) (as the snow melt factor
        # is lower than the ice melt factor, as defined)
        assert melt_f_opt_0_5 > melt_f_opt_1

        mb_mod_0_5.melt_f = melt_f_opt_0_5
        spec_0_5 = mb_mod_0_5.get_specific_mb(year=years, fls=fls)

        # check if the optimised specific mass balance using a ratio of 0.5 is similar as
        # the optimised spec. mb of no_sfc_type (i.e, ratio of 1)
        # did the optimisation work?
        assert_allclose(spec_0_5.mean(), mb_geodetic, rtol=1e-4)
        assert_allclose(spec_0_5.mean(), spec_1.mean(), rtol=1e-4)
        # the standard deviation can be quite different,
        assert_allclose(spec_0_5.std(), spec_1.std(), rtol=0.3)

        # check if mass balance gradient with 0.5 ratio is higher than with no
        # surface type distinction
        mb_annual_0_5 = mb_mod_0_5.get_annual_mb(h, year=2000, spinup=True)

        mb_gradient_0_5, _, _, _, _ = scipy.stats.linregress(h[mb_annual_0_5 < 0],
                                                         y=mb_annual_0_5[mb_annual_0_5 < 0])

        mb_gradient_1, _, _, _, _ = scipy.stats.linregress(h[mb_annual_1 < 0],
                                                             y=mb_annual_1[mb_annual_1 < 0])

        assert mb_gradient_0_5 > mb_gradient_1

        ## check if the same mb comes out from get_mb as saved in pd_mb_annual
        # -> is it the same as in pd_mb_annual???
        year_test = 2008
        mb_annual_pd_2008 = mb_mod_0_5.pd_mb_annual[year_test].values.copy()
        # now get the mb:
        mb_annual_get_2008 = mb_mod_0_5.get_annual_mb(h, year=year_test)
        # test if similar
        assert_allclose(mb_annual_pd_2008, mb_annual_get_2008)

        # let's test the auto-spinup
        mb_mod_0_5.reset_pd_mb_bucket()
        mb_mod_0_5.get_annual_mb(h, year = year_test)
        # we have 6 spinup years (default)
        assert np.all(mb_mod_0_5.pd_mb_annual.columns.values == np.arange(year_test - 6,
                                                                          year_test + 1))
        mb_mod_0_5.get_monthly_mb(h, year = date_to_floatyear(year_test, 5))
        assert np.all(mb_mod_0_5.pd_mb_annual.columns.values == np.arange(year_test - 6,
                                                                   year_test))
        assert np.all(mb_mod_0_5.pd_mb_monthly.columns.values == np.array([date_to_floatyear(year_test, m)
                                                           for m in np.arange(1,6,1)]))

    # this test is here to show that it fails
    @pytest.mark.skip(reason='this test is expected to fail')
    @pytest.mark.parametrize('mb_type', [ 'mb_monthly']) # ,'mb_pseudo_daily', 'mb_real_daily'])
    def test_monthly_mb_for_annual_mb(self, gdir, mb_type):

        # wip
        # this will actually never work unless we change TIModel_Sfc_Type with melt_f_update = 'annual'
        # if we want it to work, we would need to loop over the months inside of get_annual_mb()
        # for m in months:
        #   get_montly_mb(---)
        #

        cfg.PARAMS['hydro_month_nh'] = 1
        # just choose any random melt_f here
        melt_f = 200
        pf = 2  # precipitation factor

        baseline_climate = 'W5E5'
        if mb_type != 'mb_real_daily':
            temporal_resol = 'monthly'
        else:
            temporal_resol = 'daily'
        process_w5e5_data(gdir, temporal_resol=temporal_resol,
                          climate_type=baseline_climate,
                          )

        # those two should be similar but not equal!
        mb_mod_0_5_a = TIModel_Sfc_Type(gdir, melt_f, mb_type=mb_type,
                                  melt_f_ratio_snow_to_ice=0.5, prcp_fac=pf,
                                    melt_f_update='annual',
                                  baseline_climate=baseline_climate
                                    )
        mb_mod_0_5_m = TIModel_Sfc_Type(gdir, melt_f, mb_type=mb_type,
                                          melt_f_ratio_snow_to_ice=0.5, prcp_fac=pf,
                                          melt_f_update='monthly',
                                          baseline_climate=baseline_climate)

        h, w = gdir.get_inversion_flowline_hw()

        annual_a_via_monthly = []
        annual_m_via_monthly = []
        annual_a = []
        annual_m = []
        for y in np.arange(2000, 2003, 1):
            #annual_a = mb_mod_0_5_a.get_annual_mb(h, year=y)
            #annual_a = mb_mod_0_5_a.get_annual_mb(h, year=y)
            # after that it needs to reset again because the year just before was already computed
            for m in np.arange(1,13,1):
                floatyr = utils.date_to_floatyear(y, m)
                mb_mod_0_5_a.get_monthly_mb(h, year=floatyr)
                mb_mod_0_5_m.get_monthly_mb(h, year=floatyr)

            annual_a_via_monthly.append(mb_mod_0_5_a.get_annual_mb(h, year=y))
            annual_m_via_monthly.append(mb_mod_0_5_m.get_annual_mb(h, year=y))

        bucket_a_via_monthly_2003 = mb_mod_0_5_a.pd_bucket

        # should have spinup years and year 2000-2002
        assert_allclose(mb_mod_0_5_m.pd_mb_annual.columns.values, np.arange(2000 - 6, 2003, 1))
        assert_allclose(mb_mod_0_5_a.pd_mb_annual.columns.values, np.arange(2000 - 6, 2003, 1))
        # if annual update: should only have monthly mb starting with 2000 (not for the spinup years!)
        assert len(mb_mod_0_5_a.pd_mb_monthly.columns.values) == 12 * 3
        # if monthly update: should also have monthly mb for spinup years
        assert len(mb_mod_0_5_m.pd_mb_monthly.columns.values) == 12 * (3 + 6)

        # let's now compute the MB directly via get_annual_mb -> we should get out the same estimates at the end:
        mb_mod_0_5_m.reset_pd_mb_bucket()
        mb_mod_0_5_a.reset_pd_mb_bucket()
        for y in np.arange(2000, 2003, 1):
            annual_a.append(mb_mod_0_5_a.get_annual_mb(h, year=y))
            annual_m.append(mb_mod_0_5_m.get_annual_mb(h, year=y))
        # bucket_a_2003 = mb_mod_0_5_a.pd_bucket
        # monthly works as it should
        assert_allclose(annual_m_via_monthly, annual_m)

        # but annual has a problem!
        # this is failing, which shows that there are definitely differences
        assert_allclose(annual_a_via_monthly, annual_a, rtol=1e-2)


    #@pytest.mark.skip(reason="slow, not important for sarah")
    # in addition it only works if we allow large differences
    @pytest.mark.slow
    @pytest.mark.parametrize('mb_type', [ 'mb_monthly','mb_pseudo_daily',
                                         'mb_real_daily'
                                          ])
    def test_annual_vs_monthly_melt_f_update(self, gdir, mb_type):

        # wip
        # monthly_melt_f with melt_f_ratio=1 gives same results as monthly_melt_f with TIModel
        # needs to check whether annual and monthly melt_f give similar results
        # this works in case of monthly melt_f_update
        # but not in case of annual melt_f_update (it works here only because we allowed strong differences)

        cfg.PARAMS['hydro_month_nh'] = 1
        # just choose any random melt_f
        melt_f = 200
        pf = 2  # precipitation factor
        df = ['RGI60-11.00897']

        baseline_climate = 'W5E5'
        if mb_type != 'mb_real_daily':
            temporal_resol = 'monthly'
        else:
            temporal_resol = 'daily'
        process_w5e5_data(gdir, temporal_resol=temporal_resol,
                          climate_type=baseline_climate,
                          )

        # those two should be equal!!!
        mb_mod_monthly_1_m = TIModel_Sfc_Type(gdir, melt_f, mb_type=mb_type,
                                          melt_f_ratio_snow_to_ice=1, prcp_fac=pf,
                                          melt_f_update='monthly',
                                          baseline_climate=baseline_climate)
        mb_mod_no_sfc_type = TIModel(gdir, melt_f, mb_type=mb_type,
                                     prcp_fac=pf, baseline_climate=baseline_climate)

        # those two should be similar but not equal!
        mb_mod_annual_0_5_a = TIModel_Sfc_Type(gdir, melt_f, mb_type=mb_type,
                                  melt_f_ratio_snow_to_ice=0.5, prcp_fac=pf,
                                    melt_f_update='annual',
                                  baseline_climate=baseline_climate
                                    )
        mb_mod_monthly_0_5_m = TIModel_Sfc_Type(gdir, melt_f, mb_type=mb_type,
                                          melt_f_ratio_snow_to_ice=0.5, prcp_fac=pf,
                                          melt_f_update='monthly',
                                          baseline_climate=baseline_climate)

        h, w = gdir.get_inversion_flowline_hw()
        # monthly climate, year corresponds here to the hydro float year
        # (so it gives the values of that month)
        year = 2000
        month = 6
        floatyear = date_to_floatyear(year, month)
        # check if climate is similar (look here at monthly climate!)

        _, temp2dformelt_1_m, _, prcpsol_1_m = mb_mod_monthly_1_m.get_monthly_climate(h, floatyear)
        _, temp2dformelt_no_sfc_type, _, prcpsol_no_sfc_type = mb_mod_no_sfc_type.get_monthly_climate(h, floatyear)

        assert_allclose(temp2dformelt_1_m, temp2dformelt_no_sfc_type)
        assert_allclose(prcpsol_1_m, prcpsol_no_sfc_type)

        _, temp2dformelt_0_5_a, _, prcpsol_0_5_a = mb_mod_annual_0_5_a.get_monthly_climate(h, floatyear)
        _, temp2dformelt_0_5_m, _, prcpsol_0_5_m = mb_mod_monthly_0_5_m.get_monthly_climate(h, floatyear)

        assert_allclose(temp2dformelt_0_5_m, temp2dformelt_0_5_a)
        assert_allclose(prcpsol_0_5_m, prcpsol_0_5_a)

        # mass balance comparison
        # (monthly with ratio 1 should be equal to no sfc update!!!)
        # first the comparison of one specific month

        # need to start in January
        year = 2000
        for m in np.arange(1, 13, 1):
            floatyear = date_to_floatyear(year, m)

            mb_monthly_1_m = mb_mod_monthly_1_m.get_monthly_mb(h, year=floatyear, spinup=True)
            mb_monthly_no_sfc_type = mb_mod_no_sfc_type.get_monthly_mb(h, year=floatyear)

            assert_allclose(mb_monthly_1_m, mb_monthly_no_sfc_type)

            # for annual melt_f_update, this is a bit tricky because the _update should only be done after a full year
            # m, y = floatyear_to_date(..), if m==12: do the update

            mb_monthly_0_5_m = mb_mod_monthly_0_5_m.get_monthly_mb(h, year=floatyear, spinup=True)
            # this has to be implemented first ...
            mb_monthly_0_5_a = mb_mod_annual_0_5_a.get_monthly_mb(h, year=floatyear, spinup=True)
            # this should not be more different than 50%!!! (as melt_f of snow is 0.5*melt_f of ice!
            # the smaller the ratio, the larger are the differences
            # only differences in melt months
            if m in [4, 5, 6, 7, 8, 9]:
                assert_allclose(mb_monthly_0_5_m, mb_monthly_0_5_a, rtol=0.5)  # ?
            else:
                assert_allclose(mb_monthly_0_5_m, mb_monthly_0_5_a, rtol=0.01)  # ?

            assert np.all(mb_mod_monthly_0_5_m.pd_mb_monthly.loc[:, floatyear] == mb_monthly_0_5_m)
            assert np.all(mb_mod_annual_0_5_a.pd_mb_monthly.loc[:, floatyear] == mb_monthly_0_5_a)

            if m == 1:
                annual_a = mb_monthly_0_5_a #.values
                annual_m = mb_monthly_0_5_m #.values
            else:
                annual_a += mb_monthly_0_5_a #.values
                annual_m += mb_monthly_0_5_m #.values

            # for m >= 5: assert_allclose(annual_m, annual_a, rtol = 0.5) does not work anymore
            # that means quite some part of the very fresh snow is melted away

        # the sum over all monthly is also not equal ??? but it should !!!
        pd_sum_a = mb_mod_annual_0_5_a.pd_mb_monthly[mb_mod_annual_0_5_a.pd_mb_monthly.columns[-12:]].sum(axis=1)/12
        pd_sum_m = mb_mod_monthly_0_5_m.pd_mb_monthly[mb_mod_monthly_0_5_m.pd_mb_monthly.columns[-12:]].sum(axis=1)/12
        assert_allclose(pd_sum_a, pd_sum_m, atol=1e-8)
        assert_allclose(pd_sum_a, pd_sum_m, rtol=100) #
        #
        assert_allclose(annual_m / 12, annual_a / 12, atol=1e-8)
        assert_allclose(annual_m/12, annual_a/12, rtol=100)
        # then for the entire year!
        # just as a check:

        # test if I can get the mass balance without doing something before
        # and also check if most buckets are filled (from the spinup)
        mb_mod_monthly_0_5_m.reset_pd_mb_bucket()
        _, bucket = mb_mod_monthly_0_5_m.get_monthly_mb(h, year=date_to_floatyear(2002, 2),
                                                        spinup=True,
                                                        bucket_output=True)
        # there should be less than 5  entries
        len(bucket.columns[bucket.iloc[0] == 0]) < 5
        # all spinup years (default is 6) should be there
        assert np.all(mb_mod_monthly_0_5_m.pd_mb_annual.columns.values == np.arange(2002 - 6, 2002, 1))
        #assert np.all(mb_mod_monthly_0_5_m.pd_mb_annual.columns.values==np.arange(2002-6, 2002+1, 1))

        mb_mod_monthly_1_m.reset_pd_mb_bucket()
        mb_mod_monthly_0_5_m.reset_pd_mb_bucket()
        mb_mod_annual_0_5_a.reset_pd_mb_bucket()
        for year in np.arange(2000, 2002, 1):
            mb_annual_1_m = mb_mod_monthly_1_m.get_annual_mb(h, year=year, spinup=True)
            mb_annual_no_sfc_type = mb_mod_no_sfc_type.get_annual_mb(h, year=year)

            assert_allclose(mb_annual_1_m, mb_annual_no_sfc_type)

            # mass balance won't be the same but should at least be similar
            # need to have a spinup, otherwise difficult to compare
            mb_annual_monthly_0_5_m = mb_mod_monthly_0_5_m.get_annual_mb(h, year=year, spinup=True)
            mb_annual_0_5_a = mb_mod_annual_0_5_a.get_annual_mb(h, year=year, spinup=True)

            #TODO: at the moment this is very different, should it be more similar???
            # normally yes, mass balance should not be more different then 2*the annual value ???
            #test_m = mb_mod_monthly_0_5_m.pd_mb_monthly[mb_mod_monthly_0_5_m.pd_mb_monthly.columns[-12:]].mean(axis=1)
            #test_a = mb_mod_annual_0_5_a.pd_mb_annual[2000].values
            #mb_mod_annual_0_5_a.pd_mb_annual
            #mb_mod_annual_0_5_a.pd_mb_monthly[mb_mod_annual_0_5_a.pd_mb_monthly.columns[-12:]].mean()

            #assert_allclose(mb_annual_monthly_0_5_m, mb_annual_0_5_a, rtol=100)
            assert_allclose(np.average(mb_annual_monthly_0_5_m, weights=w),
                            np.average(mb_annual_0_5_a, weights=w),
                            atol=5e-9) # for one year ?


        # check if specific mass balance equal?
        # use the years from geodetic data
        years = np.arange(2000, 2020)
        fls = gdir.read_pickle('inversion_flowlines')

        # this should be almost equal
        # (monthly update with ratio of 1 should be equal to no sfc update)
        spec_1_m = mb_mod_monthly_1_m.get_specific_mb(year=years, fls=fls) #, spinup=True)
        spec_no_sfc_type = mb_mod_no_sfc_type.get_specific_mb(year=years, fls=fls)

        assert_allclose(spec_1_m, spec_no_sfc_type)

        # monthly update versus annual update should be similar but not equal
        spec_0_5_m = mb_mod_monthly_0_5_m.get_specific_mb(year=years, fls=fls) #, spinup=True)
        spec_0_5_a = mb_mod_annual_0_5_a.get_specific_mb(year=years, fls=fls) #, spinup=True)
        # specific mass balance is also quite different, does this make sense???
        # if monthly, specific mb rather smaller but not always!!!

        # HERE ARE LARGE DIFFERENCES THAT ARE NOT CORRECT
        if mb_type == 'mb_real_daily':
            assert_allclose(spec_0_5_m.mean(), spec_0_5_a.mean(), rtol=1.2)  # for pseudo_daily 0.45
            assert_allclose(spec_0_5_m, spec_0_5_a, atol=200) # 1.6 for mb_montly
        else:
            assert_allclose(spec_0_5_m.mean(), spec_0_5_a.mean(), rtol=0.5) # for pseudo_daily 0.45
            assert_allclose(spec_0_5_m, spec_0_5_a, rtol=3.5) # 1.6 for mb_montly

        # Check if optimizer works
        # get geodetic data
        pd_geodetic = utils.get_geodetic_mb_dataframe()
        pd_geodetic = pd_geodetic.loc[pd_geodetic.period == '2000-01-01_2020-01-01']
        mb_geodetic = pd_geodetic.loc[df].dmdtda.values * 1000

        # find and set optimal melt_factor:
        melt_f_opt_1_m = scipy.optimize.brentq(minimize_bias_geodetic, 1, 1000,
                                             xtol=0.01, args=(mb_mod_monthly_1_m,
                                                              mb_geodetic,
                                                              h, w, pf, False,
                                                              years,
                                                              False, True  # do spinup before
                                                              ), disp=True)
        mb_mod_monthly_1_m.melt_f = melt_f_opt_1_m

        melt_f_opt_no_sfc_type = scipy.optimize.brentq(minimize_bias_geodetic, 1, 1000,
                                              xtol=0.01, args=(mb_mod_no_sfc_type,
                                                               mb_geodetic,
                                                               h, w, pf, False,
                                                               years,
                                                               False, True  # do spinup before
                                                               ), disp=True)
        mb_mod_no_sfc_type.melt_f = melt_f_opt_no_sfc_type
        # they should optimize to the same melt_f
        assert_allclose(melt_f_opt_1_m, melt_f_opt_no_sfc_type)

        # check reproducibility
        #assert_allclose(melt_f_opt_1_m, 190.5106406914272)
        melt_f_check_1 = {'mb_monthly': 292.17222401789707,
                            'mb_pseudo_daily': 214.2327512287635,
                            'mb_real_daily': 218.83696832680852}
        assert_allclose(melt_f_opt_1_m, melt_f_check_1[mb_type])

        # same specific mass balance?
        spec_1_m = mb_mod_monthly_1_m.get_specific_mb(year=years, fls=fls) #, spinup=True)
        spec_no_sfc_type = mb_mod_no_sfc_type.get_specific_mb(year=years, fls=fls) #, spinup=True)

        assert_allclose(spec_1_m, spec_no_sfc_type)

        #TODO: need to check if spinup is inside or not
        # now the same but with a 0.5 ratio, comparing annual vs monthly update
        # find and set optimal melt_factor:
        melt_f_opt_0_5_m = scipy.optimize.brentq(minimize_bias_geodetic, 1, 1000,
                                             xtol=0.01, args=(mb_mod_monthly_0_5_m,
                                                              mb_geodetic,
                                                              h, w, pf, False,
                                                              years,
                                                              False, True # do spinup before
                                                              ), disp=True)
        mb_mod_monthly_0_5_m.melt_f = melt_f_opt_0_5_m

        melt_f_opt_0_5_a = scipy.optimize.brentq(minimize_bias_geodetic, 1, 1000,
                                              xtol=0.01, args=(mb_mod_annual_0_5_a,
                                                               mb_geodetic,
                                                               h, w, pf, False,
                                                               years,
                                                               False, True  # do spinup before
                                                               ), disp=True)
        mb_mod_annual_0_5_a.melt_f = melt_f_opt_0_5_a
        # they should optimize not to the same melt_f, but similar
        assert_allclose(melt_f_opt_0_5_m, melt_f_opt_0_5_a, rtol=0.2)
        # todo: is this generally true that the
        #melt factor with monthly update is lower than with annual update
        assert melt_f_opt_0_5_m < melt_f_opt_0_5_a

        # also check that they are not equal!!!,
        # TODO: which one should be typically smaller?

        # similar specific mass balance (as calibrated to same obs.)
        # so at least mean should be same
        spec_0_5_m = mb_mod_monthly_0_5_m.get_specific_mb(year=years, fls=fls) #, spinup=True)
        spec_0_5_a = mb_mod_annual_0_5_a.get_specific_mb(year=years, fls=fls) #, spinup=True)

        # did the optimisation work?
        assert_allclose(spec_0_5_m.mean(), mb_geodetic, rtol= 1e-4)
        # at least mean should be the same ...
        assert_allclose(spec_0_5_m.mean(), spec_0_5_a.mean(), rtol=1e-4)
        # the standard deviation can be different,
        assert_allclose(spec_0_5_m.std(), spec_0_5_a.std(), rtol=0.05)

        # check reproducibility
        # make a dictionary here...
        #assert_allclose(melt_f_opt_0_5_m, 290.2804601389265)
        melt_f_check_0_5_m = {'mb_monthly': 390.2045799130138,  # 403.8974519041801 annual update
                          'mb_pseudo_daily': 298.0761325686244,  # 306.448985,
                          'mb_real_daily': 308.866594596061  # 317.047318
                              }
        assert_allclose(melt_f_opt_0_5_m, melt_f_check_0_5_m[mb_type])

        # is the MB gradient with 0.5 ratio and monthly update higher than with no
        # surface type distinction
        # (need to get annual mb again, because now optimised)
        mb_annual_0_5_m = mb_mod_monthly_0_5_m.get_annual_mb(h, year=2000, spinup=True)

        mb_gradient_0_5_m, _, _, _, _ = scipy.stats.linregress(h[mb_annual_0_5_m < 0],
                                                         y=mb_annual_0_5_m[mb_annual_0_5_m < 0])

        mb_annual_no_sfc_type = mb_mod_no_sfc_type.get_annual_mb(h, year=2000, spinup=True)

        mb_gradient_no_sfc_type, _, _, _, _ = scipy.stats.linregress(h[mb_annual_no_sfc_type < 0],
                                                             y=mb_annual_no_sfc_type[mb_annual_no_sfc_type < 0])

        assert mb_gradient_0_5_m > mb_gradient_no_sfc_type

    @pytest.mark.skip(reason="need refreezing first")
    @pytest.mark.parametrize('mb_type', ['mb_monthly', 'mb_pseudo_daily',
                                         'mb_real_daily'
                                         ])
    def test_get_2d_avg_annual_air_hydro_temp(self, gdir,
                                           mb_type):

        cfg.PARAMS['hydro_month_nh'] = 1
        # just choose any random melt_f
        melt_f = 200
        pf = 2  # precipitation factor
        h, w = gdir.get_inversion_flowline_hw()

        baseline_climate = 'W5E5'
        if mb_type != 'mb_real_daily':
            temporal_resol = 'monthly'
        else:
            temporal_resol = 'daily'
        process_w5e5_data(gdir, temporal_resol=temporal_resol,
                          climate_type=baseline_climate,
                          )

        # those two should be equal!!!
        mb_mod = TIModel_Sfc_Type(gdir, melt_f, mb_type=mb_type,
                                  melt_f_ratio_snow_to_ice=0.5,
                                  prcp_fac=pf,
                                  melt_f_update='annual',
                                  baseline_climate=baseline_climate)
        year = 2000
        temp_avg_2d = mb_mod.get_2d_avg_annual_air_hydro_temp(h,
                                                              year=year)
        temp_avg_2d.mean(axis=1)
        if mb_type != 'mb_real_daily':
            assert np.shape(temp_avg_2d) == (len(h), 12)

        temp_avg_2d_clim = mb_mod.get_2d_avg_annual_air_hydro_temp(h, year=year, avg_climate=True)
        temp_avg_2d_clim.mean(axis=1)
        if mb_type != 'mb_real_daily':
            assert np.shape(temp_avg_2d_clim) == (len(h), 12*31)


    @pytest.mark.parametrize('mb_type', [#'mb_pseudo_daily', it takes so long to run all ...
                                         'mb_real_daily'#, 'mb_monthly',
                                         ])
    def test_specific_winter_mb_no_sfc_type(self, gdir, gdir_aletsch, mb_type):
        oggm_updated = False
        if oggm_updated:
            _, path = utils.get_wgms_files()
            pd_mb_overview = pd.read_csv(path[:-len('/mbdata')] + '/mb_overview_seasonal_mb_time_periods_20220301.csv',
                                         index_col='Unnamed: 0')
            #pd_wgms_data_stats = pd.read_csv(path[:-len('/mbdata')] + '/wgms_data_stats_20220301.csv',
            #                                 index_col='Unnamed: 0')
        else:
            # path_mbsandbox = MBsandbox.__file__[:-len('/__init__.py')]
            # pd_mb_overview = pd.read_csv(path_mbsandbox + '/data/mb_overview_seasonal_mb_time_periods_20220301.csv',
            #                            index_col='Unnamed: 0')
            # pd_wgms_data_stats = pd.read_csv(path_mbsandbox + '/data/wgms_data_stats_20220301.csv',
            #                                 index_col='Unnamed: 0')
            #fp = utils.file_downloader('https://cluster.klima.uni-bremen.de/~lschuster/ref_glaciers' +
            #                           '/data/mb_overview_seasonal_mb_time_periods_20220301.csv')
            fp = 'https://cluster.klima.uni-bremen.de/~lschuster/ref_glaciers/data/mb_overview_seasonal_mb_time_periods_20220301.csv'
            fp_stats = ('https://cluster.klima.uni-bremen.de/~lschuster/ref_glaciers' +
                        '/data/wgms_data_stats_20220301.csv')
            pd_mb_overview = pd.read_csv(fp, index_col='Unnamed: 0')
            #fp_stats = utils.file_downloader('https://cluster.klima.uni-bremen.de/~lschuster/ref_glaciers' +
            #                                 '/data/wgms_data_stats_20220301.csv')
            #pd_wgms_data_stats = pd.read_csv(fp_stats, index_col='Unnamed: 0')

        pd_mb_overview_sel_gdir = pd_mb_overview.loc[pd_mb_overview.rgi_id == gdir.rgi_id]
        pd_mb_overview_sel_gdir.index = pd_mb_overview_sel_gdir.Year
        assert np.all(pd_mb_overview_sel_gdir.day_BEGIN_PERIOD == 1)
        assert np.all(pd_mb_overview_sel_gdir.month_BEGIN_PERIOD == 10)
        assert np.all(pd_mb_overview_sel_gdir.month_END_WINTER == 4)
        assert np.all(pd_mb_overview_sel_gdir.day_END_WINTER == 30)
        pd_mb_overview_sel_gdir_years = pd_mb_overview_sel_gdir.Year.values

        pd_mb_overview_sel_gdir_aletsch = pd_mb_overview.loc[pd_mb_overview.rgi_id == gdir_aletsch.rgi_id]
        pd_mb_overview_sel_gdir_aletsch.index = pd_mb_overview_sel_gdir_aletsch.Year

        pd_mb_overview_sel_gdir_years_aletsch = pd_mb_overview_sel_gdir_aletsch.Year.values

        cfg.PARAMS['hydro_month_nh'] = 1
        # just choose any random melt_f
        melt_f = 200
        pf = 2  # precipitation factor
        h, w = gdir.get_inversion_flowline_hw()
        h_aletsch, w_aletsch = gdir_aletsch.get_inversion_flowline_hw()

        baseline_climate = 'W5E5'
        if mb_type != 'mb_real_daily':
            temporal_resol = 'monthly'
            input_fs = '_monthly_W5E5'
        else:
            temporal_resol = 'daily'
            input_fs = '_daily_W5E5'
        workflow.execute_entity_task(process_w5e5_data, [gdir, gdir_aletsch],
                                     temporal_resol=temporal_resol,
                                     climate_type=baseline_climate,
                                     )

        # those two should be equal!!!
        mb_mod = TIModel(gdir, melt_f, mb_type=mb_type,
                                  prcp_fac=pf,
                                  baseline_climate=baseline_climate)
        mb_mod_aletsch = TIModel(gdir_aletsch, melt_f, mb_type=mb_type,
                                          prcp_fac=pf,
                                          baseline_climate=baseline_climate)

        # in case of HEF this should be the same !!! (as HEF always has WGMS seasonal MB from Oct 1st to April 30th)
        out_w_period_from_wgms = mb_mod.get_specific_winter_mb(heights=h, year=pd_mb_overview_sel_gdir_years, widths=w,
                                                               add_climate=True,
                                                               period_from_wgms=True)
        for c in out_w_period_from_wgms[1:]:
            assert np.shape(c) == (7, len(h))
        # is the climate in the right shape (should have 1D and length of h)

        out_w_default_period = mb_mod.get_specific_winter_mb(heights=h, year=pd_mb_overview_sel_gdir_years, widths=w,
                                                             add_climate=True,
                                                             period_from_wgms=False)
        for k, _ in enumerate(out_w_default_period):
            np.testing.assert_allclose(out_w_default_period[k],
                                       out_w_period_from_wgms[k])

        # in case of Aletsch glacier they should not be equal:
        # no precomputation necessary for TIModel (without sfc type distinction)
        mb_mod_aletsch.get_specific_mb(heights=h_aletsch, widths=w_aletsch,
                                       year=np.arange(pd_mb_overview_sel_gdir_years_aletsch[0],
                                                      pd_mb_overview_sel_gdir_years_aletsch[-1] + 1, 1))
        out_w_period_from_wgms_aletsch = mb_mod_aletsch.get_specific_winter_mb(heights=h_aletsch,
                                                                               year=pd_mb_overview_sel_gdir_years_aletsch,
                                                                               widths=w_aletsch,
                                                                               add_climate=True,
                                                                               period_from_wgms=True)
        out_w_default_period_aletsch = mb_mod_aletsch.get_specific_winter_mb(heights=h_aletsch,
                                                                             year=pd_mb_overview_sel_gdir_years_aletsch,
                                                                             widths=w_aletsch,
                                                                             add_climate=True,
                                                                             period_from_wgms=False)
        # specific winter mb can be different but not too much!
        np.testing.assert_allclose(out_w_period_from_wgms_aletsch[0],
                                   out_w_default_period_aletsch[0], rtol=0.5)
        # actually differences up to rtol=0.33!!!
        for e, y in enumerate(pd_mb_overview_sel_gdir_aletsch.Year):
            # if the actual WGMS period is longer than the default Oct1st - Apr30 period, at least precipitation
            # should be more!
            condi_m_s = pd_mb_overview_sel_gdir_aletsch['BEGIN_PERIOD'].astype(np.datetime64)[y].month < 10
            condi_m_e = pd_mb_overview_sel_gdir_aletsch['END_WINTER'].astype(np.datetime64)[y].month > 4
            if condi_m_e and condi_m_s:
                ### all those that are always above zero and are summed up should be larger:
                # tfm (here it can also be equal)
                assert np.all(out_w_period_from_wgms_aletsch[-3][e] >= out_w_default_period_aletsch[-3][e])
                # liquid prcp
                assert np.all(out_w_period_from_wgms_aletsch[-2][e] > out_w_default_period_aletsch[-2][e])
                # solid prcp
                assert np.all(out_w_period_from_wgms_aletsch[-1][e] >= out_w_default_period_aletsch[-1][e])

            # if the actual WGMS period is short than the default Oct1st - Apr30 period, at least precipitation
            # should be smaller!
            condi_m_s = pd_mb_overview_sel_gdir_aletsch['BEGIN_PERIOD'].astype(np.datetime64)[y].month >= 10
            condi_m_e = pd_mb_overview_sel_gdir_aletsch['END_WINTER'].astype(np.datetime64)[y].month <= 4
            if condi_m_e and condi_m_s:
                ### all those that are always above zero and are summed up should be larger:
                # tfm (here it can also be equal)
                assert np.all(out_w_period_from_wgms_aletsch[-3][e] <= out_w_default_period_aletsch[-3][e])
                # liquid prcp
                assert np.all(out_w_period_from_wgms_aletsch[-2][e] > out_w_default_period_aletsch[-2][e])
                # solid prcp
                assert np.all(out_w_period_from_wgms_aletsch[-1][e] <= out_w_default_period_aletsch[-1][e])

        # let's calibrate the model to match winter MB bias and check if it worked
        # we only do it for Aletsch glacier here (there it is more complicated and more prone to errors)
        pd_geodetic_all = oggm.utils.get_geodetic_mb_dataframe()
        pd_geodetic = pd_geodetic_all.loc[pd_geodetic_all.period == '2000-01-01_2020-01-01']
        mb_geodetic = pd_geodetic.loc[gdir_aletsch.rgi_id].dmdtda * 1000
        pd_mb_overview_sel_gdir = pd_mb_overview.loc[pd_mb_overview.rgi_id == gdir.rgi_id]
        pd_mb_overview_sel_gdir.index = pd_mb_overview_sel_gdir.Year
        yrs_seasonal_mbs = pd_mb_overview_sel_gdir.Year.values
        assert np.all(yrs_seasonal_mbs >= 1980)
        assert np.all(yrs_seasonal_mbs < 2020)
        # yrs_seasonal_mbs = gdir.get_ref_mb_data(input_filesuffix=input_fs)['SUMMER_BALANCE'].dropna().index.values
        # yrs_seasonal_mbs = yrs_seasonal_mbs[(yrs_seasonal_mbs >= 1980) & (yrs_seasonal_mbs < 2020)] # I can't use 1979 (as then we would need climate data in winter 1978!)
        winter_mb_observed = gdir_aletsch.get_ref_mb_data(input_filesuffix=input_fs)
        winter_mb_observed = winter_mb_observed.loc[pd_mb_overview_sel_gdir_years_aletsch]['WINTER_BALANCE']
        try:
            pf_opt = scipy.optimize.brentq(minimize_winter_mb_brentq_geod_via_pf, 0.1, 5, xtol=0.1,
                                           args=(mb_mod_aletsch, mb_geodetic, winter_mb_observed,
                                                 h_aletsch, w_aletsch, pd_mb_overview_sel_gdir_years_aletsch,
                                                 True)  # period_from_wgms
                                           )
        except:
            melt_f_opt_dict = {}
            for pf in np.concatenate([np.arange(0.1, 3, 0.5), np.arange(3, 10, 2)]):
                try:
                    melt_f = scipy.optimize.brentq(minimize_bias_geodetic, 10, 1000,
                                                   xtol=0.01,
                                                   args=(mb_mod_aletsch, mb_geodetic,
                                                         h_aletsch, w_aletsch, pf),
                                                   disp=True)
                    melt_f_opt_dict[pf] = melt_f
                except:
                    pass
            pf_start = list(melt_f_opt_dict.items())[0][0]
            pf_end = list(melt_f_opt_dict.items())[-1][0]

            pf_opt = scipy.optimize.brentq(minimize_winter_mb_brentq_geod_via_pf, pf_start, pf_end, xtol=0.1,
                                           args=(mb_mod_aletsch, mb_geodetic, winter_mb_observed,
                                                 h_aletsch, w_aletsch, pd_mb_overview_sel_gdir_years_aletsch,
                                                 True)  # period_from_wgms
                                           )

        mb_mod_aletsch.prcp_fac = pf_opt
        melt_f_opt = scipy.optimize.brentq(minimize_bias_geodetic, 10, 1000,
                                                      xtol=0.01,
                                                      args=(mb_mod_aletsch, mb_geodetic,
                                                            h_aletsch, w_aletsch, pf_opt),
                                                      disp=True)
        mb_mod_aletsch.melt_f = melt_f_opt

        # first precompute it (like that also years in between (without observations) are estimated)
        # [only important for TIModel_Sfc_Type, if observations have missing years in between]

        mb_mod_aletsch.get_specific_mb(heights=h_aletsch, widths=w_aletsch,
                                       year=np.arange(pd_mb_overview_sel_gdir_years_aletsch[0],
                                                      pd_mb_overview_sel_gdir_years_aletsch[-1] + 1, 1))
        outi_right_period = mb_mod_aletsch.get_specific_winter_mb(heights=h_aletsch,
                                                                  year=pd_mb_overview_sel_gdir_years_aletsch,
                                                                  widths=w_aletsch,
                                                                  add_climate=True,
                                                                  period_from_wgms=True)
        # observed and modelled winter mb bias should be similar now (as prcp-fac was calibrated to match it!)
        np.testing.assert_allclose(outi_right_period[0].mean(),
                                   winter_mb_observed.mean(), rtol=0.01)

        # dividing through pf at the end should be sufficient and it should give
        # the same results as first changing prcp_fac and then reestimating winter prcp
        prpc_no_pf_by_division_at_end = outi_right_period[3] / mb_mod_aletsch.prcp_fac

        mb_mod_aletsch.prcp_fac = 1
        mb_mod_aletsch.get_specific_mb(heights=h_aletsch, widths=w_aletsch,
                                       year=np.arange(pd_mb_overview_sel_gdir_years_aletsch[0],
                                                      pd_mb_overview_sel_gdir_years_aletsch[-1] + 1, 1))
        outi_right_period_no_pf = mb_mod_aletsch.get_specific_winter_mb(heights=h_aletsch,
                                                                        year=pd_mb_overview_sel_gdir_years_aletsch,
                                                                        widths=w_aletsch,
                                                                        add_climate=True,
                                                                        period_from_wgms=True)
        prpc_no_pf_by_division_at_beginning = outi_right_period_no_pf[3]
        # ok, as already assumed we can also just divide at the end to get the winter prcp that is "independent" of
        # prcp. factor!!!
        np.testing.assert_allclose(prpc_no_pf_by_division_at_end, prpc_no_pf_by_division_at_beginning)

        # test if the calibrate_geod_bias_winter_mb works
        # (shoould give the same melt_f, prcp bias as manually done above)
        out = calibrate_to_geodetic_bias_winter_mb(gdir_aletsch, method='pre-check', temp_bias=0,
                                                   mb_type=mb_type,
                                                   grad_type='cte',
                                                   sfc_type_distinction=False)
        # pf_opt = out[0]
        np.testing.assert_allclose(out[0], pf_opt, rtol=0.05)
        np.testing.assert_allclose(out[1], melt_f_opt, rtol=0.05)


    @pytest.mark.slow
    @pytest.mark.parametrize('mb_type', ['mb_pseudo_daily'])#'mb_real_daily', 'mb_monthly',#])
    def test_specific_winter_mb_sfc_type_optim_temp_b(self, gdir, gdir_aletsch, mb_type):

        from MBsandbox.help_func import calibrate_to_geodetic_bias_winter_mb_different_temp_bias_fast
        baseline_climate = 'W5E5'
        if mb_type != 'mb_real_daily':
            temporal_resol = 'monthly'
            input_fs = '_monthly_W5E5'
        else:
            temporal_resol = 'daily'
            input_fs = '_daily_W5E5'

        workflow.execute_entity_task(process_w5e5_data, [gdir, gdir_aletsch],
                                     temporal_resol='monthly',
                                     climate_type=baseline_climate,
                                     )
        workflow.execute_entity_task(process_w5e5_data, [gdir, gdir_aletsch],
                                     temporal_resol='daily',
                                     climate_type=baseline_climate,
                                     )
        import time
        start = time.time()
        pd_calib = calibrate_to_geodetic_bias_winter_mb_different_temp_bias_fast(gdir_aletsch,
                                                                  temp_b_range=np.arange(-4, 4.1, 2),
                                                                  # np.arange(-6,6.1,0.5)
                                                                  method='pre-check', melt_f_update='monthly',
                                                                  sfc_type_distinction=False,
                                                                  path='return')
        end = time.time()
        print(end-start)
        # np.testing.assert_allclose(pd_calib.quot_std, 1, rtol = 0.1)


    @pytest.mark.slow
    @pytest.mark.parametrize('mb_type', ['mb_pseudo_daily',
                                         'mb_real_daily',  'mb_monthly',
                                          ])
    def test_specific_winter_mb_sfc_type(self, gdir, gdir_aletsch, mb_type):

        oggm_updated = False
        if oggm_updated:
            _, path = utils.get_wgms_files()
            pd_mb_overview = pd.read_csv(path[:-len('/mbdata')] + '/mb_overview_seasonal_mb_time_periods_20220301.csv',
                                         index_col='Unnamed: 0')
            pd_wgms_data_stats = pd.read_csv(path[:-len('/mbdata')] + '/wgms_data_stats_20220301.csv',
                                             index_col='Unnamed: 0')
        else:
            # path_mbsandbox = MBsandbox.__file__[:-len('/__init__.py')]
            # pd_mb_overview = pd.read_csv(path_mbsandbox + '/data/mb_overview_seasonal_mb_time_periods_20220301.csv',
            #                            index_col='Unnamed: 0')
            # pd_wgms_data_stats = pd.read_csv(path_mbsandbox + '/data/wgms_data_stats_20220301.csv',
            #                                 index_col='Unnamed: 0')
            #fp = utils.file_downloader('https://cluster.klima.uni-bremen.de/~lschuster/ref_glaciers' +
            #                           '/data/mb_overview_seasonal_mb_time_periods_20220301.csv')
            fp = 'https://cluster.klima.uni-bremen.de/~lschuster/ref_glaciers/data/mb_overview_seasonal_mb_time_periods_20220301.csv'
            fp_stats = ('https://cluster.klima.uni-bremen.de/~lschuster/ref_glaciers' +
                        '/data/wgms_data_stats_20220301.csv')
            pd_mb_overview = pd.read_csv(fp, index_col='Unnamed: 0')
            #fp_stats = utils.file_downloader('https://cluster.klima.uni-bremen.de/~lschuster/ref_glaciers' +
            #                                 '/data/wgms_data_stats_20220301.csv')
            pd_wgms_data_stats = pd.read_csv(fp_stats, index_col='Unnamed: 0')
        pd_mb_overview_sel_gdir = pd_mb_overview.loc[pd_mb_overview.rgi_id == gdir.rgi_id]
        pd_mb_overview_sel_gdir.index = pd_mb_overview_sel_gdir.Year
        assert np.all(pd_mb_overview_sel_gdir.day_BEGIN_PERIOD == 1)
        assert np.all(pd_mb_overview_sel_gdir.month_BEGIN_PERIOD == 10)
        assert np.all(pd_mb_overview_sel_gdir.month_END_WINTER == 4)
        assert np.all(pd_mb_overview_sel_gdir.day_END_WINTER == 30)
        pd_mb_overview_sel_gdir_years = pd_mb_overview_sel_gdir.Year.values

        pd_mb_overview_sel_gdir_aletsch = pd_mb_overview.loc[pd_mb_overview.rgi_id == gdir_aletsch.rgi_id]
        pd_mb_overview_sel_gdir_aletsch.index = pd_mb_overview_sel_gdir_aletsch.Year

        pd_mb_overview_sel_gdir_years_aletsch = pd_mb_overview_sel_gdir_aletsch.Year.values

        cfg.PARAMS['hydro_month_nh'] = 1
        # just choose any random melt_f
        melt_f = 200
        pf = 2  # precipitation factor
        h, w = gdir.get_inversion_flowline_hw()
        h_aletsch, w_aletsch = gdir_aletsch.get_inversion_flowline_hw()

        baseline_climate = 'W5E5'
        if mb_type != 'mb_real_daily':
            temporal_resol = 'monthly'
            input_fs = '_monthly_W5E5'
        else:
            temporal_resol = 'daily'
            input_fs = '_daily_W5E5'

        workflow.execute_entity_task(process_w5e5_data, [gdir, gdir_aletsch],
                                     temporal_resol=temporal_resol,
                                     climate_type=baseline_climate,
                                     )

        # those two should be equal!!!
        mb_mod = TIModel_Sfc_Type(gdir, melt_f, mb_type=mb_type,
                                  melt_f_ratio_snow_to_ice=0.5,
                                  prcp_fac=pf,
                                  melt_f_update='monthly',
                                  baseline_climate=baseline_climate)
        mb_mod_aletsch = TIModel_Sfc_Type(gdir_aletsch, melt_f, mb_type=mb_type,
                                  melt_f_ratio_snow_to_ice=0.5,
                                  prcp_fac=pf,
                                  melt_f_update='monthly',
                                  baseline_climate=baseline_climate)

        # in case of HEF this should be the same !!! (as HEF always has WGMS seasonal MB from Oct 1st to April 30th)
        out_w_period_from_wgms = mb_mod.get_specific_winter_mb(heights=h, year=pd_mb_overview_sel_gdir_years, widths=w,
                                                               add_climate=True,
                                                               period_from_wgms=True)
        for c in out_w_period_from_wgms[1:]:
            assert np.shape(c) == (7, len(h))
        # is the climate in the right shape (should have 1D and length of h)

        out_w_default_period = mb_mod.get_specific_winter_mb(heights=h, year=pd_mb_overview_sel_gdir_years, widths=w,
                                                             add_climate=True,
                                                             period_from_wgms=False)
        for k, _ in enumerate(out_w_default_period):
            np.testing.assert_allclose(out_w_default_period[k],
                                       out_w_period_from_wgms[k])

        # in case of Aletsch glacier they should not be equal:
        # first precompute it (like that also years in between (without observations) are estimated)
        mb_mod_aletsch.get_specific_mb(heights=h_aletsch, widths=w_aletsch,
                                       year=np.arange(pd_mb_overview_sel_gdir_years_aletsch[0],
                                                        pd_mb_overview_sel_gdir_years_aletsch[-1]+1, 1))
        out_w_period_from_wgms_aletsch = mb_mod_aletsch.get_specific_winter_mb(heights=h_aletsch,
                                                                       year=pd_mb_overview_sel_gdir_years_aletsch,
                                                                       widths=w_aletsch,
                                                               add_climate=True,
                                                               period_from_wgms=True)
        out_w_default_period_aletsch = mb_mod_aletsch.get_specific_winter_mb(heights=h_aletsch,
                                                                     year=pd_mb_overview_sel_gdir_years_aletsch,
                                                                     widths=w_aletsch,
                                                             add_climate=True,
                                                             period_from_wgms=False)
        # specific winter mb can be different but not too much!
        np.testing.assert_allclose(out_w_period_from_wgms_aletsch[0],
                                   out_w_default_period_aletsch[0], rtol=0.5)
        # actually differences up to rtol=0.39!!!
        for e, y in enumerate(pd_mb_overview_sel_gdir_aletsch.Year):
            # if the actual WGMS period is longer than the default Oct1st - Apr30 period, at least precipitation
            # should be more!
            condi_m_s = pd_mb_overview_sel_gdir_aletsch['BEGIN_PERIOD'].astype(np.datetime64)[y].month < 10
            condi_m_e = pd_mb_overview_sel_gdir_aletsch['END_WINTER'].astype(np.datetime64)[y].month > 4
            if condi_m_e and condi_m_s:
                ### all those that are always above zero and are summed up should be larger:
                # tfm (here it can also be equal)
                assert np.all(out_w_period_from_wgms_aletsch[-3][e] >= out_w_default_period_aletsch[-3][e])
                # liquid prcp
                assert np.all(out_w_period_from_wgms_aletsch[-2][e] > out_w_default_period_aletsch[-2][e])
                # solid prcp
                assert np.all(out_w_period_from_wgms_aletsch[-1][e] >= out_w_default_period_aletsch[-1][e])

            # if the actual WGMS period is short than the default Oct1st - Apr30 period, at least precipitation
            # should be smaller!
            condi_m_s = pd_mb_overview_sel_gdir_aletsch['BEGIN_PERIOD'].astype(np.datetime64)[y].month >= 10
            condi_m_e = pd_mb_overview_sel_gdir_aletsch['END_WINTER'].astype(np.datetime64)[y].month <= 4
            if condi_m_e and condi_m_s:
                ### all those that are always above zero and are summed up should be larger:
                # tfm (here it can also be equal)
                assert np.all(out_w_period_from_wgms_aletsch[-3][e] <= out_w_default_period_aletsch[-3][e])
                # liquid prcp
                assert np.all(out_w_period_from_wgms_aletsch[-2][e] > out_w_default_period_aletsch[-2][e])
                # solid prcp
                assert np.all(out_w_period_from_wgms_aletsch[-1][e] <= out_w_default_period_aletsch[-1][e])

        # let's calibrate the model to match winter MB bias and check if it worked
        # we only do it for Aletsch glacier here (there it is more complicated and more prone to errors)
        pd_geodetic_all = oggm.utils.get_geodetic_mb_dataframe()
        pd_geodetic = pd_geodetic_all.loc[pd_geodetic_all.period == '2000-01-01_2020-01-01']
        mb_geodetic = pd_geodetic.loc[gdir_aletsch.rgi_id].dmdtda * 1000
        pd_mb_overview_sel_gdir = pd_mb_overview.loc[pd_mb_overview.rgi_id == gdir.rgi_id]
        pd_mb_overview_sel_gdir.index = pd_mb_overview_sel_gdir.Year
        yrs_seasonal_mbs = pd_mb_overview_sel_gdir.Year.values
        assert np.all(yrs_seasonal_mbs >= 1980)
        assert np.all(yrs_seasonal_mbs < 2020)
        # yrs_seasonal_mbs = gdir.get_ref_mb_data(input_filesuffix=input_fs)['SUMMER_BALANCE'].dropna().index.values
        # yrs_seasonal_mbs = yrs_seasonal_mbs[(yrs_seasonal_mbs >= 1980) & (yrs_seasonal_mbs < 2020)] # I can't use 1979 (as then we would need climate data in winter 1978!)
        winter_mb_observed = gdir_aletsch.get_ref_mb_data(input_filesuffix=input_fs)
        winter_mb_observed = winter_mb_observed.loc[pd_mb_overview_sel_gdir_years_aletsch]['WINTER_BALANCE']
        try:
            pf_opt = scipy.optimize.brentq(minimize_winter_mb_brentq_geod_via_pf, 0.1, 5, xtol=0.1,
                                           args=(mb_mod_aletsch, mb_geodetic, winter_mb_observed,
                                                 h_aletsch, w_aletsch, pd_mb_overview_sel_gdir_years_aletsch,
                                                 True) #period_from_wgms
                                           )
        except:
            melt_f_opt_dict = {}
            for pf in np.concatenate([np.arange(0.1, 3, 0.5), np.arange(3, 10, 2)]):
                try:
                    melt_f = scipy.optimize.brentq(minimize_bias_geodetic, 10, 1000,
                                                   xtol=0.01,
                                                   args=(mb_mod_aletsch, mb_geodetic,
                                                         h_aletsch, w_aletsch, pf),
                                                   disp=True)
                    melt_f_opt_dict[pf] = melt_f
                except:
                    pass
            pf_start = list(melt_f_opt_dict.items())[0][0]
            pf_end = list(melt_f_opt_dict.items())[-1][0]

            pf_opt = scipy.optimize.brentq(minimize_winter_mb_brentq_geod_via_pf, pf_start, pf_end, xtol=0.1,
                                           args=(mb_mod_aletsch, mb_geodetic, winter_mb_observed,
                                                 h_aletsch, w_aletsch, pd_mb_overview_sel_gdir_years_aletsch,
                                                 True)  # period_from_wgms
                                           )

        mb_mod_aletsch.prcp_fac = pf_opt
        melt_f_opt = scipy.optimize.brentq(minimize_bias_geodetic, 10, 1000,
                                                      xtol=0.01,
                                                      args=(mb_mod_aletsch, mb_geodetic,
                                                            h_aletsch, w_aletsch, pf_opt),
                                                      disp=True)
        mb_mod_aletsch.melt_f = melt_f_opt

        # first precompute it (like that also years in between (without observations) are estimated)
        # [only important for TIModel_Sfc_Type, if observations have missing years in between]

        mb_mod_aletsch.get_specific_mb(heights=h_aletsch, widths=w_aletsch,
                                       year=np.arange(pd_mb_overview_sel_gdir_years_aletsch[0],
                                                      pd_mb_overview_sel_gdir_years_aletsch[-1] + 1, 1))
        outi_right_period = mb_mod_aletsch.get_specific_winter_mb(heights=h_aletsch,
                                                                  year=pd_mb_overview_sel_gdir_years_aletsch,
                                                                  widths=w_aletsch,
                                                                  add_climate=True,
                                                                  period_from_wgms=True)
        # observed and modelled winter mb bias should be similar now (as prcp-fac was calibrated to match it!)
        np.testing.assert_allclose(outi_right_period[0].mean(),
                                   winter_mb_observed.mean(), rtol=0.01)

        # dividing through pf at the end should be sufficient and it should give
        # the same results as first changing prcp_fac and then reestimating winter prcp
        prpc_no_pf_by_division_at_end = outi_right_period[3]/mb_mod_aletsch.prcp_fac

        mb_mod_aletsch.prcp_fac = 1
        mb_mod_aletsch.get_specific_mb(heights=h_aletsch, widths=w_aletsch,
                                       year=np.arange(pd_mb_overview_sel_gdir_years_aletsch[0],
                                                      pd_mb_overview_sel_gdir_years_aletsch[-1] + 1, 1))
        outi_right_period_no_pf = mb_mod_aletsch.get_specific_winter_mb(heights=h_aletsch,
                                                                  year=pd_mb_overview_sel_gdir_years_aletsch,
                                                                  widths=w_aletsch,
                                                                  add_climate=True,
                                                                  period_from_wgms=True)
        prpc_no_pf_by_division_at_beginning = outi_right_period_no_pf[3]
        # ok, as already assumed we can also just divide at the end to get the winter prcp that is "independent" of
        # prcp. factor!!!
        np.testing.assert_allclose(prpc_no_pf_by_division_at_end, prpc_no_pf_by_division_at_beginning)

        # test if the calibrate_geod_bias_winter_mb works
        # (shoould give the same melt_f, prcp bias as manually done above)
        out = calibrate_to_geodetic_bias_winter_mb(gdir_aletsch, method='pre-check', temp_bias=0,
                                                   mb_type=mb_type,
                                                   grad_type='cte',
                                                   sfc_type_distinction=True)
        # pf_opt = out[0]
        np.testing.assert_allclose(out[0], pf_opt, rtol=0.05)
        np.testing.assert_allclose(out[1], melt_f_opt, rtol=0.05)

        ### 3-step calibration -> this is WIP -> not working at the moment!!!
        # fs = '_monthly_W5E5'
        # mb_glaciological = gdir.get_ref_mb_data(input_filesuffix=fs)['ANNUAL_BALANCE']
        # ys_glac = mb_glaciological.index.values
        #
        # try:
        #     mb_mod_aletsch.temp_b = scipy.optimize.brentq(optimize_std_quot_brentq_via_temp_b_w_min_winter_geod_bias,
        #                                                   -3, 3, xtol=0.01,
        #                                                   args=(mb_mod_aletsch,
        #                                                            mb_geodetic,
        #                                                            winter_mb_observed,
        #                                                            pd_mb_overview_sel_gdir_years_aletsch,
        #                                                            mb_glaciological,
        #                                                            ys_glac,
        #                                                            h_aletsch, w_aletsch),
        #                                                   disp=True)
        # except:
        #     mb_mod_aletsch.temp_b = scipy.optimize.brentq(optimize_std_quot_brentq_via_temp_b_w_min_winter_geod_bias,
        #                                                   0, 3, xtol=0.01,
        #                                                   args=(mb_mod_aletsch,
        #                                                         mb_geodetic,
        #                                                         winter_mb_observed,
        #                                                         pd_mb_overview_sel_gdir_years_aletsch,
        #                                                         mb_glaciological,
        #                                                         ys_glac,
        #                                                         h_aletsch, w_aletsch),
        #                                                   disp=True)
        #
        # try:
        #     pf_opt = scipy.optimize.brentq(minimize_winter_mb_brentq_geod_via_pf, 0.1, 5, xtol=0.1,
        #                                    args=(mb_mod_aletsch, mb_geodetic, winter_mb_observed,
        #                                          h_aletsch, w_aletsch, pd_mb_overview_sel_gdir_years_aletsch,
        #                                          True) #period_from_wgms
        #                                    )
        # except:
        #     melt_f_opt_dict = {}
        #     for pf in np.concatenate([np.arange(0.1, 3, 0.5), np.arange(3, 10, 2)]):
        #         try:
        #             melt_f = scipy.optimize.brentq(minimize_bias_geodetic, 10, 1000,
        #                                            xtol=0.01,
        #                                            args=(mb_mod_aletsch, mb_geodetic,
        #                                                  h_aletsch, w_aletsch, pf),
        #                                            disp=True)
        #             melt_f_opt_dict[pf] = melt_f
        #         except:
        #             pass
        #     pf_start = list(melt_f_opt_dict.items())[0][0]
        #     pf_end = list(melt_f_opt_dict.items())[-1][0]
        #
        #     pf_opt = scipy.optimize.brentq(minimize_winter_mb_brentq_geod_via_pf, pf_start, pf_end, xtol=0.1,
        #                                    args=(mb_mod_aletsch, mb_geodetic, winter_mb_observed,
        #                                          h_aletsch, w_aletsch, pd_mb_overview_sel_gdir_years_aletsch,
        #                                          True)  # period_from_wgms
        #                                    )
        #
        # mb_mod_aletsch.pf_opt = pf_opt
        # mb_mod_aletsch.melt_f = scipy.optimize.brentq(minimize_bias_geodetic, 10, 1000,
        #                                               xtol=0.01,
        #                                               args=(mb_mod_aletsch, mb_geodetic,
        #                                                     h_aletsch, w_aletsch, pf_opt),
        #                                               disp=True)
        #
        # mb_mod_aletsch.get_specific_mb(heights=h_aletsch, widths=w_aletsch,
        #                                year=np.arange(1979, 2020,1) )
        # specific_winter_mb = mb_mod_aletsch.get_specific_winter_mb(heights=h_aletsch,
        #                                                           year=pd_mb_overview_sel_gdir_years_aletsch,
        #                                                           widths=w_aletsch,
        #                                                           add_climate=True,
        #                                                           period_from_wgms=True)
        #
        # # observed and modelled winter mb bias should be similar now (as prcp-fac was calibrated to match it!)
        # np.testing.assert_allclose(specific_winter_mb[0].mean(),
        #                            winter_mb_observed.mean(), rtol=0.05)
        #
        # #
        # mean_modelled_geod_period = mb_mod_aletsch.get_specific_mb(heights=h_aletsch, widths=w_aletsch,
        #                                year=np.arange(2000, 2020,1) ).mean()
        # # observed geodetic mean should be equal to modelled time period
        # np.testing.assert_allclose(mean_modelled_geod_period,
        #                            mb_geodetic, rtol=0.05)
        #
        # annual_modelled_std_period = mb_mod_aletsch.get_specific_mb(heights=h_aletsch, widths=w_aletsch,
        #                                                            year=ys_glac).mean()










class Test_geodetic_hydro1:
    # classes have to be upper case in order that they
    # all tests with hydro_month = 1
    @pytest.mark.no_w5e5
    def test_hydro_years_HEF(self, gdir):
        # only very basic test, the other stuff is done in oggm man basis
        # test if it also works for hydro_month ==1, necessary for geodetic mb
        # if hydro_month ==1, and msm start in 1979, then hydro_year should
        # also
        # be 1979, this works only with the newest OGGM dev version...
        cfg.PARAMS['hydro_month_nh'] = 1

        h, w = gdir.get_inversion_flowline_hw()
        cfg.PARAMS['baseline_climate'] = 'ERA5dr'
        oggm.shop.ecmwf.process_ecmwf_data(gdir, dataset='ERA5dr')
        oggm.core.climate.historical_climate_qc(gdir)

        f = gdir.get_filepath('climate_historical', filesuffix='')
        test_climate = xr.open_dataset(f)
        assert test_climate.time[0] == np.datetime64('1979-01-01')
        assert test_climate.time[-1] == np.datetime64('2018-12-01')

        # now test it for ERA5_daily
        cfg.PARAMS['baseline_climate'] = 'ERA5_daily'
        process_era5_daily_data(gdir, output_filesuffix='_ERA5_daily')
        f = gdir.get_filepath('climate_historical', filesuffix='_ERA5_daily')
        test_climate = xr.open_dataset(f)
        assert test_climate.time[0] == np.datetime64('1979-01-01')
        assert test_climate.time[-1] == np.datetime64('2018-12-31')

    def test_optimize_std_quot_brentq_W5E5(self, gdir):
        # check if double optimisation of bias and std_quotient works
        cfg.PARAMS['hydro_month_nh'] = 1

        grad_type = 'cte'
        N = 100
        for mb_type in ['mb_monthly', 'mb_pseudo_daily', 'mb_real_daily']:
            melt_fs = []
            prcp_facs = []
            #for climate_type in ['WFDE5_CRU', 'W5E5', 'W5E5_MSWEP']: # we don't use that anymore ...
            for climate_type in ['W5E5']:
                if mb_type != 'mb_real_daily':
                    temporal_resol = 'monthly'
                    process_w5e5_data(gdir, climate_type=climate_type,
                                      temporal_resol=temporal_resol)

                else:
                    # because of get_climate_info need ERA5_daily as
                    # baseline_climate until WFDE5_daily is included in
                    # get_climate_info
                    # cfg.PARAMS['baseline_climate'] = 'ERA5_daily'
                    temporal_resol='daily'
                    process_w5e5_data(gdir, climate_type=climate_type,
                                      temporal_resol=temporal_resol)

                hgts, widths = gdir.get_inversion_flowline_hw()
                fs = '_{}_{}'.format(temporal_resol, climate_type)
                mbdf = gdir.get_ref_mb_data(input_filesuffix=fs)
                gd_mb = TIModel(gdir, None, prcp_fac=1,
                                mb_type=mb_type,
                                grad_type=grad_type,
                                N=N, baseline_climate=climate_type)
                pf_opt = scipy.optimize.brentq(optimize_std_quot_brentq, 0.01, 20,
                                               args=(gd_mb, gdir, fs),
                                               xtol=0.01)

                melt_f_opt_pf = scipy.optimize.brentq(minimize_bias, 1, 10000,
                                                   disp=True, xtol=0.1,
                                                   args=(gd_mb, gdir,
                                                         pf_opt, False, fs))
                gd_mb.melt_f = melt_f_opt_pf
                gd_mb.prcp_fac = pf_opt
                gd_mb.historical_climate_qc_mod(gdir)
                mb_specific = gd_mb.get_specific_mb(heights=hgts, widths=widths,
                                                    year=mbdf.index.values)

                RMSD, bias, rcor, quot_std = compute_stat(mb_specific=mb_specific,
                                                          mbdf=mbdf)

                # check if the bias is optimised
                assert bias.round() == 0
                # check if the std_quotient is optimised
                assert quot_std.round(1) == 1

                # save melt_f and prcp_fac to compare between climate datasets
                melt_fs.append(melt_f_opt_pf)
                prcp_facs.append(pf_opt)
            if len(melt_fs) > 1:
                assert_allclose(melt_fs[0], melt_fs[1], rtol=0.2)
            # prcp_fac can be quite different ...
            #assert_allclose(prcp_facs[0], prcp_facs[1])

    def test_optimize_std_quot_brentq_W5E5_via_melt_f(self, gdir):
        # check if double optimisation of bias and std_quotient works
        # when calibrating pf as first variable and only tuning melt_f to match WGMS ref glacier std dev.
        from MBsandbox.help_func import (optimize_std_quot_brentq_geod_via_melt_f,
                                         minimize_bias_geodetic_via_pf_fixed_melt_f)
        cfg.PARAMS['hydro_month_nh'] = 1

        grad_type = 'cte'
        N = 100
        # get the geodetic calibration data
        pd_geodetic_all = utils.get_geodetic_mb_dataframe()
        pd_geodetic = pd_geodetic_all.loc[pd_geodetic_all.period == '2000-01-01_2020-01-01']
        mb_geodetic = pd_geodetic.loc[gdir.rgi_id].dmdtda * 1000

        melt_fs = []
        prcp_facs = []
        for mb_type in ['mb_monthly', 'mb_pseudo_daily', 'mb_real_daily']:

            for climate_type in ['W5E5']:

                if mb_type != 'mb_real_daily':
                    temporal_resol = 'monthly'
                    process_w5e5_data(gdir, climate_type=climate_type,
                                      temporal_resol=temporal_resol)
                    input_fs = '_monthly_W5E5'
                else:
                    # because of get_climate_info need ERA5_daily as
                    # baseline_climate until WFDE5_daily is included in
                    # get_climate_info
                    # cfg.PARAMS['baseline_climate'] = 'ERA5_daily'
                    temporal_resol='daily'
                    process_w5e5_data(gdir, climate_type=climate_type,
                                      temporal_resol=temporal_resol)
                    input_fs = '_daily_W5E5'

                hgts, widths = gdir.get_inversion_flowline_hw()
                fs = '_{}_{}'.format(temporal_resol, climate_type)
                mbdf = gdir.get_ref_mb_data(input_filesuffix=fs)
                ys_glac = mbdf.index.values
                gd_mb = TIModel(gdir, None, prcp_fac=1,
                                mb_type=mb_type,
                                grad_type=grad_type,
                                N=N, baseline_climate=climate_type)
                mbdf = gdir.get_ref_mb_data(input_filesuffix=input_fs)
                mb_glaciological = mbdf['ANNUAL_BALANCE']
                try:
                    melt_f_opt = scipy.optimize.brentq(optimize_std_quot_brentq_geod_via_melt_f, 10, 1000,
                                                   args=(gd_mb, mb_geodetic, mb_glaciological, hgts, widths, ys_glac),
                                                   xtol=0.1)
                except ValueError:  # (' f(a) and f(b) must have different signs'):
                    # the melt_f minimum (max) is just too low (high) to find an appropriate pf
                    # (bias is for min & max positive)
                    # try out with melt_f lying only between 50 and 600
                    # -> maybe not so good to do this, but don't have a better idea
                    try:
                        melt_f_opt = scipy.optimize.brentq(optimize_std_quot_brentq_geod_via_melt_f, 50, 600,
                                                           args=(
                                                           gd_mb, mb_geodetic, mb_glaciological, hgts, widths, ys_glac),
                                                           xtol=0.1)
                    except:
                        # if it still does not work increase to 150!!!
                        melt_f_opt = scipy.optimize.brentq(optimize_std_quot_brentq_geod_via_melt_f, 150, 600,
                                                           args=(
                                                           gd_mb, mb_geodetic, mb_glaciological, hgts, widths, ys_glac),
                                                           xtol=0.1)


                pf_opt_melt_f = scipy.optimize.brentq(minimize_bias_geodetic_via_pf_fixed_melt_f, 0.1, 10,
                                                   disp=True, xtol=0.1,
                                                   args=(gd_mb, mb_geodetic,
                                                         hgts, widths,
                                                         melt_f_opt))
                gd_mb.prcp_fac = pf_opt_melt_f
                gd_mb.melt_f = melt_f_opt
                # gd_mb.historical_climate_qc_mod(gdir)
                mb_specific = gd_mb.get_specific_mb(heights=hgts, widths=widths,
                                                    year=np.arange(2000,2020,1))

                mb_specific_opt_std = gd_mb.get_specific_mb(heights=hgts, widths=widths,
                                                    year=mbdf.index.values)

                #RMSD, bias, rcor, quot_std = compute_stat(mb_specific=mb_specific,
                #                                          mbdf=mb_geodetic)
                bias = mb_specific.mean() - mb_geodetic
                ref_std = mb_glaciological.std()
                mod_std = mb_specific_opt_std.std()
                quot_std = mod_std / ref_std

                # check if the bias is optimised
                assert bias.round() == 0
                # check if the std_quotient is optimised
                assert quot_std.round(1) == 1

                # save melt_f and prcp_fac to compare between climate datasets
                melt_fs.append(melt_f_opt)
                prcp_facs.append(pf_opt_melt_f)
            #            assert_allclose(melt_fs[0], melt_fs[1], rtol=0.2)
            # prcp_fac can be quite different ...
            #assert_allclose(prcp_facs[0], prcp_facs[1])

        assert_allclose(melt_fs[0], melt_fs[1], rtol= 0.2)
        assert_allclose(melt_fs[0], melt_fs[2], rtol= 0.2)

    def test_optimize_std_quot_brentq_W5E5_via_temp_bias(self, gdir):
        # check if double optimisation of bias and std_quotient works
        # when calibrating melt_f as first variable and tuning temp. bias to match WGMS ref glacier std dev.
        # the prcp. fac is here not used for calibration (just set to a cte "arbitrary" value)
        from MBsandbox.help_func import optimize_std_quot_brentq_geod_via_temp_bias
        # just use a possible precipitation factor (it is here constant and not calibrated)
        pf = 1

        cfg.PARAMS['hydro_month_nh'] = 1

        grad_type = 'cte'
        N = 100
        # get the geodetic calibration data
        pd_geodetic_all = utils.get_geodetic_mb_dataframe()

        pd_geodetic = pd_geodetic_all.loc[pd_geodetic_all.period == '2000-01-01_2020-01-01']
        mb_geodetic = pd_geodetic.loc[gdir.rgi_id].dmdtda * 1000

        melt_fs = []
        #prcp_facs = []
        temp_bias_s = []
        for mb_type in ['mb_monthly', 'mb_pseudo_daily', 'mb_real_daily']:
            for climate_type in ['W5E5']:
                if mb_type != 'mb_real_daily':
                    temporal_resol = 'monthly'
                    process_w5e5_data(gdir, climate_type=climate_type,
                                      temporal_resol=temporal_resol)
                    input_fs = '_monthly_W5E5'
                else:
                    # because of get_climate_info need ERA5_daily as
                    # baseline_climate until WFDE5_daily is included in
                    # get_climate_info
                    # cfg.PARAMS['baseline_climate'] = 'ERA5_daily'
                    temporal_resol='daily'
                    process_w5e5_data(gdir, climate_type=climate_type,
                                      temporal_resol=temporal_resol)
                    input_fs = '_daily_W5E5'

                hgts, widths = gdir.get_inversion_flowline_hw()
                fs = '_{}_{}'.format(temporal_resol, climate_type)
                mbdf = gdir.get_ref_mb_data(input_filesuffix=fs)
                ys_glac = mbdf.index.values
                gd_mb = TIModel(gdir, None, prcp_fac=1,
                                mb_type=mb_type,
                                grad_type=grad_type,
                                N=N, baseline_climate=climate_type)
                mbdf = gdir.get_ref_mb_data(input_filesuffix=input_fs)
                mb_glaciological = mbdf['ANNUAL_BALANCE']
                try:
                    temp_b_opt = scipy.optimize.brentq(optimize_std_quot_brentq_geod_via_temp_bias,
                                                       -6, 6, # range of allowed temp. bias values
                                                   args=(gd_mb, mb_geodetic, mb_glaciological, hgts, widths,
                                                         ys_glac, pf),
                                                   xtol=0.1)
                except ValueError:  # (' f(a) and f(b) must have different signs'):
                    # the temp. bias minimum (max) is just too low (high) to find an appropriate melt_f
                    # (bias is for min & max positive)
                    # try out with temp. bias that are less extreme
                    # -> maybe not so good to do this, but don't have a better idea
                    try:
                        temp_b_opt = scipy.optimize.brentq(optimize_std_quot_brentq_geod_via_temp_bias, -3, 3,
                                                           args=(
                                                           gd_mb, mb_geodetic, mb_glaciological, hgts, widths,
                                                           ys_glac, pf),
                                                           xtol=0.1)
                    except:
                        temp_b_opt = scipy.optimize.brentq(optimize_std_quot_brentq_geod_via_temp_bias, -2, 2,
                                                           args=(gd_mb, mb_geodetic, mb_glaciological,
                                                                 hgts, widths, ys_glac, pf),
                                                           xtol=0.1)
                gd_mb.temp_bias = temp_b_opt
                melt_f_opt = scipy.optimize.brentq(minimize_bias_geodetic, 10, 1000,
                                                   disp=True, xtol=0.01,
                                                   args=(gd_mb, mb_geodetic, hgts, widths, pf))
                gd_mb.prcp_fac = pf
                gd_mb.melt_f = melt_f_opt
                gd_mb.temp_bias = temp_b_opt
                # gd_mb.historical_climate_qc_mod(gdir)
                mb_specific = gd_mb.get_specific_mb(heights=hgts, widths=widths,
                                                    year=np.arange(2000, 2020, 1))

                mb_specific_opt_std = gd_mb.get_specific_mb(heights=hgts, widths=widths,
                                                    year=mbdf.index.values)

                #RMSD, bias, rcor, quot_std = compute_stat(mb_specific=mb_specific,
                #                                          mbdf=mb_geodetic)
                bias = mb_specific.mean() - mb_geodetic
                ref_std = mb_glaciological.std()
                mod_std = mb_specific_opt_std.std()
                quot_std = mod_std / ref_std

                # check if the bias is optimised
                assert bias.round() == 0
                # check if the std_quotient is optimised (does not need to be perfect!!!)
                assert quot_std.round(1) == 1

                # save melt_f and prcp_fac to compare between climate datasets
                melt_fs.append(melt_f_opt)
                temp_bias_s.append(temp_b_opt)
            #            assert_allclose(melt_fs[0], melt_fs[1], rtol=0.2)
            # prcp_fac can be quite different ...
            #assert_allclose(prcp_facs[0], prcp_facs[1])

        assert_allclose(melt_fs[1], melt_fs[2], rtol= 0.2)
        assert_allclose(temp_bias_s[1], temp_bias_s[2], rtol= 0.2)


    def test_minimize_geodetic_via_temp_bias(self, gdir):
        from MBsandbox.help_func import (minimize_bias_geodetic_via_temp_bias)
        cfg.PARAMS['hydro_month_nh'] = 1

        grad_type = 'cte'
        N = 100
        # get the geodetic calibration data
        pd_geodetic_all = utils.get_geodetic_mb_dataframe()

        pd_geodetic = pd_geodetic_all.loc[pd_geodetic_all.period == '2000-01-01_2020-01-01']
        mb_geodetic = pd_geodetic.loc[gdir.rgi_id].dmdtda * 1000

        melt_f_opt_ref, pf_opt_ref = (160, 2.7) # just approximate values from the optimal ref
        for mb_type in ['mb_monthly', 'mb_pseudo_daily', 'mb_real_daily']:

            for climate_type in ['W5E5']:

                if mb_type != 'mb_real_daily':
                    temporal_resol = 'monthly'
                    process_w5e5_data(gdir, climate_type=climate_type,
                                      temporal_resol=temporal_resol)
                    input_fs = '_monthly_W5E5'
                else:
                    # because of get_climate_info need ERA5_daily as
                    # baseline_climate until WFDE5_daily is included in
                    # get_climate_info
                    # cfg.PARAMS['baseline_climate'] = 'ERA5_daily'
                    temporal_resol='daily'
                    process_w5e5_data(gdir, climate_type=climate_type,
                                      temporal_resol=temporal_resol)
                    input_fs = '_daily_W5E5'

                hgts, widths = gdir.get_inversion_flowline_hw()
                fs = '_{}_{}'.format(temporal_resol, climate_type)
                mbdf = gdir.get_ref_mb_data(input_filesuffix=fs)
                ys_glac = mbdf.index.values
                gd_mb = TIModel(gdir, None, prcp_fac=1,
                                mb_type=mb_type,
                                grad_type=grad_type,
                                N=N, baseline_climate=climate_type)

                temp_bias = scipy.optimize.brentq(minimize_bias_geodetic_via_temp_bias, -3, 3,
                                                      disp=True, xtol=0.01,
                                                      args=(gd_mb, mb_geodetic,
                                                            hgts, widths,
                                                            melt_f_opt_ref, pf_opt_ref))
                gd_mb.temp_bias = temp_bias
                gd_mb.prcp_fac = pf_opt_ref
                gd_mb.melt_f = melt_f_opt_ref

                mb_specific = gd_mb.get_specific_mb(heights=hgts, widths=widths,
                                                    year=np.arange(2000, 2020, 1))

                bias = mb_specific.mean() - mb_geodetic
                # check if the bias is optimised
                assert bias.round() == 0

    @pytest.mark.skip(reason='this test is expected to fail')
    def test_daily_monthly_annual_specific_mb(self, gdir):
        # this test is "expected" to fail
        # there are small differences because of different days on a year (leap years) which are differently
        # represented inside of daily, monthly and annual specific MB at the moment!!!
        # tested for both ERA5 and WFDE5

        cfg.PARAMS['hydro_month_nh'] = 1  # 0
        h, w = gdir.get_inversion_flowline_hw()

        grad_type = 'cte'
        for dataset in ['ERA5', 'WFDE5_CRU']:
            if dataset == 'ERA5':
                pf = 2.5
            elif dataset == 'WFDE5_CRU':
                pf = 1
            for mb_type in ['mb_monthly', 'mb_real_daily']:
                if mb_type == 'mb_real_daily' and dataset == 'ERA5':
                    climate = 'ERA5_daily'
                    cfg.PARAMS['baseline_climate'] = climate
                    fs = '_daily_ERA5_daily'
                    process_era5_daily_data(gdir,
                                            output_filesuffix=fs)
                elif mb_type == 'mb_real_daily' and dataset == 'WFDE5_CRU':
                    climate = dataset
                    fs = '_daily_WFDE5_CRU'
                    process_w5e5_data(gdir, climate_type=climate,
                                      temporal_resol='daily')
                elif mb_type == 'mb_monthly' and dataset == 'ERA5':
                    climate = 'ERA5dr'
                    cfg.PARAMS['baseline_climate'] = climate
                    fs = '_monthly_ERA5dr'
                    oggm.shop.ecmwf.process_ecmwf_data(gdir,
                                                       dataset=climate,
                                                       output_filesuffix=fs)
                elif mb_type == 'mb_monthly' and dataset == 'WFDE5_CRU':
                    climate = dataset
                    fs = '_monthly_WFDE5_CRU'
                    process_w5e5_data(gdir, climate_type=dataset,
                                      temporal_resol='monthly')
                gd_mb = TIModel(gdir, 200, mb_type=mb_type, grad_type=grad_type,
                                prcp_fac=pf, input_filesuffix=fs,
                                baseline_climate=climate)

                spec_mb_annually = gd_mb.get_specific_mb(heights=h, widths=w,
                                                         year=np.arange(1980,
                                                                        2019))

                # check if annual and monthly mass balance are the same
                ann_mb = gd_mb.get_annual_mb(heights=h, year=2015)
                mon_mb_sum = 0
                for m in np.arange(1, 13):
                    mon_mb_sum += gd_mb.get_monthly_mb(heights=h,
                                                       year=date_to_floatyear(
                                                           2015, m))
                np.testing.assert_allclose(mon_mb_sum / 12, ann_mb, rtol=1e-4)

                if mb_type == 'mb_real_daily':

                    # test if the climate output of monthly
                    # and daily/annual is the same
                    # by just looking at the first month
                    clim_ann = gd_mb._get_2d_annual_climate(h, 2015)
                    clim_mon = gd_mb._get_2d_monthly_climate(h, 2015.0)
                    T_mon_from_ann = clim_ann[0][:, :np.shape(clim_mon[0])[1]]
                    np.testing.assert_allclose(T_mon_from_ann, clim_mon[0],
                                               rtol=1e-6)
                    Tfmelt_mon_from_ann = clim_ann[1][:,
                                          :np.shape(clim_mon[1])[1]]
                    np.testing.assert_allclose(Tfmelt_mon_from_ann,
                                               clim_mon[1], rtol=1e-6)
                    prcp_mon_from_ann = clim_ann[2][:,
                                        :np.shape(clim_mon[2])[1]]
                    np.testing.assert_allclose(prcp_mon_from_ann,
                                               clim_mon[2], rtol=1e-6)
                    prcp_mon_from_ann = clim_ann[2][:,
                                        :np.shape(clim_mon[2])[1]]
                    np.testing.assert_allclose(prcp_mon_from_ann,
                                               clim_mon[2], rtol=1e-6)
                    solidprcp_mon_from_ann = clim_ann[3][:,
                                             :np.shape(clim_mon[3])[1]]
                    np.testing.assert_allclose(solidprcp_mon_from_ann,
                                               clim_mon[3], rtol=1e-6)
                    # check if sum of monthly is annual
                    tfm_m_l = []
                    for m in np.arange(1, 12.1, 1):
                        floatyr = utils.date_to_floatyear(2015, m)
                        clim_mon = gd_mb._get_2d_monthly_climate(h, floatyr)
                        tfm_m_l.append(clim_mon[1].sum(axis=1))
                    tfm_y_via_m = np.array(tfm_m_l).sum(axis=0)
                    np.testing.assert_allclose(clim_ann[1].sum(axis=1), tfm_y_via_m)

                    # check if daily and annual mass balance are the same
                    # problem:
                    # as get_daily_mb accounts as well for leap years,
                    # doy are not 365.25 as in get_annual_mb but the amount
                    # of days of the year in reality!!!
                    # (needed for hydro model of Sarah Hanus)
                    # therefore we increased here rtol !!!
                    # but why are the differences in spec_mb so large this so large?
                    # @Fabi: can we do something about that?
                    day_mb = gd_mb.get_daily_mb(heights=h, year=2015)
                    day_mb_yearly_sum = []
                    for mb in day_mb:
                        day_mb_yearly_sum.append(mb.mean())
                    # problem: day_mb_yearly_sum is a bit smaller than ann_mb
                    np.testing.assert_allclose(day_mb_yearly_sum, ann_mb,
                                               rtol=1e-4)  # would need 2e-2)

                    # check if daily and yearly specific mb are the same?
                    spec_mb_daily = gd_mb.get_specific_daily_mb(heights=h,
                                                                widths=w,
                                                                year=np.arange(
                                                                    1980, 2019))
                    spec_mb_daily_yearly_sum = []
                    for mb in spec_mb_daily:
                        spec_mb_daily_yearly_sum.append(mb.sum())
                    np.testing.assert_allclose(spec_mb_daily_yearly_sum,
                                               spec_mb_annually,
                                               rtol=1e-4)  # would need 5e-2


    def test_melt_f_calib_geod_prep_inversion(self):

        # choose glaciers where we can not find a melt_f easily
        # all except for the last one do not work without
        # correcting the reference height
        cfg.initialize()
        cfg.PARAMS['use_multiprocessing'] = False
        cfg.PARAMS['hydro_month_nh'] = 1
        grad_type = 'cte'
        mb_type = 'mb_real_daily'
        climate_type = 'W5E5'  # W5E5
        temporal_resol = 'daily'
        # climate dataset goes till end of 2019
        ye = 2020
        pf = 2

        # use elevation band  flowlines
        base_url = ('https://cluster.klima.uni-bremen.de/~oggm/gdirs/oggm_v1.4/'
                    'L1-L2_files/elev_bands')


        test_dir = '/home/lilianschuster/Schreibtisch/PhD/oggm_files/MBsandbox_tests'
        if not os.path.exists(test_dir):
            test_dir = utils.gettempdir(dirname='OGGM_MBsandbox_test',
                                        reset=True)

        cfg.PATHS['working_dir'] = test_dir
        dfi =  ['RGI60-14.16678',
                'RGI60-14.08183',
                'RGI60-14.02796',
                'RGI60-14.08190']
        gdirs = workflow.init_glacier_directories(dfi,
                                                  from_prepro_level=2,
                                                  prepro_border=10,
                                                  prepro_base_url=base_url,
                                                  prepro_rgi_version='62')

        workflow.execute_entity_task(tasks.compute_downstream_line, gdirs)
        workflow.execute_entity_task(tasks.compute_downstream_bedshape, gdirs)

        for gdiri in gdirs:
            process_w5e5_data(gdiri, temporal_resol=temporal_resol,
                              climate_type=climate_type)
            melt_f_calib_geod_prep_inversion(gdiri, pf = pf,  # precipitation factor
                                             mb_type = mb_type,
                                             grad_type = grad_type,
                                             climate_type = climate_type,
                                             residual = 0,
                                             ye = ye)

            fs1 = '_{}_{}'.format(temporal_resol, climate_type)
            fpath = gdiri.get_filepath('climate_historical', filesuffix=fs1)
            with utils.ncDataset(fpath, 'a') as nc:
                corrected_ref_hgt = nc.ref_hgt
                uncorrected_ref_hgt = nc.uncorrected_ref_hgt

            fs = '_{}_{}_{}'.format(climate_type, mb_type, grad_type)
            d = gdiri.read_json(filename='melt_f_geod', filesuffix=fs)
            assert d['melt_f_pf_{}'.format(np.round(pf, 2))] > 10
            assert d['melt_f_pf_{}'.format(np.round(pf, 2))] < 1000
            # check if the historical climate file altitude is corrected as
            # we expect it
            # e.g. if no corrections, should be zero!
            assert_allclose(d['ref_hgt_calib_diff'],
                            corrected_ref_hgt - uncorrected_ref_hgt)

            if d['ref_hgt_calib_diff'] < 0:
                # if the climate was too warm for the observations
                # the melt_f  should be rather low (somewhere near 10)
                assert d['melt_f_pf_{}'.format(np.round(pf, 2))] < 100
                # the climate gridpoint altitude should be lower than
                # before the correction
                assert corrected_ref_hgt < uncorrected_ref_hgt
            elif d['ref_hgt_calib_diff'] > 0:
                # if the climate was too warm for the observations
                # the melt_f should be rather high (somewhere below 1000)
                assert d['melt_f_pf_{}'.format(np.round(pf, 2))] > 900
                # the climate gridpoint altitude should be higher than
                # before the correction
                assert corrected_ref_hgt > uncorrected_ref_hgt

# start it again to have the default hydro_month
class Test_directobs_hydro10:
    @pytest.mark.no_w5e5
    def test_minimize_bias(self, gdir):

        cfg.PARAMS['hydro_month_nh'] = 10
        # important to initialize again, otherwise hydro_month_nh=1
        # therefore gdir is included here
        # from test_hydro_years_HEF...
        # just checks if minimisation gives always same results
        grad_type = 'cte'
        N = 100
        for mb_type in ['mb_real_daily', 'mb_monthly', 'mb_pseudo_daily']:
            if mb_type != 'mb_real_daily':
                baseline_climate = 'ERA5dr'
                cfg.PARAMS['baseline_climate'] = baseline_climate #'ERA5dr'
                oggm.shop.ecmwf.process_ecmwf_data(gdir,
                                                   output_filesuffix='_monthly_ERA5dr',
                                                   )
            else:
                baseline_climate = 'ERA5'
                cfg.PARAMS['baseline_climate'] = 'ERA5_daily'
                process_era5_daily_data(gdir, output_filesuffix='_daily_ERA5')

            gd_mb = TIModel(gdir, None, mb_type=mb_type, N=N,
                            grad_type=grad_type, baseline_climate=baseline_climate)
            if mb_type == 'mb_real_daily':
                input_filesuffix = '_daily_ERA5'
            else:
                input_filesuffix = '_monthly_ERA5dr'
            melt_f_opt = scipy.optimize.brentq(minimize_bias, 10, 1000,
                                               disp=True, xtol=0.1,
                                                args=(gd_mb, gdir,
                                                  pf, False, input_filesuffix))

            hgts, widths = gdir.get_inversion_flowline_hw()
            mbdf = gdir.get_ref_mb_data(input_filesuffix=input_filesuffix)
            # check if they give the same optimal DDF
            np.testing.assert_allclose(np.round(mu_star_opt_cte[mb_type]/melt_f_opt, 4),
                                       1, rtol=1e-3)

            gd_mb.melt_f = melt_f_opt
            gd_mb.historical_climate_qc_mod(gdir)

            mb_specific = gd_mb.get_specific_mb(heights=hgts, widths=widths,
                                                year=mbdf.index.values)

            RMSD, bias, rcor, quot_std = compute_stat(mb_specific=mb_specific,
                                                      mbdf=mbdf)

            # check if the bias is optimised
            assert bias.round() == 0

    @pytest.mark.no_w5e5
    def test_optimize_std_quot_brentq_ERA5dr(self, gdir):
        # check if double optimisation of bias and std_quotient works

        cfg.PARAMS['hydro_month_nh'] = 10
        grad_type = 'cte'
        N = 100
        for mb_type in ['mb_real_daily', 'mb_monthly', 'mb_pseudo_daily']:
            if mb_type != 'mb_real_daily':
                baseline_climate='ERA5dr'
                cfg.PARAMS['baseline_climate'] = baseline_climate #'ERA5dr'
                oggm.shop.ecmwf.process_ecmwf_data(gdir,
                                                   output_filesuffix='_monthly_ERA5dr',
                                                   )
            else:
                baseline_climate='ERA5'
                cfg.PARAMS['baseline_climate'] = 'ERA5_daily'
                process_era5_daily_data(gdir, output_filesuffix='_daily_ERA5')

            gd_mb = TIModel(gdir, None, mb_type=mb_type, N=N, prcp_fac=1,
                            grad_type=grad_type, baseline_climate=baseline_climate)
            if mb_type == 'mb_real_daily':
                input_filesuffix = '_daily_ERA5'
            else:
                input_filesuffix = '_monthly_ERA5dr'

            hgts, widths = gdir.get_inversion_flowline_hw()
            mbdf = gdir.get_ref_mb_data(input_filesuffix=input_filesuffix)
            pf_opt = scipy.optimize.brentq(optimize_std_quot_brentq, 0.01, 20,
                                           args=(gd_mb, gdir, input_filesuffix),
                                           xtol=0.01)

            melt_f_opt_pf = scipy.optimize.brentq(minimize_bias, 1, 10000,
                                               disp=True, xtol=0.1,
                                               args=(gd_mb, gdir,
                                                     pf_opt, False, input_filesuffix))
            gd_mb.melt_f = melt_f_opt_pf
            gd_mb.prcp_fac = pf_opt
            gd_mb.historical_climate_qc_mod(gdir)
            mb_specific = gd_mb.get_specific_mb(heights=hgts, widths=widths,
                                                year=mbdf.index.values)

            RMSD, bias, rcor, quot_std = compute_stat(mb_specific=mb_specific,
                                                      mbdf=mbdf)

            # check if the bias is optimised
            assert bias.round() == 0
            # check if the std_quotient is optimised
            assert quot_std.round(1) == 1


    def test_TIModel_monthly(self, gdir):
        # check if massbalance.PastMassBalance equal to TIModel with cte
        # gradient and mb_monthly as options for lapse rate mb_type
        cfg.PARAMS['hydro_month_nh'] = 10
        mu_star_opt_cte_var = 195.5484547754791
        # if I use ERA5dr in PastMassBalance, it applies automatically the
        # gradient that changes with time and location
        mu_opts = [mu_star_opt_cte['mb_monthly'], mu_star_opt_cte_var]
        grads = ['cte', 'var']
        for k, clim in zip([0, 1], ['ERA5', 'ERA5dr']):
            mu_opt = mu_opts[k]
            grad_type = grads[k]
            cfg.PARAMS['baseline_climate'] = clim
            fs = '_monthly_{}'.format(clim)
            oggm.shop.ecmwf.process_ecmwf_data(gdir, dataset=clim,
                                               output_filesuffix=fs)

            mb_mod = TIModel(gdir, mu_opt, mb_type='mb_monthly',
                             prcp_fac=2.5, t_solid=0, t_liq=2, t_melt=0,
                             default_grad=-0.0065, grad_type=grad_type)

            hgts, widths = gdir.get_inversion_flowline_hw()
            mbdf = gdir.get_ref_mb_data(input_filesuffix=fs)
            mb_mod.historical_climate_qc_mod(gdir)
            tot_mb = mb_mod.get_specific_mb(heights=hgts, widths=widths,
                                            year=mbdf.index.values)

            cfg.PARAMS['temp_default_gradient'] = -0.0065
            cfg.PARAMS['prcp_scaling_factor'] = 2.5
            cfg.PARAMS['temp_all_solid'] = 0
            cfg.PARAMS['temp_all_liq'] = 2
            cfg.PARAMS['temp_melt'] = 0

            # check if the default OGGM monthly mass balance with cte gradient
            # gives the same result as the new TIModel with the options
            # mb_monthly and constant lapse rate gradient!
            mb_mod_def = massbalance.PastMassBalance(gdir, mu_star=mu_opt,
                                                     bias=0, input_filesuffix=fs,
                                                     check_calib_params=False)

            tot_mb_default = mb_mod_def.get_specific_mb(heights=hgts,
                                                        widths=widths,
                                                        year=mbdf.index.values)

            assert_allclose(tot_mb, tot_mb_default, rtol=1e-4)

    # I use here the exact same names as in test_models from OGGM.
    # but somehow the test for ref_mb_profile() is not equal
    @pytest.mark.no_w5e5
    def test_present_time_glacier_massbalance(self, gdir):

        cfg.PARAMS['hydro_month_nh'] = 10
        # check if area of  HUSS flowlines corresponds to the rgi area
        N = 100
        h, w = gdir.get_inversion_flowline_hw()
        fls = gdir.read_pickle('inversion_flowlines')
        assert_allclose(gdir.rgi_area_m2, np.sum(w * gdir.grid.dx * fls[0].dx))

        # do this for all model types
        # ONLY TEST it for ERA5dr or ERA5_daily!!!
        for climate in ['ERA5dr', 'ERA5_daily']:
            for mb_type in ['mb_monthly', 'mb_pseudo_daily', 'mb_real_daily']:
                for grad_type in ['cte', 'var_an_cycle']:
                    if grad_type == 'var_an_cycle':
                        mu_star_opt = mu_star_opt_var
                    else:
                        mu_star_opt = mu_star_opt_cte
                    if climate == 'ERA5dr':
                        cfg.PARAMS['baseline_climate'] = climate
                        fs = '_monthly_ERA5dr'
                        oggm.shop.ecmwf.process_ecmwf_data(gdir,
                                                           dataset=climate,
                                                           output_filesuffix=fs)
                    elif climate == 'ERA5_daily':
                        cfg.PARAMS['baseline_climate'] = climate
                        fs = '_daily_ERA5_daily'
                        process_era5_daily_data(gdir, output_filesuffix=fs)
                    else:
                        tasks.process_climate_data(gdir)
                        pass
                    fail_err_1 = ((mb_type == 'mb_pseudo_daily') and
                                  (climate != 'ERA5dr'))
                    fail_err_2 = ((mb_type == 'mb_monthly') and
                                  (climate == 'ERA5_daily'))
                    fail_err_3 = ((mb_type == 'mb_real_daily') and
                                  (climate == 'ERA5dr'))

                    if fail_err_1 or fail_err_2 or fail_err_3:
                        print(fail_err_1 or fail_err_2 or fail_err_3)
                        with pytest.raises(InvalidParamsError):
                            TIModel(gdir, mu_star_opt[mb_type],
                                    mb_type=mb_type, prcp_fac=pf,
                                    t_solid=0, t_liq=2, t_melt=0,
                                    default_grad=-0.0065,
                                    grad_type=grad_type,
                                    baseline_climate=climate)
                    else:
                        # this is just a test for reproducibility!
                        mb_mod = TIModel(gdir, mu_star_opt[mb_type],
                                         mb_type=mb_type, prcp_fac=pf,
                                         t_solid=0, t_liq=2, t_melt=0,
                                         default_grad=-0.0065,
                                         grad_type=grad_type, N=N,
                                         baseline_climate=climate)
                        # check climate and adapt if necessary
                        mb_mod.historical_climate_qc_mod(gdir)
                        mbdf = gdir.get_ref_mb_data(input_filesuffix=fs)
                        hgts, widths = gdir.get_inversion_flowline_hw()

                        tot_mb = []
                        refmb = []
                        grads = hgts * 0
                        for yr, mb in mbdf.iterrows():
                            refmb.append(mb['ANNUAL_BALANCE'])
                            mbh = (mb_mod.get_annual_mb(hgts, yr) *
                                   SEC_IN_YEAR * cfg.PARAMS['ice_density'])
                            grads += mbh
                            tot_mb.append(np.average(mbh, weights=widths))
                        grads /= len(tot_mb)

                        # check if calibrated total mass balance similar
                        # to observe mass balance time series
                        assert np.abs(utils.md(tot_mb, refmb)) < 50

                        # Gradient THIS GIVES an error!!!
                        # possibly because I use the HUSS flowlines ...
                        # or is it because I use another calibration?
                        # dfg = gdir.get_ref_mb_profile().mean()

                        # Take the altitudes below 3100 and fit a line
                        # dfg = dfg[dfg.index < 3100]
                        # pok = np.where(hgts < 3100)
                        # from scipy.stats import linregress
                        # slope_obs, _, _, _, _ = linregress(dfg.index,
                        #                                   dfg.values)
                        # slope_our, _, _, _, _ = linregress(hgts[pok],
                        #                                   grads[pok])
                        # np.testing.assert_allclose(slope_obs, slope_our,
                        #                           rtol=0.15)
                        # 0.15 does not work

    @pytest.mark.no_w5e5
    def test_monthly_glacier_massbalance(self, gdir):
        # I think there is a problem with SEC_IN_MONTH/SEC_IN_YEAR ...
        # do this for all model types
        # ONLY TEST it for ERA5dr or ERA5_daily!!!

        cfg.PARAMS['hydro_month_nh'] = 10
        N = 100
        for climate in ['ERA5dr', 'ERA5_daily']:
            for mb_type in ['mb_monthly', 'mb_pseudo_daily', 'mb_real_daily']:
                for grad_type in ['cte', 'var_an_cycle']:


                    if grad_type == 'var_an_cycle':
                        mu_star_opt = mu_star_opt_var
                    else:
                        mu_star_opt = mu_star_opt_cte
                    if climate == 'ERA5dr':
                        cfg.PARAMS['baseline_climate'] = climate
                        fs = '_monthly_ERA5dr'
                        oggm.shop.ecmwf.process_ecmwf_data(gdir,
                                                           dataset=climate,
                                                           output_filesuffix=fs)
                    elif climate == 'ERA5_daily':
                        cfg.PARAMS['baseline_climate'] = climate
                        fs = '_daily_ERA5_daily'
                        process_era5_daily_data(gdir, output_filesuffix=fs)
                    else:
                        tasks.process_climate_data(gdir)
                        pass

                    fail_err_1 = ((mb_type == 'mb_pseudo_daily') and
                                  (climate != 'ERA5dr'))
                    fail_err_2 = ((mb_type == 'mb_monthly') and
                                  (climate == 'ERA5_daily'))
                    fail_err_3 = ((mb_type == 'mb_real_daily') and
                                  (climate != 'ERA5_daily'))

                    if fail_err_1 or fail_err_2 or fail_err_3:
                        with pytest.raises(InvalidParamsError):
                            TIModel(gdir, mu_star_opt[mb_type],
                                     mb_type=mb_type, prcp_fac=pf,
                                     t_solid=0, t_liq=2, t_melt=0,
                                     default_grad=-0.0065,
                                     grad_type=grad_type,
                                     baseline_climate=climate)
                    else:
                        # but this is just a test for reproducibility!
                        mb_mod = TIModel(gdir, mu_star_opt[mb_type],
                                         mb_type=mb_type, prcp_fac=pf,
                                         t_solid=0, t_liq=2, t_melt=0,
                                         default_grad=-0.0065,
                                         grad_type=grad_type, N=N,
                                         baseline_climate=climate)
                        # check climate and adapt if necessary
                        mb_mod.historical_climate_qc_mod(gdir)
                        hgts, widths = gdir.get_inversion_flowline_hw()

                        rho = 900  # ice density
                        yrp = [1980, 2018]
                        for i, yr in enumerate(np.arange(yrp[0], yrp[1]+1)):
                            my_mon_mb_on_h = 0.
                            dayofyear = 0.
                            for m in np.arange(12):
                                yrm = utils.date_to_floatyear(yr, m + 1)
                                _, dayofmonth = monthrange(yr, m+1)
                                dayofyear += dayofmonth
                                tmp = (mb_mod.get_monthly_mb(hgts, yrm) *
                                       dayofmonth * SEC_IN_DAY * rho)
                                my_mon_mb_on_h += tmp
                            my_an_mb_on_h = (mb_mod.get_annual_mb(hgts, yr) *
                                             dayofyear * SEC_IN_DAY * rho)

                            # these errors come from the problematic of
                            # different amount of days in a year
                            assert_allclose(np.mean(my_an_mb_on_h -
                                                    my_mon_mb_on_h),
                                            0, atol=50)

    def test_loop(self, gdir):
        # tests whether ERA5dr works better with or without loop in mb_pseudo_daily
        # tests that both option give same results and in case that default
        # option (no loop) is 30% slower, it raises an error

        # this could be optimised and included in the above tests
        # cfg.initialize()

        cfg.PARAMS['hydro_month_nh'] = 10

        climate = 'ERA5dr'
        mb_type = 'mb_pseudo_daily'
        cfg.PARAMS['baseline_climate'] = climate
        fs = '_monthly_ERA5dr'
        oggm.shop.ecmwf.process_ecmwf_data(gdir, dataset=climate, output_filesuffix=fs)

        for grad_type in ['cte', 'var_an_cycle']:
            if grad_type == 'var_an_cycle':
                fail_err_4 = (mb_type == 'mb_monthly') and (climate == 'CRU')
                mu_star_opt = mu_star_opt_var
            else:
                fail_err_4 = False
                mu_star_opt = mu_star_opt_cte

            if fail_err_4:
                with pytest.raises(InvalidParamsError):
                    TIModel(gdir, mu_star_opt[mb_type],
                            mb_type=mb_type, prcp_fac=pf,
                            t_solid=0, t_liq=2, t_melt=0,
                            default_grad=-0.0065,
                            grad_type=grad_type,
                            baseline_climate=climate)
            else:
                mbdf = gdir.get_ref_mb_data(input_filesuffix=fs)
                ys = mbdf.index.values

                hgts, widths = gdir.get_inversion_flowline_hw()

                ex_t = time.time()
                mb_mod_noloop = TIModel(gdir, mu_star_opt[mb_type],
                                        mb_type=mb_type, prcp_fac=pf,
                                        loop=False,
                                        t_solid=0, t_liq=2, t_melt=0,
                                        default_grad=-0.0065,
                                        grad_type=grad_type,
                                        baseline_climate=climate)
                # check climate and adapt if necessary
                mb_mod_noloop.historical_climate_qc_mod(gdir)
                for t in np.arange(10):
                    totmbnoloop = mb_mod_noloop.get_specific_mb(heights=hgts,
                                                                widths=widths,
                                                                year=ys)
                ex_noloop = time.time() - ex_t

                ex_t = time.time()
                mb_mod_loop = TIModel(gdir, mu_star_opt[mb_type],
                                      mb_type=mb_type, prcp_fac=pf,
                                      loop=True, t_solid=0, t_liq=2,
                                      t_melt=0, default_grad=-0.0065,
                                      grad_type=grad_type,
                                      baseline_climate=climate)
                # check climate and adapt if necessary
                mb_mod_loop.historical_climate_qc_mod(gdir)
                for t in np.arange(10):
                    tot_mb_loop = mb_mod_loop.get_specific_mb(heights=hgts,
                                                              widths=widths,
                                                              year=ys)
                ex_loop = time.time() - ex_t

                # both should give the same results!!!
                assert_allclose(tot_mb_loop, totmbnoloop,
                                atol=1e-2)

                # if the loop would be at least 30%faster
                # than not using the loop raise an error
                assert (ex_loop-ex_noloop)/ex_noloop > -0.3
                # but in the moment loop is sometimes faster, why? !!!
                # actually it depends which one I test first,
                # the one that runs first
                # is actually fast, so when running noloop first
                # it is around 5% faster

    # %%

    def test_N(self, gdir):
        # tests whether modelled mb_pseudo_daily massbalances of different values of N
        # is similar to observed mass balances

        # this could be optimised and included in the above tests

        cfg.PARAMS['hydro_month_nh'] = 10
        climate = 'ERA5dr'
        mb_type = 'mb_pseudo_daily'
        cfg.PARAMS['baseline_climate'] = 'ERA5dr'
        fs = '_monthly_ERA5dr'
        oggm.shop.ecmwf.process_ecmwf_data(gdir, dataset=climate,
                                           output_filesuffix=fs)

        for grad_type in ['cte', 'var_an_cycle']:
            if grad_type == 'var_an_cycle':
                mu_star_opt = mu_star_opt_var
            else:
                mu_star_opt = mu_star_opt_cte


            mbdf = gdir.get_ref_mb_data(input_filesuffix=fs)
            hgts, widths = gdir.get_inversion_flowline_hw()

            tot_mbN = {}
            for N in [1000, 500, 100, 50]:
                mb_mod = TIModel(gdir, mu_star_opt[mb_type],
                                 mb_type=mb_type,
                                 prcp_fac=pf, N=N,
                                 t_solid=0, t_liq=2, t_melt=0,
                                 default_grad=-0.0065,
                                 grad_type=grad_type,
                                 baseline_climate=climate)
                # check climate and adapt if necessary
                mb_mod.historical_climate_qc_mod(gdir)

                tot_mbN[N] = mb_mod.get_specific_mb(heights=hgts,
                                                    widths=widths,
                                                    year=mbdf.index.values)

                assert np.abs(utils.md(tot_mbN[N],
                                       mbdf['ANNUAL_BALANCE'])) < 10

    def test_prcp_fac_update(self, gdir):

        cfg.PARAMS['hydro_month_nh'] = 10
        cfg.PARAMS['baseline_climate'] = 'ERA5dr'
        oggm.shop.ecmwf.process_ecmwf_data(gdir, dataset='ERA5dr',
                                           )
        gd_mb = TIModel(gdir, None, mb_type='mb_monthly', N=100, prcp_fac=2.5,
                        grad_type='cte', input_filesuffix='')
        assert gd_mb.prcp_fac == 2.5
        assert gd_mb._prcp_fac == 2.5
        prcp_old = gd_mb.prcp.copy()  # .mean()
        gd_mb.prcp_fac = 10
        assert gd_mb.prcp_fac == 10
        assert gd_mb._prcp_fac == 10
        prcp_old_regen = gd_mb.prcp * 2.5 / gd_mb.prcp_fac
        assert_allclose(prcp_old_regen, prcp_old)

        # print(gd_mb._prcp_fac)
        # print(gd_mb.prcp[0])
        gd_mb.prcp_fac = 2.5
        assert gd_mb.prcp_fac == 2.5
        assert gd_mb._prcp_fac == 2.5
        assert_allclose(gd_mb.prcp, prcp_old)
        with pytest.raises(InvalidParamsError):
            gd_mb.prcp_fac = 0
        with pytest.raises(InvalidParamsError):
            TIModel(gdir, None, mb_type='mb_monthly', prcp_fac=-1,
                    grad_type='cte', input_filesuffix='')

    @pytest.mark.no_w5e5
    def test_historical_climate_qc_mon(self, gdir):

        cfg.PARAMS['hydro_month_nh'] = 10
        h, w = gdir.get_inversion_flowline_hw()
        for mb_type in ['mb_monthly', 'mb_pseudo_daily', 'mb_real_daily']:
            for grad_type in ['cte', 'var_an_cycle']:
                if mb_type == 'mb_real_daily':
                    climate = 'ERA5_daily'
                    cfg.PARAMS['baseline_climate'] = climate
                    fs = '_daily_ERA5_daily'
                    process_era5_daily_data(gdir, output_filesuffix=fs)
                    fc = gdir.get_filepath('climate_historical', filesuffix=fs)
                else:
                    climate = 'ERA5dr'
                    cfg.PARAMS['baseline_climate'] = climate
                    fs = '_monthly_ERA5dr'
                    oggm.shop.ecmwf.process_ecmwf_data(gdir, dataset="ERA5dr",
                                                       output_filesuffix=fs)
                    # Raise ref hgt a lot
                    fc = gdir.get_filepath('climate_historical', filesuffix=fs)
                with utils.ncDataset(fc, 'a') as nc:
                    nc.ref_hgt = 10000

                mb = TIModel(gdir, 200,
                             mb_type=mb_type,
                             prcp_fac=pf,
                             t_solid=0, t_liq=2, t_melt=0,
                             default_grad=-0.0065,
                             grad_type=grad_type,
                             baseline_climate=climate)
                ref_hgt_0 = mb.uncorrected_ref_hgt
                mb.historical_climate_qc_mod(gdir)
                ref_uncorrected = mb.uncorrected_ref_hgt
                mbdf = gdir.get_ref_mb_data(input_filesuffix=fs)
                ref_hgt_1 = mb.ref_hgt
                assert (ref_hgt_1 - ref_hgt_0) < -4000
                with utils.ncDataset(fc, 'r') as nc:
                    assert (nc.ref_hgt - nc.uncorrected_ref_hgt) < -4000

                melt_f_opt = scipy.optimize.brentq(minimize_bias, 1, 10000,
                                                   disp=True, xtol=0.1,
                                                   args=(mb, gdir,
                                                         pf, False, fs))

                mb.melt_f = melt_f_opt
                mbdf['CALIB_1'] = mb.get_specific_mb(heights=h,
                                                     widths=w,
                                                     year=mbdf.index.values)

                # Lower ref hgt a lot
                with utils.ncDataset(fc, 'a') as nc:
                    nc.ref_hgt = 0
                    # as we change here the ref_hgt manually,
                    # we also have to reset the uncorrected ref hgt
                    nc.uncorrected_ref_hgt = 0
                mb = TIModel(gdir, 200,
                             mb_type=mb_type,
                             prcp_fac=pf,
                             t_solid=0, t_liq=2, t_melt=0,
                             default_grad=-0.0065,
                             grad_type=grad_type,
                             baseline_climate=climate)
                ref_hgt_0 = mb.uncorrected_ref_hgt
                mb.historical_climate_qc_mod(gdir)
                ref_hgt_1 = mb.ref_hgt
                assert (ref_hgt_1 - ref_hgt_0) > 1800
                with utils.ncDataset(fc, 'r') as nc:
                    assert (nc.ref_hgt - nc.uncorrected_ref_hgt) > 1800

                melt_f_opt = scipy.optimize.brentq(minimize_bias, 1, 10000,
                                                   disp=True, xtol=0.1,
                                                   args=(mb, gdir,
                                                         pf, False, fs))
                mb.melt_f = melt_f_opt
                mbdf['CALIB_2'] = mb.get_specific_mb(heights=h,
                                                     widths=w,
                                                     year=mbdf.index.values)

                mm = mbdf[['ANNUAL_BALANCE', 'CALIB_1',
                           'CALIB_2']].mean()
                np.testing.assert_allclose(mm['ANNUAL_BALANCE'],
                                           mm['CALIB_1'],
                                           rtol=1e-5)
                np.testing.assert_allclose(mm['ANNUAL_BALANCE'],
                                           mm['CALIB_2'],
                                           rtol=1e-5)

                cor = mbdf[['ANNUAL_BALANCE', 'CALIB_1', 'CALIB_2']].corr()
                assert cor.min().min() > 0.35

# cfg.initialize()

# test_dir = '/home/lilianschuster/Schreibtisch/PhD/oggm_files/MBsandbox_tests'
# if not os.path.exists(test_dir):
#     test_dir = utils.gettempdir(dirname='OGGM_MBsandbox_test',
#                                 reset=True)

# cfg.PATHS['working_dir'] = test_dir
# base_url = ('https://cluster.klima.uni-bremen.de/~oggm/gdirs/oggm_v1.4/'
#             'L1-L2_files/elev_bands')

# df = ['RGI60-11.00897']
# gdirs = workflow.init_glacier_directories(df, from_prepro_level=2,
#                                           prepro_border=10,
#                                           prepro_base_url=base_url,
#                                           prepro_rgi_version='62')
# gdir = gdirs[0]

# test_historical_climate_qc_mon(gdir)
# %%
# something like that could also be included later on
# bbut only if I somehow get the data from the climate files ....
# or can I use these things with the ELA... without the climate files...
# in the moment oggm.core.climate works only with default OGGM mass balance

    # def test_TIModel(self, hef_gdir):

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
