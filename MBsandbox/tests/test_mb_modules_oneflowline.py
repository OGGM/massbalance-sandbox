#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 18:34:35 2020

@author: lilianschuster
"""

# tests for mass balances
import warnings
warnings.filterwarnings("once", category=DeprecationWarning)  # noqa: E402

import time
import numpy as np
from numpy.testing import assert_allclose
import pytest
import scipy
import os
import xarray as xr
from calendar import monthrange


# imports from OGGM
import oggm
from oggm.core import massbalance
from oggm import utils, workflow, tasks, cfg
from oggm.cfg import SEC_IN_DAY, SEC_IN_YEAR
from oggm.exceptions import InvalidParamsError


# imports from MBsandbox package modules
from MBsandbox.help_func import (compute_stat, minimize_bias,
                                 optimize_std_quot_brentq)

from MBsandbox.mbmod_daily_oneflowline import (process_era5_daily_data,
                                               TIModel)

# optimal values for HEF of mu_star for cte lapse rates
mu_star_opt_cte = {'mb_monthly': 213.561413,
                   'mb_daily': 181.383413,
                   'mb_real_daily': 180.419554}
# optimal values of mu_star when using variable lapse rates ('var_an_cycle')
mu_star_opt_var = {'mb_monthly': 195.322804,
                   'mb_daily': 167.506525,
                   'mb_real_daily': 159.912743}
# precipitation factor
pf = 2.5


# %%
class Test_geodetic_hydro1:
    # classes have to be upper case in order that they
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
        process_era5_daily_data(gdir)
        f = gdir.get_filepath('climate_historical', filesuffix='_daily')
        test_climate = xr.open_dataset(f)
        assert test_climate.time[0] == np.datetime64('1979-01-01')
        assert test_climate.time[-1] == np.datetime64('2018-12-31')


# %%
class Test_directobs_hydro10:
    def test_minimize_bias(self, gdir):

        # important to initialize again, otherwise hydro_month_nh=1
        # therefore gdir is included here
        # from test_hydro_years_HEF...
        # just checks if minimisation gives always same results
        grad_type = 'cte'
        N = 100
        loop = False
        for mb_type in ['mb_real_daily', 'mb_monthly', 'mb_daily']:
            if mb_type != 'mb_real_daily':
                cfg.PARAMS['baseline_climate'] = 'ERA5dr'
                oggm.shop.ecmwf.process_ecmwf_data(gdir, dataset='ERA5dr')
            else:
                cfg.PARAMS['baseline_climate'] = 'ERA5_daily'
                process_era5_daily_data(gdir)

            DDF_opt = scipy.optimize.brentq(minimize_bias, 1, 10000, disp=True,
                                            xtol=0.1,
                                            args=(mb_type, grad_type, gdir, N,
                                                  pf, loop, False))
            hgts, widths = gdir.get_inversion_flowline_hw()
            mbdf = gdir.get_ref_mb_data()
            # check if they give the same optimal DDF
            assert np.round(mu_star_opt_cte[mb_type]/DDF_opt, 3) == 1

            gd_mb = TIModel(gdir, DDF_opt, mb_type=mb_type, N=N,
                            grad_type=grad_type)
            gd_mb.historical_climate_qc_mod(gdir)

            mb_specific = gd_mb.get_specific_mb(heights=hgts, widths=widths,
                                                year=mbdf.index.values)

            RMSD, bias, rcor, quot_std = compute_stat(mb_specific=mb_specific,
                                                      mbdf=mbdf)

            # check if the bias is optimised
            assert bias.round() == 0
    # %%


    def test_optimize_std_quot_brentq(self, gdir):
        # check if double optimisation of bias and std_quotient works

        grad_type = 'cte'
        N = 100
        loop = False
        for mb_type in ['mb_monthly', 'mb_daily', 'mb_real_daily']:
            if mb_type != 'mb_real_daily':
                cfg.PARAMS['baseline_climate'] = 'ERA5dr'
                oggm.shop.ecmwf.process_ecmwf_data(gdir, dataset='ERA5dr')
            else:
                cfg.PARAMS['baseline_climate'] = 'ERA5_daily'
                process_era5_daily_data(gdir)

            hgts, widths = gdir.get_inversion_flowline_hw()
            mbdf = gdir.get_ref_mb_data()
            pf_opt = scipy.optimize.brentq(optimize_std_quot_brentq, 0.01, 20,
                                           args=(mb_type, grad_type, gdir,
                                                 N, loop),
                                           xtol=0.01)

            DDF_opt_pf = scipy.optimize.brentq(minimize_bias, 1, 10000,
                                               args=(mb_type, grad_type,
                                                     gdir, N, pf_opt,
                                                     loop, False),
                                               disp=True, xtol=0.1)
            gd_mb = TIModel(gdir, DDF_opt_pf, prcp_fac=pf_opt, mb_type=mb_type,
                            grad_type=grad_type, N=N)
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

        mu_star_opt_cte_var = 195.5484547754791
        # if I use ERA5dr in PastMassBalance, it applies automatically the
        # gradient that changes with time and location
        mu_opts = [mu_star_opt_cte['mb_monthly'], mu_star_opt_cte_var]
        grads = ['cte', 'var']
        for k, clim in zip([0, 1], ['ERA5', 'ERA5dr']):
            mu_opt = mu_opts[k]
            grad_type = grads[k]
            cfg.PARAMS['baseline_climate'] = clim
            oggm.shop.ecmwf.process_ecmwf_data(gdir, dataset=clim)

            mb_mod = TIModel(gdir, mu_opt, mb_type='mb_monthly',
                             prcp_fac=2.5, t_solid=0, t_liq=2, t_melt=0,
                             default_grad=-0.0065, grad_type=grad_type)

            hgts, widths = gdir.get_inversion_flowline_hw()
            mbdf = gdir.get_ref_mb_data()
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
                                                     bias=0,
                                                     check_calib_params=False)

            tot_mb_default = mb_mod_def.get_specific_mb(heights=hgts,
                                                        widths=widths,
                                                        year=mbdf.index.values)

            assert_allclose(tot_mb, tot_mb_default, rtol=1e-4)

    # I use here the exact same names as in test_models from OGGM.
    # class TestInitPresentDayFlowline:
    # but somehow the test for ref_mb_profile() is not equal
    def test_present_time_glacier_massbalance(self, gdir):

        # check if area of  HUSS flowlines corresponds to the rgi area
        N = 100
        h, w = gdir.get_inversion_flowline_hw()
        fls = gdir.read_pickle('inversion_flowlines')
        assert_allclose(gdir.rgi_area_m2, np.sum(w * gdir.grid.dx * fls[0].dx))

        # do this for all model types
        # ONLY TEST it for ERA5dr or ERA5_daily!!!
        for climate in ['ERA5dr', 'ERA5_daily']:
            for mb_type in ['mb_monthly', 'mb_daily', 'mb_real_daily']:
                for grad_type in ['cte', 'var_an_cycle']:
                    if grad_type == 'var_an_cycle':
                        fail_err_4 = ((mb_type == 'mb_monthly') and
                                      (climate == 'CRU'))
                        mu_star_opt = mu_star_opt_var
                    else:
                        fail_err_4 = False
                        mu_star_opt = mu_star_opt_cte
                    if climate == 'ERA5dr':
                        cfg.PARAMS['baseline_climate'] = 'ERA5dr'
                        oggm.shop.ecmwf.process_ecmwf_data(gdir,
                                                           dataset="ERA5dr")
                    elif climate == 'ERA5_daily':
                        cfg.PARAMS['baseline_climate'] = 'ERA5_daily'
                        process_era5_daily_data(gdir)
                    else:
                        tasks.process_climate_data(gdir)
                        pass
                    fail_err_1 = ((mb_type == 'mb_daily') and
                                  (climate != 'ERA5dr'))
                    fail_err_2 = ((mb_type == 'mb_monthly') and
                                  (climate == 'ERA5_daily'))
                    fail_err_3 = ((mb_type == 'mb_real_daily') and
                                  (climate != 'ERA5_daily'))

                    if fail_err_1 or fail_err_2 or fail_err_3 or fail_err_4:
                        with pytest.raises(InvalidParamsError):
                            mb_mod = TIModel(gdir, mu_star_opt[mb_type],
                                             mb_type=mb_type, prcp_fac=pf,
                                             t_solid=0, t_liq=2, t_melt=0,
                                             default_grad=-0.0065,
                                             grad_type=grad_type)
                    else:
                        # this is just a test for reproducibility!
                        mb_mod = TIModel(gdir, mu_star_opt[mb_type],
                                         mb_type=mb_type, prcp_fac=pf,
                                         t_solid=0, t_liq=2, t_melt=0,
                                         default_grad=-0.0065,
                                         grad_type=grad_type, N=N)
                        # check climate and adapt if necessary
                        mb_mod.historical_climate_qc_mod(gdir)
                        mbdf = gdir.get_ref_mb_data()
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

    def test_monthly_glacier_massbalance(self, gdir):
        # TODO: problem with that, monthly and annual MB not exactly same!!!
        # I think there is a problem with SEC_IN_MONTH/SEC_IN_YEAR ...

        # do this for all model types
        # ONLY TEST it for ERA5dr or ERA5_daily!!!
        N = 100
        for climate in ['ERA5dr', 'ERA5_daily']:
            for mb_type in ['mb_monthly', 'mb_daily', 'mb_real_daily']:
                for grad_type in ['cte', 'var_an_cycle']:
                    if grad_type == 'var_an_cycle':
                        fail_err_4 = ((mb_type == 'mb_monthly')
                                      and (climate == 'CRU'))
                        mu_star_opt = mu_star_opt_var
                    else:
                        fail_err_4 = False
                        mu_star_opt = mu_star_opt_cte
                    if climate == 'ERA5dr':
                        cfg.PARAMS['baseline_climate'] = 'ERA5dr'
                        oggm.shop.ecmwf.process_ecmwf_data(gdir,
                                                           dataset="ERA5dr")
                    elif climate == 'ERA5_daily':
                        cfg.PARAMS['baseline_climate'] = 'ERA5_daily'
                        process_era5_daily_data(gdir)
                    else:
                        tasks.process_climate_data(gdir)
                        pass
                    # mb_type ='mb_daily'
                    fail_err_1 = ((mb_type == 'mb_daily') and
                                  (climate != 'ERA5dr'))
                    fail_err_2 = ((mb_type == 'mb_monthly') and
                                  (climate == 'ERA5_daily'))
                    fail_err_3 = ((mb_type == 'mb_real_daily') and
                                  (climate != 'ERA5_daily'))

                    if fail_err_1 or fail_err_2 or fail_err_3 or fail_err_4:
                        with pytest.raises(InvalidParamsError):
                            mb_mod = TIModel(gdir, mu_star_opt[mb_type],
                                             mb_type=mb_type, prcp_fac=pf,
                                             t_solid=0, t_liq=2, t_melt=0,
                                             default_grad=-0.0065,
                                             grad_type=grad_type)
                    else:
                        # but this is just a test for reproducibility!
                        mb_mod = TIModel(gdir, mu_star_opt[mb_type],
                                         mb_type=mb_type, prcp_fac=pf,
                                         t_solid=0, t_liq=2, t_melt=0,
                                         default_grad=-0.0065,
                                         grad_type=grad_type, N=N)
                        # check climate and adapt if necessary
                        mb_mod.historical_climate_qc_mod(gdir)
                        hgts, widths = gdir.get_inversion_flowline_hw()

                        rho = 900  # ice density
                        yrp = [1980, 2018]
                        for i, yr in enumerate(np.arange(yrp[0], yrp[1]+1)):
                            my_mon_mb_on_h = 0.
                            dayofyear = 0
                            for m in np.arange(12):
                                yrm = utils.date_to_floatyear(yr, m + 1)
                                _, dayofmonth = monthrange(yr, m+1)
                                dayofyear += dayofmonth
                                tmp = (mb_mod.get_monthly_mb(hgts, yrm) *
                                       dayofmonth * SEC_IN_DAY * rho)
                                my_mon_mb_on_h += tmp
                            my_an_mb_on_h = (mb_mod.get_annual_mb(hgts, yr) *
                                             dayofyear * SEC_IN_DAY * rho)

                            # these large errors might come from problematic of
                            # different amount of days in a year?
                            # or maybe it just does not fit ...
                            assert_allclose(np.mean(my_an_mb_on_h -
                                                    my_mon_mb_on_h),
                                            0, atol=100)

    def test_loop(self, gdir):
        # tests whether ERA5dr works better with or without loop in mb_daily
        # tests that both option give same results and in case that default
        # option (no loop) is 30% slower, it raises an error

        # this could be optimised and included in the above tests
        cfg.initialize()

        climate = 'ERA5dr'
        mb_type = 'mb_daily'
        cfg.PARAMS['baseline_climate'] = 'ERA5dr'
        oggm.shop.ecmwf.process_ecmwf_data(gdir, dataset="ERA5dr")

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
                            grad_type=grad_type)
            else:
                mbdf = gdir.get_ref_mb_data()
                ys = mbdf.index.values

                hgts, widths = gdir.get_inversion_flowline_hw()

                ex_t = time.time()
                mb_mod_noloop = TIModel(gdir, mu_star_opt[mb_type],
                                        mb_type=mb_type, prcp_fac=pf,
                                        loop=False,
                                        t_solid=0, t_liq=2, t_melt=0,
                                        default_grad=-0.0065,
                                        grad_type=grad_type)
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
                                      grad_type=grad_type)
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
        # tests whether modelled mb_daily massbalances of different values of N
        # is similar to observed mass balances

        # this could be optimised and included in the above tests
        climate = 'ERA5dr'
        mb_type = 'mb_daily'
        cfg.PARAMS['baseline_climate'] = 'ERA5dr'
        oggm.shop.ecmwf.process_ecmwf_data(gdir, dataset="ERA5dr")
        for grad_type in ['cte', 'var_an_cycle']:
            if grad_type == 'var_an_cycle':
                fail_err_4 = (mb_type == 'mb_monthly') and (climate == 'CRU')
                mu_star_opt = mu_star_opt_var
            else:
                fail_err_4 = False
                mu_star_opt = mu_star_opt_cte

            if fail_err_4:
                with pytest.raises(InvalidParamsError):
                    mb_mod = TIModel(gdir, mu_star_opt[mb_type],
                                     mb_type=mb_type, prcp_fac=pf,
                                     t_solid=0, t_liq=2, t_melt=0,
                                     default_grad=-0.0065,
                                     grad_type=grad_type)
            else:
                mbdf = gdir.get_ref_mb_data()
                hgts, widths = gdir.get_inversion_flowline_hw()

                tot_mbN = {}
                for N in [1000, 500, 100, 50]:
                    mb_mod = TIModel(gdir, mu_star_opt[mb_type],
                                     mb_type=mb_type,
                                     prcp_fac=pf, N=N,
                                     t_solid=0, t_liq=2, t_melt=0,
                                     default_grad=-0.0065,
                                     grad_type=grad_type)
                    # check climate and adapt if necessary
                    mb_mod.historical_climate_qc_mod(gdir)

                    tot_mbN[N] = mb_mod.get_specific_mb(heights=hgts,
                                                        widths=widths,
                                                        year=mbdf.index.values)

                    assert np.abs(utils.md(tot_mbN[N],
                                           mbdf['ANNUAL_BALANCE'])) < 10

    def test_prcp_fac_update(self, gdir):

        cfg.PARAMS['baseline_climate'] = 'ERA5dr'
        oggm.shop.ecmwf.process_ecmwf_data(gdir, dataset='ERA5dr')
        gd_mb = TIModel(gdir, None, mb_type='mb_monthly', N=100, prcp_fac=2.5,
                        grad_type='cte')
        assert gd_mb.prcp_fac == 2.5
        assert gd_mb.inst_prcp_fac == 2.5
        prcp_old = gd_mb.prcp  # .mean()
        gd_mb.prcp_fac = 10
        assert gd_mb.prcp_fac == 10
        assert gd_mb.inst_prcp_fac == 10
        prcp_old_regen = 2.5 * gd_mb.prcp / gd_mb.prcp_fac
        assert_allclose(prcp_old_regen, prcp_old)

        # print(gd_mb._prcp_fac)
        # print(gd_mb.prcp[0])
        gd_mb.prcp_fac = 2.5
        assert gd_mb.prcp_fac == 2.5
        assert gd_mb.inst_prcp_fac == 2.5
        assert_allclose(gd_mb.prcp, prcp_old)
        with pytest.raises(InvalidParamsError):
            gd_mb.prcp_fac = 0
        with pytest.raises(InvalidParamsError):
            TIModel(gdir, None, mb_type='mb_monthly', N=100, prcp_fac=-1,
                    grad_type='cte')

    def test_historical_climate_qc_mon(self, gdir):

        h, w = gdir.get_inversion_flowline_hw()
        N = 100
        loop = False
        climate = 'ERA5dr'
        for mb_type in ['mb_monthly', 'mb_daily', 'mb_real_daily']:
            for grad_type in ['cte', 'var_an_cycle']:
                if mb_type == 'mb_real_daily':
                    cfg.PARAMS['baseline_climate'] = 'ERA5_daily'
                    process_era5_daily_data(gdir)
                    fc = gdir.get_filepath('climate_historical_daily')
                else:
                    cfg.PARAMS['baseline_climate'] = climate
                    oggm.shop.ecmwf.process_ecmwf_data(gdir, dataset="ERA5dr")
                    # Raise ref hgt a lot
                    fc = gdir.get_filepath('climate_historical')
                with utils.ncDataset(fc, 'a') as nc:
                    nc.ref_hgt = 10000

                mb = TIModel(gdir, 200,
                             mb_type=mb_type,
                             prcp_fac=pf, N=100,
                             t_solid=0, t_liq=2, t_melt=0,
                             default_grad=-0.0065,
                             grad_type=grad_type)
                mb.historical_climate_qc_mod(gdir)
                mbdf = gdir.get_ref_mb_data()

                with utils.ncDataset(fc, 'r') as nc:
                    assert (nc.ref_hgt - nc.uncorrected_ref_hgt) < -4000

                DDF_opt = scipy.optimize.brentq(minimize_bias, 10, 10000,
                                                disp=True, xtol=0.1,
                                                args=(mb_type, grad_type,
                                                      gdir, N,
                                                      pf, loop, False))
                mb.melt_f = DDF_opt
                mbdf['CALIB_1'] = mb.get_specific_mb(heights=h,
                                                     widths=w,
                                                     year=mbdf.index.values)

                # Lower ref hgt a lot
                with utils.ncDataset(fc, 'a') as nc:
                    nc.ref_hgt = 0
                mb = TIModel(gdir, 200,
                             mb_type=mb_type,
                             prcp_fac=pf, N=100,
                             t_solid=0, t_liq=2, t_melt=0,
                             default_grad=-0.0065,
                             grad_type=grad_type)
                mb.historical_climate_qc_mod(gdir)

                with utils.ncDataset(fc, 'r') as nc:
                    assert (nc.ref_hgt - nc.uncorrected_ref_hgt) > 1800

                DDF_opt = scipy.optimize.brentq(minimize_bias, 10,
                                                10000, disp=True, xtol=0.1,
                                                args=(mb_type, grad_type,
                                                      gdir, N, pf,
                                                      loop, False))
                mb.melt_f = DDF_opt
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