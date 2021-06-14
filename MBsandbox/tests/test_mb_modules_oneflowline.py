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
import pandas as pd
from calendar import monthrange


# imports from OGGM
import oggm
from oggm.core import massbalance
from oggm import utils, workflow, tasks, cfg
from oggm.cfg import SEC_IN_DAY, SEC_IN_YEAR
from oggm.exceptions import InvalidParamsError
from oggm.utils import date_to_floatyear

from MBsandbox.wip.help_func_geodetic import minimize_bias_geodetic
# imports from MBsandbox package modules
from MBsandbox.help_func import (compute_stat, minimize_bias,
                                 optimize_std_quot_brentq)

from MBsandbox.mbmod_daily_oneflowline import (process_era5_daily_data,
                                               process_w5e5_data,
                                               TIModel, TIModel_Sfc_Type)

# optimal values for HEF of mu_star for cte lapse rates (for wgms direct MB)
mu_star_opt_cte = {'mb_monthly': 213.561413,
                   'mb_pseudo_daily': 181.383413,
                   'mb_real_daily': 180.419554}
# optimal values of mu_star when using variable lapse rates ('var_an_cycle')
mu_star_opt_var = {'mb_monthly': 195.322804,
                   'mb_pseudo_daily': 167.506525,
                   'mb_real_daily': 159.912743}
# precipitation factor
pf = 2.5


# %%
class Test_geodetic_sfc_type:
    def test_geodetic_fixed_var_melt_f(self, gdir):
        cfg.PARAMS['hydro_month_nh'] = 1
        # just choose any random melt_f
        melt_f = 200
        pf = 2.5  # precipitation factor
        df = ['RGI60-11.00897']


        cfg.PARAMS['baseline_climate'] = 'ERA5dr'
        oggm.shop.ecmwf.process_ecmwf_data(gdir, dataset='ERA5dr',
                                           output_filesuffix='_monthly_ERA5dr',
                                           )
        #
        mb_mod_1 = TIModel_Sfc_Type(gdir, melt_f, mb_type='mb_monthly',
                                  melt_f_ratio_snow_to_ice=1, prcp_fac=pf,
                                    )

        mb_mod_no_sfc_type = TIModel(gdir, melt_f, mb_type='mb_monthly',
                                     prcp_fac=pf)

        h, w = gdir.get_inversion_flowline_hw()
        year = 2000
        _, temp2dformelt_1, _, prcpsol_1 = mb_mod_1._get_2d_annual_climate(h, year)
        _, temp2dformelt_no_sfc_type, _, prcpsol_no_sfc_type = mb_mod_no_sfc_type._get_2d_annual_climate(h, year)

        assert_allclose(temp2dformelt_1, temp2dformelt_no_sfc_type)
        assert_allclose(prcpsol_1, prcpsol_no_sfc_type)

        mb_annual_1 = mb_mod_1.get_annual_mb(h, year=2000)
        mb_annual_no_sfc_type = mb_mod_no_sfc_type.get_annual_mb(h, year=2000)

        assert_allclose(mb_annual_1, mb_annual_no_sfc_type)

        # check if specific mass balance equal?
        # use the years from geodetic data
        # TODO: change this to 2020 when available
        years = np.arange(2000, 2019)
        fls = gdir.read_pickle('inversion_flowlines')
        spec_1 = mb_mod_1.get_specific_mb(year=years, fls=fls)
        spec_no_sfc_type = mb_mod_no_sfc_type.get_specific_mb(year=years, fls=fls)

        assert_allclose(spec_1, spec_no_sfc_type)

        # next: check if optimizer works for both !
        # get
        url = 'https://cluster.klima.uni-bremen.de/~oggm/geodetic_ref_mb/hugonnet_2021_ds_rgi60_pergla_rates_10_20_worldwide.csv'
        path = utils.file_downloader(url)
        pd_geodetic = pd.read_csv(path, index_col='rgiid')
        pd_geodetic = pd_geodetic.loc[pd_geodetic.period == '2000-01-01_2020-01-01']
        mb_geodetic = pd_geodetic.loc[df].dmdtda.values * 1000

        melt_f_opt_1 = scipy.optimize.brentq(minimize_bias_geodetic, 1, 1000,
                                             xtol=0.01, args=(mb_mod_1, mb_geodetic,
                                                               h, w, pf), disp=True)
        mb_mod_1.melt_f = melt_f_opt_1

        melt_f_opt_no_sfc_type = scipy.optimize.brentq(minimize_bias_geodetic, 1, 1000,
                                              xtol=0.01, args=(mb_mod_no_sfc_type,
                                                               mb_geodetic,
                                                               h, w, pf), disp=True)
        mb_mod_no_sfc_type.melt_f = melt_f_opt_no_sfc_type
        # they should optimize to the same melt_f
        assert_allclose(melt_f_opt_1, melt_f_opt_no_sfc_type)

        # check reproducibility
        assert_allclose(melt_f_opt_1, 190.5106406914272)

        spec_1 = mb_mod_1.get_specific_mb(year=years, fls=fls)
        spec_no_sfc_type = mb_mod_no_sfc_type.get_specific_mb(year=years, fls=fls)

        assert_allclose(spec_1, spec_no_sfc_type)

        # now include the surface type distinction and choose the
        # melt factor of snow to be 0.5* smaller than the melt factor of ice
        mb_mod_0_5 = TIModel_Sfc_Type(gdir, melt_f, mb_type='mb_monthly',
                                  melt_f_ratio_snow_to_ice=0.5, prcp_fac=pf)
        melt_f_opt_0_5 = scipy.optimize.brentq(minimize_bias_geodetic, 1, 1000,
                                              xtol=0.01, args=(mb_mod_0_5,
                                                               mb_geodetic,
                                                               h, w, pf),
                                              disp=True)
        # check reproducibility
        assert_allclose(melt_f_opt_0_5, 290.2804601389265)

        # the melt factor of only ice with surface type distinction should be
        # higher than the "mixed" melt factor of ice (with snow) (as the snow melt factor
        # is lower than the ice melt factor, as defined)
        assert melt_f_opt_0_5 > melt_f_opt_1

        mb_mod_0_5.melt_f = melt_f_opt_0_5
        spec_0_5 = mb_mod_0_5.get_specific_mb(year=years, fls=fls)

        # check if the optimised specific mass balance using a ratio of 0.5 is similar as
        # the optimised spec. mb of no_sfc_type (i.e, ratio of 1)
        # did the optimisation work?
        assert_allclose(spec_0_5.mean(), mb_geodetic)
        assert_allclose(spec_0_5.mean(), spec_1.mean())
        # the standard deviation can be quite different,
        assert_allclose(spec_0_5.std(), spec_1.std(), rtol=0.3)

        # check if mass balance gradient with 0.5 ratio is higher than with no
        # surface type distinction
        mb_annual_0_5 = mb_mod_0_5.get_annual_mb(h, year=2000)

        mb_gradient_0_5, _, _, _, _ = scipy.stats.linregress(h[mb_annual_0_5 < 0],
                                                         y=mb_annual_0_5[mb_annual_0_5 < 0])

        mb_gradient_1, _, _, _, _ = scipy.stats.linregress(h[mb_annual_1 < 0],
                                                             y=mb_annual_1[mb_annual_1 < 0])

        assert mb_gradient_0_5 > mb_gradient_1


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
        process_era5_daily_data(gdir, output_filesuffix='_ERA5_daily')
        f = gdir.get_filepath('climate_historical', filesuffix='_ERA5_daily')
        test_climate = xr.open_dataset(f)
        assert test_climate.time[0] == np.datetime64('1979-01-01')
        assert test_climate.time[-1] == np.datetime64('2018-12-31')


# %%
# start it again to have the default hydro_month
class Test_directobs_hydro10:
    def test_minimize_bias(self, gdir):

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
            melt_f_opt = scipy.optimize.brentq(minimize_bias, 1, 10000,
                                               disp=True, xtol=0.1,
                                                args=(gd_mb, gdir,
                                                  pf, False, input_filesuffix))

            hgts, widths = gdir.get_inversion_flowline_hw()
            mbdf = gdir.get_ref_mb_data(input_filesuffix=input_filesuffix)
            # check if they give the same optimal DDF
            assert np.round(mu_star_opt_cte[mb_type]/melt_f_opt, 3) == 1

            gd_mb.melt_f = melt_f_opt
            gd_mb.historical_climate_qc_mod(gdir)

            mb_specific = gd_mb.get_specific_mb(heights=hgts, widths=widths,
                                                year=mbdf.index.values)

            RMSD, bias, rcor, quot_std = compute_stat(mb_specific=mb_specific,
                                                      mbdf=mbdf)

            # check if the bias is optimised
            assert bias.round() == 0
    # %%


    def test_optimize_std_quot_brentq_ERA5dr(self, gdir):
        # check if double optimisation of bias and std_quotient works

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

    def test_optimize_std_quot_brentq_WFDE5(self, gdir):
        # check if double optimisation of bias and std_quotient works
        # TODO adapt this to WFDE5!!!
        grad_type = 'cte'
        N = 100
        for mb_type in ['mb_monthly', 'mb_pseudo_daily', 'mb_real_daily']:
            melt_fs = []
            prcp_facs = []
            for climate_type in ['WFDE5_CRU', 'W5E5']:

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
            assert_allclose(melt_fs[0], melt_fs[1], rtol=0.2)
            # prcp_fac can be quite different ...
            #assert_allclose(prcp_facs[0], prcp_facs[1])


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

    def test_monthly_glacier_massbalance(self, gdir):
        # I think there is a problem with SEC_IN_MONTH/SEC_IN_YEAR ...
        # do this for all model types
        # ONLY TEST it for ERA5dr or ERA5_daily!!!
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

    def test_daily_monthly_annual_specific_mb(self, gdir):
        # for both ERA5 and WFDE5
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
                    fs='_daily_WFDE5_CRU'
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
                    fs='_monthly_WFDE5_CRU'
                    process_w5e5_data(gdir, climate_type=dataset,
                                      temporal_resol='monthly')
                gd_mb = TIModel(gdir, 200, mb_type=mb_type, grad_type=grad_type,
                                prcp_fac=pf, input_filesuffix=fs,
                                baseline_climate=climate)

                spec_mb_annually = gd_mb.get_specific_mb(heights=h, widths=w,
                                                         year=np.arange(1980,
                                                                        2019))
                if mb_type == 'mb_real_daily':
                    # check if daily and yearly specific mb are the same?
                    spec_mb_daily = gd_mb.get_specific_daily_mb(heights=h,
                                                                widths=w,
                                                                year=np.arange(
                                                                    1980, 2019))
                    spec_mb_daily_yearly_sum = []
                    for mb in spec_mb_daily:
                        spec_mb_daily_yearly_sum.append(mb.sum())
                    np.testing.assert_allclose(spec_mb_daily_yearly_sum,
                                               spec_mb_annually, rtol=1e-4)

                # check if annual and monthly mass balance are the same
                ann_mb = gd_mb.get_annual_mb(heights=h, year=2015)
                mon_mb_sum = 0
                for m in np.arange(1, 13):
                    mon_mb_sum += gd_mb.get_monthly_mb(heights=h,
                                                       year=date_to_floatyear(
                                                           2015, m))
                np.testing.assert_allclose(mon_mb_sum / 12, ann_mb, rtol=1e-4)

                if mb_type == 'mb_real_daily':
                    # check if daily and annual mass balance are the same
                    day_mb = gd_mb.get_daily_mb(heights=h, year=2015)
                    day_mb_yearly_sum = []
                    for mb in day_mb:
                        day_mb_yearly_sum.append(mb.mean())
                    np.testing.assert_allclose(day_mb_yearly_sum, ann_mb,
                                               rtol=1e-4)

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

    def test_loop(self, gdir):
        # tests whether ERA5dr works better with or without loop in mb_pseudo_daily
        # tests that both option give same results and in case that default
        # option (no loop) is 30% slower, it raises an error

        # this could be optimised and included in the above tests
        # cfg.initialize()

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

    def test_historical_climate_qc_mon(self, gdir):

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
