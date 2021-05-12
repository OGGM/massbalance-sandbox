#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 19:07:38 2021

@author: lilianschuster
"""
import numpy as np
import scipy
import oggm
import pandas as pd

from oggm import cfg  # utils, workflow, tasks, graphics

# import the MSsandbox modules
from MBsandbox.mbmod_daily_oneflowline import (process_era5_daily_data, TIModel,
                                               BASENAMES, process_wfde5_data)
from MBsandbox.help_func import (compute_stat, minimize_bias,
                                 optimize_std_quot_brentq)


def minimize_bias_geodetic(x, gd_mb=None, mb_geodetic=None,
                           h=None, w=None, pf=2.5,
                           absolute_bias=False, ys=np.arange(2000, 2019, 1),
                           oggm_default_mb = False):
    """ calibrates the melt factor (melt_f) by getting the bias to zero
    comparing modelled mean specific mass balance between 2000 and 2020 to
    observed geodetic data

    Parameters
    ----------
    x : float
        what is optimised (here the melt_f)
    gd_mb: class instance
        instantiated class of TIModel, this is updated by melt_f
    mb_geodetic: float
         geodetic mass balance between 2000-2020 of the instantiated glacier
    h: np.array
        heights of the instantiated glacier
    w: np.array
        widths of the instantiated glacier
    pf: float
        precipitation scaling factor
        default is 2.5
    absolute_bias : bool
        if absolute_bias == True, the absolute value of the bias is returned.
        if optimisation is done with Powell need absolute bias.
        If optimisation is done with Brentq, absolute_bias has to be set False
        The default is False.
    ys: np.array
        years for which specific mass balance is computed
        default is 2000--2018
        TODO: change this to 2000-2019 to match better
              geodetic msm

    Returns
    -------
    float
        bias: modeled mass balance mean - reference mean (geodetic)
        if absolute_bias = True:  np.abs(bias) is returned

    """
    if oggm_default_mb:
        gd_mb.mu_star = x
    else:
        gd_mb.melt_f = x

    gd_mb.prcp_fac = pf
    mb_specific = gd_mb.get_specific_mb(heights=h,
                                        widths=w,
                                        year=ys).mean()
    if absolute_bias:
        bias_calib = np.abs(np.mean(mb_specific -
                                    mb_geodetic))
    else:
        bias_calib = np.mean(mb_specific - mb_geodetic)

    return bias_calib


def optimize_std_quot_brentq_geod(x, gd_mb=None, mb_geodetic=None,
                                  mb_glaciological=None,
                                  h=None, w=None,
                                  ys_glac=np.arange(1979, 2019, 1)):
    pf = x
    # compute optimal melt_f according to geodetic data
    melt_f_opt = scipy.optimize.brentq(minimize_bias_geodetic, 1, 10000,
                                       xtol=0.01,
                                       args=(gd_mb, mb_geodetic, h, w, pf),
                                       disp=True)

    gd_mb.melt_f = melt_f_opt
    gd_mb.prcp_fac = pf
    # now compute std over this time period using
    # direct glaciological observations
    mod_std = gd_mb.get_specific_mb(heights=h, widths=w,
                                    year=ys_glac).std()
    ref_std = mb_glaciological.loc[ys_glac].values.std()
    quot_std = mod_std / ref_std

    return 1 - quot_std


def get_opt_pf_melt_f(gd, mb_type='mb_monthly', grad_type='cte',
                      pd_calib_opt=None,
                      pd_geodetic=None, dataset='ERA5'):
    if type(pd_calib_opt) == pd.core.frame.DataFrame:
        pass
    else:
        pd_calib_opt = pd.DataFrame(
            columns=['pf_opt', 'melt_f_opt_pf', 'mb_geodetic',
                     'mb_geodetic_err', 'mb_glaciological_mean',
                     'mb_glaciological_std', 'stats_calib',  # 'stats_valid',
                     'amount_glacmsm',
                     # 'glacmsm_after_2000', 'glacmsm_before_2000',
                     'temp std', 'temp for melt std (mean)',
                     'tempfmelt_std_quot_hdiff', 'prcp std',
                     'solid prcp std (mean)', 'prcpsols_std_hstd',
                     'prcp std valid', 'solid prcp std valid', 'prcp mean',
                     'solid prcp mean', 'temp mean', 'temp for melt mean',
                     'prcp mean nopf', 'prcp mean nopf weighted',
                     'solid prcp mean nopf', 'solid prcp mean nopf weighted',
                     'solid prcp std nopf'
                     ])
        pd_calib_opt.loc[gd.rgi_id] = np.NaN
    if mb_type != 'mb_real_daily':
        if dataset == 'ERA5':
            baseline_climate = 'ERA5dr'
            oggm.shop.ecmwf.process_ecmwf_data(gd, dataset='ERA5dr')
            input_fs = ''

        elif dataset == 'WFDE5_monthly_cru':
            baseline_climate = 'WFDE5_CRU'
            process_wfde5_data(gd, temporal_resol='monthly')
            input_fs = '_monthly_WFDE5_CRU'
    elif mb_type == 'mb_real_daily':
        if dataset == 'ERA5':
            baseline_climate = 'ERA5dr'
            process_era5_daily_data(gd)
            input_fs = ''
        elif dataset == 'WFDE5_monthly_cru':
            baseline_climate = 'WFDE5_CRU'
            process_wfde5_data(gd, temporal_resol='daily')
            input_fs = '_daily_WFDE5_CRU'

    mbdf = gd.get_ref_mb_data(input_filesuffix=input_fs)
    mb_glaciological = mbdf['ANNUAL_BALANCE']
    ys_glac = mbdf.index.values
    # print(ys_glac)
    gd_mb = TIModel(gd, None, mb_type=mb_type, N=100, prcp_fac=2.5,
                    grad_type=grad_type, baseline_climate=baseline_climate)
    gd_mb.historical_climate_qc_mod(gd)

    h, w = gd.get_inversion_flowline_hw()
    mb_geodetic = pd_geodetic.loc[gd.rgi_id].dmdtda * 1000
    mb_geodetic_err = pd_geodetic.loc[gd.rgi_id].err_dmdtda * 1000

    if len(ys_glac) > 1:
        try:
            pf_opt = scipy.optimize.brentq(optimize_std_quot_brentq_geod, 0.01,
                                           20, xtol=0.01,
                                           args=(gd_mb, mb_geodetic,
                                                 mb_glaciological, h,
                                                 w, ys_glac),
                                           disp=True)
        except ValueError:  # (' f(a) and f(b) must have different signs'):
            print('{}: try out with 0.1 and 10 as pf boundaries'.format(
                gd.rgi_id))
            try:
                pf_opt = scipy.optimize.brentq(optimize_std_quot_brentq_geod,
                                               0.1,
                                               10, xtol=0.01,
                                               args=(gd_mb, mb_geodetic,
                                                     mb_glaciological, h,
                                                     w, ys_glac),
                                               disp=True)
            except ValueError:  # (' f(a) and f(b) must have different signs'):
                print('{}: try out with 0.5 and 3 as pf boundaries'.format(
                    gd.rgi_id))
                # try:
                pf_opt = scipy.optimize.brentq(optimize_std_quot_brentq_geod,
                                               0.5, 3, xtol=0.01,
                                               args=(gd_mb, mb_geodetic,
                                                     mb_glaciological, h,
                                                     w, ys_glac),
                                               disp=True)
                # except:
                #    pf_opt = 2.5

        melt_f_opt_pf = scipy.optimize.brentq(minimize_bias_geodetic, 1, 10000,
                                              xtol=0.01, args=(gd_mb,
                                                               mb_geodetic,
                                                               h, w, pf_opt),
                                              disp=True)
        gd_mb.melt_f = melt_f_opt_pf
        gd_mb.prcp_fac = pf_opt
        mb_specific_optstd = gd_mb.get_specific_mb(heights=h,
                                                   widths=w,
                                                   year=ys_glac)

        stats_calib = compute_stat(
            mb_specific=mb_specific_optstd,
            mbdf=mbdf, return_dict=True)

        cs = ['pf_opt', 'melt_f_opt_pf', 'mb_geodetic', 'mb_geodetic_err',
              'mb_glaciological_mean',
              'mb_glaciological_std', 'stats_calib',  # 'stats_valid',
              'amount_glacmsm']
        var = [pf_opt, melt_f_opt_pf, mb_geodetic,
               mb_geodetic_err, mb_glaciological.mean(),
               mb_glaciological.std(),
               str(stats_calib),  # str(stats_valid),
               len(ys_glac)]
        # len(ys_glac[ys_glac < 2000])]
        for c, v in zip(cs, var):
            # print(c, v)
            pd_calib_opt.loc[gd.rgi_id, c] = v
        pd_calib_opt.loc[gd.rgi_id, 'optimisation_possible'] = True

        ### understand which glaciers need a high pf_opt as
        # they have a high mb_glaciological_std
        gd_mb.prcp_fac = 1
        prcps_nofac = pd.DataFrame()
        prcpsols_nofac = pd.DataFrame()
        # I take only those years where measurements are available!!!!
        for y in ys_glac:
            t, tfmelt, prcp, prcpsol = gd_mb.get_annual_climate(h, year=y)
            prcps_nofac[y] = prcp
            prcpsols_nofac[y] = prcpsol
        prcp_mean_nopf = prcps_nofac.mean(axis=1).mean()
        prcp_mean_nopf_weight = (prcps_nofac.mean(axis=1) * w).mean() / w.mean()
        solid_prcp_mean_nopf = prcpsols_nofac.mean(axis=1).mean()
        solid_prcp_mean_nopf_weight = (prcpsols_nofac.mean(
            axis=1) * w).mean() / w.mean()
        prcpsols_std_nopf = prcpsols_nofac.std(axis=1).mean()

        # back to calibrated value
        gd_mb.prcp_fac = pf_opt

        # other climate data
        ts = pd.DataFrame()
        tfmelts = pd.DataFrame()
        prcps = pd.DataFrame()
        prcpsols = pd.DataFrame()
        # I take only those years where measurements are available!!!!
        for y in ys_glac:
            t, tfmelt, prcp, prcpsol = gd_mb.get_annual_climate(h, year=y)
            ts[y] = t
            tfmelts[y] = tfmelt
            prcps[y] = prcp
            prcpsols[y] = prcpsol
        # ts.index = h
        ts_std = ts.std(
            axis=1).mean()  # this is everywhere the same because of lapse rate
        ts_mean = ts.mean(
            axis=1).mean()  # this is everywhere the same because of lapse rate
        prcp_mean = prcps.mean(axis=1).mean()
        solid_prcp_mean = prcpsols.mean(axis=1).mean()
        tfmelts_std = tfmelts.std(axis=1).mean()
        tfmelts_mean = tfmelts.mean(axis=1).mean()
        tfmelts_std_quot_hdiff = tfmelts.std(axis=1).iloc[0] / \
                                 tfmelts.std(axis=1).iloc[-1]
        # this is everywhere the same because no prcp changes with height!
        prcps_std = prcps.std(axis=1).mean()

        # print(prcp_mean)
        # this changes with height
        prcpsols_std = prcpsols.std(axis=1).mean()
        prcpsols_std_hstd = prcpsols.std(axis=1).std()

        var = [ts_std, tfmelts_std, tfmelts_std_quot_hdiff, prcps_std,
               prcpsols_std, prcpsols_std_hstd, prcp_mean, solid_prcp_mean,
               ts_mean, tfmelts_mean, prcp_mean_nopf, prcp_mean_nopf_weight,
               solid_prcp_mean_nopf, solid_prcp_mean_nopf_weight,
               prcpsols_std_nopf]
        cs = ['temp std', 'temp for melt std (mean)',
              'tempfmelt_std_quot_hdiff', 'prcp std', 'solid prcp std (mean)',
              'prcpsols_std_hstd', 'prcp mean', 'solid prcp mean', 'temp mean',
              'temp for melt mean', 'prcp mean nopf', 'prcp mean nopf weighted',
              'solid prcp mean nopf', 'solid prcp mean nopf weighted',
              'solid prcp std nopf']
        for c, v in zip(cs, var):
            pd_calib_opt.loc[gd.rgi_id, c] = v
        ###

        # validation: no validation right no
        pd_calib_opt.loc[gd.rgi_id, 'prcp std valid'] = np.NaN
        pd_calib_opt.loc[gd.rgi_id, 'solid prcp std valid'] = np.NaN


    else:
        pd_calib_opt.loc[gd.rgi_id] = np.NaN
        pd_calib_opt.loc[gd.rgi_id, 'amount_glacmsm'] = len(ys_glac)
        pd_calib_opt.loc[
            gd.rgi_id, 'optimisation_possible'] = 'only 1 or less msm>=1979'

    return pd_calib_opt, gd_mb


def get_opt_pf_melt_f_above2000(gd, mb_type='mb_monthly', grad_type='cte',
                                pd_calib_opt=None,
                                pd_geodetic=None):
    if type(pd_calib_opt) == pd.core.frame.DataFrame:
        pass
    else:
        pd_calib_opt = pd.DataFrame(
            columns=['pf_opt', 'melt_f_opt_pf', 'mb_geodetic',
                     'mb_geodetic_err', 'mb_glaciological_mean',
                     'mb_glaciological_std', 'stats_calib', 'stats_valid',
                     'glacmsm_after_2000', 'glacmsm_before_2000',
                     'temp std', 'temp for melt std (mean)',
                     'tempfmelt_std_quot_hdiff', 'prcp std',
                     'solid prcp std (mean)', 'prcpsols_std_hstd',
                     'prcp std valid', 'solid prcp std valid', 'prcp mean',
                     'solid prcp mean', 'temp mean', 'temp for melt mean',
                     'prcp mean nopf', 'solid prcp mean nopf',
                     'solid prcp std nopf'
                     ])
        pd_calib_opt.loc[gd.rgi_id] = np.NaN

    cfg.PARAMS['baseline_climate'] = 'ERA5dr'
    oggm.shop.ecmwf.process_ecmwf_data(gd, dataset='ERA5dr')
    mbdf = gd.get_ref_mb_data()
    mb_glaciological = mbdf['ANNUAL_BALANCE']
    ys_glac = mbdf.index.values
    # print(ys_glac)
    gd_mb = TIModel(gd, None, mb_type=mb_type, N=100, prcp_fac=2.5,
                    grad_type=grad_type)
    gd_mb.historical_climate_qc_mod(gd)

    h, w = gd.get_inversion_flowline_hw()
    mb_geodetic = pd_geodetic.loc[gd.rgi_id].dmdtda * 1000
    mb_geodetic_err = pd_geodetic.loc[gd.rgi_id].err_dmdtda * 1000

    if len(ys_glac[ys_glac >= 2000]) > 1:
        pf_opt = scipy.optimize.brentq(optimize_std_quot_brentq_geod, 0.01, 20,
                                       xtol=0.01,
                                       args=(
                                           gd_mb, mb_geodetic, mb_glaciological,
                                           h,
                                           w, ys_glac[ys_glac >= 2000]),
                                       disp=True)

        melt_f_opt_pf = scipy.optimize.brentq(minimize_bias_geodetic, 1, 10000,
                                              xtol=0.01, args=(
                gd_mb, mb_geodetic, h, w, pf_opt),
                                              disp=True)
        gd_mb.melt_f = melt_f_opt_pf
        gd_mb.prcp_fac = pf_opt
        mb_specific_optstd = gd_mb.get_specific_mb(heights=h,
                                                   widths=w,
                                                   year=ys_glac)

        stats_calib = compute_stat(
            mb_specific=mb_specific_optstd[ys_glac >= 2000],
            mbdf=mbdf.loc[ys_glac >= 2000], return_dict=True)
        stats_valid = compute_stat(
            mb_specific=mb_specific_optstd[ys_glac < 2000],
            mbdf=mbdf.loc[ys_glac < 2000], return_dict=True)

        cs = ['pf_opt', 'melt_f_opt_pf', 'mb_geodetic', 'mb_geodetic_err',
              'mb_glaciological_mean',
              'mb_glaciological_std', 'stats_calib', 'stats_valid',
              'glacmsm_after_2000', 'glacmsm_before_2000']
        var = [pf_opt, melt_f_opt_pf, mb_geodetic,
               mb_geodetic_err, mb_glaciological[ys_glac >= 2000].mean(),
               mb_glaciological[ys_glac >= 2000].std(),
               str(stats_calib), str(stats_valid),
               len(ys_glac[ys_glac >= 2000]),
               len(ys_glac[ys_glac < 2000])]
        for c, v in zip(cs, var):
            # print(c, v)
            pd_calib_opt.loc[gd.rgi_id, c] = v
        pd_calib_opt.loc[gd.rgi_id, 'optimisation_possible'] = True

        ### understand which glaciers need a high pf_opt as
        # they have a high mb_glaciological_std
        gd_mb.prcp_fac = 1
        prcps_nofac = pd.DataFrame()
        prcpsols_nofac = pd.DataFrame()
        # I take only those years where measurements are available!!!!
        for y in ys_glac[ys_glac >= 2000]:
            t, tfmelt, prcp, prcpsol = gd_mb.get_annual_climate(h, year=y)
            prcps_nofac[y] = prcp
            prcpsols_nofac[y] = prcpsol
        prcp_mean_nopf = prcps_nofac.mean(axis=1).mean()
        solid_prcp_mean_nopf = prcpsols_nofac.mean(axis=1).mean()
        prcpsols_std_nopf = prcpsols_nofac.std(axis=1).mean()

        # back to calibrated value
        gd_mb.prcp_fac = pf_opt

        # other climate data
        ts = pd.DataFrame()
        tfmelts = pd.DataFrame()
        prcps = pd.DataFrame()
        prcpsols = pd.DataFrame()
        # I take only those years where measurements are available!!!!
        for y in ys_glac[ys_glac >= 2000]:
            t, tfmelt, prcp, prcpsol = gd_mb.get_annual_climate(h, year=y)
            ts[y] = t
            tfmelts[y] = tfmelt
            prcps[y] = prcp
            prcpsols[y] = prcpsol
        # ts.index = h
        ts_std = ts.std(
            axis=1).mean()  # this is everywhere the same because of lapse rate
        ts_mean = ts.mean(
            axis=1).mean()  # this is everywhere the same because of lapse rate
        prcp_mean = prcps.mean(axis=1).mean()
        solid_prcp_mean = prcpsols.mean(axis=1).mean()
        tfmelts_std = tfmelts.std(axis=1).mean()
        tfmelts_mean = tfmelts.mean(axis=1).mean()
        tfmelts_std_quot_hdiff = tfmelts.std(axis=1).iloc[0] / \
                                 tfmelts.std(axis=1).iloc[-1]
        # this is everywhere the same because no prcp changes with height!
        prcps_std = prcps.std(axis=1).mean()

        # print(prcp_mean)
        # this changes with height
        prcpsols_std = prcpsols.std(axis=1).mean()
        prcpsols_std_hstd = prcpsols.std(axis=1).std()

        var = [ts_std, tfmelts_std, tfmelts_std_quot_hdiff, prcps_std,
               prcpsols_std, prcpsols_std_hstd, prcp_mean, solid_prcp_mean,
               ts_mean, tfmelts_mean, prcp_mean_nopf, solid_prcp_mean_nopf,
               prcpsols_std_nopf]
        cs = ['temp std', 'temp for melt std (mean)',
              'tempfmelt_std_quot_hdiff', 'prcp std', 'solid prcp std (mean)',
              'prcpsols_std_hstd', 'prcp mean', 'solid prcp mean', 'temp mean',
              'temp for melt mean', 'prcp mean nopf', 'solid prcp mean nopf',
              'solid prcp std nopf']
        for c, v in zip(cs, var):
            pd_calib_opt.loc[gd.rgi_id, c] = v
        ###

        # validation: before 2000:
        if len(ys_glac[ys_glac < 2000]) > 1:
            prcps_val = pd.DataFrame()
            prcpsols_val = pd.DataFrame()
            for y in ys_glac[ys_glac < 2000]:
                _, _, prcp, prcpsol = gd_mb.get_annual_climate(h, year=y)
                prcps_val[y] = prcp
                prcpsols_val[y] = prcpsol
            prcps_std_val = prcps_val.std(axis=1).mean()
            prcpsols_std_val = prcpsols_val.std(axis=1).mean()
            pd_calib_opt.loc[gd.rgi_id, 'prcp std valid'] = prcps_std_val
            pd_calib_opt.loc[
                gd.rgi_id, 'solid prcp std valid'] = prcpsols_std_val
        else:
            pd_calib_opt.loc[gd.rgi_id, 'prcp std valid'] = np.NaN
            pd_calib_opt.loc[gd.rgi_id, 'solid prcp std valid'] = np.NaN


    else:
        pd_calib_opt.loc[gd.rgi_id] = np.NaN
        pd_calib_opt.loc[gd.rgi_id, 'glacmsm_after_2000'] = len(
            ys_glac[ys_glac >= 2000])
        pd_calib_opt.loc[gd.rgi_id, 'glacmsm_before_2000'] = len(
            ys_glac[ys_glac < 2000])
        pd_calib_opt.loc[
            gd.rgi_id, 'optimisation_possible'] = 'only 1 or less msm>=2000'

    return pd_calib_opt, gd_mb