#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 24 12:28:37 2020

@author: lilianschuster

some helper functions to minimize the bias, optimise std_quot and tos
compute performance statistics
"""
import scipy
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error
import pandas as pd
import logging
from oggm.core import climate
from oggm import utils, workflow, tasks, entity_task, cfg
from oggm.exceptions import MassBalanceCalibrationError

log = logging.getLogger(__name__)

# imports from local MBsandbox package modules
import MBsandbox
from MBsandbox.mbmod_daily_oneflowline import TIModel, TIModel_Sfc_Type
# from MBsandbox.help_func import compute_stat, minimize_bias, optimize_std_quot_brentq
from MBsandbox.flowline_TIModel import (run_from_climate_data_TIModel, run_constant_climate_TIModel,
                                        run_random_climate_TIModel)


# necessary for `melt_f_calib_geod_prep_inversion`
_doc = 'the calibrated melt_f according to the geodetic data with the ' \
       'chosen precipitation factor'
cfg.BASENAMES['melt_f_geod'] = ('melt_f_geod.json', _doc)

def minimize_bias(x, gd_mb=None, gdir_min=None,
                  pf=None, absolute_bias=False, input_filesuffix=''):
    """ calibrates the melt factor (melt_f) by getting the bias to zero
    comparing modelled mean specific mass balance to
    direct glaciological observations

    attention: this is a bit deprecated. It uses the direct glaciological
    observations for calibration. If you want to use the geodetic,
    you should use instead `minimize_bias_geodetic`!

    (and actually the minimisation occurs only when doing scipy.optimize.brentq(minimize_bias, 10, 100, ...)
    but we don't wan to change the function at this stage)

    Parameters
    ----------
    x : float
        what is optimised; here the melt factor (melt_f)
    gd_mb: class instance
        instantiated class of TIModel, this is updated by melt_f
    gdir_min :
        glacier directory. The default is None but this has to be set.
    pf: float: optional
        precipitation factor. The default is 2.5.
    absolute_bias : bool
        if absolute_bias == True, the absolute value of the bias is returned.
        if optimisation is done with Powell need absolute bias.
        If optimisation is done with Brentq, absolute_bias has to set False
        The default is False.
    input_filesuffix: str
        default is ''. If set, it is used to choose the right filesuffix
        for the ref mb data.

    Returns
    -------
    float
        bias: modeled mass balance mean - reference mean
        if absolute_bias = True:  np.abs(bias) is returned

    """

    h, w = gdir_min.get_inversion_flowline_hw()
    mbdf = gdir_min.get_ref_mb_data(input_filesuffix=input_filesuffix)
    gd_mb.melt_f = x
    if type(pf) == float or type(pf) == int:
        gd_mb.prcp_fac = pf

    # check climate and adapt if necessary
    gd_mb.historical_climate_qc_mod(gdir_min)
    mb_specific = gd_mb.get_specific_mb(heights=h,
                                        widths=w,
                                        year=mbdf.index.values)
    if absolute_bias:
        bias_calib = np.abs(np.mean(mb_specific -
                                    mbdf['ANNUAL_BALANCE'].values))
    else:
        bias_calib = np.mean(mb_specific - mbdf['ANNUAL_BALANCE'].values)

    return bias_calib

def minimize_bias_geodetic(x, gd_mb=None, mb_geodetic=None,
                           h=None, w=None, pf=2.5,
                           absolute_bias=False,
                           ys=np.arange(2000, 2020, 1),
                           oggm_default_mb=False,
                           spinup=True):
    """ calibrates the melt factor (melt_f) by getting the bias to zero
    comparing modelled mean specific mass balance between 2000 and 2020 to
    observed geodetic data (from Hugonnet et al. 2021)

    (and actually the minimisation occurs only when doing scipy.optimize.brentq(minimize_bias_geodetic, 10, 100, ...)
    but we don't want to change the function at this stage)

    Parameters
    ----------
    x : float
        what is optimised (here the melt_f)
    gd_mb: class instance
        instantiated class of TIModel, this is updated by melt_f
    mb_geodetic: float
         geodetic mass balance between 2000-2020 of the instantiated glacier
    h: ndarray
        heights of the instantiated glacier
    w: ndarray
        widths of the instantiated glacier.
        Important to set that otherwise it is assumed that the glacier has the same width everywhere!
    pf: float
        precipitation scaling factor
        default is 2.5
    absolute_bias : bool
        if absolute_bias == True, the absolute value of the bias is returned.
        if optimisation is done with Powell need absolute bias.
        If optimisation is done with Brentq, absolute_bias has to be set False
        The default is False.
    ys: ndarray
        years for which specific mass balance is computed
        default is 2000--2019 (when using W5E5)
    oggm_default_mb : bool
        if default oggm mass balance should be used (default is False)
    spinup : bool
        send to get_specific_mb (for sfc type distinction)

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

    try:
        mb_specific = gd_mb.get_specific_mb(heights=h,
                                            widths=w,
                                            year=ys,
                                            spinup=spinup
                                            ).mean()
    except:
        mb_specific = gd_mb.get_specific_mb(heights=h,
                                        widths=w,
                                        year=ys
                                        ).mean()

    if absolute_bias:
        bias_calib = np.abs(np.mean(mb_specific -
                                    mb_geodetic))
    else:
        bias_calib = np.mean(mb_specific - mb_geodetic)

    return bias_calib

def minimize_bias_geodetic_via_pf_fixed_melt_f(x, gd_mb=None, mb_geodetic=None,
                           h=None, w=None, melt_f=None,
                           absolute_bias=False,
                           ys=np.arange(2000, 2020, 1),
                           oggm_default_mb=False,
                           spinup=True):
    """ calibrates the precipitation factor (pf) by getting the bias to zero
    comparing modelled mean specific mass balance between 2000 and 2020 to
    observed geodetic data (from Hugonnet et al. 2021)

    Important, here the free parameter that is tuned to match the geodetic estimates is the precipitation factor
    (and not the melt_f) !!!

    (and actually the minimisation occurs only when doing
    scipy.optimize.brentq(minimize_bias_geodetic_via_pf_fixed_melt_f, 0.1, 10, ...)
    but we don't wan to change the function at this stage)

    Parameters
    ----------
    x : float
        what is optimised (here the precipitation factor pf)
    gd_mb: class instance
        instantiated class of TIModel, this is updated with the prescribed pf (& melt_f)
    mb_geodetic: float
         geodetic mass balance between 2000-2020 of the instantiated glacier
    h: np.array
        heights of the instantiated glacier
    w: np.array
        widths of the instantiated glacier
    melt_f: float
        melt factor
        has to be set!!!
    absolute_bias : bool
        if absolute_bias == True, the absolute value of the bias is returned.
        if optimisation is done with Powell need absolute bias.
        If optimisation is done with Brentq, absolute_bias has to be set False
        The default is False.
    ys: np.array
        years for which specific mass balance is computed
        default is 2000--2019 (when using W5E5)
    oggm_default_mb : bool
        if default oggm mass balance should be used (default is False)
    spinup : bool
        send to get_specific_mb (for sfc type distinction)

    Returns
    -------
    float
        bias: modeled mass balance mean - reference mean (geodetic)
        if absolute_bias = True:  np.abs(bias) is returned

    """
    # not sure if this works for the oggm default PastMassBalance instance, probably not:
    gd_mb.prcp_fac = x

    if oggm_default_mb:
        gd_mb.mu_star = melt_f
    else:
        gd_mb.melt_f = melt_f


    try:
        mb_specific = gd_mb.get_specific_mb(heights=h,
                                            widths=w,
                                            year=ys,
                                            spinup=spinup
                                            ).mean()
    except:
        mb_specific = gd_mb.get_specific_mb(heights=h,
                                        widths=w,
                                        year=ys
                                        ).mean()

    if absolute_bias:
        bias_calib = np.abs(np.mean(mb_specific -
                                    mb_geodetic))
    else:
        bias_calib = np.mean(mb_specific - mb_geodetic)

    return bias_calib


def minimize_bias_geodetic_via_temp_bias(x, gd_mb=None, mb_geodetic=None,
                           h=None, w=None, melt_f=None, pf=None,
                           absolute_bias=False,
                           ys=np.arange(2000, 2020, 1),
                           oggm_default_mb=False,
                           spinup=True):
    """ calibrates the temperature bias (t_bias) by getting the bias (|observed-geodetic|) to zero
    comparing modelled mean specific mass balance between 2000 and 2020 to
    observed geodetic data (from Hugonnet et al. 2021)

    Important, here the free parameter that is tuned to match the geodetic estimates is the temperature bias
    (and not the melt_f and also not pf). Hence, both melt_f and pf have to be prescribed !!!

    (and actually the minimisation occurs only when doing
    scipy.optimize.brentq(minimize_bias_geodetic_via_temp_bias, -5, 5, ...)
    but we don't want to change the function at this stage)

    Parameters
    ----------
    x : float
        what is optimised (here the temperature bias)
    gd_mb: class instance
        instantiated class of TIModel, this is updated with the prescribed pf & melt_f
    mb_geodetic: float
         geodetic mass balance between 2000-2020 of the instantiated glacier
    h: ndarray
        heights of the instantiated glacier
    w: ndarray
        widths of the instantiated glacier
    melt_f: float
        melt factor
        has to be set!!!
    pf : float
        precipitation factor
        has to be set!!!
    absolute_bias : bool
        if absolute_bias == True, the absolute value of the bias is returned.
        if optimisation is done with Powell need absolute bias.
        If optimisation is done with Brentq, absolute_bias has to be set False
        The default is False.
    ys: ndarray
        years for which specific mass balance is computed
        default is 2000--2019 (when using W5E5)
    oggm_default_mb : bool
        if default oggm mass balance should be used (default is False)
    spinup : bool
        send to get_specific_mb (for sfc type distinction)

    Returns
    -------
    float
        bias: modeled mass balance mean - reference mean (geodetic)
        if absolute_bias = True:  np.abs(bias) is returned

    """
    # not sure if this works for the oggm default PastMassBalance instance, probably not:
    gd_mb.temp_bias = x

    gd_mb.prcp_fac = pf

    if oggm_default_mb:
        gd_mb.mu_star = melt_f
    else:
        gd_mb.melt_f = melt_f


    try:
        mb_specific = gd_mb.get_specific_mb(heights=h,
                                            widths=w,
                                            year=ys,
                                            spinup=spinup
                                            ).mean()
    except:
        mb_specific = gd_mb.get_specific_mb(heights=h,
                                        widths=w,
                                        year=ys
                                        ).mean()

    if absolute_bias:
        bias_calib = np.abs(np.mean(mb_specific -
                                    mb_geodetic))
    else:
        bias_calib = np.mean(mb_specific - mb_geodetic)

    return bias_calib


def optimize_std_quot_brentq_geod_via_melt_f(x, gd_mb=None, mb_geodetic=None,
                                             mb_glaciological=None,
                                             h=None, w=None,
                                             ys_glac=np.arange(1979, 2020, 1),
                                             ):
    """ calibrates the optimal melt factor (melt_f) by correcting the
    standard deviation of the modelled mass balance by using the standard deviation
    from the direct glaciological measurements as reference

    for each melt_f an optimal pf is found (by using the geodetic data via
    `minimize_bias_geodetic_via_pf_fixed_melt_f`), then (1 - standard deviation quotient between modelled and
    reference mass balance) is computed, which is then minimised.

    Important, here the free parameter that is tuned to match the geodetic estimates is the precipitation factor
    (and not the melt_f) !!!

    (and actually the optimisation occurs only when doing
    scipy.optimize.brentq(optimize_std_quot_brentq_geod_via_melt_f, 10, 1000, ...)
    but we don't want to change the function at this stage)

    Parameters
    ----------
    x : float
        what is optimised (here the melt_f)
    gd_mb : class instance
        instantiated class of TIModel, this is updated by pf and melt_f
    mb_geodetic: float
        geodetic mass balance between 2000-2020 of the instantiated glacier
    mb_glaciological : pandas.core.series.Series
        direct glaciological timeseries
        e.g. gdir.get_ref_mb_data(input_filesuffix='_{}_{}'.format(temporal_resol, climate_type))['ANNUAL_BALANCE']
    h: ndarray
        heights of the instantiated glacier
    w: ndarray
        widths of the instantiated glacier
    ys_glac : ndarray
        array of years where both, glaciological observations and climate data are available
        (just use the years from the ref_mb_data file)

    Returns
    -------
    float
        1- quot_std

    """

    melt_f = x
    # compute optimal pf according to geodetic data
    pf_opt = scipy.optimize.brentq(minimize_bias_geodetic_via_pf_fixed_melt_f, 0.01, 20,  # pf range
                                       xtol=0.1,
                                       args=(gd_mb, mb_geodetic, h, w, melt_f),
                                       disp=True)

    gd_mb.prcp_fac = pf_opt
    gd_mb.melt_f = melt_f
    # now compute std over this time period using
    # direct glaciological observations
    mod_std = gd_mb.get_specific_mb(heights=h, widths=w,
                                    year=ys_glac).std()
    ref_std = mb_glaciological.loc[ys_glac].values.std()
    quot_std = mod_std / ref_std

    return 1 - quot_std

def optimize_std_quot_brentq_geod_via_temp_bias(x, gd_mb=None, mb_geodetic=None,
                                  mb_glaciological=None,
                                  h=None, w=None,
                                  ys_glac=np.arange(1979, 2020, 1),
                                    pf = None
                                  ):
    """ calibrates the optimal temperature bias by correcting the
    standard deviation of the modelled mass balance by using the standard deviation
    from the direct glaciological measurements as reference. The prcp. fac is here
     not used for calibration (just set to a cte "arbitrary" value)


    for each temp. bias an optimal melt_f is found (by using the geodetic data via `minimize_bias_geodetic`),
    then (1 - standard deviation quotient between modelled and reference mass balance) is computed,
    which is then minimised

    (and actually the optimisation occurs only when doing
    scipy.optimize.brentq(optimize_std_quot_brentq_geod_via_temp_bias, -5, 5, ...)
    but we don't want to change the function at this stage)

    Parameters
    ----------
    x : float
        what is optimised (here the temperature bias)
    gd_mb : class instance
        instantiated class of TIModel, this is updated by temperature bias and melt_f
    mb_geodetic: float
        geodetic mass balance between 2000-2020 of the instantiated glacier
    mb_glaciological : pandas.core.series.Series
        direct glaciological timeseries
        e.g. gdir.get_ref_mb_data(input_filesuffix='_{}_{}'.format(temporal_resol, climate_type))['ANNUAL_BALANCE']
    h: ndarray
        heights of the instantiated glacier
    w: ndarray
        widths of the instantiated glacier
    ys_glac : ndarray
        array of years where both, glaciological observations and climate data are available
        (just use the years from the ref_mb_data file)
    pf : float
        precipitation factor (here constant and has to be set to any value)

    Returns
    -------
    float
        1- quot_std

    """
    temp_bias = x
    gd_mb.temp_bias = temp_bias
    # compute optimal melt_f according to geodetic data for that temp_bias
    melt_f_opt = scipy.optimize.brentq(minimize_bias_geodetic, 10, 1000,
                                       xtol=0.01,
                                       args=(gd_mb, mb_geodetic, h, w, pf),
                                       disp=True)

    gd_mb.melt_f = melt_f_opt
    gd_mb.temp_bias = temp_bias
    # now compute std over this time period using
    # direct glaciological observations
    mod_std = gd_mb.get_specific_mb(heights=h, widths=w,
                                    year=ys_glac).std()
    ref_std = mb_glaciological.loc[ys_glac].values.std()
    quot_std = mod_std / ref_std

    return 1 - quot_std

def optimize_std_quot_brentq_geod(x, gd_mb=None, mb_geodetic=None,
                                  mb_glaciological=None,
                                  h=None, w=None,
                                  ys_glac=np.arange(1979, 2020, 1),
                                  ):
    """ calibrates the optimal precipitation factor (pf) by correcting the
    standard deviation of the modelled mass balance by using the standard deviation
    from the direct glaciological measurements as reference

    for each pf an optimal melt_f is found (by using the geodetic data via `minimize_bias_geodetic`),
    then (1 - standard deviation quotient between modelled and reference mass balance) is computed,
    which is then minimised

    (and actually the optimisation occurs only when doing
    scipy.optimize.brentq(optimize_std_quot_brentq_geod, 0.1, 10, ...)
    but we don't want to change the function at this stage)

    Parameters
    ----------
    x : float
        what is optimised (here the precipitation factor)
    gd_mb : class instance
        instantiated class of TIModel, this is updated by pf and melt_f
    mb_geodetic: float
        geodetic mass balance between 2000-2020 of the instantiated glacier
    mb_glaciological : pandas.core.series.Series
        direct glaciological timeseries
        e.g. gdir.get_ref_mb_data(input_filesuffix='_{}_{}'.format(temporal_resol, climate_type))['ANNUAL_BALANCE']
    h: ndarray
        heights of the instantiated glacier
    w: ndarray
        widths of the instantiated glacier
    ys_glac : ndarray
        array of years where both, glaciological observations and climate data are available
        (just use the years from the ref_mb_data file)

    Returns
    -------
    float
        1- quot_std

    """

    pf = x
    # compute optimal melt_f according to geodetic data
    melt_f_opt = scipy.optimize.brentq(minimize_bias_geodetic, 10, 1000,
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


def compute_stat(mb_specific=None, mbdf=None, return_dict=False,
                 return_plot=False, round=False):
    """ function that computes RMSD, bias, rcor, quot_std between modelled
    and reference mass balance

    Parameters
    ----------
    mb_specific : np.array or pd.Series
        modelled mass balance
    mbdf : np.array or pd.Series
        reference mass balance
    return_dict : bool
        If a dictionary instead of a list should be returned.
        The default is False
    return_plot :
        If modelled mass balance should be plotted with statistics as label,
        write the label_part1 (mb_type and grad_type) into return_plot.
        The default is False and means that no plot is returned.
    Returns
    -------
    RMSD :
        root-mean squared deviation
    bias :
        modeled mass balance mean - reference mean
    rcor :
        correlation coefficent between modelled and reference mass balance
    quot_std : TYPE
        standard deviation quotient of modelled against reference mass balance

    """
    # with np.array normalised by N / with pandas default option is N-1 !!!
    obs = mbdf['ANNUAL_BALANCE'].values  # need values otherwise problem in std
    RMSD = np.sqrt(np.sum(np.square(mb_specific - obs)))/len(obs)
    ref_std = obs.std()
    mod_std = mb_specific.std()
    bias = mb_specific.mean() - obs.mean()
    # this is treated a bit different than in mb_crossval of Matthias Dusch
    if ref_std == 0:
        # in mb_crossval: ref_std is then set equal to std of the modeled mb
        quot_std = np.NaN
        # in mb_crossval: rcor is set to 1 but I guess it should not be counted
        # because it is not sth. we want to count
        rcor = np.NaN
    else:
        quot_std = mod_std/ref_std
        rcor = np.corrcoef(mb_specific, obs)[0, 1]

    # could also be returned as dictionary instead to not confuse the results
    if return_plot is not False:
        stat_l = ('RMSD {}, rcor {}'
                  ', std_quot {}, bias {}'.format(RMSD.round(1),
                                                  rcor.round(3),
                                                  quot_std.round(3),
                                                  bias.round(2)))
        label = return_plot + stat_l
        plt.plot(mbdf.index, mb_specific, label=label)
    if round:
        RMSD = RMSD.round(1)
        bias = bias.round(1)
        rcor = rcor.round(3)
        quot_std = quot_std.round(3)
    if return_dict:
        return {'RMSD': RMSD, 'bias': bias,
                'rcor': rcor, 'quot_std': quot_std}
    else:
        return [RMSD, bias, rcor, quot_std]

# %%


def optimize_std_quot_brentq(x, gd_mb=None,
                             gdir_min=None,
                             input_filesuffix=''):
    """ calibrates the optimal precipitation factor (pf) by correcting the
    standard deviation of the modelled mass balance

    for each pf an optimal melt_f is found, then (1 - standard deviation quotient
    between modelled and reference mass balance) is computed,
    which is then minimised

    (this is a bit deprecated as we use geodetic data to calibrate melt_f via the mean MB)

    (and actually the optimisation occurs only when doing
    scipy.optimize.brentq(optimize_std_quot_brentq, 0.1, 10, ...)
    but we don't want to change the function at this stage)

    Parameters
    ----------
    x : float
        what is optimised (here the precipitation factor)
    gd_mb: class instance
        instantiated class of TIModel, this is updated by pf and melt_f
    gdir_min : optional
        glacier directory. The default is None but this has to be set.
    input_filesuffix: str
        default is ''. If set, it is used to choose the right filesuffix
        for the ref mb data.

    Returns
    -------
    float
        1- quot_std

    """
    h, w = gdir_min.get_inversion_flowline_hw()
    mbdf = gdir_min.get_ref_mb_data(input_filesuffix=input_filesuffix)
    pf = x
    melt_f_opt = scipy.optimize.brentq(minimize_bias, 1, 10000,
                                               disp=True, xtol=0.1,
                                                args=(gd_mb, gdir_min,
                                                pf, False,
                                                input_filesuffix))
    gd_mb.melt_f = melt_f_opt
    # check climate and adapt if necessary
    gd_mb.historical_climate_qc_mod(gdir_min)

    mod_std = gd_mb.get_specific_mb(heights=h, widths=w,
                                    year=mbdf.index.values).std()
    ref_std = mbdf['ANNUAL_BALANCE'].values.std()
    quot_std = mod_std/ref_std

    return 1-quot_std

def optimize_std_quot_brentq_via_temp_b_w_min_winter_geod_bias(x, gd_mb=None,
                                                               mb_geodetic=None,
                                                               winter_mb_observed = None,
                                                               yrs_seasonal_mbs = None,
                                                               mb_glaciological=None,
                                                               ys_glac=np.arange(1979, 2020, 1),
                                                               h=None, w=None,
                                                               ):
    """ calibrates the optimal temperature bias by correcting the
    standard deviation of the modelled mass balance by using the standard deviation
    from the direct glaciological measurements as reference while maintaining minimum winter bias
    and geodetic bias

    for each temp. b. an optimal pf and melt_f is found (that minimise winter and geodetic bias),
    then (1 - standard deviation quotient
    between modelled and reference mass balance) is computed,
    which is then minimised

    (and actually the optimisation occurs only when doing
    scipy.optimize.brentq(optimize_std_quot_brentq_via_temp_b_w_min_winter_geod_bias, -5, 5, ...)
    but we don't want to change the function name at this stage)

    Parameters
    ----------
    x : float
        what is optimised (here the temperature bias)
    gd_mb : class instance
        instantiated class of TIModel, this is updated by temperature bias and melt_f
    mb_geodetic: float
        geodetic mass balance between 2000-2020 of the instantiated glacier
    winter_mb_observed : pandas.core.series.Series
        winter MB
        e.g. gdir.get_ref_mb_data(input_filesuffix='_daily_W5E5').loc[yrs_seasonal_mbs]['WINTER_BALANCE']
    yrs_seasonal_mbs : np.array
        years for which we want to use winter MB (those with valid winter MB), e.g. :
        _, path = utils.get_wgms_files()
        pd_mb_overview = pd.read_csv(path[:-len('/mbdata')] + '/mb_overview_seasonal_mb_time_periods_20220301.csv',
                                     index_col='Unnamed: 0')
        or via:
        path_mbsandbox = MBsandbox.__file__[:-len('/__init__.py')]
        pd_mb_overview = pd.read_csv(path_mbsandbox + '/data/mb_overview_seasonal_mb_time_periods_20220301.csv',
                                     index_col='Unnamed: 0')
        yrs_seasonal_mbs = pd_mb_overview.loc[pd_mb_overview.rgi_id == gdir.rgi_id].Year.values
    mb_glaciological : pandas.core.series.Series
        direct glaciological timeseries
        e.g. gdir.get_ref_mb_data(input_filesuffix='_{}_{}'.format(temporal_resol, climate_type))['ANNUAL_BALANCE']
    h: ndarray
        heights of the instantiated glacier
    w: ndarray
        widths of the instantiated glacier
    ys_glac : ndarray
        array of years where both, glaciological observations and climate data are available
        (just use the years from the ref_mb_data file)

    Returns
    -------
    float
        1- quot_std

    """
    temp_bias = x
    gd_mb.temp_bias = temp_bias

    winter_mb_observed = winter_mb_observed.loc[yrs_seasonal_mbs]
    except_necessary = 0
    # minimize bias of winter MB
    try:
        pf_opt = scipy.optimize.brentq(minimize_winter_mb_brentq_geod_via_pf, 0.1, 10, xtol=0.1,
                                       args=(gd_mb, mb_geodetic, winter_mb_observed, h, w, yrs_seasonal_mbs,
                                             True)  # period_from_wgms
                                       )
    except:
        except_necessary += 1
        try:
            pf_opt = scipy.optimize.brentq(minimize_winter_mb_brentq_geod_via_pf, 0.4, 5, xtol=0.1,
                                           args=(
                                           gd_mb, mb_geodetic, winter_mb_observed, h, w, yrs_seasonal_mbs,
                                           True)  # period_from_wgms
                                           )
        except:
            melt_f_opt_dict = {}
            for pf in np.concatenate([np.arange(0.1, 3, 0.5), np.arange(3, 10, 2)]):
                try:
                    melt_f = scipy.optimize.brentq(minimize_bias_geodetic, 10, 1000,
                                                   xtol=0.01,
                                                   args=(gd_mb, mb_geodetic,
                                                         h, w, pf),
                                                   disp=True)
                    melt_f_opt_dict[pf] = melt_f
                except:
                    pass
            # print(melt_f_opt_dict)
            pf_start = list(melt_f_opt_dict.items())[0][0]
            pf_end = list(melt_f_opt_dict.items())[-1][0]
            try:
                pf_opt = scipy.optimize.brentq(minimize_winter_mb_brentq_geod_via_pf, pf_start, pf_end, xtol=0.1,
                                               args=(gd_mb, mb_geodetic, winter_mb_observed,
                                                     h, w, yrs_seasonal_mbs,
                                                     True)  # period_from_wgms
                                               )
            except:
                except_necessary += 1
                try:
                    pf_start = list(melt_f_opt_dict.items())[1][0]
                    pf_end = list(melt_f_opt_dict.items())[-2][0]
                    pf_opt = scipy.optimize.brentq(minimize_winter_mb_brentq_geod_via_pf, pf_start, pf_end, xtol=0.1,
                                                   args=(gd_mb, mb_geodetic, winter_mb_observed,
                                                         h, w, yrs_seasonal_mbs,
                                                         True)  # period_from_wgms
                                                   )
                except:
                    except_necessary += 1
                    pf_start = list(melt_f_opt_dict.items())[2][0]
                    pf_end = list(melt_f_opt_dict.items())[-3][0]
                    pf_opt = scipy.optimize.brentq(minimize_winter_mb_brentq_geod_via_pf, pf_start, pf_end, xtol=0.1,
                                                   args=(gd_mb, mb_geodetic, winter_mb_observed,
                                                         h, w, yrs_seasonal_mbs,
                                                         True)  # period_from_wgms
                                                   )



    # compute optimal melt_f according to geodetic data for that temp_bias and that pf_opt
    melt_f_opt = scipy.optimize.brentq(minimize_bias_geodetic, 10, 1000,
                                       xtol=0.01,
                                       args=(gd_mb, mb_geodetic, h, w, pf_opt),
                                       disp=True)

    gd_mb.melt_f = melt_f_opt
    gd_mb.temp_bias = temp_bias


    # now compute std over this time period using
    # direct glaciological observations
    mod_std = gd_mb.get_specific_mb(heights=h, widths=w,
                                    year=ys_glac).std()
    ref_std = mb_glaciological.loc[ys_glac].values.std()
    quot_std = mod_std / ref_std

    return 1 - quot_std

@entity_task(log)
def calibrate_to_geodetic_bias_quot_std_different_temp_bias(gdir,
                                                            temp_b_range=np.arange(-4, 4.1, 2),
                                                            # np.arange(-6,6.1,0.5)
                                                            method='pre-check', melt_f_update='monthly',
                                                            sfc_type_distinction=True,
                                                            path='/home/lilianschuster/Schreibtisch/PhD/bayes_2022/calib_winter_mb/test_run',
                                                            pf_cte_dict=False,
                                                            optimize_std_quot=True,
                                                            pf_cte_via= ''):
    ''' todo: do documentation '''
    j = 0
    # sfc_type_distinction types
    n = 2
    if not sfc_type_distinction:
        melt_f_update = np.NaN
        n = 1

    pd_geodetic_all = utils.get_geodetic_mb_dataframe()
    pd_geodetic = pd_geodetic_all.loc[pd_geodetic_all.period == '2000-01-01_2020-01-01']
    mb_geodetic = pd_geodetic.loc[gdir.rgi_id].dmdtda * 1000

    climate_type = 'W5E5'
    # actually it does not matter which climate input fs we use as long it exists (they all have the same time period)
    try:
        mbdf = gdir.get_ref_mb_data(input_filesuffix='_monthly_W5E5')
    except:
        mbdf = gdir.get_ref_mb_data(input_filesuffix='_daily_W5E5')

    # get the filtered seasonal MB data, if it is not available or has less than 5 measurements
    # just take as years the annual years and as values NaNs
    oggm_updated = False
    if oggm_updated:
        _, pathi = utils.get_wgms_files()
        pd_mb_overview = pd.read_csv(
            pathi[:-len('/mbdata')] + '/mb_overview_seasonal_mb_time_periods_20220301.csv',
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
    pd_mb_overview = pd_mb_overview[pd_mb_overview['at_least_5_winter_mb']]
    try:
        pd_mb_overview_sel_gdir = pd_mb_overview.loc[pd_mb_overview.rgi_id == gdir.rgi_id]
        pd_mb_overview_sel_gdir.index = pd_mb_overview_sel_gdir.Year
        yrs_seasonal_mbs = pd_mb_overview_sel_gdir.Year.values
        assert np.all(yrs_seasonal_mbs >= 1980)
        assert np.all(yrs_seasonal_mbs < 2020)
    except:
        # just take the years where annual MB exists (even if no winter MB exist)--> get np.NaN for winter_mb_observed!
        yrs_seasonal_mbs = mbdf.index
        # actually it does not matter which climate input fs we use as long it exists (they all have the same time period)
    try:
        winter_mb_observed = gdir.get_ref_mb_data(input_filesuffix='_monthly_W5E5').loc[yrs_seasonal_mbs][
            'WINTER_BALANCE']
        mean_mb_prof = get_mean_mb_profile_filtered(gdir, input_fs='_monthly_W5E5', obs_ratio_needed=0.6)

    except:
        winter_mb_observed = gdir.get_ref_mb_data(input_filesuffix='_daily_W5E5').loc[yrs_seasonal_mbs][
            'WINTER_BALANCE']
        mean_mb_prof = get_mean_mb_profile_filtered(gdir, input_fs='_daily_W5E5', obs_ratio_needed=0.6)

    # annual_mb_observed = gdir.get_ref_mb_data(input_filesuffix='_daily_W5E5').loc[yrs_seasonal_mbs][
    #    'ANNUAL_BALANCE']
    hgts, widths = gdir.get_inversion_flowline_hw()

    pd_calib = pd.DataFrame(np.NaN, index=np.arange(0, int(3 * 2 * n * len(temp_b_range))),  # *len(gdirs))),
                            columns=['rgi_id', 'temp_bias', 'pf_opt', 'melt_f',
                                     'winter_prcp_mean', 'winter_solid_prcp_mean',
                                     'specific_melt_winter_kg_m2', 'except_necessary', 'quot_std', 'mae_mb_profile',
                                     'bias_winter_mb',
                                     'mb_type', 'grad_type', 'melt_f_change', 'melt_f_update','tau_e_fold_yr'])
    for mb_type in ['mb_monthly', 'mb_pseudo_daily', 'mb_real_daily']:
        for grad_type in ['cte', 'var_an_cycle']:
            for melt_f_change_r in ['linear', 'neg_exp_t0.5yr', 'neg_exp_t1.0yr']:
                if 'neg_exp' in melt_f_change_r:
                    melt_f_change = 'neg_exp'
                    if melt_f_change_r == 'neg_exp_t0.5yr':
                        tau_e_fold_yr = 0.5
                    elif melt_f_change_r == 'neg_exp_t1.0yr':
                        tau_e_fold_yr = 1
                else:
                    melt_f_change = melt_f_change_r
                    tau_e_fold_yr = np.NaN
                if sfc_type_distinction:
                    gd_mb = TIModel_Sfc_Type(gdir, 200,
                                             prcp_fac=1,
                                             mb_type=mb_type,
                                             grad_type=grad_type,
                                             baseline_climate=climate_type,
                                             melt_f_ratio_snow_to_ice=0.5, melt_f_update=melt_f_update,
                                             melt_f_change=melt_f_change,
                                             tau_e_fold_yr=tau_e_fold_yr
                                             )
                else:
                    gd_mb = TIModel(gdir, 200,
                                    prcp_fac=1,
                                    mb_type=mb_type,
                                    grad_type=grad_type,
                                    baseline_climate=climate_type,
                                    )

                if not sfc_type_distinction and melt_f_change == 'neg_exp':
                    pass
                else:
                    if pf_cte_via == '_pf_via_winter_mb_log_fit':
                        path_folder = '/home/lilianschuster/Schreibtisch/PhD/Schuster_et_al_phd_paper_1/data/'
                        #pd_pf = pd.read_csv(
                        #    f'{path_folder}winter_prcp_mean_general_log_relation_pf_winter_mb_match.csv',
                        #    index_col='rgi_id')
                        pd_pf = pd.read_csv(f'{path_folder}winter_daily_prcp_mean_general_log_relation_pf_winter_mb_match.csv', index_col='rgi_id')
                        # old
                        # pf_cte = pd_pf.loc[gdir.rgi_id]['pf_via_log_regression']
                        hemisphere = gdir.hemisphere
                        # if NH ---
                        import xarray as xr
                        fpath_clim = gdir.get_filepath('climate_historical', filesuffix='_daily_W5E5')
                        ds_prcp = xr.open_dataset(fpath_clim).prcp
                        if hemisphere == 'nh':
                            ds_prcp_winter = ds_prcp.where(ds_prcp['time.month'].isin([10, 11, 12, 1, 2, 3, 4]),
                                                           drop=True)
                        else:
                            ds_prcp_winter = ds_prcp.where(ds_prcp['time.month'].isin([4, 5, 6, 7, 8, 9, 10]),
                                                           drop=True)
                        prcp_winter_daily_mean = ds_prcp_winter.mean().values  # kg m-2 day-1
                        def log_func(x, a, b):
                            r = a * np.log(x) + b
                            # don't allow extremely low/high prcp. factors!!!
                            if np.shape(r) == ():
                                if r > 10:
                                    r = 10
                                if r < 0.1:
                                    r = 0.1
                            else:
                                r[r > 10] = 10
                                r[r < 0.1] = 0.1
                            return r
                        # the log_func params are all the same over the columnspd_pf['a_log_multiplier'].iloc[0]
                        pf_cte = log_func(prcp_winter_daily_mean, pd_pf['a_log_multiplier'].iloc[0],
                                          pd_pf['b_intercept'].iloc[0])
                    elif pf_cte_via == '' or pf_cte_via == '_pf_cte_via_std':
                        if not sfc_type_distinction:
                            melt_f_change = np.NaN
                            if pf_cte_dict is False:
                                pf_cte = pf_cte_dict
                            else:
                                pf_cte = pf_cte_dict[f'sfc_type_False_{mb_type}_{grad_type}']
                        else:
                            if pf_cte_dict is False:
                                pf_cte = pf_cte_dict
                            else:
                                pf_cte = pf_cte_dict[f'{melt_f_update}_melt_f_update_sfc_type_{melt_f_change_r}_{mb_type}_{grad_type}']
                    for temp_bias in temp_b_range:
                        pd_calib.loc[j] = np.NaN
                        try:
                            out = calibrate_to_geodetic_bias_std_quot_fast(gd_mb, method=method,
                                                                            temp_bias=temp_bias,
                                                                            hgts=hgts,
                                                                            widths=widths,
                                                                            mb_geodetic=mb_geodetic,
                                                                            mbdf=mbdf,
                                                                            winter_mb_observed=winter_mb_observed,
                                                                            mean_mb_prof=mean_mb_prof,
                                                                            mb_type=mb_type,
                                                                            sfc_type_distinction=sfc_type_distinction,
                                                                            optimize_std_quot =optimize_std_quot,
                                                                            pf_cte=pf_cte)
                            (pd_calib.loc[j, 'pf_opt'], pd_calib.loc[j, 'melt_f'],
                             pd_calib.loc[j, 'winter_prcp_mean'], pd_calib.loc[j, 'winter_solid_prcp_mean'],
                             pd_calib.loc[j, 'specific_melt_winter_kg_m2'], pd_calib.loc[j, 'except_necessary'],
                             pd_calib.loc[j, 'quot_std'], pd_calib.loc[j, 'mae_mb_profile'],
                             pd_calib.loc[j, 'bias_winter_mb']) = out
                        except:
                            pass
                        pd_calib.loc[j, 'rgi_id'] = gdir.rgi_id
                        pd_calib.loc[j, 'temp_bias'] = temp_bias
                        pd_calib.loc[j, 'melt_f_change'] = melt_f_change
                        pd_calib.loc[j, 'mb_type'] = mb_type
                        pd_calib.loc[j, 'grad_type'] = grad_type
                        pd_calib.loc[j, 'tau_e_fold_yr'] = tau_e_fold_yr

                        j += 1
    # print(gdir.rgi_id)
    # except:
    # pass
    # pd_calib['rgi_id'] = gdir.rgi_id
    pd_calib['winter_prcp_mean_no_pf'] = pd_calib['winter_prcp_mean'] / pd_calib['pf_opt']
    pd_calib['winter_solid_prcp_mean_no_pf'] = pd_calib['winter_solid_prcp_mean'] / pd_calib['pf_opt']
    pd_calib['melt_f_update'] = melt_f_update
    pd_calib['sfc_type_distinction'] = sfc_type_distinction

    if path == 'return':
        return pd_calib
    else:
        if optimize_std_quot:
            if sfc_type_distinction is True:
                pd_calib.to_csv(f'{path}/calib_std_quot_{melt_f_update}_melt_f_update_{gdir.rgi_id}_method{method}.csv')
            else:
                pd_calib.to_csv(f'{path}/calib_std_quot_no_sfc_type_{gdir.rgi_id}_method{method}.csv')
        else:
            if sfc_type_distinction is True:
                pd_calib.to_csv(f'{path}/calib_only_geod_{melt_f_update}_melt_f_update{pf_cte_via}_{gdir.rgi_id}_method{method}.csv')
            else:
                pd_calib.to_csv(f'{path}/calib_only_geod_no_sfc_type{pf_cte_via}_{gdir.rgi_id}_method{method}.csv')


def calibrate_to_geodetic_bias_winter_mb(gdir,  # temp_b_range = np.arange(-3,3,1),
                                         mb_type='mb_monthly', grad_type='cte',
                                         melt_f_update='monthly',
                                         climate_type='W5E5', method='nested', temp_bias=0,
                                         melt_f_change='linear',
                                         sfc_type_distinction=True):

    """
    todo: make documentation

    Parameters
    ----------
    gdir
    mb_type
    grad_type
    melt_f_update
    climate_type
    method
    temp_bias

    Returns
    -------

    """
    if melt_f_update == 'annual':
        raise NotImplementedError('using annual melt_f_update is at the moment not compatible with'
                                  'estimating winter mass-balance!')
    pd_geodetic_all = utils.get_geodetic_mb_dataframe()
    pd_geodetic = pd_geodetic_all.loc[pd_geodetic_all.period == '2000-01-01_2020-01-01']
    mb_geodetic = pd_geodetic.loc[gdir.rgi_id].dmdtda * 1000
    if mb_type != 'mb_real_daily':
        input_fs = '_monthly_W5E5'
    else:
        input_fs = '_daily_W5E5'

    mbdf = gdir.get_ref_mb_data(input_filesuffix=input_fs)
    ys_glac = mbdf.index.values
    mb_glaciological = mbdf['ANNUAL_BALANCE']

    oggm_updated = False
    if oggm_updated:
        _, pathi = utils.get_wgms_files()
        pd_mb_overview = pd.read_csv(
            pathi[:-len('/mbdata')] + '/mb_overview_seasonal_mb_time_periods_20220301.csv',
            index_col='Unnamed: 0')
    else:
        # path_mbsandbox = MBsandbox.__file__[:-len('/__init__.py')]
        # pd_mb_overview = pd.read_csv(path_mbsandbox + '/data/mb_overview_seasonal_mb_time_periods_20220301.csv',
        #                            index_col='Unnamed: 0')
        # pd_wgms_data_stats = pd.read_csv(path_mbsandbox + '/data/wgms_data_stats_20220301.csv',
        #                                 index_col='Unnamed: 0')

        # path_mbsandbox = MBsandbox.__file__[:-len('/__init__.py')]
        # pd_mb_overview = pd.read_csv(path_mbsandbox + '/data/mb_overview_seasonal_mb_time_periods_20220301.csv',
        #                            index_col='Unnamed: 0')
        # pd_wgms_data_stats = pd.read_csv(path_mbsandbox + '/data/wgms_data_stats_20220301.csv',
        #                                 index_col='Unnamed: 0')
        fp = utils.file_downloader('https://cluster.klima.uni-bremen.de/~lschuster/ref_glaciers' +
                                   '/data/mb_overview_seasonal_mb_time_periods_20220301.csv')
        pd_mb_overview = pd.read_csv(fp, index_col='Unnamed: 0')
        fp_stats = utils.file_downloader('https://cluster.klima.uni-bremen.de/~lschuster/ref_glaciers' +
                                         '/data/wgms_data_stats_20220301.csv')
        pd_wgms_data_stats = pd.read_csv(fp_stats, index_col='Unnamed: 0')
    pd_mb_overview_sel_gdir = pd_mb_overview.loc[pd_mb_overview.rgi_id == gdir.rgi_id]
    pd_mb_overview_sel_gdir.index = pd_mb_overview_sel_gdir.Year
    yrs_seasonal_mbs = pd_mb_overview_sel_gdir.Year.values
    assert np.all(yrs_seasonal_mbs >= 1980)
    assert np.all(yrs_seasonal_mbs < 2020)
    # yrs_seasonal_mbs = gdir.get_ref_mb_data(input_filesuffix=input_fs)['SUMMER_BALANCE'].dropna().index.values
    # yrs_seasonal_mbs = yrs_seasonal_mbs[(yrs_seasonal_mbs >= 1980) & (yrs_seasonal_mbs < 2020)] # I can't use 1979 (as then we would need climate data in winter 1978!)
    winter_mb_observed = gdir.get_ref_mb_data(input_filesuffix=input_fs).loc[yrs_seasonal_mbs][
        'WINTER_BALANCE']
    #annual_mb_observed = gdir.get_ref_mb_data(input_filesuffix='_daily_W5E5').loc[yrs_seasonal_mbs][
    #    'ANNUAL_BALANCE']
    hgts, widths = gdir.get_inversion_flowline_hw()

    ###
    if sfc_type_distinction:
        gd_mb = TIModel_Sfc_Type(gdir, 200,
                                     prcp_fac=1,
                                     mb_type=mb_type,
                                     grad_type=grad_type,
                                     baseline_climate=climate_type,
                                     melt_f_ratio_snow_to_ice=0.5, melt_f_update=melt_f_update,
                                     melt_f_change=melt_f_change,
                                     )
    else:
        gd_mb = TIModel(gdir, 200,
                        prcp_fac=1,
                        mb_type=mb_type,
                        grad_type=grad_type,
                        baseline_climate=climate_type,
                         )
    gd_mb.temp_bias = temp_bias
    except_necessary = 0
    try:
        pf_opt = scipy.optimize.brentq(minimize_winter_mb_brentq_geod_via_pf, 0.1, 10, xtol=0.1,
                                       args=(
                                       gd_mb, mb_geodetic, winter_mb_observed, hgts, widths, yrs_seasonal_mbs,
                                       True)  # period_from_wgms
                                       )
    except:
        except_necessary += 1
        try:
            pf_opt = scipy.optimize.brentq(minimize_winter_mb_brentq_geod_via_pf, 0.4, 5, xtol=0.1,
                                           args=(gd_mb, mb_geodetic, winter_mb_observed, hgts, widths,
                                                 yrs_seasonal_mbs,
                                                 True)  # period_from_wgms
                                           )
        except:
            except_necessary += 1

            if method == 'nested':
                try:
                    # first try to see if the problem is on high or low pfs!
                    scipy.optimize.brentq(minimize_bias_geodetic, 10, 1000,
                                          xtol=0.01,
                                          args=(gd_mb, mb_geodetic, hgts, widths, 0.1),
                                          disp=True)

                    try:

                        pf_opt = scipy.optimize.brentq(minimize_winter_mb_brentq_geod_via_pf, 0.1, 2, xtol=0.1,
                                                       args=(
                                                       gd_mb, mb_geodetic, winter_mb_observed, hgts, widths,
                                                       yrs_seasonal_mbs,
                                                       True))  # period_from_wgms
                    except:
                        except_necessary += 1
                        try:
                            pf_opt = scipy.optimize.brentq(minimize_winter_mb_brentq_geod_via_pf, 0.1, 1, xtol=0.1,
                                                           args=(
                                                           gd_mb, mb_geodetic, winter_mb_observed, hgts, widths,
                                                           yrs_seasonal_mbs,
                                                           True)  # period_from_wgms
                                                           )
                        except:
                            except_necessary += 1
                            pf_opt = scipy.optimize.brentq(minimize_winter_mb_brentq_geod_via_pf, 0.3, 0.9,
                                                           xtol=0.1,
                                                           args=(
                                                           gd_mb, mb_geodetic, winter_mb_observed, hgts, widths,
                                                           yrs_seasonal_mbs,
                                                           True)  # period_from_wgms
                                                           )
                except:
                    try:
                        pf_opt = scipy.optimize.brentq(minimize_winter_mb_brentq_geod_via_pf, 2, 5, xtol=0.1,
                                                       args=(
                                                       gd_mb, mb_geodetic, winter_mb_observed, hgts, widths,
                                                       yrs_seasonal_mbs,
                                                       True)  # period_from_wgms
                                                       )
                    except:
                        pf_opt = scipy.optimize.brentq(minimize_winter_mb_brentq_geod_via_pf, 1, 3, xtol=0.1,
                                                       args=(
                                                       gd_mb, mb_geodetic, winter_mb_observed, hgts, widths,
                                                       yrs_seasonal_mbs,
                                                       True)  # period_from_wgms
                                                       )
            elif method == 'pre-check':
                melt_f_opt_dict = {}
                for pf in np.concatenate([np.arange(0.1, 3, 0.5), np.arange(3, 10, 2)]):
                    try:
                        melt_f = scipy.optimize.brentq(minimize_bias_geodetic, 10, 1000,
                                                       xtol=0.01,
                                                       args=(gd_mb, mb_geodetic,
                                                             hgts, widths, pf),
                                                       disp=True)
                        melt_f_opt_dict[pf] = melt_f
                    except:
                        pass
                #print(melt_f_opt_dict)
                pf_start = list(melt_f_opt_dict.items())[0][0]
                pf_end = list(melt_f_opt_dict.items())[-1][0]
                try:
                    pf_opt = scipy.optimize.brentq(minimize_winter_mb_brentq_geod_via_pf, pf_start, pf_end,
                                                   xtol=0.1,
                                                   args=(gd_mb, mb_geodetic, winter_mb_observed,
                                                         hgts, widths, yrs_seasonal_mbs,
                                                         True)  # period_from_wgms
                                                   )
                except:
                    except_necessary += 1
                    try:
                        pf_start = list(melt_f_opt_dict.items())[1][0]
                        pf_end = list(melt_f_opt_dict.items())[-2][0]
                        pf_opt = scipy.optimize.brentq(minimize_winter_mb_brentq_geod_via_pf, pf_start, pf_end,
                                                       xtol=0.1,
                                                       args=(gd_mb, mb_geodetic, winter_mb_observed,
                                                             hgts, widths, yrs_seasonal_mbs,
                                                             True)  # period_from_wgms
                                                       )
                    except:
                        except_necessary += 1
                        pf_start = list(melt_f_opt_dict.items())[2][0]
                        pf_end = list(melt_f_opt_dict.items())[-3][0]
                        pf_opt = scipy.optimize.brentq(minimize_winter_mb_brentq_geod_via_pf, pf_start, pf_end,
                                                       xtol=0.1,
                                                       args=(gd_mb, mb_geodetic, winter_mb_observed,
                                                             hgts, widths, yrs_seasonal_mbs,
                                                             True)  # period_from_wgms
                                                       )

    gd_mb.prcp_fac = pf_opt
    gd_mb.melt_f = scipy.optimize.brentq(minimize_bias_geodetic, 10, 1000,
                                             xtol=0.01,
                                             args=(gd_mb, mb_geodetic, hgts, widths, pf_opt),
                                             disp=True)

    # first precompute it for the entire time period
    # more practical also for std and MB profile computation
    # (like that also years in between (without observations) are estimated)
    # [only important for TIModel_Sfc_Type, if observations have missing years in between]
    gd_mb.get_specific_mb(heights=hgts, widths=widths,
                              year=np.arange(1979, # yrs_seasonal_mbs[0]
                                             2019 + 1, 1))
    # in case of HEF this should be the same !!! (as HEF always has WGMS seasonal MB from Oct 1st to April 30th)
    outi_right_period = gd_mb.get_specific_winter_mb(heights=hgts, year=yrs_seasonal_mbs, widths=widths,
                                                         add_climate=True,
                                                         period_from_wgms=True)
    np.testing.assert_allclose(outi_right_period[0].mean(),
                               winter_mb_observed.mean(), rtol=0.05)
    #outi = gd_mb.get_specific_winter_mb(heights=hgts, year=yrs_seasonal_mbs, widths=widths, add_climate=True,
    #                                    period_from_wgms=False)
    # for k,_ in enumerate(outi):
    #    np.testing.assert_allclose(outi[k],
    #                               outi_right_period[k])
    t, tfm, prcp, prcpsol = outi_right_period[1:]

    pd_prcp = pd.DataFrame(prcp).T
    pd_prcp.index.name = 'distance_along_flowline'
    pd_prcp.columns = yrs_seasonal_mbs
    winter_prcp_mean = pd_prcp.mean().mean()

    pd_solid_prcp = pd.DataFrame(prcpsol).T
    pd_solid_prcp.index.name = 'distance_along_flowline'
    pd_solid_prcp.columns = yrs_seasonal_mbs
    winter_solid_prcp_mean = np.average(pd_solid_prcp.mean(axis=1), weights=widths)

    if mb_type == 'mb_real_daily':
        fact = 12 / 365.25
    else:
        fact = 1

    pd_tfm = pd.DataFrame(tfm).T
    pd_tfm.index.name = 'distance_along_flowline'
    pd_tfm.columns = yrs_seasonal_mbs
    # how much minimum melt happened over winter months?
    if sfc_type_distinction:
        melt_w_month_kg_m2_2d = pd_tfm * fact * gd_mb.melt_f_buckets[0]
    else:
        melt_w_month_kg_m2_2d = pd_tfm * fact * gd_mb.melt_f

    specific_melt_winter = np.average(melt_w_month_kg_m2_2d.mean(axis=1), weights=widths)

    # also compute quot_std
    mod_std = gd_mb.get_specific_mb(heights=hgts, widths=widths,
                                    year=ys_glac).std()
    ref_std = mb_glaciological.loc[ys_glac].values.std()
    quot_std = mod_std / ref_std

    # now also compute mean absolute error of mean MB profile: (if available!)
    # observed MB profile
    # todo: optimize this that this is not called every time but just once!
    outi_prof = get_mean_mb_profile_filtered(gdir, input_fs=input_fs, obs_ratio_needed=0.6)
    if outi_prof is None:
        # no MB profile exist -> mae has to be np.NaN
        mae = np.NaN
    else:
        obs_mean_mb_profile_filtered, obs_mean_mb_profile_years = outi_prof
        # get modelled MB profile
        fac = cfg.SEC_IN_YEAR * cfg.PARAMS['ice_density']
        mb_annual = []
        if sfc_type_distinction:
            gd_mb.reset_pd_mb_bucket()
        for y in obs_mean_mb_profile_years:
            # if isinstance(gd_mb, TIModel):
            mb_y = gd_mb.get_annual_mb(hgts, y) * fac
            # print(h, mb_y)
            # else:
            #   mb_y = gd_mb.pd_mb_annual[y]
            mb_annual.append(mb_y)
        y_modelled = np.array(mb_annual).mean(axis=0)
        h_condi = obs_mean_mb_profile_filtered.index
        condi1 = ((hgts > min(h_condi)) & (hgts < max(h_condi)))
        f = scipy.interpolate.interp1d(h_condi,
                                       obs_mean_mb_profile_filtered.values,
                                       kind='cubic')
        y_interp_obs = f(hgts[condi1])
        mae = mean_absolute_error(y_interp_obs, y_modelled[condi1])

    return (gd_mb.prcp_fac, gd_mb.melt_f, winter_prcp_mean, winter_solid_prcp_mean,
            specific_melt_winter, except_necessary, quot_std, mae)



def calibrate_to_geodetic_bias_std_quot_fast(gd_mb,  # temp_b_range = np.arange(-3,3,1),
                                              hgts=None, widths=None,
                                              mb_geodetic = None,
                                              mbdf=None,
                                              winter_mb_observed=None,
                                              mean_mb_prof= None,
                                         mb_type='mb_monthly',  method='nested', temp_bias=0,
                                         sfc_type_distinction=True, optimize_std_quot=True, pf_cte = False):

    """
    todo: make documentation

    Parameters
    ----------
    gd_mb
    optimize_std_quot :
        default is True. If False, it will only calibrate to the geodetic bias. In this case, you have to set a "global"
        precipitation factor.
    Returns
    -------

    """


    ###
    ys_glac = mbdf.index.values
    mb_glaciological = mbdf['ANNUAL_BALANCE']
    yrs_seasonal_mbs = winter_mb_observed.index

    gd_mb.temp_bias = temp_bias
    except_necessary = 0
    if optimize_std_quot:
        try:
            pf_opt = scipy.optimize.brentq(optimize_std_quot_brentq_geod, 0.1, 10, xtol=0.01,
                                           args=(
                                           gd_mb, mb_geodetic, mb_glaciological, hgts, widths, ys_glac,
                                           )  # period_from_wgms
                                           )
        except:
            except_necessary += 1
            try:
                pf_opt = scipy.optimize.brentq(optimize_std_quot_brentq_geod, 0.4, 5, xtol=0.01,
                                               args=(gd_mb, mb_geodetic, mb_glaciological, hgts, widths, ys_glac)  # period_from_wgms
                                               )
            except:
                except_necessary += 1

                if method == 'nested':
                    try:
                        # first try to see if the problem is on high or low pfs!
                        scipy.optimize.brentq(minimize_bias_geodetic, 10, 1000,
                                              xtol=0.01,
                                              args=(gd_mb, mb_geodetic, hgts, widths, 0.01),
                                              disp=True)

                        try:

                            pf_opt = scipy.optimize.brentq(optimize_std_quot_brentq_geod, 0.1, 2, xtol=0.01,
                                                           args=(
                                                           gd_mb, mb_geodetic, mb_glaciological, hgts, widths, ys_glac)
                                                           # period_from_wgms
                                                           )  # period_from_wgms
                        except:
                            except_necessary += 1
                            try:
                                pf_opt = scipy.optimize.brentq(optimize_std_quot_brentq_geod, 0.1, 1, xtol=0.01,
                                                               args=(
                                                               gd_mb, mb_geodetic, mb_glaciological, hgts, widths, ys_glac)
                                                               # period_from_wgms
                                                               # period_from_wgms
                                                               )
                            except:
                                except_necessary += 1
                                pf_opt = scipy.optimize.brentq(optimize_std_quot_brentq_geod, 0.3, 0.9,
                                                               xtol=0.01,
                                                               args=(
                                                               gd_mb, mb_geodetic, mb_glaciological, hgts, widths, ys_glac)
                                                               # period_from_wgms
                                                               # period_from_wgms
                                                               )
                    except:
                        try:
                            pf_opt = scipy.optimize.brentq(optimize_std_quot_brentq_geod, 2, 5, xtol=0.01,
                                                           args=(
                                                           gd_mb, mb_geodetic, mb_glaciological, hgts, widths, ys_glac)
                                                           # period_from_wgms
                                                           # period_from_wgms
                                                           )
                        except:
                            pf_opt = scipy.optimize.brentq(optimize_std_quot_brentq_geod, 1, 3, xtol=0.01,
                                                           args=(
                                                           gd_mb, mb_geodetic, mb_glaciological, hgts, widths, ys_glac)
                                                           # period_from_wgms
                                                           # period_from_wgms
                                                           )
                elif method == 'pre-check':
                    melt_f_opt_dict = {}
                    for pf in np.concatenate([np.arange(0.1, 3, 0.5), np.arange(3, 10, 2)]):
                        try:
                            melt_f = scipy.optimize.brentq(minimize_bias_geodetic, 10, 1000,
                                                           xtol=0.01,
                                                           args=(gd_mb, mb_geodetic,
                                                                 hgts, widths, pf),
                                                           disp=True)
                            melt_f_opt_dict[pf] = melt_f
                        except:
                            pass
                    #print(melt_f_opt_dict)
                    pf_start = list(melt_f_opt_dict.items())[0][0]
                    pf_end = list(melt_f_opt_dict.items())[-1][0]
                    try:
                        pf_opt = scipy.optimize.brentq(optimize_std_quot_brentq_geod, pf_start, pf_end,
                                                       xtol=0.01,
                                                       args=(gd_mb, mb_geodetic, mb_glaciological, hgts, widths, ys_glac)
                                                       # period_from_wgms
                                                       # period_from_wgms
                                                       )
                    except:
                        except_necessary += 1
                        try:
                            pf_start = list(melt_f_opt_dict.items())[1][0]
                            pf_end = list(melt_f_opt_dict.items())[-2][0]
                            pf_opt = scipy.optimize.brentq(optimize_std_quot_brentq_geod, pf_start, pf_end,
                                                           xtol=0.01,
                                                           args=(
                                                           gd_mb, mb_geodetic, mb_glaciological, hgts, widths, ys_glac)
                                                           # period_from_wgms
                                                           # period_from_wgms
                                                           )
                        except:
                            except_necessary += 1
                            pf_start = list(melt_f_opt_dict.items())[2][0]
                            pf_end = list(melt_f_opt_dict.items())[-3][0]
                            pf_opt = scipy.optimize.brentq(optimize_std_quot_brentq_geod, pf_start, pf_end,
                                                           xtol=0.01,
                                                           args=(
                                                           gd_mb, mb_geodetic, mb_glaciological, hgts, widths, ys_glac)
                                                           # period_from_wgms
                                                           # period_from_wgms
                                                           )
    else:
        except_necessary = 0
        try:
            assert type(pf_cte) == float, 'pf_cte has to be set and has to be a float!'
        except:
            assert type(pf_cte) == np.float64, 'pf_cte has to be set and has to be a float!'
        pf_opt = pf_cte

    gd_mb.prcp_fac = pf_opt
    try:
        gd_mb.melt_f = scipy.optimize.brentq(minimize_bias_geodetic, 10, 1000,
                                                 xtol=0.01,
                                                 args=(gd_mb, mb_geodetic, hgts, widths, pf_opt),
                                                 disp=True)
    except:
        gd_mb.melt_f = scipy.optimize.brentq(minimize_bias_geodetic, 100, 400,
                                             xtol=0.01,
                                             args=(gd_mb, mb_geodetic, hgts, widths, pf_opt),
                                             disp=True)

    # first precompute it for the entire time period
    # more practical also for std and MB profile computation
    # (like that also years in between (without observations) are estimated)
    # [only important for TIModel_Sfc_Type, if observations have missing years in between]
    gd_mb.get_specific_mb(heights=hgts, widths=widths,
                          year=np.arange(1979, # yrs_seasonal_mbs[0]
                                         2019 + 1, 1))
    try:
        melt_f_update = gd_mb.melt_f_update
    except:
        melt_f_update = np.NaN
    if melt_f_update != 'annual':
        # in case of HEF this should be the same !!! (as HEF always has WGMS seasonal MB from Oct 1st to April 30th)
        try:
            outi_right_period = gd_mb.get_specific_winter_mb(heights=hgts, year=yrs_seasonal_mbs, widths=widths,
                                                                 add_climate=True,
                                                                 period_from_wgms=True)


            ## estimate winter mb bias and save it (not the absolute value!!!)
            winter_mb_bias = outi_right_period[0].mean() - winter_mb_observed.mean()

            t, tfm, prcp, prcpsol = outi_right_period[1:]

            pd_prcp = pd.DataFrame(prcp).T
            pd_prcp.index.name = 'distance_along_flowline'
            pd_prcp.columns = yrs_seasonal_mbs
            winter_prcp_mean = pd_prcp.mean().mean()

            pd_solid_prcp = pd.DataFrame(prcpsol).T
            pd_solid_prcp.index.name = 'distance_along_flowline'
            pd_solid_prcp.columns = yrs_seasonal_mbs
            winter_solid_prcp_mean = np.average(pd_solid_prcp.mean(axis=1), weights=widths)

            if mb_type == 'mb_real_daily':
                fact = 12 / 365.25
            else:
                fact = 1

            pd_tfm = pd.DataFrame(tfm).T
            pd_tfm.index.name = 'distance_along_flowline'
            pd_tfm.columns = yrs_seasonal_mbs
            # how much minimum melt happened over winter months?
            if sfc_type_distinction:
                melt_w_month_kg_m2_2d = pd_tfm * fact * gd_mb.melt_f_buckets[0]
            else:
                melt_w_month_kg_m2_2d = pd_tfm * fact * gd_mb.melt_f

            specific_melt_winter = np.average(melt_w_month_kg_m2_2d.mean(axis=1), weights=widths)
        except:
            winter_prcp_mean = np.NaN
            winter_solid_prcp_mean = np.NaN
            specific_melt_winter = np.NaN
            winter_mb_bias = np.NaN
    else:
        winter_prcp_mean = np.NaN
        winter_solid_prcp_mean = np.NaN
        specific_melt_winter = np.NaN
        winter_mb_bias = np.NaN

    # also compute quot_std
    mod_std = gd_mb.get_specific_mb(heights=hgts, widths=widths,
                                    year=ys_glac).std()
    ref_std = mb_glaciological.loc[ys_glac].values.std()
    quot_std = mod_std / ref_std
    if optimize_std_quot:
        # check if std. quotient is calibrated right
        np.testing.assert_allclose(quot_std, 1, rtol=0.01)
    # now also compute mean absolute error of mean MB profile: (if available!)
    # observed MB profile
    # todo: optimize this that this is not called every time but just once!
    if mean_mb_prof is None:
        # no MB profile exist -> mae has to be np.NaN
        mae = np.NaN
    else:
        obs_mean_mb_profile_filtered, obs_mean_mb_profile_years = mean_mb_prof
        # get modelled MB profile
        fac = cfg.SEC_IN_YEAR * cfg.PARAMS['ice_density']
        mb_annual = []
        if sfc_type_distinction:
            gd_mb.reset_pd_mb_bucket()
        for y in obs_mean_mb_profile_years:
            # if isinstance(gd_mb, TIModel):
            mb_y = gd_mb.get_annual_mb(hgts, y) * fac
            # print(h, mb_y)
            # else:
            #   mb_y = gd_mb.pd_mb_annual[y]
            mb_annual.append(mb_y)
        y_modelled = np.array(mb_annual).mean(axis=0)
        h_condi = obs_mean_mb_profile_filtered.index
        condi1 = ((hgts > min(h_condi)) & (hgts < max(h_condi)))
        f = scipy.interpolate.interp1d(h_condi,
                                       obs_mean_mb_profile_filtered.values,
                                       kind='cubic')
        y_interp_obs = f(hgts[condi1])
        mae = mean_absolute_error(y_interp_obs, y_modelled[condi1])
    return (gd_mb.prcp_fac, gd_mb.melt_f, winter_prcp_mean, winter_solid_prcp_mean,
            specific_melt_winter, except_necessary, quot_std, mae, winter_mb_bias)


def calibrate_to_geodetic_bias_winter_mb_fast(gd_mb,  # temp_b_range = np.arange(-3,3,1),
                                              hgts=None, widths=None,
                                              mb_geodetic = None,
                                              mbdf=None,
                                              winter_mb_observed=None,
                                              mean_mb_prof= None,
                                         mb_type='mb_monthly',  method='nested', temp_bias=0,
                                         sfc_type_distinction=True):

    """
    todo: make documentation

    Parameters
    ----------
    gd_mb

    Returns
    -------

    """


    ###
    ys_glac = mbdf.index.values
    mb_glaciological = mbdf['ANNUAL_BALANCE']
    yrs_seasonal_mbs = winter_mb_observed.index

    gd_mb.temp_bias = temp_bias
    except_necessary = 0
    try:
        pf_opt = scipy.optimize.brentq(minimize_winter_mb_brentq_geod_via_pf, 0.1, 10, xtol=0.1,
                                       args=(
                                       gd_mb, mb_geodetic, winter_mb_observed, hgts, widths, yrs_seasonal_mbs,
                                       True)  # period_from_wgms
                                       )
    except:
        except_necessary += 1
        try:
            pf_opt = scipy.optimize.brentq(minimize_winter_mb_brentq_geod_via_pf, 0.4, 5, xtol=0.1,
                                           args=(gd_mb, mb_geodetic, winter_mb_observed, hgts, widths,
                                                 yrs_seasonal_mbs,
                                                 True)  # period_from_wgms
                                           )
        except:
            except_necessary += 1

            if method == 'nested':
                try:
                    # first try to see if the problem is on high or low pfs!
                    scipy.optimize.brentq(minimize_bias_geodetic, 10, 1000,
                                          xtol=0.01,
                                          args=(gd_mb, mb_geodetic, hgts, widths, 0.1),
                                          disp=True)

                    try:

                        pf_opt = scipy.optimize.brentq(minimize_winter_mb_brentq_geod_via_pf, 0.1, 2, xtol=0.1,
                                                       args=(
                                                       gd_mb, mb_geodetic, winter_mb_observed, hgts, widths,
                                                       yrs_seasonal_mbs,
                                                       True))  # period_from_wgms
                    except:
                        except_necessary += 1
                        try:
                            pf_opt = scipy.optimize.brentq(minimize_winter_mb_brentq_geod_via_pf, 0.1, 1, xtol=0.1,
                                                           args=(
                                                           gd_mb, mb_geodetic, winter_mb_observed, hgts, widths,
                                                           yrs_seasonal_mbs,
                                                           True)  # period_from_wgms
                                                           )
                        except:
                            except_necessary += 1
                            pf_opt = scipy.optimize.brentq(minimize_winter_mb_brentq_geod_via_pf, 0.3, 0.9,
                                                           xtol=0.1,
                                                           args=(
                                                           gd_mb, mb_geodetic, winter_mb_observed, hgts, widths,
                                                           yrs_seasonal_mbs,
                                                           True)  # period_from_wgms
                                                           )
                except:
                    try:
                        pf_opt = scipy.optimize.brentq(minimize_winter_mb_brentq_geod_via_pf, 2, 5, xtol=0.1,
                                                       args=(
                                                       gd_mb, mb_geodetic, winter_mb_observed, hgts, widths,
                                                       yrs_seasonal_mbs,
                                                       True)  # period_from_wgms
                                                       )
                    except:
                        pf_opt = scipy.optimize.brentq(minimize_winter_mb_brentq_geod_via_pf, 1, 3, xtol=0.1,
                                                       args=(
                                                       gd_mb, mb_geodetic, winter_mb_observed, hgts, widths,
                                                       yrs_seasonal_mbs,
                                                       True)  # period_from_wgms
                                                       )
            elif method == 'pre-check':
                melt_f_opt_dict = {}
                for pf in np.concatenate([np.arange(0.1, 3, 0.5), np.arange(3, 10, 2)]):
                    try:
                        melt_f = scipy.optimize.brentq(minimize_bias_geodetic, 10, 1000,
                                                       xtol=0.01,
                                                       args=(gd_mb, mb_geodetic,
                                                             hgts, widths, pf),
                                                       disp=True)
                        melt_f_opt_dict[pf] = melt_f
                    except:
                        pass
                #print(melt_f_opt_dict)
                pf_start = list(melt_f_opt_dict.items())[0][0]
                pf_end = list(melt_f_opt_dict.items())[-1][0]
                try:
                    pf_opt = scipy.optimize.brentq(minimize_winter_mb_brentq_geod_via_pf, pf_start, pf_end,
                                                   xtol=0.1,
                                                   args=(gd_mb, mb_geodetic, winter_mb_observed,
                                                         hgts, widths, yrs_seasonal_mbs,
                                                         True)  # period_from_wgms
                                                   )
                except:
                    except_necessary += 1
                    try:
                        pf_start = list(melt_f_opt_dict.items())[1][0]
                        pf_end = list(melt_f_opt_dict.items())[-2][0]
                        pf_opt = scipy.optimize.brentq(minimize_winter_mb_brentq_geod_via_pf, pf_start, pf_end,
                                                       xtol=0.1,
                                                       args=(gd_mb, mb_geodetic, winter_mb_observed,
                                                             hgts, widths, yrs_seasonal_mbs,
                                                             True)  # period_from_wgms
                                                       )
                    except:
                        except_necessary += 1
                        pf_start = list(melt_f_opt_dict.items())[2][0]
                        pf_end = list(melt_f_opt_dict.items())[-3][0]
                        pf_opt = scipy.optimize.brentq(minimize_winter_mb_brentq_geod_via_pf, pf_start, pf_end,
                                                       xtol=0.1,
                                                       args=(gd_mb, mb_geodetic, winter_mb_observed,
                                                             hgts, widths, yrs_seasonal_mbs,
                                                             True)  # period_from_wgms
                                                       )

    gd_mb.prcp_fac = pf_opt
    gd_mb.melt_f = scipy.optimize.brentq(minimize_bias_geodetic, 10, 1000,
                                             xtol=0.01,
                                             args=(gd_mb, mb_geodetic, hgts, widths, pf_opt),
                                             disp=True)

    # first precompute it for the entire time period
    # more practical also for std and MB profile computation
    # (like that also years in between (without observations) are estimated)
    # [only important for TIModel_Sfc_Type, if observations have missing years in between]
    gd_mb.get_specific_mb(heights=hgts, widths=widths,
                          year=np.arange(1979, # yrs_seasonal_mbs[0]
                                             2019 + 1, 1))
    # in case of HEF this should be the same !!! (as HEF always has WGMS seasonal MB from Oct 1st to April 30th)
    outi_right_period = gd_mb.get_specific_winter_mb(heights=hgts, year=yrs_seasonal_mbs, widths=widths,
                                                         add_climate=True,
                                                         period_from_wgms=True)
    np.testing.assert_allclose(outi_right_period[0].mean(),
                               winter_mb_observed.mean(), rtol=0.05)
    #outi = gd_mb.get_specific_winter_mb(heights=hgts, year=yrs_seasonal_mbs, widths=widths, add_climate=True,
    #                                    period_from_wgms=False)
    # for k,_ in enumerate(outi):
    #    np.testing.assert_allclose(outi[k],
    #                               outi_right_period[k])
    t, tfm, prcp, prcpsol = outi_right_period[1:]

    pd_prcp = pd.DataFrame(prcp).T
    pd_prcp.index.name = 'distance_along_flowline'
    pd_prcp.columns = yrs_seasonal_mbs
    winter_prcp_mean = pd_prcp.mean().mean()

    pd_solid_prcp = pd.DataFrame(prcpsol).T
    pd_solid_prcp.index.name = 'distance_along_flowline'
    pd_solid_prcp.columns = yrs_seasonal_mbs
    winter_solid_prcp_mean = np.average(pd_solid_prcp.mean(axis=1), weights=widths)

    if mb_type == 'mb_real_daily':
        fact = 12 / 365.25
    else:
        fact = 1

    pd_tfm = pd.DataFrame(tfm).T
    pd_tfm.index.name = 'distance_along_flowline'
    pd_tfm.columns = yrs_seasonal_mbs
    # how much minimum melt happened over winter months?
    if sfc_type_distinction:
        melt_w_month_kg_m2_2d = pd_tfm * fact * gd_mb.melt_f_buckets[0]
    else:
        melt_w_month_kg_m2_2d = pd_tfm * fact * gd_mb.melt_f

    specific_melt_winter = np.average(melt_w_month_kg_m2_2d.mean(axis=1), weights=widths)

    # also compute quot_std
    mod_std = gd_mb.get_specific_mb(heights=hgts, widths=widths,
                                    year=ys_glac).std()
    ref_std = mb_glaciological.loc[ys_glac].values.std()
    quot_std = mod_std / ref_std

    # now also compute mean absolute error of mean MB profile: (if available!)
    # observed MB profile
    # todo: optimize this that this is not called every time but just once!
    if mean_mb_prof is None:
        # no MB profile exist -> mae has to be np.NaN
        mae = np.NaN
    else:
        obs_mean_mb_profile_filtered, obs_mean_mb_profile_years = mean_mb_prof
        # get modelled MB profile
        fac = cfg.SEC_IN_YEAR * cfg.PARAMS['ice_density']
        mb_annual = []
        if sfc_type_distinction:
            gd_mb.reset_pd_mb_bucket()
        for y in obs_mean_mb_profile_years:
            # if isinstance(gd_mb, TIModel):
            mb_y = gd_mb.get_annual_mb(hgts, y) * fac
            # print(h, mb_y)
            # else:
            #   mb_y = gd_mb.pd_mb_annual[y]
            mb_annual.append(mb_y)
        y_modelled = np.array(mb_annual).mean(axis=0)
        h_condi = obs_mean_mb_profile_filtered.index
        condi1 = ((hgts > min(h_condi)) & (hgts < max(h_condi)))
        f = scipy.interpolate.interp1d(h_condi,
                                       obs_mean_mb_profile_filtered.values,
                                       kind='cubic')
        y_interp_obs = f(hgts[condi1])
        mae = mean_absolute_error(y_interp_obs, y_modelled[condi1])

    return (gd_mb.prcp_fac, gd_mb.melt_f, winter_prcp_mean, winter_solid_prcp_mean,
            specific_melt_winter, except_necessary, quot_std, mae)

@entity_task(log)
def calibrate_to_geodetic_bias_winter_mb_different_temp_bias(gdir,
                                                             temp_b_range = np.arange(-4, 4.1, 2),  # np.arange(-6,6.1,0.5)
                                                             method='pre-check', melt_f_update='monthly',
                                                             sfc_type_distinction=True,
                                                             path = '/home/lilianschuster/Schreibtisch/PhD/bayes_2022/calib_winter_mb/test_run'):
    ''' is probably deprecated ... todo: do documentation '''
    j = 0
    # sfc_type_distinction types
    n = 2
    if not sfc_type_distinction:
        melt_f_update = np.NaN
        n = 1

    pd_calib = pd.DataFrame(np.NaN, index=np.arange(0, int(3*2*n*len(temp_b_range))), #*len(gdirs))),
                            columns=['rgi_id', 'temp_bias', 'pf_opt', 'melt_f',
                                 'winter_prcp_mean', 'winter_solid_prcp_mean',
                                 'specific_melt_winter_kg_m2', 'except_necessary', 'quot_std', 'mae_mb_profile',
                                     'mb_type', 'grad_type', 'melt_f_change', 'melt_f_update'])
    for mb_type in ['mb_monthly', 'mb_pseudo_daily', 'mb_real_daily']:
        #print(mb_type)
        for grad_type in ['cte', 'var_an_cycle']:
            for melt_f_change in ['linear', 'neg_exp']:
                if not sfc_type_distinction and melt_f_change == 'neg_exp':
                    pass
                else:
                    if not sfc_type_distinction:
                        melt_f_change = np.NaN
                    for temp_bias in temp_b_range:
                        pd_calib.loc[j] = np.NaN

                        try:
                            out = calibrate_to_geodetic_bias_winter_mb(gdir, method = method,
                                                                       temp_bias = temp_bias, mb_type=mb_type,
                                                                       grad_type=grad_type,
                                                         melt_f_update=melt_f_update,
                                                         melt_f_change=melt_f_change,
                                                                      sfc_type_distinction=sfc_type_distinction)
                            (pd_calib.loc[j,'pf_opt'], pd_calib.loc[j,'melt_f'],
                             pd_calib.loc[j, 'winter_prcp_mean'], pd_calib.loc[j, 'winter_solid_prcp_mean'],
                             pd_calib.loc[j, 'specific_melt_winter_kg_m2'], pd_calib.loc[j,'except_necessary'],
                             pd_calib.loc[j, 'quot_std'], pd_calib.loc[j, 'mae_mb_profile']) = out
                        except:
                            pass
                        pd_calib.loc[j, 'rgi_id'] = gdir.rgi_id
                        pd_calib.loc[j, 'temp_bias'] = temp_bias
                        pd_calib.loc[j, 'melt_f_change'] = melt_f_change
                        pd_calib.loc[j, 'mb_type'] = mb_type
                        pd_calib.loc[j, 'grad_type'] = grad_type

                        j += 1
    #print(gdir.rgi_id)
                #except:
                #pass
            #pd_calib['rgi_id'] = gdir.rgi_id
    pd_calib['winter_prcp_mean_no_pf'] = pd_calib['winter_prcp_mean']/pd_calib['pf_opt']
    pd_calib['winter_solid_prcp_mean_no_pf'] = pd_calib['winter_solid_prcp_mean']/pd_calib['pf_opt']
    pd_calib['melt_f_update'] = melt_f_update
    pd_calib['sfc_type_distinction'] = sfc_type_distinction

    if sfc_type_distinction is True:
        pd_calib.to_csv(f'{path}/calib_winter_mb_monthly_melt_f_update_{gdir.rgi_id}_method{method}.csv')
    else:
        pd_calib.to_csv(f'{path}/calib_winter_mb_monthly_no_sfc_type_{gdir.rgi_id}_method{method}.csv')

@entity_task(log)
def calibrate_to_geodetic_bias_winter_mb_different_temp_bias_fast(gdir,
                                                             temp_b_range = np.arange(-4, 4.1, 2),  # np.arange(-6,6.1,0.5)
                                                             method='pre-check', melt_f_update='monthly',
                                                             sfc_type_distinction=True,
                                                             path = '/home/lilianschuster/Schreibtisch/PhD/bayes_2022/calib_winter_mb/test_run'):
    ''' todo: do documentation
    here only monthly melt_f_update  makes sense!!!
    '''
    j = 0
    # sfc_type_distinction types
    n = 2
    if not sfc_type_distinction:
        melt_f_update = np.NaN
        n = 1

    pd_geodetic_all = utils.get_geodetic_mb_dataframe()
    pd_geodetic = pd_geodetic_all.loc[pd_geodetic_all.period == '2000-01-01_2020-01-01']
    mb_geodetic = pd_geodetic.loc[gdir.rgi_id].dmdtda * 1000

    climate_type = 'W5E5'
    # actually it does not matter which climate input fs we use as long it exists (they all have the same time period)
    try:
        mbdf = gdir.get_ref_mb_data(input_filesuffix='_monthly_W5E5')
    except:
        mbdf = gdir.get_ref_mb_data(input_filesuffix='_daily_W5E5')



    # get the filtered seasonal MB data, if it is not available or has less than 5 measurements
    # just take as years the annual years and as values NaNs
    oggm_updated = False
    if oggm_updated:
        _, pathi = utils.get_wgms_files()
        pd_mb_overview = pd.read_csv(
            pathi[:-len('/mbdata')] + '/mb_overview_seasonal_mb_time_periods_20220301.csv',
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
    pd_mb_overview = pd_mb_overview[pd_mb_overview['at_least_5_winter_mb']]
    try:
        pd_mb_overview_sel_gdir = pd_mb_overview.loc[pd_mb_overview.rgi_id == gdir.rgi_id]
        pd_mb_overview_sel_gdir.index = pd_mb_overview_sel_gdir.Year
        yrs_seasonal_mbs = pd_mb_overview_sel_gdir.Year.values
        assert np.all(yrs_seasonal_mbs >= 1980)
        assert np.all(yrs_seasonal_mbs < 2020)
    except:
        # just take the years where annual MB exists (even if no winter MB exist)--> get np.NaN for winter_mb_observed!
        yrs_seasonal_mbs = mbdf.index
        # actually it does not matter which climate input fs we use as long it exists (they all have the same time period)
    try:
        winter_mb_observed = gdir.get_ref_mb_data(input_filesuffix='_monthly_W5E5').loc[yrs_seasonal_mbs][
            'WINTER_BALANCE']
        mean_mb_prof = get_mean_mb_profile_filtered(gdir, input_fs='_monthly_W5E5', obs_ratio_needed=0.6)

    except:
        winter_mb_observed = gdir.get_ref_mb_data(input_filesuffix='_daily_W5E5').loc[yrs_seasonal_mbs][
            'WINTER_BALANCE']
        mean_mb_prof = get_mean_mb_profile_filtered(gdir, input_fs='_daily_W5E5', obs_ratio_needed=0.6)

    hgts, widths = gdir.get_inversion_flowline_hw()

    pd_calib = pd.DataFrame(np.NaN, index=np.arange(0, int(3*2*n*len(temp_b_range))), #*len(gdirs))),
                            columns=['rgi_id', 'temp_bias', 'pf_opt', 'melt_f',
                                 'winter_prcp_mean', 'winter_solid_prcp_mean',
                                 'specific_melt_winter_kg_m2', 'except_necessary', 'quot_std', 'mae_mb_profile',
                                     'mb_type', 'grad_type', 'melt_f_change', 'melt_f_update', 'tau_e_fold_yr'])
    for mb_type in ['mb_monthly', 'mb_pseudo_daily', 'mb_real_daily']:
        #print(mb_type)
        for grad_type in ['cte', 'var_an_cycle']:
            for melt_f_change_r in ['linear', 'neg_exp_t0.5yr', 'neg_exp_t1yr']:
                if 'neg_exp' in melt_f_change_r:
                    melt_f_change = 'neg_exp'
                    if melt_f_change_r == 'neg_exp_t0.5yr':
                        tau_e_fold_yr = 0.5
                    elif melt_f_change_r == 'neg_exp_t1yr':
                        tau_e_fold_yr = 1
                else:
                    melt_f_change = melt_f_change_r
                    tau_e_fold_yr = np.NaN
                if sfc_type_distinction:
                    gd_mb = TIModel_Sfc_Type(gdir, 200,
                                             prcp_fac=1,
                                             mb_type=mb_type,
                                             grad_type=grad_type,
                                             baseline_climate=climate_type,
                                             melt_f_ratio_snow_to_ice=0.5, melt_f_update=melt_f_update,
                                             melt_f_change=melt_f_change,
                                             tau_e_fold_yr=tau_e_fold_yr
                                             )
                else:
                    gd_mb = TIModel(gdir, 200,
                                    prcp_fac=1,
                                    mb_type=mb_type,
                                    grad_type=grad_type,
                                    baseline_climate=climate_type,
                                    )

                if not sfc_type_distinction and melt_f_change == 'neg_exp':
                    pass
                else:
                    if not sfc_type_distinction:
                        melt_f_change = np.NaN
                    for temp_bias in temp_b_range:
                        pd_calib.loc[j] = np.NaN

                        try:
                            out = calibrate_to_geodetic_bias_winter_mb_fast(gd_mb, method = method,
                                                                       temp_bias = temp_bias,
                                                                            hgts=hgts,
                                                                            widths=widths,
                                                                            mb_geodetic = mb_geodetic,
                                                                            mbdf = mbdf,
                                                                            winter_mb_observed=winter_mb_observed,
                                                                            mean_mb_prof= mean_mb_prof,
                                                                            mb_type=mb_type,
                                                                      sfc_type_distinction=sfc_type_distinction)
                            (pd_calib.loc[j,'pf_opt'], pd_calib.loc[j,'melt_f'],
                             pd_calib.loc[j, 'winter_prcp_mean'], pd_calib.loc[j, 'winter_solid_prcp_mean'],
                             pd_calib.loc[j, 'specific_melt_winter_kg_m2'], pd_calib.loc[j,'except_necessary'],
                             pd_calib.loc[j, 'quot_std'], pd_calib.loc[j, 'mae_mb_profile']) = out
                        except:
                            pass
                        pd_calib.loc[j, 'rgi_id'] = gdir.rgi_id
                        pd_calib.loc[j, 'temp_bias'] = temp_bias
                        pd_calib.loc[j, 'melt_f_change'] = melt_f_change
                        pd_calib.loc[j, 'mb_type'] = mb_type
                        pd_calib.loc[j, 'grad_type'] = grad_type
                        pd_calib.loc[j, 'tau_e_fold_yr'] = tau_e_fold_yr

                        j += 1
    #print(gdir.rgi_id)
                #except:
                #pass
            #pd_calib['rgi_id'] = gdir.rgi_id
    pd_calib['winter_prcp_mean_no_pf'] = pd_calib['winter_prcp_mean']/pd_calib['pf_opt']
    pd_calib['winter_solid_prcp_mean_no_pf'] = pd_calib['winter_solid_prcp_mean']/pd_calib['pf_opt']
    pd_calib['melt_f_update'] = melt_f_update
    pd_calib['sfc_type_distinction'] = sfc_type_distinction

    if path == 'return':
        return pd_calib
    else:
        if sfc_type_distinction is True:
            pd_calib.to_csv(f'{path}/calib_winter_mb_monthly_melt_f_update_{gdir.rgi_id}_method{method}.csv')
        else:
            pd_calib.to_csv(f'{path}/calib_winter_mb_no_sfc_type_{gdir.rgi_id}_method{method}.csv')



@entity_task(log)
def melt_f_calib_geod_prep_inversion(gdir, mb_type='mb_monthly', grad_type='cte',
                                     pf=None, climate_type='W5E5',
                                     residual=0, ye=None,
                                     mb_model_sub_class=TIModel,
                                     kwargs_for_TIModel_Sfc_Type={},
                                     spinup=True,
                                     min_melt_f=10,
                                     max_melt_f=1000,
                                     step_height_for_corr=25,
                                     max_height_change_for_corr=3000,
                                     ):
    """ calibrates the melt factor using the TIModel mass-balance model,
    computes the apparent mass balance for the inversion
    and saves the melt_f, the applied pf and the possible climate data
    ref_hgt correction into a json file inside of the glacier directory

    This has to be run before e.g. `run_from_climate_data_TIModel` !

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory to process
    mb_type: str
            three types: 'mb_pseudo_daily' (default: use temp_std and N percentiles),
            'mb_monthly' (same as default OGGM PastMassBalance),
            'mb_real_daily' (use daily temperature values).
            the mb_type only work if the baseline_climate of gdir is right
    grad_type : str
            three types of applying the temperature gradient:
            'cte' (default, constant lapse rate, set to default_grad,
                   same as in default OGGM)
            'var_an_cycle' (varies spatially and over annual cycle,
                            but constant over the years)
            'var' (varies spatially & temporally as in the climate files, deprecated!)
    pf : float
        multiplicative precipitation factor, has to be calibrated for each option and
        each climate dataset,
        (no default value, has to be set)
    climate_type : str
        which climate to use, default is 'W5E5'
        (the one that has been preprocessed before)
    residual : float
        default is zero, do not change this !
    ye : int
        the last year +1 ! that should be used for calibration,
        for W5E5, default is 2020, as the climate data goes till end 2019
    mb_model_sub_class : class, optional
            the mass-balance model to use: either TIModel (default)
            or TIModel_Sfc_Type
    kwargs_for_TIModel_Sfc_Type : {}
        stuff passed to the TIModel_Sfc_Type instance!
    spinup : bool, optional
        default is True, only used
        when mb_model_sub_class is TIModel_Sfc_Type
    min_melt_f: float, optional
        minimum melt factor that is allowed
        default is 10, can also be set higher (e.g. 60)
        unit is (kg /m² /mth /K)
    max_melt_f: float, optional
        minimum melt factor that is allowed
        default is 1000, can also be set lower (e.g. 600)
        unit is (kg /m² /mth /K)
    step_height_for_corr: float, optional
        step altitude change to correct climate gridpoint reference height
        default is 25 meters
    max_height_change_for_corr: float, optional
        maximum change of altitude that is allowed to correct the climate
        gridpoint, default is 3000 meters

    """

    # new method that uses the corrected geodetic MB:
    mb_geodetic = utils.get_geodetic_mb_dataframe().loc[gdir.rgi_id]
    ref_period = '2000-01-01_2020-01-01'
    mb_geodetic = float(mb_geodetic.loc[mb_geodetic['period'] == ref_period]['dmdtda'])
    # dmdtda: in meters water-equivalent per year -> we convert
    mb_geodetic *= 1000  # kg m-2 yr-1

    # instantiate the mass-balance model
    # this is used instead of the melt_f
    mb_mod = mb_model_sub_class(gdir, None, mb_type=mb_type,
                                grad_type=grad_type,
                                baseline_climate=climate_type, residual=residual,
                                **kwargs_for_TIModel_Sfc_Type)

    # old:
    # if necessary add a temperature bias (reference height change)
    # mb_mod.historical_climate_qc_mod(gdir),
    # instead do sth. similar but only when no melt_f found ...
    # see below

    if mb_type != 'mb_real_daily':
        temporal_resol = 'monthly'
    else:
        temporal_resol = 'daily'
    fs = '_{}_{}'.format(temporal_resol, climate_type)

    fpath = gdir.get_filepath('climate_historical', filesuffix=fs)
    with utils.ncDataset(fpath, 'a') as nc:
        start = getattr(nc, 'uncorrected_ref_hgt', nc.ref_hgt)
        nc.uncorrected_ref_hgt = start
        nc.ref_hgt = start

    h, w = gdir.get_inversion_flowline_hw()
    # do the climate calibration:
    # get here the right melt_f fitting to that precipitation factor
    # find the melt factor that minimises the bias to the geodetic observations
    try:
        melt_f_opt = scipy.optimize.brentq(minimize_bias_geodetic,
                                           min_melt_f,
                                           max_melt_f,
                                           disp=True, xtol=0.1,
                                           args=(mb_mod, mb_geodetic, h, w,
                                                 pf, False,
                                                 np.arange(2000, ye, 1),  # time period that we want to calibrate
                                                 False, spinup))
        # we don't need a reference height correction if melt_f calibration has worked!
        ref_hgt_calib_diff = 0
    except ValueError or TypeError:
        # This happens when out of bounds
        # same methods as in mu_star_calibration_from_geodetic_mb applied
        # Check in which direction we should correct the temp
        _lim0 = minimize_bias_geodetic(min_melt_f, gd_mb=mb_mod,
                                       mb_geodetic=mb_geodetic,
                                       h=h, w=w, pf=pf,
                                       ys=np.arange(2000, ye, 1))
        # _lim0 = modelled - observed(geodetic)

        if _lim0 < 0:
            # The mass-balances are too positive to be matched - we need to
            # cool down the climate data
            # here we lower the reference height of the climate gridpoint
            # hence the difference between glacier and ref_hgt of climate data
            # gets higher
            step = -step_height_for_corr
            end = -max_height_change_for_corr
        else:
            # The other way around
            step = step_height_for_corr
            end = max_height_change_for_corr

        steps = np.arange(start, start + end, step, dtype=np.int64)
        melt_f_candidates = steps * np.NaN
        for i, ref_h in enumerate(steps):
            with utils.ncDataset(fpath, 'a') as nc:
                nc.ref_hgt = ref_h
                # problem, climate needs to be read in again??
                # fabi???
                # I guess, I need to change the ref_hgt of the mb_mod
                # in order that this works for me!!!
            mb_mod.ref_hgt = ref_h
            try:
                melt_f = scipy.optimize.brentq(minimize_bias_geodetic,
                                               min_melt_f,
                                               max_melt_f,
                                               disp=True, xtol=0.1,
                                               args=(mb_mod, mb_geodetic, h,
                                                     w, pf, False,
                                                     np.arange(2000, ye, 1),
                                                     # time period that we want to calibrate
                                                     False, spinup))


            except ValueError or TypeError:
                melt_f = np.NaN
                # Done - store for later
            melt_f_candidates[i] = melt_f
            # @Fabi, if we find one working melt_f we can actually stop
            # the loop, or???
            if np.isfinite(melt_f):
                break

        # only choose the one that worked
        sel_steps = steps[np.isfinite(melt_f_candidates)]
        sel_melt_f = melt_f_candidates[np.isfinite(melt_f_candidates)]
        if len(sel_melt_f) == 0:
            # Yeah nothing we can do here
            raise MassBalanceCalibrationError('We could not find a way to '
                                              'correct the climate data and '
                                              'fit within the prescribed '
                                              'bounds for the melt_f.')
        # now we only have one candidate (the one where we changes the least the ref_hgt)
        # so we prefer to have a melt_f very near to the allowed range
        # than having a very large temp_bias (i.e ref_hgt_calib_diff)
        # @fabi: mu_star_calibration_from_geodetic_mb we take the first one
        # but compute all, but maybe it makes sense to only compute the first one
        # because this is the nearest to the ref_hgt???
        melt_f_opt = sel_melt_f[0]
        # Final correction of the data
        with utils.ncDataset(fpath, 'a') as nc:
            nc.ref_hgt = sel_steps[0]
        ref_hgt_calib_diff = sel_steps[0] - start
        gdir.add_to_diagnostics('ref_hgt_calib_diff',
                                ref_hgt_calib_diff)
        mb_mod.ref_hgt = sel_steps[0]

    mb_mod.melt_f = melt_f_opt
    mb_mod.prcp_fac = pf

    # just check if calibration worked ...
    spec_mb = mb_mod.get_specific_mb(heights=h, widths=w,
                                     year=np.arange(2000, ye, 1)).mean()
    np.testing.assert_allclose(mb_geodetic, spec_mb, rtol=1e-2)
    if mb_model_sub_class == TIModel_Sfc_Type:
        mb_mod.reset_pd_mb_bucket()
    # Fabi?: which starting year? I set it to 2000 !!! Is this right?
    # get the apparent_mb (necessary for inversion)
    climate.apparent_mb_from_any_mb(gdir, mb_model=mb_mod,
                                    mb_years=np.arange(2000, ye, 1))

    # TODO: maybe also add type of mb_model_sub_class into fs ???
    fs_new = '_{}_{}_{}'.format(climate_type, mb_type,
                                grad_type)
    d = {'melt_f_pf_{}'.format(np.round(pf, 2)): melt_f_opt,
         'ref_hgt_calib_diff':  ref_hgt_calib_diff}
    gdir.write_json(d, filename='melt_f_geod', filesuffix=fs_new)


def minimize_winter_mb_brentq_geod_via_pf(x, gd_mb=None, mb_geodetic=None, mb_glaciological_winter=None, h=None,
                                           w=None, ys_glac=None, period_from_wgms=False, tipe='bias'):
    """ calibrates the optimal precipitation factor (pf) to match best the observed winter mass balance
     to the modelled mass balance by getting the bias near to zero (-> could change this to mean absolute error?!)

    for each pf an optimal melt_f is found (by using the geodetic data via `minimize_bias_geodetic`),
    then the bias (mean absolute error) is computed, which is then brought to zero (minimised)

    Attention mb_glaciological_winter should be the MB of the years from ys_glac!!

    --> Work In Process:
    todo: documentation

    (and actually the optimisation occurs only when doing
    scipy.optimize.brentq(minimize_winter_mb_brentq_geod_via_pf, 0.1, 10, ...)
    but we don't want to change the function at this stage)

    """

    pf = x
    melt_f_opt = scipy.optimize.brentq(minimize_bias_geodetic, 10, 1000,
                                       xtol=0.01,
                                       args=(gd_mb, mb_geodetic, h, w, pf),
                                       disp=True)
    gd_mb.melt_f = melt_f_opt
    gd_mb.prcp_fac = pf

    # now compute mean absolute error over given time period using direct glaciological observations
    # mb_sum = []
    mb_winter = []
    #for y in ys_glac:
    # mb_sum.append(gd_mb.get_specific_summer_mb(y))
    # first precompute it without missing years
    gd_mb.get_specific_mb(heights=h, widths=w,
                              year=np.arange(ys_glac[0],
                                             ys_glac[-1] + 1, 1))
    mb_winter.append(gd_mb.get_specific_winter_mb(h, year=ys_glac, widths=w, period_from_wgms=period_from_wgms))

    bias = np.array(mb_winter).mean() - mb_glaciological_winter.mean()
    # mae_win = mean_absolute_error(mb_winter, mb_glaciological_winter)
    if tipe == 'bias':
        return bias
    elif tipe == 'absolute_bias':
        return np.absolute(bias)


def calib_inv_run(gdir=np.NaN, kwargs_for_TIModel_Sfc_Type={'melt_f_change': 'linear',
                                                            'tau_e_fold_yr': 0.5,
                                                            'spinup_yrs': 6,
                                                            'melt_f_update': 'annual',
                                                            'melt_f_ratio_snow_to_ice': 0.5},
                  mb_elev_feedback='annual',
                  nyears=300, seed=0,
                  spinup=True, interpolation_optim=False,
                  run_type='constant',
                  mb_model_sub_class=TIModel_Sfc_Type, pf=2,
                  mb_type='mb_monthly',
                  grad_type='cte', y0=2004, hs=10,
                  store_monthly_step=False, unique_samples=False,
                  climate_type='W5E5',
                  ensemble='mri-esm2-0_r1i1p1f1', ssp='ssp126',
                  ye=2100):

    """
    Does the calibration, the inversion and then the run using GCM data:
    First it runs, melt_f_calib_geod_prep_inversion,
    then calibrate_inversion_from_consensus,
    then init_present_time_glacier,
    finally it runs either constant, random or from climate for a given period!

    attention: you need to process first the climate calibration data (e.g. `process_w5e5_data`)
    and then process the projection gcm data (e.g. `process_isimip_data`)

    attention: here glen-a is calibrated to match a single glaciers
    volume. Hence, glen-a is different for each glacier if this is applied on several glaciers.
    In that case, it would be better to write a new function and to tune only one glen-a such
    that the Farinotti(2019) volume is matched for all projected glaciers ...

    todo: add better documentation and add a test suite for that !!!

    Parameters
    ----------
    gdir
    kwargs_for_TIModel_Sfc_Type
    mb_elev_feedback
    nyears
    seed
    spinup
    interpolation_optim
    run_type
    mb_model_sub_class
    pf
    mb_type
    grad_type
    y0
    hs
    store_monthly_step
    unique_samples
    climate_type
    ensemble
    ssp
    ye

    Returns
    -------

    """
    # end of climate calibration dataset & geodetic data
    ye_calib = 2019
    dataset = climate_type
    nosigmaadd = ''

    if mb_model_sub_class == TIModel:
        kwargs_for_TIModel_Sfc_Type = {}
    kwargs_for_TIModel_Sfc_Type_calib = kwargs_for_TIModel_Sfc_Type.copy()

    if run_type == 'constant' and mb_model_sub_class == TIModel_Sfc_Type:
        # for calibration, no interpolation will be done ...
        kwargs_for_TIModel_Sfc_Type_calib['interpolation_optim'] = False

    workflow.execute_entity_task(melt_f_calib_geod_prep_inversion, [gdir],
                                 pf=pf,  # precipitation factor
                                 mb_type=mb_type, grad_type=grad_type,
                                 climate_type=climate_type, residual=0,
                                 ye=ye_calib+1,
                                 mb_model_sub_class=mb_model_sub_class,
                                 kwargs_for_TIModel_Sfc_Type=kwargs_for_TIModel_Sfc_Type_calib,
                                 spinup=spinup)

    # here glen-a is calibrated to match each glaciers volume (glen-a is different for each glacier!!!)
    border = 80
    filter = border >= 20
    pd_inv_melt_f = workflow.calibrate_inversion_from_consensus([gdir],
                                                                apply_fs_on_mismatch=False,
                                                                error_on_mismatch=False,
                                                                filter_inversion_output=filter)
    workflow.execute_entity_task(tasks.init_present_time_glacier, [gdir])

    a_factor = gdir.get_diagnostics()['inversion_glen_a'] / cfg.PARAMS['inversion_glen_a']
    # just a check if a-factor is set to the same value
    np.testing.assert_allclose(a_factor,
                               gdir.get_diagnostics()['inversion_glen_a'] / cfg.PARAMS['inversion_glen_a'])

    # double check: volume sum of gdirs from Farinotti estimate is equal to oggm estimates
    np.testing.assert_allclose(pd_inv_melt_f.sum()['vol_itmix_m3'],
                               pd_inv_melt_f.sum()['vol_oggm_m3'], rtol=1e-2)

    ######
    if mb_model_sub_class == TIModel_Sfc_Type:
        add_msm = 'sfc_type_{}_tau_{}_{}_update_ratio{}_mb_fb_{}'.format(kwargs_for_TIModel_Sfc_Type['melt_f_change'],
                                                                         kwargs_for_TIModel_Sfc_Type['tau_e_fold_yr'],
                                                                         kwargs_for_TIModel_Sfc_Type['melt_f_update'],
                                                                         kwargs_for_TIModel_Sfc_Type['melt_f_ratio_snow_to_ice'],
                                                                         mb_elev_feedback)
    else:
        add_msm = 'TIModel_no_sfc_type_distinction_mb_fb_{}'
    j = 'test'
    add = 'pf{}'.format(pf)
    rid = '{}_{}_{}'.format('ISIMIP3b', ensemble, ssp)
    if run_type == 'constant':
        run_model = run_constant_climate_TIModel(gdir, bias=0, nyears=nyears,
                                                 y0=y0, halfsize=hs,
                                                 mb_type=mb_type,
                                                 climate_filename='gcm_data',
                                                 grad_type=grad_type,
                                                 precipitation_factor=pf,
                                                 climate_type=climate_type, # dataset for the calibration
                                                 # changed this here to also include possible temp.bias correction
                                                 # but not yet tested!!!
                                                 # melt_f=gdir.read_json(filename='melt_f_geod',
                                                 # filesuffix=fs)[f'melt_f_pf_{pf}'],
                                                 melt_f='from_json',
                                                 climate_input_filesuffix=rid,  # dataset for the run
                                                 output_filesuffix='_{}{}_ISIMIP3b_{}_{}_{}_{}{}_hist_{}_{}'.format(
                                                     nosigmaadd, add_msm,
                                                     dataset, ensemble, mb_type, grad_type,
                                                     add, ssp, j),
                                                 mb_model_sub_class=mb_model_sub_class,
                                                 kwargs_for_TIModel_Sfc_Type=kwargs_for_TIModel_Sfc_Type,
                                                 mb_elev_feedback=mb_elev_feedback,
                                                 interpolation_optim=interpolation_optim,
                                                 store_monthly_step=store_monthly_step)
    elif run_type == 'random':
        run_model = run_random_climate_TIModel(gdir, nyears=nyears, y0=y0, halfsize=hs,
                                               mb_model_sub_class=mb_model_sub_class,
                                               temperature_bias=None,
                                               mb_type=mb_type, grad_type=grad_type,
                                               bias=0, seed=seed,
                                               climate_type=climate_type,  # dataset for the calibration
                                               # changed this here to also include possible temp.bias correction
                                               # but not yet tested!!!
                                               # melt_f=gdir.read_json(filename='melt_f_geod',
                                               # filesuffix=fs)[f'melt_f_pf_{pf}'],
                                               melt_f='from_json',
                                               climate_input_filesuffix=rid,  # dataset for the run
                                               precipitation_factor=pf,
                                               climate_filename='gcm_data',
                                               output_filesuffix='_{}{}_ISIMIP3b_{}_{}_{}_{}{}_hist_{}_{}'.format(
                                                   nosigmaadd, add_msm,
                                                   dataset, ensemble, mb_type, grad_type, add, ssp, j),
                                               kwargs_for_TIModel_Sfc_Type=kwargs_for_TIModel_Sfc_Type,
                                               # {'melt_f_change':'linear'},
                                               mb_elev_feedback=mb_elev_feedback,
                                               store_monthly_step=store_monthly_step,
                                               unique_samples=unique_samples)

    elif run_type == 'from_climate':
        run_model = run_from_climate_data_TIModel(gdir, bias=0, min_ys=y0,
                                                  ye=2100, mb_type=mb_type,
                                                  climate_filename='gcm_data',
                                                  grad_type=grad_type, precipitation_factor=pf,
                                                  climate_type=climate_type,  # dataset for the calibration
                                                  # changed this here to also include possible temp.bias correction
                                                  # but not yet tested!!!
                                                  # melt_f=gdir.read_json(filename='melt_f_geod',
                                                  # filesuffix=fs)[f'melt_f_pf_{pf}'],
                                                  melt_f='from_json',
                                                  climate_input_filesuffix=rid,  # dataset for the run
                                                  output_filesuffix='_{}{}_ISIMIP3b_{}_{}_{}_{}{}_hist_{}_{}'.format(
                                                      nosigmaadd, add_msm,
                                                      dataset, ensemble, mb_type, grad_type, add, ssp, j),
                                                  mb_model_sub_class=mb_model_sub_class,
                                                  kwargs_for_TIModel_Sfc_Type=kwargs_for_TIModel_Sfc_Type,
                                                  # {'melt_f_change':'linear'},
                                                  mb_elev_feedback=mb_elev_feedback,
                                                  store_monthly_step=store_monthly_step)
    ds = utils.compile_run_output([gdir],
                                  input_filesuffix='_{}{}_ISIMIP3b_{}_{}_{}_{}{}_hist_{}_{}'.format(nosigmaadd, add_msm,
                                                                                                    dataset, ensemble,
                                                                                                    mb_type, grad_type,
                                                                                                    add, ssp, j))

    fs = '_{}_{}_{}'.format(climate_type, mb_type, grad_type)

    return ds, gdir.read_json(filename='melt_f_geod', filesuffix=fs)[f'melt_f_pf_{pf}'], run_model


def get_mean_mb_profile_filtered(gdir, input_fs=None,
                                 # y0 = 1979, y1 = 2019, (this is already done because of the climate file
                                 obs_ratio_needed=0.6):
    mb_profile_filtered = gdir.get_ref_mb_profile(input_filesuffix=input_fs, constant_dh=True,
                                                  obs_ratio_needed=obs_ratio_needed)
    mb = gdir.get_ref_mb_data(input_filesuffix=input_fs)
    if (mb_profile_filtered is None) or (np.shape(mb_profile_filtered) == (0, 0)):
        # ref glacier but has no elevation band details
        pass
    else:

        # years with direct glaciological observaitions
        ys_glac_direct_glac = mb.index.values
        # at least 10 glacier measurements of direct glaciological observations
        condi_glac_msm = len(ys_glac_direct_glac) >= 10

        ys_glac_sel = mb_profile_filtered.index.values
        # at least 5 years with MB profile observations
        condi_mb_profile_msm_yr = len(mb_profile_filtered.index) >= 5

        # at least 5 elevation band measurements (filtered; for at least 60% of the msm years)
        condi_mb_profile_h_n = len(mb_profile_filtered.columns) >= 5

        if (condi_glac_msm) & (condi_mb_profile_msm_yr) & (condi_mb_profile_h_n):
            # enough mb profiles and with at least 10 direct glaciological observation

            return mb_profile_filtered.mean(axis=0), ys_glac_sel
        else:
            pass