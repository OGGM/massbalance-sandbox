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

# imports from local MBsandbox package modules
from MBsandbox.mbmod_daily_oneflowline import TIModel

# %%
def minimize_bias(x, gd_mb=None, gdir_min=None,
                  pf=None, absolute_bias=False):
    """ calibrates the melt factor (melt_f) by getting the bias to zero
    comparing modelled mean specific mass balance to
    direct glaciological observations

    Parameters
    ----------
    x : float
        what is optimised; here the melt factor (melt_f)
    gd_mb: class instance
        instantiated class of TIModel, this is updated by melt_f
    gdir_min :
        glacier directory. The default is None but this has to be set.
    N : int, optional
        Amount of percentiles, only used for mb_type ='mb_daily'.
        The default is 1000.
    pf: float: optional
        precipitation factor. The default is 2.5.
    loop : bool, optional
        If loop is applied, only used for mb_type ='mb_daily'.
        The default is False.
    absolute_bias : bool
        if absolute_bias == True, the absolute value of the bias is returned.
        if optimisation is done with Powell need absolute bias.
        If optimisation is done with Brentq, absolute_bias has to set False
        The default is False.

    Returns
    -------
    float
        bias: modeled mass balance mean - reference mean
        if absolute_bias = True:  np.abs(bias) is returned

    """

    h, w = gdir_min.get_inversion_flowline_hw()
    mbdf = gdir_min.get_ref_mb_data()
    gd_mb.melt_f = x
    if type(pf)==float or type(pf)==int:
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
# %%


def minimize_bias_geodetic(x, gd_mb=None, mb_geodetic=None,
                           h=None, w=None, pf=2.5,
                           absolute_bias=False, ys=np.arange(2000, 2019, 1)):
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
        If optimisation is done with Brentq, absolute_bias has to set False
        The default is False.
    ys: np.array
        years for which specific mass balance is computed
        default is 2000--2018
        TODO: change this to to 2000-2019 to match better
              geodetic msm

    Returns
    -------
    float
        bias: modeled mass balance mean - reference mean (geodetic)
        if absolute_bias = True:  np.abs(bias) is returned

    """
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




def compute_stat(mb_specific=None, mbdf=None, return_dict=False,
                 return_plot=False, round = False):
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
                             gdir_min=None):
    """ calibrates the optimal precipitation factor (pf) by correcting the
    standard deviation of the modelled mass balance

    for each pf an optimal melt_f is found, then (1 - standard deviation quotient
    between modelled and reference mass balance) is computed,
    which is then minimised

    TODO: only change melt_f and pf, and not instantiate the model always again

    Parameters
    ----------
    x : float
        what is optimised (here the precipitation factor)
    gd_mb: class instance
        instantiated class of TIModel, this is updated by pf and melt_f
    gdir_min : optional
        glacier directory. The default is None but this has to be set.


    Returns
    -------
    float
        1- quot_std

    """
    h, w = gdir_min.get_inversion_flowline_hw()
    mbdf = gdir_min.get_ref_mb_data()
    pf = x
    melt_f_opt = scipy.optimize.brentq(minimize_bias, 1, 10000,
                                               disp=True, xtol=0.1,
                                                args=(gd_mb, gdir_min,
                                                pf, False))
    gd_mb.melt_f = melt_f_opt
    # check climate and adapt if necessary
    gd_mb.historical_climate_qc_mod(gdir_min)

    mod_std = gd_mb.get_specific_mb(heights=h, widths=w,
                                    year=mbdf.index.values).std()
    ref_std = mbdf['ANNUAL_BALANCE'].values.std()
    quot_std = mod_std/ref_std

    return 1-quot_std
