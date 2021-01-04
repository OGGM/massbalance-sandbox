#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 24 12:28:37 2020

@author: lilianschuster

some helper functions to minimize the bias, optimise std_quot and tos
compute performance statistics
"""
from oggm import utils, workflow, tasks, cfg
import scipy
import matplotlib.pyplot as plt

import numpy as np
from mbmod_daily_oneflowline import *
#base_url = 'https://cluster.klima.uni-bremen.de/~fmaussion/gdirs/prepro_l2_202010/elevbands_fl'


# df = utils.get_rgi_glacier_entities(['RGI60-11.00897'])
# gdirs = workflow.init_glacier_directories(df, from_prepro_level=2,
#                                          prepro_border=40, 
#                                  prepro_base_url=base_url,
#                                  prepro_rgi_version='62')
# gd = gdirs[0]
# Why does this happen, if I uncomment this, always this gd is used instead
# of the one already defined ...
# cfg.PARAMS['baseline_climate'] = 'ERA5dr'
# oggm.shop.ecmwf.process_ecmwf_data(gd, dataset = 'ERA5dr')


# %%
def minimize_bias(x, mb_type = 'mb_daily', grad ='cte', gdir_min= None, N = 1000,
                  pf = 2.5, loop = False, absolute_bias = False):
    
    """ calibrates the degree day factor or mu_star by getting the bias to zero
    
    

    Parameters
    ----------
    x : float
        what is optimised (here the DDF/mu_star)
    mb_type : TYPE, optional
        DESCRIPTION. The default is 'mb_daily'.
    grad : TYPE, optional
        DESCRIPTION. The default is 'cte'.
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
    mu_star = x 
    mbmod_s = mb_modules(gdir_min, mu_star, mb_type=mb_type, grad_type=grad,
                         N = N, prcp_fac = pf, loop = loop)
    mb_specific = mbmod_s.get_specific_mb(heights = h,
                                          widths = w,
                                          year = mbdf.index.values)
    if absolute_bias:
        bias_calib = np.abs(np.mean(mb_specific-
                                    mbdf['ANNUAL_BALANCE'].values))
    else:
        bias_calib = np.mean(mb_specific-mbdf['ANNUAL_BALANCE'].values)
        
    return bias_calib
# %%

def compute_stat(mb_specific=None, mbdf=None, return_dict = False,
                 return_plot = False):
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
    RMSD = np.sqrt(np.sum(np.square(mb_specific-mbdf['ANNUAL_BALANCE'])))/len(mbdf)
    ref_std = mbdf.ANNUAL_BALANCE.std()
    mod_std = mb_specific.std()
    bias = mb_specific.mean() - mbdf.ANNUAL_BALANCE.mean()
    # this is treated a bit different than in mb_crossval of Matthias Dusch
    if ref_std == 0:
        # in mb_crossval: ref_std is then set equal to std of the modeled mb
        quot_std = np.NaN 
        # in mb_crossval: rcor is set to 1 but I guess it should not be counted
        # because it is not sth. we want to count 
        rcor = np.NaN 
    else:
        quot_std = mod_std/ref_std
        rcor = np.corrcoef(mb_specific, mbdf.ANNUAL_BALANCE)[0, 1]

    # could also be returned as dictionary instead to not confuse the results
    if return_plot!=False:
        label = return_plot + 'RMSD {}, rcor {}, std_quot {}, bias {}'.format(RMSD.round(1), rcor.round(3), quot_std.round(3), bias.round(2))
        plt.plot(mbdf.index, mb_specific,
         label = label)
    
    if return_dict:
        return  {'RMSD' : RMSD, 'bias' : bias, 
                 'rcor' : rcor, 'quot_std' : quot_std}
    else:
        return  [RMSD,bias,rcor,quot_std]

# %%

def optimize_std_quot_brentq(x, mb_type = 'mb_daily', grad ='cte',
                             gdir_min= None,
                      N= 1000,loop = False):
    """ calibrates the optimal precipitation factor (pf) by correcting the
    standard deviation of the modelled mass balance
    
    for each pf an optimal DDF is found, then (1 - standard deviation quotient 
    between modelled and reference mass balance) is computed,
    which is then minimised
    

    Parameters
    ----------
    x : float
        what is optimised (here the precipitation factor)
    mb_type : TYPE, optional
        DESCRIPTION. The default is 'mb_daily'.
    grad : TYPE, optional
        DESCRIPTION. The default is 'cte'.
    gdir_min : optional
        glacier directory. The default is None but this has to be set. 
    N : int, optional
        Amount of percentiles, only used for mb_type ='mb_daily'.
        The default is 1000.
    loop : bool, optional
        If loop is applied, only used for mb_type ='mb_daily'. 
        The default is False.

    Returns
    -------
    float
        1- quot_std

    """
    h, w = gdir_min.get_inversion_flowline_hw()
    mbdf = gdir_min.get_ref_mb_data()
    pf = x
    DDF_opt = scipy.optimize.brentq(minimize_bias, 1, 10000, 
                                    xtol = 0.01,
                                    args=(mb_type, grad, gdir_min, N, pf, loop,
                                          False),
                                    disp=True) 
    mbmod_s = mb_modules(gdir_min, DDF_opt, mb_type = mb_type, prcp_fac = pf, 
                                       grad_type=grad, N = N)
    
    mod_std = mbmod_s.get_specific_mb(heights = h, widths = w,
                                      year = mbdf.index.values).std()
    ref_std = mbdf['ANNUAL_BALANCE'].values.std()
    quot_std = mod_std/ref_std

    return 1-quot_std

