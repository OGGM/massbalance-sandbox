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
from oggm.core import climate
import pandas as pd
import logging
from oggm import utils, workflow, tasks, entity_task

log = logging.getLogger(__name__)

# imports from local MBsandbox package modules
from MBsandbox.mbmod_daily_oneflowline import TIModel, TIModel_Sfc_Type
# from MBsandbox.help_func import compute_stat, minimize_bias, optimize_std_quot_brentq
from MBsandbox.flowline_TIModel import (run_from_climate_data_TIModel, run_constant_climate_TIModel,
                                        run_random_climate_TIModel)


# %%
def minimize_bias(x, gd_mb=None, gdir_min=None,
                  pf=None, absolute_bias=False, input_filesuffix=''):
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
        Amount of percentiles, only used for mb_type ='mb_pseudo_daily'.
        The default is 100.
    pf: float: optional
        precipitation factor. The default is 2.5.
    loop : bool, optional
        If loop is applied, only used for mb_type ='mb_pseudo_daily'.
        The default is False.
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
# %%
def minimize_bias_geodetic(x, gd_mb=None, mb_geodetic=None,
                           h=None, w=None, pf=2.5,
                           absolute_bias=False,
                           ys=np.arange(2000, 2019, 1),
                           oggm_default_mb=False,
                           spinup=True):
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


def optimize_std_quot_brentq_geod(x, gd_mb=None, mb_geodetic=None,
                                  mb_glaciological=None,
                                  h=None, w=None,
                                  ys_glac=np.arange(1979, 2019, 1),
                                  ):
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
                             gdir_min=None,
                             input_filesuffix=''):
    """ calibrates the optimal precipitation factor (pf) by correcting the
    standard deviation of the modelled mass balance

    for each pf an optimal melt_f is found, then (1 - standard deviation quotient
    between modelled and reference mass balance) is computed,
    which is then minimised

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


###
from oggm import cfg
_doc = 'the calibrated melt_f according to the geodetic data with the ' \
       'chosen precipitation factor'
cfg.BASENAMES['melt_f_geod'] = ('melt_f_geod.json', _doc)

# TODO:@entity_task(log) -> allows multi-processing
#  but at the moment still gives an error
#  AttributeError: Can't get attribute 'melt_f_calib_geod_prep_inversion' on <module '__main__'>
#  Process ForkPoolWorker-3:
@entity_task(log)
def melt_f_calib_geod_prep_inversion(gdir, mb_type='mb_monthly', grad_type='cte',
                                     pf=None, climate_type='W5E5',
                                     residual=0, path_geodetic=None, ye=None,
                                     mb_model_sub_class=TIModel,
                                     kwargs_for_TIModel_Sfc_Type={},
                                     spinup=True):
    """ calibrates the melt factor using the TIModel mass-balance model,
    computes the apparent mass balance for the inversion
    and saves the melt_f and the applied pf into a json inside of the glacier directory

    TODO: make documentation
    """
    # get the geodetic data
    pd_geodetic = pd.read_csv(path_geodetic, index_col='rgiid')
    pd_geodetic = pd_geodetic.loc[pd_geodetic.period == '2000-01-01_2020-01-01']
    # for that glacier
    mb_geodetic = pd_geodetic.loc[gdir.rgi_id].dmdtda * 1000  # want it to be in kg/m2

    # instantiate the mass-balance model
    # this is used instead of the melt_f
    mb_mod = mb_model_sub_class(gdir, None, mb_type=mb_type,
                                grad_type=grad_type,
                     baseline_climate=climate_type, residual=residual,
                                **kwargs_for_TIModel_Sfc_Type)

    # if necessary add a temperature bias (reference height change)
    mb_mod.historical_climate_qc_mod(gdir)

    # do the climate calibration:

    # and get here the right melt_f fitting to that precipitation factor
    h, w = gdir.get_inversion_flowline_hw()
    # find the melt factor that minimises the bias to the geodetic observations
    melt_f_opt = scipy.optimize.brentq(minimize_bias_geodetic, 10, 1000,
                                       disp=True, xtol=0.1,
                                       args=(mb_mod, mb_geodetic, h, w,
                                             pf, False,
                                             np.arange(2000, ye, 1),  # time period that we want to calibrate
                                             False, spinup))
    mb_mod.melt_f = melt_f_opt
    mb_mod.prcp_fac = pf

    # just check if calibration worked ...
    spec_mb = mb_mod.get_specific_mb(heights=h, widths=w,
                                     year=np.arange(2000, ye, 1)).mean()
    np.testing.assert_allclose(mb_geodetic, spec_mb, rtol=1e-2)
    if mb_model_sub_class == TIModel_Sfc_Type:
        mb_mod.reset_pd_mb_bucket()
    # Todo: which starting year if set to 1979 I get problems !!!
    # get the apparent_mb (necessary for inversion)
    climate.apparent_mb_from_any_mb(gdir, mb_model=mb_mod,
                                    mb_years=np.arange(2000, ye, 1))
    # TODO: maybe also add type of mb_model_sub_class into fs ???
    fs = '_{}_{}_{}'.format(climate_type, mb_type, grad_type)
    d = {'melt_f_pf_{}'.format(np.round(pf, 2)): melt_f_opt}
    gdir.write_json(d, filename='melt_f_geod', filesuffix=fs)


def calib_inv_run(gdir=np.NaN, kwargs_for_TIModel_Sfc_Type={'melt_f_change': 'linear',
                                               'tau_e_fold_yr': 0.5,
                                               'spinup_yrs': 6,
                                               'melt_f_update': 'annual',
                                               'melt_f_ratio_snow_to_ice': 0.5},
                  mb_elev_feedback='annual',
                  nyears=300, seed=0,
                  spinup=True, interpolation_optim=False, run_type='constant',
                  mb_model_sub_class=TIModel_Sfc_Type, pf=2, mb_type='mb_monthly',
                  grad_type='cte', y0=2004, hs=10,
                  store_monthly_step=False, unique_samples=False, climate_type='W5E5',
                  ensemble = 'mri-esm2-0_r1i1p1f1', ssp = 'ssp126' ):
    # ye_h = 2014
    # y0 = 1979
    dataset = climate_type
    #gdirs = [gdir]
    url = 'https://cluster.klima.uni-bremen.de/~oggm/geodetic_ref_mb/hugonnet_2021_ds_rgi60_pergla_rates_10_20_worldwide.csv'
    path_geodetic = utils.file_downloader(url)

    ye = 2100
    ye_calib = 2020
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
                                 path_geodetic=path_geodetic, ye=ye_calib,
                                 mb_model_sub_class=mb_model_sub_class,
                                 kwargs_for_TIModel_Sfc_Type=kwargs_for_TIModel_Sfc_Type_calib,
                                 spinup=spinup)

    # here glen-a is calibrated to mathch gdirs glaciers in total
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
                                                                         kwargs_for_TIModel_Sfc_Type[
                                                                             'melt_f_ratio_snow_to_ice'],
                                                                         mb_elev_feedback)
    else:
        add_msm = 'TIModel_no_sfc_type_distinction_mb_fb_{}'
    j = 'test'
    add = 'pf{}'.format(pf)
    fs = '_{}_{}_{}'.format(climate_type, mb_type, grad_type)
    gdir.read_json(filename='melt_f_geod', filesuffix=fs)
    rid = '{}_{}_{}'.format('ISIMIP3b', ensemble, ssp)
    if run_type == 'constant':
        run_model = run_constant_climate_TIModel(gdir, bias=0, nyears=nyears,
                                                 y0=y0, halfsize=hs,
                                                 mb_type=mb_type,
                                                 climate_filename='gcm_data',
                                                 grad_type=grad_type, precipitation_factor=pf,
                                                 melt_f=gdir.read_json(filename='melt_f_geod', filesuffix=fs)[
                                                     f'melt_f_pf_{pf}'],
                                                 climate_input_filesuffix=rid,  # dataset,
                                                 output_filesuffix='_{}{}_ISIMIP3b_{}_{}_{}_{}{}_hist_{}_{}'.format(
                                                     nosigmaadd, add_msm,
                                                     dataset, ensemble, mb_type, grad_type,
                                                     add, ssp, j),
                                                 mb_model_sub_class=mb_model_sub_class,
                                                 kwargs_for_TIModel_Sfc_Type=kwargs_for_TIModel_Sfc_Type,
                                                 # {'melt_f_change':'linear'},
                                                 mb_elev_feedback=mb_elev_feedback,
                                                 interpolation_optim=interpolation_optim,
                                                 store_monthly_step=store_monthly_step)
    elif run_type == 'random':
        run_model = run_random_climate_TIModel(gdir, nyears=nyears, y0=y0, halfsize=hs,
                                               mb_model_sub_class=mb_model_sub_class,
                                               temperature_bias=None,
                                               mb_type=mb_type, grad_type=grad_type,
                                               bias=0, seed=seed,
                                               melt_f=gdir.read_json(filename='melt_f_geod', filesuffix=fs)[
                                                   f'melt_f_pf_{pf}'],
                                               precipitation_factor=pf,
                                               climate_filename='gcm_data',
                                               climate_input_filesuffix=rid,
                                               output_filesuffix='_{}{}_ISIMIP3b_{}_{}_{}_{}{}_hist_{}_{}'.format(
                                                   nosigmaadd, add_msm,
                                                   dataset, ensemble, mb_type, grad_type, add, ssp, j),
                                               kwargs_for_TIModel_Sfc_Type=kwargs_for_TIModel_Sfc_Type,
                                               # {'melt_f_change':'linear'},
                                               mb_elev_feedback=mb_elev_feedback, store_monthly_step=store_monthly_step,
                                               unique_samples=unique_samples)

    elif run_type == 'from_climate':
        run_model = run_from_climate_data_TIModel(gdir, bias=0, min_ys=y0,
                                                  ye=2100, mb_type=mb_type,
                                                  climate_filename='gcm_data',
                                                  grad_type=grad_type, precipitation_factor=pf,
                                                  melt_f=gdir.read_json(filename='melt_f_geod', filesuffix=fs)[
                                                      f'melt_f_pf_{pf}'],
                                                  climate_input_filesuffix=rid,  # dataset,
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

    # init_model_filesuffix='{}{}_ISIMIP3b_{}_{}_{}_{}{}_historical_{}'.format(nosigmaadd,add_msm,

    return ds, gdir.read_json(filename='melt_f_geod', filesuffix=fs)[f'melt_f_pf_{pf}'], run_model

