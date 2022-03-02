import pymc3 as pm
# 	conda install -c conda-forge python-graphviza
import numpy as np
import pandas as pd
import xarray as xr
import seaborn as sns
import pickle
import ast
import warnings

import matplotlib.pyplot as plt
import matplotlib

# %matplotlib inline
import statsmodels as stats
import scipy
import scipy.stats as stats
from IPython.core.pylabtools import figsize
import os
import oggm
from oggm import cfg, utils, workflow, tasks, graphics
from oggm.core import massbalance, flowline

# import aesara.tensor as aet
# import aesara

# from drounce_analyze_mcmc import effective_n, mcse_batchmeans
# plotting bayesian stuff
import arviz as az

az.rcParams['stats.hdi_prob'] = 0.95

# import the MSsandbox modules
from MBsandbox.mbmod_daily_oneflowline import process_era5_daily_data, TIModel, \
    BASENAMES
from MBsandbox.help_func import compute_stat, minimize_bias, \
    optimize_std_quot_brentq

from MBsandbox.wip.help_func_geodetic import minimize_bias_geodetic, \
    optimize_std_quot_brentq_geod, get_opt_pf_melt_f

import theano
import theano.tensor as aet


# general parameters
# ys = np.arange(2000, 2019)  # data only goes till 2018 ref_df.index.values[ref_df.index.values>1999]


def get_TIModel_clim_model_type(gd, mb_type='mb_monthly', grad_type='cte',
                                pd_geodetic_loc=None,  # pd_geodetic_comp
                                ):
    '''

    changed it that it runs now with area in m2 input !!!
    Parameters
    ----------
    gd
    mb_type
    grad_type
    pd_geodetic_loc

    Returns
    -------

    '''

    rho_geodetic = 850

    # get volume estimates
    dV = pd.read_hdf(utils.get_demo_file('rgi62_itmix_df.h5'))

    # for potential restricting total ice melt
    V_gd_m3 = dV.loc[gd.rgi_id]['vol_itmix_m3']  # m3 volume of HEF
    total_mass_gd = V_gd_m3 * rho_geodetic
    # this is the area from 2000, could use another estimate (e.g. mean between 2000 and 2020...)
    gd_area = pd_geodetic_loc.loc[gd.rgi_id]['area']  # in m2 now !!!
    # convert kg --> kg/km2
    max_allowed_specificMB = - total_mass_gd / gd_area

    h, w = gd.get_inversion_flowline_hw()
    if mb_type != 'mb_real_daily':
        cfg.PARAMS['baseline_climate'] = 'ERA5dr'
        oggm.shop.ecmwf.process_ecmwf_data(gd, dataset='ERA5dr')
    else:
        cfg.PARAMS['baseline_climate'] = 'ERA5_daily'
        process_era5_daily_data(gd)

    ref_df = gd.get_ref_mb_data()

    # at instantiation use prcp_fac = 2.5, change this in def_get_mb later on
    gd_mb = TIModel(gd, 150, mb_type=mb_type, N=100, prcp_fac=2.5,
                    grad_type=grad_type)
    #gd_mb.historical_climate_qc_mod(gd)

    return gd_mb, ref_df, h, w, max_allowed_specificMB


def get_slope_pf_melt_f(gd_mb, h=None, w=None, ys=None):
    ### this has to go in tests later on!
    # check if the mass balance is equal for both methods!

    pfs = np.array([0.5, 1, 2, 3, 4, 5, 20])
    melt_fs = np.array([5, 50, 100, 150, 200, 300, 1000])
    pd_mb = pd.DataFrame(np.NaN, index=pfs,
                         columns=melt_fs)
    for pf in pfs:
        for melt_f in melt_fs:
            gd_mb.melt_f = melt_f
            gd_mb.prcp_fac = pf
            mb = gd_mb.get_specific_mb(heights=h, widths=w,
                                       year=ys)
            pd_mb.loc[pf, melt_f] = np.array(mb).mean()

    slope_pf = (pd_mb.loc[pfs[-1]] - pd_mb.loc[pfs[0]]) / (pfs[-1] - pfs[0])
    slope_melt_f = (pd_mb.loc[:, melt_fs[-1]] - pd_mb.loc[:, melt_fs[0]]) / (
            melt_fs[-1] - melt_fs[0])

    try:
        # print(ys, pf.std()/slope_pf.mean())
        assert slope_pf.std() / slope_pf.mean() < 1e-5
        assert slope_melt_f.std() / slope_melt_f.mean() < 1e-5
    except:
        warnings.warn(
            '{}: there might be some nonlinearities occuring in the mass balance. When using get_slope_pf_melt_f, '
            'the slopes are more different than rtol=1e-5, we will try again with rtol=1e-3')
        assert slope_pf.std() / slope_pf.mean() < 1e-3
        assert slope_melt_f.std() / slope_melt_f.mean() < 1e-3
    # recompute it the simple way without directly using oggm get_specific_mb ...
    pd_mb_simple = pd.DataFrame(np.NaN, index=pfs,
                                columns=melt_fs)
    for pf in pfs:
        for melt_f in melt_fs:
            pd_mb_simple.loc[
                pf, melt_f] = slope_pf.mean() * pf + slope_melt_f.mean() * melt_f
    np.testing.assert_allclose(pd_mb_simple, pd_mb, rtol=0.005)

    return slope_pf, slope_melt_f
    # could also add testing with always initiating again ... just to be sure


# df = ['RGI60-11.00897']
# gdirs = workflow.init_glacier_directories(df, from_prepro_level=2,
#                                          prepro_border=10,
#                                          prepro_base_url=base_url,
#                                          prepro_rgi_version='62')
# gd = gdirs[0]
# mb_type = 'mb_monthly'
# grad_type = 'cte'
# uniform = False
# gd_mb, ref_df, h, w, max_allowed_specificMB = get_TIModel_clim_model_type(gd, mb_type = mb_type, grad_type = grad_type, pd_geodetic_loc = pd_geodetic_comp)
# ys = None

def bayes_dummy_model_better(uniform,
                             max_allowed_specificMB=None,
                             gd=None, sampler='nuts',
                             ys=np.arange(2000, 2019, 1),
                             gd_mb=None, h=None, w=None, use_two_msm=True,
                             nosigma=False, model=None, pd_calib_opt=None,
                             first_ppc=True, predict_historic=True,
                             first_ppc_200=False, random_seed=42,
                             cores=4,
                             pd_geodetic_comp=None, y0=None, y1=None):
    if use_two_msm:
        slope_pfs = []
        slope_melt_fs = []
        for y in ys:
            slope_pf, slope_melt_f = get_slope_pf_melt_f(gd_mb, h=h, w=w, ys=y)
            slope_pfs.append(slope_pf.mean())
            slope_melt_fs.append(slope_melt_f.mean())
    else:
        slope_pf, slope_melt_f = get_slope_pf_melt_f(gd_mb, h=h, w=w, ys=ys)
    if model == None:
        model_T = pm.Model()
    else:
        model_T = model
    with model_T:
        if uniform == True:
            melt_f = pm.Uniform("melt_f", lower=10, upper=1000)
            pf = pm.Uniform('pf', lower=0.1, upper=10)
        else:
            if model == None:
                pf = pm.TruncatedNormal('pf', mu=pd_calib_opt['pf_opt'][
                    pd_calib_opt.reg == 11.0].dropna().mean(),
                                        sigma=pd_calib_opt['pf_opt'][
                                            pd_calib_opt.reg == 11.0].dropna().std(),
                                        lower=0.5, upper=10)
                melt_f = pm.TruncatedNormal('melt_f',
                                            mu=pd_calib_opt['melt_f_opt_pf'][
                                                pd_calib_opt.reg == 11.0].dropna().mean(),
                                            sigma=pd_calib_opt['melt_f_opt_pf'][
                                                pd_calib_opt.reg == 11.0].dropna().std(),
                                            lower=1, upper=1000)
            else:
                pass  # melt_f = melt_f

            # slopes have to be defined as theano constants

        # aet_slope_pf_0 = aet.constant(np.array(slope_pfs)[ys<=2010].mean())
        # aet_slope_pf_1 = aet.constant(np.array(slope_pfs)[ys>2010].mean())
        # aet_slope_melt_f_0 = aet.constant(np.array(slope_melt_fs)[ys<=2010].mean())
        # aet_slope_melt_f_1 = aet.constant(np.array(slope_melt_fs)[ys>2010].mean())
        if use_two_msm:
            aet_slope_melt_fs = pm.Data('aet_slope_melt_fs', [
                np.array(slope_melt_fs)[ys <= 2010].mean(),
                np.array(slope_melt_fs)[ys > 2010].mean()])
            aet_slope_pfs = pm.Data('aet_slope_pfs', (
                [np.array(slope_pfs)[ys <= 2010].mean(),
                 np.array(slope_pfs)[ys > 2010].mean()]))
        else:
            aet_slope_melt_fs = pm.Data('aet_slope_melt_fs',
                                        [np.array(slope_melt_f).mean()])
            aet_slope_pfs = pm.Data('aet_slope_pfs',
                                    [np.array(slope_pf).mean()])
        # aet_mbs = [aet_slope_pf_0, aet_slope_pf_1] *pf + aet_slope_melt_fs*melt_f

        # aet_mb_0 = aet_slope_pf_0 *pf + aet_slope_melt_f_0*melt_f
        # aet_mb_1 = aet_slope_pf_1 *pf + aet_slope_melt_f_1*melt_f
        if model == None:
            aet_mbs = aet_slope_pfs * pf + aet_slope_melt_fs * melt_f
        else:
            aet_mbs = aet_slope_pfs * model.pf + aet_slope_melt_fs * model.melt_f
        # aet_mbs = aet.as_tensor_variable([aet_mb_0, aet_mb_1])
        # aet_slope_melt_fs = aet.vector(np.array([np.array(slope_melt_fs)[ys<=2010].mean(), np.array(slope_melt_fs)[ys>2010].mean()]))

        # this is not the new simple theano compatible
        # mass balance function that depends on pf and melt_f
        # aet_mbs = [aet_slope_pf_0, aet_slope_pf_1] *pf + aet_slope_melt_fs*melt_f

        # make a deterministic out of it to save it also in the traces
        mb_mod = pm.Deterministic('mb_mod', aet_mbs)
    with model_T:
        if use_two_msm:
            sigma = pm.Data('sigma', pd_geodetic_comp.loc[gd.rgi_id][
                ['err_dmdtda_2000_2010', 'err_dmdtda_2010_2020']].values * 1000)
            observed = pm.Data('observed', pd_geodetic_comp.loc[gd.rgi_id][
                ['dmdtda_2000_2010', 'dmdtda_2010_2020']].values * 1000)
            if nosigma == False:
                geodetic_massbal = pm.Normal('geodetic_massbal',
                                             mu=mb_mod,
                                             sigma=sigma,  # standard devia
                                             observed=observed)  # likelihood
            else:
                geodetic_massbal = pm.Normal('geodetic_massbal',
                                             mu=mb_mod,
                                             observed=observed)  # likelihood

            diff_geodetic_massbal = pm.Deterministic("diff_geodetic_massbal",
                                                     geodetic_massbal - observed)
        else:
            # sigma and observed need to have dim 1 (not zero), --> [value]
            observed = pm.Data('observed', [
                pd_geodetic_comp.loc[gd.rgi_id]['dmdtda'] * 1000])
            if nosigma == False:
                sigma = pm.Data('sigma', [
                    pd_geodetic_comp.loc[gd.rgi_id]['err_dmdtda'] * 1000])
                geodetic_massbal = pm.TruncatedNormal('geodetic_massbal',
                                                      mu=mb_mod,
                                                      sigma=sigma,  # standard devia
                                                      observed=observed,
                                                      lower=max_allowed_specificMB)  # likelihood
            else:
                geodetic_massbal = pm.TruncatedNormal('geodetic_massbal',
                                                      mu=mb_mod,
                                                      observed=observed,
                                                      lower=max_allowed_specificMB)  # likelihood
            diff_geodetic_massbal = pm.Deterministic("diff_geodetic_massbal",
                                                     geodetic_massbal - observed)
        # constrained already by using TruncatedNormal geodetic massbalance ...
        # pot_max_melt = pm.Potential('pot_max_melt', aet.switch(
        #    geodetic_massbal < max_allowed_specificMB, -np.inf, 0))

        # also compute this difference just to be sure ...
        prior = pm.sample_prior_predictive(random_seed=random_seed, #cores=cores,
                                           samples=1000)  # , keep_size = True)
    with model_T:
        if sampler == 'nuts':
            trace = pm.sample(20000, chains=3, tune=20000, target_accept=0.999,
                              compute_convergence_checks=True, cores=cores,
                              return_inferencedata=True)
            # increased target_accept because of divergences ...
        #                 #start={'pf':2.5, 'melt_f': 200})
        elif sampler == 'jax':
            import pymc3.sampling_jax
            trace = pm.sampling_jax.sample_numpyro_nuts(20000, chains=3,
                                                        tune=20000,
                                                        target_accept=0.98)  # , compute_convergence_checks= True)

    with model_T:
        burned_trace = trace.sel(draw=slice(5000, None))
        burned_trace.posterior['draw'] = np.arange(0, len(burned_trace.posterior.draw))
        burned_trace.log_likelihood['draw'] = np.arange(0, len(burned_trace.posterior.draw))
        burned_trace.sample_stats['draw']  = np.arange(0, len(burned_trace.posterior.draw))
        
        
        # trace = pm.sample(10000, chains=4, tune=10000, target_accept = 0.98)
        # need high target_accept to have no divergences, effective sample number
        #  and # We have stored the paths of all our variables, or "traces", in the trace variable.,
        # these paths are the routes the unknown parameters (here just 'n') have taken thus far.
        # Inference using the first few thousand points is a bad idea, as they are unrelated to the
        # final distribution we are interested in.
        # Thus is it a good idea to discard those samples before using the samples for inference.
        # We call this period before converge the burn-in period.
        # burned_trace = trace[1000:]
        # if arviz dataset
        if first_ppc:
            # TODO: then sometimes a problem occurs that a warning is raised
            #  about more chains (1000) than draws (2) ... why ???
            ppc = pm.sample_posterior_predictive(burned_trace, #cores=cores,
                                                 random_seed=random_seed,
                                                 var_names=['geodetic_massbal',
                                                            'pf', 'melt_f',
                                                            'mb_mod',
                                                            'diff_geodetic_massbal'],
                                                 keep_size=True)
            az.concat(burned_trace,
                      az.from_dict(posterior_predictive=ppc, prior=prior),
                      inplace=True)
        if first_ppc_200:
            ppc = pm.sample_posterior_predictive(burned_trace, #=cores,
                                                 samples=200,
                                                 random_seed=random_seed,
                                                 var_names=['geodetic_massbal',
                                                            'pf', 'melt_f',
                                                            'mb_mod',
                                                            'diff_geodetic_massbal'])

            az.concat(burned_trace,
                      az.from_dict(posterior_predictive=ppc, prior=prior),
                      inplace=True)

    #
    if predict_historic:
        try:
            ys_ref = gd.get_ref_mb_data(y0=y0, y1=y1).index.values
        except:
            ys_ref = np.arange(1979, 2019, 1)
        with model_T:
            slope_pf_new = []
            slope_melt_f_new = []
            for y in ys_ref:
                slope_pf, slope_melt_f = get_slope_pf_melt_f(gd_mb, h=h, w=w,
                                                             ys=y)
                slope_pf_new.append(slope_pf.mean())
                slope_melt_f_new.append(slope_melt_f.mean())
            pm.set_data(new_data={'aet_slope_melt_fs': slope_melt_f_new,
                                  'aet_slope_pfs': slope_pf_new,
                                  'observed': np.empty(len(ys_ref)),
                                  'sigma': np.empty(len(ys_ref))})
            ppc_new = pm.sample_posterior_predictive(burned_trace, #cores=cores,
                                                     random_seed=random_seed,
                                                     var_names=[
                                                         'geodetic_massbal',
                                                         'pf', 'melt_f',
                                                         'mb_mod',
                                                         'diff_geodetic_massbal'],
                                                     keep_size=True)
        predict_data = az.from_dict(posterior_predictive=ppc_new)
    else:
        predict_data = None
    return burned_trace, model_T, predict_data


def bayes_dummy_model_better_OLD(uniform,
                                 max_allowed_specificMB=None,
                                 gd=None, sampler='nuts',
                                 ys=np.arange(2000, 2019, 1),
                                 gd_mb=None, h=None, w=None, use_two_msm=True,
                                 nosigma=False, model=None, pd_calib_opt=None,
                                 first_ppc=True, pd_geodetic_comp=None,
                                 random_seed=42, y0=None, y1=None):
    if use_two_msm:
        slope_pfs = []
        slope_melt_fs = []
        for y in ys:
            slope_pf, slope_melt_f = get_slope_pf_melt_f(gd_mb, h=h, w=w, ys=y)
            slope_pfs.append(slope_pf.mean())
            slope_melt_fs.append(slope_melt_f.mean())
    else:
        slope_pf, slope_melt_f = get_slope_pf_melt_f(gd_mb, h=h, w=w, ys=ys)
    if model == None:
        model_T = pm.Model()
    else:
        model_T = model
    with model_T:
        if uniform == True:
            melt_f = pm.Uniform("melt_f", lower=10, upper=1000)
            pf = pm.Uniform('pf', lower=0.1, upper=10)
        else:
            if model == None:
                pf = pm.TruncatedNormal('pf', mu=pd_calib_opt['pf_opt'][
                    pd_calib_opt.reg == 11.0].dropna().mean(),
                                        sigma=pd_calib_opt['pf_opt'][
                                            pd_calib_opt.reg == 11.0].dropna().std(),
                                        lower=0.1, upper=10)
                melt_f = pm.TruncatedNormal('melt_f',
                                            mu=pd_calib_opt['melt_f_opt_pf'][
                                                pd_calib_opt.reg == 11.0].dropna().mean(),
                                            sigma=pd_calib_opt['melt_f_opt_pf'][
                                                pd_calib_opt.reg == 11.0].dropna().std(),
                                            lower=10, upper=1000)
            else:
                pass  # melt_f = melt_f

            # slopes have to be defined as theano constants

        # aet_slope_pf_0 = aet.constant(np.array(slope_pfs)[ys<=2010].mean())
        # aet_slope_pf_1 = aet.constant(np.array(slope_pfs)[ys>2010].mean())
        # aet_slope_melt_f_0 = aet.constant(np.array(slope_melt_fs)[ys<=2010].mean())
        # aet_slope_melt_f_1 = aet.constant(np.array(slope_melt_fs)[ys>2010].mean())
        if use_two_msm:
            aet_slope_melt_fs = pm.Data('aet_slope_melt_fs', [
                np.array(slope_melt_fs)[ys <= 2010].mean(),
                np.array(slope_melt_fs)[ys > 2010].mean()])
            aet_slope_pfs = pm.Data('aet_slope_pfs', (
                [np.array(slope_pfs)[ys <= 2010].mean(),
                 np.array(slope_pfs)[ys > 2010].mean()]))
        else:
            aet_slope_melt_fs = pm.Data('aet_slope_melt_fs',
                                        [np.array(slope_melt_f).mean()])
            aet_slope_pfs = pm.Data('aet_slope_pfs',
                                    [np.array(slope_pf).mean()])
        # aet_mbs = [aet_slope_pf_0, aet_slope_pf_1] *pf + aet_slope_melt_fs*melt_f

        # aet_mb_0 = aet_slope_pf_0 *pf + aet_slope_melt_f_0*melt_f
        # aet_mb_1 = aet_slope_pf_1 *pf + aet_slope_melt_f_1*melt_f
        if model == None:
            aet_mbs = aet_slope_pfs * pf + aet_slope_melt_fs * melt_f
        else:
            aet_mbs = aet_slope_pfs * model.pf + aet_slope_melt_fs * model.melt_f
        # aet_mbs = aet.as_tensor_variable([aet_mb_0, aet_mb_1])
        # aet_slope_melt_fs = aet.vector(np.array([np.array(slope_melt_fs)[ys<=2010].mean(), np.array(slope_melt_fs)[ys>2010].mean()]))

        # this is not the new simple theano compatible
        # mass balance function that depends on pf and melt_f
        # aet_mbs = [aet_slope_pf_0, aet_slope_pf_1] *pf + aet_slope_melt_fs*melt_f

        # make a deterministic out of it to save it also in the traces
        mb_mod = pm.Deterministic('mb_mod', aet_mbs)
    with model_T:
        if use_two_msm:
            sigma = pm.Data('sigma', pd_geodetic_comp.loc[gd.rgi_id][
                ['err_dmdtda_2000_2010', 'err_dmdtda_2010_2020']].values * 1000)
            observed = pm.Data('observed', pd_geodetic_comp.loc[gd.rgi_id][
                ['dmdtda_2000_2010', 'dmdtda_2010_2020']].values * 1000)
            if nosigma == False:

                geodetic_massbal = pm.Normal('geodetic_massbal',
                                             mu=mb_mod,
                                             sigma=sigma,  # standard devia
                                             observed=observed)  # likelihood
            else:
                geodetic_massbal = pm.Normal('geodetic_massbal',
                                             mu=mb_mod,
                                             observed=observed)  # likelihood

            diff_geodetic_massbal = pm.Deterministic("diff_geodetic_massbal",
                                                     geodetic_massbal - observed)
        else:
            # sigma and observed need to have dim 1 (not zero), --> [value]
            sigma = pm.Data('sigma', [
                pd_geodetic_comp.loc[gd.rgi_id]['err_dmdtda'] * 1000])
            observed = pm.Data('observed', [
                pd_geodetic_comp.loc[gd.rgi_id]['dmdtda'] * 1000])
            geodetic_massbal = pm.TruncatedNormal('geodetic_massbal',
                                                  mu=mb_mod,
                                                  sigma=sigma,  # standard devia
                                                  observed=observed,
                                                  lower=max_allowed_specificMB)  # likelihood
            diff_geodetic_massbal = pm.Deterministic("diff_geodetic_massbal",
                                                     geodetic_massbal - observed)
        # constrained already by using TruncatedNormal geodetic massbalance ...
        # pot_max_melt = pm.Potential('pot_max_melt', aet.switch(
        #    geodetic_massbal < max_allowed_specificMB, -np.inf, 0))

        # also compute this difference just to be sure ...
        prior = pm.sample_prior_predictive(random_seed=random_seed,
                                           samples=1000)  # , keep_size = True)
    with model_T:
        if sampler == 'nuts':
            trace = pm.sample(20000, chains=4, tune=20000, target_accept=0.98,
                              compute_convergence_checks=True,
                              return_inferencedata=True)
        #                 #start={'pf':2.5, 'melt_f': 200})
        elif sampler == 'jax':
            import pymc3.sampling_jax
            trace = pm.sampling_jax.sample_numpyro_nuts(20000, chains=4,
                                                        tune=20000,
                                                        target_accept=0.98)  # , compute_convergence_checks= True)

    with model_T:
        burned_trace = trace.sel(draw=slice(5000, None))

        # trace = pm.sample(10000, chains=4, tune=10000, target_accept = 0.98)
        # need high target_accept to have no divergences, effective sample number
        #  and # We have stored the paths of all our variables, or "traces", in the trace variable.,
        # these paths are the routes the unknown parameters (here just 'n') have taken thus far.
        # Inference using the first few thousand points is a bad idea, as they are unrelated to the
        # final distribution we are interested in.
        # Thus is it a good idea to discard those samples before using the samples for inference.
        # We call this period before converge the burn-in period.
        # burned_trace = trace[1000:]
        # if arviz dataset
        if first_ppc:
            # TODO: then sometimes a problem occurs that a warning is raised
            #  about more chains (1000) than draws (2) ... why ???
            ppc = pm.sample_posterior_predictive(burned_trace,
                                                 random_seed=random_seed,
                                                 var_names=['geodetic_massbal',
                                                            'pf', 'melt_f',
                                                            'mb_mod',
                                                            'diff_geodetic_massbal'],
                                                 keep_size=True)
            az.concat(burned_trace,
                      az.from_dict(posterior_predictive=ppc, prior=prior),
                      inplace=True)

    ys_ref = gd.get_ref_mb_data(y0=y0, y1=y1).index.values
    with model_T:
        slope_pf_new = []
        slope_melt_f_new = []
        for y in ys_ref:
            slope_pf, slope_melt_f = get_slope_pf_melt_f(gd_mb, h=h, w=w, ys=y)
            slope_pf_new.append(slope_pf.mean())
            slope_melt_f_new.append(slope_melt_f.mean())
        pm.set_data(new_data={'aet_slope_melt_fs': slope_melt_f_new,
                              'aet_slope_pfs': slope_pf_new,
                              'observed': np.empty(len(ys_ref)),
                              'sigma': np.empty(len(ys_ref))})
        ppc_new = pm.sample_posterior_predictive(burned_trace,
                                                 random_seed=random_seed,
                                                 var_names=['geodetic_massbal',
                                                            'pf', 'melt_f',
                                                            'mb_mod',
                                                            'diff_geodetic_massbal'],
                                                 keep_size=True)
    predict_data = az.from_dict(posterior_predictive=ppc_new)
    return burned_trace, model_T, predict_data


def bayes_dummy_model_ref_std(uniform,
                              max_allowed_specificMB=None,
                              gd=None,
                              sampler='nuts', ys=np.arange(1979, 2020, 1), # with var_an_cycle only works with 1980 ...
                              gd_mb=None,
                              h=None, w=None, use_two_msm=True, nosigma=False,
                              nosigmastd=False, first_ppc=True,
                              pd_calib_opt=None,
                              pd_geodetic_comp=None, random_seed=42,
                              y0=None, y1=None):
    # test
    slope_pfs = []
    slope_melt_fs = []
    for y in ys:
        slope_pf, slope_melt_f = get_slope_pf_melt_f(gd_mb, h=h, w=w, ys=y)
        slope_pfs.append(slope_pf.mean())
        slope_melt_fs.append(slope_melt_f.mean())
    with pm.Model() as model_T:
        if uniform:
            melt_f = pm.Uniform("melt_f", lower=10, upper=1000)
            pf = pm.Uniform('pf', lower=0.1, upper=10)
        else:
            pf = pm.TruncatedNormal('pf', mu=pd_calib_opt['pf_opt'][
                pd_calib_opt.reg == 11.0].dropna().mean(),
                                    sigma=pd_calib_opt['pf_opt'][
                                        pd_calib_opt.reg == 11.0].dropna().std(),
                                    lower=0.1, upper=10)
            melt_f = pm.TruncatedNormal('melt_f',
                                        mu=pd_calib_opt['melt_f_opt_pf'][
                                            pd_calib_opt.reg == 11.0].dropna().mean(),
                                        sigma=pd_calib_opt['melt_f_opt_pf'][
                                            pd_calib_opt.reg == 11.0].dropna().std(),
                                        lower=10, upper=1000)

        ##
        if use_two_msm:
            # should not use the stuff before 2000
            aet_slope_melt_fs_two = pm.Data('aet_slope_melt_fs_two',
                                            [np.array(slope_melt_fs)[
                                                 (ys >= 2000) & (
                                                             ys <= 2009)].mean(),
                                             np.array(slope_melt_fs)[
                                                 ys >= 2010].mean()])
            aet_slope_pfs_two = pm.Data('aet_slope_pfs_two',
                                        ([np.array(slope_pfs)[(ys >= 2000) & (
                                                    ys <= 2009)].mean(),
                                          np.array(slope_pfs)[
                                              ys >= 2010].mean()]))
        else:
            aet_slope_melt_fs_two = pm.Data('aet_slope_melt_fs_two',
                                            [np.array(slope_melt_fs)[
                                                 ys >= 2000].mean()])
            aet_slope_pfs_two = pm.Data('aet_slope_pfs_two',
                                        [np.array(slope_pfs)[
                                             ys >= 2000].mean()])
        aet_mbs_two = aet_slope_pfs_two * pf + aet_slope_melt_fs_two * melt_f
        # make a deterministic out of it to save it also in the traces
        mb_mod = pm.Deterministic('mb_mod', aet_mbs_two)

        # std
        # need to put slope_melt_fs and slope_pfs into []???
        aet_slope_melt_fs = pm.Data('aet_slope_melt_fs',
                                    slope_melt_fs)  # pd.DataFrame(slope_melt_fs, columns=['slope_melt_fs'])['slope_melt_fs'])
        aet_slope_pfs = pm.Data('aet_slope_pfs',
                                slope_pfs)  # pd.DataFrame(slope_pfs, columns=['slope_pfs'])['slope_pfs'])
        aet_mbs = aet_slope_pfs * pf + aet_slope_melt_fs * melt_f
        mod_std = pm.Deterministic('mod_std', aet_mbs.std())

        if use_two_msm:
            sigma = pm.Data('sigma', pd_geodetic_comp.loc[gd.rgi_id][
                ['err_dmdtda_2000_2010', 'err_dmdtda_2010_2020']].values * 1000)
            observed = pm.Data('observed', pd_geodetic_comp.loc[gd.rgi_id][
                ['dmdtda_2000_2010', 'dmdtda_2010_2020']].values * 1000)
            if nosigma == False:
                geodetic_massbal = pm.Normal('geodetic_massbal',
                                             mu=mb_mod,
                                             sigma=sigma,  # standard devia
                                             observed=observed)  # likelihood
            else:
                geodetic_massbal = pm.Normal('geodetic_massbal',
                                             mu=mb_mod,
                                             observed=observed)  # likelihood
            # diff_geodetic_massbal = pm.Deterministic("diff_geodetic_massbal",
            #                                      geodetic_massbal - observed)
        else:
            # sigma and observed need to have dim 1 (not zero), --> [value]
            sigma = pm.Data('sigma', [
                pd_geodetic_comp.loc[gd.rgi_id]['err_dmdtda'] * 1000])
            observed = pm.Data('observed', [
                pd_geodetic_comp.loc[gd.rgi_id]['dmdtda'] * 1000])
            if nosigma == False:
                # likelihood
                geodetic_massbal = pm.TruncatedNormal('geodetic_massbal',
                                                      mu=mb_mod,
                                                      sigma=sigma,
                                                      # standard devia
                                                      observed=observed,
                                                      lower=max_allowed_specificMB)
            else:
                geodetic_massbal = pm.TruncatedNormal('geodetic_massbal',
                                                      mu=mb_mod,
                                                      observed=observed,
                                                      lower=max_allowed_specificMB)  # likelihood

            # constrained already by using TruncatedNormal geodetic massbalance ...
            # pot_max_melt = pm.Potential('pot_max_melt', aet.switch(
            #    geodetic_massbal < max_allowed_specificMB, -np.inf, 0))
        diff_geodetic_massbal = pm.Deterministic("diff_geodetic_massbal",
                                                 geodetic_massbal - observed)

        # pot_max_melt = pm.Potential('pot_max_melt', aet.switch(geodetic_massbal < max_allowed_specificMB, -np.inf, 0) )

        # std
        # sigma = pm.Data('sigma', 100) # how large are the uncertainties of the direct glaciological method !!!
        ref_df = gd.get_ref_mb_data(y0=y0, y1=y1)
        sigma_std = aet.constant((ref_df['ANNUAL_BALANCE'].values / 10).std())  # how large are the uncertainties of the direct glaciological method !!!
        observed_std = aet.constant(ref_df['ANNUAL_BALANCE'].values.std())

        # std should always be above zero
        if nosigmastd:
            glaciological_std = pm.TruncatedNormal('glaciological_std',
                                                   mu=mod_std,
                                                   # sigma=sigma_std,
                                                   observed=observed_std,
                                                   lower=0.001)  # likelihood
        else:
            glaciological_std = pm.TruncatedNormal('glaciological_std',
                                                   mu=mod_std, sigma=sigma_std,
                                                   observed=observed_std,
                                                   lower=0.001)  # likelihood

        quot_std = pm.Deterministic("quot_std",
                                    glaciological_std / observed_std)
        # pot_std = pm.Potential('pot_std', aet.switch(mod_std <= 0, -np.inf, 0) )
        prior = pm.sample_prior_predictive(random_seed=random_seed,
                                           samples=1000)  # , keep_size = True)

    with model_T:
        # sampling
        if sampler == 'nuts':
            trace = pm.sample(20000, chains=3, tune=20000, target_accept=0.99, # 25000
                              compute_convergence_checks=True,
                              return_inferencedata=True)
        #                 #start={'pf':2.5, 'melt_f': 200})
        elif sampler == 'jax':
            import pymc3.sampling_jax
            trace = pm.sampling_jax.sample_numpyro_nuts(20000, chains=4,
                                                        tune=20000,
                                                        target_accept=0.98)  # , compute_convergence_checks= True)

        burned_trace = trace.sel(draw=slice(5000, None))
        burned_trace.posterior['draw'] = np.arange(0, len(burned_trace.posterior.draw))
        burned_trace.log_likelihood['draw'] = np.arange(0, len(burned_trace.posterior.draw))
        burned_trace.sample_stats['draw']  = np.arange(0, len(burned_trace.posterior.draw))
        
        if first_ppc:
            print(az.summary(burned_trace.posterior))
            ppc = pm.sample_posterior_predictive(burned_trace,
                                                 random_seed=random_seed,
                                                 var_names=['geodetic_massbal',
                                                            'glaciological_std',
                                                            'pf', 'melt_f',
                                                            'mb_mod',
                                                            'diff_geodetic_massbal',
                                                            'quot_std'],
                                                 keep_size=True)
            az.concat(burned_trace, az.from_dict(posterior_predictive=ppc,
                                                 prior=prior), inplace=True)
    with model_T:
        slope_pf_new = []
        slope_melt_f_new = []
        for y in ys:
            slope_pf, slope_melt_f = get_slope_pf_melt_f(gd_mb, h=h, w=w, ys=y)
            slope_pf_new.append(slope_pf.mean())
            slope_melt_f_new.append(slope_melt_f.mean())
        pm.set_data(new_data={'aet_slope_melt_fs_two': slope_melt_f_new,
                              'aet_slope_pfs_two': slope_pf_new,
                              'observed': np.empty(len(ys)),
                              'sigma': np.empty(len(ys))})
        ppc_new = pm.sample_posterior_predictive(burned_trace,
                                                 random_seed=random_seed,
                                                 var_names=['geodetic_massbal',
                                                            'pf', 'melt_f',
                                                            'mb_mod',
                                                            'diff_geodetic_massbal'],
                                                 keep_size=True)
    predict_data = az.from_dict(posterior_predictive=ppc_new)
    return burned_trace, model_T, predict_data


#########################
def bayes_dummy_model_ref(uniform,
                          max_allowed_specificMB=None, gd=None,
                          sampler='nuts',
                          ys=None, gd_mb=None, h=None, w=None, use_two_msm=True,
                          nosigma=False, pd_calib_opt=None,
                          random_seed=4, y0=None, y1=None):
    # if use_two_msm:
    slope_pfs = []
    slope_melt_fs = []
    for y in ys:
        slope_pf, slope_melt_f = get_slope_pf_melt_f(gd_mb, h=h, w=w, ys=y)
        slope_pfs.append(slope_pf.mean())
        slope_melt_fs.append(slope_melt_f.mean())
    with pm.Model() as model_T:
        if uniform:
            melt_f = pm.Uniform("melt_f", lower=10, upper=1000)
            pf = pm.Uniform('pf', lower=0.1, upper=10)
        else:
            pf = pm.TruncatedNormal('pf', mu=pd_calib_opt['pf_opt'][
                pd_calib_opt.reg == 11.0].dropna().mean(),
                                    sigma=pd_calib_opt['pf_opt'][
                                        pd_calib_opt.reg == 11.0].dropna().std(),
                                    lower=0.5, upper=10)
            melt_f = pm.TruncatedNormal('melt_f',
                                        mu=pd_calib_opt['melt_f_opt_pf'][
                                            pd_calib_opt.reg == 11.0].dropna().mean(),
                                        sigma=pd_calib_opt['melt_f_opt_pf'][
                                            pd_calib_opt.reg == 11.0].dropna().std(),
                                        lower=1, upper=1000)
            # need to put slope_melt_fs and slope_pfs into [], other wise it does not work for jay
        aet_slope_melt_fs = pm.Data('aet_slope_melt_fs',
                                    slope_melt_fs)  # pd.DataFrame(slope_melt_fs, columns=['slope_melt_fs'])['slope_melt_fs'])
        aet_slope_pfs = pm.Data('aet_slope_pfs',
                                slope_pfs)  # pd.DataFrame(slope_pfs, columns=['slope_pfs'])['slope_pfs'])
        aet_mbs = aet_slope_pfs * pf + aet_slope_melt_fs * melt_f
        mb_mod = pm.Deterministic('mb_mod', aet_mbs)
    with model_T:
        ref_df = gd.get_ref_mb_data(y0=y0, y1=y1)
        # sigma = pm.Data('sigma', 100) # how large are the uncertainties of the direct glaciological method !!!
        sigma = pm.Data('sigma',
                        100)  # np.abs(ref_df['ANNUAL_BALANCE'].values/10)) # how large are the uncertainties of the direct glaciological method !!!
        observed = pm.Data('observed', ref_df['ANNUAL_BALANCE'].values)
        if nosigma:
            geodetic_massbal = pm.TruncatedNormal('geodetic_massbal',
                                                  mu=mb_mod,  # sigma=sigma,
                                                  observed=observed,
                                                  lower=max_allowed_specificMB)
        else:
            geodetic_massbal = pm.TruncatedNormal('geodetic_massbal',
                                                  mu=mb_mod, sigma=sigma,
                                                  observed=observed,
                                                  lower=max_allowed_specificMB)  # likelihood

        diff_geodetic_massbal = pm.Deterministic("diff_geodetic_massbal",
                                                 geodetic_massbal - observed)
        # pot_max_melt = pm.Potential('pot_max_melt', aet.switch(geodetic_massbal < max_allowed_specificMB, -np.inf, 0) )
        prior = pm.sample_prior_predictive(random_seed=random_seed,
                                           samples=1000)  # , keep_size = True)
        if sampler == 'nuts':
            trace = pm.sample(10000, chains=4, tune=10000, target_accept=0.98,
                              compute_convergence_checks=True,
                              return_inferencedata=True)
        #                 #start={'pf':2.5, 'melt_f': 200})
        elif sampler == 'jax':
            import pymc3.sampling_jax
            trace = pm.sampling_jax.sample_numpyro_nuts(20000, chains=4,
                                                        tune=20000,
                                                        target_accept=0.98)  # , compute_convergence_checks= True)

    with model_T:
        burned_trace = trace.sel(draw=slice(5000, None))
        az.summary(burned_trace.posterior)
        ppc = pm.sample_posterior_predictive(burned_trace,
                                             random_seed=random_seed,
                                             var_names=['geodetic_massbal',
                                                        'pf', 'melt_f',
                                                        'mb_mod',
                                                        'diff_geodetic_massbal'],
                                             keep_size=True)
        az.concat(burned_trace,
                  az.from_dict(posterior_predictive=ppc, prior=prior),
                  inplace=True)

    # with model_T:
    #     slope_pf_new = []
    #     slope_melt_f_new = []
    #     for y in ys:
    #             slope_pf, slope_melt_f = get_slope_pf_melt_f(gd_mb, h = h, w =w, ys = y)
    #             slope_pf_new.append(slope_pf.mean())
    #             slope_melt_f_new.append(slope_melt_f.mean())
    #     if nosigma:
    #         pm.set_data(new_data={'aet_slope_melt_fs': slope_melt_f_new, 'aet_slope_pfs':slope_pf_new,
    #                 'observed':np.empty(len(ys))}) # , 'sigma':np.empty(len(ys))})
    ##    else:
    #        pm.set_data(new_data={'aet_slope_melt_fs': slope_melt_f_new, 'aet_slope_pfs':slope_pf_new,
    #                'observed':np.empty(len(ys)), 'sigma':np.empty(len(ys))})
    ##   ppc_new = pm.sample_posterior_predictive(burned_trace, random_seed=random_seed,
    #                               var_names=['geodetic_massbal', 'pf', 'melt_f', 'mb_mod','diff_geodetic_massbal'],
    #                               keep_size = True)
    # predict_data = az.from_dict(posterior_predictive=ppc_new)
    return burned_trace, model_T  # , predict_data
    # idata_kwargs={"density_dist_obs": False}