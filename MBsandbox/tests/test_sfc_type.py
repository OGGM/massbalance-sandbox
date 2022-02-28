import numpy as np
import pandas as pd
import xarray as xr
import scipy
import oggm
from oggm import cfg, utils, workflow, tasks
import pytest
from numpy.testing import assert_allclose

# import the MBsandbox modules
from MBsandbox.mbmod_daily_oneflowline import (process_w5e5_data, TIModel_Sfc_Type, TIModel,
                                               ConstantMassBalance_TIModel, RandomMassBalance_TIModel)

from MBsandbox.help_func import melt_f_calib_geod_prep_inversion, calib_inv_run
from MBsandbox.flowline_TIModel import (run_from_climate_data_TIModel,
                                        run_random_climate_TIModel,
                                        run_constant_climate_TIModel)
from MBsandbox.wip.projections_bayescalibration import process_isimip_data


class Test_sfc_type_run:
    #@pytest.mark.parametrize(update=['annual', 'monthly'])
    @pytest.mark.parametrize("melt_f_update", ['annual', 'monthly'])
    def test_run_constant_mb(self, gdir, melt_f_update):
        # test if we get the same result out when using TIModel and
        # using TIModel_Sfc_type using the interpolation and with ratio of 1
        # (at the beginning this was not the case, because the mb was not
        # computed again in case of monthly update stuff ... )
        cfg.PARAMS['hydro_month_nh'] = 1
        gdirs = [gdir]
        workflow.execute_entity_task(tasks.compute_downstream_line, gdirs)
        workflow.execute_entity_task(tasks.compute_downstream_bedshape, gdirs)

        # if you have a precipitation factor from the hydrological model you can change it here
        pf = 2  # we set the precipitation factor here to 1
        climate_type = 'W5E5'  # W5E5 # 'WFDE5_CRU'
        # climate_type='WFDE5_CRU' -> need to use other pf and temp_bias ...
        # mb_type = 'mb_real_daily' #real daily input, this would be monthly input:'mb_monthly' #'mb_real_daily' # 'mb_monthly'#
        mb_type = 'mb_monthly'
        grad_type = 'cte'
        ensemble = 'mri-esm2-0_r1i1p1f1'
        ssp = 'ssp126'
        dataset = climate_type
        temporal_resol = 'monthly'

        # get the climate data (either w5e5 or WFDE5_CRU)
        workflow.execute_entity_task(process_w5e5_data, gdirs,
                                     temporal_resol=temporal_resol, climate_type=climate_type)
        process_isimip_data(gdir, ensemble=ensemble,
                                     ssp=ssp, temporal_resol=temporal_resol,
                                     climate_historical_filesuffix='_{}_{}'.format(temporal_resol,
                                                                                   dataset)
                            )

        nyears = 10

        out = calib_inv_run(gdir=gdirs[-1],
                            kwargs_for_TIModel_Sfc_Type={'melt_f_change': 'linear',
                                                         'tau_e_fold_yr': 20,
                                                         'spinup_yrs': 6,
                                                        'melt_f_update': melt_f_update,
                                                        'melt_f_ratio_snow_to_ice': 1},
                            mb_elev_feedback='annual',
                            nyears=nyears,
                            mb_type=mb_type,
                            grad_type=grad_type,
                            pf=pf, hs=1,
                            run_type='constant',
                            interpolation_optim=True, unique_samples=True)
        ds_cte_ratio1_a_fb, melt_f_cte_ratio1_a_fb, run_model_cte_ratio1_a_fb = out
        out_2 = calib_inv_run(gdir=gdirs[-1], mb_model_sub_class=TIModel,
                              mb_elev_feedback='annual',
                              nyears=nyears, mb_type=mb_type,
                              grad_type=grad_type, pf=pf,
                              run_type='constant', hs=1, unique_samples=True)
        ds_default_cte_a_fb, melt_f_default_cte_a_fb, run_model_cte_a_fb = out_2
        np.testing.assert_allclose(ds_cte_ratio1_a_fb.volume,
                                   ds_default_cte_a_fb.volume)
    def test_run_climate_mb(self, gdir):

        cfg.PARAMS['hydro_month_nh'] = 1
        gdirs = [gdir]
        workflow.execute_entity_task(tasks.compute_downstream_line, gdirs)
        workflow.execute_entity_task(tasks.compute_downstream_bedshape, gdirs)
        # if you have a precipitation factor from the hydrological model you can change it here
        pf = 2  # we set the precipitation factor here to 1
        climate_type = 'W5E5'  # W5E5 # 'WFDE5_CRU'
        # climate_type='WFDE5_CRU' -> need to use other pf and temp_bias ...
        # mb_type = 'mb_real_daily' #real daily input, this would be monthly input:'mb_monthly' #'mb_real_daily' # 'mb_monthly'#
        mb_type = 'mb_monthly'
        grad_type = 'cte'
        ensemble = 'mri-esm2-0_r1i1p1f1'
        ssp = 'ssp126'
        dataset = climate_type
        temporal_resol = 'monthly'

        # get the climate data (either w5e5 or WFDE5_CRU)
        workflow.execute_entity_task(process_w5e5_data, gdirs,
                                     temporal_resol=temporal_resol, climate_type=climate_type)
        workflow.execute_entity_task(process_isimip_data, gdirs, ensemble=ensemble,
                                     ssp=ssp, temporal_resol=temporal_resol,
                                     climate_historical_filesuffix='_{}_{}'.format(temporal_resol,
                                                                                   dataset));

        nyears = 100
        for melt_f_update in ['annual', 'monthly']:
            ds_default_from_climate_TIModel, melt_f_default_from_climate_TIModel,_ = calib_inv_run(gdir=gdir,
                                                                                                 mb_model_sub_class=TIModel,
                                                                                                 mb_elev_feedback=melt_f_update,
                                                                                                 mb_type=mb_type,
                                                                                                 grad_type=grad_type, pf=pf,
                                                                                                 nyears=nyears,
                                                                                                 run_type='from_climate')
            ds_default_from_climate_TIModel_ratio1, melt_f_default_from_climate_TIModel_ratio1,_ = calib_inv_run(
                gdir=gdir, mb_model_sub_class=TIModel_Sfc_Type,
                kwargs_for_TIModel_Sfc_Type={'melt_f_change': 'linear', 'tau_e_fold_yr': 20,
                                             'spinup_yrs': 6,
                                             'melt_f_update': melt_f_update,
                                             'melt_f_ratio_snow_to_ice': 1},
                mb_elev_feedback=melt_f_update, mb_type=mb_type,
                grad_type=grad_type, pf=pf,
                nyears=nyears,
                run_type='from_climate')
            np.testing.assert_allclose(ds_default_from_climate_TIModel.volume,
                                       ds_default_from_climate_TIModel_ratio1.volume)

    def test_run_random_mb(self,gdir):
        # test if we get the same result out when using TIModel and using TIModel_Sfc_type
        # with ratio of 1 (at the beginning this was not the case, because the mb was not
        # computed again in case of monthly update stuff ...
        cfg.PARAMS['hydro_month_nh'] = 1
        gdirs = [gdir]
        workflow.execute_entity_task(tasks.compute_downstream_line, gdirs)
        workflow.execute_entity_task(tasks.compute_downstream_bedshape, gdirs)

        # if you have a precipitation factor from the hydrological model you can change it here
        pf = 2  # we set the precipitation factor here to 1
        climate_type = 'W5E5'  # W5E5 # 'WFDE5_CRU'
        # climate_type='WFDE5_CRU' -> need to use other pf and temp_bias ...
        # mb_type = 'mb_real_daily' #real daily input, this would be monthly input:'mb_monthly' #'mb_real_daily' # 'mb_monthly'#
        mb_type = 'mb_monthly'
        grad_type = 'cte'
        ensemble = 'mri-esm2-0_r1i1p1f1'
        ssp = 'ssp126'
        dataset = climate_type
        temporal_resol = 'monthly'

        # get the climate data (either w5e5 or WFDE5_CRU)
        workflow.execute_entity_task(process_w5e5_data, gdirs,
                                     temporal_resol=temporal_resol, climate_type=climate_type)
        workflow.execute_entity_task(process_isimip_data, gdirs, ensemble=ensemble,
                                     ssp=ssp, temporal_resol=temporal_resol,
                                     climate_historical_filesuffix='_{}_{}'.format(temporal_resol,
                                                                                   dataset));

        #calib_inv_run(gdir, )
        melt_f_update = 'monthly'
        nyears = 10
        out = calib_inv_run(gdir=gdirs[-1],  kwargs_for_TIModel_Sfc_Type={'melt_f_change': 'linear',
                                                                          'tau_e_fold_yr': 20, 'spinup_yrs': 6,
                                                                           'melt_f_update': melt_f_update,
                                                                            'melt_f_ratio_snow_to_ice': 1},
                           mb_elev_feedback='annual',
                           nyears=nyears,
                           mb_type=mb_type,
                           grad_type=grad_type,
                           pf=pf, hs=1,
                           run_type='random',
                           unique_samples=True)
        ds_random_ratio1_a_fb, melt_f_random_ratio1_a_fb, run_model_random_ratio1_a_fb = out
        out_2 = calib_inv_run(gdir=gdirs[-1], mb_model_sub_class=TIModel,
                              mb_elev_feedback='annual',
                              nyears=nyears, mb_type=mb_type, grad_type=grad_type, pf=pf,
                              run_type='random', hs=1, unique_samples=True)
        ds_default_random_a_fb, melt_f_default_random_a_fb, run_model_random_a_fb = out_2
        np.testing.assert_allclose(ds_random_ratio1_a_fb.volume, ds_default_random_a_fb.volume)

    # def test_random_mass_balance(self,gdir):
    #     cfg.PARAMS['use_multiprocessing'] = False
    #     update = 'annual'
    #     kwargs_for_TIModel_Sfc_Type = {'melt_f_change': 'linear',
    #                                    'tau_e_fold_yr': 0.5,
    #                                    'spinup_yrs': 6,
    #                                    'melt_f_update': update,
    #                                    'melt_f_ratio_snow_to_ice': 0.5,
    #                                    }
    #     # Add debug vars
    #     cfg.PARAMS['hydro_month_nh'] = 1
    #
    #     climate_type = 'W5E5'
    #     mb_type = 'mb_monthly'
    #     grad_type = 'var_an_cycle'
    #     dataset = climate_type
    #     ###
    #
    #     if mb_type == 'mb_real_daily':
    #         temporal_resol = 'daily'
    #     else:
    #         temporal_resol = 'monthly'
    #     process_w5e5_data(gdir, temporal_resol=temporal_resol,
    #                       climate_type=climate_type)
    #     rid = '_monthly_W5E5'
    #     mb_mod = RandomMassBalance_TIModel(gdir, melt_f = 400, prcp_fac=2,
    #                                 mb_model_sub_class=TIModel_Sfc_Type, residual=0,
    #                                 #nyears=20,
    #                                          y0=2003 - 5, halfsize=5,
    #                                 #output_filesuffix='_annual',
    #                                 #climate_
    #                                 #input_filesuffix=rid,
    #                                 baseline_climate=dataset,
    #                                 mb_type=mb_type, grad_type=grad_type,
    #                                        seed=0,
    #                                 **kwargs_for_TIModel_Sfc_Type
    #                                 )
    #     mb_mod.get_specific_mb(year=2003)
    def test_constant_mass_balance_sfc_type(self, gdir):

        cfg.PARAMS['use_multiprocessing'] = False
        update = 'annual'
        kwargs_for_TIModel_Sfc_Type = {'melt_f_change': 'linear',
                                       'tau_e_fold_yr': 0.5,
                                       'spinup_yrs': 6,
                                       'melt_f_update': update,
                                       'melt_f_ratio_snow_to_ice': 0.5,
                                       }
        # Add debug vars
        cfg.PARAMS['hydro_month_nh'] = 1

        climate_type = 'W5E5'
        mb_type = 'mb_monthly'
        grad_type = 'var_an_cycle'
        dataset = climate_type
        ###

        if mb_type == 'mb_real_daily':
            temporal_resol = 'daily'
        else:
            temporal_resol = 'monthly'
        process_w5e5_data(gdir, temporal_resol=temporal_resol,
                          climate_type=climate_type)
        rid = '_monthly_W5E5'
        mb_mod = ConstantMassBalance_TIModel(gdir, melt_f = 400, prcp_fac=2,
                                    mb_model_sub_class=TIModel_Sfc_Type, residual=0,
                                    #nyears=20,
                                             y0=2003 - 5, halfsize=5,
                                    #output_filesuffix='_annual',
                                    #climate_
                                    #input_filesuffix=rid,
                                    baseline_climate=dataset,
                                    mb_type=mb_type, grad_type=grad_type,
                                             interpolation_optim=True,
                                    **kwargs_for_TIModel_Sfc_Type
                                    )
        h, w = gdir.get_inversion_flowline_hw()
        mb_mod.reset_pd_mb_bucket()
        mb_mod.mbmod.reset_pd_mb_bucket()

        mb = mb_mod.get_annual_mb(h,
                                  spinup=True)
        # the available mb should include the spinup from the first year (2003-5-6)
        # this checks if the spinup is computed again or just computed once and then not
        # anymore

        mb0 = mb.copy()
        mb0_pd_bucket = mb_mod.mbmod.pd_bucket.copy()
        mb0_mb_annual = mb_mod.mbmod.pd_mb_annual.copy()
        mb1 = mb_mod.get_annual_mb(h,
                                  spinup=False)

        mb1_pd_bucket = mb_mod.mbmod.pd_bucket.copy()
        mb1_mb_annual = mb_mod.mbmod.pd_mb_annual.copy()

        np.testing.assert_allclose(mb0, mb1)
        np.testing.assert_allclose(mb0_pd_bucket, mb1_pd_bucket)
        np.testing.assert_allclose(mb0_mb_annual, mb1_mb_annual)

        pd_bucket = mb_mod.mbmod.pd_bucket.copy()
        assert np.all(mb_mod.mbmod.pd_mb_annual.columns == np.arange(2003-5-5-6,
                                                                     2003-5+5+1))
        # now compute it again, we should not get exactly the same output,
        # because of changes in the buckets ?!
        for n in np.arange(0, 20):
            mb20 = mb_mod.get_annual_mb(h, year=2000)
        pd_bucket20 = mb_mod.mbmod.pd_bucket.copy()
        # ok, apparently the mass balances are the same
        np.testing.assert_allclose(mb, mb20)
        np.testing.assert_allclose(pd_bucket.values, pd_bucket20.values)
        ### WHY is it the same???
        #kwargs_for_TIModel_Sfc_Type['interpolation_optim'] = True
        # ds = run_constant_climate_TIModel(gdir, bias=0,
        #                                   nyears=20, y0=2003 - 5, halfsize=5,
        #                                   output_filesuffix='_annual',
        #                                   melt_f=200,
        #                                   precipitation_factor=2,
        #                                   climate_input_filesuffix=rid,
        #                                   mb_type=mb_type, grad_type=grad_type,
        #                                   kwargs_for_TIModel_Sfc_Type=kwargs_for_TIModel_Sfc_Type)

    @pytest.mark.skip(reason="this test does not make sense anymore")
    # this does not work even with the old setup ...
    # might delete this at some point!
    def test_sfc_type_diff_heights(self, gdir  # inversion_params,
                                    ):
        # TODO: need to make this test compatible !!!
        cfg.PARAMS['use_multiprocessing'] = False
        update = 'monthly'
        kwargs_for_TIModel_Sfc_Type = {'melt_f_change': 'linear',
                                       'tau_e_fold_yr': 0.5, 'spinup_yrs': 6,
                                       'melt_f_update': update,
                                       'melt_f_ratio_snow_to_ice': 0.5}
        # Add debug vars
        cfg.PARAMS['hydro_month_nh'] = 1

        pf = 2
        climate_type = 'W5E5'
        mb_type = 'mb_monthly'
        grad_type = 'var_an_cycle'
        ensemble = 'mri-esm2-0_r1i1p1f1'
        ssp = 'ssp126'
        dataset = climate_type
        ###
        y0 = 1979
        ye = 2100
        ye_calib = 2020
        nosigmaadd = ''

        if mb_type == 'mb_real_daily':
            temporal_resol = 'daily'
        else:
            temporal_resol = 'monthly'
        process_w5e5_data(gdir, temporal_resol=temporal_resol,
                          climate_type=climate_type)
        process_isimip_data(gdir, ensemble=ensemble,
                                     ssp=ssp, temporal_resol=temporal_resol,
                                     climate_historical_filesuffix='_{}_{}'.format(temporal_resol, dataset));
        ###
        melt_f_calib_geod_prep_inversion(gdir,
                                         pf=pf,  # precipitation factor
                                         mb_type=mb_type, grad_type=grad_type,
                                         climate_type=climate_type, residual=0,
                                         ye=ye_calib,
                                         mb_model_sub_class=TIModel_Sfc_Type,
                                         kwargs_for_TIModel_Sfc_Type=kwargs_for_TIModel_Sfc_Type)
        # make sure melt factor is within a range
        fs = '_{}_{}_{}'.format(climate_type, mb_type, grad_type)
        melt_f = gdir.read_json(filename='melt_f_geod', filesuffix=fs).get('melt_f_pf_2')
        assert 10 <= melt_f <= 1000

        # here just calibrate a-factor to that single glacier
        workflow.execute_entity_task(tasks.compute_downstream_line, [gdir])
        workflow.execute_entity_task(tasks.compute_downstream_bedshape, [gdir])

        oggm.workflow.calibrate_inversion_from_consensus([gdir],
                                                         apply_fs_on_mismatch=False,
                                                         error_on_mismatch=False,
                                                         )
        workflow.execute_entity_task(tasks.init_present_time_glacier, [gdir])

        #mod = run_from_climate_data_TIModel(gdir,
        #                         temperature_bias=0,
        #                         ys=2003, # if WFDE5_CRU need to set y0=2013 because no data for 2019
        #                         store_monthly_step=False,
        #                         output_filesuffix='_constant_spinup_test',
        #                         bias=0,  # only tested with bias=0 !!!, don't change!
        #                         mb_type=mb_type,
        #                         grad_type=grad_type,
        #                         precipitation_factor=pf,
         #                        melt_f='from_json', #melt_f_file=melt_f_file, # need to set these to find the right calibrated melt_f
         #                        climate_input_filesuffix=climate_type,
         #                        mb_model_sub_class = TIModel_Sfc_Type,
         #                        kwargs_for_TIModel_Sfc_Type = {'melt_f_change':'linear'},
         #                        mb_elev_feedback = 'annual'
         #                        )

        a_factor = gdir.get_diagnostics()['inversion_glen_a'] / cfg.PARAMS['inversion_glen_a']
        # just a check if a-factor is set to the same value
        np.testing.assert_allclose(a_factor,
                                   gdir.get_diagnostics()['inversion_glen_a'] / cfg.PARAMS['inversion_glen_a'])

        # double check: volume sum of gdirs from Farinotti estimate is equal to oggm estimates
        #np.testing.assert_allclose(pd_inv_melt_f.sum()['vol_itmix_m3'], pd_inv_melt_f.sum()['vol_oggm_m3'], rtol=1e-2)

        ######

        add_msm = 'sfc_type_{}_tau_{}_{}_update'.format(kwargs_for_TIModel_Sfc_Type['melt_f_change'],
                                                        kwargs_for_TIModel_Sfc_Type['tau_e_fold_yr'],
                                                        kwargs_for_TIModel_Sfc_Type['melt_f_update'])
        j = 'test'

        add = 'pf{}'.format(pf)

        #fs = '_{}_{}_{}'.format(climate_type, mb_type, grad_type)
        #gdir.read_json(filename='melt_f_geod', filesuffix=fs)
        rid = '{}_{}_{}'.format('ISIMIP3b', ensemble, ssp)



        ds = run_constant_climate_TIModel(gdir, bias = 0,
                                          store_monthly_hydro = False,
                                     nyears = 20, y0 = 2003 - 5, halfsize = 5,
                                     output_filesuffix = '_annual',
                                     melt_f = 'from_json', precipitation_factor=pf,
                                     climate_input_filesuffix = rid,
                                     mb_type = mb_type, grad_type = grad_type)


        run_from_climate_data_TIModel(gdir, bias=0, min_ys=y0,
                                      ye=ye,
                                      mb_type=mb_type,
                                      climate_filename='gcm_data',
                                      grad_type=grad_type, precipitation_factor=pf,
                                      melt_f=melt_f,
                                      climate_input_filesuffix=rid,  # dataset,
                                      output_filesuffix='_{}{}_ISIMIP3b_{}_{}_{}_{}{}_hist_{}_{}'.format(nosigmaadd,
                                                                                                         add_msm,
                                                                                                         dataset,
                                                                                                         ensemble,
                                                                                                         mb_type,
                                                                                                         grad_type, add,
                                                                                                         ssp, j),
                                      mb_model_sub_class=TIModel_Sfc_Type,
                                      kwargs_for_TIModel_Sfc_Type=kwargs_for_TIModel_Sfc_Type,
                                      # {'melt_f_change':'linear'},
                                      #mb_elev_feedback='monthly'
                                      mb_elev_feedback = 'annual'
                                        )
        ds = utils.compile_run_output(gdir,
                                      input_filesuffix='_{}{}_ISIMIP3b_{}_{}_{}_{}{}_hist_{}_{}'.format(nosigmaadd,
                                                                                                        add_msm,
                                                                                                        dataset,
                                                                                                        ensemble,
                                                                                                        mb_type,
                                                                                                        grad_type, add,
                                                                                                        ssp, j))