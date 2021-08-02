import numpy as np
import pandas as pd
import xarray as xr
import scipy
import oggm
from oggm import cfg, utils, workflow, tasks
import pytest
from numpy.testing import assert_allclose

# import the MBsandbox modules
from MBsandbox.mbmod_daily_oneflowline import process_w5e5_data, TIModel_Sfc_Type
from MBsandbox.help_func import melt_f_calib_geod_prep_inversion
from MBsandbox.flowline_TIModel import (run_from_climate_data_TIModel,
                                        run_random_climate_TIModel,
                                        run_constant_climate_TIModel)
from MBsandbox.wip.projections_bayescalibration import process_isimip_data


# get the geodetic calibration data
url = 'https://cluster.klima.uni-bremen.de/~oggm/geodetic_ref_mb/hugonnet_2021_ds_rgi60_pergla_rates_10_20_worldwide.csv'
path = utils.file_downloader(url)
pd_geodetic = pd.read_csv(path, index_col='rgiid')
pd_geodetic = pd_geodetic.loc[pd_geodetic.period == '2000-01-01_2020-01-01']



class Test_sfc_type_run:
    #@pytest.mark.parametrize(update=['annual', 'monthly'])
    def test_sfc_type_diff_heights(self, gdir  # inversion_params,
                                    ):
        # TODO: need to make this test compatible !!!
        cfg.PARAMS['use_multiprocessing'] = False
        update = 'monthly'
        kwargs_for_TIModel_Sfc_Type = {'melt_f_change': 'linear', 'tau_e_fold_yr': 0.5, 'spinup_yrs': 6,
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
                                         path_geodetic=path, ye=ye_calib,
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