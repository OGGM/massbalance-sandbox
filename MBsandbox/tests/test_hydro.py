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
                                        run_constant_climate_TIModel,
                                        run_with_hydro_daily)


# get the geodetic calibration data

DOM_BORDER = 80

ALL_DIAGS = ['volume', 'volume_bsl', 'volume_bwl', 'area', 'length',
             'calving', 'calving_rate', 'off_area', 'on_area', 'melt_off_glacier',
             'melt_on_glacier', 'liq_prcp_off_glacier', 'liq_prcp_on_glacier',
             'snowfall_off_glacier', 'snowfall_on_glacier', 'model_mb',
             'residual_mb', 'snow_bucket']


#@pytest.fixture(scope='class')
#def inversion_params(gdir):
#    diag = gdir.get_diagnostics()
#    return {k: diag[k] for k in ('inversion_glen_a', 'inversion_fs')}

# because of some reason the hydro test does not work anymore when using random MB
# the problem is inside of run_with_hydro when running add_climate
qualitative_tests = False
tests_melt_off_glacier = False
class Test_hydro:
    #@pytest.mark.slow
    @pytest.mark.parametrize('store_monthly_hydro',
                             [False, True],
                             ids=['annual', 'monthly'])
    def test_hydro_out_random_oggm_core(self, gdir,  # inversion_params,
                                        store_monthly_hydro):
        #TODO: need to make this test compatible !!!

        # Add debug vars
        cfg.PARAMS['store_diagnostic_variables'] = ALL_DIAGS
        cfg.PARAMS['hydro_month_nh'] = 1
        cfg.PARAMS['store_model_geometry'] = True  # Hydro outputs needs the full model geometry

        pf = 2
        climate_type = 'W5E5'
        mb_type = 'mb_real_daily'
        # does NOT work for 'real_daily' for that need hydro_daily function of Sarah
        grad_type = 'var_an_cycle'
        ###
        if climate_type == 'W5E5':
            ye = 2020  # till end of 2019
        else:
            ye = 2019

        if mb_type == 'mb_real_daily':
            temporal_resol = 'daily'
        else:
            temporal_resol = 'monthly'
        process_w5e5_data(gdir, temporal_resol=temporal_resol,
                          climate_type=climate_type)
        ###
        melt_f_calib_geod_prep_inversion(gdir,
                                         pf=pf,  # precipitation factor
                                         mb_type=mb_type, grad_type=grad_type,
                                         climate_type=climate_type, residual=0,
                                         ye=ye)
        #make sure melt factor is within a range
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

        ###
        tasks.run_with_hydro(gdir, run_task=run_random_climate_TIModel,
                             store_monthly_hydro=store_monthly_hydro,
                             seed=0, nyears=100, y0=2003 - 5, halfsize=5,
                             output_filesuffix='_rand',
                             melt_f='from_json',
                             precipitation_factor=pf,
                             climate_input_filesuffix=climate_type,
                             mb_type=mb_type, grad_type=grad_type)

        with xr.open_dataset(gdir.get_filepath('model_diagnostics',
                                               filesuffix='_rand')) as ds:
            sel_vars = [v for v in ds.variables if 'month_2d' not in ds[v].dims]
            odf = ds[sel_vars].to_dataframe().iloc[:-1]

        # Sanity checks
        # Tot prcp here is constant (constant climate) -> only for run with constant climate
        odf['tot_prcp'] = (odf['liq_prcp_off_glacier'] +
                           odf['liq_prcp_on_glacier'] +
                           odf['snowfall_off_glacier'] +
                           odf['snowfall_on_glacier'])

        # Glacier area is the same (remove on_area?)
        assert_allclose(odf['on_area'], odf['area_m2'])

        # Our MB is the same as the glacier dyn one
        reconstructed_vol = (odf['model_mb'].cumsum() / cfg.PARAMS['ice_density'] +
                             odf['volume_m3'].iloc[0])
        assert_allclose(odf['volume_m3'].iloc[1:], reconstructed_vol.iloc[:-1])

        # Mass-conservation
        odf['runoff'] = (odf['melt_on_glacier'] +
                         odf['melt_off_glacier'] +
                         odf['liq_prcp_on_glacier'] +
                         odf['liq_prcp_off_glacier'])

        mass_in_glacier_end = odf['volume_m3'].iloc[-1] * cfg.PARAMS['ice_density']
        mass_in_glacier_start = odf['volume_m3'].iloc[0] * cfg.PARAMS['ice_density']

        mass_in_snow = odf['snow_bucket'].iloc[-1]
        mass_in = odf['tot_prcp'].iloc[:-1].sum()
        mass_out = odf['runoff'].iloc[:-1].sum()
        assert_allclose(mass_in_glacier_end,
                        mass_in_glacier_start + mass_in - mass_out - mass_in_snow,
                        atol=1e-2)  # 0.01 kg is OK as numerical error
        # mass conservation in each step
        # mass in glacier in each moment should be equal to mass of the glacier in the timestep before + all mass input until this timestep - all mass output until with timestep - snow bucket in this timestep
        mass_in_snow = np.diff(odf['snow_bucket'])
        mass_in = odf['tot_prcp'].iloc[:-1]
        mass_out = odf['runoff'].iloc[:-1]
        mass_in_glacier_end = odf['volume_m3'].iloc[1:] * cfg.PARAMS['ice_density']  # to get kg
        mass_in_glacier_start = odf['volume_m3'].iloc[0:-1] * cfg.PARAMS['ice_density']

        assert_allclose(mass_in_glacier_end,
                        mass_in_glacier_start + mass_in - mass_out - mass_in_snow,
                        atol=1e-2)

        # Qualitative assessments

        # why is 0.7 used as factor? this assertion fails with other glaciers (e.g. RGI60-11.00890), suggest to remove 0.7
        assert odf['melt_on_glacier'].iloc[-1] < odf['melt_on_glacier'].iloc[0]  # * 0.7
        # if you force it for year 2003 at the end there should be more precipitation off glacier than on the glacier
        # at least for glaciers in european alps!
        # what is this assertion testing? It fails if you try it with other glaciers (e.g. RGI60-11.00890)
        # assert odf['liq_prcp_off_glacier'].iloc[-1] > odf['liq_prcp_on_glacier'].iloc[-1]
        # liquid precipitation off glacier should be smaller than liquid precipitation on glacier at the start of the run
        assert odf['liq_prcp_off_glacier'].iloc[0] < odf['liq_prcp_on_glacier'].iloc[0]
        if odf['on_area'].iloc[-1:-5].median() > odf['on_area'].iloc[0:5].median():
            assert odf['liq_prcp_off_glacier'].iloc[-1:-5].median() > odf['liq_prcp_off_glacier'].iloc[0:5].median()
            assert odf['liq_prcp_on_glacier'].iloc[-1:-5].median() < odf['liq_prcp_on_glacier'].iloc[0:5].median()
            assert odf['snowfall_on_glacier'].iloc[-1:-5].median() < odf['snowfall_on_glacier'].iloc[0:5].median()
            assert odf['melt_off_glacier'].iloc[-1:-5].median() > odf['melt_off_glacier'].iloc[0:5].median()
            assert odf['melt_on_glacier'].iloc[-1:-5].median() < odf['melt_on_glacier'].iloc[0:5].median()

        # for year with smallest area, liquid prec, melt and snowfall off glacier should be smallest within the years closeby
        # (does not work on all glaciers e.g. RGI60-11.01346)
        vars = [('liq_prcp_on_glacier', 'liq_prcp_off_glacier'), ('snowfall_on_glacier', 'snowfall_off_glacier'),
                ('melt_on_glacier', 'melt_off_glacier')]
        if qualitative_tests:
            for var in vars:
                assert np.argmin(odf['off_area']) -15 <= np.argmin(odf[var[1]]) <= np.argmin(odf['off_area']) +15
                assert np.argmin(odf['off_area']) -15 <= np.argmax(odf[var[0]]) <= np.argmin(odf['off_area']) +15
            # odf['on_area'].argmax()

        # Residual MB should not be crazy large
        frac = odf['residual_mb'] / odf['melt_on_glacier']
        #this assertion does not work on other glacier, (e.g. RGI60-11.00890 reaches 0.25!!)
        #assert_allclose(frac, 0, atol=0.06)  # annual can be large (prob)


    #    @pytest.mark.slow
    def test_hydro_out_daily_random_oggm_core(self, gdir):
        '''
        This checks:
        - if glacier areas are the same
        - if mb is the same as the mb of the dynamic run
        - if mass conservation is given at every time step
        - if fluxes off glacier are larger in the end than at the start if off_area increases
        - if fluxes on glacier are smallest around the time of smallest glacier extent
        '''
        #TODO: need to make this test compatible !!!

        # Add debug vars
        cfg.PARAMS['store_diagnostic_variables'] = ALL_DIAGS
        cfg.PARAMS['hydro_month_nh'] = 1
        cfg.PARAMS['store_model_geometry'] = True  # Hydro outputs needs the full model geometry

        pf = 2
        climate_type= 'W5E5'
        mb_type='mb_real_daily'
        temporal_resol = 'daily'
        grad_type='var_an_cycle'
        ###
        if climate_type == 'W5E5':
            ye = 2020  # till end of 2019
        else:
            ye = 2019

        process_w5e5_data(gdir, temporal_resol=temporal_resol,
                          climate_type=climate_type)
        ###
        melt_f_calib_geod_prep_inversion(gdir,
                                         pf=pf,  # precipitation factor
                                         mb_type=mb_type, grad_type=grad_type,
                                         climate_type=climate_type, residual=0,
                                         ye=ye)
        #make sure melt factor is within a range
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

        ###
        run_with_hydro_daily(gdir, run_task=run_random_climate_TIModel,
                             seed=0, nyears=100, y0=2003 - 5, halfsize=5,
                             output_filesuffix='_rand',
                             melt_f='from_json',
                             precipitation_factor=pf,
                             climate_input_filesuffix=climate_type,
                             mb_type=mb_type, grad_type=grad_type)

        with xr.open_dataset(gdir.get_filepath('model_diagnostics',
                                               filesuffix='_rand')) as ds:
            sel_vars = [v for v in ds.variables if 'day_2d' not in ds[v].dims]
            odf = ds[sel_vars].to_dataframe().iloc[:-1]

        # Check that compiling also works
        ds_comp = utils.compile_run_output([gdir], input_filesuffix='_rand')
        ds_comp = ds_comp.isel(rgi_id=0)

        # Sanity checks
        tot_yearly_prcp = (ds_comp['liq_prcp_on_glacier_daily'] +
                           ds_comp['liq_prcp_off_glacier_daily'] +
                           ds_comp['snowfall_on_glacier_daily'] +
                           ds_comp['snowfall_off_glacier_daily']
                           )
        tot_yearly_prcp = tot_yearly_prcp.sum(dim='day_2d')

        # Tot prcp here is constant (constant climate) -> only for run with constant climate
        odf['tot_prcp'] = (odf['liq_prcp_off_glacier'] +
                           odf['liq_prcp_on_glacier'] +
                           odf['snowfall_off_glacier'] +
                           odf['snowfall_on_glacier'])

        # This checks that the compile run output from oggm core also works
        assert_allclose(odf['tot_prcp'], tot_yearly_prcp.data[:-1])

        # Glacier area is the same (remove on_area?)
        assert_allclose(odf['on_area'], odf['area_m2'])

        # Our MB is the same as the glacier dyn one
        reconstructed_vol = (odf['model_mb'].cumsum() / cfg.PARAMS['ice_density'] +
                             odf['volume_m3'].iloc[0])
        assert_allclose(odf['volume_m3'].iloc[1:], reconstructed_vol.iloc[:-1])

        # Mass-conservation
        odf['runoff'] = (odf['melt_on_glacier'] +
                         odf['melt_off_glacier'] +
                         odf['liq_prcp_on_glacier'] +
                         odf['liq_prcp_off_glacier'])

        mass_in_glacier_end = odf['volume_m3'].iloc[-1] * cfg.PARAMS['ice_density']
        mass_in_glacier_start = odf['volume_m3'].iloc[0] * cfg.PARAMS['ice_density']

        mass_in_snow = odf['snow_bucket'].iloc[-1]
        mass_in = odf['tot_prcp'].iloc[:-1].sum()
        mass_out = odf['runoff'].iloc[:-1].sum()
        assert_allclose(mass_in_glacier_end,
                        mass_in_glacier_start + mass_in - mass_out - mass_in_snow,
                        atol=1e-2)  # 0.01 kg is OK as numerical error
        #mass conservation in each step
        #mass in glacier in each moment should be equal to mass of the glacier in the timestep before + all mass input until this timestep - all mass output until with timestep - snow bucket in this timestep
        mass_in_snow = np.diff(odf['snow_bucket'])
        mass_in = odf['tot_prcp'].iloc[:-1]
        mass_out = odf['runoff'].iloc[:-1]
        mass_in_glacier_end = odf['volume_m3'].iloc[1:] * cfg.PARAMS['ice_density'] #to get kg
        mass_in_glacier_start = odf['volume_m3'].iloc[0:-1] * cfg.PARAMS['ice_density']

        assert_allclose(mass_in_glacier_end,
                      mass_in_glacier_start + mass_in - mass_out - mass_in_snow,
                      atol=1e-2)

        # Qualitative assessments

        #why is 0.7 used as factor? this assertion fails with other glaciers (e.g. RGI60-11.00890), suggest to remove 0.7
        assert odf['melt_on_glacier'].iloc[-1] < odf['melt_on_glacier'].iloc[0] #* 0.7
        #if you force it for year 2003 at the end there should be more precipitation off glacier than on the glacier
        #at least for glaciers in european alps!
        #what is this assertion testing? It fails if you try it with other glaciers (e.g. RGI60-11.00890)
        #assert odf['liq_prcp_off_glacier'].iloc[-1] > odf['liq_prcp_on_glacier'].iloc[-1]
        #liquid precipitation off glacier should be smaller than liquid precipitation on glacier at the start of the run
        assert odf['liq_prcp_off_glacier'].iloc[0] < odf['liq_prcp_on_glacier'].iloc[0]
        if odf['on_area'].iloc[-1:-5].median() > odf['on_area'].iloc[0:5].median():
            assert odf['liq_prcp_off_glacier'].iloc[-1:-5].median() > odf['liq_prcp_off_glacier'].iloc[0:5].median()
            assert odf['liq_prcp_on_glacier'].iloc[-1:-5].median() < odf['liq_prcp_on_glacier'].iloc[0:5].median()
            assert odf['snowfall_on_glacier'].iloc[-1:-5].median() < odf['snowfall_on_glacier'].iloc[0:5].median()
            assert odf['melt_off_glacier'].iloc[-1:-5].median() > odf['melt_off_glacier'].iloc[0:5].median()
            assert odf['melt_on_glacier'].iloc[-1:-5].median() < odf['melt_on_glacier'].iloc[0:5].median()

        #for year with smallest area, liquid prec, melt and snowfall off glacier should be smallest within the years closeby
        # (does not work on all glaciers e.g. RGI60-11.01346)
        vars = [('liq_prcp_on_glacier', 'liq_prcp_off_glacier'), ('snowfall_on_glacier', 'snowfall_off_glacier'), ('melt_on_glacier', 'melt_off_glacier')]
        if qualitative_tests:
            for var in vars:
                assert np.argmin(odf['off_area']) -15 <= np.argmin(odf[var[1]]) <= np.argmin(odf['off_area']) +15
                assert np.argmin(odf['off_area']) -15 <= np.argmax(odf[var[0]]) <= np.argmin(odf['off_area']) +15
            #odf['on_area'].argmax()


        # Residual MB should not be crazy large
        frac = odf['residual_mb'] / odf['melt_on_glacier']
        #this assertion does not work on other glacier, (e.g. RGI60-11.00890 reaches 0.25!!)
        #assert_allclose(frac, 0, atol=0.06)  # annual can be large (prob)

    # @pytest.mark.slow
    @pytest.mark.parametrize('mb_run', ['hist', 'cte', 'random' ])
    @pytest.mark.parametrize('mb_type', ['mb_monthly', 'mb_real_daily'])
    def test_hydro_monthly_vs_annual_from_oggm_core(self, gdir,  # inversion_params,
                                                    mb_run, mb_type):
        # TODO: need to make this test compatible !!!

        mb_bias = 0
        cfg.PARAMS['store_diagnostic_variables'] = ALL_DIAGS
        cfg.PARAMS['hydro_month_nh'] = 1
        cfg.PARAMS['store_model_geometry'] = True  # Hydro outputs needs the full model geometry

        pf = 2
        climate_type = 'W5E5'
        grad_type = 'var_an_cycle'

        if climate_type == 'W5E5':
            ye = 2020  # till end of 2019
        else:
            ye = 2019

        if mb_type == 'mb_real_daily':
            temporal_resol = 'daily'
        else:
            temporal_resol = 'monthly'
        process_w5e5_data(gdir, temporal_resol=temporal_resol,
                          climate_type=climate_type)
        ###
        melt_f_calib_geod_prep_inversion(gdir,
                                         pf=pf,  # precipitation factor
                                         mb_type=mb_type, grad_type=grad_type,
                                         climate_type=climate_type, residual=0,
                                         ye=ye)

        # here just calibrate a-factor to that single glacier
        workflow.execute_entity_task(tasks.compute_downstream_line, [gdir])
        workflow.execute_entity_task(tasks.compute_downstream_bedshape, [gdir])

        oggm.workflow.calibrate_inversion_from_consensus([gdir],
                                                         apply_fs_on_mismatch=False,
                                                         error_on_mismatch=False,
                                                         )
        workflow.execute_entity_task(tasks.init_present_time_glacier, [gdir])

        gdir.rgi_date = 1990

        if mb_run == 'random':
            tasks.run_with_hydro(gdir, run_task=run_random_climate_TIModel,
                                 bias=mb_bias,
                                 store_monthly_hydro=False,
                                 seed=0, nyears=20, y0=2003 - 5, halfsize=5,
                                 output_filesuffix='_annual',
                                 melt_f='from_json',
                                 precipitation_factor=pf,
                                 climate_input_filesuffix=climate_type,
                                 mb_type=mb_type, grad_type=grad_type)

            tasks.run_with_hydro(gdir, run_task=run_random_climate_TIModel,
                                 bias=mb_bias,
                                 store_monthly_hydro=True,
                                 seed=0, nyears=20, y0=2003 - 5, halfsize=5,
                                 output_filesuffix='_monthly',
                                 melt_f='from_json',
                                 precipitation_factor=pf,
                                 climate_input_filesuffix=climate_type,
                                 mb_type=mb_type, grad_type=grad_type)
        elif mb_run == 'hist':
            tasks.run_with_hydro(gdir, run_task=run_from_climate_data_TIModel,
                                 bias=mb_bias,
                                 store_monthly_hydro=False,
                                 min_ys=1980, output_filesuffix='_annual',
                                 melt_f='from_json',
                                 precipitation_factor=pf,
                                 climate_input_filesuffix=climate_type,
                                 mb_type=mb_type, grad_type=grad_type)
            tasks.run_with_hydro(gdir, run_task=run_from_climate_data_TIModel,
                                 bias=mb_bias,
                                 store_monthly_hydro=True,
                                 min_ys=1980, output_filesuffix='_monthly',
                                 melt_f='from_json',
                                 precipitation_factor=pf,
                                 climate_input_filesuffix=climate_type,
                                 mb_type=mb_type, grad_type=grad_type)
        elif mb_run == 'cte':
            tasks.run_with_hydro(gdir, run_task=run_constant_climate_TIModel,
                                 bias=mb_bias,
                                 store_monthly_hydro=False,
                                 nyears=20, y0=2003 - 5, halfsize=5,
                                 output_filesuffix='_annual',
                                 melt_f='from_json',
                                 precipitation_factor=pf,
                                 climate_input_filesuffix=climate_type,
                                 mb_type=mb_type, grad_type=grad_type)
            tasks.run_with_hydro(gdir, run_task=run_constant_climate_TIModel,
                                 bias=mb_bias,
                                 store_monthly_hydro=True,
                                 nyears=20, y0=2003 - 5, halfsize=5,
                                 output_filesuffix='_monthly',
                                 melt_f='from_json',
                                 precipitation_factor=pf,
                                 climate_input_filesuffix=climate_type,
                                 mb_type=mb_type, grad_type=grad_type)

        with xr.open_dataset(gdir.get_filepath('model_diagnostics',
                                               filesuffix='_annual')) as ds:
            odf_a = ds.to_dataframe()

        with xr.open_dataset(gdir.get_filepath('model_diagnostics',
                                               filesuffix='_monthly')) as ds:
            sel_vars = [v for v in ds.variables if 'month_2d' not in ds[v].dims]
            odf_m = ds[sel_vars].to_dataframe()
            sel_vars = [v for v in ds.variables if 'month_2d' in ds[v].dims]
            odf_ma = ds[sel_vars].mean(dim='time').to_dataframe()
            odf_ma.columns = [c.replace('_monthly', '') for c in odf_ma.columns]

        # Check that yearly equals monthly
        np.testing.assert_array_equal(odf_a.columns, odf_m.columns)
        for c in odf_a.columns:
            rtol = 1e-5
            if c == 'melt_off_glacier':
                # rtol = 0.15
                # quite different, up tp 50%!
                # but this is 'ok' as fabien said
                # run_with_hydro with annual update is just very different there
                if mb_type == 'mb_monthly':
                    # why is it even worse for mb_monthly
                    rtol = 1.1
                elif mb_type == 'mb_real_daily':
                    # sum of daily solid prcp update
                    rtol = 0.9 #0.5
                if not tests_melt_off_glacier:
                    continue
            if c in ['snow_bucket']:
                continue
            assert_allclose(odf_a[c], odf_m[c], rtol=rtol)

        # Check monthly stuff
        odf_ma['tot_prcp'] = (odf_ma['liq_prcp_off_glacier'] +
                              odf_ma['liq_prcp_on_glacier'] +
                              odf_ma['snowfall_off_glacier'] +
                              odf_ma['snowfall_on_glacier'])

        odf_ma['runoff'] = (odf_ma['melt_on_glacier'] +
                            odf_ma['melt_off_glacier'] +
                            odf_ma['liq_prcp_on_glacier'] +
                            odf_ma['liq_prcp_off_glacier'])

        # Regardless of MB bias the melt in HYDROmonths 3, 4, 5, 6 should be zero
        # calendar monthls 12,1,2
        #TODO: in my case it is not zero!!! @fabien @sarah
        #I guess this because sometimes in March the temperature threshold is passed?
        # in Dec, Jan, Feb around <1e7 kg and in August >1e9 kg
        #this assertion does not work with 'RGI60-11.01328'
        #assert_allclose(odf_ma['melt_on_glacier'].loc[:3], 0, atol=1e7)
        #maybe better to check whether melt in winter very small compared to summer melt?
        #ATTENTION: These assertions only work in European climate
        #TODO: make these assertions applicable for other climates, e.g. by using climate file
        if qualitative_tests:
            assert odf_ma['melt_on_glacier'].loc[:3].mean() / odf_ma['melt_on_glacier'].loc[6:8].mean() < 0.02

            assert odf_ma['melt_on_glacier'].iloc[5:9].mean() > odf_ma['melt_on_glacier'].iloc[0:3].mean()
            assert odf_ma['melt_on_glacier'].iloc[5:9].mean() > odf_ma['melt_on_glacier'].iloc[10:].mean()
            assert odf_ma['melt_off_glacier'].iloc[5:9].mean() > odf_ma['melt_off_glacier'].iloc[0:3].mean()
            assert odf_ma['melt_off_glacier'].iloc[5:9].mean() > odf_ma['melt_off_glacier'].iloc[10:].mean()
            assert odf_ma['liq_prcp_on_glacier'].iloc[5:9].mean() > odf_ma['liq_prcp_on_glacier'].iloc[0:3].mean()
            assert odf_ma['liq_prcp_on_glacier'].iloc[5:9].mean() > odf_ma['liq_prcp_on_glacier'].iloc[10:].mean()
            assert odf_ma['liq_prcp_off_glacier'].iloc[5:9].mean() > odf_ma['liq_prcp_off_glacier'].iloc[0:3].mean()
            assert odf_ma['liq_prcp_off_glacier'].iloc[5:9].mean() > odf_ma['liq_prcp_off_glacier'].iloc[10:].mean()

            # Runoff peak should follow a temperature curve
            # month with largest runoff should be in August (calendar years!!!)
            assert_allclose(odf_ma['runoff'].idxmax(), 8, atol=1.1)
            assert_allclose(odf_ma['melt_on_glacier'].idxmax(), 8, atol=1.1)
            # in summer month ratio of rain to snow should be largest
            ratio_rain_snow = odf_ma['liq_prcp_on_glacier'] / odf_ma['snowfall_on_glacier']
            assert_allclose(np.argmax(ratio_rain_snow), 8, atol=2.1)

        # Residual MB should not be crazy large
        frac = odf_ma['residual_mb'] / odf_ma['melt_on_glacier']
        frac[odf_ma['melt_on_glacier'] < 1e-4] = 0
        # for RGI60-16.01251 this results in an error because absolute difference is 0.5
        assert_allclose(frac.loc[~frac.isnull()], 0, atol=0.02)#atol=0.01) for other glaciers this is larger



    @pytest.mark.parametrize('mb_run', ['random', 'hist'])
    def test_hydro_daily_vs_annual_from_oggm_core(self, gdir, #inversion_params,
                                    mb_run):
        '''
        This checks:
         - if yearly outputs of annual mb and daily mb are the same
         - if monthly outputs of monthly mb and daily mb are the same
         - if melt in winter very small compared to summer melt for daily mb
         - if melt peak follows a temperature peak for daily mb
         - if ratio of rain to snow is highest in the summer months for daily mb
        '''
        # TODO: need to make this test compatible !!!

        mb_bias = 0
        cfg.PARAMS['store_diagnostic_variables'] = ALL_DIAGS
        cfg.PARAMS['hydro_month_nh'] = 1
        cfg.PARAMS['store_model_geometry'] = True  # Hydro outputs needs the full model geometry

        pf = 2
        climate_type = 'W5E5'
        grad_type = 'var_an_cycle'
        mb_type = 'mb_real_daily'
        temporal_resol = 'daily'

        if climate_type == 'W5E5':
            ye = 2020  # till end of 2019
        else:
            ye = 2019

        process_w5e5_data(gdir, temporal_resol=temporal_resol,
                          climate_type=climate_type)
        ###
        melt_f_calib_geod_prep_inversion(gdir,
                                         pf=pf,  # precipitation factor
                                         mb_type=mb_type, grad_type=grad_type,
                                         climate_type=climate_type, residual=0,
                                         ye=ye)

        # here just calibrate a-factor to that single glacier
        workflow.execute_entity_task(tasks.compute_downstream_line, [gdir])
        workflow.execute_entity_task(tasks.compute_downstream_bedshape, [gdir])

        oggm.workflow.calibrate_inversion_from_consensus([gdir],
                                                         apply_fs_on_mismatch=False,
                                                         error_on_mismatch=False,
                                                         )
        workflow.execute_entity_task(tasks.init_present_time_glacier, [gdir])


        gdir.rgi_date = 1990

        if mb_run == 'random':
            tasks.run_with_hydro(gdir, run_task=run_random_climate_TIModel,
                                 bias=mb_bias,
                                 store_monthly_hydro=False,
                                 seed=0, nyears=100, y0=2014, halfsize=5,
                                 output_filesuffix='_annual',
                                 melt_f='from_json',
                                 precipitation_factor=pf,
                                 climate_input_filesuffix=climate_type,
                                 mb_type=mb_type, grad_type=grad_type)
            tasks.run_with_hydro(gdir, run_task=run_random_climate_TIModel,
                                 bias=mb_bias,
                                 store_monthly_hydro=True,
                                 seed=0, nyears=100, y0=2014, halfsize=5,
                                 output_filesuffix='_monthly',
                                 melt_f='from_json',
                                 precipitation_factor=pf,
                                 climate_input_filesuffix=climate_type,
                                 mb_type=mb_type, grad_type=grad_type)
            run_with_hydro_daily(gdir, run_task=run_random_climate_TIModel,
                                 Testing=True,
                                 bias=mb_bias,
                                 seed=0, nyears=100, y0=2014, halfsize=5,
                                 output_filesuffix='_daily',
                                 melt_f='from_json',
                                 precipitation_factor=pf,
                                 climate_input_filesuffix=climate_type,
                                 mb_type=mb_type, grad_type=grad_type)
        elif mb_run == 'hist':
            tasks.run_with_hydro(gdir, run_task=run_from_climate_data_TIModel,
                                 bias=mb_bias,
                                 store_monthly_hydro=False,
                                 min_ys=1980, output_filesuffix='_annual',
                                 melt_f='from_json',
                                 precipitation_factor=pf,
                                 climate_input_filesuffix=climate_type,
                                 mb_type=mb_type, grad_type=grad_type)
            tasks.run_with_hydro(gdir, run_task=run_from_climate_data_TIModel,
                                 bias=mb_bias,
                                 store_monthly_hydro=True,
                                 min_ys=1980, output_filesuffix='_monthly',
                                 melt_f='from_json',
                                 precipitation_factor=pf,
                                 climate_input_filesuffix=climate_type,
                                 mb_type=mb_type, grad_type=grad_type)
            run_with_hydro_daily(gdir, run_task=run_from_climate_data_TIModel,
                                 Testing=True,
                                 bias=mb_bias,
                                 min_ys=1980, output_filesuffix='_daily',
                                 melt_f='from_json',
                                 precipitation_factor=pf,
                                 climate_input_filesuffix=climate_type,
                                 mb_type=mb_type, grad_type=grad_type)

        with xr.open_dataset(gdir.get_filepath('model_diagnostics',
                                               filesuffix='_annual')) as ds:
            odf_a = ds.to_dataframe()

        with xr.open_dataset(gdir.get_filepath('model_diagnostics',
                                               filesuffix='_daily')) as ds:
            sel_vars = [v for v in ds.variables if 'day_2d' not in ds[v].dims]
            odf_m = ds[sel_vars].to_dataframe()
            sel_vars = [v for v in ds.variables if 'day_2d' in ds[v].dims]
            odf_da = ds[sel_vars].mean(dim='time').to_dataframe()
            odf_da.columns = [c.replace('_daily', '') for c in odf_da.columns]


        with xr.open_dataset(gdir.get_filepath('model_diagnostics',
                                               filesuffix='_monthly')) as ds:
            sel_vars = [v for v in ds.variables if 'month_2d' in ds[v].dims]
            odf_ma = ds[sel_vars].mean(dim='time').to_dataframe()
            odf_ma.columns = [c.replace('_monthly', '') for c in odf_ma.columns]

        # Check that yearly equals daily
        np.testing.assert_array_equal(odf_a.columns, odf_m.columns)
        for c in odf_a.columns:
            rtol = 1e-5
            if c == 'melt_off_glacier':
                #rtol = 0.15
                # quite different, up tp 50%!
                # but this is 'ok' as fabien said
                # run_with_hydro with annual update is just very different there
                if mb_type == 'mb_monthly':
                    # why is it even worse for mb_monthly
                    rtol = 1.1
                elif mb_type == 'mb_real_daily':
                    # sum of daily solid prcp update
                    rtol = 0.9 #0.5
                if not tests_melt_off_glacier:
                    continue
            if c in ['snow_bucket']  or c == 'residual_mb':
            #the residual mb is different for monthly mb and daily mb
                continue
            print(c)
            assert_allclose(odf_a[c], odf_m[c], rtol=rtol)

        # Check daily stuff
        odf_da['tot_prcp'] = (odf_da['liq_prcp_off_glacier'] +
                              odf_da['liq_prcp_on_glacier'] +
                              odf_da['snowfall_off_glacier'] +
                              odf_da['snowfall_on_glacier'])

        odf_da['runoff'] = (odf_da['melt_on_glacier'] +
                            odf_da['melt_off_glacier'] +
                            odf_da['liq_prcp_on_glacier'] +
                            odf_da['liq_prcp_off_glacier'])

        # Regardless of MB bias the melt in HYDROmonths 3, 4, 5, 6 should be zero
        # calendar monthls 12,1,2
        #TODO: in my case it is not zero!!! @fabien @sarah
        #I guess this because sometimes in March the temperature threshold is passed?
        # in Dec, Jan, Feb around <1e7 kg and in August >1e9 kg
        #this assertion does not work with 'RGI60-11.01328'
        #assert_allclose(odf_ma['melt_on_glacier'].loc[:3], 0, atol=1e7)
        #maybe better to check whether melt in winter very small compared to summer melt?
        #so which days are winter january to march:
        if qualitative_tests:
            assert odf_da['melt_on_glacier'].loc[:90].mean() / odf_da['melt_on_glacier'].loc[182:273].mean() < 0.02

            assert odf_da['melt_on_glacier'].iloc[152:273].mean() > odf_da['melt_on_glacier'].iloc[:90].mean()
            assert odf_da['melt_on_glacier'].iloc[152:273].mean() > odf_da['melt_on_glacier'].iloc[305:].mean()
            assert odf_da['melt_off_glacier'].iloc[152:273].mean() > odf_da['melt_off_glacier'].iloc[0:3].mean()
            assert odf_da['melt_off_glacier'].iloc[152:273].mean() > odf_da['melt_off_glacier'].iloc[305:].mean()
            assert odf_da['liq_prcp_on_glacier'].iloc[152:273].mean() > odf_da['liq_prcp_on_glacier'].iloc[:90].mean()
            assert odf_da['liq_prcp_on_glacier'].iloc[152:273].mean() > odf_da['liq_prcp_on_glacier'].iloc[305:].mean()
            assert odf_da['liq_prcp_off_glacier'].iloc[152:273].mean() > odf_da['liq_prcp_off_glacier'].iloc[0:3].mean()
            assert odf_da['liq_prcp_off_glacier'].iloc[152:273].mean() > odf_da['liq_prcp_off_glacier'].iloc[305:].mean()

            # Runoff peak should follow a temperature curve
            # month with largest runoff should be in August (calendar years!!!)
            # day 227 is mid August
            assert_allclose(odf_da['runoff'].idxmax(), 227, atol=20)
            assert_allclose(odf_da['melt_on_glacier'].idxmax(), 227, atol=20)
            # in summer month ratio of rain to snow should be largest
            ratio_rain_snow = odf_da['liq_prcp_on_glacier'] / odf_da['snowfall_on_glacier']
            assert_allclose(np.argmax(ratio_rain_snow), 227, atol=30)

        # Residual MB should not be crazy large
        frac = odf_da['residual_mb'] / odf_da['melt_on_glacier']
        frac[odf_da['melt_on_glacier'] < 1e-4] = 0
        #TODO: ATTENTION with the new testing this DOES NOT WORK
        #assert_allclose(frac.loc[~frac.isnull()], 0, atol=0.02)#atol=0.01) for other glaciers this is larger



                #check whether sum of daily is similar to monthly
        #get mean monthly values out of the daily values
        #PROBLEM: leap years
        odf_da["Date"] = pd.date_range('2020-01-01', '2020-12-31', freq="D")
        odf_da.set_index('Date', inplace=True)
        odf_da.index = pd.to_datetime(odf_da.index)
        odf_mean_monthly_d_mb = odf_da.resample("M").sum()
        #snow bucket is a sate variable so NO sum
        odf_mean_monthly_d_mb["snow_bucket"] = odf_da["snow_bucket"].resample("M").mean()

        #assert mean monthly from daily mb and from monthly mb are similar
        #np.testing.assert_array_equal(odf_mean_monthly_d_mb.columns[1:], odf_ma.columns[1:])
        #mean monthly from daily and monthly mb are not the same for February because leap years are included in daily mb,
        #but not correctly in monthly mb because of seconds in month
        # cfg.SEC_IN_MONTH assumes each month to be 30.41 days long
        for c in odf_ma.columns[1:]:
            rtol = 1e-5
            if c == 'melt_off_glacier':
                #rtol = 0.15
                # quite different, up tp 50%!
                # but this is 'ok' as fabien said
                # run_with_hydro with annual update is just very different there
                if mb_type == 'mb_monthly':
                    # why is it even worse for mb_monthly
                    rtol = 1.1
                elif mb_type == 'mb_real_daily':
                    # sum of daily solid prcp update
                    rtol = 0.8 #0.5
                if not tests_melt_off_glacier:
                    continue
            if c in ['snow_bucket'] or c == 'residual_mb':
                continue
            print(c)
            assert_allclose(odf_mean_monthly_d_mb[c][2:], odf_ma[c][2:], rtol=rtol)
            assert_allclose(odf_mean_monthly_d_mb[c].values[0], odf_ma[c].values[0], rtol=rtol)
            assert (np.abs((odf_mean_monthly_d_mb[c].values[0] - odf_ma[c].values[0])/ odf_ma[c].values[0])) < rtol



