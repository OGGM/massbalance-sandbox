import warnings
warnings.filterwarnings("once", category=DeprecationWarning)  # noqa: E402

import os
from functools import partial
import shutil
import copy
import time
import numpy as np
import xarray as xr
import pandas as pd
import shapely.geometry as shpg
from numpy.testing import assert_allclose
import pytest

from oggm import cfg, utils, workflow, tasks, graphics
from oggm.core import massbalance, flowline, climate
import logging
#log = logging.getLogger(__name__)
#import aesara.tensor as aet
import oggm
from MBsandbox.mbmod_daily_oneflowline import process_w5e5_data
from MBsandbox.wip.projections_bayescalibration import process_isimip_data, run_from_climate_data_TIModel, MultipleFlowlineMassBalance_TIModel
ensemble = 'mri-esm2-0_r1i1p1f1'

base_url = ('https://cluster.klima.uni-bremen.de/~oggm/gdirs/oggm_v1.4/'
            'L1-L2_files/elev_bands')
#@pytest.fixture(scope='class')
#def get_hef_gcms(gdir):

@pytest.mark.skip(reason="no way of currently testing this")
def test_extend_past_climate():
    # wip
    from MBsandbox.mbmod_daily_oneflowline import (
        compile_fixed_geometry_mass_balance_TIModel,
        extend_past_climate_run_TIModel)
    cfg.initialize()
    cfg.PATHS['working_dir'] = '/home/lilianschuster/Schreibtisch/PhD/oggm_files/all'
    gdirs = workflow.init_glacier_directories(['RGI60-11.00897'])
    gdir = gdirs[0]
    dataset = 'WFDE5_CRU'
    typ = 'mb_pseudo_daily_cte'
    ensemble = 'mri-esm2-0_r1i1p1f1'

    cfg.PARAMS['hydro_month_nh'] = 1
    ssp = 'ssp126'
    #process_w5e5_data(gdir, temporal_resol='monthly',
    #                   output_filesuffix='_monthly_WFDE5_CRU')
    #process_isimip_data(gdir, ensemble=ensemble, ssp=ssp,
    #                    climate_historical_filesuffix='_monthly_WFDE5_CRU',
    #                    temporal_resol='monthly'
    #                    )

    utils._workflow.compile_glacier_statistics([gdir], filesuffix=dataset)
    oggm.utils._workflow.compile_run_output([gdir],
                                            input_filesuffix='_ISIMIP3b_{}_mri-esm2-0_r1i1p1f1_{}_historical_test'.format(
                                                dataset, typ))
    compile_fixed_geometry_mass_balance_TIModel([gdir],
                                                climate_input_filesuffix=dataset,
                                                filesuffix='_{}_{}'.format(
                                                    dataset, typ),
                                                csv=True,
                                                ys=1999, ye=2013,
                                                prcp_fac=2, melt_f=200)

    gpath = '/home/lilianschuster/Schreibtisch/PhD/oggm_files/all/per_glacier/RGI60-11/RGI60-11.00/RGI60-11.00897/'
    extend_past_climate_run_TIModel(past_run_file='/home/lilianschuster/Schreibtisch/PhD/oggm_files/all/run_output_ISIMIP3b_WFDE5_CRU_mri-esm2-0_r1i1p1f1_mb_pseudo_daily_cte_historical_test.nc',
                                fixed_geometry_mb_file='/home/lilianschuster/Schreibtisch/PhD/oggm_files/all/fixed_geometry_mass_balance_{}_{}.csv'.format(dataset, typ),
                                glacier_statistics_file= '/home/lilianschuster/Schreibtisch/PhD/oggm_files/all/glacier_statistics_{}_{}.csv'.format(dataset, typ),
                                path=gpath + 'extended_test.nc',
                                use_compression=True)



#@pytest.mark.usefixtures('get_hef_gcms')
class TestProcessIsimipData:

    def test_process_isimip_data_monthly_w5e5(self, gdir):
        cfg.PARAMS['hydro_month_nh'] = 1
        ssp ='ssp126'

        process_w5e5_data(gdir, temporal_resol='monthly',
                           climate_type='W5E5')

        process_isimip_data(gdir, ensemble=ensemble, ssp=ssp,
                            climate_historical_filesuffix='_monthly_W5E5')
        process_isimip_data(gdir, ensemble=ensemble, ssp=ssp, correct=False,
                            climate_historical_filesuffix='_monthly_W5E5')

        fh = gdir.get_filepath('climate_historical',
                               filesuffix='_monthly_W5E5')
        fgcm = gdir.get_filepath('gcm_data',
                                 filesuffix='_monthly_ISIMIP3b_{}_{}'.format(ensemble, ssp))

        fgcm_nc = gdir.get_filepath('gcm_data',
                                 filesuffix='_monthly_ISIMIP3b_{}_{}_no_correction'.format(ensemble, ssp))

        with xr.open_dataset(fh) as clim, xr.open_dataset(fgcm) as gcm, \
                xr.open_dataset(fgcm_nc) as gcm_nc:
            # Let's do some basic checks
            sclim = clim.sel(time=slice('1979', '2014'))
            # print(sclim.temp)
            sgcm = gcm.load().isel(time=((gcm['time.year'] >= 1979) &
                                         (gcm['time.year'] <= 2014)))


            sgcm_nc = gcm_nc.load().isel(time=((gcm_nc['time.year'] >= 1979) &
                                         (gcm_nc['time.year'] <= 2014)))
            #sgcm_nc = gcm_nc.load().isel(time=((gcm['time.year'] >= 1979) &
            #                                   (gcm['time.year'] <= 2014)))

            # print(sclim.temp.mean(), sgcm.temp.mean(), sgcm_nc.temp.mean())
            # first check if the same grid point was chosen and the same ref_hgt:
            np.testing.assert_allclose(sgcm.ref_hgt, sgcm_nc.ref_hgt)
            np.testing.assert_allclose(sgcm_nc.ref_hgt, sclim.ref_hgt)

            np.testing.assert_allclose(sgcm.ref_pix_lon, sgcm_nc.ref_pix_lon)
            np.testing.assert_allclose(sgcm_nc.ref_pix_lon, sclim.ref_pix_lon)

            np.testing.assert_allclose(sgcm.ref_pix_lat, sgcm_nc.ref_pix_lat)
            np.testing.assert_allclose(sgcm_nc.ref_pix_lat, sclim.ref_pix_lat)


            # Climate during the chosen period should be the same when corrected
            np.testing.assert_allclose(sclim.temp.mean(),
                                       sgcm.temp.mean(),
                                       rtol=1e-3)
            np.testing.assert_allclose(sclim.temp_std.mean(),
                                       sgcm.temp_std.mean(),
                                       rtol=1e-3)
            np.testing.assert_allclose(sclim.prcp.mean(),
                                       sgcm.prcp.mean(),
                                       rtol=1e-3)
            # even if not corrected the climate should be quite similar because ISIMIP3b was internally
            # bias corrected to match W5E5
            np.testing.assert_allclose(sclim.temp.mean(),
                                       sgcm_nc.temp.mean(),
                                       rtol=2e-2)
            np.testing.assert_allclose(sclim.temp_std.mean(),
                                       sgcm_nc.temp_std.mean(),
                                       rtol=5e-2)
            np.testing.assert_allclose(sclim.prcp.mean(),
                                       sgcm_nc.prcp.mean(),
                                       rtol=2e-2)


            # Here no std dev!
            # do not look at the lapse rate gradient here, because this is set constant
            # for gcms (for clim it varies, but when using 'var_an_cycle', only the mean
            # annual lapse rate cycle is applied anyway
            _sclim = sclim.groupby('time.month').std(dim='time')
            _sgcm = sgcm.groupby('time.month').std(dim='time')
            _sgcm_nc = sgcm_nc.groupby('time.month').std(dim='time')
            # need higher tolerance here:
            np.testing.assert_allclose(_sclim.temp, _sgcm.temp, rtol=0.08)  # 1e-3
            np.testing.assert_allclose(_sclim.temp_std, _sgcm.temp_std, rtol=0.01)
            # even higher for non-OGGM bias ccrrection
            np.testing.assert_allclose(_sclim.temp, _sgcm_nc.temp, rtol=0.3)  # 1e-3
            np.testing.assert_allclose(_sclim.temp_std, _sgcm_nc.temp_std, rtol=0.4)
            # not done for precipitation!

            # check the gradient, this is independent of the correction
            np.testing.assert_allclose(sclim.gradient.groupby('time.month').mean(),
                                       sgcm.gradient.groupby('time.month').mean(),
                                       rtol=1e-5)
            np.testing.assert_allclose(sgcm.gradient.groupby('time.month').mean(),
                                       sgcm_nc.gradient.groupby('time.month').mean(),
                                       rtol=1e-5)
            np.testing.assert_allclose(sgcm.gradient.groupby('time.month').std(), 0, atol=1e-6)
            np.testing.assert_allclose(sgcm_nc.gradient.groupby('time.month').std(), 0, atol=1e-6)



            # And also the annual cycle
            sclim = sclim.groupby('time.month').mean(dim='time')
            sgcm = sgcm.groupby('time.month').mean(dim='time')
            sgcm_nc = sgcm_nc.groupby('time.month').mean(dim='time')

            np.testing.assert_allclose(sclim.temp, sgcm.temp, rtol=1e-3)
            np.testing.assert_allclose(sclim.temp_std, sgcm.temp_std, rtol=1e-3)
            np.testing.assert_allclose(sclim.prcp, sgcm.prcp, rtol=1e-3)

            # same for non corrected stuff
            np.testing.assert_allclose(sclim.temp, sgcm_nc.temp, rtol=8e-2)
            np.testing.assert_allclose(sclim.temp_std, sgcm_nc.temp_std, rtol=2e-1)
            np.testing.assert_allclose(sclim.prcp, sgcm_nc.prcp, rtol=2e-1)

            # How did the annual cycle change with time?
            sgcm1 = gcm.load().isel(time=((gcm['time.year'] >= 1979) &
                                          (gcm['time.year'] <= 2019)))
            sgcm2 = gcm.isel(time=((gcm['time.year'] >= 2060) &
                                   (gcm['time.year'] <= 2100)))
            sgcm1_nc = gcm_nc.load().isel(time=((gcm_nc['time.year'] >= 1979) &
                                          (gcm_nc['time.year'] <= 2019)))
            sgcm2_nc = gcm_nc.isel(time=((gcm_nc['time.year'] >= 2060) &
                                   (gcm_nc['time.year'] <= 2100)))

            _sgcm1_std = sgcm1.groupby('time.month').mean(dim='time').std()
            _sgcm2_std = sgcm2.groupby('time.month').mean(dim='time').std()

            _sgcm1_nc_std = sgcm1_nc.groupby('time.month').mean(dim='time').std()
            _sgcm2_nc_std = sgcm2_nc.groupby('time.month').mean(dim='time').std()
            # the mean standard deviation over the year between the months
            # should be different for the time periods
            assert not np.allclose(_sgcm1_std.temp, _sgcm2_std.temp, rtol=1e-2)
            assert not np.allclose(_sgcm1_nc_std.temp, _sgcm2_nc_std.temp, rtol=1e-2)
            # but should be similar between corrected and not corrected
            np.testing.assert_allclose(_sgcm1_std.temp, _sgcm1_nc_std.temp, rtol=1e-2)
            np.testing.assert_allclose(_sgcm2_std.temp, _sgcm2_nc_std.temp, rtol=1e-2)


            sgcm1 = sgcm1.groupby('time.month').mean(dim='time')
            sgcm2 = sgcm2.groupby('time.month').mean(dim='time')
            sgcm1_nc = sgcm1_nc.groupby('time.month').mean(dim='time')
            sgcm2_nc = sgcm2_nc.groupby('time.month').mean(dim='time')
            # It has warmed at least 1 degree for each scenario???
            assert sgcm1.temp.mean() < (sgcm2.temp.mean() - 1)
            assert sgcm1_nc.temp.mean() < (sgcm2_nc.temp.mean() - 1)
            # No prcp change more than 30%? (silly test)
            np.testing.assert_allclose(sgcm1.prcp, sgcm2.prcp, rtol=0.3)
            np.testing.assert_allclose(sgcm1_nc.prcp, sgcm2_nc.prcp, rtol=0.3)

            # mean temperature similar between OGGM and ISIMIP corrected
            np.testing.assert_allclose(sgcm1.temp.mean(),
                                       sgcm1_nc.temp.mean(), rtol=0.05)
            np.testing.assert_allclose(sgcm2.temp.mean(),
                                       sgcm2_nc.temp.mean(), rtol=0.05)

            # mean prcp similar between OGGM and ISIMIP corrected
            np.testing.assert_allclose(sgcm1.prcp.mean(),
                                       sgcm1_nc.prcp.mean(), rtol=0.05)
            np.testing.assert_allclose(sgcm2.prcp.mean(),
                                       sgcm2_nc.prcp.mean(), rtol=0.05)


            # Check that temp still correlate a lot between non corrected
            # and corrected gcm:
            n = 36 *12 +1
            ss1 = gcm.temp.rolling(time=n, min_periods=1, center=True).std()
            ss2 = gcm_nc.temp.rolling(time=n, min_periods=1, center=True).std()
            assert utils.corrcoef(ss1, ss2) > 0.99

    def test_process_isimip_data_daily_w5e5(self, gdir):
        ssp = 'ssp126'
        cfg.PARAMS['use_multiprocessing'] = False

        cfg.PARAMS['hydro_month_nh'] = 1

        process_w5e5_data(gdir, temporal_resol='daily',
                          climate_type='W5E5'
                          )
        process_isimip_data(gdir, ensemble=ensemble, ssp=ssp,
                            temporal_resol='daily',
                            climate_historical_filesuffix='_daily_W5E5')
        process_isimip_data(gdir, ensemble=ensemble, ssp=ssp,
                            temporal_resol='daily',
                            climate_historical_filesuffix='_daily_W5E5', correct=False)
        # process_isimip_data(gdir, ensemble=ensemble, ssp=ssp,
        #                    temporal_resol='daily', correct=False,
        #                    climate_historical_filesuffix='_daily_WFDE5_CRU')

        fh_d = gdir.get_filepath('climate_historical',
                                 filesuffix='_daily_W5E5')
        fgcm_d = gdir.get_filepath('gcm_data',
                                   filesuffix='_daily_ISIMIP3b_{}_{}'.format(ensemble, ssp))
        fgcm_nc_d = gdir.get_filepath('gcm_data',
                                   filesuffix='_daily_ISIMIP3b_{}_{}_no_correction'.format(ensemble, ssp))

        with xr.open_dataset(fh_d) as clim, xr.open_dataset(fgcm_d) as gcm, \
                xr.open_dataset(fgcm_nc_d) as gcm_nc:
            # Let's do some basic checks
            sclim = clim.sel(time=slice('1979', '2014'))
            # print(sclim.temp)
            sgcm = gcm.load().isel(time=((gcm['time.year'] >= 1979) &
                                         (gcm['time.year'] <= 2014)))
            sgcm_nc = gcm.load().isel(time=((gcm_nc['time.year'] >= 1979) &
                                            (gcm_nc['time.year'] <= 2014)))

            # first check if the same grid point was chosen and the same ref_hgt:
            np.testing.assert_allclose(sgcm.ref_hgt, sgcm_nc.ref_hgt)
            np.testing.assert_allclose(sgcm_nc.ref_hgt, sclim.ref_hgt)

            np.testing.assert_allclose(sgcm.ref_pix_lon, sgcm_nc.ref_pix_lon)
            np.testing.assert_allclose(sgcm_nc.ref_pix_lon, sclim.ref_pix_lon)

            np.testing.assert_allclose(sgcm.ref_pix_lat, sgcm_nc.ref_pix_lat)
            np.testing.assert_allclose(sgcm_nc.ref_pix_lat, sclim.ref_pix_lat)

            # Climate during the chosen period should be the same
            # print(sclim.temp.mean(), sgcm.temp.mean(), sgcm_nc.temp.mean())
            np.testing.assert_allclose(sclim.temp.mean(),
                                       sgcm.temp.mean(),
                                       rtol=1e-3)
            np.testing.assert_allclose(sclim.prcp.mean(),
                                       sgcm.prcp.mean(),
                                       rtol=1e-3)
            # even if not corrected the climate should be quite similar because ISIMIP3b was internally
            # bias corrected to match W5E5
            np.testing.assert_allclose(sclim.temp.mean(),
                                       sgcm_nc.temp.mean(),
                                       rtol=2e-2)
            np.testing.assert_allclose(sclim.prcp.mean(),
                                       sgcm_nc.prcp.mean(),
                                       rtol=2e-2)

            # Here no std dev of temperatuer
            # we also do not look at the lapse rate gradient here, because this is set constant
            # for gcms (for clim it varies, but when using 'var_an_cycle', only the mean
            # annual lapse rate cycle is applied anyway
            _sclim = sclim.groupby('time.dayofyear').std(dim='time')
            _sgcm = sgcm.groupby('time.dayofyear').std(dim='time')
            _sgcm_nc = sgcm_nc.groupby('time.dayofyear').std(dim='time')
            # need a higher tolerance here! for cluster 0.12
            np.testing.assert_allclose(_sclim.temp, _sgcm.temp, rtol=0.12)
            # even higher for non-OGGM bias ccrrection
            np.testing.assert_allclose(_sclim.temp, _sgcm_nc.temp, rtol=0.3)  # 1e-3
            # not done for precipitation!

            # gradient stuff
            np.testing.assert_allclose(
                sclim.gradient.groupby('time.dayofyear').mean(),
                sgcm.gradient.groupby('time.dayofyear').mean(), rtol=1e-5)
            np.testing.assert_allclose(
                sclim.gradient.groupby('time.dayofyear').mean(),
                sgcm_nc.gradient.groupby('time.dayofyear').mean(), rtol=1e-5)

            np.testing.assert_allclose(
                sgcm.gradient.groupby('time.dayofyear').std(), 0, atol=1e-6)
            np.testing.assert_allclose(
                sgcm_nc.gradient.groupby('time.dayofyear').std(), 0, atol=1e-6)

            # And also the annual cycle
            sclim = sclim.groupby('time.dayofyear').mean(dim='time')
            sgcm = sgcm.groupby('time.dayofyear').mean(dim='time')
            sgcm_nc = sgcm_nc.groupby('time.dayofyear').mean(dim='time')

            np.testing.assert_allclose(sclim.temp, sgcm.temp, rtol=2e-3)
            np.testing.assert_allclose(sclim.prcp, sgcm.prcp, rtol=2e-3)

            # same for non corrected stuff
            np.testing.assert_allclose(sclim.temp, sgcm_nc.temp, rtol=8e-2)
            np.testing.assert_allclose(sclim.prcp, sgcm_nc.prcp, rtol=2e-1)

            # How did the annual cycle change with time?
            sgcm1 = gcm.load().isel(time=((gcm['time.year'] >= 1979) &
                                          (gcm['time.year'] <= 2018)))
            sgcm2 = gcm.isel(time=((gcm['time.year'] >= 2060) &
                                   (gcm['time.year'] <= 2100)))
            sgcm1_nc = gcm_nc.load().isel(time=((gcm_nc['time.year'] >= 1979) &
                                                (gcm_nc['time.year'] <= 2019)))
            sgcm2_nc = gcm_nc.isel(time=((gcm_nc['time.year'] >= 2060) &
                                         (gcm_nc['time.year'] <= 2100)))

            _sgcm1_std = sgcm1.groupby('time.dayofyear').mean(dim='time').std()
            _sgcm2_std = sgcm2.groupby('time.dayofyear').mean(dim='time').std()
            _sgcm1_nc_std = sgcm1_nc.groupby('time.dayofyear').mean(dim='time').std()
            _sgcm2_nc_std = sgcm2_nc.groupby('time.dayofyear').mean(dim='time').std()
            # the mean standard deviation over the year between the months
            # should be different for the time periods
            assert not np.allclose(_sgcm1_std.temp, _sgcm2_std.temp, rtol=1e-2)
            assert not np.allclose(_sgcm1_nc_std.temp, _sgcm2_nc_std.temp, rtol=1e-2)

            # but should be similar between corrected and not corrected
            np.testing.assert_allclose(_sgcm1_std.temp, _sgcm1_nc_std.temp, rtol=3e-2)
            np.testing.assert_allclose(_sgcm2_std.temp, _sgcm2_nc_std.temp, rtol=3e-2)

            sgcm1 = sgcm1.groupby('time.dayofyear').mean(dim='time')
            sgcm2 = sgcm2.groupby('time.dayofyear').mean(dim='time')
            sgcm1_nc = sgcm1_nc.groupby('time.month').mean(dim='time')
            sgcm2_nc = sgcm2_nc.groupby('time.month').mean(dim='time')
            # It has warmed at least 1 degree for each scenario???
            assert sgcm1.temp.mean() < (sgcm2.temp.mean() - 1)
            assert sgcm1_nc.temp.mean() < (sgcm2_nc.temp.mean() - 1)

            # mean temperature similar between OGGM and ISIMIP corrected
            np.testing.assert_allclose(sgcm1.temp.mean(),
                                       sgcm1_nc.temp.mean(), rtol=0.05)
            np.testing.assert_allclose(sgcm2.temp.mean(),
                                       sgcm2_nc.temp.mean(), rtol=0.05)

            # mean prcp similar between OGGM and ISIMIP corrected
            np.testing.assert_allclose(sgcm1.prcp.mean(),
                                       sgcm1_nc.prcp.mean(), rtol=0.1)
            np.testing.assert_allclose(sgcm2.prcp.mean(),
                                       sgcm2_nc.prcp.mean(), rtol=0.1)

    @pytest.mark.skip(reason="daily prcp bias correction with OGGM still does not work,"
                             " we just use the internal ISIMIP bias correction instead!!!")
    def test_process_isimip_data_prcp_bias_correction_issue(self, gdir):
        cfg.PARAMS['hydro_month_nh'] = 1
        ssp = 'ssp126'

        base_url = ('https://cluster.klima.uni-bremen.de/~oggm/gdirs/oggm_v1.4/'
                    'L1-L2_files/elev_bands')
        df = ['RGI60-01.00570']
        #gdirs = workflow.init_glacier_directories(df, from_prepro_level=2,
        #                                          prepro_border=10,
        #                                          prepro_base_url=base_url,
        #                                          prepro_rgi_version='62')
        gdir_strange = gdir #s[0]


        process_w5e5_data(gdir_strange, temporal_resol='monthly',
                          climate_type='W5E5')

        process_isimip_data(gdir_strange, ensemble=ensemble, ssp=ssp,
                            climate_historical_filesuffix='_monthly_W5E5')
        process_isimip_data(gdir_strange, ensemble=ensemble, ssp=ssp, correct=False,
                            climate_historical_filesuffix='_monthly_W5E5')
        process_w5e5_data(gdir_strange, temporal_resol='daily',
                          climate_type='W5E5'
                          )
        process_isimip_data(gdir_strange, ensemble=ensemble, ssp=ssp,
                            temporal_resol='daily',
                            climate_historical_filesuffix='_daily_W5E5')
        process_isimip_data(gdir_strange, ensemble=ensemble, ssp=ssp,
                            temporal_resol='daily',
                            climate_historical_filesuffix='_daily_W5E5', correct=False)


        #fh = gdir_strange.get_filepath('climate_historical',
        #                       filesuffix='_monthly_W5E5')
        fgcm = gdir_strange.get_filepath('gcm_data',
                                 filesuffix='_monthly_ISIMIP3b_{}_{}'.format(ensemble, ssp))

        fgcm_nc = gdir_strange.get_filepath('gcm_data',
                                    filesuffix='_monthly_ISIMIP3b_{}_{}_no_correction'.format(ensemble, ssp))

        #fh_d = gdir_strange.get_filepath('climate_historical',
        #                         filesuffix='_daily_W5E5')
        fgcm_d = gdir_strange.get_filepath('gcm_data',
                                   filesuffix='_daily_ISIMIP3b_{}_{}'.format(ensemble, ssp))
        fgcm_nc_d = gdir_strange.get_filepath('gcm_data',
                                      filesuffix='_daily_ISIMIP3b_{}_{}_no_correction'.format(ensemble, ssp))

        with xr.open_dataset(fgcm) as gcm, xr.open_dataset(fgcm_d) as gcm_d,\
              xr.open_dataset(fgcm_nc) as gcm_nc, xr.open_dataset(fgcm_nc_d) as gcm_nc_d:

            # daily and monthly should be similar
            # first the non-OGGM corrected stuff
            # temperature
            yr_mean_nc_temp = gcm_nc.sel(time=slice('2015-01-01', '2100-12-01')).load().temp.groupby('time.year').mean()
            yr_mean_nc_d_temp = gcm_nc_d.sel(time=slice('2015-01-01', '2100-12-01')).load().temp.groupby('time.year').mean()
            np.testing.assert_allclose(yr_mean_nc_temp, yr_mean_nc_d_temp, rtol=1e-2)
            # precipitation
            yr_mean_nc_prcp = gcm_nc.sel(time=slice('2015-01-01', '2100-12-01')).load().prcp.groupby('time.year').mean()
            yr_mean_nc_d_prcp = gcm_nc_d.sel(time=slice('2015-01-01', '2100-12-01')).load().prcp.groupby('time.year').mean()
            np.testing.assert_allclose(yr_mean_nc_prcp, yr_mean_nc_d_prcp, rtol=1e-2)

            # now the OGGM bias-corrected stuff
            # temperature
            yr_mean_temp = gcm.sel(time=slice('2015-01-01', '2100-12-01')).load().temp.groupby('time.year').mean()
            yr_mean_d_temp = gcm_d.sel(time=slice('2015-01-01', '2100-12-01')).load().temp.groupby('time.year').mean()
            np.testing.assert_allclose(yr_mean_temp, yr_mean_d_temp, rtol=1e-2)
            # precipitation
            yr_mean_prcp = gcm.sel(time=slice('2015-01-01', '2100-12-01')).load().prcp.groupby('time.year').mean()
            yr_mean_d_prcp = gcm_d.sel(time=slice('2015-01-01', '2100-12-01')).load().prcp.groupby('time.year').mean()
            np.testing.assert_allclose(yr_mean_prcp, yr_mean_d_prcp, rtol=1e-2)

            # historical stuff should all be around the same
            yr_mean_nc_h_temp = gcm_nc.sel(time=slice('1979-01-01', '2014-12-01')).load().temp.groupby('time.year').mean()
            yr_mean_nc_d_h_temp = gcm_nc_d.sel(time=slice('1979-01-01', '2014-12-01')).load().temp.groupby('time.year').mean()
            yr_mean_h_temp = gcm.sel(time=slice('1979-01-01', '2014-12-01')).load().temp.groupby('time.year').mean()
            yr_mean_d_h_temp = gcm_d.sel(time=slice('1979-01-01', '2014-12-01')).load().temp.groupby('time.year').mean()

            np.testing.assert_allclose(yr_mean_nc_h_temp.mean(), yr_mean_nc_d_h_temp.mean())

    @pytest.mark.skip(reason="no one uses wfde5_cru ath the moment")
    def test_process_isimip_data_monthly_wfde5_cru(self, gdir):
        cfg.PARAMS['hydro_month_nh'] = 1
        ssp ='ssp126'
        ensemble = 'mri-esm2-0_r1i1p1f1'

        process_w5e5_data(gdir, temporal_resol='monthly',
                           climate_type='WFDE5_CRU')

        process_isimip_data(gdir, ensemble=ensemble, ssp=ssp,
                            climate_historical_filesuffix='_monthly_WFDE5_CRU')
        process_isimip_data(gdir, ensemble=ensemble, ssp=ssp, correct=False,
                            climate_historical_filesuffix='_monthly_WFDE5_CRU')

        fh = gdir.get_filepath('climate_historical',
                               filesuffix='_monthly_WFDE5_CRU')
        fgcm = gdir.get_filepath('gcm_data',
                                 filesuffix='_monthly_ISIMIP3b_{}_{}'.format(ensemble, ssp))

        fgcm_nc = gdir.get_filepath('gcm_data',
                                 filesuffix='_monthly_ISIMIP3b_{}_{}_no_correction'.format(ensemble, ssp))

        with xr.open_dataset(fh) as clim, xr.open_dataset(fgcm) as gcm, \
                xr.open_dataset(fgcm_nc) as gcm_nc:
            # Let's do some basic checks
            sclim = clim.sel(time=slice('1979', '2014'))
            # print(sclim.temp)
            sgcm = gcm.load().isel(time=((gcm['time.year'] >= 1979) &
                                         (gcm['time.year'] <= 2014)))
            #sgcm_nc = gcm_nc.load().isel(time=((gcm['time.year'] >= 1979) &
            #                                   (gcm['time.year'] <= 2014)))

            # Climate during the chosen period should be the same
            # print(sclim.temp.mean(), sgcm.temp.mean(), sgcm_nc.temp.mean())
            np.testing.assert_allclose(sclim.temp.mean(),
                                       sgcm.temp.mean(),
                                       rtol=1e-3)
            np.testing.assert_allclose(sclim.temp_std.mean(),
                                       sgcm.temp_std.mean(),
                                       rtol=1e-3)
            np.testing.assert_allclose(sclim.prcp.mean(),
                                       sgcm.prcp.mean(),
                                       rtol=1e-3)

            # Here no std dev!
            # do not look at the lapse rate gradient here, because this is set constant
            # for gcms (for clim it varies, but when using 'var_an_cycle', only the mean
            # annual lapse rate cycle is applied anyway
            _sclim = sclim.groupby('time.month').std(dim='time')
            _sgcm = sgcm.groupby('time.month').std(dim='time')
            # need higher tolerance here:
            np.testing.assert_allclose(_sclim.temp, _sgcm.temp, rtol=0.08)  # 1e-3
            np.testing.assert_allclose(_sclim.temp_std, _sgcm.temp_std, rtol=0.01)
            # not done for precipitation!

            # check the gradient
            np.testing.assert_allclose(sclim.gradient.groupby('time.month').mean(),
                                       sgcm.gradient.groupby('time.month').mean(), rtol=1e-5)
            np.testing.assert_allclose(sgcm.gradient.groupby('time.month').std(), 0, atol=1e-6)


            # And also the annual cycle
            sclim = sclim.groupby('time.month').mean(dim='time')
            sgcm = sgcm.groupby('time.month').mean(dim='time')
            np.testing.assert_allclose(sclim.temp, sgcm.temp, rtol=1e-3)
            np.testing.assert_allclose(sclim.temp_std, sgcm.temp_std, rtol=1e-3)
            np.testing.assert_allclose(sclim.prcp, sgcm.prcp, rtol=1e-3)

            # How did the annual cycle change with time?
            sgcm1 = gcm.load().isel(time=((gcm['time.year'] >= 1979) &
                                          (gcm['time.year'] <= 2018)))
            sgcm2 = gcm.isel(time=((gcm['time.year'] >= 2060) &
                                   (gcm['time.year'] <= 2100)))

            _sgcm1_std = sgcm1.groupby('time.month').mean(dim='time').std()
            _sgcm2_std = sgcm2.groupby('time.month').mean(dim='time').std()
            # the mean standard deviation over the year between the months
            # should be different for the time periods
            assert not np.allclose(_sgcm1_std.temp, _sgcm2_std.temp, rtol=1e-2)

            sgcm1 = sgcm1.groupby('time.month').mean(dim='time')
            sgcm2 = sgcm2.groupby('time.month').mean(dim='time')
            # It has warmed at least 1 degree for each scenario???
            assert sgcm1.temp.mean() < (sgcm2.temp.mean() - 1)
            # N more than 30%? (silly test)
            np.testing.assert_allclose(sgcm1.prcp, sgcm2.prcp, rtol=0.3)

            # Check that temp still correlate a lot between non corrected
            # and corrected gcm:
            n = 36 *12 +1
            ss1 = gcm.temp.rolling(time=n, min_periods=1, center=True).std()
            ss2 = gcm_nc.temp.rolling(time=n, min_periods=1, center=True).std()
            assert utils.corrcoef(ss1, ss2) > 0.9

    @pytest.mark.skip(reason="no one uses wfde5_cru ath the moment")
    def test_process_isimip_data_daily_wfde5_cru(self, gdir):
        ssp ='ssp126'

        cfg.PARAMS['hydro_month_nh'] = 1

        process_w5e5_data(gdir, temporal_resol='daily',
                           climate_type='WFDE5_CRU'
                           )
        process_isimip_data(gdir, ensemble=ensemble, ssp=ssp,
                            temporal_resol='daily',
                            climate_historical_filesuffix='_daily_WFDE5_CRU')
        #process_isimip_data(gdir, ensemble=ensemble, ssp=ssp,
        #                    temporal_resol='daily', correct=False,
        #                    climate_historical_filesuffix='_daily_WFDE5_CRU')


        fh_d = gdir.get_filepath('climate_historical',
                               filesuffix='_daily_WFDE5_CRU')
        fgcm_d = gdir.get_filepath('gcm_data',
                                 filesuffix='_daily_ISIMIP3b_{}_{}'.format(ensemble, ssp))

        with xr.open_dataset(fh_d) as clim, xr.open_dataset(fgcm_d) as gcm:
            # Let's do some basic checks
            sclim = clim.sel(time=slice('1979', '2014'))
            # print(sclim.temp)
            sgcm = gcm.load().isel(time=((gcm['time.year'] >= 1979) &
                                         (gcm['time.year'] <= 2014)))

            # Climate during the chosen period should be the same
            # print(sclim.temp.mean(), sgcm.temp.mean(), sgcm_nc.temp.mean())
            np.testing.assert_allclose(sclim.temp.mean(),
                                       sgcm.temp.mean(),
                                       rtol=1e-3)
            np.testing.assert_allclose(sclim.prcp.mean(),
                                       sgcm.prcp.mean(),
                                       rtol=1e-3)

            # Here no std dev of temperatuer
            # we also do not look at the lapse rate gradient here, because this is set constant
            # for gcms (for clim it varies, but when using 'var_an_cycle', only the mean
            # annual lapse rate cycle is applied anyway
            _sclim = sclim.groupby('time.dayofyear').std(dim='time')
            _sgcm = sgcm.groupby('time.dayofyear').std(dim='time')
            # need a higher tolerance here! for cluster 0.12
            np.testing.assert_allclose(_sclim.temp, _sgcm.temp, rtol=0.12)
            # not done for precipitation!

            # gradient stuff
            np.testing.assert_allclose(
                sclim.gradient.groupby('time.dayofyear').mean(),
                sgcm.gradient.groupby('time.dayofyear').mean(), rtol=1e-5)

            np.testing.assert_allclose(
                sgcm.gradient.groupby('time.dayofyear').std(), 0, atol=1e-6)

            # And also the annual cycle
            sclim = sclim.groupby('time.dayofyear').mean(dim='time')
            sgcm = sgcm.groupby('time.dayofyear').mean(dim='time')
            np.testing.assert_allclose(sclim.temp, sgcm.temp, rtol=2e-3)
            np.testing.assert_allclose(sclim.prcp, sgcm.prcp, rtol=2e-3)

            # How did the annual cycle change with time?
            sgcm1 = gcm.load().isel(time=((gcm['time.year'] >= 1979) &
                                          (gcm['time.year'] <= 2018)))
            sgcm2 = gcm.isel(time=((gcm['time.year'] >= 2060) &
                                   (gcm['time.year'] <= 2100)))

            _sgcm1_std = sgcm1.groupby('time.dayofyear').mean(dim='time').std()
            _sgcm2_std = sgcm2.groupby('time.dayofyear').mean(dim='time').std()
            # the mean standard deviation over the year between the months
            # should be different for the time periods
            assert not np.allclose(_sgcm1_std.temp, _sgcm2_std.temp, rtol=1e-2)

            sgcm1 = sgcm1.groupby('time.dayofyear').mean(dim='time')
            sgcm2 = sgcm2.groupby('time.dayofyear').mean(dim='time')
            # It has warmed at least 1 degree for each scenario???
            assert sgcm1.temp.mean() < (sgcm2.temp.mean() - 1)

