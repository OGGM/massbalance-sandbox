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



def test_ssp585_problem():
    from oggm import cfg, utils, workflow, tasks, graphics
    from oggm.core import massbalance, flowline, climate
    import logging
    log = logging.getLogger(__name__)
    cfg.initialize()
    cfg.PATHS['working_dir'] = utils.gettempdir(dirname='test', reset=False)
    gdirs = workflow.init_glacier_directories(['RGI60-11.03238'], #'RGI60-11.03677'], #'RGI60-11.03471'],
                                              from_prepro_level=2,
                                              prepro_border=10,
                                              prepro_base_url=base_url,
                                              prepro_rgi_version='62',
                                              )
    gdir = gdirs[0]
    cfg.PARAMS['hydro_month_nh'] = 1
    ssp ='ssp585'

    process_w5e5_data(gdir, temporal_resol='monthly',
                       climate_type='WFDE5_CRU'
                       )
    process_isimip_data(gdir, ensemble=ensemble, ssp=ssp,
                        climate_historical_filesuffix='_monthly_WFDE5_CRU',
                        temporal_resol='monthly'
                        )


#@pytest.mark.usefixtures('get_hef_gcms')
@pytest.mark.skip(reason="this won't work on other computers")
class TestProcessIsimipData:

    def test_process_isimip_data_monthly(self, gdir):
        cfg.PARAMS['hydro_month_nh'] = 1
        ssp ='ssp126'

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


    def test_process_isimip_data_daily(self, gdir):
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
            # N more than 30%? (silly test)
            # this does NOT work for daily values
            # np.testing.assert_allclose(sgcm1.prcp, sgcm2.prcp, rtol=0.3)
