#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 18:34:35 2020

@author: lilianschuster
"""
import warnings
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from numpy.testing import assert_allclose
import oggm

# imports from oggm
from oggm.exceptions import InvalidParamsError
from oggm.shop.ecmwf import get_ecmwf_file
from oggm import tasks, cfg
# imports from MBsandbox package modules
from MBsandbox.mbmod_daily_oneflowline import (process_era5_daily_data,
                                               process_w5e5_data, get_w5e5_file)

warnings.filterwarnings("once", category=DeprecationWarning)
# %%

class Test_climate_daily_datasets:
    @pytest.mark.no_w5e5
    def test_ERA5_daily_dataset(self):

        dataset = 'ERA5_daily'

        ds_ERA5_daily = xr.open_dataset(get_ecmwf_file(dataset, 'tmp'))
        ds_ERA5_daily['time.month'][0] == 1
        ds_ERA5_daily['time.month'][-1] == 12

        # checks if it is in Kelvin
        assert np.all(ds_ERA5_daily.t2m > 0)
        # not too high temperatures
        assert np.max(ds_ERA5_daily.t2m) < 350

        # ERA5 daily should start in 1979 and end in 2018
        assert ds_ERA5_daily['time.year'][0] == 1979
        assert ds_ERA5_daily['time.year'][-1] == 2018
        assert ds_ERA5_daily['time.month'][0] == 1
        assert ds_ERA5_daily['time.month'][-1] == 12

        # compare the daily dataset to the monthly:
        ds_ERA5 = xr.open_dataset(get_ecmwf_file('ERA5', 'tmp'))

        # check if for Hintereisferner, both datasets produce similar monthly
        # temperature time series if ERA5_daily is resampled over month
        lon = 10.7584
        lat = 46.8003

        # compute all the distances and choose nearest gridpoint
        # this also checks if the flattened version is used!
        c = (ds_ERA5_daily.longitude - lon)**2 + (ds_ERA5_daily.latitude - lat)**2
        ds_ERA5_daily_g = ds_ERA5_daily.isel(points=c.argmin())

        ds_ERA5_g = ds_ERA5.sel(time=slice('1979-01-01',
                                           '2018-12-01')).sel(longitude=lon,
                                                              latitude=lat,
                                                              method='nearest')
        # do we use the same longitude/latitudes
        assert ds_ERA5_daily_g.longitude.values == ds_ERA5_g.longitude.values
        assert ds_ERA5_daily_g.latitude.values == ds_ERA5_g.latitude.values

        # do the two datasets have the same monthly temperatures?
        # at the neares gridpoint to HEF (some rounding errors are allowed)
        assert_allclose(ds_ERA5_daily_g.resample(time='1M').mean().t2m.values,
                        ds_ERA5_g.t2m.values, rtol=1e-4)

    @pytest.mark.no_w5e5
    def test_WFDE5_W5E5_daily_dataset(self):

        dataset = 'WFDE5_CRU_daily'
        path_tmp = get_w5e5_file(dataset, 'tmp')
        path_prcp = get_w5e5_file(dataset, 'prcp')
        ds_WFDE5_daily_tmp = xr.open_dataset(path_tmp)
        ds_WFDE5_daily_prcp = xr.open_dataset(path_prcp)

        dataset = 'W5E5_daily'
        path_tmp = get_w5e5_file(dataset, 'tmp')
        path_prcp = get_w5e5_file(dataset, 'prcp')
        #pathi = ('/home/lilianschuster/Schreibtisch/PhD/WP0_bayesian/'
        #                 'WPx_WFDE5/wfde5_cru/daily/v1.1/'
        #                 'wfde5_cru_tmp_1979-2018_flat.nc')
        ds_W5E5_daily_tmp = xr.open_dataset(path_tmp)
        ds_W5E5_daily_prcp = xr.open_dataset(path_prcp)

        # xr.open_dataset(get_ecmwf_file(dataset, 'tmp'))
        for ds in [ds_WFDE5_daily_tmp, ds_WFDE5_daily_prcp,
                   ds_W5E5_daily_tmp, ds_W5E5_daily_prcp]:
            assert ds['time.month'][0] == 1
            assert ds['time.month'][-1] == 12
            assert ds['time.year'][0] == 1979

        for ds_W5E5 in [ds_W5E5_daily_tmp, ds_W5E5_daily_prcp]:
            assert ds_W5E5['time.year'][-1] == 2019

        # WFDE5 old version should start in 1979 and end in 2018
        for ds_WFDE5 in [ds_WFDE5_daily_tmp, ds_WFDE5_daily_prcp]:
            assert ds_WFDE5['time.year'][-1] == 2018

        # temperature checks
        # checks if it is in Kelvin
        assert np.all(ds_WFDE5_daily_tmp.Tair > 0)
        assert np.all(ds_W5E5_daily_tmp.tas>0)
        # not too high temperatures
        assert np.max(ds_WFDE5_daily_tmp.Tair) < 350
        assert np.max(ds_W5E5_daily_tmp.tas) < 350

        # prcp checks
        assert np.all(ds_WFDE5_daily_prcp.tp >= 0)
        assert np.all(ds_W5E5_daily_prcp.pr >= 0)
        # they have different prcp units
        # (is accounted for in process_W5E5_data)
        assert ds_W5E5_daily_prcp.pr.units == 'kg m-2 s-1'
        assert ds_W5E5_daily_prcp.pr.max() < 1e-2
        assert ds_WFDE5_daily_prcp.tp.units == 'kg m-2 day-1 ~ mm/day'
        assert ds_WFDE5_daily_prcp.tp.max() > 100

        # compare the daily dataset to the monthly:
        ds_ERA5 = xr.open_dataset(get_ecmwf_file('ERA5', 'tmp'))

        # check if for Hintereisferner, both datasets produce similar monthly
        # temperature time series if ERA5_daily is resampled over month
        lon = 10.7584
        lat = 46.8003

        # compute all the distances and choose nearest gridpoint
        # this also checks if the flattened version is used!
        c_wfde5 = ((ds_WFDE5_daily_tmp.longitude - lon)**2
             + (ds_WFDE5_daily_tmp.latitude - lat)**2)
        ds_WFDE5_daily_tmp_g = ds_WFDE5_daily_tmp.isel(points=c_wfde5.argmin())
        c_w5e5 = ((ds_W5E5_daily_tmp.longitude - lon)**2
             + (ds_W5E5_daily_tmp.latitude - lat)**2)
        ds_W5E5_daily_tmp_g = ds_W5E5_daily_tmp.isel(points=c_w5e5.argmin())
        ds_W5E5_daily_tmp_g = ds_W5E5_daily_tmp_g.sel(time=slice('1979-01-01',
                                           '2018-12-01'))

        ds_ERA5_g = ds_ERA5.sel(time=slice('1979-01-01',
                                           '2018-12-01')).sel(longitude=lon,
                                                              latitude=lat,
                                                              method='nearest')
        # do we use a similar longitude/latitude? (not exactly the same, as ERA5 is
        # finer than WFDE5/W5E5)
        assert_allclose(ds_WFDE5_daily_tmp_g.longitude.values,
                        ds_ERA5_g.longitude.values,
                        atol=0.6)
        assert_allclose(ds_W5E5_daily_tmp_g.longitude.values,
                        ds_ERA5_g.longitude.values,
                        atol=0.6)
        assert_allclose(ds_WFDE5_daily_tmp_g.latitude.values,
                        ds_ERA5_g.latitude.values,
                        atol=0.6)
        assert_allclose(ds_W5E5_daily_tmp_g.latitude.values,
                        ds_ERA5_g.latitude.values,
                        atol=0.6)

        # do the three datasets have similar monthly temperature for HEF
        # at the nearest gridpoint to HEF (wfde5 temp against HEF temp. )
        wfde5_tmp_m = ds_WFDE5_daily_tmp_g.resample(time='MS').mean().Tair.values
        w5e5_tmp_m = ds_W5E5_daily_tmp_g.resample(time='MS').mean().tas.values

        tmp_corr_wfde5 = np.corrcoef(wfde5_tmp_m,
                              ds_ERA5_g.t2m.values)[0][1]
        tmp_corr_w5e5 = np.corrcoef(w5e5_tmp_m,
                               ds_ERA5_g.t2m.values)[0][1]
        assert tmp_corr_wfde5 > 0.95
        assert tmp_corr_w5e5 > 0.95



class Test_process_era5_daily_wfde5_w5e5_hef:

    @pytest.mark.no_w5e5
    def test_process_era5_daily_data(self, gdir):
        process_era5_daily_data(gdir, y0=1979, y1=2018)

        fs = '_daily_ERA5'
        fpath = gdir.get_filepath('climate_historical', filesuffix=fs)

        # check the climate files of an individual glacier (Hintereisferner)
        xr_nc = xr.open_dataset(fpath)
        assert np.all(xr_nc.prcp) >= 0
        # daily precipitation amount in kg m-2  (not in kg m-2 s-1)
        assert xr_nc.prcp.max() > 1
        # to be sure that there are no erroneaous filling values inside
        assert np.all(xr_nc.prcp) < 10000
        # temperature values are in °C and in the right range
        assert np.all(xr_nc.temp) > -100
        assert np.all(xr_nc.temp) < 100

        # temperature gradient should be in the following range
        assert np.all(xr_nc.gradient > -0.015)
        assert np.all(xr_nc.gradient < -0.002)

        # all lapse rates/ precipitation values in one month should be equal
        # because only temperature is on daily basis
        np.testing.assert_allclose(xr_nc.resample(time='1M').std().prcp,
                                   0, atol=1e-3)
        np.testing.assert_allclose(xr_nc.resample(time='1M').std().gradient,
                                   0, atol=1e-3)

        # summed up monthly precipitation from daily dataset
        xr_nc_prcp_m = xr_nc.prcp.resample(time='1M').sum()

        oggm.shop.ecmwf.process_ecmwf_data(gdir, dataset="ERA5",
                                           y0=1979, y1=2018)
        filename = 'climate_historical'

        fpath = gdir.get_filepath(filename)
        xr_nc_monthly = xr.open_dataset(fpath)

        # check if summed up monthly precipitation from daily
        # dataset equals approx. to the ERA5 monthly prcp
        np.testing.assert_allclose(xr_nc_prcp_m.values,
                                   xr_nc_monthly.prcp.values, rtol=1e-4)

        xr_nc_temp_m = xr_nc.temp.resample(time='1M').mean()
        # check if mean temperature from daily dataset equals
        # approx. to the ERA5 monthly temp.
        np.testing.assert_allclose(xr_nc_temp_m.values,
                                   xr_nc_monthly.temp.values, atol=0.05)

        with pytest.raises(InvalidParamsError):
            # dataset only goes from 1979--2018
            process_era5_daily_data(gdir, y0=1979, y1=2019)

            # in cfg.PARAMS that is initiated during testing,
            # cfg.PARAMS[hydro_month_nh = 10], this is in conflict with 8
            process_era5_daily_data(gdir, y0=1979, y1=2018, hydro_month_nh=8)

    @pytest.mark.no_w5e5
    def test_process_w5e5_data(self, gdir):

        # first with daily resolution
        cfg.PARAMS['hydro_month_nh'] = 1
        process_w5e5_data(gdir, y0=1979, y1=2019, temporal_resol='daily',
                          climate_type='W5E5')
        process_w5e5_data(gdir, y0=1979, y1=2018, temporal_resol='daily',
                          climate_type='WFDE5_CRU')


        filename = 'climate_historical'
        fpath = gdir.get_filepath(filename, filesuffix='_daily_WFDE5_CRU')
        # check the climate files of an individual glacier (Hintereisferner)
        xr_nc = xr.open_dataset(fpath)
        fpath_w5e5 = gdir.get_filepath(filename, filesuffix='_daily_W5E5')
        # check the climate files of an individual glacier (Hintereisferner)
        xr_nc_w5e5 = xr.open_dataset(fpath_w5e5)

        assert np.all(xr_nc.prcp) >= 0
        assert np.all(xr_nc_w5e5.prcp) >= 0

        # daily precipitation amount in kg m-2  (not in kg m-2 s-1)
        assert xr_nc.prcp.max() > 1
        assert xr_nc_w5e5.prcp.max() > 1
        # to be sure that there are no erroneous filling values inside
        assert np.all(xr_nc.prcp) < 10000
        assert np.all(xr_nc_w5e5.prcp) < 10000
        # temperature values are in °C and in the right range
        assert np.all(xr_nc.temp) > -100
        assert np.all(xr_nc.temp) < 100
        assert np.all(xr_nc_w5e5.temp) > -100
        assert np.all(xr_nc_w5e5.temp) < 100
        # temperature gradient should be in the following range
        assert np.all(xr_nc.gradient > -0.015)
        assert np.all(xr_nc.gradient < -0.002)
        assert np.all(xr_nc_w5e5.gradient > -0.015)
        assert np.all(xr_nc_w5e5.gradient < -0.002)

        # all lapse rates values in one month should be equal
        # because only temperature and prcp is on daily basis
        #np.testing.assert_allclose(xr_nc.resample(time='1M').std().prcp,
        #                           0, atol=1e-3)
        np.testing.assert_allclose(xr_nc.resample(time='MS').std().gradient,
                                   0, atol=1e-3)
        np.testing.assert_allclose(xr_nc_w5e5.resample(time='MS').std().gradient,
                                   0, atol=1e-3)

        # summed up monthly precipitation from daily dataset
        xr_nc_prcp_m = xr_nc.prcp.resample(time='MS').sum()
        xr_nc_prcp_m_w5e5 = xr_nc_w5e5.prcp.resample(time='MS').sum()


        oggm.shop.ecmwf.process_ecmwf_data(gdir, dataset="ERA5",
                                           y0=1979, y1=2018)
        filename = 'climate_historical'
        fpath = gdir.get_filepath(filename)
        xr_nc_monthly_ERA5 = xr.open_dataset(fpath)

        # check if summed up monthly precipitation from daily
        # dataset equals approx to the WFDE5 monthly prcp
        # first with daily resolution
        process_w5e5_data(gdir, y0=1979, y1=2018, temporal_resol='monthly',
                           climate_type='WFDE5_CRU')
        process_w5e5_data(gdir, y0=1979, y1=2019, temporal_resol='monthly',
                          climate_type='W5E5')
        # an output filesuffix is needed, otherwise the
        # ERA5 dataset from above is used as it has the same name ...
        fpath_monthly = gdir.get_filepath(filename,
                                          filesuffix='_monthly_WFDE5_CRU')
        xr_nc_monthly = xr.open_dataset(fpath_monthly)
        fpath_monthly = gdir.get_filepath(filename,
                                          filesuffix='_monthly_W5E5')
        xr_nc_monthly_W5E5 = xr.open_dataset(fpath_monthly)
        assert np.all(xr_nc_monthly.prcp) > 0
        assert np.all(xr_nc_monthly_W5E5.prcp) > 0
        assert xr_nc_monthly.prcp.max() > 1
        assert xr_nc_monthly_W5E5.prcp.max() > 1
        # to be sure that there are no erroneous filling values inside
        assert np.all(xr_nc_monthly.prcp) < 10000
        assert np.all(xr_nc_monthly_W5E5.prcp) < 10000
        # temperature values are in °C and in the right range
        assert np.all(xr_nc_monthly.temp) > -100
        assert np.all(xr_nc_monthly.temp) < 100
        assert np.all(xr_nc_monthly_W5E5.temp) > -100
        assert np.all(xr_nc_monthly_W5E5.temp) < 100
        # temperature gradient should be in the following range
        assert np.all(xr_nc_monthly.gradient > -0.015)
        assert np.all(xr_nc_monthly.gradient < -0.002)
        assert np.all(xr_nc_monthly_W5E5.gradient > -0.015)
        assert np.all(xr_nc_monthly_W5E5.gradient < -0.002)
        assert_allclose(xr_nc_prcp_m.values,
                        xr_nc_monthly.prcp.values, rtol=1e-4)
        assert_allclose(xr_nc_prcp_m_w5e5.values,
                        xr_nc_monthly_W5E5.prcp.values,
                        rtol=1e-4)
        # check if summed up monthly precipitation from daily
        # dataset correlate for prpc ...
        assert np.corrcoef(xr_nc_prcp_m.values,
                           xr_nc_monthly_ERA5.prcp.values)[0][1] > 0.75
        xr_nc_prcp_m_w5e5 = xr_nc_prcp_m_w5e5.sel(time=slice('1979', '2018'))
        assert np.corrcoef(xr_nc_prcp_m.values,
                           xr_nc_prcp_m_w5e5)[0][1] > 0.75
        #np.testing.assert_allclose(xr_nc_prcp_m.values,
        #                           xr_nc_monthly.prcp.values, rtol=1e-4)

        xr_nc_temp_m = xr_nc.temp.resample(time='MS').mean()
        xr_nc_temp_m_w5e5 = xr_nc_w5e5.temp.resample(time='MS').mean()

        # check if mean temperature from daily dataset equals
        # approx. the WFDE5 monthly temp.
        assert_allclose(xr_nc_temp_m.values,
                        xr_nc_monthly.temp.values, atol=1e-4)
        assert_allclose(xr_nc_temp_m_w5e5.values,
                        xr_nc_monthly_W5E5.temp.values, atol=1e-4)

        # check if mean temperature from daily dataset equals
        # approx. to the ERA5 monthly temp.
        assert np.corrcoef(xr_nc_monthly_ERA5.temp.values,
                           xr_nc_temp_m.values)[0][1] > 0.75
        xr_nc_temp_m = xr_nc_temp_m.sel(time=slice('1979', '2018'))
        assert np.corrcoef(xr_nc_monthly_ERA5.temp.values,
                           xr_nc_temp_m.values)[0][1] > 0.75
        with pytest.raises(InvalidParamsError):
            # dataset only goes from 1979--2018
            process_era5_daily_data(gdir, y0=1979, y1=2019)

            # in cfg.PARAMS that is initiated during testing,
            # cfg.PARAMS[hydro_month_nh = 1], this is in conflict with 8
            process_era5_daily_data(gdir, y0=1979, y1=2018, hydro_month_nh=8)

    @pytest.mark.no_w5e5
    def test_process_mswep_data(self, gdir):

        # first with daily resolution
        cfg.PARAMS['hydro_month_nh'] = 1
        process_w5e5_data(gdir, y0=1979, y1=2019, temporal_resol='monthly',
                          climate_type='W5E5_MSWEP')
        process_w5e5_data(gdir, y0=1979, y1=2019, temporal_resol='daily',
                          climate_type='W5E5_MSWEP')

        filename = 'climate_historical'
        fpath_w5e5_mswep_m = gdir.get_filepath(filename, filesuffix='_monthly_W5E5_MSWEP')
        # check the climate files of an individual glacier (Hintereisferner)
        xr_nc_w5e5_mswep_m = xr.open_dataset(fpath_w5e5_mswep_m)
        fpath_w5e5_mswep = gdir.get_filepath(filename, filesuffix='_daily_W5E5_MSWEP')
        # check the climate files of an individual glacier (Hintereisferner)
        xr_nc_w5e5_mswep_d = xr.open_dataset(fpath_w5e5_mswep)

        assert np.all(xr_nc_w5e5_mswep_m.prcp) >= 0
        assert np.all(xr_nc_w5e5_mswep_d.prcp) >= 0

        # daily precipitation amount in kg m-2  (not in kg m-2 s-1)
        assert xr_nc_w5e5_mswep_m.prcp.max() > 1
        assert xr_nc_w5e5_mswep_d.prcp.max() > 1
        # to be sure that there are no erroneous filling values inside
        assert np.all(xr_nc_w5e5_mswep_m.prcp) < 10000
        assert np.all(xr_nc_w5e5_mswep_d.prcp) < 10000
        # temperature values are in °C and in the right range
        assert np.all(xr_nc_w5e5_mswep_m.temp) > -100
        assert np.all(xr_nc_w5e5_mswep_m.temp) < 100
        assert np.all(xr_nc_w5e5_mswep_d.temp) > -100
        assert np.all(xr_nc_w5e5_mswep_d.temp) < 100
        # temperature gradient should be in the following range
        assert np.all(xr_nc_w5e5_mswep_m.gradient > -0.015)
        assert np.all(xr_nc_w5e5_mswep_m.gradient < -0.002)
        assert np.all(xr_nc_w5e5_mswep_d.gradient > -0.015)
        assert np.all(xr_nc_w5e5_mswep_d.gradient < -0.002)

        # all lapse rates values in one month should be equal
        # because only temperature and prcp is on daily basis
        #np.testing.assert_allclose(xr_nc.resample(time='1M').std().prcp,
        #                           0, atol=1e-3)
        #np.testing.assert_allclose(xr_nc_w5e5_mswep_m.resample(time='MS').std().gradient,
        #                           0, atol=1e-3)
        np.testing.assert_allclose(xr_nc_w5e5_mswep_d.resample(time='MS').std().gradient,
                                   0, atol=1e-3)

        # summed up monthly precipitation from daily dataset
        # xr_nc_w5e5_mswep_m #.prcp.resample(time='MS').sum()
        xr_nc_prcp_m_mswep = xr_nc_w5e5_mswep_d.prcp.resample(time='MS').sum()

        oggm.shop.ecmwf.process_ecmwf_data(gdir, dataset="ERA5",
                                           y0=1979, y1=2018)
        filename = 'climate_historical'
        fpath = gdir.get_filepath(filename)
        xr_nc_monthly_ERA5 = xr.open_dataset(fpath)

        # check if summed up monthly precipitation from daily
        # dataset equals approx to MSWEP data
        # first with daily resolution
        assert_allclose(xr_nc_prcp_m_mswep.values,
                        xr_nc_w5e5_mswep_m.prcp.values, rtol=1e-4)
        # check if summed up monthly precipitation from daily
        # dataset correlate for prpc ... with ERA5
        xr_nc_prcp_m_mswep = xr_nc_prcp_m_mswep.sel(time=slice('1979', '2018'))
        # not so much correlated (also does not use exactly the same gridpoint) !!!
        assert np.corrcoef(xr_nc_prcp_m_mswep.values,
                           xr_nc_monthly_ERA5.prcp.values)[0][1] > 0.45
        #assert np.corrcoef(xr_nc_w5e5_mswep_m.values,
        #                   xr_nc_prcp_m_mswep)[0][1] > 0.75
        #np.testing.assert_allclose(xr_nc_prcp_m.values,
        #                           xr_nc_monthly.prcp.values, rtol=1e-4)

        xr_nc_temp_m_w5e5 = xr_nc_w5e5_mswep_d.temp.resample(time='MS').mean()

        # check if mean temperature from daily dataset equals
        # approx. the WFDE5 monthly temp.
        assert_allclose(xr_nc_temp_m_w5e5.values,
                        xr_nc_w5e5_mswep_m.temp.values,
                        atol=1e-4)
        #assert_allclose(xr_nc_temp_m_w5e5.values,
        #                xr_nc_monthly_W5E5.temp.values, atol=1e-4)

        # check if mean temperature from daily dataset equals
        # approx. to the ERA5 monthly temp.
        #assert np.corrcoef(xr_nc_monthly_ERA5.temp.values,
        #                   xr_nc_temp_m_w5e5.values)[0][1] > 0.75
        xr_nc_temp_m = xr_nc_temp_m_w5e5.sel(time=slice('1979', '2018'))
        assert np.corrcoef(xr_nc_monthly_ERA5.temp.values,
                           xr_nc_temp_m.values)[0][1] > 0.75


    # this could be replaced in OGGM base code test_shop.py when merged
    #@pytest.mark.parametrize('hydro_month', [10, 1])
    @pytest.mark.skip(reason="too slow")
    def test_all_at_once(self, gdir):
        # Init
        cfg.PARAMS['hydro_month_nh'] = 1
        exps = ['CRU',
                'ERA5',
                #'CERA', HISTALP', 'ERA5L',
                'ERA5_daily', 'WFDE5_CRU_daily', 'W5E5_daily',
                'W5E5_MSWEP_daily', 'WFDE5_CRU_monthly',
                'W5E5_monthly', 'W5E5_MSWEP_monthly']
        ref_hgts = []
        dft = []
        dfp = []
        for base in exps:
            if base not in ['ERA5_daily', 'WFDE5_CRU_daily',
                            'W5E5_daily',
                            'WFDE5_CRU_monthly', 'W5E5_monthly',
                            'W5E5_MSWEP_daily', 'W5E5_MSWEP_monthly']:
                cfg.PARAMS['baseline_climate'] = base
                tasks.process_climate_data(gdir, output_filesuffix=base)
                fs = base
            elif base == 'ERA5_daily':
                cfg.PARAMS['baseline_climate'] = base
                process_era5_daily_data(gdir, output_filesuffix=base)
                fs = base
            elif base == 'WFDE5_CRU_daily' or base == 'W5E5_daily' or base == 'W5E5_MSWEP_daily':
                process_w5e5_data(gdir, climate_type=base[:-6],
                                  temporal_resol='daily')
                fs = f'_daily_{base[:-6]}'
            elif '_monthly' in base:
                # wfde5_cru and w5e5
                process_w5e5_data(gdir, climate_type=base[:-8],
                                  temporal_resol='monthly')
                fs = f'_monthly_{base[:-8]}'

            f = gdir.get_filepath('climate_historical',
                                   filesuffix=fs)

            with xr.open_dataset(f) as ds:
                ref_hgts.append(ds.ref_hgt)
                assert ds.ref_pix_dis < 30000
                dft.append(ds.temp.to_series())
                dfp.append(ds.prcp.to_series())
        dft = pd.concat(dft, axis=1, keys=exps)
        dfp = pd.concat(dfp, axis=1, keys=exps)

        # compare daily mean temperatures of ERA5 and WFDE5
        assert dft[['ERA5_daily', 'WFDE5_CRU_daily']].corr().min().min() > 0.95
        assert dft[['W5E5_daily', 'WFDE5_CRU_daily']].corr().min().min() > 0.95
        # compare temp W5E5 to W5E5_MSWEP -> should be the same!!!
        assert dft[['W5E5_daily', 'W5E5_MSWEP_daily']].corr().min().min() > 0.9999
        assert dft[['W5E5_monthly', 'W5E5_MSWEP_monthly']].corr().min().min() > 0.9999

        # want to compare mean monthly temperatures
        # (daily resolution datasets have to be resampled)
        dft = dft.resample('MS').mean()
        print(dft)
        # Common period
        #dft = dft.resample(time='1M').mean()
        dfy = dft.resample('AS').mean().dropna().iloc[1:]
        dfm = dft.groupby(dft.index.month).mean()
        assert dfy.corr().min().min() > 0.44  # ERA5L and CERA do not correlate
        assert dfm.corr().min().min() > 0.97
        dfavg = dfy.describe()

        # Correct for hgt
        ref_h = ref_hgts[0]
        for h, d in zip(ref_hgts, exps):
            dfy[d] = dfy[d] - 0.0065 * (ref_h - h)
            dfm[d] = dfm[d] - 0.0065 * (ref_h - h)
        dfavg_cor = dfy.describe()

        # After correction less spread
        assert dfavg_cor.loc['mean'].std() < 0.8 * dfavg.loc['mean'].std()
        print(dfavg_cor)
        # with ERA5_daily and WFDE5_daily, smaller std (from <2.1 -> <1.7)
        assert dfavg_cor.loc['mean'].std() < 1.7

        # PRECIP
        # want to compare summed up monthly precipitation
        # (daily resolution datasets, so far only wfde5, have to resample)
        dfp = dfp.resample('MS').sum(min_count =1)
        # Common period
        dfy = dfp.resample('AS').mean().dropna().iloc[1:] * 12
        dfm = dfp.groupby(dfp.index.month).mean()
        assert dfy.corr().min().min() > 0.3 # CERA and MSWEP quite different
        # without MSWEP (because different gridpoint)
        assert dfy[dfy.columns.drop(['W5E5_MSWEP_monthly',
                                    'W5E5_MSWEP_daily'])].corr().min().min() > 0.6
        # monthly prcp cycle
        # WFDE5 is quite different > 0.8 -> > 0.75
        assert dfm.corr().min().min() > 0.7  # 0.8
        #assert dfm[dfm.columns.drop(['W5E5_MSWEP_monthly',
        #               'W5E5_MSWEP_daily'])].corr().min().min() > 0.7
        dfavg = dfy.describe()
        assert dfavg.loc['mean'].std() / dfavg.loc['mean'].mean() < 0.32 #28  # %


