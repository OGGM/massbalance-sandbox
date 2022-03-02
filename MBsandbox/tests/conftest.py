"""Pytest fixtures to be used in other test modules"""

import os

# imports from OGGM
from oggm import utils, workflow, cfg



import logging
import getpass
from functools import wraps

import numpy as np
import pytest
import shapely.geometry as shpg
import matplotlib.pyplot as plt

from oggm import cfg, tasks
from oggm import utils
from oggm.utils import mkdir, _downloads
from oggm.tests import HAS_MPL_FOR_TESTS, HAS_INTERNET

logger = logging.getLogger(__name__)


def pytest_configure(config):
    for marker in ["slow", "download", "creds", "internet", "test_env",
                   "graphic "]:
        config.addinivalue_line("markers", marker)
    if config.pluginmanager.hasplugin('xdist'):
        try:
            from ilock import ILock
            utils.lock = ILock("oggm_xdist_download_lock_" + getpass.getuser())
            logger.info("ilock locking setup successfully for xdist tests")
        except BaseException:
            logger.warning("could not setup ilock locking for distributed "
                           "tests")


def pytest_addoption(parser):
    parser.addoption("--run-slow", action="store_true", default=False,
                     help="Run slow tests")
    parser.addoption("--run-download", action="store_true", default=False,
                     help="Run download tests")
    parser.addoption("--run-creds", action="store_true", default=False,
                     help="Run download tests requiring credentials")
    parser.addoption("--run-test-env", metavar="ENVNAME", default="",
                     help="Run only specified test env")
    parser.addoption("--no-run-internet", action="store_true", default=False,
                     help="Don't run any tests accessing the internet")


def pytest_collection_modifyitems(config, items):
    use_internet = HAS_INTERNET and not config.getoption("--no-run-internet")
    skip_slow = not config.getoption("--run-slow")
    skip_download = not use_internet or not config.getoption("--run-download")
    skip_cred = skip_download or not config.getoption("--run-creds")
    run_test_env = config.getoption("--run-test-env")

    slow_marker = pytest.mark.skip(reason="need --run-slow option to run")
    download_marker = pytest.mark.skip(reason="need --run-download option to "
                                              "run, internet access is "
                                              "required")
    cred_marker = pytest.mark.skip(reason="need --run-creds option to run, "
                                          "internet access is required")
    internet_marker = pytest.mark.skip(reason="internet access is required")
    test_env_marker = pytest.mark.skip(reason="only test_env=%s tests are run"
                                              % run_test_env)
    graphic_marker = pytest.mark.skip(reason="requires mpl V1.5+ and "
                                             "pytest-mpl")

    for item in items:
        if skip_slow and "slow" in item.keywords:
            item.add_marker(slow_marker)
        if skip_download and "download" in item.keywords:
            item.add_marker(download_marker)
        if skip_cred and "creds" in item.keywords:
            item.add_marker(cred_marker)
        if not use_internet and "internet" in item.keywords:
            item.add_marker(internet_marker)

        if run_test_env:
            test_env = item.get_closest_marker("test_env")
            if not test_env or test_env.args[0] != run_test_env:
                item.add_marker(test_env_marker)

        if "graphic" in item.keywords:
            def wrap_graphic_test(test):
                @wraps(test)
                def test_wrapper(*args, **kwargs):
                    try:
                        return test(*args, **kwargs)
                    finally:
                        plt.close()
                return test_wrapper
            item.obj = wrap_graphic_test(item.obj)

            if not HAS_MPL_FOR_TESTS:
                item.add_marker(graphic_marker)




@pytest.fixture(scope='class')
def gdir():
    """ Provides a copy of the base Hintereisenferner glacier directory in
        a case directory specific to the current test class using the single
        elev_bands as flowlines. All cases in
        the test class will use the same copy of this glacier directory.
    """

    cfg.initialize()
    cfg.PARAMS['use_multiprocessing'] = False
    cfg.PARAMS['hydro_month_nh'] = 1
    cfg.PARAMS['hydro_month_sh'] = 1
    test_dir = '/home/lilianschuster/Schreibtisch/PhD/oggm_files/MBsandbox_tests'
    if not os.path.exists(test_dir):
        test_dir = utils.gettempdir(dirname='OGGM_MBsandbox_test',
                                    reset=True)

    cfg.PATHS['working_dir'] = test_dir
    base_url = ('https://cluster.klima.uni-bremen.de/~oggm/gdirs/oggm_v1.4/'
                'L1-L2_files/elev_bands')

    df = ['RGI60-11.00897'] # #897 HEF #1328 largest glacier Rhine, #1346 second largest glacier rhine
    #Pakistan: biafo glacier RGI60-14.00005, hispar glacier 14.04477, 14.06794 baltoro glacier
    #Andes: 'RGI60-16.01053', RGI60-16.01251
    gdirs = workflow.init_glacier_directories(df, from_prepro_level=2,
                                              prepro_border=10,
                                              prepro_base_url=base_url,
                                              prepro_rgi_version='62')
    return gdirs[0]


@pytest.fixture(scope='class')
def gdir_aletsch():
    """ Provides a copy of the base Hintereisenferner glacier directory in
        a case directory specific to the current test class using the single
        elev_bands as flowlines. All cases in
        the test class will use the same copy of this glacier directory.
    """

    cfg.initialize()
    cfg.PARAMS['use_multiprocessing'] = False
    cfg.PARAMS['hydro_month_nh'] = 1
    cfg.PARAMS['hydro_month_sh'] = 1
    test_dir = '/home/lilianschuster/Schreibtisch/PhD/oggm_files/MBsandbox_tests'
    if not os.path.exists(test_dir):
        test_dir = utils.gettempdir(dirname='OGGM_MBsandbox_test',
                                    reset=False)

    cfg.PATHS['working_dir'] = test_dir
    base_url = ('https://cluster.klima.uni-bremen.de/~oggm/gdirs/oggm_v1.4/'
                'L1-L2_files/elev_bands')

    df = ['RGI60-11.01450'] # #897 HEF #1328 largest glacier Rhine, #1346 second largest glacier rhine
    #Pakistan: biafo glacier RGI60-14.00005, hispar glacier 14.04477, 14.06794 baltoro glacier
    #Andes: 'RGI60-16.01053', RGI60-16.01251
    gdirs = workflow.init_glacier_directories(df, from_prepro_level=2,
                                              prepro_border=10,
                                              prepro_base_url=base_url,
                                              prepro_rgi_version='62')
    return gdirs[0]