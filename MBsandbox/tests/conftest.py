"""Pytest fixtures to be used in other test modules"""

import pytest
import os
import oggm

# imports from OGGM
from oggm import utils, workflow, cfg

@pytest.fixture(scope='class')
def gdir():
    """ Provides a copy of the base Hintereisenferner glacier directory in
        a case directory specific to the current test class using the single
        elev_bands as flowlines. All cases in
        the test class will use the same copy of this glacier directory.
    """

    cfg.initialize()

    test_dir = '/home/lilianschuster/Schreibtisch/PhD/oggm_files/MBsandbox_tests'
    if not os.path.exists(test_dir):
        test_dir = utils.gettempdir(dirname='OGGM_MBsandbox_test',
                                    reset=True)

    cfg.PATHS['working_dir'] = test_dir
    base_url = ('https://cluster.klima.uni-bremen.de/~oggm/gdirs/oggm_v1.4/'
                'L1-L2_files/elev_bands')

    df = ['RGI60-11.00897']
    gdirs = workflow.init_glacier_directories(df, from_prepro_level=2,
                                              prepro_border=10,
                                              prepro_base_url=base_url,
                                              prepro_rgi_version='62')
    return gdirs[0]