import numpy as np
import oggm
from oggm import entity_task

from oggm.core.flowline import FileModel
from oggm.exceptions import InvalidWorkflowError

# import the MBsandbox modules
from MBsandbox.mbmod_daily_oneflowline import TIModel, RandomMassBalance_TIModel
from MBsandbox.mbmod_daily_oneflowline import \
    MultipleFlowlineMassBalance_TIModel
from oggm.core.flowline import flowline_model_run
from oggm.core.massbalance import ConstantMassBalance
import logging

log = logging.getLogger(__name__)

### maybe these won't be necessary if the OGGM core flowline run_from_climate_data
# and run_from_constant_data are enough flexible to use another MultipleFlowlineMassBalance
# model ...

# # do it similar as in run_from_climate_data()




@entity_task(log)
def run_from_climate_data_TIModel(gdir, ys=None, ye=None, min_ys=None,
                                  max_ys=None,
                                  store_monthly_step=False,
                                  climate_filename='climate_historical',
                                  climate_input_filesuffix='',
                                  output_filesuffix='',
                                  init_model_filesuffix=None,
                                  init_model_yr=None,
                                  init_model_fls=None,
                                  zero_initial_glacier=False,
                                  bias=0,
                                  melt_f=None,
                                  precipitation_factor=None,
                                  temperature_bias=None,
                                  mb_type='mb_monthly', grad_type='cte',
                                  mb_model_class=TIModel,
                                  **kwargs):
    """ Runs a glacier with climate input from e.g. W5E5 or a GCM.

    This will initialize a
    :py:class:`MBsandbox.MultipleFlowlineMassBalance_TIModel`,
    and run a :py:func:`oggm.core.flowline.flowline_model_run`.

    same as in run_from_climate_data but compatible with TIModel


    Parameters:
    ----------------------------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory to process
    ys : int
        start year of the model run (default: from the glacier geometry
        date if init_model_filesuffix is None, else init_model_yr)
    ye : int
        end year of the model run (default: last year of the provided
        climate file)
    min_ys : int
        if you want to impose a minimum start year, regardless if the glacier
        inventory date is earlier (e.g. if climate data does not reach).
    max_ys : int
        if you want to impose a maximum start year, regardless if the glacier
        inventory date is later (e.g. if climate data does not reach).
    store_monthly_step : bool
        whether to store the diagnostic data at a monthly time step or not
        (default is yearly)
    #TODO: should this be included?
    #store_model_geometry : bool
    #    whether to store the full model geometry run file to disk or not.
    #    (new in OGGM v1.4.1: default is to follow
    #    cfg.PARAMS['store_model_geometry'])
    climate_filename : str
        name of the climate file, e.g. 'climate_historical' (default) or
        'gcm_data'
    climate_input_filesuffix: str
        filesuffix for the input climate file, use e.g. 'W5E5' or 'WFDE5_CRU'
    output_filesuffix : str
        for the output file
    init_model_filesuffix : str
        if you want to start from a previous model run state. Can be
        combined with `init_model_yr`
    init_model_yr : int
        the year of the initial run you want to start from. The default
        is to take the last year of the simulation.
    init_model_fls : []
        list of flowlines to use to initialise the model (the default is the
        present_time_glacier file from the glacier directory).
        Ignored if `init_model_filesuffix` is set
    zero_initial_glacier : bool
        if true, the ice thickness is set to zero before the simulation
    bias : float
        equal to the residual in TIModel, best is to leave it at 0 !
    melt_f:
        calibrated melt_f (float) or 'from_json', then the saved json
        file from the right prcp-fac and climate is opened and that melt_f is chosen
    temperature_bias : float
        add a bias to the temperature timeseries
    precipitation_factor: float
        multiply a factor to the precipitation time series
        use the value from the calibration!
    kwargs : dict
        kwargs to pass to the FluxBasedModel instance
    """

    if init_model_filesuffix is not None:
        fp = gdir.get_filepath('model_geometry',
                               filesuffix=init_model_filesuffix)

        fmod = FileModel(fp)
        if init_model_yr is None:
            init_model_yr = fmod.last_yr
        fmod.run_until(init_model_yr)
        init_model_fls = fmod.fls
        if ys is None:
            ys = init_model_yr

    # Take from rgi date if not set yet
    if ys is None:
        try:
            ys = gdir.rgi_date.year
        except AttributeError:
            ys = gdir.rgi_date
        # The RGI timestamp is in calendar date - we convert to hydro date,
        # i.e. 2003 becomes 2004 (so that we don't count the MB year 2003
        # in the simulation)
        # See also: https://github.com/OGGM/oggm/issues/1020
        ys += 1

    # Final crop
    if min_ys is not None:
        ys = ys if ys > min_ys else min_ys
    if max_ys is not None:
        ys = ys if ys < max_ys else max_ys

    if melt_f == 'from_json':
        fs = '_{}_{}_{}'.format(climate_input_filesuffix, mb_type, grad_type)
        d = gdir.read_json(filename='melt_f_geod', filesuffix=fs)
        # get the calibrated melt_f that suits to the prcp factor
        try:
            melt_f_chosen = d['melt_f_pf_{}'.format(np.round(precipitation_factor, 2))]
        except:
            raise InvalidWorkflowError('there is no calibrated melt_f for this precipitation factor, glacier, climate'
                                       'mb_type and grad_type, need to run first melt_f_calib_geod_prep_inversion'
                                       'with these options!')
        #pd_inv_melt_f = pd.read_csv(melt_f_file, index_col='RGIId')
        #melt_f_chosen = pd_inv_melt_f['melt_f_opt'].loc[gdir.rgi_id]
        # use same pf as from initialisation and calibration
        #np.testing.assert_allclose(precipitation_factor, pd_inv_melt_f['pf'])
    else:
        melt_f_chosen = melt_f

    mb = MultipleFlowlineMassBalance_TIModel(gdir, mb_model_class=mb_model_class,
                                             prcp_fac=precipitation_factor,
                                             melt_f=melt_f_chosen,
                                             filename=climate_filename,
                                             bias=bias,
                                             input_filesuffix=climate_input_filesuffix,
                                             mb_type=mb_type,
                                             grad_type=grad_type,
                                             # check_calib_params=check_calib_params,
                                             )

    # if temperature_bias is not None:
    #    mb.temp_bias = temperature_bias
    if precipitation_factor is not None:
        mb.prcp_fac = precipitation_factor

    if temperature_bias is not None:
        mb.temp_bias = temperature_bias
    else:
        # do the quality check!
        mb.flowline_mb_models[-1].historical_climate_qc_mod(gdir)

    if ye is None:
        # Decide from climate (we can run the last year with data as well)
        ye = mb.flowline_mb_models[0].ye + 1

    return flowline_model_run(gdir, output_filesuffix=output_filesuffix,
                              mb_model=mb, ys=ys, ye=ye,
                              store_monthly_step=store_monthly_step,
                              init_model_fls=init_model_fls,
                              zero_initial_glacier=zero_initial_glacier,
                              **kwargs)


@entity_task(log)
def run_random_climate_TIModel(gdir, nyears=1000, y0=None, halfsize=15,
                               mb_model_sub_class=TIModel,
                               temperature_bias=None,
                               mb_type='mb_monthly', grad_type='cte',
                               bias=0, seed=None,
                       melt_f=None,
                       precipitation_factor=None,
                       store_monthly_step=False,
                       store_model_geometry=None,
                       climate_filename='climate_historical',
                       climate_input_filesuffix='',
                       output_filesuffix='', init_model_fls=None,
                       zero_initial_glacier=False,
                       unique_samples=False, #melt_f_file=None,
                               **kwargs):
    """Runs the random mass-balance model for a given number of years.

    copy of run_random_climate --> needs to be tested ...

    This will initialize a
    :py:class:`MBsandbox.MultipleFlowlineMassBalance_TIModel`,
    and run a :py:func:`oggm.core.flowline.flowline_model_run`.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory to process
    nyears : int
        length of the simulation
    y0 : int, optional
        central year of the random climate period. The default is to be
        centred on t*.
    halfsize : int, optional
        the half-size of the time window (window size = 2 * halfsize + 1)
    bias : float
        equal to the residual in TIModel, best is to leave it at 0 !
    seed : int
        seed for the random generator. If you ignore this, the runs will be
        different each time. Setting it to a fixed seed across glaciers can
        be useful if you want to have the same climate years for all of them
    store_monthly_step : bool
        whether to store the diagnostic data at a monthly time step or not
        (default is yearly)
    #TODO: should this be included?
    #store_model_geometry : bool
    #    whether to store the full model geometry run file to disk or not.
    #    (new in OGGM v1.4.1: default is to follow
    #    cfg.PARAMS['store_model_geometry'])
    climate_filename : str
        name of the climate file, e.g. 'climate_historical' (default) or
        'gcm_data'
    climate_input_filesuffix: str
        filesuffix for the input climate file, use e.g. 'W5E5' or 'WFDE5_CRU'
    output_filesuffix : str
        for the output file
    init_model_filesuffix : str
        if you want to start from a previous model run state. Can be
        combined with `init_model_yr`
    init_model_yr : int
        the year of the initial run you want to start from. The default
        is to take the last year of the simulation.
    init_model_fls : []
        list of flowlines to use to initialise the model (the default is the
        present_time_glacier file from the glacier directory).
        Ignored if `init_model_filesuffix` is set
    zero_initial_glacier : bool
        if true, the ice thickness is set to zero before the simulation
    melt_f:
        calibrated melt_f (float) or 'from_json', then the saved json
        file from the right prcp-fac and climate is opened and that melt_f is chosen
    temperature_bias : float
        add a bias to the temperature timeseries
    precipitation_factor: float
        multiply a factor to the precipitation time series
        use the value from the calibration!
    unique_samples: bool
        if true, chosen random mass-balance years will only be available once
        per random climate period-length
        if false, every model year will be chosen from the random climate
        period with the same probability
    kwargs : dict
        kwargs to pass to the FluxBasedModel instance
    """
    if melt_f == 'from_json':
        fs = '_{}_{}_{}'.format(climate_input_filesuffix, mb_type, grad_type)
        d = gdir.read_json(filename='melt_f_geod', filesuffix=fs)
        # get the calibrated melt_f that suits to the prcp factor
        try:
            melt_f_chosen = d['melt_f_pf_{}'.format(np.round(precipitation_factor, 2))]
        except:
            raise InvalidWorkflowError('there is no calibrated melt_f for this precipitation factor, glacier, climate'
                                       'mb_type and grad_type, need to run first melt_f_calib_geod_prep_inversion'
                                       'with these options!')
        # old method: use csv file to get the calibrated melt_f
        #pd_inv_melt_f = pd.read_csv(melt_f_file, index_col='RGIId')
        #melt_f_chosen = pd_inv_melt_f['melt_f_opt'].loc[gdir.rgi_id]
        # use same pf as from initialisation and calibration
        #np.testing.assert_allclose(precipitation_factor, pd_inv_melt_f['pf'])
    else:
        melt_f_chosen = melt_f

    mb = MultipleFlowlineMassBalance_TIModel(gdir,
                                             mb_model_class=RandomMassBalance_TIModel,
                                             y0=y0, halfsize=halfsize,
                                             melt_f=melt_f_chosen,
                                             prcp_fac=precipitation_factor,
                                             mb_type=mb_type,
                                             grad_type=grad_type,
                                             bias = bias,
                                             seed=seed,
                                             mb_model_sub_class = mb_model_sub_class,
                                     filename=climate_filename,
                                     input_filesuffix=climate_input_filesuffix,
                                     unique_samples=unique_samples)

    if precipitation_factor is not None:
        mb.prcp_fac = precipitation_factor
    if temperature_bias is not None:
        mb.temp_bias = temperature_bias
    else:
        # do the quality check!
        mb.flowline_mb_models[-1].historical_climate_qc_mod(gdir)


    return flowline_model_run(gdir, output_filesuffix=output_filesuffix,
                              mb_model=mb, ys=0, ye=nyears,
                              store_monthly_step=store_monthly_step,
                              store_model_geometry=store_model_geometry,
                              init_model_fls=init_model_fls,
                              zero_initial_glacier=zero_initial_glacier,
                              **kwargs)


# work in Process:
# problem: don't have a constant mb TIModel, this would be quite a lot of work ...
# not yet adapted at all, first need a new ConstantMbModel_TIModel!!
@entity_task(log)
def run_constant_climate_TIModel(gdir, nyears=1000, y0=None, halfsize=15,
                         bias=None, temperature_bias=None,
                         precipitation_factor=None,
                         store_monthly_step=False,
                         store_model_geometry=None,
                         init_model_filesuffix=None,
                         init_model_yr=None,
                         output_filesuffix='',
                         climate_filename='climate_historical',
                         climate_input_filesuffix='',
                         init_model_fls=None,
                         zero_initial_glacier=False, **kwargs):
    """Runs the constant mass-balance model for a given number of years.

    This will initialize a
    :py:class:`oggm.core.massbalance.MultipleFlowlineMassBalance`,
    and run a :py:func:`oggm.core.flowline.flowline_model_run`.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory to process
    nyears : int
        length of the simulation (default: as long as needed for reaching
        equilibrium)
    y0 : int
        central year of the requested climate period. The default is to be
        centred on t*.
    halfsize : int, optional
        the half-size of the time window (window size = 2 * halfsize + 1)
    bias : float
        bias of the mb model. Default is to use the calibrated one, which
        is often a better idea. For t* experiments it can be useful to set it
        to zero
    temperature_bias : float
        add a bias to the temperature timeseries
    precipitation_factor: float
        multiply a factor to the precipitation time series
        default is None and means that the precipitation factor from the
        calibration is applied which is cfg.PARAMS['prcp_scaling_factor']
    store_monthly_step : bool
        whether to store the diagnostic data at a monthly time step or not
        (default is yearly)
    store_model_geometry : bool
        whether to store the full model geometry run file to disk or not.
        (new in OGGM v1.4.1: default is to follow
        cfg.PARAMS['store_model_geometry'])
    init_model_filesuffix : str
        if you want to start from a previous model run state. Can be
        combined with `init_model_yr`
    init_model_yr : int
        the year of the initial run you want to start from. The default
        is to take the last year of the simulation.
    climate_filename : str
        name of the climate file, e.g. 'climate_historical' (default) or
        'gcm_data'
    climate_input_filesuffix: str
        filesuffix for the input climate file
    output_filesuffix : str
        this add a suffix to the output file (useful to avoid overwriting
        previous experiments)
    zero_initial_glacier : bool
        if true, the ice thickness is set to zero before the simulation
    init_model_fls : []
        list of flowlines to use to initialise the model (the default is the
        present_time_glacier file from the glacier directory)
    kwargs : dict
        kwargs to pass to the FluxBasedModel instance
    """

    NotImplementedError('work in process...')

    if init_model_filesuffix is not None:
        fp = gdir.get_filepath('model_geometry',
                               filesuffix=init_model_filesuffix)
        fmod = FileModel(fp)
        if init_model_yr is None:
            init_model_yr = fmod.last_yr
        fmod.run_until(init_model_yr)
        init_model_fls = fmod.fls

    mb = MultipleFlowlineMassBalance_TIModel(gdir, mb_model_class=ConstantMassBalance,
                                     y0=y0, halfsize=halfsize,
                                     bias=bias, filename=climate_filename,
                                     input_filesuffix=climate_input_filesuffix)

    if temperature_bias is not None:
        mb.temp_bias = temperature_bias
    if precipitation_factor is not None:
        mb.prcp_fac = precipitation_factor

    return flowline_model_run(gdir, output_filesuffix=output_filesuffix,
                              mb_model=mb, ys=0, ye=nyears,
                              store_monthly_step=store_monthly_step,
                              store_model_geometry=store_model_geometry,
                              init_model_fls=init_model_fls,
                              zero_initial_glacier=zero_initial_glacier,
                              **kwargs)