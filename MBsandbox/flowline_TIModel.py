import numpy as np
import xarray as xr

import logging
import warnings
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

from oggm import cfg, utils
from oggm.exceptions import InvalidParamsError

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

@entity_task(log)
def run_with_hydro_daily(gdir, run_task=None, ref_area_from_y0=False, Testing=False, **kwargs):
    """Run the flowline model and add hydro diagnostics on daily basis (experimental!).
    Parameters
    ----------
    run_task : func
        any of the `run_*`` tasks in the MBSandbox.flowline_TIModel module.
        The mass-balance model used needs to have the `add_climate` output
        kwarg available though.
    ref_area_from_y0 : bool
        the hydrological output is computed over a reference area, which
        per default is the largest area covered by the glacier in the simulation
        period. Use this kwarg to force a specific area to the state of the
        glacier at the provided simulation year.
    Testing: if set to true, the 29th of February is set to nan values in non-leap years, so that the remaining days
        are at the same index in non-leap and leap years, if set to false the last 366th day in non-leap years
        is set to zero
    **kwargs : all valid kwargs for ``run_task``
    """

    # Make sure it'll return something
    kwargs['return_value'] = True

    # Check that kwargs are compatible
    if kwargs.get('store_monthly_step', False):
        raise InvalidParamsError('run_with_hydro only compatible with '
                                 'store_monthly_step=False.')
    if kwargs.get('mb_elev_feedback', 'annual') != 'annual':
        raise InvalidParamsError('run_with_hydro_daily only compatible with '
                                 "mb_elev_feedback='annual' (yes, even "
                                 "when asked for monthly hydro output).")
    out = run_task(gdir, **kwargs)
    if out is None:
        raise InvalidWorkflowError('The run task ({}) did not run '
                                   'successfully.'.format(run_task.__name__))

    # Mass balance model used during the run
    mb_mod = out.mb_model

    # Glacier geometry during the run
    suffix = kwargs.get('output_filesuffix', '')

    # We start by fetching mass balance data and geometry for all years
    # model_geometry files always retrieve yearly timesteps
    fmod = FileModel(gdir.get_filepath('model_geometry', filesuffix=suffix))
    # The last one is the final state - we can't compute MB for that
    years = fmod.years[:-1]

    # Geometry at y0 to start with + off-glacier snow bucket
    bin_area_2ds = []
    bin_elev_2ds = []
    ref_areas = []
    snow_buckets = []
    for fl in fmod.fls:
        # Glacier area on bins
        bin_area = fl.bin_area_m2

        ref_areas.append(bin_area)
        # snow_buckets.append(bin_area * 0)
        # snow_buckets.append(np.zeros(len(bin_area)))
        snow_buckets.append(np.zeros(len(bin_area)))

        # Output 2d data
        shape = len(years), len(bin_area)
        bin_area_2ds.append(np.empty(shape, np.float64))
        bin_elev_2ds.append(np.empty(shape, np.float64))

    # Ok now fetch all geometry data in a first loop
    # We do that because we might want to get the largest possible area (default)
    # and we want to minimize the number of calls to run_until
    for i, yr in enumerate(years):
        fmod.run_until(yr)
        for fl_id, (fl, bin_area_2d, bin_elev_2d) in \
                enumerate(zip(fmod.fls, bin_area_2ds, bin_elev_2ds)):
            # Time varying bins
            bin_area_2d[i, :] = fl.bin_area_m2
            bin_elev_2d[i, :] = fl.surface_h

    if not ref_area_from_y0:
        # Ok we get the max area instead
        for ref_area, bin_area_2d in zip(ref_areas, bin_area_2ds):
            ref_area[:] = bin_area_2d.max(axis=0)

    # Ok now we have arrays, we can work with that
    # -> second time varying loop is for mass-balance

    ntime = len(years) + 1
    # for each year store 366 values #last one should be 0 or nann in non leap years
    oshape = (ntime, 366)
    # for daily usage
    seconds = cfg.SEC_IN_DAY

    out = {
        'off_area': {
            'description': 'Off-glacier area',
            'unit': 'm 2',
            'data': np.zeros(ntime),
        },
        'on_area': {
            'description': 'On-glacier area',
            'unit': 'm 2',
            'data': np.zeros(ntime),
        },
        'melt_off_glacier': {
            'description': 'Off-glacier melt',
            'unit': 'kg yr-1',
            'data': np.zeros(oshape),
        },
        'melt_on_glacier': {
            'description': 'On-glacier melt',
            'unit': 'kg yr-1',
            'data': np.zeros(oshape),
        },
        'melt_residual_off_glacier': {
            'description': 'Off-glacier melt due to MB model residual',
            'unit': 'kg yr-1',
            'data': np.zeros(oshape),
        },
        'melt_residual_on_glacier': {
            'description': 'On-glacier melt due to MB model residual',
            'unit': 'kg yr-1',
            'data': np.zeros(oshape),
        },
        'liq_prcp_off_glacier': {
            'description': 'Off-glacier liquid precipitation',
            'unit': 'kg yr-1',
            'data': np.zeros(oshape),
        },
        'liq_prcp_on_glacier': {
            'description': 'On-glacier liquid precipitation',
            'unit': 'kg yr-1',
            'data': np.zeros(oshape),
        },
        'snowfall_off_glacier': {
            'description': 'Off-glacier solid precipitation',
            'unit': 'kg yr-1',
            'data': np.zeros(oshape),
        },
        'snowfall_on_glacier': {
            'description': 'On-glacier solid precipitation',
            'unit': 'kg yr-1',
            'data': np.zeros(oshape),
        },
        'snow_bucket': {
            'description': 'Off-glacier snow reservoir (state variable)',
            'unit': 'kg',
            'data': np.zeros(oshape),
        },
        'model_mb': {
            'description': 'Annual mass-balance from dynamical model',
            'unit': 'kg yr-1',
            'data': np.zeros(ntime),
        },
        'residual_mb': {
            'description': 'Difference (before correction) between mb model and dyn model melt',
            'unit': 'kg yr-1',
            'data': np.zeros(oshape),
        },
    }

    # Initialize
    fmod.run_until(years[0])
    prev_model_vol = fmod.volume_m3

    for i, yr in enumerate(years):

        for fl_id, (ref_area, snow_bucket, bin_area_2d, bin_elev_2d) in \
                enumerate(zip(ref_areas, snow_buckets, bin_area_2ds, bin_elev_2ds)):

            bin_area = bin_area_2d[i, :]
            bin_elev = bin_elev_2d[i, :]

            # Make sure we have no negative contribution when glaciers are out
            off_area = utils.clip_min(ref_area - bin_area, 0)

            try:
                try:
                    mb_out = mb_mod.get_daily_mb(bin_elev, fl_id=fl_id,
                                                 year=yr,
                                                 add_climate=True)
                    mb, _, _, prcp, prcpsol = mb_out
                except:
                    raise InvalidWorkflowError('Run with hydro daily needs a daily MB '
                                               'model, so it can only run with TIModel.')

            except ValueError as e:
                if 'too many values to unpack' in str(e):
                    raise InvalidWorkflowError('Run with hydro needs a MB '
                                               'model able to add climate '
                                               'info to `get_annual_mb`.')
                raise

            # Here we use mass (kg/time) not ice volume (mb is m ice per second)
            mb *= seconds * cfg.PARAMS['ice_density']

            # Bias of the mb model is a fake melt term that we need to deal with

            #check if year is leap year
            days_in_year = len(np.sum(prcpsol, axis=0))
            SEC_IN_YEAR = cfg.SEC_IN_DAY * days_in_year

            mb_bias = mb_mod.bias * seconds / SEC_IN_YEAR #cfg.SEC_IN_YEAR

            # on daily basis prcp has shape (bins, days in year) bin_area must have shape (bins,1)
            bin_area = bin_area[:, np.newaxis]
            off_area = off_area[:, np.newaxis]
            liq_prcp_on_g = (prcp - prcpsol) * bin_area
            liq_prcp_off_g = (prcp - prcpsol) * off_area

            prcpsol_on_g = prcpsol * bin_area
            prcpsol_off_g = prcpsol * off_area

            # IMPORTANT: this does not guarantee that melt cannot be negative
            # the reason is the MB residual that here can only be understood
            # as a fake melt process.
            # In particular at the monthly scale this can lead to negative
            # or winter positive melt - we try to mitigate this
            # issue at the end of the year
            melt_on_g = (prcpsol - mb) * bin_area
            melt_off_g = (prcpsol - mb) * off_area

            # This is the bad boy
            bias_on_g = mb_bias * bin_area
            bias_off_g = mb_bias * off_area

            # Update bucket with accumulation and melt
            # snow bucket has size (heights, 366) but prcpsol_off_g only has (heights, 365 if no leap year)
            # so we have to add a column to
            if days_in_year == 365:
                prcpsol_off_g = np.c_[prcpsol_off_g, np.zeros(len(bin_elev))]
                melt_off_g = np.c_[melt_off_g, np.zeros(len(bin_elev))]

            # loop through all days to get snow bucket correctly
            snow_bucket_daily = np.zeros((len(snow_bucket), 366))
            for day in range(366):
                # you have to have snow bucket from day before that gets updated
                # but you also have to store it before
                snow_bucket += prcpsol_off_g[:, day]
                # It can only melt that much
                melt_off_g[:, day] = np.where((snow_bucket - melt_off_g[:, day]) >= 0, melt_off_g[:, day], snow_bucket)
                # Update bucket
                snow_bucket -= melt_off_g[:, day]
                snow_bucket_daily[:, day] = snow_bucket

            # Daily out
            # we want daily output of all bins, so the bins have to be summed up
            # if not a leap year, the last day will remain 0
            out['melt_off_glacier']['data'][i, :] = np.sum(melt_off_g, axis=0)
            out['melt_on_glacier']['data'][i, :days_in_year] = np.sum(melt_on_g, axis=0)
            out['melt_residual_off_glacier']['data'][i, :days_in_year] = np.sum(bias_off_g, axis=0)
            out['melt_residual_on_glacier']['data'][i, :days_in_year] = np.sum(bias_on_g, axis=0)
            out['liq_prcp_off_glacier']['data'][i, :days_in_year] = np.sum(liq_prcp_off_g, axis=0)
            out['liq_prcp_on_glacier']['data'][i, :days_in_year] = np.sum(liq_prcp_on_g, axis=0)
            out['snowfall_off_glacier']['data'][i, :] = np.sum(prcpsol_off_g, axis=0)
            out['snowfall_on_glacier']['data'][i, :days_in_year] = np.sum(prcpsol_on_g, axis=0)

            # Snow bucket is a state variable - stored at end of timestamp
            # last day of year has to be stored as the first one for next year
            out['snow_bucket']['data'][i + 1, 0] += np.sum(snow_bucket_daily, axis=0)[-1]
            out['snow_bucket']['data'][i, 1:] += np.sum(snow_bucket_daily, axis=0)[:-1]

        # Update the annual data
        out['off_area']['data'][i] = np.sum(off_area)
        out['on_area']['data'][i] = np.sum(bin_area)

        # put the residual where we can
        for melt, bias in zip(
                [
                    out['melt_on_glacier']['data'][i, :],
                    out['melt_off_glacier']['data'][i, :],
                ],
                [
                    out['melt_residual_on_glacier']['data'][i, :],
                    out['melt_residual_off_glacier']['data'][i, :],
                ],
        ):

            real_melt = melt - bias
            to_correct = utils.clip_min(real_melt, 0)
            to_correct_sum = np.sum(to_correct)
            if (to_correct_sum > 1e-7) and (np.sum(melt) > 0):
                # Ok we correct the positive melt instead
                fac = np.sum(melt) / to_correct_sum
                melt[:] = to_correct * fac

        # Correct for mass-conservation and match the ice-dynamics model
        fmod.run_until(yr + 1)
        model_mb = (fmod.volume_m3 - prev_model_vol) * cfg.PARAMS['ice_density']
        prev_model_vol = fmod.volume_m3

        reconstructed_mb = (out['snowfall_on_glacier']['data'][i, :].sum() -
                            out['melt_on_glacier']['data'][i, :].sum())
        residual_mb = model_mb - reconstructed_mb

        # Now correct
        # We try to correct the melt only where there is some
        asum = out['melt_on_glacier']['data'][i, :].sum()
        if asum > 1e-7 and (residual_mb / asum < 1):
            # try to find a fac
            fac = 1 - residual_mb / asum
            corr = out['melt_on_glacier']['data'][i, :] * fac
            residual_mb = out['melt_on_glacier']['data'][i, :] - corr
            out['melt_on_glacier']['data'][i, :] = corr
        else:
            # We simply spread over the days
            # TO DO: more sophisticated approach??
            # with this approach proably more negative melt contributions??
            residual_mb /= days_in_year
            out['melt_on_glacier']['data'][i, :] = (out['melt_on_glacier']['data'][i, :] -
                                                    residual_mb)

        out['model_mb']['data'][i] = model_mb
        out['residual_mb']['data'][i] = residual_mb

        vars = ['melt_off_glacier', 'melt_on_glacier', 'melt_residual_off_glacier', 'melt_residual_on_glacier',
                'liq_prcp_off_glacier', 'liq_prcp_on_glacier', 'snowfall_off_glacier', 'snowfall_on_glacier']
        if days_in_year == 365:
            for var in vars:
                out[var]['data'][i, -1] = np.NaN
                if Testing == True:
                    # this is basically just for testing (to see whether monthly volumes of daily and monthly mb match
                    # don't use it for the real version
                    #only works if sm = 1
                    out[var]['data'][i, :] = np.concatenate(
                        (out[var]['data'][i, :59], np.array([np.NaN]), out[var]['data'][i, 59:-1]))

    # Convert to xarray
    out_vars = cfg.PARAMS['store_diagnostic_variables']
    ods = xr.Dataset()
    ods.coords['time'] = fmod.years
    ods.coords['day_2d'] = ('day_2d', np.arange(1, 367))
    # For the user later
    # if nh = 10, if sh = 4
    # first day of October 274 (leapyear 275)
    # first day of April 91 (leapyear 92)
    sm = cfg.PARAMS['hydro_month_' + mb_mod.hemisphere]
    # in Lili's massbalance this is 1 for nh and 4 for sh
    # TO DO: something is wrong here!!!
    # sm should be one
    if sm == 10:
        dayofyear = 275
    elif sm == 4:
        dayofyear = 92
    elif sm == 1:
        # like in Lili's model
        dayofyear = 1
    if sm != 1 and Testing == True:
        warnings.warn("Testing assumes calendar to start at beginning of year but this does not seem to be the case")

    ods.coords['calendar_day_2d'] = ('day_2d', (np.arange(366) + dayofyear - 1) % 366 + 1)
    # so the dataset is made with leapyears, because that is longest, for non leap year this has to be kept in mind
    for varname, d in out.items():
        data = d.pop('data')
        if varname not in out_vars:
            continue
        if len(data.shape) == 2:
            # First the annual agg
            if varname == 'snow_bucket':
                # Snowbucket is a state variable
                ods[varname] = ('time', data[:, 0])
            else:
                # Last year is never good
                data[-1, :] = np.NaN
                var_annual = np.nansum(data, axis=1)
                var_annual[-1] = np.NaN
                ods[varname] = ('time', var_annual)
            # Then the daily ones
            ods[varname + '_daily'] = (('time', 'day_2d'), data)
        else:
            assert varname != 'snow_bucket'
            data[-1] = np.NaN
            ods[varname] = ('time', data)
        for k, v in d.items():
            ods[varname].attrs[k] = v

    # Append the output to the existing diagnostics
    fpath = gdir.get_filepath('model_diagnostics', filesuffix=suffix)
    ods.to_netcdf(fpath, mode='a')
    return ods

