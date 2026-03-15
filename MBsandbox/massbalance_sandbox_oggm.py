"""Mass balance models - next generation"""


# Built ins
import logging
import os
# External libs
import cftime
import numpy as np
import xarray as xr
import pandas as pd
from scipy.interpolate import interp1d
from scipy import optimize
# Locals
import oggm.cfg as cfg
from oggm.cfg import SEC_IN_YEAR, SEC_IN_MONTH
from oggm.utils import (SuperclassMeta, get_geodetic_mb_dataframe,
                        floatyear_to_date, date_to_floatyear,
                        monthly_timeseries, ncDataset,
                        clip_min, clip_max, clip_array, clip_scalar,
                        weighted_average_1d, lazy_property)
from oggm.exceptions import (InvalidWorkflowError, InvalidParamsError,
                             MassBalanceCalibrationError)
from oggm import entity_task

# Module logger
log = logging.getLogger(__name__)

# Climate relevant global params - not optimised
MB_GLOBAL_PARAMS = ['temp_default_gradient',
                    'temp_all_solid',
                    'temp_all_liq',
                    'temp_melt']

from oggm.core.massbalance import MassBalanceModel


class TestTIModel(MassBalanceModel):
    """Monthly temperature index model.
    """
    def __init__(self, gdir,
                 filename='climate_historical',
                 input_filesuffix='',
                 fl_id=None,
                 melt_f=None,
                 temp_bias=None,
                 prcp_fac=None,
                 bias=0,
                 ys=None,
                 ye=None,
                 repeat=False,
                 check_calib_params=True,
                 ):
        """Initialize.

        Parameters
        ----------
        gdir : GlacierDirectory
            the glacier directory
        filename : str, optional
            set to a different BASENAME if you want to use alternative climate
            data. Default is 'climate_historical'
        input_filesuffix : str, optional
            append a suffix to the filename (useful for GCM runs).
        fl_id : int, optional
            if this flowline has been calibrated alone and has specific
            model parameters.
        melt_f : float, optional
            set to the value of the melt factor you want to use,
            here the unit is kg m-2 day-1 K-1
            (the default is to use the calibrated value).
        temp_bias : float, optional
            set to the value of the temperature bias you want to use
            (the default is to use the calibrated value).
        prcp_fac : float, optional
            set to the value of the precipitation factor you want to use
            (the default is to use the calibrated value).
        bias : float, optional
            set to the alternative value of the calibration bias [mm we yr-1]
            you want to use (the default is to use the calibrated value)
            Note that this bias is *substracted* from the computed MB. Indeed:
            BIAS = MODEL_MB - REFERENCE_MB.
        ys : int
            The start of the climate period where the MB model is valid
            (default: the period with available data)
        ye : int
            The end of the climate period where the MB model is valid
            (default: the period with available data)
        repeat : bool
            Whether the climate period given by [ys, ye] should be repeated
            indefinitely in a circular way
        check_calib_params : bool
            OGGM will try hard not to use wrongly calibrated parameters
            by checking the global parameters used during calibration and
            the ones you are using at run time. If they don't match, it will
            raise an error. Set to "False" to suppress this check.
        """

        super(TestTIModel, self).__init__()
        self.valid_bounds = [-1e4, 2e4]  # in m
        self.fl_id = fl_id  # which flowline are we the model of?
        self.gdir = gdir

        if melt_f is None:
            melt_f = self.calib_params['melt_f']

        if temp_bias is None:
            temp_bias = self.calib_params['temp_bias']

        if prcp_fac is None:
            prcp_fac = self.calib_params['prcp_fac']

        # Check the climate related params to the GlacierDir to make sure
        if check_calib_params:
            mb_calib = self.calib_params['mb_global_params']
            for k, v in mb_calib.items():
                if v != cfg.PARAMS[k]:
                    msg = ('You seem to use different mass balance parameters '
                           'than used for the calibration: '
                           f"you use cfg.PARAMS['{k}']={cfg.PARAMS[k]} while "
                           f"it was calibrated with cfg.PARAMS['{k}']={v}. "
                           'Set `check_calib_params=False` to ignore this '
                           'warning.')
                    raise InvalidWorkflowError(msg)
            src = self.calib_params['baseline_climate_source']
            src_calib = gdir.get_climate_info()['baseline_climate_source']
            if src != src_calib:
                msg = (f'You seem to have calibrated with the {src} '
                       f"climate data while this gdir was calibrated with "
                       f"{src_calib}. Set `check_calib_params=False` to "
                       f"ignore this warning.")
                raise InvalidWorkflowError(msg)

        self.melt_f = melt_f
        self.bias = bias

        # Global parameters
        self.t_solid = cfg.PARAMS['temp_all_solid']
        self.t_liq = cfg.PARAMS['temp_all_liq']
        self.t_melt = cfg.PARAMS['temp_melt']

        # check if valid prcp_fac is used
        if prcp_fac <= 0:
            raise InvalidParamsError('prcp_fac has to be above zero!')
        default_grad = cfg.PARAMS['temp_default_gradient']

        # Public attrs
        self.hemisphere = gdir.hemisphere
        self.repeat = repeat

        # Private attrs
        # to allow prcp_fac to be changed after instantiation
        # prescribe the prcp_fac as it is instantiated
        self._prcp_fac = prcp_fac
        # same for temp bias
        self._temp_bias = temp_bias

        # Read climate file
        fpath = gdir.get_filepath(filename, filesuffix=input_filesuffix)
        with ncDataset(fpath, mode='r') as nc:
            # time
            time = nc.variables['time']
            time = cftime.num2date(time[:], time.units, calendar=time.calendar)
            ny, r = divmod(len(time), 12)
            if r != 0:
                raise ValueError('Climate data should be N full years')

            # We check for calendar years
            if (time[0].month != 1) or (time[-1].month != 12):
                raise InvalidWorkflowError('We now work exclusively with '
                                           'calendar years.')

            # Quick trick because we now the size of our array
            years = np.repeat(np.arange(time[-1].year - ny + 1,
                                        time[-1].year + 1), 12)
            pok = slice(None)  # take all is default (optim)
            if ys is not None:
                pok = years >= ys
            if ye is not None:
                try:
                    pok = pok & (years <= ye)
                except TypeError:
                    pok = years <= ye

            self.years = years[pok]
            self.months = np.tile(np.arange(1, 13), ny)[pok]

            # Read timeseries and correct it
            self.temp = nc.variables['temp'][pok].astype(np.float64) + self._temp_bias
            self.prcp = nc.variables['prcp'][pok].astype(np.float64) * self._prcp_fac
            if 'gradient' in nc.variables and cfg.PARAMS['temp_use_local_gradient']:
                grad = nc.variables['gradient'][pok].astype(np.float64)
                # Security for stuff that can happen with local gradients
                g_minmax = cfg.PARAMS['temp_local_gradient_bounds']
                grad = np.where(~np.isfinite(grad), default_grad, grad)
                grad = clip_array(grad, g_minmax[0], g_minmax[1])
            else:
                grad = self.prcp * 0 + default_grad
            self.grad = grad
            self.ref_hgt = nc.ref_hgt
            self.climate_source = nc.climate_source
            self.ys = self.years[0]
            self.ye = self.years[-1]

    def __repr__(self):
        """String Representation of the mass balance model"""
        summary = ['<oggm.MassBalanceModel>']
        summary += ['  Class: ' + self.__class__.__name__]
        summary += ['  Attributes:']
        # Add all scalar attributes
        done = []
        for k in ['hemisphere', 'climate_source', 'melt_f', 'prcp_fac', 'temp_bias', 'bias']:
            done.append(k)
            v = self.__getattribute__(k)
            if k == 'climate_source':
                if v.endswith('.nc'):
                    v = os.path.basename(v)
            nofloat = ['hemisphere', 'climate_source']
            nbform = '    - {}: {}' if k in nofloat else '    - {}: {:.02f}'
            summary += [nbform.format(k, v)]
        for k, v in self.__dict__.items():
            if np.isscalar(v) and not k.startswith('_') and k not in done:
                nbform = '    - {}: {}'
                summary += [nbform.format(k, v)]
        return '\n'.join(summary) + '\n'

    @property
    def monthly_melt_f(self):
        return self.melt_f * 365 / 12

    # adds the possibility of changing prcp_fac
    # after instantiation with properly changing the prcp time series
    @property
    def prcp_fac(self):
        """Precipitation factor (default: cfg.PARAMS['prcp_scaling_factor'])

        Called factor to make clear that it is a multiplicative factor in
        contrast to the additive temperature bias
        """
        return self._prcp_fac

    @prcp_fac.setter
    def prcp_fac(self, new_prcp_fac):
        # just to check that no invalid prcp_factors are used
        if np.any(np.asarray(new_prcp_fac) <= 0):
            raise InvalidParamsError('prcp_fac has to be above zero!')

        if len(np.atleast_1d(new_prcp_fac)) == 12:
            # OK so that's monthly stuff
            new_prcp_fac = np.tile(new_prcp_fac, len(self.prcp) // 12)

        self.prcp *= new_prcp_fac / self._prcp_fac
        self._prcp_fac = new_prcp_fac

    @property
    def temp_bias(self):
        """Add a temperature bias to the time series"""
        return self._temp_bias

    @temp_bias.setter
    def temp_bias(self, new_temp_bias):

        if len(np.atleast_1d(new_temp_bias)) == 12:
            # OK so that's monthly stuff
            new_temp_bias = np.tile(new_temp_bias, len(self.temp) // 12)

        self.temp += new_temp_bias - self._temp_bias
        self._temp_bias = new_temp_bias

    @lazy_property
    def calib_params(self):
        if self.fl_id is None:
            return self.gdir.read_json('mb_calib')
        else:
            try:
                return self.gdir.read_json('mb_calib',
                                           filesuffix=f'_fl{self.fl_id}')
            except FileNotFoundError:
                return self.gdir.read_json('mb_calib')

    def is_year_valid(self, year):
        return self.ys <= year <= self.ye

    def get_monthly_climate(self, heights, year=None):
        """Monthly climate information at given heights.

        Note that prcp is corrected with the precipitation factor and that
        all other model biases (temp and prcp) are applied.

        Returns
        -------
        (temp, tempformelt, prcp, prcpsol)
        """

        y, m = floatyear_to_date(year)
        if self.repeat:
            y = self.ys + (y - self.ys) % (self.ye - self.ys + 1)
        if not self.is_year_valid(y):
            raise ValueError('year {} out of the valid time bounds: '
                             '[{}, {}]'.format(y, self.ys, self.ye))
        pok = np.where((self.years == y) & (self.months == m))[0][0]

        # Read already (temperature bias and precipitation factor corrected!)
        itemp = self.temp[pok]
        iprcp = self.prcp[pok]
        igrad = self.grad[pok]

        # For each height pixel:
        # Compute temp and tempformelt (temperature above melting threshold)
        npix = len(heights)
        temp = np.ones(npix) * itemp + igrad * (heights - self.ref_hgt)
        tempformelt = temp - self.t_melt
        clip_min(tempformelt, 0, out=tempformelt)

        # Compute solid precipitation from total precipitation
        prcp = np.ones(npix) * iprcp
        fac = 1 - (temp - self.t_solid) / (self.t_liq - self.t_solid)
        prcpsol = prcp * clip_array(fac, 0, 1)

        return temp, tempformelt, prcp, prcpsol

    def _get_2d_annual_climate(self, heights, year):
        # Avoid code duplication with a getter routine
        year = np.floor(year)
        if self.repeat:
            year = self.ys + (year - self.ys) % (self.ye - self.ys + 1)
        if not self.is_year_valid(year):
            raise ValueError('year {} out of the valid time bounds: '
                             '[{}, {}]'.format(year, self.ys, self.ye))
        pok = np.where(self.years == year)[0]
        if len(pok) < 1:
            raise ValueError('Year {} not in record'.format(int(year)))

        # Read already (temperature bias and precipitation factor corrected!)
        itemp = self.temp[pok]
        iprcp = self.prcp[pok]
        igrad = self.grad[pok]

        # For each height pixel:
        # Compute temp and tempformelt (temperature above melting threshold)
        heights = np.asarray(heights)
        npix = len(heights)
        grad_temp = np.atleast_2d(igrad).repeat(npix, 0)
        grad_temp *= (heights.repeat(12).reshape(grad_temp.shape) -
                      self.ref_hgt)
        temp2d = np.atleast_2d(itemp).repeat(npix, 0) + grad_temp
        temp2dformelt = temp2d - self.t_melt
        clip_min(temp2dformelt, 0, out=temp2dformelt)

        # Compute solid precipitation from total precipitation
        prcp = np.atleast_2d(iprcp).repeat(npix, 0)
        fac = 1 - (temp2d - self.t_solid) / (self.t_liq - self.t_solid)
        prcpsol = prcp * clip_array(fac, 0, 1)

        return temp2d, temp2dformelt, prcp, prcpsol

    def get_annual_climate(self, heights, year=None):
        """Annual climate information at given heights.

        Note that prcp is corrected with the precipitation factor and that
        all other model biases (temp and prcp) are applied.

        Returns
        -------
        (temp, tempformelt, prcp, prcpsol)
        """
        t, tmelt, prcp, prcpsol = self._get_2d_annual_climate(heights, year)
        return (t.mean(axis=1), tmelt.sum(axis=1),
                prcp.sum(axis=1), prcpsol.sum(axis=1))

    def get_monthly_mb(self, heights, year=None, add_climate=False, **kwargs):

        t, tmelt, prcp, prcpsol = self.get_monthly_climate(heights, year=year)
        mb_month = prcpsol - self.monthly_melt_f * tmelt
        mb_month -= self.bias * SEC_IN_MONTH / SEC_IN_YEAR
        if add_climate:
            return (mb_month / SEC_IN_MONTH / self.rho, t, tmelt,
                    prcp, prcpsol)
        return mb_month / SEC_IN_MONTH / self.rho

    def get_annual_mb(self, heights, year=None, add_climate=False, **kwargs):

        t, tmelt, prcp, prcpsol = self._get_2d_annual_climate(heights, year)
        mb_annual = np.sum(prcpsol - self.monthly_melt_f * tmelt, axis=1)
        mb_annual = (mb_annual - self.bias) / SEC_IN_YEAR / self.rho
        if add_climate:
            return (mb_annual, t.mean(axis=1), tmelt.sum(axis=1),
                    prcp.sum(axis=1), prcpsol.sum(axis=1))
        return mb_annual




class TIModel_Parent(MassBalanceModel):
    """ Parent class that works for different temperature-index models, this is only instanciated
    via the child classes TIModel or TIModel_Sfc_Type. It is just a container with shared code
    to get annual, monthly and daily climate. The actual mass balance can only be computed in child classes as
    there the methods between using surface type and not using sfc type differ.

    Different mass balance modules compatible to OGGM with one flowline,
    so far this is only tested for the elevation-band flowlines

    """

    def __init__(self, gdir, melt_f=melt_f, prcp_fac=2.5, residual=0,
                 mb_type='mb_pseudo_daily', N=100, loop=False,
                 temp_std_const_from_hist = False,
                 grad_type='cte', filename='climate_historical',
                 repeat=False, ys=None, ye=None,
                 t_solid=0, t_liq=2, t_melt=0,
                 default_grad=-0.0065,
                 temp_local_gradient_bounds=[-0.009, -0.003],
                 SEC_IN_YEAR=SEC_IN_YEAR,
                 SEC_IN_MONTH=SEC_IN_MONTH,
                 SEC_IN_DAY=SEC_IN_DAY,
                 baseline_climate=None,
                 input_filesuffix='default',
                 input_filesuffix_h_for_temp_std = '_monthly_W5E5'
                 ):
        """ Initialize.
        Parameters
        ----------
        gdir : GlacierDirectory
            the glacier directory
        melt_f : float
            melt temperature sensitivity factor per month (kg /m² /mth /K),
            need to be prescribed, e.g. such that
            |mean(MODEL_MB)-mean(REF_MB)|--> 0
        prcp_fac : float
            multiplicative precipitation factor, has to be calibrated for each option and
            each climate dataset, default is 2.5,
        residual : float, optional
            default is to use a residual of zero [mm we yr-1 ~ kg/m2/mth]
            Note that this residual is *substracted* from the computed MB.
            do not change this unless you know what you do!
            Indeed: residual = MODEL_MB - REFERENCE_MB.
            ToDO: maybe change the sign?,
            opposite to OGGM "MB terms + residual"
        mb_type: str
            three types: 'mb_pseudo_daily' (default: use temp_std and N percentiles),
            'mb_monthly' (same as default OGGM PastMassBalance),
            'mb_real_daily' (use daily temperature values).
            the mb_type only work if the baseline_climate of gdir is right
        N : int
            number of percentiles used to generate gaussian-like daily
            temperatures from daily std and mean monthly temp (default is N=100)
        loop : bool
            the way how the matrix multiplication is done,
            using np.matmul or a loop(default: False)
            only applied if mb_type is 'mb_pseudo_daily'
            todo: can actually remove loop=True, because this is not anymore used!
             (loop=True is deprecated)
        grad_type : str
            three types of applying the temperature gradient:
            'cte' (default, constant lapse rate, set to default_grad,
                   same as in default OGGM)
            'var_an_cycle' (varies spatially and over annual cycle,
                            but constant over the years)
            'var' (varies spatially & temporally as in the climate files, deprecated!)
        temp_std_const_from_hist : boolean, optional
            whether a variable temp_std is used or one that is constant but just change
            for each month of the year (avg. over calib period 2000-2019). Only is used
            in combination with mb_type='mb_pseudo_daily'. If set to True, mb_pseudo_daily
            is more similar to the approach inside of GloGEM, GloGEMFlow ...
        filename : str, optional
            set it to a different BASENAME if you want to use alternative climate
            data, default is 'climate_historical'
        repeat : bool
            Whether the climate period given by [ys, ye] should be repeated
            indefinitely in a circular way (default is False)
            todo: add when this is used
        ys : int
            The start of the climate period where the MB model is valid
            (default: the period with available data)
        ye : int
            The end of the climate period where the MB model is valid
            (default: the period with available data)
        t_solid : float
            temperature threshold for solid precipitation
            (degree Celsius, default 0)
        t_liq: float
            temperature threshold for liquid precipitation
            (degree Celsius, default 2)
        t_melt : float
            temperature threshold where snow/ice melts
            (degree Celsius, default 0)
        default_grad : float,
            constant lapse rate (temperature gradient, default: -0.0065 m/K)
            if grad_type != cte, then this value is not used
            but instead the changing lapse rate from the climate datasets
        temp_local_gradient_bounds : [float, float],
            if grad_type != cte and the estimated lapse rate (from ERA5)
            does not lie in this range,
            set gradient instead to these minimum, maximum gradients
            (default: [-0.009, -0.003] m/K)
        SEC_IN_YEAR: float
            seconds in a year (default: 31536000s),
            comment: maybe this could be changed
        SEC_IN_MONTH: float
            seconds in a month (default: 2628000s),
            comment: maybe this could be changed as not each
            month has the same amount of seconds,
            in February can be a difference of 8%
        baseline_climate: str
            climate that should be applied, e.g. ERA5dr, WFDE5_CRU, W5E5
        input_filesuffix: str
            if set to 'default', it is set depending on mb_type and
            baseline_climate, but can change this here,
            e.g. change it to '' to work without filesuffix as
            default in oggm PastMassBalance
        input_filesuffix_h_for_temp_std : str
            historical climate_historical filesuffix which was used for calibration,
            only used if  filename is not
            climate_historical and if mb_type = 'mb_pseudo_daily' and if
            temp_std_const_from_hist is True. It is then necessary to get the 12 monthly
            daily std from the historical climate file.

        Attributes
        ----------
        temp_bias : float,
            Add a temperature bias to the time series (default is 0)
            todo: maybe also add it as keyword argument?
        prcp_fac : float, >0
            multiplicative precipitation correction factor (default 2.5)
        melt_f : float, >0
            melt temperature sensitivity factor per month (kg /m² /mth /K),
            need to be prescribed, e.g. such that
            |mean(MODEL_MB)-mean(REF_MB)|--> 0
        """
        # melt_f is only initiated here, and not used in __init__
        # so it does not matter if it is changed
        # just enforce it then it is easier for run_from_climate_data ...
        if mb_type == 'mb_pseudo_daily_fake':
            temp_std_const_from_hist = True
            mb_type = 'mb_pseudo_daily'
        self.mb_type = mb_type

        self._melt_f = melt_f
        if self._melt_f != None and self._melt_f <= 0:
            raise InvalidParamsError('melt_f has to be above zero!')
        if prcp_fac <= 0:
            raise InvalidParamsError('prcp_fac has to be above zero!')
        # as self.prcp is produced by changing prcp_fac
        # we need to update the prcp via prcp_fac by a property (see code after __init__)
        # (reason why there is here only a self._prcp_fac and no self.prcp_fac)
        #  to allow prcp_fac to be changed after instantiation
        #  prescribe the prcp_fac as it is instantiated
        self._prcp_fac = prcp_fac
        # same is true for temp bias
        self._temp_bias = 0.
        self.residual = residual

        # Parameters (that stay cte)
        # comment: those ones are not allowed to be changed after initiation !!!
        # (for my use case, I don't need to change them,
        # but if necessary would need to set properties as well)
        self.t_solid = t_solid
        self.t_liq = t_liq
        self.t_melt = t_melt
        self.N = N

        self.loop = loop
        self.grad_type = grad_type
        # default rho is 900  kg/m3
        # (to convert from kg/m2 into m ice per second=
        self.rho = cfg.PARAMS['ice_density']

        # Public attrs
        self.hemisphere = gdir.hemisphere
        self.repeat = repeat

        self.SEC_IN_YEAR = SEC_IN_YEAR
        self.SEC_IN_MONTH = SEC_IN_MONTH
        self.SEC_IN_DAY = SEC_IN_DAY
        # valid_bounds: The altitudinal bounds where the MassBalanceModel is valid.
        # This is necessary for automated ELA search.
        self.valid_bounds = [-1e4, 2e4]

        # only have one flowline when using elevation bands
        self.fl = gdir.read_pickle('inversion_flowlines')[-1]

        # check if the right climate is used for the right mb_type
        if baseline_climate == None:
            try:
                baseline_climate = gdir.get_climate_info()['baseline_climate_source']
            except:
                baseline_climate = cfg.PARAMS['baseline_climate']
            if baseline_climate != cfg.PARAMS['baseline_climate']:
                raise InvalidParamsError('need to use filesuffixes to define the right climate!')

        if input_filesuffix == 'default':
            if mb_type != 'mb_real_daily':
                input_filesuffix = '_monthly_{}'.format(baseline_climate)
            else:
                input_filesuffix = '_daily_{}'.format(baseline_climate)
        else:
            warnings.warn('you changed the default input_filesuffix of the climate,'
                           'make sure that the default climate (without filesuffix)'
                           'is what you want and is compatible to the chosen temporal resolution!')

        self._input_filesuffix = input_filesuffix
        monthly_climates = ['CRU', 'ERA5dr', 'HISTALP', 'CERA']
        if (self.mb_type == 'mb_real_daily' and
            (baseline_climate in monthly_climates)):
            text = ('wrong climate for mb_real_daily, need to do e.g. '
                    'process_era5_daily_data(gd) to produce daily_ERA5dr'
                    'or process_w5e5_data(gd) for daily_WFDE5_CRU')
            raise InvalidParamsError(text)
        # mb_monthly does not work when daily temperatures are used
        if self.mb_type == 'mb_monthly' and \
                (baseline_climate == 'ERA5_daily' or 'daily' in input_filesuffix):
            text = ('wrong climate for mb_monthly, need to do e.g.'
                    'oggm.shop.ecmwf.process_ecmwf_data(gd, dataset="ERA5dr")')
            raise InvalidParamsError(text)
        # mb_pseudo_daily needs temp_std
        if self.mb_type == 'mb_pseudo_daily' and baseline_climate == 'ERA5_daily':
            text = 'wrong climate for mb_pseudo_daily, need to do e.g. \
            oggm.shop.ecmwf.process_ecmwf_data(gd, dataset = "ERA5dr")'
            raise InvalidParamsError(text)

        # Read climate file
        fpath = gdir.get_filepath(filename, filesuffix=input_filesuffix)

        # todo exectime:
        #  I am using xarray instead of netCDF4, which is slower and I should change this at some point to netCDF4
        with xr.open_dataset(fpath) as xr_nc:
            if self.mb_type == 'mb_real_daily' or self.mb_type == 'mb_monthly':
                # even if there is temp_std inside the dataset, we won't use
                # it for these mb_types
                self.temp_std = np.NaN
            else:
                try:
                    if temp_std_const_from_hist:
                        # really take the temp_std from historical calibration time period!!!
                        # this has to take the historical data even if we look here into gcm data
                        if int(xr_nc.time[0].dt.month.values) != 1 or cfg.PARAMS[f'hydro_month_{self.hemisphere}'] != 1:
                            raise InvalidWorkflowError('temp_std_const_from_hist is only implemented and tested'
                                                       'with hydro_month = 1')
                        if filename == 'climate_historical':
                            xr_nc_h = xr_nc
                        else:
                            fpath_h = gdir.get_filepath('climate_historical',
                                                        filesuffix=input_filesuffix_h_for_temp_std)
                            xr_nc_h = xr.open_dataset(fpath_h)
                        # here we need to have the lenth of used climate file (which can be the gcm file)
                        # so, we need xr_nc and not xr_nc_h !!!
                        n_yrs = len(xr_nc.time.groupby('time.year').mean())
                        temp_std_cte = xr_nc_h['temp_std'].sel(time=slice('2000',
                                                                          '2020')).groupby('time.month').mean()
                        self.temp_std = np.concatenate([temp_std_cte.values.astype(np.float64)] * n_yrs)
                        xr_nc_h.close()
                    else:
                        self.temp_std = xr_nc['temp_std'].values.astype(np.float64)

                except KeyError:
                    text = ('The applied climate has no temp std, do e.g.'
                            'oggm.shop.ecmwf.process_ecmwf_data'
                            '(gd, dataset="ERA5dr")')
                    raise InvalidParamsError(text)

            # goal is to get self.years/self.months in hydro_years
            # (only important for TIModel if we do not use calendar years,
            # for TIModel_Sfc_Type, we can only use calendar years anyways!)

            # get the month where the hydrological month starts
            # as chosen from the gdir climate file
            hydro_month_start = int(xr_nc.time[0].dt.month.values)
            # if we actually use TIModel_Sfc_Type -> hydro_month has to be 1
            if type(self) == TIModel_Sfc_Type:
                if hydro_month_start != 1 or cfg.PARAMS[f'hydro_month_{self.hemisphere}'] != 1:
                    raise InvalidWorkflowError('TIModel_Sfc_Type works only with calendar years, set '
                                               'cfg.PARAMS["hydro_month_nh"] (or sh)'
                                               ' to 1 and process the climate data again')

            if self.mb_type != 'mb_real_daily':
                ny, r = divmod(len(xr_nc.time), 12)
                if r != 0:
                    raise ValueError('Climate data should be N full years')
                # This is where we switch to hydro float year format
                # Last year gives the tone of the hydro year
                self.years = np.repeat(np.arange(xr_nc.time[-1].dt.year-ny+1,
                                                 xr_nc.time[-1].dt.year+1),
                                       12)
                self.months = np.tile(np.arange(1, 13), ny)

            elif self.mb_type == 'mb_real_daily':
                # use pandas to convert month/year to hydro_years
                # this has to be done differently than above because not
                # every month, year has the same amount of days
                pd_test = pd.DataFrame(xr_nc.time.to_series().dt.year.values,
                                       columns=['year'])
                pd_test.index = xr_nc.time.to_series().values
                pd_test['month'] = xr_nc.time.to_series().dt.month.values
                pd_test['hydro_year'] = np.NaN

                if hydro_month_start == 1:
                    # hydro_year corresponds to normal year
                    pd_test.loc[pd_test.index.month >= hydro_month_start,
                                'hydro_year'] = pd_test['year']
                else:
                    pd_test.loc[pd_test.index.month < hydro_month_start,
                                'hydro_year'] = pd_test['year']
                    # otherwise, those days with a month>=hydro_month_start
                    # belong to the next hydro_year
                    pd_test.loc[pd_test.index.month >= hydro_month_start,
                                'hydro_year'] = pd_test['year']+1
                # month_hydro is 1 if it is hydro_month_start
                month_hydro = pd_test['month'].values+(12-hydro_month_start+1)
                month_hydro[month_hydro > 12] += -12
                pd_test['hydro_month'] = month_hydro
                pd_test = pd_test.astype('int')
                self.years = pd_test['hydro_year'].values
                ny = self.years[-1] - self.years[0]+1
                self.months = pd_test['hydro_month'].values
            # Read timeseries and correct it
            self.temp = xr_nc['temp'].values.astype(np.float64) + self._temp_bias
            # this is prcp computed by instantiation
            # this changes if prcp_fac is updated (see @property)
            self.prcp = xr_nc['prcp'].values.astype(np.float64) * self._prcp_fac

            # lapse rate (temperature gradient)
            if self.grad_type == 'var' or self.grad_type == 'var_an_cycle':
                try:
                    # need this to ensure that gradients are not fill-values
                    xr_nc['gradient'] = xr_nc['gradient'].where(xr_nc['gradient'] < 1e12)
                    grad = xr_nc['gradient'].values.astype(np.float64)
                    # Security for stuff that can happen with local gradients
                    g_minmax = temp_local_gradient_bounds

                    # if gradient is not a number, or positive/negative
                    # infinity, use the default gradient
                    grad = np.where(~np.isfinite(grad), default_grad, grad)

                    # if outside boundaries of default -0.009 and above
                    # -0.003 -> use the boundaries instead
                    grad = clip_array(grad, g_minmax[0], g_minmax[1])

                    if self.grad_type == 'var_an_cycle':
                        # if we want constant lapse rates over the years
                        # that change over the annual cycle, but not over time
                        if self.mb_type == 'mb_real_daily':
                            grad_gb = xr_nc['gradient'].groupby('time.month')
                            grad = grad_gb.mean().values
                            g_minmax = temp_local_gradient_bounds

                            # if gradient is not a number, or positive/negative
                            # infinity, use the default gradient
                            grad = np.where(~np.isfinite(grad), default_grad,
                                            grad)
                            assert np.all(grad < 1e12)
                            # if outside boundaries of default -0.009 and above
                            # -0.003 -> use the boundaries instead
                            grad = clip_array(grad, g_minmax[0], g_minmax[1])

                            stack_grad = grad.reshape(-1, 12)
                            grad = np.tile(stack_grad.mean(axis=0), ny)
                            reps_day1 = xr_nc.time[xr_nc.time.dt.day == 1]
                            reps = reps_day1.dt.daysinmonth
                            grad = np.repeat(grad, reps)

                        else:
                            stack_grad = grad.reshape(-1, 12)
                            grad = np.tile(stack_grad.mean(axis=0), ny)
                except KeyError:
                    text = ('there is no gradient available in chosen climate'
                            'file, try instead e.g. ERA5_daily or ERA5dr e.g.'
                            'oggm.shop.ecmwf.process_ecmwf_data'
                            '(gd, dataset="ERA5dr")')

                    raise InvalidParamsError(text)

            elif self.grad_type == 'cte':
                # if grad_type is chosen cte, we use the default_grad!
                grad = self.prcp * 0 + default_grad
            else:
                raise InvalidParamsError('grad_type can be either cte,'
                                         'var or var_an_cycle')
            self.grad = grad
            self.ref_hgt = xr_nc.ref_hgt
            # if climate dataset has been corrected once again
            # or non corrected reference height!
            try:
                self.uncorrected_ref_hgt = xr_nc.uncorrected_ref_hgt
            except:
                self.uncorrected_ref_hgt = xr_nc.ref_hgt

            self.ys = self.years[0] if ys is None else ys
            self.ye = self.years[-1] if ye is None else ye

        self.fpath = fpath

    @property
    def prcp_fac(self):
        """ prints the _prcp_fac as initiated or changed"""
        return self._prcp_fac

    @prcp_fac.setter
    def prcp_fac(self, new_prcp_fac):
        """ sets new prcp_fac, changes prcp time series
         and if TIModel_Sfc_Type resets the pd_mb and buckets
        """
        if new_prcp_fac <= 0:
            raise InvalidParamsError('prcp_fac has to be above zero!')
        # attention, prcp_fac should not be called here
        # otherwise there is recursion occurring forever...
        # use new_prcp_fac to not get maximum recusion depth error
        self.prcp *= new_prcp_fac / self._prcp_fac

        if type(self) == TIModel_Sfc_Type:
            # if the prcp_fac is set to another value,
            # need to reset pd_mb_annual, pd_mb_monthly
            # and also reset pd_bucket
            self.reset_pd_mb_bucket()

        # update old prcp_fac in order that it can be updated
        # again ...
        self._prcp_fac = new_prcp_fac

    @property
    def melt_f(self):
        """ prints the _melt_f """
        return self._melt_f

    @melt_f.setter
    def melt_f(self, new_melt_f):
        """ sets new melt_f and if TIModel_Sfc_Type resets the pd_mb and buckets
        """
        # first update self._melt_f
        self._melt_f = new_melt_f

        if type(self) == TIModel_Sfc_Type:
            # if the prcp_fac is set to another value,
            # need to reset pd_mb_annual, pd_mb_monthly
            # and also reset pd_bucket
            self.reset_pd_mb_bucket()
            # In addition, we need to recompute here once the melt_f buckets
            # (as they depend on self._melt_f)
            # IMPORTANT: this has to be done AFTER self._melt_f got updated!!!
            self.recompute_melt_f_buckets()

    # same for temp_bias:
    @property
    def temp_bias(self):
        return self._temp_bias

    @temp_bias.setter
    def temp_bias(self, new_temp_bias):
        self.temp += new_temp_bias - self._temp_bias

        if type(self) == TIModel_Sfc_Type:
            # if the prcp_fac is set to another value,
            # need to reset pd_mb_annual, pd_mb_monthly
            # and also reset pd_bucket
            self.reset_pd_mb_bucket()

        # update old temp_bias in order that it can be updated again ...
        self._temp_bias = new_temp_bias

    def historical_climate_qc_mod(self, gdir,
                                  climate_qc_months=3):
        """ Check the "quality" of climate data and correct it if needed.

        Similar to historical_climate_qc from oggm.core.climate but checks
        that climate that is used in TIModels directly

        This forces the climate data to have at least one month of melt
        per year at the terminus of the glacier (i.e. simply shifting
        temperatures up
        when necessary), and at least one month where accumulation is possible
        at the glacier top (i.e. shifting the temperatures down).

        This has a similar effect as introducing a temperature bias

        (deprecated at the moment, i.e. we don't apply it for any workflow)
        """
        if type(self) == TIModel_Sfc_Type:
            # if the prcp_fac is set to another value,
            # need to reset pd_mb_annual, pd_mb_monthly
            # and also reset pd_bucket
            self.reset_pd_mb_bucket()
        # Parameters
        if self.temp_bias != 0:
            raise InvalidParamsError('either use no temp_bias or do no quality'
                                     'check corrections, as they have the '
                                     'same effects!')
        fpath = self.fpath
        grad = self.grad
        # get non-corrected quality check
        ref_hgt = self.uncorrected_ref_hgt
        itemp = self.temp
        temp_m = self.t_melt
        temp_s = (self.t_liq + self.t_solid) / 2
        if ('daily' in self._input_filesuffix):
            # different amount of days per year ...
            d_m = 30
            pass
        else:
            d_m = 1
            ny = len(grad) // 12
            assert ny == len(grad) / 12

        # Geometry data
        fls = gdir.read_pickle('inversion_flowlines')
        heights = np.array([])
        for fl in fls:
            heights = np.append(heights, fl.surface_h)
        top_h = np.max(heights)
        bot_h = np.min(heights)

        # First check - there should be at least "climate_qc_months"
        # month of melt every year
        prev_ref_hgt = ref_hgt
        while True:
            # removed default_grad and uses instead grad!
            ts_bot = itemp + grad * (bot_h - ref_hgt)
            # reshape does not work , because of different amount of days
            # per year ...
            pd_ts = pd.DataFrame({'ts_threshold': ts_bot > temp_m,
                                  'year': self.years})
            ts_bot = pd_ts.groupby('year').sum()['ts_threshold'].values
            # ts_bot = (ts_bot.reshape((ny, 12)) > temp_m).sum(axis=1)
            if np.all(ts_bot >= climate_qc_months * d_m):
                # Ok all good
                break
            # put ref hgt a bit higher so that we warm things a bit
            ref_hgt += 10

        # If we changed this it makes no sense to lower it down again,
        # so resume here:
        if ref_hgt != prev_ref_hgt:
            with utils.ncDataset(fpath, 'a') as nc:
                nc.ref_hgt = ref_hgt
                nc.uncorrected_ref_hgt = prev_ref_hgt
            gdir.add_to_diagnostics('ref_hgt_qc_diff',
                                    int(ref_hgt - prev_ref_hgt))
            # need to save the new ref_hgt
            self.ref_hgt = ref_hgt
            return

        # Second check - there should be at least "climate_qc_months"
        # month of acc every year
        while True:
            # grad instead of default_grad
            ts_top = itemp + grad * (top_h - ref_hgt)
            # reshape does not work , because of different amount of days
            # per year ...
            pd_ts = pd.DataFrame({'ts_threshold': ts_top < temp_s,
                                  'year': self.years})
            ts_top = pd_ts.groupby('year').sum()['ts_threshold'].values
            # ts_top = (ts_top.reshape((ny, 12)) < temp_s).sum(axis=1)
            if np.all(ts_top >= climate_qc_months * d_m):
                # Ok all good
                break
            # put ref hgt a bit lower so that we cold things a bit
            ref_hgt -= 10

        if ref_hgt != prev_ref_hgt:
            with utils.ncDataset(fpath, 'a') as nc:
                nc.ref_hgt = ref_hgt
                nc.uncorrected_ref_hgt = prev_ref_hgt
            gdir.add_to_diagnostics('ref_hgt_qc_diff',
                                    int(ref_hgt - prev_ref_hgt))
            # need to save the new ref_hgt
            self.ref_hgt = ref_hgt
            return

    def _get_climate(self, heights, climate_type, year=None):
        """ Climate information at given heights.

        Note that prcp is corrected with the precipitation factor and that
        all other model biases (temp and prcp) are applied.

        same as in OGGM default except that tempformelt is computed by
        self._get_tempformelt

        Parameters
        -------
        heights : np.array or list
            heights along flowline
        climate_type : str
            either 'monthly' or 'annual', if annual floor of year is used,
            if monthly float year is converted into month and year
        year : float
            float hydro year from what both, year and month, are taken if climate_type is monthly.
            hence year 2000 -> y=2000, m = 1, & year = 2000.09, y=2000, m=2 ...
            which corresponds to the real year 1999 and months October or November
            if hydro year starts in October

        Returns
        -------
        (temp, tempformelt, prcp, prcpsol)
        """

        y, m = floatyear_to_date(year)
        if self.repeat:
            y = self.ys + (y - self.ys) % (self.ye - self.ys + 1)
        if y < self.ys or y > self.ye:
            raise ValueError('year {} out of the valid time bounds: '
                             '[{}, {}]'.format(y, self.ys, self.ye))

        if self.mb_type == 'mb_real_daily' or climate_type == 'annual':
            if climate_type == 'annual':
                # todo: exectime the line code below is "a bit" expensive in TIModel
                pok = np.where(self.years == year)[0]
                if len(pok) < 1:
                    raise ValueError('Year {} not in record'.format(int(year)))
            else:
                pok = np.where((self.years == y) & (self.months == m))[0]
                if len(pok) < 28:
                    warnings.warn('something goes wrong with amount of entries\
                                  per month for mb_real_daily')
        else:
            pok = np.where((self.years == y) & (self.months == m))[0][0]
        # Read time series
        # (already temperature bias and precipitation factor corrected!)
        itemp = self.temp[pok]
        iprcp = self.prcp[pok]
        igrad = self.grad[pok]

        # For each height pixel:
        # Compute temp and tempformelt (temperature above melting threshold)
        heights = np.asarray(heights)
        npix = len(heights)
        if self.mb_type == 'mb_real_daily' or climate_type == 'annual':
            grad_temp = np.atleast_2d(igrad).repeat(npix, 0)
            if len(pok) != 12 and self.mb_type != 'mb_real_daily':
                warnings.warn('something goes wrong with amount of entries'
                              'per year')
            # todo: exectime the line code below is quite expensive in TIModel
            grad_temp *= (heights.repeat(len(pok)).reshape(grad_temp.shape) -
                          self.ref_hgt)
            temp2d = np.atleast_2d(itemp).repeat(npix, 0) + grad_temp

            # temp_for_melt is computed separately depending on mb_type
            # todo: exectime the line code below is quite expensive in TIModel
            temp2dformelt = self._get_tempformelt(temp2d, pok)

            # Compute solid precipitation from total precipitation
            prcp = np.atleast_2d(iprcp).repeat(npix, 0)
            fac = 1 - (temp2d - self.t_solid) / (self.t_liq - self.t_solid)
            # line code below also quite expensive!
            prcpsol = prcp * clip_array(fac, 0, 1)
            return temp2d, temp2dformelt, prcp, prcpsol
        else:
            temp = np.ones(npix) * itemp + igrad * (heights - self.ref_hgt)

            # temp_for_melt is computed separately depending on mb_type
            tempformelt = self._get_tempformelt(temp, pok)

            prcp = np.ones(npix) * iprcp
            fac = 1 - (temp - self.t_solid) / (self.t_liq - self.t_solid)
            prcpsol = prcp * clip_array(fac, 0, 1)

            return temp, tempformelt, prcp, prcpsol

    def get_2d_avg_annual_air_hydro_temp(self, heights, year=None,
                                         avg_climate = False, hs = 15,
                                         hydro_m = 10):
        raise NotImplementedError('not tested and only useful'
                                  ' for estimating refreezing')
        # WIP
        # (I won't work on this for the next half a year, but leave
        # it inside as a template)
        # only computes avg annual air temperature in hydrological years,
        # this is necessary for the refreezing estimate of Woodward et al.1997
        # only tested for NH here
        # IMPORTANT: this does not take into account that months
        # have different amount of days (should I include the weighting as
        # it is inside of PyGEM??? [see: https://github.com/drounce/PyGEM/blob/ac619bd1fd862b93a01068a0887efa2a97478b99/pygem/utils/_funcs.py#L10 )
        # pok = np.where(self.years == year)[0]
        if avg_climate:
            year0 = year - hs
            year1 = year + hs
        else:
            year0 = year
            year1 = year
        if self.mb_type != 'mb_real_daily':
            pok_begin = np.where((self.years == year0-1) & (self.months == hydro_m))[0]
            pok_end = np.where((self.years == year1) & (self.months == hydro_m-1))[0]
            pok_hydro = np.arange(pok_begin, pok_end + 1, 1)
        else:
            pok_begin = np.where((self.years == year0-1) & (self.months == hydro_m))[0][0]
            pok_end = np.where((self.years == year1) & (self.months == hydro_m-1))[0][-1]
            pok_hydro = np.arange(pok_begin, pok_end + 1, 1)
        assert self.months[pok_hydro[0]] == hydro_m
        assert self.months[pok_hydro[-1]] == hydro_m - 1
        if len(pok_hydro) < 1:
            raise ValueError('Year {} not in record'.format(int(year)))
        # Read time series
        # (already temperature bias and precipitation factor corrected!)
        itemp = self.temp[pok_hydro]
        igrad = self.grad[pok_hydro]

        heights = np.asarray(heights)
        npix = len(heights)

        grad_temp = np.atleast_2d(igrad).repeat(npix, 0)
        if self.mb_type != 'mb_real_daily' and hydro_m == 10 and not avg_climate:
            assert np.all(self.months[pok_hydro] == np.array([10, 11, 12,  1,  2,  3,  4,  5,  6,  7,  8,  9]))
        if len(pok_hydro) != 12 and self.mb_type != 'mb_real_daily' and not avg_climate:
            warnings.warn('something goes wrong with amount of entries'
                          'per year')
        grad_temp *= (heights.repeat(len(pok_hydro)).reshape(grad_temp.shape) -
                      self.ref_hgt)
        temp2d = np.atleast_2d(itemp).repeat(npix, 0) + grad_temp

        return temp2d

    def _get_2d_monthly_climate(self, heights, year=None):
        # only works with real daily climate data!
        # (is used by get_monthly_mb of TIModel)
        # comment: as it is not used in TIModel_Sfc_type it could also directly go inside of TIModel ?!
        if self.mb_type == 'mb_real_daily':
            return self._get_climate(heights, 'monthly', year=year)
        else:
            raise InvalidParamsError('_get_2d_monthly_climate works only\
                                     with mb_real_daily as mb_type!!!')

    def get_monthly_climate(self, heights, year=None):
        if self.mb_type == 'mb_real_daily':
            # first get the climate data
            t, tfmelt, prcp, prcpsol = self._get_climate(heights, 'monthly',
                                                         year=year)
            return (t.mean(axis=1), tfmelt.sum(axis=1),
                    prcp.sum(axis=1), prcpsol.sum(axis=1))
        else:
            return self._get_climate(heights, 'monthly', year=year)
            # if it is mb_real_daily, the data has daily resolution (2d array then)

    def get_daily_climate(self, heights, year=None):
        raise NotImplementedError('look at _get_2d_daily_climate instead')

    def _get_2d_annual_climate(self, heights, year):
        return self._get_climate(heights, 'annual', year=year)

    def _get_2d_daily_climate(self, heights, year = None):
        return self._get_climate(heights, 'annual', year=year)

    def _get_tempformelt(self, temp, pok):
        """ Helper function to compute tempformelt to avoid code duplication
        in get_monthly_climate() and _get2d_annual_climate()

        comment: I can't use  _get_tempformelt outside the class, but sometimes this could be useful.
        If using this again outside of this class, need to remove the "self",
        such as for 'mb_climate_on_height' in climate.py, that has no self....
        (would need to change temp, t_melt ,temp_std, mb_type, N, loop)

        Parameters
        -------
            temp: ndarray
                temperature time series
            pok: ndarray
                indices of time series

        Returns
        -------
        (tempformelt)
        """

        tempformelt_without_std = temp - self.t_melt

        # computations change only if 'mb_pseudo_daily' as mb_type!
        if self.mb_type == 'mb_monthly' or self.mb_type == 'mb_real_daily':
            tempformelt = tempformelt_without_std
        elif self.mb_type == 'mb_pseudo_daily':
            itemp_std = self.temp_std[pok]

            # matrix with N values that are distributed around 0
            # showing how much fake 'daily' values vary from the mean
            z_scores_mean = stats.norm.ppf(np.arange(1/self.N-1/(2*self.N),
                                                     1, 1/self.N))

            z_std = np.matmul(np.atleast_2d(z_scores_mean).T,
                              np.atleast_2d(itemp_std))

            # there are two possibilities,
            # not using the loop is most of the times faster
            if self.loop is False:
                # without the loop: but not much faster ..
                tempformelt_daily = np.atleast_3d(tempformelt_without_std).T + \
                                    np.atleast_3d(z_std)
                clip_min(tempformelt_daily, 0, out=tempformelt_daily)
                tempformelt_with_std = tempformelt_daily.mean(axis=0).T
            else:
                shape_tfm = np.shape(tempformelt_without_std)
                tempformelt_with_std = np.full(shape_tfm, np.NaN)
                for h in np.arange(0, np.shape(tempformelt_without_std)[0]):
                    h_tfm_daily_ = np.atleast_2d(tempformelt_without_std[h, :])
                    h_tempformelt_daily = h_tfm_daily_ + z_std
                    clip_min(h_tempformelt_daily, 0, out=h_tempformelt_daily)
                    h_tempformelt_monthly = h_tempformelt_daily.mean(axis=0)
                    tempformelt_with_std[h, :] = h_tempformelt_monthly
            tempformelt = tempformelt_with_std

        else:
            raise InvalidParamsError('mb_type can only be "mb_monthly,\
                                     mb_pseudo_daily or mb_real_daily" ')
        # replace all values below zero to zero
        # todo: exectime this is also quite expensive
        clip_min(tempformelt, 0, out=tempformelt)

        return tempformelt

    def get_annual_climate(self, heights, year=None):
        """Annual climate information at given heights.

        Note that prcp is corrected with the precipitation factor and that
        all other model biases (temp and prcp) are applied.

        same as in OGGM default but only for a single elevation band flowline

        Attention: temperature for melt is either in sum over monthly
        mean or in annual sum (if real_daily)!

        Parameters
        -------
        heights: ndarray
            the altitudes at which the mass-balance will be computed (has to be set)
        year: int
            hydro integer year (has to be set)

        Returns
        -------
        (temp, tempformelt, prcp, prcpsol)
        """
        t, tfmelt, prcp, prcpsol = self._get_2d_annual_climate(heights, year)
        return (t.mean(axis=1), tfmelt.sum(axis=1),
                prcp.sum(axis=1), prcpsol.sum(axis=1))

    def get_specific_winter_mb(self, heights, year=None, #begin_winter_mb=None, end_winter_mb=None,
                               widths=None, add_climate=False, period_from_wgms=False,
                               **kwargs_monthly_mb):
        """outputs specific winter MB in kg/m2. The actual winter time period can be (at the moment) either
        default October 1st to April 30th  or the one as observed by the WGMS
        (different for each glacier and observation year)

        todo: copy this and create a similar get_specific_summer_mb() -
         -> and maybe a get_specific_hydro_mb() (which would be the doing first get_specific_winter_mb
         and then get_specific_summer_mb

        Parameters
        -------
        heights: ndarray
            the altitudes at which the mass-balance will be computed (has to be set)
        year: int
            hydro integer year, (e.g. 2020 means winter 2019/2020),  (has to be set)
        #begin_winter_mb : float or np.array
        #    deprecated at the moment
        #    if several years, has to be an array. Corresponds to yearmonthday of the starting day of the winter MB.
        #    If perfect, should be YYYY1001 .
        #end_winter_mb : float or np.array
        #    deprecated at the moment
        #    if several years, has to be an array. Corresponds to yearmonthday of the end day of the winter MB.
        #    If perfect, should be YYYY0430 .
        widths : ndarray
            widths that correspond to the given heights
            (if not given specific MB is estimated without weighting over the widths which is actually not wanted,
            -> that's why you have to set the widths!)
        add_climate : bool
            default is False. If True, climate (temperature, temp_for_melt, prcp, prcp_solid) are also given as output.
            Prcp and temp_for_melt as sum over winter months, temperature as mean.
        period_from_wgms : bool
            if we compute MB by using the same observed time periods as the WGMS does (hence: time period is different
            for each year and glacier). Default is False (computing instead fixed date winter MB
            on northern Hemisphere, always from October 1st - April 30th).
        **kwargs_monthly_mb:
            todo: do I need that?

        Returns
        -------
        either just specific winter mb , or (spec_winter_mb, temp, tempformelt, prcp, prcpsol). Note that temp. is the
        mean temperature and all other variables are the sum over the winter months. Attention: temperature for melt
        is either in sum over monthly mean winter months or in annual sum (if real_daily)!
        """
        # replace this to oggm-sample-data path and use utils.file_downloader
        if period_from_wgms:
            _, path = utils.get_wgms_files()
            oggm_updated = False
            if oggm_updated:
                _, path = utils.get_wgms_files()
                pd_mb_overview = pd.read_csv(
                    path[:-len('/mbdata')] + '/mb_overview_seasonal_mb_time_periods_20220301.csv',
                    index_col='Unnamed: 0')
            else:
                #path_mbsandbox = MBsandbox.__file__[:-len('/__init__.py')]
                #pd_mb_overview = pd.read_csv(path_mbsandbox + '/data/mb_overview_seasonal_mb_time_periods_20220301.csv',
                #                            index_col='Unnamed: 0')
                #fp = utils.file_downloader('https://cluster.klima.uni-bremen.de/~lschuster/ref_glaciers'+
                #                                '/data/mb_overview_seasonal_mb_time_periods_20220301.csv')
                fp = 'https://cluster.klima.uni-bremen.de/~lschuster/ref_glaciers/data/mb_overview_seasonal_mb_time_periods_20220301.csv'
                pd_mb_overview = pd.read_csv(fp, index_col='Unnamed: 0')
            pd_mb_overview = pd_mb_overview[pd_mb_overview['at_least_5_winter_mb']]

            pd_mb_overview_sel_gdir = pd_mb_overview.loc[pd_mb_overview.rgi_id == self.fl.rgi_id]
            pd_mb_overview_sel_gdir_yr = pd_mb_overview_sel_gdir.loc[pd_mb_overview_sel_gdir.Year == year]
            ### starting period
            m_start = pd_mb_overview_sel_gdir_yr['BEGIN_PERIOD'].astype('datetime64[ns]').iloc[0].month
            d_start = pd_mb_overview_sel_gdir_yr['BEGIN_PERIOD'].astype('datetime64[ns]').iloc[0].day
            m_start_days_in_month = pd_mb_overview_sel_gdir_yr['BEGIN_PERIOD'].astype('datetime64[ns]').iloc[0].days_in_month
            # ratio of 1st month that we want to estimate?
            # if d_start is 1 -> ratio should be 1 --> the entire month should be added to the winter MB
            ratio_m_start = 1 - (d_start-1)/m_start_days_in_month
            ### end period
            m_end = pd_mb_overview_sel_gdir_yr['END_WINTER'].astype('datetime64[ns]').iloc[0].month + 1
            m_end_days_in_month = pd_mb_overview_sel_gdir_yr['END_WINTER'].astype('datetime64[ns]').iloc[0].days_in_month
            d_end = pd_mb_overview_sel_gdir_yr['END_WINTER'].astype('datetime64[ns]').iloc[0].day
            # ratio of last month that we want to estimate?
            # if d_end == m_end_days_in_month, then the entire month should be used
            ratio_m_end = d_end/m_end_days_in_month

        else:
            if self.hemisphere == 'nh':
                m_start = 10
                m_end = 4 + 1  # until end of April
            else:
                # just a bit arbitrarily
                m_start = 4
                m_end = 10 + 1
            ratio_m_start = 1
            ratio_m_end = 1

        if widths is None:
            raise InvalidParamsError('need to set widths to get correct specific winter MB')

        if len(np.atleast_1d(year)) > 1:
            if isinstance(self, TIModel_Sfc_Type):
                year_spinup = np.append(year[0] - 1, year)
                # first pre-compute all MB, then take only those months that are wished!
                self.get_specific_mb(heights, year=year_spinup, widths=widths)
                # they are saved, so computational time should not increase much

            if add_climate:
                out = []
                t_list = []
                tfm_list = []
                prcp_list = []
                prcp_sol_list = []
                for k in np.arange(0, len(year), 1):
                    out_k = self.get_specific_winter_mb(heights, year=year[k],
                                                        widths=widths,
                                                        add_climate=add_climate,
                                                        period_from_wgms=period_from_wgms,
                                                        **kwargs_monthly_mb)
                    out.append(out_k[0])
                    for numi, listi, in enumerate([t_list, tfm_list, prcp_list, prcp_sol_list]):
                        listi.append(out_k[numi+1])
                return (np.asarray(out), np.asarray(t_list), np.asarray(tfm_list),
                        np.asarray(prcp_list), np.asarray(prcp_sol_list))


            else:
                out = [self.get_specific_winter_mb(heights, year=year[k],
                                                   widths=widths,
                                                   add_climate=add_climate,
                                                   period_from_wgms=period_from_wgms,
                                                   **kwargs_monthly_mb) for k in np.arange(0, len(year), 1)]
            return np.asarray(out)

        #if year == 1980 and isinstance(self, TIModel_Sfc_Type):
        #    # need to do a "special" spinup!
        #    for m in np.arange(1, m_start, 1):
        #        floatyr = utils.date_to_floatyear(year-1, m)
        #        self.get_monthly_mb(heights, year=floatyr)
        mbs_winter = 0
        if add_climate:
            t_winter_sum = 0
            tfm_winter_sum = 0
            prcp_winter_sum = 0
            prcp_sol_winter_sum = 0
        yr_changes = m_end < m_start
        if yr_changes:
            m_winter_mb = np.concatenate([np.arange(m_start, 13, 1), np.arange(1, m_end, 1)])
        else:
            m_winter_mb = np.arange(m_start, m_end, 1)
        for m in m_winter_mb:
            if (m in np.arange(m_start, 13, 1)) and (yr_changes):
                floatyr = utils.date_to_floatyear(year-1, m)
            else:
                floatyr = utils.date_to_floatyear(year, m)
            out = self.get_monthly_mb(heights, year=floatyr, add_climate=add_climate)
            if m == m_winter_mb[0]:
                ratio = ratio_m_start
            elif m == m_winter_mb[-1]:
                ratio = ratio_m_end
            else:
                # take the entire months if these
                # are not starting or ending months
                ratio = 1
            if add_climate:
                out, t, tfm, prcp, prcp_sol = out
                t_winter_sum += t * ratio
                tfm_winter_sum += tfm * ratio
                prcp_winter_sum += prcp * ratio
                prcp_sol_winter_sum += prcp_sol * ratio
            mbs_winter += out * ratio
            #mbs_winter += mb_winter_m
        mbs_winter = mbs_winter * self.SEC_IN_MONTH * self.rho
        if add_climate:
            # we took the sum of the temperature of the winter months
            # so now we need to divide by the amount of months
            m_length_corrected = len(m_winter_mb) - 2 + ratio + ratio # use the ratio of the winter MBs
            t_winter_mean = np.array(t_winter_sum)/m_length_corrected
            # need to correct the winter temp. mean
            return (np.average(mbs_winter, weights=widths), t_winter_mean,
                    np.array(tfm_winter_sum), np.array(prcp_winter_sum), np.array(prcp_sol_winter_sum))
        else:
            return np.average(mbs_winter, weights=widths)

   # def get_specific_summer_mb(self, heights, year=None, widths=None, add_climate=False, **kwargs_monthly_mb):
   #     """outputs specific summer MB in kg/m2.
   #     This is not yet tested and only works for northern Hemisphere at the moment.
   #     year corresponds to hydro "integer" year
   #
   #     WORK IN PROCESS -> adapt it if get_specific_winter_mb is corrected the right way!
   #     """
   #     raise NotImplementedError('todo: has to be copied / adapted from get_specific_winter_mb')
   #     m_start = 5 # start in May
   #     m_end = 10  # until end of September
   #     mbs_summer = 0
   #     if widths is None:
   #         raise InvalidParamsError('need to set widths to get correct specific winter MB')
   #     for m in np.arange(m_start, m_end , 1):
   #         floatyr = utils.date_to_floatyear(year, m)
   #         mbs_summer += self.get_monthly_mb(heights, year=floatyr, **kwargs_monthly_mb)
   #     mbs_summer = mbs_summer * self.SEC_IN_MONTH * self.rho
   #     if add_climate:
   #         raise NotImplementedError('TODO: add_climate has to be implemented!')
   #     return np.average(mbs_summer, weights=widths)

    def get_specific_summer_mb(self, heights, year=None,  # begin_winter_mb=None, end_winter_mb=None,
                               widths=None, add_climate=False, period_from_wgms=False,
                               **kwargs_monthly_mb):
        """outputs specific summer MB in kg/m2. The actual summer time period can be (at the moment) either
        default May 1st to end of September  or the one as observed by the WGMS
        (different for each glacier and observation year)

        todo: maybe also create a get_specific_hydro_mb() (which would be the doing first get_specific_winter_mb
         and then get_specific_summer_mb

        Parameters
        -------
        heights: ndarray
            the altitudes at which the mass-balance will be computed (has to be set)
        year: int
            hydro integer year,  (has to be set)
        widths : ndarray
            widths that correspond to the given heights
            (if not given specific MB is estimated without weighting over the widths which is actually not wanted,
            -> that's why you have to set the widths!)
        add_climate : bool
            default is False. If True, climate (temperature, temp_for_melt, prcp, prcp_solid) are also given as output.
            Prcp and temp_for_melt as sum over winter months, temperature as mean.
        period_from_wgms : bool
            if we compute MB by using the same observed time periods as the WGMS does (hence: time period is different
            for each year and glacier). Default is False (computing instead fixed date winter MB
            on northern Hemisphere, always from May 1st - end of September).
        **kwargs_monthly_mb:
            todo: do I need that?

        Returns
        -------
        either just specific summer mb , or (spec_summmer_mb, temp, tempformelt, prcp, prcpsol). Note that temp. is the
        mean temperature and all other variables are the sum over the winter months. Attention: temperature for melt
        is either in sum over monthly mean winter months or in annual sum (if real_daily)!
        """
        # replace this to oggm-sample-data path and use utils.file_downloader
        if period_from_wgms:
            _, path = utils.get_wgms_files()
            oggm_updated = False
            if oggm_updated:
                _, path = utils.get_wgms_files()
                pd_mb_overview = pd.read_csv(
                    path[:-len('/mbdata')] + '/mb_overview_seasonal_mb_time_periods_20220301.csv',
                    index_col='Unnamed: 0')
            else:
                # path_mbsandbox = MBsandbox.__file__[:-len('/__init__.py')]
                # pd_mb_overview = pd.read_csv(path_mbsandbox + '/data/mb_overview_seasonal_mb_time_periods_20220301.csv',
                #                            index_col='Unnamed: 0')
                #fp = utils.file_downloader('https://cluster.klima.uni-bremen.de/~lschuster/ref_glaciers' +
                #                           '/data/mb_overview_seasonal_mb_time_periods_20220301.csv')
                fp = 'https://cluster.klima.uni-bremen.de/~lschuster/ref_glaciers/data/mb_overview_seasonal_mb_time_periods_20220301.csv'
                pd_mb_overview = pd.read_csv(fp, index_col='Unnamed: 0')
            pd_mb_overview = pd_mb_overview[pd_mb_overview['at_least_5_winter_mb']]
            pd_mb_overview_sel_gdir = pd_mb_overview.loc[pd_mb_overview.rgi_id == self.fl.rgi_id]
            pd_mb_overview_sel_gdir_yr = pd_mb_overview_sel_gdir.loc[pd_mb_overview_sel_gdir.Year == year]
            ### starting period
            m_start = pd_mb_overview_sel_gdir_yr['END_WINTER'].astype('datetime64[ns]').iloc[0].month
            d_start = pd_mb_overview_sel_gdir_yr['END_WINTER'].astype('datetime64[ns]').iloc[0].day
            m_start_days_in_month = pd_mb_overview_sel_gdir_yr['END_WINTER'].astype('datetime64[ns]').iloc[0].days_in_month
            # ratio of 1st month that we want to estimate?
            # if d_start is 1 -> ratio should be 1 --> the entire month should be added to the winter MB
            ratio_m_start = 1 - (d_start - 1) / m_start_days_in_month
            ### end period
            m_end = pd_mb_overview_sel_gdir_yr['END_PERIOD'].astype('datetime64[ns]').iloc[0].month + 1
            m_end_days_in_month = pd_mb_overview_sel_gdir_yr['END_PERIOD'].astype('datetime64[ns]').iloc[0].days_in_month
            d_end = pd_mb_overview_sel_gdir_yr['END_PERIOD'].astype('datetime64[ns]').iloc[0].day
            # ratio of last month that we want to estimate?
            # if d_end == m_end_days_in_month, then the entire month should be used
            ratio_m_end = d_end / m_end_days_in_month

        else:
            m_start = 5
            m_end = 9 + 1  # until end of April
            ratio_m_start = 1
            ratio_m_end = 1

        if widths is None:
            raise InvalidParamsError('need to set widths to get correct specific winter MB')

        if len(np.atleast_1d(year)) > 1:
            if isinstance(self, TIModel_Sfc_Type):
                year_spinup = np.append(year[0] - 1, year)
                # first pre-compute all MB, then take only those months that are wished!
                self.get_specific_mb(heights, year=year_spinup, widths=widths)
                # they are saved, so computational time should not increase much

            if add_climate:
                out = []
                t_list = []
                tfm_list = []
                prcp_list = []
                prcp_sol_list = []
                for k in np.arange(0, len(year), 1):
                    out_k = self.get_specific_summer_mb(heights, year=year[k],
                                                        widths=widths,
                                                        add_climate=add_climate,
                                                        period_from_wgms=period_from_wgms,
                                                        **kwargs_monthly_mb)
                    out.append(out_k[0])
                    for numi, listi, in enumerate([t_list, tfm_list, prcp_list, prcp_sol_list]):
                        listi.append(out_k[numi + 1])
                return (np.asarray(out), np.asarray(t_list), np.asarray(tfm_list),
                        np.asarray(prcp_list), np.asarray(prcp_sol_list))


            else:
                out = [self.get_specific_summer_mb(heights, year=year[k],
                                                   widths=widths,
                                                   add_climate=add_climate,
                                                   period_from_wgms=period_from_wgms,
                                                   **kwargs_monthly_mb) for k in np.arange(0, len(year), 1)]
            return np.asarray(out)

        # if year == 1980 and isinstance(self, TIModel_Sfc_Type):
        #    # need to do a "special" spinup!
        #    for m in np.arange(1, m_start, 1):
        #        floatyr = utils.date_to_floatyear(year-1, m)
        #        self.get_monthly_mb(heights, year=floatyr)
        mbs_summer = 0
        if add_climate:
            t_summer_sum = 0
            tfm_summer_sum = 0
            prcp_summer_sum = 0
            prcp_sol_summer_sum = 0
        yr_changes = m_end < m_start
        if yr_changes:
            m_summer_mb = np.concatenate([np.arange(m_start, 13, 1), np.arange(1, m_end, 1)])
        else:
            m_summer_mb = np.arange(m_start, m_end, 1)
        for m in m_summer_mb:
            if (m in np.arange(m_start, 13, 1)) and (yr_changes):
                floatyr = utils.date_to_floatyear(year - 1, m)
            else:
                floatyr = utils.date_to_floatyear(year, m)
            out = self.get_monthly_mb(heights, year=floatyr, add_climate=add_climate)
            if m == m_summer_mb[0]:
                ratio = ratio_m_start
            elif m == m_summer_mb[-1]:
                ratio = ratio_m_end
            else:
                # take the entire months if these
                # are not starting or ending months
                ratio = 1
            if add_climate:
                out, t, tfm, prcp, prcp_sol = out
                t_summer_sum += t * ratio
                tfm_summer_sum += tfm * ratio
                prcp_summer_sum += prcp * ratio
                prcp_sol_summer_sum += prcp_sol * ratio
            mbs_summer += out * ratio
            # mbs_winter += mb_winter_m
        mbs_summer = mbs_summer * self.SEC_IN_MONTH * self.rho
        if add_climate:
            # we took the sum of the temperature of the winter months
            # so now we need to divide by the amount of months
            m_length_corrected = len(m_summer_mb) - 2 + ratio + ratio  # use the ratio of the winter MBs
            t_winter_mean = np.array(t_summer_sum) / m_length_corrected
            # need to correct the winter temp. mean
            return (np.average(mbs_summer, weights=widths), t_winter_mean,
                    np.array(tfm_summer_sum), np.array(prcp_summer_sum), np.array(prcp_sol_summer_sum))
        else:
            return np.average(mbs_summer, weights=widths)


class TIModel(TIModel_Parent):
    """ Temperature-Index model without surface type distinction (but different climate resolution
    and temperature lapse rate options (child class of TIModel_Parent)"""

    def __init__(self, *args, **kwargs):
        # same as those from TIModel_Parent
        super().__init__(*args, **kwargs)

    def get_monthly_mb(self, heights, year=None, add_climate=False,
                       **kwargs):
        """ computes annual mass balance in m of ice per second!

        Attention year is here in hydro float year

        year has to be given as float hydro year from what the month is taken,
        hence year 2000 -> y=2000, m = 1, & year = 2000.09, y=2000, m=2 ...
        which corresponds to the real year 1999 and months October or November
        if hydro year starts in October
        """
        # todo: can actually remove **kwargs???
        # comment: get_monthly_mb and get_annual_mb are only different
        #  to OGGM default for mb_real_daily

        if self.mb_type == 'mb_real_daily':
            # get 2D values, dependencies on height and time (days)
            out = self._get_2d_monthly_climate(heights, year)
            t, temp2dformelt, prcp, prcpsol = out
            # (days per month)
            # dom = 365.25/12  # len(prcpsol.T)
            fact = 12/365.25
            # attention, I should not use the days of years as the melt_f is
            # per month ~mean days of that year 12/daysofyear
            # to have the same unit of melt_f, which is
            # the monthly temperature sensitivity (kg /m² /mth /K),
            mb_daily = prcpsol - self.melt_f * temp2dformelt * fact

            mb_month = np.sum(mb_daily, axis=1)
            # more correct than using a mean value for days in a month
            #warnings.warn('there might be a problem with SEC_IN_MONTH'
            #              'as February changes amount of days inbetween the years'
            #              ' see test_monthly_glacier_massbalance()')
        else:
            # get 1D values for each height, no dependency on days
            t, temp2dformelt, prcp, prcpsol = self.get_monthly_climate(heights, year=year)
            mb_month = prcpsol - self.melt_f * temp2dformelt

        # residual is in mm w.e per year, so SEC_IN_MONTH ... but mb_month
        # should be per month!
        mb_month -= self.residual * self.SEC_IN_MONTH / self.SEC_IN_YEAR
        # this is for mb_pseudo_daily otherwise it gives the wrong shape
        mb_month = mb_month.flatten()
        if add_climate:
            if self.mb_type == 'mb_real_daily':
                # for run_with_hydro want to get monthly output (sum of daily),
                # if we want daily output in run_with_hydro need to directly use get_daily_mb()
                prcp = prcp.sum(axis=1)
                prcpsol = prcpsol.sum(axis=1)
                t = t.mean(axis=1)
                temp2dformelt = temp2dformelt.sum(axis=1)
            if self.mb_type == 'mb_pseudo_daily':
                temp2dformelt = temp2dformelt.flatten()
            return (mb_month / SEC_IN_MONTH / self.rho, t, temp2dformelt,
                    prcp, prcpsol)
        # instead of SEC_IN_MONTH, use instead len(prcpsol.T)==daysinmonth
        return mb_month / self.SEC_IN_MONTH / self.rho

    def get_annual_mb(self, heights, year=None, add_climate=False,
                      **kwargs):
        """ computes annual mass balance in m of ice per second !
        Attention: temperature for melt of add_climate
        is either in sum over monthly mean or in annual sum (if real_daily)!
        """
        # todo: can actually remove **kwargs???
        # comment: get_monthly_mb and get_annual_mb are only different
        # to OGGM default for mb_real_daily

        t, temp2dformelt, prcp, prcpsol = self._get_2d_annual_climate(heights,
                                                                   year)
        # *12/daysofthisyear in order to have the same unit of melt_f, which
        # is the monthly temperature sensitivity (kg /m² /mth /K),
        if self.mb_type == 'mb_real_daily':
            # in this case we have the temp2dformelt for each day
            # but self.melt_f is in per month -> divide trough days/month
            # more correct than using a mean value for days in a year
            fact = 12/365.25
            # len(prcpsol.T): make it more consistent as melt_f is described
            # per month independent of which month it is ...
        else:
            fact = 1  # eventually correct here with 365.25
        # todo: exectime -> this line code is quite expensive in TIModel (e.g. ~12% of computation time)
        mb_annual = np.sum(prcpsol - self.melt_f * temp2dformelt*fact,
                           axis=1)
        # and also this one is expensive
        mb_annual = (mb_annual - self.residual) / self.SEC_IN_YEAR / self.rho
        if add_climate:
            # for run_with_hydro, want climate as sum over year (for prcp...)
            return (mb_annual, t.mean(axis=1), temp2dformelt.sum(axis=1),
                    prcp.sum(axis=1), prcpsol.sum(axis=1))
        return mb_annual

    def get_daily_mb(self, heights, year=None,
                     add_climate=False):
        """computes daily mass balance in m of ice per second

        attention this accounts as well for leap years, hence
        doy are not 365.25 as in get_annual_mb but the amount of days the year
        has in reality!!! (needed for hydro model of Sarah Hanus)

        year has to be given as float hydro year from what the month is taken,
        hence year 2000 -> y=2000, m = 1, & year = 2000.09, y=2000, m=2 ...
        which corresponds to the real year 1999 and months October or November
        if hydro year starts in October

        (implemented in order that Sarah Hanus can use daily input and also gets daily output)

        """

        # todo: make this more user friendly
        if type(year) == float:
            raise InvalidParamsError('here year has to be the integer year')
        else:
            pass

        if self.mb_type == 'mb_real_daily':
            # get 2D values, dependencies on height and time (days)
            out = self._get_2d_daily_climate(heights, year)
            t, temp2dformelt, prcp, prcpsol = out
            # days of year
            doy = len(prcpsol.T)  # 365.25
            # assert doy > 360
            # to have the same unit of melt_f, which is
            # the monthly temperature sensitivity (kg /m² /mth /K),
            melt_f_daily = self.melt_f * 12/doy
            mb_daily = prcpsol - melt_f_daily * temp2dformelt

            # mb_month = np.sum(mb_daily, axis=1)
            # more correct than using a mean value for days in a month
            warnings.warn('be cautiuous when using get_daily_mb and test yourself if it does '
                          'what you expect')

            # residual is in mm w.e per year, so SEC_IN_MONTH .. but mb_daily
            # is per day!
            mb_daily -= self.residual * doy
            # this is for mb_daily otherwise it gives the wrong shape
            # mb_daily = mb_month.flatten()
            # instead of SEC_IN_MONTH, use instead len(prcpsol.T)==daysinmonth
            if add_climate:
                # these are here daily values as output for the entire year
                # might need to be changed a bit to be used for run_with_hydro
                return (mb_daily / self.SEC_IN_DAY / self.rho,
                        t, temp2dformelt, prcp, prcpsol)
            return mb_daily / self.SEC_IN_DAY / self.rho
        else:
            raise InvalidParamsError('get_daily_mb works only with'
                                     'mb_real_daily as mb_type!')

    def get_specific_daily_mb(self, heights=None, widths=None, year=None):
        """ returns specific daily mass balance in kg m-2 day

        (implemented in order that Sarah Hanus can use daily input and also gets daily output)
        """
        if len(np.atleast_1d(year)) > 1:
            out = [self.get_specific_daily_mb(heights=heights, widths=widths,
                                              year=yr) for yr in year]
            return np.asarray(out)

        mb = self.get_daily_mb(heights, year=year)
        spec_mb = np.average(mb * self.rho * SEC_IN_DAY, weights=widths, axis=0)
        assert len(spec_mb) > 360
        return spec_mb

class TIModel_Sfc_Type(TIModel_Parent):
    """ Temperature-Index model with surface type distinction using a bucket system
    (child class of TIModel_Parent)

    this works only with calendar years (i.e. hydro_month = 1)

    """

    def __init__(self, gdir, melt_f,
                 melt_f_ratio_snow_to_ice=0.5,
                 melt_f_update='annual',
                 spinup_yrs=6,
                 tau_e_fold_yr=1,  #0.5,
                 melt_f_change='linear',
                 check_availability=True,
                 interpolation_optim=False,
                 hbins=np.NaN,
                 **kwargs):

        # doc_TIModel_Sfc_Type =
        """
        Other terms are equal to TIModel_Parent!
        todo Fabi: need to inherit them here
        The following parameters are initialized specifically only for TIModel_Sfc_Type

        Parameters
        ----------
        melt_f_ratio_snow_to_ice : float
            ratio of snow melt factor to ice melt factor (from 0 to 1),
            if set to 1, it is equal to no surface type distinction
            (default is 0.5 same as in GloGEM)
        melt_f_update : str, default: 'annual'
            'annual' or 'monthly' , how
            often the melt factor should be updated and how
            many buckets exist. If annual it uses 1 snow
            and 5 firn buckets with yearly melt factor updates. If monthly,
            each month the snow is ageing over the amount of spinup years
            (default 6yrs, i.e., 72 months).
        spinup_yrs : int
            amount of years to compute as spinup (if spinup=True in
            get_annual_mb or get_monthly_mb) before computing the actual year.
            Default is 6 years. After these 6 years every bucket has the opportunity
            to be filled (as we have 1 snow bucket and 5 firn buckets, or
            72 monthly buckets)
        tau_e_fold_yr : float
            default is 1, (before it was set to 0.5!!!)
            only used if melt_f_change is 'neg_exp',
            it describes how fast the snow melt
            factor approximates to the ice melt factor via
            melt_f = melt_f_ice + (melt_f_snow - melt_f_ice)* np.exp(-time_yr/tau_e_fold_yr) # s: in months
            do not set to 0, otherwise the melt_f of the first bucket is np.NaN
        melt_f_change : str
            default is linear,
            how the snow melt factor changes to the ice melt factor, either 'linear'
            or 'neg_exp' (see `tau_e_fold_yr` for the equation)
        check_availability: boolean
            default is True, checks if year / month has already been computed, if true it gives that output
            (which comes from either pd_mb_monthly or pd_mb_annual). This ensures to give always the same
            values for a specific year/month and is also faster.
            However, in case of random_climate or constant_climate, we have to set this to False!!!
        interpolation_optim:
            default is False. If it is true, a cte bucket profile (via spinup) is estimated and then kept cte,
            so we just do once the computation of the mass balance for different heights
             & reuse it then again without considering that the surface type changes over time
             ... as we need to use an emulation anyway this is deprecated (but was once used for equilibration runs)
             todo: maybe remove?
        hbins:
            default is np.NaN. Here you can set different height bins for the sfc type distinction method.
            Best is to keep it at the default. Only necessary when using e.g. ConstantMBModel
            todo: which other specific things?
        kwargs:
            those keyword arguments that are equal to TIModel_Parent
            todo: or should I just brutally copy the docs from TIModel_Parent?
        """
        # todo Fabi: how can I best inherit the init docs from TIModel_Parent to be visible as well in TIModel_Sfc_Type
        #self.__init__.__doc__ = super().__init__.__doc__ + doc_TIModel_Sfc_Type

        super().__init__(gdir, melt_f, **kwargs)

        self.hbins = hbins
        self.interpolation_optim = interpolation_optim

        self.tau_e_fold_yr = tau_e_fold_yr
        if melt_f_change == 'neg_exp':
            assert tau_e_fold_yr > 0, "tau_e_fold_yr has to be above zero!"
        self.melt_f_change = melt_f_change
        assert melt_f_change in ['linear', 'neg_exp'], "melt_f_change has to be either 'linear' or 'neg_exp'"
        # ratio of snow melt_f to ice melt_f
        self.melt_f_ratio_snow_to_ice = melt_f_ratio_snow_to_ice
        self.melt_f_update = melt_f_update
        self.spinup_yrs = spinup_yrs
        # amount of bucket and bucket naming depends on melt_f_update:
        if self.melt_f_update == 'annual':
            self.buckets = ['snow', 'firn_yr_1', 'firn_yr_2', 'firn_yr_3',
                            'firn_yr_4', 'firn_yr_5']
        elif self.melt_f_update == 'monthly':
            # for each month over 6 years a bucket-> in total 72!
            self.buckets = np.arange(0, 12 * 6, 1).tolist()
        else:
            raise InvalidParamsError('melt_f_update can either be annual or monthly!')
        # first bucket: if annual it is 'snow', if monthly update it is 0
        self.first_snow_bucket = self.buckets[0]

        self.columns = self.buckets + ['delta_kg/m2']
        # TODO: maybe also include snow_delta_kg/m2, firn_yr_1_delta_kg/m2...
        #  (only important if I add refreezing or a densification scheme)
        # comment: I don't need an ice bucket because this is assumed to be "infinite"
        # (instead just have a 'delta_kg/m2' bucket)

        # save the inversion height to later check if the same height is applied!!!
        self.inv_heights = self.fl.surface_h
        self.check_availability = check_availability

        # container template (has to be updatable -> pd_mb_template is property/setter thing)
        # use the distance_along_flowline as index
        self._pd_mb_template = pd.DataFrame(0, index=self.fl.dx_meter * np.arange(self.fl.nx),
                                            columns=[]) # exectime-> directly addind columns here should be faster
        self._pd_mb_template.index.name = 'distance_along_flowline'

        # bucket template:
        # make a different template for the buckets, because we want directly the right columns inside
        self._pd_mb_template_bucket = pd.DataFrame(0, index=self.fl.dx_meter * np.arange(self.fl.nx),
                                            columns=self.columns)  # exectime-> directly addind columns here should be faster
        self._pd_mb_template_bucket.index.name = 'distance_along_flowline'

        # storage containers for monthly and annual mb
        # columns are the months or years respectively
        # IMPORTANT: those need other columns
        self.pd_mb_monthly = self._pd_mb_template.copy()
        self.pd_mb_annual = self._pd_mb_template.copy()

        # bucket containers with buckets as columns
        pd_bucket = self._pd_mb_template_bucket.copy()
        # exectime comment:  this line is quite expensive -> actually this line can be removed as
        # self._pd_mb_template was defined more clever !!!
        # pd_bucket[self.columns] = 0
        # I don't need a total_kg/m2 because I don't know it anyway
        # as we do this before inversion!



        # storage container for buckets (in kg/m2)
        # columns are the buckets
        # (6*12 or 6 buckets depending if melt_f_update is monthly or annual)
        self.pd_bucket = pd_bucket

    @property
    def melt_f_buckets(self):
        # interpolated from snow melt_f to ice melt_f by using melt_f_ratio_snow_to_ice and tau_e_fold_yr
        # to get the bucket melt_f. Need to do this here and not in init, because otherwise it does not get updated.
        # Or I would need to set melt_f as property / setter function that updates self.melt_f_buckets....

        # exectime too long --> need to make that more efficient:
        #  only "update" melt_f_buckets and recompute them if melt_f
        #  is changed (melt_f_ratio_snow_to_ice, tau_e_fold_yr and amount of buckets do not change after instantiation!)
        # Only if _melt_f_buckets does not exist, compute it out of self.melt_f ... and the other stuff!!
        try:
            return self._melt_f_buckets
        except:
            if self.melt_f_change == 'linear':
                self._melt_f_buckets = dict(zip(self.buckets + ['ice'],
                                                np.linspace(self.melt_f * self.melt_f_ratio_snow_to_ice,
                                                            self.melt_f, len(self.buckets)+1)
                                                ))
            elif self.melt_f_change == 'neg_exp':
                time = np.linspace(0, 6, len(self.buckets + ['ice']))
                melt_f_snow = self.melt_f_ratio_snow_to_ice * self.melt_f
                self._melt_f_buckets = dict(zip(self.buckets + ['ice'],
                                                self.melt_f + (melt_f_snow - self.melt_f) * np.exp(-time/self.tau_e_fold_yr)
                                                ))

            # at the moment we don't allow to set externally the melt_f_buckets but this could be added later ...
            return self._melt_f_buckets

    def recompute_melt_f_buckets(self):
        # This is called always when self.melt_f is updated  (the other self.PARAMS are not changed after instantiation)
        # IMPORTANT:
        # - we need to use it after having updated self._melt_f inside of melt_f setter  (is in TIModel_Sfc_Type)!!!
        # - we need to use self._melt_f -> as this is what is before changed inside of self.melt_f!!!

        # interpolate from snow melt_f to ice melt_f by using melt_f_ratio_snow_to_ice and tau_e_fold_yr
        # to get the bucket melt_f. Need to do this here and not in init, because otherwise it does not get updated.

        # comment exectime: made that more efficient. Now it only "updates" melt_f_buckets and recomputes them if melt_f
        # is changed (melt_f_ratio_snow<_to_ice, tau_e_fold_yr and amount of buckets do not change after instantiation)!
        if self.melt_f_change == 'linear':
            self._melt_f_buckets = dict(zip(self.buckets + ['ice'],
                                            np.linspace(self._melt_f * self.melt_f_ratio_snow_to_ice,
                                                        self._melt_f, len(self.buckets)+1)
                                            ))
        elif self.melt_f_change == 'neg_exp':
            time = np.linspace(0, 6, len(self.buckets + ['ice']))
            melt_f_snow = self.melt_f_ratio_snow_to_ice * self._melt_f
            self._melt_f_buckets = dict(zip(self.buckets + ['ice'],
                                            self._melt_f +
                                            (melt_f_snow - self._melt_f) * np.exp(-time/self.tau_e_fold_yr)
                                            ))

    def reset_pd_mb_bucket(self,
                           init_model_fls='use_inversion_flowline'):
        """ resets pandas mass balance bucket monthly and annual dataframe as well as the bucket dataframe
        (basically removes all years/months and empties the buckets so that everything is as if freshly instantiated)
        It is called when setting new melt_f, prcp_fac, temp_bias, ...
        """
        # comment: do I need to reset sometimes only the buckets,
        #  or only mb_monthly??? -> for the moment I don't think so
        if self.interpolation_optim:
            self._pd_mb_template_bucket = pd.DataFrame(0, index=self.hbins[::-1],
                                                columns=self.columns)
            self._pd_mb_template_bucket.index.name = 'hbins_height'
            self._pd_mb_template = pd.DataFrame(0, index=self.hbins[::-1],
                                                columns=[])
            self._pd_mb_template.index.name = 'distance_along_flowline'

            self.pd_mb_monthly = self._pd_mb_template.copy()
            self.pd_mb_annual = self._pd_mb_template.copy()

            pd_bucket = self._pd_mb_template_bucket.copy()
            # pd_bucket[self.columns] = 0
            self.pd_bucket = pd_bucket
        else:
            if init_model_fls != 'use_inversion_flowline':
                #fls = gdir.read_pickle('model_flowlines')
                self.fl = copy.deepcopy(init_model_fls)[-1]
                self.mod_heights = self.fl.surface_h

                # container templates
                # use the distance_along_flowline as index
                self._pd_mb_template = pd.DataFrame(0, index=self.fl.dx_meter * np.arange(self.fl.nx),
                                                           columns=[])
                self._pd_mb_template.index.name = 'distance_along_flowline'
                self._pd_mb_template_bucket = pd.DataFrame(0, index=self.fl.dx_meter * np.arange(self.fl.nx),
                                                    columns=self.columns)
                self._pd_mb_template_bucket.index.name = 'distance_along_flowline'


            self.pd_mb_monthly = self._pd_mb_template.copy()
            self.pd_mb_annual = self._pd_mb_template.copy()

            pd_bucket = self._pd_mb_template_bucket.copy()
            # this expensive code line below  is not anymore necessary if we define columns directly when creating
            # self._pd_mb_template during instantiation!!! (took up to 15% of
            # pd_bucket[self.columns] = 0
            self.pd_bucket = pd_bucket

    def _add_delta_mb_vary_melt_f(self, heights, year=None,
                                  climate_resol='annual',
                                  add_climate=False):
        """ get the mass balance of all buckets and add/remove the
         new mass change for each bucket

         this is probably where most computational time is needed

         Parameters
         ----
         heights: np.array()
            heights of elevation flowline, at the monent only inversion height works!!!
            Only put heights inside that fit to "distance_along_flowline.
         year: int or float
            if melt_f_update='annual', integer calendar float year,
            if melt_f_update=='monthly', year should be a calendar float year,
            so it corresponds to 1 month of a specific year, i
         climate_resol: 'annual' or 'monthly
            if melt_f_update -> climate_resol has to be always monthly
            but if annual melt_f_update can have either annual climate_resol (if get_annual_mb is used)
            or monthly climate_resol (if get_monthly_mb is used)
         add_climate : bool
            default is False. If True, climate (temperature, temp_for_melt, prcp, prcp_solid) are also given as output
            practical for get_monthly_mb with add_climate=True, as then we do not need to compute climate twice!!!

         """

        # new: get directly at the beginning the pd_bucket and convert it to np.array, at the end it is reconverted and
        # saved again under self.pd_bucket:
        np_pd_bucket = self.pd_bucket.values  # last column is delta_kg/m2

        if self.melt_f_update == 'monthly':
            if climate_resol != 'monthly':
                raise InvalidWorkflowError('Need monthly climate_resol if melt_f_update is monthly')
            if not isinstance(year, float):
                raise InvalidParamsError('Year has to be the calendar float year '
                                         '_add_delta_mb_vary_melt_f with monthly melt_f_update,'
                                         'year needs to be a float')

        #########
        # todo: if I put heights inside that are not fitting to
        #  distance_along_flowline, it can get problematic, I can only check it by testing if
        #  length of the heights are the same as distance along
        #  flowline of pd_bucket dataframe
        #  but the commented check is not feasible as it does not work for all circumstances
        #  however, we check at other points if the height array has at least the right order!
        # if len(heights) != len(self.fl.dis_on_line):
        #    raise InvalidParamsError('length of the heights should be the same as '
        #                             'distance along flowline of pd_bucket dataframe,'
        #                             'use for heights e.g. ...fl.surface_h()')
        ##########

        # check if the first bucket is be empty if:
        condi1 = climate_resol == 'annual' and self.melt_f_update == 'annual'
        # from the last year, all potential snow should be no firn, and from this year, the
        # new snow is not yet added, so snow buckets should be empty
        # or
        condi2 = self.melt_f_update == 'monthly'
        # from the last month, all potential snow should have been update and should be now
        # in the next older month bucket
        # or 1st month with annual update and get_monthly_mb
        # !!! in case of climate_resol =='monthly' but annual update,
        # first_snow_bucket is not empty if mc != 1!!!
        _, mc = floatyear_to_date(float(year))
        condi3 = climate_resol == 'monthly' and self.melt_f_update == 'annual' and mc == 1
        if condi1 or condi2 or condi3:
            # if not np.any(self.pd_bucket[self.first_snow_bucket] == 0):
            # comment exectime: this code here below is faster than the one from above
            np_pd_values_first_snow_bucket = np_pd_bucket[:, 0]  # -> corresponds to first_snow_bucket
            # todo exectime : this still takes long to evaluate !!! is there a betterway ???
            if len(np_pd_values_first_snow_bucket[np_pd_values_first_snow_bucket != 0]) > 0:
                raise InvalidWorkflowError('the first snow buckets should be empty in this use case '
                                           'but it is not, try e.g. to do '
                                           'reset_pd_mb_buckets() before and then rerun the task')

        # Let's do the same as in get_annual_mb of TIModel but with varying melt_f:
        # first get the climate
        if climate_resol == 'annual':
            t, temp2dformelt, prcp, prcpsol = self._get_2d_annual_climate(heights, year)
            if add_climate:
                raise NotImplementedError('Not implemented. Need to check here in the code'
                                      'that the dimensions are right. ')
            # t = t.mean(axis=1)
            # temp2dformelt = temp2dformelt.sum(axis=1)
            # prcp = prcp.sum(axis=1)
            # prcpsol = prcpsol.sum(axis=1)
        elif climate_resol == 'monthly':
            if isinstance(type(year), int):
                raise InvalidWorkflowError('year should be a float with monthly climate resolution')
            # year is here float year, so it corresponds to 1 month of a specific year
            t, temp2dformelt, prcp, prcpsol = self.get_monthly_climate(heights, year)

        # put first all solid precipitation into first snow bucket  (actually corresponds to first part of the loop)
        if climate_resol == 'annual':
            # if melt_f_update annual: treat snow as the amount of solid prcp over that year
            # first add all solid prcp amount to the bucket
            # (the amount of snow that has melted in the same year is taken away in the for loop)
            np_pd_bucket[:, 0] = prcpsol.sum(axis=1)  # first snow bucket should be filled with solid prcp
            # at first, remaining temp for melt energy is all temp for melt (tfm)
            # when looping over the buckets this term will get gradually smaller until the remaining tfm corresponds
            # to the potential ice melt
            remaining_tfm = temp2dformelt.sum(axis=1)
            # delta has to be set to the solid prcp (the melting part comes in during the for loop)
            # self.pd_bucket['delta_kg/m2']
            np_pd_bucket[:, -1] = prcpsol.sum(axis=1)

        elif climate_resol == 'monthly':
            # if self.melt_f_update is  'monthly': snow is the amount
            # of solid prcp over that month, ...
            # SLOW
            # self.pd_bucket[self.first_snow_bucket] = self.pd_bucket[self.first_snow_bucket].values.copy() + prcpsol.flatten() # .sum(axis=1)
            # faster
            np_pd_bucket[:, 0] = np_pd_bucket[:, 0] + prcpsol.flatten()
            remaining_tfm = temp2dformelt.flatten()
            # the last column corresponds to self.pd_bucket['delta_kg/m2'] -> np. is faster
            np_pd_bucket[:, -1] = prcpsol.flatten()

        # need the right unit
        if self.mb_type == 'mb_real_daily':
            fact = 12/365.25
        else:
            fact = 1

        # now do the melting processes for each bucket in a loop:
        # how much tempformelt (tfm) would we need to remove all snow, firn of each bucket
        # in the case of snow it corresponds to tfm to melt all solid prcp.
        # To convert from kg/m2 in the buckets to tfm [K], we use the melt_f values
        # of each bucket accordingly. No need of .copy(), because directly changed by divisor (/)
        # this can actually be done outside the for-loop!!!

        # just get the melt_f_buckets once
        melt_f_buckets = self.melt_f_buckets
        # do not include delta_kg/m2 in pd_buckets, and also do not include melt_f of 'ice'
        tfm_to_melt_b = np_pd_bucket[:, :-1] / (np.array(list(melt_f_buckets.values()))[:-1]*fact)  # in K
        # need to transpose it to have the right shape
        tfm_to_melt_b = tfm_to_melt_b.T

        ### loop:
        # todo Fabi: exectime , can I use another kind of loop -> e.g. "apply" or sth.?
        #  maybe I can convert this first into a np.array then do the loop there and reconvert it at the end
        #  again into a pd.DataFrame for better handling?

        # faster: (old pandas code commented below)
        np_delta_kg_m2 = np_pd_bucket[:, -1]
        for e, b in enumerate(self.buckets):
            # there is no ice bucket !!!

            # now recompute buckets mass:
            # kg/m2 of that bucket that is not lost (that has not melted) in that year / month (i.e. what will
            # stay in that bucket after that month/year -> and gets later on eventually updated -> older)
            # if all lost -> set it to 0
            # -> to get this need to reconvert the tfm energy unit into kg/m2 by using the right melt_factor
            # e.g. at the uppest layers there is new snow added ...
            # todo Fabi: exectime now one of slowest lines (e.g. ~8% of computational time)
            not_lost_bucket = utils.clip_min(tfm_to_melt_b[e] - remaining_tfm, 0) * melt_f_buckets[b] * fact

            # if all of the bucket is removed (not_lost_bucket=0), how much energy (i.e. tfm) is left
            # to remove mass from other buckets?
            # remaining tfm to melt older firn layers -> for the next loop ...
            remaining_tfm = utils.clip_min(remaining_tfm - tfm_to_melt_b[e], 0)
            # exectime: is this faster?
            # find sth. faster

            # in case of ice, the remaining_tfm is only used to update once again delta_kg/m2
            # todo : exectime time Fabi check if I can merge not_lost_bucket and remaining_tfm computation
            #  (so that we don't do the clip_min twice)?
            #  or maybe I should only compute that if not_lost_bucket == 0:

            # amount of kg/m2 lost in this bucket -> this will be added to delta_kg/m2
            # corresponds to: not yet updated total bucket - amount of not lost mass of that bucket
            # comment: the line below was very expensive (Number 1, e.g. 22% of computing time)-> rewrite to numpy?
            # .copy(), not necessary because minus
            # self.pd_bucket['delta_kg/m2'] += not_lost_bucket - self.pd_bucket[b].values
            # comment:  now faster
            # np_delta_kg_m2 = (np_delta_kg_m2 + not_lost_bucket - self.pd_bucket[b].values)
            # but we want it even faster: removed all panda stuff
            np_delta_kg_m2 = (np_delta_kg_m2 + not_lost_bucket - np_pd_bucket[:, e])

            # update pd_bucket with what has not melted from the bucket
            # how can we make this faster?
            # self.pd_bucket[b] = not_lost_bucket
            # new: removed all panda stuff
            # exectime: not so slow anymore (e.g. 3% of computational time)
            np_pd_bucket[:, e] = not_lost_bucket

            # new since autumn 2021: if all the remaining_tfm is zero, then go out of the loop,
            # to make the code faster!
            # e.g. in winter this should mean that we don't loop over all buckets (as tfm is very small)
            # if np.all(remaining_tfm == 0):
            # is this faster?
            if len(remaining_tfm[remaining_tfm != 0]) == 0:
                break

        # we assume that the ice bucket is infinite,
        # so everything that could be melted is included inside of delta_kg/m2
        # that means all the remaining tfm energy is used to melt the infinite ice bucket
        # self.pd_bucket['delta_kg/m2'] = np_delta_kg_m2 -remaining_tfm * self.melt_f_buckets['ice'] * fact
        # use the np.array and recompute delta_kg_m2
        np_pd_bucket[:, -1] = np_delta_kg_m2 -remaining_tfm * melt_f_buckets['ice'] * fact
        # create pd_bucket again at the end
        # todo Fabi: exectime -> this is also now quite time expensive (~8-13%)
        #  but on the other hand, it takes a lot of time to
        #  create a bucket system without pandas at all !
        #  if we would always stay in the self.pd_bucket.values system instead of np_pd_bucket,
        #  we would not need to recreate the pd_bucket, maybe this would be then faster?
        self.pd_bucket = pd.DataFrame(np_pd_bucket, columns=self.pd_bucket.columns.values,
                                      index=self.pd_bucket.index)
        if add_climate:
            return self.pd_bucket, t, temp2dformelt, prcp, prcpsol
        else:
            return self.pd_bucket

        # old:
        # loop_via_pd = False
        # if loop_via_pd:
        #     for e, b in enumerate(self.buckets):
        #         # there is no ice bucket !!!
        #
        #         # now recompute buckets mass:
        #         # kg/m2 of that bucket that is not lost (that has not melted) in that year / month (i.e. what will
        #         # stay in that bucket after that month/year -> and gets later on eventually updated ->older)
        #         # if all lost -> set it to 0
        #         # -> to get this need to reconvert the tfm energy unit into kg/m2 by using the right melt_factor
        #         # e.g. at the uppest layers there is new snow added ...
        #         not_lost_bucket = utils.clip_min(tfm_to_melt_b[e] - remaining_tfm, 0) * self.melt_f_buckets[b] * fact
        #
        #         # if all of the bucket is removed (not_lost_bucket=0), how much energy (i.e. tfm) is left
        #         # to remove mass from other buckets?
        #         # remaining tfm to melt older firn layers -> for the next loop ...
        #         remaining_tfm = utils.clip_min(remaining_tfm - tfm_to_melt_b[e], 0)
        #         # exectime: is this faster?
        #         # find sth. faster
        #
        #         # in case of ice, the remaining_tfm is only used to update once again delta_kg/m2
        #
        #         # amount of kg/m2 lost in this bucket -> this will be added to delta_kg/m2
        #         # corresponds to: not yet updated total bucket - amount of not lost mass of that bucket
        #         # comment: this line was very expensive (Number 1, e.g. 22% of computing time)-> rewrite to numpy?
        #         # self.pd_bucket['delta_kg/m2'] += not_lost_bucket - self.pd_bucket[b].values # .copy(), not necessary because minus
        #         self.pd_bucket['delta_kg/m2'] = (self.pd_bucket['delta_kg/m2'].values
        #                                          + not_lost_bucket - self.pd_bucket[
        #                                              b].values)  # .copy(), not necessary because minus
        #
        #         # is this faster:
        #         # self.pd_bucket['delta_kg/m2'] = not_lost_bucket - self.pd_bucket[b].values # .copy(), not necessary because minus
        #
        #         # update pd_bucket with what has not melted from the bucket
        #         # comment: exectime this line was very expensive (Number 2, e.g. 11% of computing time)->
        #         # how can we make this faster?
        #         self.pd_bucket[b] = not_lost_bucket
        #
        #         # new since autumn 2021: if all the remaining_tfm is zero, then go out of the loop,
        #         # to make the code faster!
        #         # e.g. in winter this should mean that we don't loop over all buckets (as tfm is very small)
        #         if np.all(remaining_tfm == 0):
        #             break
        #     # we assume that the ice bucket is infinite,
        #     # so everything that could be melted is included inside of delta_kg/m2
        #     # that means all the remaining tfm energy is used to melt the infinite ice bucket
        #     self.pd_bucket['delta_kg/m2'] += - remaining_tfm * self.melt_f_buckets['ice'] * fact




    # comment: should this be a setter ??? because no argument ...
    # @update_buckets.setter
    def _update(self):
        """ this is called by get_annual_mb or get_monthly_mb after one year/one month to update
        the buckets as they got older

        if it is called on monthly or annual basis depends if we set melt_f_update to monthly or to annual!


        new: I removed a loop here because I can just rename the bucket column names in order that they are
        one bucket older, then set the first_snow_bucket to 0, and the set pd_bucket[kg/m2] to np.NaN
        just want to shift all buckets from one column to another one .> can do all at once via self.buckets[::-1].iloc[1:] ???
        """
        # first convert pd_bucket to np dataframe -> at the end reconvert to pd.DataFrame
        np_pd_bucket = self.pd_bucket.values

        # this here took sometimes 1.1%
        # if np.any(np.isnan(self.pd_bucket['delta_kg/m2'])):
        # now faster (~0.2%)
        if np.any(np.isnan(np_pd_bucket[:, -1])): # delta_kg/m2 is in the last column !!!
            raise InvalidWorkflowError('the buckets have been updated already, need'
                                       'to add_delta_mb first')

        # exectime: this here took sometimes 7.5% of total computing time !!!
        # if np.any(self.pd_bucket[self.buckets] < 0):
        #    raise ValueError('the buckets should only have positive values')
        # a bit faster now: but can be still ~5%
        # if self.pd_bucket[self.buckets].values.min() < 0:
        # np_pd_values = self.pd_bucket[self.buckets].values
        # buckets are in all columns except the last column!!!
        if len(np_pd_bucket[:, :-1][np_pd_bucket[:, :-1] < 0]) > 0:
            raise ValueError('the buckets should only have positive values')

        # old snow without last bucket and without delta_kg/ms
        # all buckets to update: np_pd_bucket[1:-1]
        # kg/m2 in those buckets before the update: self.bucket[0:-2]
        pd_bucket_val_old = np_pd_bucket[:, :-2]
        len_h = len(self.pd_bucket.index)
        # exectime: <0.1%
        np_updated_bucket = np.concatenate([np.zeros(len_h).reshape(len_h, 1),  # first bucket should be zero
                                            pd_bucket_val_old,
                                            np.full((len_h, 1), np.NaN)],  # kg/m2 bucket should be np.NaN
                                            axis=1)  # , np.NaN(len(self.pd_bucket.index))])
        # as we add a zero array in the front, we don't have any snow more, what has been in the youngest bucket
        # went into the next older bucket and so on!!!

        # recreate pd_bucket:
        # todo Fabi exectime: if we want to save more time would need to restructure all -> slowest line
        #  (9-12% computational time)
        #  if we would always stay in the self.pd_bucket.values system instead of np_pd_bucket,
        #  we would not need to recreate the pd_bucket, maybe this would be then faster?
        self.pd_bucket = pd.DataFrame(np_updated_bucket, columns=self.pd_bucket.columns.values,
                                      index=self.pd_bucket.index)
        return self.pd_bucket

        # OLD stuff
        # other idea -> but did not work!
        # just rename the columns : and add afterwards the snow bucket inside
        # self.pd_bucket = self.pd_bucket.drop(columns=['delta_kg/m2', self.buckets[-1]])
        # problem here: need to have it in the right order (first snow, then firn buckets -> otherwise tests fail)
        # self.pd_bucket.columns = self.buckets[1:]

        # pd_version of above -> was still too slow!
        # exectime: this line was very expensive (sometimes even number 1 with 40% of computing time)
        # self.pd_bucket[self.buckets[1:]] = self.pd_bucket[self.buckets[:-1]].values

        # self.pd_bucket[self.first_snow_bucket] = 0
        # self.pd_bucket['delta_kg/m2'] = np.NaN

        # the new version should work as the old version (e.g. this test here should check it: test_sfc_type_update)
        # the other loop version will be removed
        # else:
        #     for e, b in enumerate(self.buckets[::-1]):
        #         # start with updating oldest snow bucket (in reversed order!= ...
        #         if b != self.first_snow_bucket:
        #             # e.g. update ice by using old firn_yr_5 ...
        #             # @Fabi: do I need the copy here?
        #             self.pd_bucket[b] = self.pd_bucket[self.buckets[::-1][e + 1]].copy()
        #             # we just overwrite it so we don't need to reset it to zero
        #             # self.pd_bucket[self.buckets[::-1][e+1]] = 0 #pd_bucket['firn_yr_4']
        #         elif b == self.first_snow_bucket:
        #             # the first_snow bucket is set to 0 after the update
        #             # (if annual update there is only one snow bucket)
        #             self.pd_bucket[b] = 0
        #     # reset delta_kg/m2 to make clear that it is updated
        #     self.pd_bucket['delta_kg/m2'] = np.NaN

    def get_annual_mb(self, heights, year=None, unit='m_of_ice',
                      bucket_output=False, spinup=True,
                      add_climate=False,
                      auto_spinup=True,
                      **kwargs):
        """
        computes annual mass balance in m of ice per second

        Parameters
        ----------
        heights : np.array
            at the moment works only with inversion heights!
        year: int
            integer CALENDAR year! if melt_f_update='monthly', it will loop over each month.
        unit : str
            default is 'm of ice', nothing else implemented at the moment!
            comment: include option of metre of glacier where the different densities
             are taken into account but in this case would need to add further columns in pd_buckets
             like: snow_delta_kg/m2 ... and so on (only important if I add refreezing or a densification scheme)
        bucket_output: bool (default is False)
            if True, returns as second output the pd.Dataframe with details about
            amount of kg/m2 for each bucket and height grid point,
            set it to True to visualize how the buckets change over time or for testing
            (these are the buckets before they got updated for the next year!)
        spinup : bool (default is True)
            if a spinup is applied to fill up sfc type buckets beforehand
            (default are 6 years, check under self.spinup_yrs)
        add_climate: bool (default is False)
            for run_with_hydro (not yet implemented!)
            todo: implement and test it
        auto_spinup: bool (default is true)
            if True, it automatically computes the spinup years (default is 6) beforehand (however in this case,
            these 6 years at the beginning, although saved in pd_mb_annual, had no spinup...)
            todo: maybe need to add a '_no_spinup' to those that had no spinup?
             or save them in a separate pd.DataFrame?
        **kwargs:
            other stuff passed to get_monthly_mb or to _add_delta_mb_vary_melt_f
            **kwargs necessary to take stuff we don't use (like fls...)

        """

        # when we set spinup_yrs to zero, then there should be no spinup occurring, even if spinup
        # is set to True
        if self.spinup_yrs == 0:
            spinup = False

        if len(self.pd_mb_annual.columns) > 0:
            if self.pd_mb_annual.columns[0] > self.pd_mb_annual.columns[-1]:
                raise InvalidWorkflowError('need to run years in ascending order! Maybe reset first!')
        # dirty check that the given heights are in the right order
        assert heights[0] > heights[-1], "heights should be in descending order!"
        # otherwise pd_buckets does not work ...
        # below would be a more robust test, but this is not compatible to all use cases:
        # np.testing.assert_allclose(heights, self.inv_heights,
        #                           err_msg='the heights should correspond to the inversion heights',
        #                           )
        # np.testing.assert_allclose(heights, self.mod_heights)

        # we just convert to integer without checking ...
        year = int(year)
        # comment: there was some reason why I commented that code again
        # if not isinstance(year, int):
        #    raise InvalidParamsError('Year has to be the full year for get_annual_mb,'
        #                             'year needs to be an integer')
        if year < 1979+self.spinup_yrs:
            # most climatic data starts in 1979, so if we want to
            # get the first 6 years can not use a spinup!!!
            # (or would need to think about sth. else, but not so
            # important right now!!!)
            spinup = False
        if year in self.pd_mb_annual.columns and self.check_availability:
            # print('takes existing annual mb')
            # if that year has already been computed, and we did not change any parameter settings
            # just get the annual_mb without redoing all the computations
            mb_annual = self.pd_mb_annual[year].values
            if bucket_output:
                raise InvalidWorkflowError('if you want to output the buckets, you need to do'
                                           'reset_pd_mb_bucket() and rerun')

            if add_climate:
                t, temp2dformelt, prcp, prcpsol = self._get_2d_annual_climate(heights,
                                                                              year)
                return (mb_annual, t.mean(axis=1), temp2dformelt.sum(axis=1),
                        prcp.sum(axis=1), prcpsol.sum(axis=1))
                # raise NotImplementedError('TODO: add_climate has to be implemented!')
            else:
                return mb_annual
        else:
            # do we need to do the spinup beforehand
            # if any of the default 6 spinup years before was not computed,
            # we need to reset all and do the spinup properly -> (i.e. condi = True)
            if self.melt_f_update == 'annual':
                # so we really need to check if every year exists!!!
                condis = []
                for bef in np.arange(1, self.spinup_yrs, 1):
                    condis.append(int(year - bef) not in self.pd_mb_annual.columns)
                condi = np.any(np.array(condis))
            elif self.melt_f_update == 'monthly':
                try:
                    # check if first and last month of each spinup years before exists
                    condis = []
                    for bef in np.arange(1, self.spinup_yrs, 1):
                        condis.append(date_to_floatyear(year - bef, 1) not in self.pd_mb_monthly.columns)
                        condis.append(date_to_floatyear(year - bef, 12) not in self.pd_mb_monthly.columns)
                    condi = np.any(np.array(condis))
                except:
                    condi = True

            if condi and spinup and auto_spinup:
                # reset and do the spinup:
                # comment: I think we should always reset when
                # doing the spinup and not having computed year==2000
                # problem: the years before year should not be saved up (in get_annual_mb)!
                # (because they are not computed right!!! (they don't have a spinup)
                self.reset_pd_mb_bucket()
                for yr in np.arange(year-self.spinup_yrs, year):
                    self.get_annual_mb(heights, year=yr, unit=unit,
                                       bucket_output=False,
                                       spinup=False, add_climate=False,
                                       auto_spinup=False,
                                       **kwargs)

            if spinup:
                # check if the spinup years had been computed (should be inside of pd_mb_annual)
                # (it should have been computed automatically if auto_spinup=True)
                for bef in np.arange(1, 6, 1):
                    if int(year-bef) not in self.pd_mb_annual.columns.values:
                        raise InvalidWorkflowError('need to do get_annual_mb of all spinup years'
                                                   '(default is 6) beforehand')
            if self.melt_f_update == 'annual':
                self.pd_bucket = self._add_delta_mb_vary_melt_f(heights, year=year,
                                                                climate_resol='annual')
                mb_annual = ((self.pd_bucket['delta_kg/m2'].values
                              - self.residual) / self.SEC_IN_YEAR / self.rho)
                if bucket_output:
                    # copy because we want to output the bucket that is not yet updated!!!
                    pd_bucket = self.pd_bucket.copy()
                # update to one year later ... (i.e. put the snow / firn into the next older bucket)
                self._update()
                # save the annual mb
                # todo Fabi exectime: --> would need to restructure code to remove pd stuff
                self.pd_mb_annual[year] = mb_annual
                # this is done already if melt_f_update is monthly (see get_monthly_mb for m == 12)
            elif self.melt_f_update == 'monthly':
                # will be summed up over each month by getting pd_mb_annual
                # that is set in december (month=12)
                for m in np.arange(1, 13, 1):
                    floatyear = date_to_floatyear(year, m)
                    out = self.get_monthly_mb(heights, year=floatyear, unit=unit,
                                              bucket_output=bucket_output, spinup=spinup,
                                              add_climate=add_climate,
                                              auto_spinup=auto_spinup,
                                              **kwargs)
                    if bucket_output and m == 12:
                        pd_bucket = out[1]
                # get mb_annual that is produced by
                mb_annual = self.pd_mb_annual[year].values
            if bucket_output:
                return mb_annual, pd_bucket
            else:
                if add_climate:
                    t, temp2dformelt, prcp, prcpsol = self._get_2d_annual_climate(heights,
                                                                                  year)
                    return (mb_annual, t.mean(axis=1), temp2dformelt.sum(axis=1),
                            prcp.sum(axis=1), prcpsol.sum(axis=1))
                    #raise NotImplementedError('TODO: add_climate has to be implemented!')
                else:
                    return mb_annual

            #todo
            # if add_climate:
            #    return (mb_annual, t.mean(axis=1), tmelt.sum(axis=1),
            #            prcp.sum(axis=1), prcpsol.sum(axis=1))

    def get_monthly_mb(self, heights, year=None, unit='m_of_ice',
                       bucket_output=False, spinup=True,
                       add_climate=False,
                       auto_spinup=True,
                       **kwargs):
        """
        computes monthly mass balance in m of ice per second!

        year should be the calendar float year,
        so it corresponds to 1 month of a specific year

        Parameters
        ----------
        heights : np.array
            at the moment only works with inversion heights!
        year: int
            year has to be given as CALENDAR float year from what the (integer) year and month is taken,
            hence year 2000 -> y=2000, m = 1, & year = 2000.09, y=2000, m=2 ...
        unit : str
            default is 'm of ice', nothing else implemented at the moment!
            TODO: include option of metre of glacier where the different densities
             are taken into account but in this case would need to add further columns in pd_buckets
             like: snow_delta_kg/m2 ... (only important if I add refreezing or a densification scheme)
        bucket_output: bool (default is False)
            if True, returns as second output the pd.Dataframe with details about
            amount of kg/m2 for each bucket and height grid point,
            set it to True to visualize how the buckets change over time or for testing
            (these are the buckets before they got updated for the next year!)
        spinup : bool (default is True)
            if a spinup is applied to fill up sfc type buckets beforehand
            (default are 6 years, check under self.spinup_yrs), if month is not January,
            also checks if the preceding monts of thar year have been computed
        add_climate: bool (default is False)
            for run_with_hydro or for get_specific_winter_mb to get winter precipitation
            If True, climate (temperature, temp_for_melt, prcp, prcp_solid) are also given as output.
            prcp and temp_for_melt as monthly sum, temp. as mean.
        auto_spinup: bool (default is true)
            if True, it automatically computes the spinup years (default is 6) beforehand and
            all months before the existing year
            (however in this case, these spinup years, although saved in pd_mb_annual, had no spinup...)
            todo: maybe need to add a '_no_spinup' to those that had no spinup?
             or save them in a separate pd.DataFrame?
        **kwargs:
            other stuff passed to get_annual_mb or to _add_delta_mb_vary_melt_f
            todo: not yet necessary, maybe later? **kwargs necessary to take stuff we don't use (like fls...)
        """

        if self.spinup_yrs == 0:
            spinup = False
        if year < 1979+self.spinup_yrs:
            # comment: most climatic data starts in 1979, so if we want to
            # get the first 6 years can not use a spinup!!!
            # (or would need to think about sth. else, but not so
            # important right now!!!)
            spinup = False

        if len(self.pd_mb_monthly.columns) > 1:
            if self.pd_mb_monthly.columns[0] > self.pd_mb_monthly.columns[-1]:
                raise InvalidWorkflowError('need to run months in ascending order! Maybe reset first!')

        assert heights[0] > heights[-1], "heights should be in descending order!"
        # below would be a more robust test, but this is not compatible to all use cases:
        # np.testing.assert_allclose(heights, self.inv_heights,
        #                           err_msg='the heights should correspond to the inversion heights',
        #                           )

        if year in self.pd_mb_monthly.columns and self.check_availability:
            # if that float year has already been computed
            # just get the saved monthly_mb
            # (but don't need to compute something or update pd_buckets)
            # print('takes existing monthly mb')
            mb_month = self.pd_mb_monthly[year].values
            if bucket_output:
                raise InvalidWorkflowError('if you want to output the buckets, you need to do'
                                           'reset_pd_mb_bucket() and rerun')
            if add_climate:
                if isinstance(type(year), int):
                    raise InvalidWorkflowError('year should be a float with monthly climate resolution')
                # year is here float year, so it corresponds to 1 month of a specific year
                t, temp2dformelt, prcp, prcpsol = self.get_monthly_climate(heights, year)
                if self.mb_type == 'mb_pseudo_daily':
                    temp2dformelt = temp2dformelt.flatten()
                return mb_month, t, temp2dformelt, prcp, prcpsol
            else:
                return mb_month
        else:
            # only allow float years
            if not isinstance(year, float):
                raise InvalidParamsError('Year has to be the calendar float year '
                                         'for get_monthly_mb,'
                                         'year needs to be a float')
            # need to somehow check if the months before that were computed as well
            # otherwise monthly mass-balance does not make sense when using sfc type distinction ...
            y, m = floatyear_to_date(year)

            ### spinup
            # find out if the spinup has to be computed
            # this is independent of melt_f_update, always need to reset if the spinup years before
            # were not computed
            try:
                # it is sufficient to check if the first month of that year is there
                # if there is any other problem it will raise an error later anyways
                condi = date_to_floatyear(y - self.spinup_yrs, 1) not in self.pd_mb_monthly.columns
            except:
                condi = True
            # if annual melt_f_update and annual_mb exist from previous years
            # (by doing , get_annual_mb), and there exist not yet the mass-balance for the next year
            # then no reset & new spinup is necessary
            # comment (Feb2022): why did I have here before:
            # condi_add = int(y + 1) in self.pd_mb_annual.columns
            # NEW: don't reset if the years before already exist (by checking pd_mb_annual)
            if condi:
                try:
                    if np.any(self.pd_mb_annual.columns[-self.spinup_yrs:] == np.arange(y - self.spinup_yrs, y, 1)):
                        no_spinup_necessary = True
                    else:
                        no_spinup_necessary = False
                except:
                    no_spinup_necessary = False
                if self.melt_f_update == 'annual' and no_spinup_necessary:
                    condi = False

            if condi and spinup and auto_spinup:
                # reset and do the spinup:
                self.reset_pd_mb_bucket()
                for yr in np.arange(y-self.spinup_yrs, y):
                    self.get_annual_mb(heights, year=yr, unit=unit, bucket_output=False,
                                       spinup=False, add_climate=False,
                                       auto_spinup=False, **kwargs)
                # need to also run the months before our actual wanted month m
                if m > 1:
                    for mb in np.arange(1, m, 1):
                        floatyear_before = date_to_floatyear(y, mb)
                        self.get_monthly_mb(heights, year=floatyear_before,
                                            bucket_output=False,
                                            spinup=False, add_climate=False,
                                            auto_spinup=False, **kwargs)

            if spinup:
                # check if the years before that had been computed
                # (if 2000, it should have been computed above)
                for bef in np.arange(1, 6, 1):
                    if int(y-bef) not in self.pd_mb_annual.columns:
                        raise InvalidWorkflowError('need to do get_monthly_mb of all spinup years '
                                                   '(default is 6) beforehand')
                # check if the months before had been computed for that year
                for mb in np.arange(1, m, 1):
                   floatyear_before = date_to_floatyear(y, mb)
                   if floatyear_before not in self.pd_mb_monthly.columns:
                       raise InvalidWorkflowError('need to get_monthly_mb of each month '
                                                  'of that year beforehand')

            # year is here in float year!
            if add_climate:
                self.pd_bucket, t, temp2dformelt, prcp, prcpsol = self._add_delta_mb_vary_melt_f(heights,
                                                                year=year,
                                                                climate_resol='monthly',
                                                                add_climate=add_climate)
            else:
                self.pd_bucket = self._add_delta_mb_vary_melt_f(heights,
                                                                year=year,
                                                                climate_resol='monthly',
                                                                add_climate=add_climate)

            # ?Fabi: if I want to have it in two lines, can I then  still put the .copy() away?
            #mb_month = self.pd_bucket['delta_kg/m2'].copy().values
            #mb_month -= self.residual * self.SEC_IN_MONTH / self.SEC_IN_YEAR
            mb_month = self.pd_bucket['delta_kg/m2'].values - \
                       self.residual * self.SEC_IN_MONTH / self.SEC_IN_YEAR
            # need to flatten for mb_pseudo_daily otherwise it gives the wrong shape
            mb_month = mb_month.flatten() / self.SEC_IN_MONTH / self.rho
            # save the mass-balance to pd_mb_monthly
            # todo Fabi exectime: -> this line is also quite expensive -> to make it faster would need
            #  to entirely remove the pd_structure!
            self.pd_mb_monthly[year] = mb_month
            self.pd_mb_monthly[year] = mb_month

            # the update happens differently fot the two cases -> so we need to differentiate here:
            if self.melt_f_update == 'annual':
                warnings.warn('get_monthly_mb with annual melt_f_update results in a different summed up annual MB'
                              'than using get_annual_mb. This is because we use bulk estimates in get_annual_mb.'
                              'Only use it when you know what you do!!! ')
                # todo: maybe better to set a NotImplementedError
                # raise NotImplementedError('get_monthly_mb works at the moment '
                #                          'only when melt_f is updated monthly')

                # if annual, only want to update at the end of the year
                if m == 12:
                    # sum up the monthly mb and update the annual pd_mb
                    # but as the mb is  m of ice per second -> use the mean !!!
                    # only works if same height ... and if
                    if bucket_output:
                        pd_bucket = self.pd_bucket.copy()
                    self._update()
                    if int(year) not in self.pd_mb_annual.columns:
                        condi = [int(c) == int(year) for c in self.pd_mb_monthly.columns]
                        # first check if we have all 12 months of the year together
                        if len(self.pd_mb_monthly.loc[:, condi].columns) != 12:
                            raise InvalidWorkflowError('Not all months were computed beforehand,'
                                                       'need to do get_monthly_mb for all months before')
                        # get all columns that correspond to that year and do the mean to get the annual estimate
                        # mean because unit is in m of ice per second
                        self.pd_mb_annual[int(year)] = self.pd_mb_monthly.loc[:, condi].mean(axis=1).values

                if bucket_output and m != 12:
                    pd_bucket = self.pd_bucket.copy()

            elif self.melt_f_update == 'monthly':

                # get the output before the update
                if bucket_output:
                    pd_bucket = self.pd_bucket.copy()
                # need to update it after each month
                self._update()
                # if December, also want to save it to the annual mb
                if m == 12:
                    # sum up the monthly mb and update the annual pd_mb
                    # run this if this year not yet inside or if we don't want to check availability
                    # e.g. for random / constant runs !!!
                    if int(year) not in self.pd_mb_annual.columns or not self.check_availability:
                        condi = [int(c) == int(year) for c in self.pd_mb_monthly.columns]
                        # first check if we have all 12 months of the year together
                        if len(self.pd_mb_monthly.loc[:, condi].columns) != 12:
                            raise InvalidWorkflowError('Not all months were computed beforehand,'
                                                       'need to do get_monthly_mb for all months before')
                        # get all columns that correspond to that year and do the mean to get the annual estimate
                        # mean because unit is in m of ice per second
                        self.pd_mb_annual[int(year)] = self.pd_mb_monthly.loc[:, condi].mean(axis=1).values

            if add_climate and not bucket_output:
                if self.mb_type == 'mb_pseudo_daily':
                    temp2dformelt = temp2dformelt.flatten()
                return mb_month, t, temp2dformelt, prcp, prcpsol
            elif bucket_output:
                return mb_month, pd_bucket
            elif add_climate and bucket_output:
                raise InvalidWorkflowError('either set bucket_output or add_climate to True, not both. '
                                           'Otherwise you have to change sth. in the code!')
            else:
                return mb_month

    def get_daily_mb(self):
        raise NotImplementedError('this has not been implemented with surface type distinction')

    def get_ela(self, year=None, **kwargs):
        """Compute the equilibrium line altitude for a given year.

        copied and adapted from OGGM main -> had to remove the invalid ELA check, as it won't work
        together with sfc type distinction. It still does not work, as the buckets are not made to be applied just
        that


        Parameters
        ----------
        year: float, optional
            the time (in the "hydrological floating year" convention)
        **kwargs: any other keyword argument accepted by self.get_annual_mb
        Returns
        -------
        the equilibrium line altitude (ELA, units: m)
        """
        if len(np.atleast_1d(year)) > 1:
            return np.asarray([self.get_ela(year=yr, **kwargs) for yr in year])

        if self.valid_bounds is None:
            raise ValueError('attribute `valid_bounds` needs to be '
                             'set for the ELA computation.')

        # Check for invalid ELAs
        #b0, b1 = self.valid_bounds
        #if (np.any(~np.isfinite(
        #        self.get_annual_mb([b0, b1], year=year, **kwargs))) or
        #        (self.get_annual_mb([b0], year=year, **kwargs)[0] > 0) or
        #        (self.get_annual_mb([b1], year=year, **kwargs)[0] < 0)):
        #    return np.NaN

        def to_minimize(x):
            return (self.get_annual_mb([x], year=year, **kwargs)[0] *
                    SEC_IN_YEAR * self.rho)

        return optimization.brentq(to_minimize, *self.valid_bounds, xtol=0.1)

