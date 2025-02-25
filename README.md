# OGGM Mass-Balance sandbox


Next generation of OGGM's mass-balance models. Work in process! 

---
**The OGGM mass-balance sandbox does currently not work with OGGM>=v.1.6. More info under # How to install!** 

---

> You are welcome to discover some new mass-balance options such as different climate resolutions or surface
> type distinction. However, we want to make clear that this is 
> work in process and is part of my 
> [PhD project](https://www.uibk.ac.at/acinn/research/ice-and-climate/projects/uncertainties-glacier-smb.html.en):
> - the new mass-balance module from this sandbox is of course less stable, less robust, slower and less documented than the default MBmodel (i.e. the 
> [PastMassBalance](https://docs.oggm.org/en/stable/_modules/oggm/core/massbalance.html#PastMassBalance) 
> class in OGGM) 
> - it has no pre-mass-balance-calibrated `gdirs` 
> - although I try to include [tests](https://github.com/OGGM/massbalance-sandbox/tree/master/MBsandbox/tests)
    as I can, they might be less rigorous as the ones from pure OGGM
> 

> The OGGM mass-balance sandbox has been used by the paper  ["Glacier projections sensitivity to temperature-index model 
choices and calibration strategies" (Schuster et al. 2023)](https://doi.org/10.1017/aog.2023.57). Most options are described there in details. If you use the OGGM mass-balance sandbox, please cite the paper. The scripts to reproduce the analysis of the paper and that use this OGGM mass-balance sandbox are available here: https://github.com/lilianschuster/oggm_mb_sandbox_option_intercomparison.  
> 
> If you can't wait until the OGGM massbalance-sandbox is integrated into OGGM default and want to
> use it already now for your specific OGGM application,
> please contact me first either by: 
> - opening an [issue](https://github.com/OGGM/massbalance-sandbox/issues)
> - writing an [e-mail](mailto:lilian.schuster@uibk.ac.at) to me 
     (Lilian Schuster)
> - or, the easiest way, discuss it inside of our OGGM slack channel

---

At the moment these **options of climate resolution** are available inside `TIModel`:
- to compute degree days:
    - using monthly temperatures ('mb_monthly'), default option in OGGM
    - using monthly temperatures and daily temperature standard deviations (monhtly averages) from the past to generate daily temp. assuming normal distributed data ('mb_pseudo_daily_fake') 
    - using monthly temperatures and daily temp std to generate daily temp. assuming normal distributed data ('mb_pseudo_daily')
    - using daily temperatures ('mb_real_daily')
- temperature lapse rates:
    - using a constant calibrated value independent of location and season (-6.5 K/km, grad_type: cte), default option in OGGM
    - using lapse rates from ERA5 that vary throughout the year and inbetween glacier locations, but that are constant inbetween the years (grad_type: 'var_an_cycle')
    - ( this has not been tested: using lapse rates from ERA5 that vary throughout the year and inbetween glacier locations, different for each year (grad_type: 'var') )

In addition, a **surface type distinction model is included with a bucket system together with a melt_f that varies with age** inside of `TIModel_Sfc_Type`:
- there are two options for how often the melt factor should be updated and how many buckets exist
    - `melt_f_update=annual`
        - If annual, then it uses 1 snow and 5 firn buckets with yearly melt factor updates.
    - `melt_f_update=monthly`:
        -  If monthly, each month the snow is ageing over 6 years (i.e., 72 months -> 72 buckets).
    - the ice bucket is thought as an "infinite" bucket (because we do not know the ice thickness at this model stage)
    - Melt factors are interpolated linearly inbetween the buckets.
      TODO: include non-linear melt factor change as an option!
- there are different option of how and how fast the snow/firn melt factor approximates to the ice melt factor. 
    - `melt_f_change=linear`
        - just a linear change assumed
    - `melt_f_change='neg_exp'`
        - a negative exponential change is assumed via the eq. melt_f = melt_f_ice + (melt_f_snow - melt_f_ice)* np.exp(-time_yr/tau_e_fold_yr)
        - `tau_e_fold_yr` is per default 1 year (but can be changed to another value)
- default is to use a **spinup** of 6 years. So to compute the specific mass balance between 2000 and 2020, with `spinup=True`, the annual mb is computed since 1994 where at first everything is ice, and then it accumulates over the next years, so that in 2000 there is something in each bucket ...

- the ratio of snow melt factor to ice melt factor is set to 0.5 (as in GloGEM, PyGEM, ...) but it can be changed via `melt_f_ratio_snow_to_ice`
    - if we set `melt_f_ratio_snow_to_ice=1` the melt factor is equal for all buckets, hence the results are equal to no surface type distinction (as in `TIModel`)
- `get_annual_mb` and `get_monthly_mb` work as in the OGGM v153 `PastMassBalance` class, however they only accept the height array that corresponds to the inversion height (so no mass-balance elevation feedback can be included at the moment!)
    - that means the given mass-balance ist the mass-balance over the inversion heights (before doing the inversion and so on)
- the buckets are automatically updated when using `get_annual_mb` or `get_monthly_mb` via the `TIModel_Sfc_Type.pd_bucket` dataframe 
- to make sure that we do not compute mass-balance twice and to always have a spin-up of 6 years, we save the mass balance under 
    - `get_annual_mb.pd_mb_annual`: for each year
        - when using `get_monthly_mb` for several years, after computing the December month, the `pd_mb_annual` dataframe is updated
    - `get_annual_mb.pd_mb_monthly`: for each month
        - note that this stays empty if we only use get_annual_mb with annual melt_f_update


All options have been tested with the elevation-band flowlines.

# How to install!
<!-- structure as in https://github.com/fmaussion/scispack and oggm/oggm -->
The OGGM MB sandbox needs OGGMv153 and does currently not work with the latest OGGMv16. It also needs some developments after the OGGM v153 release. Thus, it is best if you install a more recent OGGM development version which is still before OGGM v16 (e.g. "OGGM version: '1.5.4.dev60+g9d17303'": https://github.com/OGGM/oggm/commit/9d173038862f36a21838034da07243bd189ef2d0) by doing:

    $ conda create --name env_mb
    $ source activate env_mb
    $ pip install --no-deps "git+https://github.com/OGGM/oggm.git@9d173038862f36a21838034da07243bd189ef2d0"
    $ git clone https://github.com/OGGM/massbalance-sandbox
    $ cd massbalance-sandbox
    $ pip install -e .
    

Test the installation via pytest while being in the massbalance-sandbox folder, best is if you do :

    $ pytest -v -m "not no_w5e5"
    
(Attention: Just doing `pytest .`, downloads several climate datasets and does example ensemble projections into the future. If you only want to use W5E5 climate data, the tests run with the line above are sufficient)

If you have issues to install the right package versions, you can install the packages dependent on oggm_v153 by using for example the following `.yml` file: https://github.com/OGGM/OGGM-dependency-list/blob/master/Linux-64/oggmdev-1.5.3.202209061450_20221107_py39.yml

The MBsandbox package can be imported in python by

    >>> import MBsandbox

# code inside of MBsandbox

- [mbmod_daily_oneflowline.py](MBsandbox/mbmod_daily_oneflowline.py): 
    - process different climate data (W5E5, WFDE5_CRU, ERA5_daily, W5E5_MSWEP(prcp from MSWEP, temp. from W5E5)),
    - new mass-balance model `TIModel_Parent` with children `TIModel` and `TIModel_Sfc_Type`
- [flowline_TIModel.py](MBsandbox/flowline_TIModel.py): copies of run_from_climate, run_random_climate that are compatible with `TIModel`, not yet tested for `TIModel_Sfc_Type`
- [help_func.py](MBsandbox/help_func.py): helper functions to minimize the bias, optimise standard deviation quotient for reference glaciers, to calibrate the melt factor given the precipitation factor and geodetic observations, and to compute performance statistics
- [tests](MBsandbox/tests): tests for different functions
- [wip](MBsandbox/wip): work in process folder without documentation

# How to use!
There is not a lot of documentation, but here are some example notebooks with explanations inside of the [docs folder](docs). More example scripts of the OGGM massbalance-sandbox are in the https://github.com/lilianschuster/oggm_mb_sandbox_option_intercomparison repository. 
- [how_to_use.ipynb](docs/how_to_use.ipynb) : simple use case that shows how to use different climate resolutions
- [surface_type_distinction/oggm_sfc_type_bucket.ipynb](docs/surface_type_distinction/oggm_sfc_type_bucket.ipynb) : use case of how the surface type distinction with different melt factors works
    - inside of the folder [surface_type_distinction](docs/surface_type_distinction) there are some plots that show how the sfc type distinction and melt_f variation works

- [how_to_daily_input_daily_output.ipynb](docs/how_to_daily_input_daily_output.ipynb) : how to get daily mb output and calibrate with geodetic data

**hydro_compatility**:
- [hydro_compatibility/hydro_compatibility_workflow_daily.ipynb](docs/hydro_compatibility/hydro_compatibility_workflow_daily.ipynb) : workflow how to use MBsandbox with run_with_hydro together with using daily climate data and getting daily runoff data out

**equilibrium runs**
- [equilibrium_runs_OGGM_MBsandbox.ipynb](docs/equilibrium_runs_OGGM_MBsandbox.ipynb) : 
    - analyses climate equilibrium experiments with different MB model options

**other notebooks not directly related to the MBSandbox**:
- [geodetic_mb_calibration_to_volume_changes.ipynb](docs/geodetic_mb_calibration_to_volume_changes.ipynb) : template how to calibrate a glacier by using the geodetic estimates (for the default `PastMassBalanceModel` of OGGMv153!!! )
    - also explains docs/calib_log_fit_pf_distribution_change_monthly_cte_melt_f_minus_1.png
    
- [emulator_bayes_calib](docs/emulator_bayes_calib) :
    - trial notebooks to use Bayesian calibration with PyMC3 -> work in process


