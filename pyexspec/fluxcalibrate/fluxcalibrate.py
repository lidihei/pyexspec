import numpy as np
from pyexspec.utils import sigma_clip, rvcorr_spec
from pypeit.core import flux_calib
from scipy import interpolate
from astropy import constants, units

class Fcalibrate():

    def __init__(self, waveobs=None, fluxobs=None, fluxobs_err = None,
                 waveobs_std = None, fluxobs_std = None, fluxobs_std_err = None,
                 wavesyn=None, fluxsyn=None):
        '''
        Calibrate observation flux with a synthectic spectum
        waveobs, fluxobs, fluxobs_err are the spectrum of science object, flux in unit of counts
        waveobs_std, fluxobs_std, fluxobs_std_err are the spectrum of the standard star, flux in unit of counts
        wavesyn, fluxsyn are the synthetic spectrum of the standard star, flux in units of erg/s/cm2/A
        '''
        self.waveobs = waveobs
        self.fluxobs = fluxobs
        self.fluxobs_err = fluxobs_err
        if fluxobs_err is not None: self.fluxobs_ivar = 1/fluxobs_err**2
        self.waveobs_std = waveobs_std
        self.fluxobs_std = fluxobs_std
        self.fluxobs_std_err = fluxobs_std_err
        if fluxobs_std_err is not None: self.fluxobs_std_ivar = 1/fluxobs_std_err**2
        self.wavesyn = wavesyn
        self.fluxsyn = fluxsyn

    def rvcorr_syn(self, rv, wave=None):
        if wave is None: wave = self.wavesyn
        self.wavesyn_rvcorr = rvcorr_spec(wave, rv, returnwvl=True)

    def calibrate_flux_Poly1DFitter(self, waveobs=None, fluxobs=None, wavesyn=None, fluxsyn=None, fluxobs_err=None,
                                    num_sigclip =5,  deg=5, min_select=400, verbose=True, lograte=True):
        '''
        using polynomial fit to calibrate the observed flux: twodspec.polynomial.Poly1DFitter
        polyfit = polyfit(fluxsyn/fluxobs)
        flux_calibrate = fluxobs*polyfit
        flux should not be normalized
        paramters:
        -------------
        deg [int] Degree of the fitting polynomial
        '''
        from twodspec.polynomial import Poly1DFitter
        waveobs = self.waveobs if waveobs is None else waveobs
        fluxobs = self.fluxobs if fluxobs is None else fluxobs
        fluxobs_err = self.fluxobs_err if fluxobs_err is None else fluxobs_err
        wavesyn = self.wavesyn if wavesyn is None else wavesyn
        fluxsyn = self.fluxsyn if fluxsyn is None else fluxsyn
        _fluxsyn = np.interp(waveobs, wavesyn, fluxsyn)
        x = waveobs
        rate = _fluxsyn/fluxobs
        if lograte: rate = np.log(rate)
        z = rate.copy()
        indselect = np.ones_like(x, dtype=bool)
        iiter = 0
        while True:
            pf1 = Poly1DFitter(x[indselect], z[indselect], deg=deg, pw=1, robust=False)
            z_pred = pf1.predict(x)
            indselect, iiter, break_bool = sigma_clip(z, z_pred, indselect, iiter,num_sigclip=num_sigclip, min_select=min_select, verbose=verbose)
            if break_bool: break
        yfit = pf1.predict(x)
        self.polyfit_Poly1DFitter = yfit
        if lograte: yfit = np.exp(yfit)
        self.indselect_Poly1DFitter = indselect
        self.fluxobs_calibrate_Poly1DFitter = fluxobs*yfit
        self.polyfunc_Poly1DFitter = pf1
        self.fluxobs2fluxsyn_Poly1DFitter = rate
        self.fluxobs_err_calibrate_Poly1DFitter = np.nan if fluxobs_err is None else fluxobs_err*yfit

    def calibrate_flux_poly1d(self, waveobs=None, fluxobs=None, wavesyn=None, fluxsyn=None, fluxobs_err=None,
                              num_sigclip=5, deg=5, min_select=400, verbose=True, lograte=True):
        ''' 
        using np.poly1d to fit fluxsyn/fluxobs then calibrating the observed flux
        polyfit = polyfit(fluxsyn/fluxobs)
        flux_calibrate = fluxobs*polyfit
        flux should not be normalized
        paramters:
        -------------
        deg [int] Degree of the fitting polynomial
        '''
        waveobs = self.waveobs if waveobs is None else waveobs
        fluxobs = self.fluxobs if fluxobs is None else fluxobs
        fluxobs_err = self.fluxobs_err if fluxobs_err is None else fluxobs_err
        wavesyn = self.wavesyn if wavesyn is None else wavesyn
        fluxsyn = self.fluxsyn if fluxsyn is None else fluxsyn
        _fluxsyn = np.interp(waveobs, wavesyn, fluxsyn)
        x = waveobs.copy()
        rate = _fluxsyn/fluxobs
        if lograte: rate = np.log(rate)  
        z = rate.copy()
        indselect = np.ones_like(x, dtype=bool)
        def getfunc(x, y, deg, rcond=None, full=False, w=None, cov=False, **argkeys):
            zz = np.polyfit(x, y, deg, rcond=rcond, full=full, w=w, cov=cov)
            pfunc1 = np.poly1d(zz)
            return pfunc1
        iiter = 0
        while True:
            pf1 = getfunc(x[indselect], z[indselect], deg)
            z_pred = pf1(x)
            indselect, iiter, break_bool = sigma_clip(z, z_pred, indselect, iiter,num_sigclip=num_sigclip, min_select=min_select, verbose=verbose)
            if break_bool: break
        yfit = pf1(waveobs)
        self.polyfit = yfit
        if lograte: yfit = np.exp(yfit)
        self.fluxobs_calibrate_poly1d = fluxobs*yfit
        self.polyfunc_poly1d = pf1
        self.fluxobs2fluxsyn_poly1d = rate
        self.fluxobs_err_calibrate_poly1d = np.nan if fluxobs_err is None else fluxobs_err*yfit


    def stellar_model(self, Radius, parallax, sptype, V, wave_syn=None, flux_syn=None, Ebv=None):
        """
        Get the Kurucz stellar model for a given apparent magnitude and spectral type of your standard star.
        The function is modified from pypeit.core.flux_calib.stellar_model
        the synthetic sectrum can be generated by using spectrum (https://www.appstate.edu/~grayro/spectrum/spectrum.html)
        note the command: $spectrum -f -1

        This routine first get the temperature, logg, and bolometric luminosity from the Schmidt-Kaler (1982) table
        for the given spectral type. It then find the nearest neighbour in the Kurucz stellar atmosphere ATLAS.
        Finally, the wavelength was converted to Angstrom and the flux density (cgs units) was calculated.

        Parameters
        ----------
        Radius [float] stellar radius in unit of Rsun
        parallax [float] in unit of mas
        sptype [str]  e.g. F2
        V [str] Apparent magnitude of the standard star
        Returns
        -------
        dictionary which contains
        wave: `numpy.ndarray`_
            in uint of angstrom
        flux: `numpy.ndarray`_
            flux density f_lambda (10^17 erg/s/cm2/A cgs units)
        """

        star_lam = self.wavesyn if wave_syn is None else wave_syn
        flux = self.fluxsyn if flux_syn is None else flux_syn
        if Ebv is not None:
            from speedysedfit.reddening import redden
            flux = redden(flux,wave=star_lam,ebv= Ebv,rtype='flux',law='cardelli1989')
        # Grab telluric star parameters
        # log(g) of the Sun
        PYPEIT_FLUX_SCALE = 1e-17
        self.PYPEIT_FLUX_SCALE = PYPEIT_FLUX_SCALE
        # Flux factor (absolute/apparent V mag)
        # Define constants
        parsec = constants.pc.cgs  # 3.086e18
        R_sol = constants.R_sun.cgs  # 6.96e10

        # Distance modulus
        D = parsec * (1000/parallax)
        R = R_sol * Radius

        # Factor converts the kurucz surface flux densities to flux observed on Earth
        flux_factor = (R / D.value) ** 2

        star_flux = flux * flux_factor
        # Generate a dict matching the output of find_standard_file
        std_dict = dict(cal_file='KuruczTelluricModel', name=sptype, Vmag=V, std_ra=None, std_dec=None)
        std_dict['std_source'] = 'KuruczTelluricModel'
        std_dict['wave'] = star_lam * units.AA
        flam_true = star_flux / PYPEIT_FLUX_SCALE * units.erg / units.s / units.cm ** 2 / units.AA
        std_dict['flux'] = flam_true
        self.flam_syn = flam_true.value
        self.lam_syn = star_lam
        return std_dict

    def standard_zeropoint(self, exptime, airmass, longitude, latitude, extinctfilepar='palomarextinct.dat',
                           wave=None, counts=None, counts_ivar=None, counts_mask=None, flam_true = None,
                           maxiter=35, upper=2, lower=2, polyorder=10,
                           balm_mask_wid=5., nresln=4, resolution=2500.,
                           polycorrect=True, polyfunc=False, debug=True
                          ):
        '''
        Generate a sensitivity function based on observed flux and synthectic spectrum of standard star

        Parameters
        ----------
        exptime (float): Exposure time in seconds
        airmass (float): Airmass
        longitude (float): Telescope longitude, used for extinction correction.
        latitude (float): Telescope latitude, used for extinction correction
        extinctfilepar (str):[sensfunc][UVIS][extinct_file] parameter
            Used for extinction correction, e.g. extinctfilepar='palomarextinct.dat
        wave : `numpy.ndarray`_
            wavelength as observed
        wave (`numpy.ndarray`_):
            Wavelength of the star. Shape (nspec,)
        counts (`numpy.ndarray`_):
            Flux (in counts) of the star. Shape (nspec,)
        counts_ivar (`numpy.ndarray`_):
            Inverse variance of the star counts. Shape (nspec,)
        counts_mask (`numpy.ndarray`_):
        Good pixel mask for the counts.
        flam_true : Quantity array
            standard star true flux (erg/s/cm^2/A)
        maxiter : integer
            maximum number of iterations for polynomial fit
        upper : integer
            number of sigma for rejection in polynomial
        lower : integer
            number of sigma for rejection in polynomial
        polyorder : integer
            order of polynomial fit
        balm_mask_wid: float
            Mask parameter for Balmer absorption. A region equal to balm_mask_wid in
            units of angstrom is masked.
        nresln: integer/float
            number of resolution elements between breakpoints
        resolution: integer/float.
            The spectral resolution.  This paramters should be removed in the
            future. The resolution should be estimated from spectra directly.
        debug : bool
            if True shows some dubugging plots

        Returns
        -------
        zeropoint_data: `numpy.ndarray`_ 
            Sensitivity function with same shape as wave (nspec,)
        zeropoint_data_gpm: `numpy.ndarray`_
            Good pixel mask for sensitivity function with same shape as wave (nspec,)
        zeropoint_fit: `numpy.ndarray`_
            Fitted sensitivity function with same shape as wave (nspec,)
        zeropoint_fit_gpm: `numpy.ndarray`_
        Good pixel mask for fitted sensitivity function with same shape as wave (nspec,)
        '''
        wave = self.waveobs_std if wave is None else wave
        counts = self.fluxobs_std if counts is None else counts
        counts_ivar = self.fluxobs_std_ivar if counts_ivar is None else counts_ivar
        counts_mask = np.ones_like(counts, dtype=bool) if counts_mask is None else counts_mask
        if flam_true is None: flam_true = np.interp(wave, self.lam_syn, self.flam_syn)
        mask_bad, mask_recomb, mask_tell =\
                    flux_calib.get_mask(wave, counts, counts_ivar, counts_mask,  mask_hydrogen_lines=True,
                                     mask_helium_lines=False, mask_telluric=True, hydrogen_mask_wid=10.0,  trans_thresh=0.9)
        Nlam_star, Nlam_ivar_star, gpm_star = \
                    flux_calib.counts2Nlam(wave, counts, counts_ivar, counts_mask,
                                    exptime, airmass, longitude, latitude, extinctfilepar)
        zeropoint_data, zeropoint_data_gpm, zeropoint_fit, zeropoint_fit_gpm =\
                    flux_calib.standard_zeropoint(wave, Nlam_star, Nlam_ivar_star, mask_bad, 
                                    flam_true, mask_recomb=mask_recomb,  mask_tell=mask_tell, maxiter=maxiter, upper=upper,
                                    lower=lower, polyorder=polyorder, balm_mask_wid=balm_mask_wid, nresln=nresln, resolution=resolution,
                                    polycorrect=polycorrect, polyfunc=polyfunc, debug=debug)
        self.zeropoint_data = zeropoint_data
        self.zeropoint_data_gpm = zeropoint_data_gpm
        self.zeropoint_fit = zeropoint_fit
        self.zeropoint_fit_gpm = zeropoint_fit_gpm
        return zeropoint_data, zeropoint_data_gpm, zeropoint_fit, zeropoint_fit_gpm

    def standard_zeropoint_Poly1DFitter(self, wave=None, zeropoint_data=None, zeropoint_data_gpm=None, 
                                         num_sigclip=1, deg=10, min_select=400, verbose=True):
        '''
        Using twodspec.polynomial.Poly1DFitter to fit the zeropoint_data
        wave : `numpy.ndarray`_
            wavelength as observed
        zeropoint_data: `numpy.ndarray`_ 
            Sensitivity function with same shape as wave (nspec,)
        zeropoint_data_gpm: `numpy.ndarray`_
            Good pixel mask for sensitivity function with same shape as wave (nspec,)
        '''
        from twodspec.polynomial import Poly1DFitter
        x = self.waveobs_std if wave is None else wave
        z = self.zeropoint_data if zeropoint_data is None else zeropoint_data
        indselect = np.ones_like(x, dtype=bool) if zeropoint_data_gpm is None else zeropoint_data_gpm
        iiter = 0
        while True:
            pf1 = Poly1DFitter(x[indselect], z[indselect], deg=deg, pw=1, robust=False)
            z_pred = pf1.predict(x)
            indselect, iiter, break_bool = sigma_clip(z, z_pred, indselect, iiter,num_sigclip=num_sigclip, min_select=min_select, verbose=verbose)
            if break_bool: break
        yfit = pf1.predict(x)
        self.zeropoint_Poly1DFitter = pf1.predict
        self.zeropoint_Poly1DFit = yfit
        self.zeropoint_fit_indselect_Poly1DFit = indselect

    def get_sensfunc_factor(self, exptime, wave_obs=None, PolyFitter=None, wave_zp=None, zeropoint=None, tellmodel=None, extinct_correct=False,
                         airmass=None, longitude=None, latitude=None, extinctfilepar=None, extrap_sens=False, use_PolyFitter=True, **keywords):
        """
        Get the final sensitivity function factor that will be multiplied into a spectrum in units of counts to flux calibrate it.
        This code interpolates the sensitivity function and can also multiply in extinction and telluric corrections. 
        FLAM = counts*sensfunc_factor
        FLAM_err = counts_err*sensfunc_factor

        Paramters:
        -----------------
        exptime (float): exposure time
        wave_obs (float `numpy.ndarray`_): shape = (nspec,)
        PolyFitter [function]  zeropoint_obs = Poly1DFitter(wave)
        wave_zp (float `numpy.ndarray`_):
           Zerooint wavelength vector shape = (nsens,)
        zeropoint (float `numpy.ndarray`_): shape = (nsens,)
           Zeropoint, i.e. sensitivity function
        tellmodel (float  `numpy.ndarray`_, optional): shape = (nspec,)
           Apply telluric correction if it is passed it. Note this is deprecated.
        extinct_correct (bool, optional)
           If True perform an extinction correction. Deafult = False
        airmass (float, optional):
           Airmass used if extinct_correct=True. This is required if extinct_correct=True
        longitude (float, optional):
            longitude in degree for observatory
            Required for extinction correction
        latitude:
            latitude in degree for observatory
            Required  for extinction correction
        extinctfilepar (str):
                [sensfunc][UVIS][extinct_file] parameter
                Used for extinction correction, e.g. extinctfilepar='palomarextinct.dat
        use_PolyFitter  (bool, optional)
           if True: zeropoint_obs = PolyFitter(wave)
           else: zeropoint_obs = interpolate.interp1d(wave_zp, zeropoint, bounds_error=True)(wave[wave_mask])
        extrap_sens (bool, optional):
            Extrapolate the sensitivity function (instead of crashing out)

        Returns
        -------
        sensfunc_factor: `numpy.ndarray`_
            This quantity is defined to be sensfunc_interp/exptime/delta_wave. shape = (nspec,)

        """
        from pypeit.core.wavecal import wvutils
        wave = self.waveobs if wave_obs is None else wave_obs
        wave_zp = self.waveobs_std if wave_zp is None else wave_zp
        PolyFitter = self.zeropoint_Poly1DFitter if PolyFitter is None else PolyFitter
        wave_mask = wave > 1.0  # filter out masked regions or bad wavelengths
        #delta_wave = wvutils.get_delta_wave(wave, wave_mask)
        if use_PolyFitter:
            zeropoint_obs = PolyFitter(wave)
        else:
            zeropoint_obs = np.zeros_like(wave)
            try:
                zeropoint_obs[wave_mask] \
                        = interpolate.interp1d(wave_zp, zeropoint, bounds_error=True)(wave[wave_mask])
            except ValueError:
                if extrap_sens:
                    zeropoint_obs[wave_mask] \
                            = interpolate.interp1d(wave_zp, zeropoint, bounds_error=False)(wave[wave_mask])
                    msgs.warn("Your data extends beyond the bounds of your sensfunc. You should be "
                              "adjusting the par['sensfunc']['extrap_blu'] and/or "
                              "par['sensfunc']['extrap_red'] to extrapolate further and recreate your "
                              "sensfunc. But we are extrapolating per your direction. Good luck!")
                else:
                    msgs.error("Your data extends beyond the bounds of your sensfunc. " + msgs.newline() +
                               "Adjust the par['sensfunc']['extrap_blu'] and/or "
                               "par['sensfunc']['extrap_red'] to extrapolate further and recreate "
                               "your sensfunc.")

        # This is the S_lam factor required to convert N_lam = counts/sec/Ang to
        # F_lam = 1e-17 erg/s/cm^2/Ang, i.e.  F_lam = S_lam*N_lam
        sensfunc_obs = flux_calib.Nlam_to_Flam(wave, zeropoint_obs)

        # TODO Telluric corrections via this method are deprecated
        # Did the user request a telluric correction?
        if tellmodel is not None:
            # This assumes there is a separate telluric key in this dict.
            msgs.info('Applying telluric correction')
            sensfunc_obs = sensfunc_obs * (tellmodel > 1e-10) / (tellmodel + (tellmodel < 1e-10))


        if extinct_correct:
            if longitude is None or latitude is None:
                msgs.error('You must specify longitude and latitude if we are extinction correcting')
            # Apply Extinction if optical bands
            msgs.info("Applying extinction correction")
            msgs.warn("Extinction correction applied only if the spectra covers <10000Ang.")
            extinct = flux_calib.load_extinction_data(longitude, latitude, extinctfilepar)
            ext_corr = flux_calib.extinction_correction(wave * units.AA, airmass, extinct)
            senstot = sensfunc_obs * ext_corr
        else:
            senstot = sensfunc_obs.copy()

        # senstot is the conversion from N_lam to F_lam, and the division by exptime and delta_wave are to convert
        # the spectrum in counts/pixel into units of N_lam = counts/sec/angstrom
        #sensfunc_factor = senstot/exptime/delta_wave
        return sensfunc_obs

    def calibrateflux(self, exptime, wave_obs, counts, counts_ivar, counts_mask=None,
                         PolyFitter=None, wave_zp=None, zeropoint=None, tellmodel=None, extinct_correct=False,
                         airmass=None, longitude=None, latitude=None, extinctfilepar=None, extrap_sens=False, use_PolyFitter=True, **keywords):
        '''
        Get the final sensitivity function factor that will be multiplied into a spectrum in units of counts to flux calibrate it.
        This code interpolates the sensitivity function and can also multiply in extinction and telluric corrections.

        FLAM = counts*sensfunc_factor
        FLAM_err = counts_err*sensfunc_factor

        Paramters:
        -----------------
        exptime (float): exposure time
        wave_obs (float `numpy.ndarray`_): shape = (nspec,)
        counts  [array]
        counts_ivar [array]
        PolyFitter [function]  zeropoint_obs = Poly1DFitter(wave)
        wave_zp (float `numpy.ndarray`_):
           Zerooint wavelength vector shape = (nsens,)
        zeropoint (float `numpy.ndarray`_): shape = (nsens,)
           Zeropoint, i.e. sensitivity function
        tellmodel (float  `numpy.ndarray`_, optional): shape = (nspec,)
           Apply telluric correction if it is passed it. Note this is deprecated.
        extinct_correct (bool, optional)
           If True perform an extinction correction. Deafult = False
        airmass (float, optional):
           Airmass used if extinct_correct=True. This is required if extinct_correct=True
        longitude (float, optional):
            longitude in degree for observatory
            Required for extinction correction
        latitude:
            latitude in degree for observatory
            Required  for extinction correction
        extinctfilepar (str):
                [sensfunc][UVIS][extinct_file] parameter
                Used for extinction correction, e.g. extinctfilepar='palomarextinct.dat
        use_PolyFitter  (bool, optional)
           if True: zeropoint_obs = PolyFitter(wave)
           else: zeropoint_obs = interpolate.interp1d(wave_zp, zeropoint, bounds_error=True)(wave[wave_mask])
        extrap_sens (bool, optional):
            Extrapolate the sensitivity function (instead of crashing out)
        Returns:
        ----------------
        Flam [array] in unit of erg/s/cm2/A
        Flam _err [array] error of Flam
        gpm_star [array bool] good pixel mask
        '''
        if counts_mask is None: counts_mask = np.ones_like(counts, dtype=bool)
        Nlam_star, Nlam_ivar_star, gpm_star = \
                    flux_calib.counts2Nlam(wave_obs, counts, counts_ivar, counts_mask,
                                    exptime, airmass, longitude, latitude, extinctfilepar)
        sensfunc_factor = self.get_sensfunc_factor(exptime, wave_obs=wave_obs, PolyFitter=PolyFitter, wave_zp=wave_zp, zeropoint=zeropoint,
                         tellmodel=tellmodel, extinct_correct=extinct_correct, airmass=airmass, longitude=longitude,  latitude=latitude, 
                         extinctfilepar=extinctfilepar, extrap_sens=extrap_sens, use_PolyFitter=use_PolyFitter)
        Flam = Nlam_star * sensfunc_factor
        Flam_ivar = Nlam_ivar_star/sensfunc_factor**2
        Flam_err = np.sqrt(1/Flam_ivar)
        return Flam, Flam_err, gpm_star
