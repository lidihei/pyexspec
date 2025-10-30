import numpy as np

def rvcorr_spec(wave, rv, flux=None, fluxerr=None, wave_new=None, left=np.nan, right=np.nan, interp1d=None, returnwvl=False):
    ''' correct spectrum with radial velocity
    parameters:
    ------------
    wave [1d array]
    rv [float] radial velocity in units of km/s
    flux [1d array] = None if (returnwvl is False) else 1D array has the same size of wave
    fluxerr [1d array] = None if (returnwvl is False) else 1D array has the same size of wave 
    wave_new [1d array] should be linear
    returnwvl: [bool] if true, only return the cerrected wave
    returns:
    ----------
    flux_rvc [1d array]
    fluxerr_rvc [1d array]
    '''
    wvl = wave
    flux = flux
    ## Page 71 of An Introduction to Close Binary Stars
    c = 299792.458
    beta = rv/c
    lgwvl = np.log(wvl)
    gamma =(1+beta)/(1-beta)
    _lgwvl = lgwvl + 0.5*np.log(gamma)
    if returnwvl:
        return np.exp(_lgwvl)
    else:
        if wave_new is not None:
           lgwvl = np.log(wave_new)
        if interp1d is None:
           flux_rvc = np.interp(lgwvl, _lgwvl, flux, left=left, right=right)
           err2 = np.interp(lgwvl, _lgwvl, fluxerr**2, left=left, right=right)
        else:
           flux_rvc = interp1d(_lgwvl, flux, kind='linear',fill_value='extrapolate')(lgwvl)
           err2 = interp1d(_lgwvl, fluxerr**2, kind='linear',fill_value='extrapolate')(lgwvl)
        fluxerr_rvc = np.sqrt(err2)
        return flux_rvc, fluxerr_rvc

def vacuum2air(self, wave):
    return wave / (1.0 + 2.735182e-4 + 131.4182 / wave**2 + 2.76249e8 / wave**4)


def get_loglam(R, lam_start, lam_end, N=3):
    ''' get log10(lambda) with a proper sample interval
    parameters:
    ---------------
    R: [float] spectral resolution (BFOSC: R ~ 1600)
    lam_start: [float]: the start of wavelength
    lam_end: [float]: the end of wavelength
    N: [int] oversampling, (typical oversampling: N=3 --> 3pixel=FWHM)
       R = lambda/FWHM --> (log10(lambda))' =1/(lambda*ln(10)) --> FWHM = 1/(R*ln(10))
    returns:
    log10lam: [array] log10(lambda)
    '''
    deltax = 1/R/N/np.log(10)
    log10lam = np.arange(np.log10(lam_start), np.log10(lam_end), deltax)
    return log10lam
