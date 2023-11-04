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
