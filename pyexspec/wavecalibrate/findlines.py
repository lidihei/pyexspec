# -*- coding: utf-8 -*-
"""

@author: Jiao

@SONG: RMS =  0.00270124312246
delta_rv = 299792458/5500*0.00270124312246 = 147.23860278870518 m/s

LAMOST: R~1800,  299792.458/6000*3A = 150km/s WCALIB: 10km/s delta_rv = 5km/s
MMT: R~2500,  299792.458/R = 119.9km/s  rms=0.07A, delta_rv = 5km/s
SONG: 1800,  299792.458/6000*3A = 150km/s delta_rv = 5km/s
DBSP: 2500,  299792.458/5000*3A = 150km/s delta_rv = 5km/s
"""

import numpy as np
from astropy import table
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from scipy.optimize import minimize
from scipy.fftpack import fft
from scipy.fftpack import ifft
from pyexspec.fitfunc.function import gaussian_linear_func


class findline():

    def calCCF(self, xcoord, flux, x_template: np.array = None,
               flux_template:np.array = None, step: float = 0.1, show: bool = False):
        '''
        calculate CCF of sample spectrum and template arc lamp spectrum
        parameters:
        -----------------
        xcoord [1D int array] the index coordiates of sample spectrum, generally started with 0 in unit of pixel
        flux [1D array]: flux of sample spectrum
        x_template [1D int array] the index coordiates of the template arc lamp spectrum, generally started with 0 in unit of pixel
        flux_template [1D array] the flux of the template arc lamp spectrum
        step [float]: a step to interpolate the index in unit of pixel
        returns:
        shifts [1D array] shift of the index of CCF in unit of pixel
        CCF [1D array]
        '''
        x_template = self.x_template if x_template is None else x_template
        flux_template = self.flux_template if flux_template is None else flux_template
        xmax = np.min([xcoord[-1], x_template[-1]])
        x = np.arange(xcoord[0], xmax+step, step)
        shifts = x-xmax/2
        fluxr = np.interp(x, xcoord, flux)
        ftmp = np.interp(x, x_template, flux_template)
        tflux = fft(fluxr)
        tftmp = fft(ftmp)
        conj_tflux = np.conj(tflux)
        num = int(len(tftmp)/2)+1
        tmp = abs(ifft(conj_tflux*tftmp))
        ccf = np.hstack((tmp[num:], tmp[:num]))
        self.ccf = ccf
        self.shifts = shifts
        if show:
           fig, axs = plt.subplots(2, 1, figsize=(7, 6))
           plt.sca(axs[0])
           plt.title(f'meidian flux template={np.median(flux_template):.2f}; sample={np.median(flux):.2f} counts')
           plt.plot(xcoord, flux/np.median(flux), label='sample', lw=1)
           plt.plot(x_template, flux_template/np.median(flux_template), label='template', lw=1)
           plt.title(f'meidian flux template={np.median(flux_template):.2f}; sample={np.median(flux):.2f} counts')
           plt.xlabel('x index (pixel)')
           plt.ylabel('Flux/median(Flux)')
           plt.legend()
           plt.sca(axs[1])
           xshift = shifts[np.argmax(ccf)]
           plt.title(f'shift ={xshift}')
           plt.plot(shifts, ccf, lw=1)
           plt.xlabel('x shifts (pixel)')
           plt.ylabel('CCF')
           self.fig_QA_ccf = fig
        return shifts, ccf


def corr_arc(wave_temp, arc_temp, arc_obs, maxshift=100):
    """
    Calculate the shift between arc_obs and arc_temp using correlation.
    Returns the shifted interpolated wavelength array

    Parameters
    ----------
    wave_temp : ndarray
        wavelength template.
    arc_temp : ndarray
        arc template.
    arc_obs : ndarray
        arc observation.
    maxshift : int, optional
        max shift. The default is 100.

    Returns
    -------
    wave_corr : ndarray
        shifted interpolated wavelength array.

    """
    # assert number of orders are the same
    assert arc_obs.shape == arc_temp.shape
    nrow, ncol = arc_temp.shape
    icol0 = np.int(0.25 * ncol)
    icol1 = np.int(0.75 * ncol)
    assert icol0 > maxshift

    nshift_grid = np.arange(-maxshift, maxshift + 1)
    nshift_dotmax = np.zeros(nrow)
    for irow in np.arange(nrow):
        ind_dotmax = np.argmax(
            [np.dot(arc_temp[irow][icol0 + ishift:icol1 + ishift], arc_obs[irow][icol0:icol1]) for ishift in
             nshift_grid])
        nshift_dotmax[irow] = nshift_grid[ind_dotmax]
    bulkshift = np.median(nshift_dotmax)
    npm1 = np.sum(np.abs(nshift_dotmax - bulkshift) <= 1)
    npm2 = np.sum(np.abs(nshift_dotmax - bulkshift) <= 2)
    assert npm2 > 0.5 * nrow
    print("@corr_arc: bulkshift={} (±1 {}/{}) (±2 {}/{})".format(bulkshift, npm1, nrow, npm1, nrow))

    xcoord = np.arange(ncol)
    wave_corr = interp1d(xcoord - bulkshift, wave_temp, kind="linear", fill_value="extrapolate")(xcoord)
    return wave_corr


def ccfmax_gauss(x, b=0, c=1):
    """
    Generate a Gaussian array, centering at b, with width of c.

    Parameters
    ----------
    x : ndarray
        x array.
    b : float, optional
        The center of Gaussian. The default is 0.
    c : float, optional
        The width of Gaussian. The default is 1.

    Returns
    -------
    y
        The Gaussian array.

    """
    y = np.exp(-0.5 * ((x - b) / c) ** 2.)
    return y / np.sum(y)


def ccfmax_cost(x0, x, y, width=2):
    """
    The cost function for ccfmax

    Parameters
    ----------
    x0 : float
        initial guess of max position.
    x : array
        x.
    y : array
        y.
    width : float, optional
        the width of Gaussian. The default is 2.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    y_model = ccfmax_gauss(x, x0, width)
    return - np.dot(y, y_model)


def ccfmax(x0, x, y, width=2, method="Nelder-Mead"):
    """
    The ccf max method

    Parameters
    ----------
    x0 : float
        initial guess of max position.
    x : array
        x.
    y : array
        y.
    width : float, optional
        the width of Gaussian. The default is 2.
    method : str, optional
        minimization method. The default is "Nelder-Mead".

    Returns
    -------
    pccf : TYPE
        DESCRIPTION.

    """
    pccf = minimize(ccfmax_cost, x0=x0, args=(x, y, width), method=method)
    return pccf


def gauss(x, a, b, c):
    """
    generate a Gaussian function

    Parameters
    ----------
    x : array like
        x coordinates.
    a : float
        amplitude (centeral height).
    b : float
        center.
    c : float
        sigma.

    Returns
    -------
    y
        y = Gaussian(x | a, b, c).

    """
    c = np.abs(c)
    return a / np.sqrt(2. * np.pi) / c * np.exp(-0.5 * ((x - b) / c) ** 2.)


def find_lines(wave_init, arc_obs, arc_line_list, npix_chunk=20, ccf_kernel_width=2):
    """
    Find emission lines in ThAr spectrum.

    Parameters
    ----------
    wave_init : array
        initial wavelength solution.
    arc_obs : array
        observed arc spectrum.
    arc_line_list : array
        the arc line list (a list of wavelengths).
    npix_chunk : int, optional
        the chunk length (half). The default is 20.

    Returns
    -------
    tlines : astropy.table.Table
        A table of identified lines.

    """
    # get a chunk [5*5] ±5sigma
    # shift: 1-2 pixel
    # typical oversampling: 1/3R --> 3pixel=FWHM
    # LAMOST MRS overestimate: 0.7 / 0.1 --> 7pixel=FWHM
    norder, npix = wave_init.shape
    xcoord = np.arange(npix)

    tlines = []
    # for each order
    for iorder in range(norder):
        # this order
        this_wave_init = wave_init[iorder]
        this_wave_min = np.min(this_wave_init)
        this_wave_max = np.max(this_wave_init)
        this_arc_obs = arc_obs[iorder]
        this_line_list = arc_line_list[np.logical_and(arc_line_list > this_wave_min, arc_line_list < this_wave_max)]

        # for each line
        for this_line in this_line_list:
            # init x position
            this_line_x_init = np.interp(this_line, this_wave_init, xcoord)
            this_line_x_init_int = np.int(this_line_x_init)  # np.argmin(np.abs((this_wave_init-this_line)))

            # get a chunk
            if npix_chunk < this_line_x_init_int < npix - npix_chunk:
                this_line_slc = slice(this_line_x_init_int - npix_chunk, this_line_x_init_int + npix_chunk)
                this_line_xcoord = xcoord[this_line_slc]
                this_line_arc = this_arc_obs[this_line_slc]
                this_line_base = np.percentile(this_line_arc, q=20)  # 25th percentile as baseline
                # if this_line_base < 0:
                #     continue

                # 1. Gaussian fit
                try:
                    y = this_line_arc - this_line_base
                    popt, pcov = curve_fit(gauss, this_line_xcoord, y,
                                           p0=[np.max(y)/2, this_line_x_init, 1.5], )
                    # bounds=(np.array([0,-np.inf,1]), np.array([np.inf,np.inf,np.inf])))
                    this_line_a_gf = popt[0]
                    this_line_c_gf = popt[2]
                    this_line_x_gf = popt[1]
                    this_line_wave_init_gf = np.interp(popt[1], xcoord, this_wave_init)
                except:
                    this_line_a_gf = np.nan
                    this_line_c_gf = np.nan
                    this_line_x_gf = np.nan
                    this_line_wave_init_gf = np.nan
                # 2. CCF method
                try:
                    pccf = ccfmax(this_line_x_init, this_line_xcoord, this_line_arc, width=ccf_kernel_width, method="Nelder-Mead")
                    this_line_x_ccf = np.float(pccf.x)
                    this_line_wave_init_ccf = np.interp(this_line_x_ccf, xcoord, this_wave_init)
                    this_line_peakflux = np.interp(this_line_x_ccf, this_line_xcoord, this_line_arc)
                except:
                    this_line_x_ccf = np.nan
                    this_line_wave_init_ccf = np.nan
                    this_line_peakflux = np.nan

                # gather results
                this_result = dict(
                    order=iorder,
                    line=this_line,
                    line_x_init=this_line_x_init,
                    # gf
                    line_x_gf=this_line_x_gf,
                    line_a_gf=this_line_a_gf,
                    line_c_gf=this_line_c_gf,
                    line_wave_init_gf=this_line_wave_init_gf,
                    # ccf
                    line_x_ccf=this_line_x_ccf,
                    line_wave_init_ccf=this_line_wave_init_ccf,
                    line_base=this_line_base,
                    # peakflux
                    line_peakflux=this_line_peakflux
                )
                # np.array([iorder, this_line_xcenter, this_line, line_wave_gf, line_wave_ccf,
                #           this_line_base, popt[0]/np.sqrt(2.*np.pi)/popt[2],
                #           *popt, np.float(pccf.x)]))
                tlines.append(this_result)
    tlines = table.Table(tlines)
    print("  |-find_lines: {}/{} lines using GF / CCF!".format(
        np.sum(np.isfinite(tlines["line_x_gf"])),
        np.sum(np.isfinite(tlines["line_x_ccf"]))
    ))

    return tlines


