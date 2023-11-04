from twodspec.polynomial import Poly1DFitter
from scipy.fftpack import fft
from scipy.fftpack import ifft
import numpy as np
import matplotlib.pyplot as plt

class longslit():

    def __init__(self, wave_template:np.array=None,
                 flux_template: np.array=None, linelist: np.array=None):
        '''
        calibrate wavelength by using a know arc template observe by the same telescope
        parameters:
        wave_template: [1D array] the wave should be sorted from small to large
        flux_template: [1D array]
        linelist: [1D array] a list of wavelength of emission lines of arc lamp spectrum
        '''
        self.wave_template = wave_template
        self.x_template = None if (wave_template is None) else np.arange(len(wave_template))
        self.flux_template = flux_template
        self.linelist = linelist


    def read_arc_pypiet(self, fname: str):
        hdu = fits.open(fname)
        self.wave_pypiet = hdu[2].data['wave_soln'][0]
        self.flux_pypiet = hdu[4].data[:, 0]
        self.x_pypiet = np.arange(len(self.wave_pypiet))

    def find_lines(self, wave_init: np.array = None, flux: np.array = None, npix_chunk=8, ccf_kernel_width=1.5):
        '''
        find the index of the emission lines of the arc lamp spectrum, detials see. twodspec.thar.fine_lines
        returns:
        -------------
        tab_lines: [astropy.table] a table of emission lines of the sample spectrum
        '''
        from twodspec import thar
        wave_init = self.wave_init if (wave_init is None) else wave_init
        flux = self.flux_pypiet if flux is None else flux
        if len(wave_init.shape) ==1:
            wave_init = np.array([wave_init])
            flux = np.array([flux])
        tab_lines = thar.find_lines(wave_init, flux, self.linelist, npix_chunk=npix_chunk, ccf_kernel_width=ccf_kernel_width)
        return tab_lines

    def estimate_wave_init(self, x: np.array = None, xshift: float=None,
                           x_template = None, wave_template = None,
                            **argwords):
        '''
        estimate the initall wavelength by using template arc lam spectrum with twodspec.polynomial.Poly1DFitter
        x [1D array] the index of sample spectrum wave
        xshift [float]:the shifted x-axis coordinates of the sample spectrum comparing with template in unit of pixel
        x_template [1D array] the index of the template spectrum wave
        wave_template [1D array]: the wavelength of the template spectrum
        returns:
        ----------------
        wave_init [1D array] the initall wavelength
        '''
        if x_template is None:
           x_template = self.x_template
        if wave_template is None:
           wave_template = self.wave_template
        pf1, _indselect = self.grating_equation(x_template, wave_template, **argwords)
        #self.pf1 = pf1
        if x is None: x = self.x_pypiet
        if xshift is None: xshift = self.xshift
        x = x+xshift
        self.func_polyfit_template = pf1 # the poly fitted function of template arc lamp sectrum
        wave_init = pf1.predict(x)
        self.wave_init = wave_init
        return wave_init

    def grating_equation(self, x, z, deg=4, nsigma=3, min_select=None, verbose=True, **argwords):
        """
        Fit a grating equation (1D polynomial function) to data
        Parameters
        ----------
        x : array
            x coordinates of emission lines
        z : array
            The true wavelengths of lines.
        deg : tuple, optional
            The degree of the 1D polynomial. The default is (4, 10).
        nsigma : float, optional
            The data outside of the nsigma*sigma radius is rejected iteratively. The default is 3.
        min_select : int or None, optional
            The minimal number of selected lines. The default is None.
        verbose :
            if True, print info

        Returns
        -------
        pf1, indselect

        """
        indselect = np.ones_like(x, dtype=bool)
        iiter = 0
        # pf1
        while True:
            pf1 = Poly1DFitter(x[indselect], z[indselect], deg=deg, pw=1, robust=False)
            z_pred = pf1.predict(x)
            z_res = z_pred - z
            sigma = np.std(z_res[indselect])
            indreject = np.abs(z_res[indselect]) > nsigma * sigma
            n_reject = np.sum(indreject)
            if n_reject == 0:
                # no lines to kick
                break
            elif isinstance(min_select, int) and min_select >= 0 and np.sum(indselect) <= min_select:
                # selected lines reach the threshold
                break
            else:
                # continue to reject lines
                indselect &= np.abs(z_res) < nsigma * sigma
                iiter += 1
            if verbose:
                print("  |-@grating_equation: iter-{} \t{} lines kicked, {} lines left, rms={:.5f} A".format(
                    iiter, n_reject, np.sum(indselect), sigma))
        pf1.rms = sigma

        if verbose:
            print("  |-@grating_equation: {} iterations, rms = {:.5f} A".format(iiter, pf1.rms))
        return pf1, indselect

    def calibrate(self, xcoord, tabline, flux=None, deg=4, line_peakflux= 50, num_sigclip=2.5, line_type = 'line_x_ccf',
                  min_select_lines = 10, verbose:bool =False, show:bool =False):
        '''
        Fitting index, lines in tabline using grating_equation, and predict wavelength for xcoord
        parameters:
        --------------------
        xcoord [1D array] index of wave of the sample spectrum
        tabline [astropy.table]: a table of the emission lines of the sample spectrum.
        flux [1D array] flux of the sample spectum, could be None (only for plot QA)
        line_type [str]: e.g. line_x_gf	, line_x_ccf (the index coordinates of the peak of emission lines)
        min_select_lines [int]: the minimum lines for fitting.
        returns:
        --------------------
        wave_solu [1D array] the wavelength of xcoord fitted by emission line in tabline
        '''
        from astropy import table
        tlines = tabline
        ind_good = np.isfinite(tlines["line_x_ccf"]) & (np.abs(tlines["line_x_ccf"] - tlines["line_x_init"]) < 10) & (
                    (tlines["line_peakflux"] - tlines["line_base"]) > line_peakflux) & (
                               np.abs(tlines["line_wave_init_ccf"] - tlines["line"]) < 3)
        indselect0 = np.zeros(len(tlines), dtype=bool)
        tlines.add_column(table.Column(ind_good, "ind_good"))
        def clean(pw=1, deg=3, threshold=0.1, min_select=10):
            order = tlines["order"].data
            ind_good = tlines["ind_good"].data
            linex = tlines["line_x_ccf"].data
            z = tlines["line"].data

            u_order = np.unique(order)
            for _u_order in u_order:
                ind = (order == _u_order) & ind_good
                if np.sum(ind) > min_select:
                    # in case some orders have only a few lines
                    p1f = Poly1DFitter(linex[ind], z[ind], deg=deg, pw=pw)
                    res = z[ind] - p1f.predict(linex[ind])
                    ind_good[ind] &= np.abs(res) < threshold
            tlines["ind_good"] = ind_good
            return
        print("  |- {} lines left".format(len(tlines)))
        clean(pw=1, deg=deg, threshold=0.8, min_select=20)
        clean(pw=1, deg=deg, threshold=0.4, min_select=20)
        clean(pw=1, deg=deg, threshold=0.2, min_select=20)
        print("  |- {} lines left".format(np.sum(tlines["ind_good"])))
        tab_lines = tlines.copy()
        tlines = tlines[tlines["ind_good"]]

        """ fitting grating equation """
        x = tlines["line_x_ccf"]  # line_x_ccf/line_x_gf
        y = tlines["order"]
        z = tlines["line"]
        pf1, indselect = self.grating_equation(
               x, z, deg=deg, nsigma=num_sigclip, min_select=min_select_lines, verbose=verbose)
        print("  |- {} lines selected".format(np.sum(indselect)))
        #tlines.add_column(table.Column(indselect, "indselect"))
        indselect0[tab_lines["ind_good"]] = indselect
        tab_lines.add_column(table.Column(indselect0, "indselect"))
        mpflux = np.median(tab_lines[tab_lines["indselect"]]['line_peakflux'])
        rms = np.std((pf1.predict(x) - z)[indselect])
        nlines = np.sum(indselect)
        wave_solu = pf1.predict(xcoord)  # polynomial fitter
        self.wave_solu = wave_solu
        self.tab_lines = tab_lines
        self.func_polyfit = pf1
        self.rms = rms
        self.tab_lines = tab_lines
        print("  |- nlines={}  rms={:.4}A  median peak flux={:.1f}".format(nlines, rms, mpflux))
        if show:
            ##### plot QA of wavelength calibration
            fig1, axs = plt.subplots(3,1,figsize=(7,9), gridspec_kw={'height_ratios': [3,1,3]}, sharex=True)
            plt.subplots_adjust(hspace=0)
            plt.suptitle(f'rms = {rms:.3}')
            plt.sca(axs[0])
            _tab = tab_lines[indselect0]
            plt.plot(xcoord, wave_solu)
            plt.scatter(_tab['line_x_ccf'], _tab['line'], marker='x')
            plt.ylabel('Wavelength')
            plt.sca(axs[1])
            plt.scatter(_tab['line_x_ccf'], _tab['line']-pf1.predict(_tab['line_x_ccf']), marker='x')
            plt.ylabel('Residual')
            plt.sca(axs[2])
            _tab = tab_lines
            plt.scatter(_tab['line_x_ccf'], _tab['line_peakflux'], marker='+', label=r'$Peak_{\rm CCF}$')
            _tab = tab_lines[indselect0]
            plt.scatter(_tab['line_x_ccf'], _tab['line_peakflux'], marker='o', label=r'$Peak_{\rm CCF}: good$', facecolors='none', edgecolors='r')
            plt.plot(xcoord, flux, lw=1, c='k')
            plt.legend()
            #plt.xlabel(r'Wavelength ${\rm \AA}$')
            plt.xlabel('x index (pixel)')
            plt.ylabel(r'Counts')
            self.fig_QA_wave_calibrate = fig1
        return wave_solu

    def get_xshift(self, xcoord, flux, x_template: np.array = None,
                   flux_template:np.array = None, step: float = 0.1, show=False):
        '''
        get the shifted x-axis coordinates of the sample spectrum comparing with template
        returns:
        --------------
        xshift [float]: the x-axis coordinate of the maximum CCF point in unit of pixel.
        '''
        shfits, ccf = self.calCCF(xcoord, flux, x_template, flux_template, show=show)
        #print(np.argmax(ccf))
        xshift = shfits[np.argmax(ccf)]
        self.xshift = xshift
        return xshift

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
