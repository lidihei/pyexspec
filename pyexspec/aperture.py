from twodspec.aperture import *

from skimage.filters import gaussian
from scipy.signal import medfilt2d
from .fitfunc import Poly1DFitter
import numpy as np

def polyfit_backgroud(x, z, deg=4, min_select=10, nsigma=5, verbose=True, **argwords):
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
            The minimal number of selected points. The default is None.
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
        #pf1 = Poly1DFitter(x[indselect], z[indselect], deg=deg, pw=1, robust=False)
        pp = np.polyfit(x[indselect], z[indselect], deg)
        pf1 = np.poly1d(pp)
        z_pred = pf1(x)#pf1.predict(x)
        z_res = z_pred - z
        sigma = np.std(z_res[indselect])
        indreject = np.abs(z_res[indselect]) > nsigma * sigma
        n_reject = np.sum(indreject)
        if n_reject == 0:
            # no lines to kick
            break
        elif isinstance(min_select, int) and (min_select <0 ) and np.sum(indselect) <= min_select:
            # selected lines reach the threshold
            break
        else:
            # continue to reject lines
            indselect &= np.abs(z_res) < nsigma * sigma
            iiter += 1
        if verbose:
            print("  |-@polyfit_backgroud: iter-{} \t{} points kicked, {} points left, rms={:.5f} counts".format(
                iiter, n_reject, np.sum(indselect), sigma))
    #pf1.rms = sigma
    return pf1, indselect

def straight_line(x, y):
    """ return coefs for a straight line """
    a = np.float(np.diff(y) / np.diff(x))
    b = np.float(y[0] - a * x[0])
    return a, b



def sort_apertures(ap_trace: np.ndarray):
    """ sort ascend """
    nap = ap_trace.shape[0]
    ind_sort = np.arange(nap, dtype=int)
    for i in range(nap - 1):
        for j in range(i + 1, nap):
            ind_common = (ap_trace[i] >= 0) & (ap_trace[j] >= 0)
            if np.median(ap_trace[i][ind_common]) > np.median(ap_trace[j][ind_common]) and ind_sort[i] < ind_sort[j]:
                ind_sort[i], ind_sort[j] = ind_sort[j], ind_sort[i]
                # print(ind_sort)
    return ind_sort

class Aperturenew(Aperture):

    def polyfitnew(self, deg=2, ystart=0, yend=None):
        """ fit using chebyshev polynomial for adopted apertures
            in interval [ystart, yend]"""
        self.polydeg = deg
        n_row = self.imshape[0]
        if yend is None: yend = n_row
        nap = self.nap
        ap_col_interp = np.arange(0, n_row, dtype=int)
        ap_upper_interp = []  # interpolated
        ap_lower_interp = []
        ap_center_interp = []
        ap_upper_chebcoef = []  # chebcoef
        ap_lower_chebcoef = []
        ap_center_chebcoef = []
        for i in range(nap):
            ind_fit = self.ap_center[i] > -1
            ind_fit = ind_fit & (self.y >= ystart) & (self.y <= yend)
            this_chebcoef = np.polynomial.chebyshev.chebfit(
                self.y[ind_fit], self.ap_upper[i][ind_fit], deg=deg)
            ap_upper_chebcoef.append(this_chebcoef)
            ap_upper_interp.append(
                np.polynomial.chebyshev.chebval(ap_col_interp, this_chebcoef))
            # for lower
            this_chebcoef = np.polynomial.chebyshev.chebfit(
                self.y[ind_fit], self.ap_lower[i][ind_fit], deg=deg)
            ap_lower_chebcoef.append(this_chebcoef)
            ap_lower_interp.append(
                np.polynomial.chebyshev.chebval(ap_col_interp, this_chebcoef))
            # for center
            this_chebcoef = np.polynomial.chebyshev.chebfit(
                self.y[ind_fit], self.ap_center[i][ind_fit], deg=deg)
            ap_center_chebcoef.append(this_chebcoef)
            ap_center_interp.append(
                np.polynomial.chebyshev.chebval(ap_col_interp, this_chebcoef))

        # transform to numpy.array format
        self.ap_upper_interp = np.array(ap_upper_interp)
        self.ap_lower_interp = np.array(ap_lower_interp)
        self.ap_center_interp = np.array(ap_center_interp)
        self.ap_upper_chebcoef = np.array(ap_upper_chebcoef)
        self.ap_lower_chebcoef = np.array(ap_lower_chebcoef)
        self.ap_center_chebcoef = np.array(ap_center_chebcoef)
        # center trace: center is not fitted but averaged from edges
        # self.ap_center_interp = (ap_upper_interp + ap_lower_interp) / 2.

        self.ispolyfitted = True
        return

    def backgroundnew(self, img, ap_center=None, q=(5, 40), npix_inter=10, sigma=None, kernel_size=None, ap_width=None, Napw=1.5,
                      longslit = True, Napw_bg=1, deg=4, num_sigclip=5, **argwords ):
        """ determine background/scattered light using inter-aperture pixels
        Parameters
        ----------
        img: ndarray
            the image whose background is to be determined
        ap_center: ndarray
            aperture center array, (n_aperture, n_pixel)
        q: tuple of float
            the starting and ending percentile
        npix_inter:
            the number of pixel that will be used to determine the background
        sigma: tuple
            gaussian smoothing parameter
        kernel_size: tuple
            median smoothing parameter
        ap_width: int
            the width of aperture
        Napw: float
           N times ap_width away from the aperture center which is the right (and left) edge of the background
        Napw_bg: float
           N times ap_width area used to fitting background, |Napw_bg*ap_width-|Napw*ap_width -|center|+ Napw*ap_width| + Napw_bg*ap_width|
        Returns
        -------
        bg0

        """
        if ap_center is None: ap_center = self.ap_center
        if ap_width is None: ap_width = self.ap_width
        n_ap = ap_center.shape[0]
        nrow, ncol = img.shape
        x = np.arange(ncol, dtype=np.float)
        npix_inter_hf = np.int(npix_inter / 2)
        if isinstance(q, tuple):
            q = np.linspace(q[0], q[1], n_ap)

        bg0 = img.copy()

        self.Napw = Napw
        self.Napw_bg = Napw_bg
        if longslit:
           if n_ap < 2:
              i_ap = 0
              for i_row in range(nrow):
                  i_med_r = int(ap_center[i_ap][i_row] + Napw*ap_width)
                  i_med_l = int(ap_center[i_ap][i_row] - Napw*ap_width)
                  zi = img[i_row]
                  apw_bg = int(ap_width*Napw_bg)
                  ind_ls, ind_le = i_med_l-apw_bg, i_med_l
                  ind_rs, ind_re = i_med_r, i_med_r+apw_bg
                  xi_background = np.hstack([x[ind_ls:ind_le], x[ind_rs:ind_re]])
                  zi_background = np.hstack([zi[ind_ls:ind_le], zi[ind_rs:ind_re]])
                  self.xi_background = xi_background
                  self.zi_background = zi_background
                  pf1, _ = polyfit_backgroud(xi_background, zi_background, deg=deg, nsigma=num_sigclip, **argwords)
                  bg0[i_row][ind_ls:ind_re] = pf1(x[ind_ls:ind_re])
        else:
            if n_ap < 2:
                i_ap = 0
                for i_row in range(nrow):
                    i_med_r = np.int(ap_center[i_ap][i_row] + Napw*ap_width)
                    i_med_l = np.int(ap_center[i_ap][i_row] - Napw*ap_width)
                    y_med_r = np.percentile(
                        img[i_row, np.max((0, i_med_r - npix_inter_hf)):np.max((1, i_med_r + npix_inter_hf + 1))], q[i_ap])
                    y_med_l = np.percentile(
                        img[i_row, np.max((0, i_med_l - npix_inter_hf)):np.max((1, i_med_l + npix_inter_hf + 1))], q[i_ap])
                    a, b = straight_line([x[i_med_l], x[i_med_r]], [y_med_l, y_med_r])
                    bg0[i_row, :i_med_r] = a * x[:i_med_r] + b

            else:
                for i_row in range(nrow):
                    # each row
                    for i_ap in range(n_ap):
                        # each aperture
                        if i_ap == 0:
                            # the first aperture
                            i_med_r = (ap_center[i_ap][i_row] + ap_center[i_ap + 1][
                                i_row]) / 2
                            i_med = ap_center[i_ap][i_row]
                            i_med_l = 2 * i_med - i_med_r

                            i_med_r = np.int(i_med_r)
                            # i_med = np.int(i_med)
                            i_med_l = np.int(i_med_l)

                            y_med_r = np.percentile(
                                img[i_row, np.max((0, i_med_r - npix_inter_hf)):np.max((1, i_med_r + npix_inter_hf + 1))], q[i_ap])
                            y_med_l = np.percentile(
                                img[i_row, np.max((0, i_med_l - npix_inter_hf)):np.max((1, i_med_l + npix_inter_hf + 1))], q[i_ap])

                            a, b = straight_line([x[i_med_l], x[i_med_r]], [y_med_l, y_med_r])
                            bg0[i_row, :i_med_r] = a * x[:i_med_r] + b

                        if i_ap == n_ap - 1:
                            # the last aperture
                            i_med = ap_center[i_ap][i_row]
                            i_med_l = (ap_center[i_ap - 1][i_row] + ap_center[i_ap][
                                i_row]) / 2
                            i_med_r = 2 * i_med - i_med_l

                            # i_med = np.int(i_med)
                            i_med_r = np.int(i_med_r)
                            i_med_l = np.int(i_med_l)

                            y_med_r = np.percentile(
                                img[i_row, i_med_r - npix_inter_hf:i_med_r + npix_inter_hf + 1], q[i_ap])
                            y_med_l = np.percentile(
                                img[i_row, i_med_l - npix_inter_hf:i_med_l + npix_inter_hf + 1], q[i_ap])

                            a, b = straight_line([x[i_med_l], x[i_med_r]], [y_med_l, y_med_r])
                            bg0[i_row, i_med_l:] = a * x[i_med_l:] + b

                        else:
                            # the middle aperture
                            # i_med = ap_center[i_ap][i_row]
                            i_med_l = (ap_center[i_ap - 1][i_row] + ap_center[i_ap][i_row]) / 2
                            i_med_r = (ap_center[i_ap][i_row] + ap_center[i_ap + 1][i_row]) / 2

                            # i_med = np.int(i_med)
                            i_med_r = np.int(i_med_r)
                            i_med_l = np.int(i_med_l)

                            y_med_r = np.percentile(
                                img[i_row, np.min((ncol-1, i_med_r - npix_inter_hf)):np.min((ncol, i_med_r + npix_inter_hf + 1))], q[i_ap])
                            y_med_l = np.percentile(
                                img[i_row, np.min((ncol-1, i_med_l - npix_inter_hf)):np.min((ncol, i_med_l + npix_inter_hf + 1))], q[i_ap])

                            a, b = straight_line([x[i_med_l], x[i_med_r]], [y_med_l, y_med_r])
                            bg0[i_row, i_med_l:i_med_r] = a * x[i_med_l:i_med_r] + b

        # do a smooth                                                                                                                                                         
        # bgg = gaussian(bg0, sigma=sigma)
        #
        # bgm = medfilt2d(bg0, kernel_size=kernel_size)
        # bgmg = gaussian(bgm, sigma=sigma)

        if kernel_size is not None:
            bg0 = medfilt2d(bg0, kernel_size=kernel_size)
        if sigma is not None:
            bg0 = gaussian(bg0, sigma=sigma)
        return bg0

