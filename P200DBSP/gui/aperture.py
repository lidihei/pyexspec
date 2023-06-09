from twodspec.aperture import *

from skimage.filters import gaussian
from scipy.signal import medfilt2d


def straight_line(x, y):
    """ return coefs for a straight line """
    a = np.float(np.diff(y) / np.diff(x))
    b = np.float(y[0] - a * x[0])
    return a, b

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

    def backgroundnew(self, img, ap_center=None, q=(40, 5), npix_inter=5, sigma=(10, 10), kernel_size=(11, 11), ap_width=None, Napw=3):
        """ newly developed on 2017-05-28, with best performance """
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
        Napw: int
           N times ap_width for which find background
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

        bg0 = np.zeros_like(img, np.float)

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

