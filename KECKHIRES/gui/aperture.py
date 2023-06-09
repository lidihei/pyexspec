from twodspec.aperture import *

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

