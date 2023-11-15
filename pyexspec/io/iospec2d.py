import numpy as np
from astropy.io import fits

class iospec2d():

    def __init__(self, rot90 = False, trimx = None, trimy = None, ihdu=0):
        '''
        Read 2D image of a spectrum, and trim the image
        parameters:
        ------------------
        rot90 [bool]: if True rotate the image with 90 degree
        trimx [list, or tuple]: Trim image on x axis, e.g. trimx = [0, 100]
        trimy [list, or tuple]: Trim image on y axis, e.g. trimx = [0, 100]
                note, if rot90 = True, trim image after rotation
        '''
        self.trimx = trimx
        self.trimy = trimy
        self.rot90 = rot90
        self.ihdu = ihdu

    def _read_img(self, fp_star):
        '''
        parameters:
        ------------
        fp_star [str] file name including full path
        returns:
        ------------
        data [2D array] data array of trimed image
        '''
        hdu = fits.open(fp_star)
        ihdu = self.ihdu
        data = hdu[ihdu].data
        if self.rot90: data = np.rot90(data)
        ye, xe= data.shape
        trimx = [0, xe] if self.trimx is None else self.trimx
        trimy = [0, ye] if self.trimy is None else self.trimy
        xs, xe = trimx
        ys, ye = trimy
        data = data[ys:ye, xs:xe]
        hdu.close()
        #return np.rot90(data - self.master_bias)
        return data


    def cal_image_error_square(self, image, bias_err_squared, gain, readnoise):
        '''
        calculate error**2 of each pixel of image
        paramters:
        -------------------------
        image [2D array]
        bias_err_squared [float], np.std(master_biase)**2
        gain [float]  in unit of e-/ADU
        readnoise [float]
        returns:
        ------------
        image_err_squared
        '''
        image_err_squared = image*gain
        _ind = image_err_squared < 0
        image_err_squared[_ind] = 0
        image_err_squared = (image_err_squared + readnoise**2)/gain**2+bias_err_squared
        return image_err_squared

    def read_star(self, fp_star, master_bias=None, master_bias_err_squared=None, gain=None, readnoise=None, remove_cosmic_ray=False):
        '''
        paramters:
        --------------
        fp_star [str] file name including full path
        gain [float] in units of e-/ADU
        readnoise [float] system noise in units of e-
        returns:
        --------------
        image [2D array] the trimed image which have substracted bias
        '''
        master_bias = self.master_bias if master_bias is None else master_bias
        hdu = fits.open(fp_star)
        ihdu = self.ihdu
        data = self._read_img(fp_star)
        if gain is None: gain = hdu[ihdu].header['gain']
        if readnoise is None: readnoise = hdu[ihdu].header['RON']
        hdu.close()
        image = data- master_bias
        bias_err_squared = np.nanvar(master_bias)
        bias_err_squared = self.master_bias_err_squared if  master_bias_err_squared is None else master_bias_err_squared
        image_err_squared = self.cal_image_error_square(image, bias_err_squared, gain, readnoise).copy()
        self.image = image
        self.image_err_squared = image_err_squared
        if remove_cosmic_ray:
           pass
        return image
