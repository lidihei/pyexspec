import joblib
import numpy as np
from astropy.io import fits

class iospec1d():


    def readspec1d_dump(self, filename, wave_key = 'wave_solu_barycorr',
                       flux_key = 'spec_extr_divide_flat', flux_err_key = 'err_extr_divide_flat'):
        '''
        Read 1D spectrum from the dump file extraceted by pyexspec
        returns:
        wave, flux, flux_err
        '''
        dumpdic = joblib.load(filename)
        self.dumpdic = dumpdic
        wave = dumpdic[wave_key]
        flux = dumpdic[flux_key]
        flux_err = dumpdic[flux_err_key]
        self.wave = wave,
        self.flux = flux
        self.flux_err = flux_err
        return wave, flux, flux_err


    def readp200spec1dfits(self, fname):
        '''
        Read P200 DBPS spectrum extracted by pyexspec
        parameters:
        ----------------
        fname [str] file name
        returs:
        -------------
        wave [1D array] is in uint of angstrom and in air
        flux [1D array] is in units of 10^17 erg/s/cm2/A
        fluxerr [1D array]
        '''
        from astropy.io import fits
        hdu = fits.open(fname)
        data = hdu[1].data
        header = hdu[0].header
        wave = data['wave']
        flux = data['flux']
        fluxerr = data['error']
        self.wave_fits = wave
        self.flux_fits = flux
        self.flux_err_fits = fluxerr
        self.header_fits = header
        return wave, flux, fluxerr

    def correct_rv(self, rv, wave=None):
        '''
        Correct the wavelenth to the rest frame
        parameters:
        -------------
        rv [float] the radial velocity, in unit of km/s
        returns
        ------------
        wave_rvcorr [1D array]
        '''
        from pyexspec.utils.spectools import rvcorr_spec
        if wave is None: wave = self.wave
        wave_rvcorr = rvcorr_spec(wave, -rv, returnwvl=True)
        self.wave_rvcorr = wave_rvcorr
        return wave_rvcorr

    def write2fits_iraf(self, header, data, fname_out, overwrite=True):
        '''
        Write a spectum into a fits file that can be read by iraf.
        I write this function for manually wave calibrate using iraf
        data = dump_lamp['lamp1d'][0]
        '''
        data = np.array(data, dtype=np.float32)
        hdu = fits.HDUList([fits.PrimaryHDU(header=_header, data=_data)])
        hdu.writeto(fname_out, overwrite=overwrite)

    def write2fits(self, wave, flux, flux_err, header, fileout, overwrite=True):
        '''
        Write  spectum into a fits
        fileout [str] the output file name
        '''
        table = fits.BinTableHDU.from_columns([
                    fits.Column(name='wave',  format='E', array=np.array(wave, dtype=np.float32)),
                    fits.Column(name='flux',format='E', array=np.array(flux, dtype=np.float32)),
                    fits.Column(name='error',format='E', array=np.array(flux_err, dtype=np.float32))
                                         ])
        data0 = np.zeros(1,dtype=np.float32)
        hdu = fits.PrimaryHDU(data0)
        hdu.header = header
        hdul = fits.HDUList([hdu, table])
        hdul.writeto(fileout, overwrite=overwrite)
