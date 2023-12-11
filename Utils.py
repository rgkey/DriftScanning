from astropy.io import fits
from astropy.wcs import WCS
import astroscrappy
import warnings
import numpy as np

class ImageImport:
    
    def __init__(self, ccd_name, filepath):
        
        self.ccd = ccd_name
        self.filepath = filepath
    
        with fits.open(self.filepath) as file:
            self.hdr = file[0].header
            self.ccd_hdr = file[self.ccd].header
            self.data = file[self.ccd].data
        self.wcs = WCS(self.ccd_hdr)
                                
    
    def cosmicray_removal(self, gain_keyword = ['GAINA', 'GAINB'], 
                          saturation_keyword = ['SATURATA', 'SATURATB'], readnoise_keyword = ['RDNOISEA', 'RDNOISEB']):

        self.gain = np.median([self.ccd_hdr[gain] for gain in gain_keyword])
        self.readnoise = np.median([self.ccd_hdr[readnoise] for readnoise in readnoise_keyword])
        self.saturation = np.median([self.ccd_hdr[saturate] for saturate in saturation_keyword])
        
        clean_mask, clean_data = astroscrappy.detect_cosmics(self.data, gain=self.gain,
                                                             readnoise=self.readnoise, satlevel=self.saturation, cleantype='medmask')

        self.cr_mask = clean_mask
        self.clean_data = clean_data
