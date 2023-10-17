import numpy as np
import scipy as scp
import pandas as pd
import astroscrappy
import warnings

from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import match_coordinates_sky
from astropy.stats import sigma_clipped_stats, gaussian_sigma_to_fwhm, gaussian_fwhm_to_sigma
from astropy import units as u
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.modeling.models import  Gaussian2D, Const2D
from astropy.utils.exceptions import AstropyUserWarning
from astropy.table import Table
from astropy.nddata import NDData
from photutils.psf import extract_stars

from photutils.psf import IntegratedGaussianPRF
from astropy.modeling.fitting import LevMarLSQFitter
from photutils.psf import DAOPhotPSFPhotometry
from photutils.detection import find_peaks

#Define a function that MODIFIES from Photutils source code (centroid functionality) which fits a known centroid location with a 
#2D gaussian function. Fitting is LevMarLSQ architecture within Astropy, and returns the major and minor FWHM of the best-fit model. 

#This is used to find the average FWHM of the image without eyeballing it. It's also used to define other parameters within the PSF photometry stage.
def FWHM_2dg(data, error=None, mask=None):

    # prevent circular import
    from photutils.morphology import data_properties

    data = np.ma.asanyarray(data)

    if mask is not None and mask is not np.ma.nomask:
        mask = np.asanyarray(mask)
        if data.shape != mask.shape:
            raise ValueError('data and mask must have the same shape.')
        data.mask |= mask

    if np.any(~np.isfinite(data)):
        data = np.ma.masked_invalid(data)
        warnings.warn('Input data contains non-finite values (e.g., NaN or '
                      'infs) that were automatically masked.',
                      AstropyUserWarning)

    if error is not None:
        error = np.ma.masked_invalid(error)
        if data.shape != error.shape:
            raise ValueError('data and error must have the same shape.')
        data.mask |= error.mask
        weights = 1.0 / error.clip(min=1.0e-30)
    else:
        weights = np.ones(data.shape)

    if np.ma.count(data) < 7:
        raise ValueError('Input data must have a least 7 unmasked values to '
                         'fit a 2D Gaussian plus a constant.')

    # assign zero weight to masked pixels
    if data.mask is not np.ma.nomask:
        weights[data.mask] = 0.0

    mask = data.mask
    data.fill_value = 0.0
    data = data.filled()

    # Subtract the minimum of the data as a rough background estimate.
    # This will also make the data values positive, preventing issues with
    # the moment estimation in data_properties. Moments from negative data
    # values can yield undefined Gaussian parameters, e.g., x/y_stddev.
    props = data_properties(data - np.min(data), mask=mask)

    constant_init = 0.0  # subtracted data minimum above
    g_init = (Const2D(constant_init)
              + Gaussian2D(amplitude=np.ptp(data),
                           x_mean=props.xcentroid,
                           y_mean=props.ycentroid,
                           x_stddev=props.semimajor_sigma.value,
                           y_stddev=props.semiminor_sigma.value,
                           theta=props.orientation.value))
    fitter = LevMarLSQFitter()
    y, x = np.indices(data.shape)
    gfit = fitter(g_init, x, y, data, weights=weights)
    
    fwhm_y = gfit[1].y_stddev.value*gaussian_sigma_to_fwhm
    fwhm_x = gfit[1].x_stddev.value*gaussian_sigma_to_fwhm 
    
    return np.array([fwhm_x, fwhm_y])

def select_point_source():
    

#read in instant calibration version of guided field (otherwise will need a debiasing step if using pointed raw images)
#The instant calibration is done by NOIRLab community pipeline, should already have correction WCS solution.
g_path = '/Users/reneekey/Documents/drift_testset/g_ooi_exp012923.fits'
r_path = '/Users/reneekey/Documents/drift_testset/r_ooi_exp013253.fits'

#select two CCD names - Northern Edge (S31), Southern Middle (N16)
ccd_names = ['N16', 'S31']
i = 1

with fits.open(g_path) as g_file:
    g_hdr = g_file[0].header
    g_img_hdr = g_file[ccd_names[i]].header
    g_img_data = g_file[ccd_names[i]].data

w_g = WCS(g_img_hdr)   #read in WCS of both images, this is corrected by NOIRLab

with fits.open(r_path) as r_file:
    r_hdr = r_file[0].header
    r_img_hdr = r_file[ccd_names[i]].header
    r_img_data = r_file[ccd_names[i]].data

w_r = WCS(r_img_hdr)

#Remove cosmic rays via LACosmic (use median combination of image values as better theshold estimate than the standard instrumental report from NOIRlab)
gain_g = np.median([g_img_hdr['GAINA'], g_img_hdr['GAINB']], axis = 0)
readnoise_g = np.median([g_img_hdr['RDNOISEA'], g_img_hdr['RDNOISEA']], axis = 0)
satlevel_g = np.median([g_img_hdr['SATURATA'], g_img_hdr['SATURATB']], axis = 0)

mask_g, data_g = astroscrappy.detect_cosmics(g_img_data, gain=gain_g, readnoise=readnoise_g, satlevel=satlevel_g, 
                                            cleantype='medmask')

gain_r = np.median([r_img_hdr['GAINA'], r_img_hdr['GAINB']], axis = 0)
readnoise_r = np.median([r_img_hdr['RDNOISEA'], r_img_hdr['RDNOISEA']], axis = 0)
satlevel_r = np.median([r_img_hdr['SATURATA'], r_img_hdr['SATURATB']], axis = 0)

mask_r, data_r = astroscrappy.detect_cosmics(r_img_data, gain=gain_r, readnoise=readnoise_r, satlevel=satlevel_r, 
                                            cleantype='medmask')


#This section is automated to read DAOPhotPSFPhotom parameters as estimate from image without needing user inputs.
#Threshold detection limit, finding the average background
thresh_init_g = sigma_clipped_stats(data_g, sigma=3.0)[1] #(median value)
thresh_init_r = sigma_clipped_stats(data_r, sigma=3.0)[1] #(median value)

#Automate the FWHM detection using bright, isolated reference stars
target_num = 20 #set number of ePSF reference stars
pix_scale = u.pixel_scale(g_hdr['PIXSCAL1']*u.arcsec/u.pixel) #global value across any DECam filter
size = 15  #edge case removal, pixel width from edge of image to use in extracting single reference stars
hsize = (size - 1) / 2

#Find all bright stars as a first pass
g_ref_peaks = find_peaks(data_g, threshold=thresh_init_g)  
r_ref_peaks = find_peaks(data_r, threshold=thresh_init_r)  

'''SECTION A: find well behaved, bright, isolated stars:
To use with building the PSF model for all stars'''

#find nearest neighbour to each star
#since WCS is internal to chip image, no recalibration needed as offset is ~linearly constant
wcs_peaks_g = w_g.pixel_to_world(g_ref_peaks['x_peak'], g_ref_peaks['y_peak'])
g_match, g_sep, g_d3sep = match_coordinates_sky(wcs_peaks_g, wcs_peaks_g, nthneighbor = 2) #nthneighbour =2 for nearest neighbour, 1 for self-matching
g_sep_arcsec = g_sep.arcsecond*u.arcsecond  #annoying Astropy convention
g_sep_pix = g_sep_arcsec.to(u.pixel, pix_scale)
g_ref_peaks['ref_id'] = np.arange(len(g_ref_peaks))
g_ref_peaks['neighbour'] = g_match
g_ref_peaks['pix_sep'] = g_sep_pix.value

#cut reference peaks based on separation in pixels and then brightness
g_sep_cut = np.quantile(g_sep_pix.value,(0.99))    #the most isolated stars
g_sep_mask = g_ref_peaks['pix_sep'] > g_sep_cut
g_iso_ref = g_ref_peaks[g_sep_mask]

g_flux_cut = np.quantile(g_iso_ref['peak_value'].data,(0.5)) #upper 50% of brightest stars
g_flux_mask = g_iso_ref['peak_value'] > g_flux_cut
g_bright_ref = g_iso_ref[g_flux_mask]

#sort and add in overfill to target number slice to make sure extractions fall within edge of chip
g_bright_ref = g_bright_ref[g_bright_ref['pix_sep'].argsort()][::-1][0:target_num+10]   

g_x = g_bright_ref['x_peak']  
g_y = g_bright_ref['y_peak']  
g_edge = ((g_x > hsize) & (g_x < (data_g.shape[1] -1 - hsize)) &
        (g_y > hsize) & (g_y < (data_g.shape[0] -1 - hsize)))  

g_stars_tbl = Table()
g_stars_tbl['x'] = g_x[g_edge]  
g_stars_tbl['y'] = g_y[g_edge]  

g_median_val = sigma_clipped_stats(data_g, sigma=2.0)[1]
g_data_clean = data_g - g_median_val  

g_nddata = NDData(data=g_data_clean) 
g_stars = extract_stars(g_nddata, g_stars_tbl, size=25)


#use only sources with circular morphology (removes saturated sources and galaxies)
g_symmetry = []
for star in g_stars:
    cat = data_properties(star)
    columns = ['label', 'xcentroid', 'ycentroid', 'semimajor_sigma',
           'semiminor_sigma', 'orientation']
    tbl = cat.to_table(columns=columns)
    a = tbl['semimajor_sigma']
    b = tbl['semiminor_sigma']
    sym = a/b
    g_symmetry.append(sym.value[0])
g_iso_source = np.where(np.array(symmetry) < 1.1)[0]


g_points = [g_stars[i].data for i in g_iso_source]
g_fwhms = [np.median(FWHM_2dg(star)) for star in g_points]
fwhm_init_g = sigma_clipped_stats(g_fwhms, sigma=3.0)[1]  #gives us the FWHM from the reference stars

#Redo the above code for r-band image (Could remove this and make this whole section a function)
wcs_peaks_r = w_r.pixel_to_world(r_ref_peaks['x_peak'], r_ref_peaks['y_peak'])
r_match, r_sep, r_d3sep = match_coordinates_sky(wcs_peaks_r, wcs_peaks_r, nthneighbor = 2) 
r_sep_arcsec = r_sep.arcsecond*u.arcsecond  #annoying Astropy convention
r_sep_pix = r_sep_arcsec.to(u.pixel, pix_scale)
r_ref_peaks['ref_id'] = np.arange(len(r_ref_peaks))
r_ref_peaks['neighbour'] = r_match
r_ref_peaks['pix_sep'] = r_sep_pix.value

#cut reference peaks based on separation in pixels and then brightness
r_sep_cut = np.quantile(r_sep_pix.value,(0.99))    #the most isolated stars
r_sep_mask = r_ref_peaks['pix_sep'] > r_sep_cut
r_iso_ref = r_ref_peaks[r_sep_mask]

r_flux_cut = np.quantile(r_iso_ref['peak_value'].data,(0.5)) #upper 50% of brightest stars
r_flux_mask = r_iso_ref['peak_value'] > r_flux_cut
r_bright_ref = r_iso_ref[r_flux_mask]

#sort and add in overfill to target number slice to make sure extractions fall within edge of chip
r_bright_ref = r_bright_ref[r_bright_ref['pix_sep'].argsort()][::-1][0:target_num+10]   


r_x = r_bright_ref['x_peak']  
r_y = r_bright_ref['y_peak']  
r_edge = ((r_x > hsize) & (r_x < (data_r.shape[1] -1 - hsize)) &
        (r_y > hsize) & (r_y < (data_r.shape[0] -1 - hsize)))  

r_stars_tbl = Table()
r_stars_tbl['x'] = r_x[r_edge]  
r_stars_tbl['y'] = r_y[r_edge]  

r_median_val = sigma_clipped_stats(data_r, sigma=2.0)[1]
r_data_clean = data_r - r_median_val  

r_nddata = NDData(data=r_data_clean) 
r_stars = extract_stars(r_nddata, r_stars_tbl, size=25)

r_symmetry = []
for star in r_stars:
    cat = data_properties(star)
    columns = ['label', 'xcentroid', 'ycentroid', 'semimajor_sigma',
           'semiminor_sigma', 'orientation']
    tbl = cat.to_table(columns=columns)
    a = tbl['semimajor_sigma']
    b = tbl['semiminor_sigma']
    sym = a/b
    r_symmetry.append(sym.value[0])
r_iso_source = np.where(np.array(symmetry) < 1.1)[0]


r_points = [r_stars[i].data for i in r_iso_source]
r_fwhms = [np.median(FWHM_2dg(star)) for star in r_points]
fwhm_init_r = sigma_clipped_stats(r_fwhms, sigma=3.0)[1]

#Use FWHM to define the PSF sigma guess for the PRF model
g_sigma_psf = fwhm_init_g * gaussian_fwhm_to_sigma

#Run photometry on the CCD images
fitter = LevMarLSQFitter()
g_psf_model = IntegratedGaussianPRF(sigma=g_sigma_psf) 
photometry_g = DAOPhotPSFPhotometry(crit_separation = 3*fwhm_init_g, threshold = thresh_init_g, 
                                    fwhm = fwhm_init_g, sigma = 3, psf_model=g_psf_model, fitter=LevMarLSQFitter(),
                                    niters=1, fitshape=(7,7))

table_g = photometry_g(image=data_g)   
residual_g = photometry_g.get_residual_image()

#background threshold changes for r_band as exposure is longer
r_sigma_psf = fwhm_init_r * gaussian_fwhm_to_sigma
r_psf_model = IntegratedGaussianPRF(sigma=r_sigma_psf)
photometry_r = DAOPhotPSFPhotometry(crit_separation = 3*fwhm_init_r, threshold = thresh_init_r, 
                                    fwhm = fwhm_init_r, sigma = 3, psf_model=r_psf_model, fitter=LevMarLSQFitter(),
                                    niters=1, fitshape=(7,7))

table_r = photometry_r(image=data_r)   
residual_r = photometry_r.get_residual_image()

#Use WCS convention to get estimate of local RA, Dec for X-match between r and g band
wcs_g = w_g.pixel_to_world(table_g['x_fit'], table_g['y_fit'])
wcs_r = w_r.pixel_to_world(table_r['x_fit'], table_r['y_fit'])
match, sep, d3sep = match_coordinates_sky(wcs_r, wcs_g) 

#Make final frame
comb_phot = pd.DataFrame()

#Global info
comb_phot['chp'] = [ccd_names[i]]*len(wcs_r)
comb_phot['skysep_rg'] = sep.deg             

#g-band info
comb_phot['id_g'] = table_g['id'][match]
comb_phot['x_g'] = table_g['x_fit'][match]
comb_phot['y_g'] = table_g['y_fit'][match]
comb_phot['flux_g'] = table_g['flux_fit'][match]
comb_phot['flux_unc_g'] = table_g['flux_unc'][match]
comb_phot['ra_g'] = np.array([i.ra.degree for i in wcs_g])[match]
comb_phot['dec_g'] = np.array([i.dec.degree for i in wcs_g])[match]

#r-band info
comb_phot['id_r'] = table_r['id']
comb_phot['x_r'] = table_r['x_fit']
comb_phot['y_r'] = table_r['y_fit']
comb_phot['flux_r'] = table_r['flux_fit']
comb_phot['flux_unc_r'] = table_r['flux_unc']
comb_phot['ra_r'] = np.array([i.ra.degree for i in wcs_r])
comb_phot['dec_r'] = np.array([i.dec.degree for i in wcs_r])

'''This is where a GAIA TAP query should X_match between local and external frames'''
#send comb_phot to csv file
comb_phot.to_csv('catalogue_photometry_S31.csv', sep = ',')