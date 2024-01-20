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

from DriftScanning.utils import ImageImport

def select_reference_stars(data, WCS,pixscale, Num_target = 20, edge_crit = 0.05, Iso_Perc = 0.99, Flux_Perc = 0.5,
                          Morph_Perc = 0.25, plot_psf_stars = True, timestamp = None):
    
    """This function takes several inputs
    data: CCD image of a field of stars, assumed to be calibrated with the usual bias and flat-fielding. 
    It is suggested that cosmic-ray removal also be completed on the CCD prior to analysis.
    
    WCS: an astropy.WCS.wcs() object containing the WCS header information linked to the data CCD.
    
    pixscale: the pixelscale of the image.
    
    Num_target: the number of target stars to be returned as references for astrometric correction to DSI and/or as
    a sample for psf modelling. Default value is 20 stars. 
    
    edge_crit: the percentage of each axis length on the data image to denote as the image edge. Stars within 
    this edge_criterion are considered to be too close to the edge of the image, and won't be part of the final
    reference star selection. Default value is 5% from each edge = 0.05*data.shape.
    
    Iso_Perc: the percentile threshold to determine the most isolated stars, with isolation calculated as on-image pixel
    distance to nearest on-sky neighbour (using Astropy.match_coordinates_sky()). Default value is the 99th upper
    percentile of all stars. 
    
    Flux_Perc: the percentile threshold to determine the brightest stars. It is suggested to select a moderate value 
    for the Flux threshold. This threhold is more concerned with removing faint stars from the reference catalogue,
    rather than finding the most bright of the stars. It is possible that a stringent upper threshold for Flux_Perc
    may either a: return saturated or contaminated stars, or b: return a minimal sample of reference stars below the
    required Num_target. Default value is 50th upper percentile of all stars. 
    
    Morph_Perc: the percentile threshold to determine the least elliptical sources. Ellipticity is
    measure as '1.0 minus the ratio of the lengths of the semimajor and semiminor axes' (PhotUtils, 2023).
    This threshold works to avoid selecting saturated stars as reference objects. 
    Default values is the lower 25th percentile.
    
    plot_psf_stars: (bool) saves to file the plot each of the reference stars 
    as cutouts on a 5X(Num_target//5) figure. File saved to working directory as '/reference_psf_stars.plt'.
    Default value is True. THIS PARAMETER IS WORK IN PROGRESS
    
    timestamp: a string from the Header or data file containing the date of the exposure. Intended for saving plots.
    Default value is the datetime at plotting, but can take a FITS Header['DATE'] as well.
    
    RETURNS:
    A Pandas dataframe containing the x,y pixel, skycoordinates, peak flux, morphology measurements and nearest
    neighbour pixel separation of Num_target bright, isolated reference stars."""
    
    #converts fractional edge_crit to a number of pixels to avoid searching
    edge_a = data.shape[0]*edge_crit
    edge_b = data.shape[1]*edge_crit

    pix_scale = u.pixel_scale(pixscale*u.arcsec/u.pixel)
    background = sigma_clipped_stats(data, sigma=3.0)[1] #(median value)
    
    all_peaks = find_peaks(data, threshold=background)  
    all_peaks.rename_column('x_peak', 'x') #makes life easier with nndata conventions
    all_peaks.rename_column('y_peak', 'y')
    
    #Match all peaks to nearest neighbour and calculate pixel separation
    wcs_peaks = WCS.pixel_to_world(all_peaks['x'], all_peaks['y'])
    
    #nthneighbour =2 for nearest neighbour, 1 for self-matching
    match, sep, d3sep = match_coordinates_sky(wcs_peaks, wcs_peaks, nthneighbor = 2)
    sep_arcsec = sep.arcsecond*u.arcsecond  #annoying Astropy convention
    sep_pix = sep_arcsec.to(u.pixel, pix_scale)
    
    
    #add data columns
    all_peaks['ref_id'] = np.arange(len(all_peaks))
    all_peaks['neighbour'] = match
    all_peaks['pix_sep'] = sep_pix.value
    
    #Strip edge sources from detected peaks
    central_peaks = all_peaks[(all_peaks['x'] < data.shape[0]-edge_a) & (all_peaks['x'] > edge_a) & 
              (all_peaks['y'] < data.shape[1]-edge_b) & (all_peaks['y'] > edge_b)]
    
    
    #Strip neighbouring sources closer than the Iso_Perc percentile 
    sep_cut = np.quantile(sep_pix.value,(Iso_Perc))    #the most isolated stars
    sep_mask = central_peaks['pix_sep'] > sep_cut
    iso_peaks = central_peaks[sep_mask]

    #Strip the lower Flux_Perc percentile of faint sources
    flux_cut = np.quantile(iso_peaks['peak_value'].data,(Flux_Perc)) #upper 50% of brightest stars
    flux_mask = iso_peaks['peak_value'] > flux_cut
    inter_stars = iso_peaks[flux_mask]  #intermediate selection of isolated, bright stars
    
    #Extract all intermediate stars
    star_nddata = NDData(data=data) 
    all_stars = extract_stars(star_nddata, inter_stars, size=15) #to do, estimate size from image and pixelscale
    
    #Transform to pandas table for functionality
    inter_stars = inter_stars.to_pandas()

    #Mask unuseable extracted stars (from overlapping cutout regions)
    extract = pd.DataFrame(all_stars.center_flat, columns = ['x', 'y'])

    df_all = inter_stars[['x','y']].merge(extract.drop_duplicates(), on=['x','y'], 
                       how='left', indicator=True)

    ref_stars = inter_stars[df_all['_merge'] == 'both']

    #Provide morphological properties of stars THIS COULD BE BETTER
    ref_stars['semimajor_sigma'] = [data_properties(star).semimajor_sigma.value for star in all_stars]
    ref_stars['semiminor_sigma'] = [data_properties(star).semiminor_sigma.value for star in all_stars]
    ref_stars['fwhm'] = [data_properties(star).fwhm.value for star in all_stars]
    ref_stars['ellipticity'] = [data_properties(star).ellipticity.value for star in all_stars]
    
    #Mask on ellipticity
    morph_lower = np.quantile(ref_stars['ellipticity'],(Morph_Perc)) 
    ref_stars = ref_stars[(ref_stars['ellipticity'] <= morph_lower)]
    
    #Return final random sample COULD be a sampling function (linear, highest, lowest, random)
    ref_sample = ref_stars.sample(Num_target)
    
    
    #get the extracted data array for the psf_stars 
    if plot_psf_stars == True: 
        if not timestamp:
            timestamp = str(datetime.datetime.now().isoformat())
        sample_ind = ref_sample.index.values
        psf_stars = [all_stars[i] for i in sample_ind]
        
        nrows = 5
        ncols = int(len(psf_stars)/nrows) #will not plot all stars if Num_target is not a multiple of 5. 

        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 20),
                               squeeze=True)
        ax = ax.ravel()
        for i in range(nrows*ncols):
            norm = simple_norm(psf_stars[i], 'log', percent=99.0)
            ax[i].imshow(psf_stars[i], norm=norm, origin='lower', cmap='viridis')
        fig_save = plt.gcf()
        fig_save.savefig(f'psf_stars_{timestamp}.png')
        plt.show()
        
        
    return(ref_sample)

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



'''SECTION A: find well behaved, bright, isolated stars:
To use with building the PSF model for all stars'''

g_stars_tbl = select_reference_stars(data_g, w_g, g_hdr['PIXSCAL1'])

g_median_val = sigma_clipped_stats(data_g, sigma=2.0)[1]
g_data_clean = data_g - g_median_val  

g_nddata = NDData(data=g_data_clean) 
g_stars = extract_stars(g_nddata, g_stars_tbl, size=25)

'''TO DO: Build this into the select_reference_stars function'''
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

#Redo the above code for r-band image (Am in the process of removing this and make this whole section a function call in main)
r_stars_tbl = select_reference_stars(data_r, w_r, r_hdr['PIXSCAL1'])

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
