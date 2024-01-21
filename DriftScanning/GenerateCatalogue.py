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
class Cataloguer:

  #Need a function that initialises what ccd_name and file_path
  #def __init__(self):
  #Image = ImageImport(ccd_name, filepath_
  #return Image
  
  def pointed_catalogue(self, ccd_name, filepath):
    Image = ImageImport(ccd_name, filepath)
    
    CRTask = cosmicray_removal(Image)
    REFtask = select_reference_stars(Image.data, Image.wcs, Image.hdr['PIXSCAL1'])
  
    #Use FWHM to define the PSF sigma guess for the PRF model
    psf_fwhm = sigma_clipped_stats(REFtask['fwhm'].values, sigma=3.0)[1]
    sigma_psf = psf_fwhm * gaussian_fwhm_to_sigma
    background = Image.hdr['BACKGROUND'] #DEBUG THIS HEADER NAME
    
    #Run photometry on the CCD images
    fitter = LevMarLSQFitter()
    psf_model = IntegratedGaussianPRF(sigma=sigma_psf) 
    photometry = DAOPhotPSFPhotometry(crit_separation = 3*psf_fwhm, threshold = background, 
                                      fwhm = psf_fwhm, sigma = 3, psf_model=psf_model, fitter=LevMarLSQFitter(),
                                      niters=1, fitshape=(7,7))
  
    table = photometry(image=Image.data)   
    residual = photometry.get_residual_image()
    
    #Use WCS convention to get estimate of local RA, Dec for X-match between r and g band
    sky_positions = w.pixel_to_world(table_g['x_fit'], table_g['y_fit'])
    
    #Make final frame
    photometry_dataframe = pd.DataFrame()
    comb_phot['chp'] = [ccd_names[i]]*len(w)           
    comb_phot['id'] = table['id'][match]
    comb_phot['x'] = table['x_fit'][match]
    comb_phot['y'] = table['y_fit'][match]
    comb_phot['flux'] = table['flux_fit'][match]
    comb_phot['flux_unc'] = table['flux_unc'][match]
    comb_phot['ra'] = np.array([i.ra.degree for i in w])[match]
    comb_phot['dec'] = np.array([i.dec.degree for i in w])[match]
    
    return(comb_phot)


#Call this in the main function
comb_phot.to_csv('catalogue_photometry_S31.csv', sep = ',')
