import pandas as pd

import astropy.units as u
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import fit_wcs_from_points
from astropy.coordinates import SkyCoord, match_coordinates_sky

from astropy.modeling.models import Box2D
from astropy.convolution.kernels import Model2DKernel
from astropy.convolution import convolve

from photutils.detection import find_peaks
from photutils.centroids import centroid_sources, centroid_com

#Read in VR image (eventually using utils class)
VR_path = '/Users/reneekey/Documents/Drift_Scripts/TestImages/c4d_211211_031625_ooi_VR_v1.fits.fz'
ccd_name = 'N16'
with fits.open(VR_path) as hdul_VR:
    header = hdul_VR[0].header
    ccd_header = hdul_VR[ccd_name].header
    ccd_VR = hdul_VR[ccd_name].data
    w_VR = WCS(hdul_VR[ccd_name].header)

#Read in main catalogue
main_catalogue = pd.read_csv("path/to/catalogue")

def drift_width(fwhm_estimate, unit = 'pixel', pixelscale = None):
    #Estimate width of star (here we use Noirlab Community pipeline FITs headers in this function call)
    if unit == 'arcsec':
        fwhm_estimate = fwhm_estimate * pixelscale
      
    return (fwhm_estimate)


def drift_length(track_rate, exp_time, unit = 'arcsec', pixelscale = 0.27):
    #calculates drift length given user defined tracking rate, and FITs header cards for the
    #exposure duration and pixelscale
    #NOTE: rate is given in arcsec/second ---> exp_time must be in seconds
    
    if unit == 'arcsec':
        length = track_rate * exp_time / pixelscale
    
    if unit == 'pixel':
        length = track_rate * exp_time
    
    return (length)

def drift_model(a, b, unit = 'pixel', pixelscale = None):
    #define a Box2D model that makes a theoretical model drift given drift rate, exposure time
    #or width, length depending on unit
    
    #unit are pixel or 'arcsec'
    if unit == 'arcsec':
        a = a * pixelscale
        b = b * pixelscale
        
    model = Box2D(x_0 = 0, x_width = b, y_0 = 0, y_width = a)
    return model

def drift_centroids(image, background, WCS, drift_model, Nsources):
    
    """
    image: the drifted VR image 
    
    background: the image background or a defined threshold level to search for flux peaks. Below background
    values, no peaks will be defined (i.e for faint drifts).
    
    WCS: the original WCS of the image
    
    drift_model: a simplistic model of the typical driftscan (2D rectangle of constant flux)
    
    Nsources: the number of sources to search for. From decreasing brightness the first Nsources will be
    returned
    """
    
    shape = int(drift_model.x_width.value), int(drift_model.y_width.value) #must be integer values for kernel
    
    drift_kernel = Model2DKernel(drift_model, x_size=int(shape[0]*2 - 1)) #what does xsize do?
    drift_conv = convolve(image, drift_kernel)
    drift_map = find_peaks(drift_conv, background, box_size=20, npeaks = Nsources, wcs = WCS) #redefine 
    
    x_peaks = drift_map['x_peak']
    y_peaks = drift_map['y_peak']
    
    x, y = centroid_sources(image, x_peaks, y_peaks, box_size = shape, centroid_func=centroid_com)
    
    drift_map['x_cent'] = x
    drift_map['y_cent'] = y
    
    skycoord = WCS.pixel_to_world(drift_map['x_cent'], drift_map['y_cent'])
    drift_map['ra_cent'] = [i.ra.deg for i in skycoord]
    drift_map['dec_cent'] = [i.dec.deg for i in skycoord]
    
    return drift_map, drift_conv

def drift_centroids(image, background, WCS, drift_model, Nsources):
    
    """
    image: the drifted VR image 
    
    background: the image background or a defined threshold level to search for flux peaks. Below background
    values, no peaks will be defined (i.e for faint drifts).
    
    WCS: the original WCS of the image
    
    drift_model: a simplistic model of the typical driftscan (2D rectangle of constant flux)
    
    Nsources: the number of sources to search for. From decreasing brightness the first Nsources will be
    returned
    """
    
    shape = int(drift_model.x_width.value), int(drift_model.y_width.value) #must be integer values for kernel
    
    drift_kernel = Model2DKernel(drift_model, x_size=int(shape[0]*2 - 1)) #what does xsize do?
    drift_conv = convolve(image, drift_kernel)
    drift_map = find_peaks(drift_conv, background, box_size=20, npeaks = Nsources, wcs = WCS) #redefine 
    
    x_peaks = drift_map['x_peak']
    y_peaks = drift_map['y_peak']
    
    x, y = centroid_sources(image, x_peaks, y_peaks, box_size = shape, centroid_func=centroid_com)
    
    drift_map['x_cent'] = x
    drift_map['y_cent'] = y
    
    skycoord = WCS.pixel_to_world(drift_map['x_cent'], drift_map['y_cent'])
    drift_map['ra_cent'] = [i.ra.deg for i in skycoord]
    drift_map['dec_cent'] = [i.dec.deg for i in skycoord]
    
    return drift_map, drift_conv


def project_centroids(centroid_table, WCS, anchor, drift_model):
    
    """Take a centroided drift position and shift it left or right to the start of end of the drift, or keep it
    at the midpoint of the drift.
    
    This function anchors the drift ra, dec to a specific point along the drift. 
    The exposure starts with specific coordinates of the field, and our specific tracking rate in 
    declination (along the y-axis) builds the trail of the drift. Technically the WCS is built from the
    initial starting image of the start at exp_time = 0 at position (x, y) = (0,0), 
    and at exp_time = 20 seconds, the star is elsewhere on the detector image (+0, + 40) pixels.
    
    Our positions in the WCS shift should be measured with respect to the star position at the beginning
    of the exposure.
    
    The opposite anchor point ('r') would be equivalent to a reversed tracking rate of -0.5 arcsecs/sec in dec. 
    I've included this point as a input in case other drift datasets have different tracking directions. 
    
    Eventually this could become more sophisticated, where the user has tracking_vector = (0.5, 0.1) in (ra, dec)
    and the projection would work out the relevant x and y anchor points for the starting location of the source. 
    
    centroid_table: A csv/table format of the centroid locations of driftscans
    
    anchor: a string to represent the anchor location (x,y), For now, this is a linear shift in y 
    'r' is a leftways move to +y_offset, 'l' is a rightways move to -y_offset. 'm' is midway, and keeps the
    anchor at the centroid location
    
    drift_model: an astropy 2D rectangular model of the driftscan
    """
    
    
    #from the centroid locations, plop down a centroided model and use a specific x,y pixel as the anchor
    #for all drifts. i.e 'l' = left, 'r' = right, 'm' = middle/centroid....
    
    half_y = model.y_width.value/2
    anchor_x = 0  #no linear shift to centroid midline position
    
    if anchor == 'r':
        anchor_y = -half_y  #no linear shift to centroid midline position
        
    if anchor == 'l':
        anchor_y = half_y
        
    if anchor == 'm':
        anchor_y = 0
        
    centroid_table['x_a'] = centroid_table['x_cent'] + anchor_x
    centroid_table['y_a'] = centroid_table['y_cent'] + anchor_y
    
    
    skycoord = WCS.pixel_to_world(centroid_table['x_a'], centroid_table['y_a'])
    centroid_table['ra_a'] = [i.ra.deg for i in skycoord]
    centroid_table['dec_a'] = [i.dec.deg for i in skycoord]

    return centroid_table

def match_drift_ref(ref_cat_pos, centroid_pos, wcs):
    """
    ref_cat_pos = list or tuple of ra, dec positions for main catalogue
    centroid_table = table or list of ra, dec positions for VR images
    """
    
    #match between ra and dec positions on sky between the reference stars and the drifts
    #Noted (20th Feb 2024) that this function is slow, and we could speed up using Stilts
    ref_coords = SkyCoord(ref_cat_pos[0]*u.deg, ref_cat_pos[1]*u.deg, frame = 'fk5')
    drift_coords = SkyCoord(centroid_pos[0]*u.deg, centroid_pos[1]*u.deg, frame = 'fk5')

    idx, d2d, d3d = drift_coords.match_to_catalog_sky(ref_coords)

    #apply linear shift (Renee to test on Thursday/Friday)
    ref_pixel = wcs.world_to_pixels(ref_coords)
    xshift = ref_pixel[0] - centroid_pos['x_a']
    yshift = ref_pixel[1] - centroid_pos['y_a']

    drift_xy = centroid_pos['x_a'] + xshift, centroid_pos['y_a'] + yshift
    #take the matched drift x and y positions to the reference star skycoordinates in g, r images
    w_SHIFT = fit_wcs_from_points(xy = np.array(drift_xy), world_coords = ref_coords, projection='TAN')
    
    return w_SHIFT
