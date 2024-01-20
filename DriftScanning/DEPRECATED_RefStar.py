
##DEPRECATED: NOW A FUNCTION IN MAINCAT.PY

import pandas as pd
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u



#Uses a single chip on the DECam FOV to find a target number of reference stars for both astrometric calibration of drifts on same chip 
# and for atmospheric calibration pairings for S/N improvements in the ePSF fitting of drifts.
# NOTE: all comparisons to catalogue are lead by the r-band photometry of the pointed images as this is closest to the VR band of the drifts.


g_path = '/Users/reneekey/Documents/drift_testset/g_ooi_exp012923.fits'
r_path = '/Users/reneekey/Documents/drift_testset/r_ooi_exp013253.fits'

ccd_names = ['N16', 'S31']
i = 0
with fits.open(g_path) as g_file:
    g_hdr = g_file[0].header
    g_img_hdr = g_file[ccd_names[i]].header
    g_img_data = g_file[ccd_names[i]].data

w_g = WCS(g_img_hdr)

with fits.open(r_path) as r_file:
    r_hdr = r_file[0].header
    r_img_hdr = r_file[ccd_names[i]].header
    r_img_data = r_file[ccd_names[i]].data

w_r = WCS(r_img_hdr)

#read in GAIA DR3 photometry catalogue per chp
catalogue = pd.read_csv('/Users/reneekey/Documents/Drift_Scripts/gaia_N16_crossmatch.csv', sep=',')

#select reference cases from longer band centroids
num_reference = 50
edge_crit = 200      #number of pixels inwards from the edges to ignore (safeguard against pointing displacements across the night)
chip_dim = [0, 1000, 0, 1000]      #x1, x2, y1, y2

#edge condition for stars not within 100 pixels of the each edge of the chip
cand = catalogue[(catalogue['x_r'] > chip_dim[0]+edge_crit) & (catalogue['x_r'] < chip_dim[1]-edge_crit) & 
              (catalogue['y_r'] > chip_dim[2]+edge_crit) & (catalogue['y_r'] < chip_dim[3]-edge_crit)]

#select step 1 reference stars with a quartile magnitude cut off
flux_quantile = cand['flux_r'].quantile(0.75)
bright = cand[cand['flux_r'] > flux_quantile]

#select step 2 with search around star condition to avoid overlapping drifts
skycoord = SkyCoord(ra=bright['RAJ2000']*u.degree, dec=bright['DEJ2000']*u.degree)
idxc, idxcatalog, d2d, d3d = skycoord.search_around_sky(skycoord, 0.0008*u.degree)

#Get histogram of neighbours within 3 arcseconds of each star
hist = np.histogram(idxc, bins = len(bright), range = (idxc.min(), idxc.max())) #number of bins = 1 bin per star in dataframe 'bright'
bright['SAS_3arcsec']  = hist[0]
#choose final N reference stars
reference = bright[bright['SAS_3arcsec'] == 1]  #choose only stars that have 1 (is self-matched) match within 3arcseconds.

#NOTE: we do want to select the brighest sources from the reference images since these are longer bands, 
# even saturated stars will translate into 'normal' fluxes on the detector since the shorter exp timing 
# combined with the trailing effect will dampen peak brightess
star_list = reference.sort_values(by=['flux_r'], ascending = False).head(num_reference)
star_list.to_csv(f'{ccd_names[i]}_reference_stars.csv', sep = ',')
