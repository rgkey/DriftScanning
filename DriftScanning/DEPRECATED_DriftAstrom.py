import astropy 
import numpy as np
import scipy as scp
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
from astropy.io import fits
import math

#for cosmic ray removal
import astroscrappy

from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.coordinates import match_coordinates_sky
from astropy.wcs.utils import fit_wcs_from_points

import matplotlib.pyplot as plt
from matplotlib import colors
from astropy.visualization import SqrtStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from astropy.visualization.wcsaxes import SphericalCircle
from photutils.aperture import CircularAperture, SkyCircularAperture

#for centroiding
from astropy.modeling.models import Box2D
from astropy.convolution.kernels import Model2DKernel
from astropy.convolution import convolve
from photutils.detection import find_peaks
from photutils.centroids import centroid_sources, centroid_com

g_path = '/Users/reneekey/Documents/drift_testset/g_ooi_exp012923.fits'
r_path = '/Users/reneekey/Documents/drift_testset/r_ooi_exp013253.fits'
vr_pointed = '/Users/reneekey/Documents/drift_testset/c4d_2019B0071_exp917982.fits'

ccd_names = ['N16', 'S31']
i = 1
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

with fits.open(vr_pointed) as vr_file:
    vr_hdr = vr_file[0].header
    vr_img_hdr = vr_file[ccd_names[i]].header
    vr_img_data = vr_file[ccd_names[i]].data

w_vr = WCS(vr_img_hdr)

#externally cross-match with Gaia to the closest match
catalogue = pd.read_csv(f'/Users/reneekey/Documents/Drift_Scripts/VR2019_crossmatch_gaia.csv', sep=',')

#read in 2019 data in VR, same exposure length (gives better star match)

VR_path = f'/Users/reneekey/Documents/drift_testset/VR_{ccd_names[i]}_ori_exp1058800.fits'
bias1_path = '/Users/reneekey/Documents/drift_testset/VR_bias_zri_exp043052.fits'
bias2_path = '/Users/reneekey/Documents/drift_testset/VR_bias_zri_exp043115.fits'


with fits.open(VR_path) as VR_file:
    VR_hdr = VR_file[0].header
    VR_img_hdr = VR_file[1].header
    VR_img_data = VR_file[1].data

w_VR = WCS(VR_img_hdr)

with fits.open(bias1_path) as bias1_file:
    bias1_hdr = bias1_file[0].header
    bias1_img_hdr = bias1_file['S31'].header
    bias1_img_data = bias1_file['S31'].data

with fits.open(bias2_path) as bias2_file:
    bias2_hdr = bias2_file[0].header
    bias2_img_hdr = bias2_file['S31'].header
    bias2_img_data = bias2_file['S31'].data

clean_mask, clean_VR = astroscrappy.detect_cosmics(VR_img_data, gain=4.0, 
                                    readnoise=7.0, satlevel=65536.0, cleantype='medmask')

#debias
debias_VR = clean_VR - mean_bias

#define drift convolution matched filter kernel
drift_model = Box2D(x_0 = 0, x_width = 40, y_0 = 0, y_width = 5)
drift_kernel = Model2DKernel(drift_model, x_size=49)

#convolve drift with the kernel
drift_conv = convolve(debias_VR, drift_kernel)

#Astometric shift using the brightest stars

N = 200   #find only the N brightest drifts for astrometric shift calculation
drift_map = find_peaks(drift_conv, 500, box_size=20, npeaks = N, wcs = w_VR)

#find COM on unconvolved image
x_init = drift_map['x_peak']
y_init = drift_map['y_peak']
x, y = centroid_sources(clean_VR, x_init, y_init, box_size=19, centroid_func=centroid_com)

#now guestimate on the starting and ending positions of the drifts - 
#perhaps centroid is too far along DSI to be matched

#start of reference DSIs
x_s = x - 30
y_s = y

#end of reference DSIs
x_f = x + 30
y_f = y

DSI_pos = pd.DataFrame()
DSI_pos['xpix_s'] = x_s
DSI_pos['ypix_S'] = y_s

worldcoords = w_VR.pixel_to_world(x_s, y_s)

#add in ra and dec in WCS_VR in order to use astropy
DSI_pos['ra_s'] = [i.ra.deg for i in worldcoords]
DSI_pos['dec_s'] = [i.dec.deg for i in worldcoords]

brightest = catalogue.sort_values(['Gmag'], ascending=[True]).head(N)

#Match between catalogues
cd = SkyCoord(ra=DSI_pos['ra_s'].values*u.degree, dec=DSI_pos['dec_s'].values*u.degree, frame="fk5")
cg = SkyCoord(ra=brightest['RAJ2000'].values*u.degree, dec=brightest['DEJ2000'].values*u.degree, frame="icrs")
idxg, d2dg, d3dg = cd.match_to_catalog_sky(cg, nthneighbor=1)

offset = pd.DataFrame()
offset['x_DSI'] = DSI_pos['xpix_s']
offset['x_point'] = brightest['x'].values[idxg]
offset['y_DSI'] = DSI_pos['ypix_S']
offset['y_point'] = brightest['y'].values[idxg]
offset['idxg'] = idxg
offset['d2dg'] = d2dg.deg
offset['ra'] = brightest['RAJ2000'].values[idxg]
offset['dec'] = brightest['DEJ2000'].values[idxg]

#go through gaia xmatch VR catalogue and remove stars within a radius, get brightest isolated stars
select = offset.sort_values('d2dg', ascending=True).drop_duplicates('idxg').sort_index()
select['xoff'] = np.floor(select['x_DSI'] - select['x_point'])
select['yoff'] = np.floor(select['y_DSI'] - select['y_point'])

histx1, bin_x1 = np.histogram(select['xoff'].values, bins = len(select)//5)
x_bound = np.extract(histx1 == histx1.max(), bin_x1)

histx2, bin_x2 = np.histogram(select['xoff'].values, bins = len(select), range = (x_bound[0] - 10, x_bound[0] + 50))
xoff = np.extract(histx2 == histx2.max(), bin_x2)[0]

histy1, bin_y1 = np.histogram(select['yoff'].values, bins = len(select)//5)
y_bound = np.extract(histy1 == histy1.max(), bin_y1)

histy2, bin_y2 = np.histogram(select['yoff'].values, bins = len(select), range = (y_bound[0] - 10, y_bound[0] + 50))
yoff = np.extract(histy2 == histy2.max(), bin_y2)[0]

#Now use the information of changed X, Y positions to update the WCS header
shifted_pix = np.array([select['x_point'].values+xoff, select['y_point'].values+yoff])
assigned_coords = SkyCoord(ra = select['ra'].values*u.degree, dec = select['dec'].values*u.degree, frame="fk5")

w_SHIFT = fit_wcs_from_points(xy = shifted_pix, world_coords = assigned_coords, projection=w_VR)
vr_skycoords = SkyCoord(catalogue['RAJ2000'],catalogue['DEJ2000'], unit='deg')
x_vr, y_vr = w_SHIFT.world_to_pixel(vr_skycoords)

#Need to save this WCS output to meta data!!
