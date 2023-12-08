import numpy as np

#plots
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.patches as mpatches
import cmasher as cm

from astropy.io import fits
from astropy.nddata.utils import Cutout2D

from photutils.aperture import CircularAperture, RectangularAperture
from skimage.draw import circle_perimeter, rectangle_perimeter

def aperture_sum(apertures, data_shape):
    '''
    Function takes a list of apertures and calculates the image of the aperture onto an array of size data_shape.
    
    apertures: list of Aperture objects from Photutils
    
    data_shape: tuple or list of the array shape as number of pixels (nx, ny) 

    returns the combined aperture as a boolean mask.
    '''
    mask_sum = sum(aper.to_mask(method = 'center').to_image(data_shape) for aper in apertures)
    aperture = np.where(mask_sum != 0, 1, mask_sum)
    return(aperture)

def stadium_perimeter(cutout_data, x0, y0, length, radius, pad = None):
    import cmasher as cm
    import matplotlib.patches as mpatches
    '''
    This function uses combinations of scikit.draw perimeter objects to draw the contours of a stadium apeture
    Note that the apeture is 'jagged', it traces whole pixels; a direct result of ndarray indexing.
    NOTE: This function is only used for plotting apetures, since some pixel conventions differ from Photutils.
    Potential: there may be a smart way of joining the stadium across fractions of pixels to form a precise apeture.
    
    chip_data: the full, relevant data array of the CCD chip. Passing a slice or subsection of the data may
    result in Cutout2D value errors. 
    
    cutout_size: the shape of the Cutout2D object around x0, y0.

    (x0, y0): the pixel x,y image centroids of detected driftscans, scaled to the image cutout.
    
    length: the driftscan length in the x-direction. 
    
    radius: the radius of the semi-circular caps of the stadium shape, and half the width of the rectangular body
    in the y-direction.

    returns: a mask of stadium perimeter edges for a single star. 
    '''
    #plotting global definition of colourmaps
    drift_cmap = cm.rainforest
    contour_cmap = cm.take_cmap_colors(cm.guppy, 2, return_fmt='hex')
    
    
    #Make individual apeture perimiters
    contour = np.zeros(cutout_data.shape)
    rlhs, clhs = circle_perimeter(y0,x0 - length//2, radius)
    rrhs, crhs = circle_perimeter(y0,x0 + length//2, radius)
    
    start = ((y0-1)+radius, x0 - length//2)
    end=((y0+1)-radius, x0 + length//2)
    rRect, cRect = rectangle_perimeter(start = start, end=end, shape=cutout_data.shape)
    
    #define the additive contour lines
    contour[rlhs, clhs] = 1
    contour[rrhs, crhs] = 1
    contour[rRect, cRect] = 1
    
    #hollow out the inside of the apeture to avoid plotting intersections and cross-hairs
    contour[end[0]:start[0]+1, start[1]-1:end[1]+2] = 0  
    
    
    #if pad is not None, define a second aperture to plot the annulus of the driftscan
    if pad:
        outer_contour = np.zeros(cutout_data.shape)
        rlhs, clhs = circle_perimeter(y0,x0 - length//2, radius+pad)
        rrhs, crhs = circle_perimeter(y0,x0 + length//2, radius+pad)

        start = ((y0-1)+(radius+pad), x0 - length//2)
        end=((y0+1)-(radius+pad), x0 + length//2)
        rRect, cRect = rectangle_perimeter(start = start, end=end, shape=cutout_data.shape)

        
        outer_contour[rlhs, clhs] = 1
        outer_contour[rrhs, crhs] = 1
        outer_contour[rRect, cRect] = 1
        outer_contour[end[0]:start[0]+1, start[1]-1:end[1]+2] = 0  
    
        #plot
        plt.figure(figsize = (5,5), dpi = 150)
        plt.imshow(cutout_data, norm = LogNorm(), cmap = drift_cmap)
        
        plt.imshow(contour, cmap = colors.ListedColormap(['None', contour_cmap[0]]))
        plt.imshow(outer_contour, cmap = colors.ListedColormap(['None', contour_cmap[1]]))
        
        plt.scatter([x0], [y0], c ='k', marker = '+', s = 100)

        #Combine Legend objects
        labels = {0:'DSI Centroid', 1:'Inner Aperture', 2:'Outer Aperture'}
        combined_cmaps = ['k', contour_cmap[0], contour_cmap[1]]
        patches =[mpatches.Patch(color=combined_cmaps[i],label=labels[i]) for i in labels]
        plt.legend(handles=patches, loc = 'best')
        fig = plt.gcf()
        fig.savefig('aperture_plot.png')
        return(None)
    
    else:
        plt.figure(figsize = (5,5), dpi = 150)
        plt.imshow(cutout_data, norm = LogNorm(), cmap = drift_cmap)
        
        plt.imshow(contour, cmap = colors.ListedColormap(['None', contour_cmap[0]]))
        
        plt.scatter([x0], [y0], c ='k', marker = '+', s = 100)

        #Combine Legend objects
        labels = {0:'DSI Centroid', 1:'Inner Aperture'}
        combined_cmaps = ['k', contour_cmap[0]]
        patches =[mpatches.Patch(color=combined_cmaps[i],label=labels[i]) for i in labels]
        plt.legend(handles=patches, loc = 'best')
        fig = plt.gcf()
        fig.savefig('aperture_plot.png')
        return(None)
def stadium_annulus(chip_data, x0, y0, cutout_size, length, radius, pad, plot_star = True, verbose_save = 0):
        
    '''
    Function uses combinations of Photutils aperture objects to create a stadium annulus and aperture for
    a driftscan.
    
    chip_data: the full, relevant data array of the CCD chip. Passing a slice or subsection of the data may
    result in Cutout2D value errors. 

    (x0, y0): the pixel x,y image centroids of detected driftscans.
    
    cutout_size: the size of the cutout centered around the driftscan (in pixels)
    
    length: the driftscan length in the x-direction. 
    radius: the radius of the semi-circular caps of the stadium shape, and half the width of the rectangular body
    in the y-direction.

    pad: the number of pixels within the annulus, projected around the perimeter of the inner stadium aperture. 

    plot_stars:  (default = True). If True, creates an imshow figure of the cutout of the star with inner and outer stadium apertures. 
    Plot is saved to the working directory. If False, no plots are made
    
    verbose_save: (default = 0) defines the amount of annulus information saved to working directory
    0: no information saved, function returns as normal.
    1: the data within the annulus is saved
    2: both the mask and data of the annulus is saved #check if I want to project this to chip_data x,y coordinates
    '''
    
    #make a cutout2D object of the drift around x0, y0
    cutout = Cutout2D(full_data, (x0, y0), (cutout_size, cutout_size))
    xi, yi = cutout.to_cutout_position((x0, y0))
    
    aperRect = RectangularAperture((xi, yi), w = length, h = radius*2)
    aperCirc_LHS = CircularAperture((xi - length//2, yi), radius)
    aperCirc_RHS = CircularAperture((xi + length//2, yi), radius)

    inner_aperture = aperture_sum([aperRect, aperCirc_LHS, aperCirc_RHS], cutout.shape)

    #Make an annulus using the same method but concentric circles
    annuRect = RectangularAperture((xi, yi), w = length + pad, h = (radius+pad)*2)
    annuCirc_LHS = CircularAperture((xi - length//2, yi), radius+pad)
    annuCirc_RHS = CircularAperture((xi + length//2, yi), radius+pad)

    outer_aperture = aperture_sum([annuRect, annuCirc_LHS, annuCirc_RHS], cutout.shape)
    annulus_mask = outer_aperture - inner_aperture
    annulus_data = cutout.data*annulus_mask
    
    #calculate the sky within the annulus with sigma_clipping to avoid blended pixels
   
    clipped_sky = sigma_clip(annulus_data, sigma=2, maxiters=10).data

    #verbose saves and plots
    if plot_star:
        stadium_perimeter(cutout.data, xi, yi, length, radius, pad = pad)
        
    if verbose_save == 0:
        pass
    if verbose_save == 1:
        #save annulus data as an array
        np.save(f'annulus_data_{dsi_ID}.npy', annulus_data)
    if verbose_save == 2:
        #save annulus data and annulus mask as arrays
        np.save(f'annulus_eval_{dsi_ID}.npy', [annulus_data, annulus_mask])
    
    return(clipped_sky)

VR_path = '/Test_exposures/VR_N16_ori_exp1058800.fits'

hdulist = fits.open(VR_path)
full_data = hdulist[1].data

length = 40
radius = 7
pad = 5
cutout_size = 100

x0 = 67 + 240 + 1500   #known centroids from the astrometry positioning
y0 = 21 + 410 + 1000

sky = stadium_annulus(full_data, x0, y0, cutout_size, length, radius, pad)

#TO ADD: DriftAstrom.py builds a dataframe of x0, y0 positions and the sky function is processed as lambda:row()
#df['sky'] = df.apply(lambda row: stadium_annulus(full_data, row['x0'], row['y0'], cutout_size, length, radius, pad), axis = 1)
