import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import scipy.stats                         # for Pearson r
from scipy.optimize import curve_fit
from scipy.stats import norm
from scipy.integrate import simps
import os



def L3(y, LF, SF, MF):    
    N = 2.75                                           # ignore background for a start
    P = LF*SF**N / (np.abs(y-MF)**N + SF**N) 
    return P


#make a little more streamlined with photutils

def S_fun1(xs,ys):         # xs and ys are LH end of driftscan to nearest pix           

# estimate sky level with a couple of 2-sigma clip cycles and subtract from star        

    star = scidata[ys-5:ys+6, xs-6:xs+45]            # star has shape (11,51)           
    sky = scidata[ys-35:ys+35,xs-15:xs+55]           # sky has shape (70,70)            
    sky_flat = sky.flatten()                                                            
    m_sky = sky_flat.mean()                                                             
    std_sky = sky_flat.std()                                                            
    
    c0 = abs(sky_flat - m_sky)< 2.0 * std_sky        # clip cycle0                      
    sky_clip0 = sky_flat[c0]                                                            
    skybar0 = np.median(sky_clip0)                       
    skysig0 = np.std(sky_clip0)
    
    c1 = abs(sky_clip0 - skybar0)< 2.0 * skysig0     # clip cycle1
    sky_clip1 = sky_clip0[c1]
    skybar1 = np.median(sky_clip1)
    skysig1 = np.std(sky_clip1)
    
    star = star - skybar1                            # subtract sky from the star

    return star

ccd_image = '/Users/reneekey/Documents/drift_testset/VR_S31_ori_exp1058800.fits'
bias_frame1 = '/Users/reneekey/Documents/drift_testset/VR_bias_zri_exp043052.fits'
bias_frame2 = '/Users/reneekey/Documents/drift_testset/VR_bias_zri_exp043115.fits'

with fits.open(ccd_image) as VR_file:
    VR_hdr = VR_file[0].header
    VR_img_hdr = VR_file[1].header
    data = VR_file[1].data

#debiasing step
with fits.open(bias_frame1) as bias1_file:
    bias1 = bias1_file[36].data
with fits.open(bias_frame2) as bias2_file:
    bias2 = bias2_file[36].data
bias = np.nanmedian([bias1, bias2], axis = 0)

scidata = data-bias

#reference stars in a list - This will be a read in from another function
s_list = [(1942,2543), (1665, 1757), (1156, 1495), (1479,1141), (725, 3298), (972, 201), (2021, 1577),\
         (324, 247), (247, 1202), (817, 2497), (170, 1295), (894, 1356), (771, 833), (1620, 252)]

L = len(s_list)                                                  # 20 
xstar = np.zeros(L,dtype='int32')
ystar = np.zeros(L,dtype='int32')
star = np.zeros((L,11,51))                             # flat part of star DSI is x = 10:41

for k in np.arange(L):                                                     
    xstar[k] = s_list[k][0]
    ystar[k] = s_list[k][1]    
     
    star[k,:,:] = S_fun1(xstar[k],ystar[k])          
    
# save star data
# np.save('star',star)        
# read as np.load('star.npy')           

#This needs to be fixed!!

print(L, 'template stars extracted')
LI = 32                                     # fit to flat part of DSI: x = 10:41 incl : 32 pixels

L0 = np.zeros((L,LI))                      # parameters for DSI for L3 fits to each of 20 stars
S0 = np.zeros((L,LI))
M0 = np.zeros((L,LI))

eL = np.zeros((L,LI))                      # errors for L0,S0,M0
eS = np.zeros((L,LI))
eM = np.zeros((L,LI))

y = np.arange(11)                           # pixel coordinate across the drift

for k in np.arange(L):                     # loop over the L=20 template stare

    Py = star[k,:,:]                        # shape(Py) = (11,51)

    for i in np.arange(LI):                 # fit Py(x) for x = 10:41 incl = 32 pixels
                                            # flat part of DSI is x = 9:40 inclusive
                    
        P0 = [Py[:,i+10].max(),2.,3.5]
        print(P0)
        try: popt1, pcov1 = curve_fit(L3, y, Py[:,i+10],P0)          # catch curve_fit error
        except RuntimeError:
            print('continuing after RTE at k,i = ',k,i)
            
        L0[k,i] = popt1[0]                  # shape(L0) = (20,32)
        S0[k,i] = popt1[1]
        M0[k,i] = popt1[2]
        perr = np.sqrt(np.diag(pcov1))      # errors for L0,S0,M0
        eL[k,i]=perr[0]
        eS[k,i]=perr[1]
        eM[k,i]=perr[2]
print('L3 fits done for ',L,' stars')


T = np.sum(star,axis=1)[:,10:42]            # shape(T) = (L,32)
L0S0 = L0*S0                                # shape(L0S0) = (L,32)

ratioT_L0S0 = (T / L0S0).mean()
stdT_L0S0 = (T / L0S0).std()
SNR_T = np.mean(T,axis=1) / np.std(T,axis=1)
SNR_L0S0 = np.mean(L0S0,axis=1) / np.std(L0S0,axis=1)


#YET TO BE CLEANED UP

'''m_best = np.zeros(L, dtype='int64')         # index
r_best = np.zeros(L)
r = np.zeros((L,L))
T_corr = np.zeros((L,LI))
SNR_T_corr = np.zeros(L)

# ------------- normalise T to their median         
                
T_norm = (T.T / np.median(T, axis=1)).T               # T normalised to median(T)                                      

# ------------- calculate Pearsonr correlations all star pairs  (20,20) 

for k in np.arange(L): 
    for m in np.arange(L):
        r[k,m],p = scipy.stats.pearsonr(T_norm[k],T_norm[m])
             
# ------------- for each k, find index m_best for star with the second-largest r:  i.e  largest r with r != 1 
      
    m_best[k] = np.argsort(r[k,:])[L-2]                                        

# ------------- divide each star by the other star in the set of L stars with which it had the largest r

    T_corr[k] = T_norm[k,:] / T_norm[m_best[k],:]
    
# ----------- now calculate the SNR for the corrected spectra
#         the r-parameter for the best match to star k is r[k,m_best[k]]

    SNR_T_corr[k] = np.median(T_corr[k,:])/np.std(T_corr[k,:])
    r_best[k] =  r[k,m_best[k]]  '''
