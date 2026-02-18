import numpy as np
import healpy as hp

########### Histogram 2d map ##################

def cat_to_hpx(theta, phi, nside, radec=True):
    
    indices = hp.ang2pix(nside, phi, theta)
    npix = hp.nside2npix(nside)

    indx, counts = np.unique(indices, return_counts=True)
    hpx_map = np.zeros(npix, dtype=int)
    hpx_map[indx] = counts

    return hpx_map

########### One way of Colourmap of Healpy map ##################

def map_weight(theta, phi, third,nside, radec=True):
    
    ipix = hp.ang2pix(nside, phi, theta)

    mean_age = (np.bincount(ipix, minlength=hp.nside2npix(nside), weights=third) / 
                np.bincount(ipix, minlength=hp.nside2npix(nside)))

    return mean_age
    
########### Second way of Colourmap of Healpy map ##################

def cat_to_hpx_third3(thetas, phis, fs ,nside, radec=True):
    
    indices = hp.ang2pix(nside, phis, thetas)
    npix = hp.nside2npix(nside)
    hpxmap = np.zeros(npix, dtype=float)
    indx,counts = np.unique(indices, return_counts=True)
    
    for ind in indx:
        hpxmap[ind] =np.mean(fs[indices==ind])        
    return hpxmap

