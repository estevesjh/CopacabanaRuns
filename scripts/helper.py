#!/usr/bin/env python -W ignore::DeprecationWarning
# Purpose: Take 1000 galaxy clusters from the Buzzard v2.0 catalog
# Load files in Parallel only works for getdata()

import numpy as np
from astropy.table import Table, vstack, join

import fitsio
import healpy as hp

from astropy.io.fits import getdata
from astropy.io import fits as pyfits

from astropy.cosmology import FlatLambdaCDM
from astropy import units as u
import esutil

import dask
from joblib import Parallel, delayed

import os
from time import sleep
from time import time

import glob



import matplotlib 
import matplotlib.pyplot as plt
matplotlib.rcParams.update({'font.size': 22})

#from sklearn.model_selection import StratifiedShuffleSplit
try: from sklearn.model_selection import train_test_split
except: from sklearn.cross_validation import train_test_split


cosmo = FlatLambdaCDM(H0=70, Om0=0.283)
Mpc2cm = 3.086e+24
Msol = 1.98847e33
h = 0.7


## Dask bug on NERSC cori
# __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"

def split_text(infile):
    return int(infile.split('.')[-2])

def loadfiles(filenames, columns=None, mask=None):
    '''
    Read a set of filenames with fitsio.read and return a concatenated array
    '''
    out = []
    i = 1
    print()
    for f in filenames:
        print('File {i}: {f}'.format(i=i, f=f))
        if not os.path.isfile(f):
            print('file not found: %s'%f)
        else:
            w = get_mask(f,mask) ## if mask None returns all the rows
            ## loading subset
            g0= Table(fitsio.read(f,rows=w,columns=columns))
            g0['hpx8'] = split_text(f)
            out.append(g0)
        i += 1
    mygals  = [data for data in out if data is not None]
    data       = vstack(mygals)
    return data

def get_mask(f,mask):
    fits = fitsio.FITS(f)
    if mask is not None: w = fits[1].where(mask)
    else: w = np.arange(fits[1].get_nrows(),dtype=np.int64)
    fits.close()
    return w

def remove_outlier(x):
    p16,p84 = np.percentile(x,[16,84])
    iqr = p84-p16
    nh = p84+2.5*iqr
    nl = p16-2.5*iqr
    w, = np.where((x>=nl)&(x<=nh))
    return w

def chunks(ids1, ids2):
    """Yield successive n-sized chunks from data"""
    for id in ids2:
        w, = np.where( ids1==id )
        yield w

def rhoc(z):
    try:
        rho_c = float(cosmo.critical_density(z)/(u.g/u.cm**3)) # in g/cm**3
    except:
        rho_c = [float(cosmo.critical_density(zi)/(u.g/u.cm**3)) for zi in z]
        rho_c = np.array(rho_c)
    
    rho_c = rho_c*(Mpc2cm**3)/Msol # in Msol/Mpc**3
    return rho_c

def convertM200toR200(M200,z):
    ## M200 in solar masses
    ## R200 in Mpc
    rho = rhoc(z)
    R200 = np.power( M200/(200*4*np.pi*rho/3) , 1/3.  )
    return R200

def AngularDistance(z):
    DA = ( (cosmo.luminosity_distance(z)/(1+z)**2)/u.Mpc ) # em Mpc
    return DA
AngularDistance = np.vectorize(AngularDistance)

rad2deg = 180/np.pi
## functions
def get_healpix_radec(center_ra,center_dec,radius=1.,nside=32):
    center_ra_rad = np.radians(center_ra)
    center_dec_rad = np.radians(center_dec)

    center_vec = np.array([np.cos(center_dec_rad)*np.cos(center_ra_rad),
                        np.cos(center_dec_rad)*np.sin(center_ra_rad),
                        np.sin(center_dec_rad)])

    healpix_list = hp.query_disc(nside, center_vec, np.radians(radius), nest=True, inclusive=True)
    healpix_list = np.append(healpix_list,radec_pix(center_ra,center_dec,nside=nside))
    
    return healpix_list

def get_healpix_map(ra,dec,nside=1024):
    # Set the number of sources and the coordinates for the input
    nsources = len(ra)
    npix = hp.nside2npix(nside)

    thetas,phis = np.radians(90-dec),np.radians(ra)

    # Go from HEALPix coordinates to indices
    indices = hp.ang2pix(nside, thetas, phis,nest=True)

    hpxmap = np.zeros(npix, dtype=np.float)
    w, values = np.unique(indices,return_counts=True)
    hpxmap[w] = values

    return hpxmap

def compute_area_fraction(cat,hpxmap,rmax=1,nside=1024):
    ncls=len(cat)
    out = []
    for i in range(ncls):
        ra,dec = cat['RA'][i], cat['DEC'][i]
        zcls = cat['Z'][i]

        DA = float(AngularDistance(zcls))
        radius = (float(rmax)/DA)*rad2deg ## degrees

        hps_rad = get_healpix_radec(ra,dec,radius=radius,nside=nside)
        hpx_circle  = np.unique(hps_rad)
        
        ## matched healpix table
        ncommon = np.count_nonzero(hpxmap[hpx_circle]) #np.unique(hpx_gals[(np.in1d(hpx_gals,hpx_circle))])

        ## area fraction: number of matched healpix / number of healpix in the cluster circle
        area_fraction = 1.*ncommon/len(hpx_circle)
        out.append(area_fraction)
    
    return np.array(out)

def get_healpix_list(cat,nside=1024):
    healpix_list = np.empty((0),dtype=int)

    for ra,dec,rmax in zip(cat["RA"],cat["DEC"],cat['rmax']):
        hps_rad = get_healpix_radec(ra,dec,radius=rmax/60,nside=nside)
        healpix_list = np.append(healpix_list,hps_rad)
    
    # cat["healpix_list"] = healpix_list

    return np.unique(healpix_list) ## get unique healpix number

def hpix2ang(pix,nside=1024):
    lon,lat = hp.pix2ang(nside,pix,nest=True)
    dec,ra=(90-(lon)*(180/np.pi)),(lat*(180/np.pi))
    return ra,dec

def radec_pix(ra,dec,nside=1024):
    return np.array(hp.ang2pix(nside,np.radians(90-dec),np.radians(ra),nest=True),dtype=np.int64)

def make_hpx_map(ra,dec,Nside,hpx_file,save=True):
    hpx_map = get_healpix_map(ra,dec,nside=Nside)
    hpx_indices = np.arange(1,len(hpx_map)+1,1,dtype=np.int)

    
    data = [hpx_indices,hpx_map]
    cols = ['hpx_pixel','hpx_value']
    if save:
        data = Table(data,names=cols)
        data.write(hpx_file,format='fits',overwrite=True)
        print('healpix map, nside %i'%Nside)
        print('-> %s'%hpx_file)
    return data

def get_critical_mass(cat2,files_truth,MassCut=1e12,zmin=0.,zmax=1.):
    ### Truth table
    print('Load the halos files')
    cat= loadfiles(files_truth,columns=['HALOID','Z','M200c','M200b','RVIR','Z_COS'])
    #cat.rename_column('HOST_HALOID','HALOID')
    cat.rename_column('M200C','M200')

    print('\n Making cuts')
    print('clusters with M200<%.2E and z<%.2f'%(MassCut,zmax))
    cat = cat[(cat['Z']>=zmin)&(cat['Z']<=zmax)]
    cat = cat[cat['M200B']>=MassCut]
    cat.remove_columns('hpx8')

    print('matching halos truth files')
    cat2.rename_column('M200','M200_old')
    new_cat = join(cat2['HALOID','RA','DEC','hpx8','M200_old','sample'],cat,keys=['HALOID'])
    
    print('switching M200b and R200b to M200c and R200c')
    Z = new_cat['Z']
    M200c = new_cat['M200'][:]     ## [Msun/h] in physical units
    M200b = new_cat['M200B'][:]     ## [Msun/h] in physical units

    r200c = convertM200toR200(M200c*h,Z)/h  ## [Mpc/h] in physical units
    r200b = convertM200toR200(M200b*h,Z)/h  ## [Mpc/h] in physical units

    new_cat['R200B'] = r200b      ## [Mpc/h] mean, in comoving units 
    new_cat['R200'] = r200c
    
    return new_cat, cat

def get_random_selection(cat2,Nsize=3000):
    df = cat2.to_pandas()
    train_set, test_set = train_test_split(df, test_size=Nsize, random_state=42)

    cat3 = Table.from_pandas(test_set)
    return cat3

def get_high_mass_selection(cat2,Nsize=1000,nbins=25):
    Nsize+= 1200

    M200 = cat2["M200"]
    Mh = M200[cat2["M200"]>=5e13]
    z  = cat2['Z']
    df = cat2.to_pandas()

    data = []
    mbins = np.logspace(np.log10(np.min(Mh)),np.log10(np.max(Mh)),nbins)
    zbins = np.linspace(0.1,1.,11)
    for zl,zh in zip(zbins[:-1],zbins[1:]):
        for ml,mh in zip(mbins[:-1],mbins[1:]):
            w, = np.where((M200>=ml)&(M200<=mh)&(z<=zh)&(z>=zl))
            dfb= df.iloc[w]
            size=int(Nsize/nbins/10)
            if w.size <size: size=len(dfb)-1
            if len(dfb)>0:
                print('bin size:',size)
                if (size>0)&(size<=len(dfb)):
                    try:
                        train_set, test_set = train_test_split(dfb, test_size=size, random_state=42)
                        data.append(Table.from_pandas(test_set))
                    except:
                        print('Error while taking the mass bin %.1e<M200<%.1e'%(ml,mh))
                else:
                    print('Error: no halos within this limits')
    cat_high_mass = vstack(data)
    return cat_high_mass

import seaborn as sns
sns.set_style('darkgrid')

def plot_scatter_hist(x,y,xlabel=r'$z$',ylabel=r'$\log(M_{200,c})\,\,[M_{\odot}\; h^{-1}]$',save='./img/bla.png'):
    fig = plt.figure(figsize=(10,8))

    # definitions for the axes
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    spacing = 0.01
    scale = 70
    
    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom + height + spacing, width, 0.2]
    rect_histy = [left + width + spacing, bottom, 0.175, height]

    scatter_axes = plt.axes(rect_scatter)
    scatter_axes.tick_params(direction='in', top=True, right=True)
    x_hist_axes = plt.axes(rect_histx,sharex=scatter_axes)
    x_hist_axes.tick_params(direction='in')
    y_hist_axes = plt.axes(rect_histy,sharey=scatter_axes)
    y_hist_axes.tick_params(direction='in')
    
    xmax = np.nanmax(x)
    ymin, ymax = np.nanmin(y), np.nanmax(y)
    
    binx = np.linspace(0.1, xmax, num=15)
    biny = np.linspace(ymin, ymax, num=15)  # 1500/100.

    pal = sns.dark_palette("#3498db", as_cmap=True)
    # sns.kdeplot(x, y, ax=scatter_axes, cmap=pal, zorder=3)  # n_levels=10

    scatter_axes.set_xlabel(xlabel, fontsize=25)
    scatter_axes.set_ylabel(ylabel, fontsize=25)

    scatter_axes.scatter(x, y, s=scale, color='b', marker='o', alpha=0.3, zorder=0)
#     scatter_axes.axhline(np.mean(y),linestyle='--',color='b')

    x_hist_axes.hist(x, bins=binx, density=False, alpha=0.3, color='b')
    y_hist_axes.hist(y, bins=biny, density=False, orientation='horizontal', alpha=0.3, color='b')
#     y_hist_axes.axhline(np.mean(y),linestyle='--', color='b')
    
    x_hist_axes.set_yticklabels([])
    y_hist_axes.set_xticklabels([])

    if ymax<0:
        ymin,ymax=13.5,15.3

        scatter_axes.set_ylim(ymin, ymax)
        y_hist_axes.set_ylim(ymin, ymax)

    scatter_axes.xaxis.set_tick_params(labelsize=15)
    scatter_axes.yaxis.set_tick_params(labelsize=15)
    x_hist_axes.xaxis.set_tick_params(labelsize=0.05, labelcolor='white')
    y_hist_axes.yaxis.set_tick_params(labelsize=0.05, labelcolor='white')

    fig.subplots_adjust(hspace=.01, wspace=0.01)