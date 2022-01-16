#!/usr/bin/env python -W ignore::DeprecationWarning

import numpy as np
from astropy.table import Table, vstack, join

import fitsio
import healpy as hp

from astropy.io.fits import getdata
from astropy.io import fits as pyfits

from astropy.cosmology import FlatLambdaCDM
from astropy import units as u
from scipy.interpolate import interp1d

import dask
from joblib import Parallel, delayed

import os
from time import sleep
from time import time

import glob
import healpy
import esutil
import matplotlib 
import matplotlib.pyplot as plt
import h5py
import pandas as pd

matplotlib.rcParams.update({'font.size': 22})

cosmo = FlatLambdaCDM(H0=70, Om0=0.283)
Mpc2cm = 3.086e+24
Msol = 1.98847e33
h = 0.7


## Dask bug on NERSC cori
# __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"

def split_text(infile):
    return int(infile.split('.')[-2])

def load_hdf_files(infile,indices,columns,path='catalog/gold/'):
    master = h5py.File(infile,'r')
    gold   = master[path]
    
    mydict = dict().fromkeys(columns)
    
    mydict['id'] = indices
    for col in columns:
        mydict[col] = gold[col][:][indices]
    
    master.close()
    return Table(mydict)
def save_hdf5_output(gal,cat,outfile):
    df  = gal.to_pandas()
    df.to_hdf(outfile, key='members', mode='w')

    gal = 0

    dfc = cat.to_pandas()
    dfc.to_hdf(outfile, key='cluster', mode='a')

def remove_outlier(x,ns=1.5):
    p16,p84 = np.percentile(x,[16,84])
    iqr = p84-p16
    nh = p84+ns*iqr
    nl = p16-ns*iqr
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

def _AngularDistance(z):
    DA = ( (cosmo.luminosity_distance(z)/(1+z)**2)/u.Mpc ) # em Mpc
    return DA
AngularDistance = np.vectorize(_AngularDistance)

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
