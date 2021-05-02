#!/usr/bin/env python -W ignore::DeprecationWarning
# Purpose: Take 3000 galaxy clusters from the Buzzard v2.0 catalog
# This code is slow because it loads multiple times the same file but it prevents a memory issue

print('Importing')
import numpy as np
import healpy as hp

import os
from time import sleep
from time import time

import glob

import matplotlib 
import matplotlib.pyplot as plt
matplotlib.rcParams.update({'font.size': 16})

import pandas

from helper import *

cosmo = FlatLambdaCDM(H0=70, Om0=0.283)
Mpc2cm = 3.086e+24
Msol = 1.98847e33

######### Setting the Code #########

rmax = 8 # Mpc

####### Output Files
outdir = '/global/project/projectdirs/des/jesteves/buzzardSelection/v2.0.0/'
outfile_base = outdir+'healpix8/buzzard_v2.0.0_{}.hdf'
#######

####### Input Files
indir         ='/global/project/projectdirs/des/jderose/Chinchilla/Herd/Chinchilla-3/v1.9.9/addgalspostprocess/'
files         = glob.glob(indir+'truth/Chinchilla-3_lensed_rs_shift_rs_scat_cam*')
cinfile       = outdir+'buzzard_v2.0.0_hod_halos.fits'
base          = indir + 'truth/Chinchilla-3_lensed_rs_shift_rs_scat_cam.%i.fits'

######### Starting the Code #########
cat = Table(getdata(cinfile))

#### Make cutout
DA = AngularDistance(np.array(cat['Z']))
cat['rmax'] = 60*(float(rmax)/DA)*rad2deg ## arcmin

## for tests purpose
#cat = cat[:3]

#### healpix list
hpx_list = np.unique(cat['hpx8'])

# given a healpix
for hpx in hpx_list:
    gg = get_galaxy_sample(hpx,cat)

    ## find the healpix files around the cluster centers
    gg.get_healpix_neighbours()

    ## load files within Mr<=-18
    d = gg.load_files(base,amagMax=-18.)

    ## get only true members for the silver cluster sample
    ds= gg.get_silver_sample(d)

    ## get the galaxies within 8Mpc from the golden cluster centers
    dg= gg.get_golden_sample(d)

    ## drop the whole galaxy sample
    d = 0

    ## compute cluster richness for all the sample ; updates ngals on the cluster sample
    gg.compute_cluster_richness(ds)
    
    ## writing the output files
    outfile = outfile_base.format(hpx)
    save_hdf5_output(dg,gg.cat[gg.indices],outfile)
    
    print(gg.cat[gg.indices])

    hpx_time = (time()-gg.t0)/60
    print('partial time: %.2f min \n'%(hpx_time))
    
    ## drop the other samples
    ds = dg = 0