#!/usr/bin/env python -W ignore::DeprecationWarning
# Purpose: Take 3000 galaxy clusters from the Buzzard v2.0 catalog
# This code is slow because it loads multiple times the same file but it prevents a memory issue

print('Importing')
import glob
import numpy as np
import healpy as hp
from time import time

## local library    
from helper import *

######### Setting the Code #########

rmax = 8 # Mpc

####### Output Files
outdir = '/global/project/projectdirs/des/jesteves/buzzardSelection/v2.0.0/'
outfile_base = outdir+'healpix8/buzzard_v2.0.0_{}.hdf'
#######

####### Input Files
indir         ='/global/project/projectdirs/des/jderose/Chinchilla/Herd/Chinchilla-3/v1.9.9/addgalspostprocess/'
files         = glob.glob(indir+'truth/Chinchilla-3_lensed_rs_shift_rs_scat_cam*')
cinfile       = outdir+'buzzard_v2.0.0_halos_hod.fits'
base          = indir + 'truth/Chinchilla-3_lensed_rs_shift_rs_scat_cam.%i.fits'

######### Starting the Code #########
cat = Table(getdata(cinfile))

#### Make cutout
sample = cat['sample']
nsample= np.logical_not(sample)

DA     = AngularDistance(np.array(cat['Z']))

cat['rmax'] = 0.
cat['rmax'][sample]  = 60*(float(rmax)/DA[sample])*rad2deg ## arcmin
cat['rmax'][nsample] = 60*(3./DA[nsample])*rad2deg ## arcmin

## for tests purpose
#cat = cat[:3]

#### healpix list
hpx_list = np.unique(cat['hpx8'])

## columns
# List all catalogs whose names start with the word "buzzard"
GCRCatalogs.get_available_catalog_names(include_default_only=False, name_startswith="buzzard")

catalog =  GCRCatalogs.load_catalog('buzzard_v2.0.0_3')
# catalog =  GCRCatalogs.load_catalog('buzzard_v2.0.0_test', config_overwrite={'healpix_pixels': [8786, 8787, 8788]})
columns = ['halo_id','galaxy_id','is_central','halo_mass','healpix_pixel','ra','dec','redshift','Mag_true_r_des_z01','Mag_true_i_des_z01','truth/RHALO']
filters = ['g','r','i','z']
columns+= ['mag_%s_lsst'%mag    for mag in filters]
columns+= ['magerr_%s_des'%mag for mag in filters]

## mag_%s_des and magerr_%s_des are empty columns.

# print('columns')
# print('\n'.join(sorted(columns)))
# print('\n')

# given a healpix
for hpx in hpx_list:
    gg = get_galaxy_sample(hpx,cat)

    ## find the healpix files around the cluster centers
    gg.get_healpix_neighbours()

    ## load files within Mr<=-18
    d = gg.load_files_gcr(catalog,columns,amagMax=-18.)

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