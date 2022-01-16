#!/usr/bin/env python -W ignore::DeprecationWarning

import os
import numpy as np
import h5py
import esutil

from time import time

from astropy.table import Table, vstack
from astropy.io.fits import getdata

## local library    
from helper import get_healpix_list, AngularDistance,\
                   load_hdf_files, save_hdf5_output

######### Setting the Code #########
print(5*'--'+'Starting Code'+5*'--')
print()
rmax = 6. # Mpc

####### Output Files
outdir = '/data/des61.a/data/johnny/DESY3/data/cutouts/'
outfile_base = outdir+'y3_gold_2.2.1_wide_sofcol_run_redmapper_v6.4.22_wv1.2_full_hpx8_{:05d}.hdf5'
cluster_file = outdir+'y3_gold_2.2.1_wide_sofcol_run_redmapper_v6.4.22_wv1.2_full.fits'
#######

####### Input Files
path  = '/data/des81.b/data/mariaeli/y3_cats/full/'
fname = path+'Y3_GOLD_2_2.1_12_3_19.h5' 

# path2  = '/data/des61.a/data/johnny/DESY3/data/photoz/dnf_gold_2_2/'
# fname_aux= path2+'y3_gold_2_2.1_dnf_jesteves.h5'

####### Columns
columns = ['hpix_16384','coadd_object_id','ra','dec','extended_class_mash_sof','flags_gold']
columns+= ['sof_cm_mag_corrected_%s'%(ix) for ix in ['g','r','i','z']]
columns+= ['sof_cm_mag_err_%s'%(ix) for ix in ['g','r','i','z']]

print('Load Gold Catalog\n')
####### Load Variables
master = h5py.File(fname,'r')
mag_i      = master['catalog/gold/sof_cm_mag_corrected_i'][:][:]
maglim_idx = np.where((mag_i<=23.5)&(mag_i>=0.))[0]
hpx16384   = master['catalog/gold/hpix_16384'][:][maglim_idx]#.astype(np.int64)
master.close()

# indexes = h5py.File(fname_aux,'r')
# dnf       = indexes['catalog']
# d_cid     = dnf['coadd_object_id'][:][:]
# d_zmean   = dnf[u'DNF_ZMEAN_SOF'][:][:]
# d_zt      = dnf[u'Z'][:][:]
# d_sigma   = dnf[u'DNF_ZSIGMA_SOF'][:][:]
# d_indices = dnf[u'indices'][:][:]
# indexes.close()

print('Load Cluster Catalog\n')
####### Load Cluster catalog
cat = Table(getdata(cluster_file))

cluster_tile = np.array(cat['tile']).astype(int)
tiles = np.unique(cluster_tile)

## compute the radii max
rad2deg  = 180/np.pi
rmax     = 8 #Mpc around each cluster
DA = AngularDistance(np.array(cat['Z_LAMBDA']))
cat['rmax']  = 60*(float(rmax)/DA)*rad2deg ## arcmin

print('Starting Query Procedure\n')
t0 = time()
time_evolution = np.empty((len(tiles)+1,),dtype=np.float64)
time_evolution[0] = t0
for i, tile in enumerate(tiles[3:]):
    print(10*'--')
    print('start tile: %i'%(tile))
    outfile = outfile_base.format(tile)
    w       = np.where(cluster_tile==tile)[0]

    if (len(w)>1) & (not os.path.isfile(outfile)):
        circles= get_healpix_list(cat[w],nside=16384)
        match  = esutil.numpy_util.match(circles,hpx16384)
        indices= maglim_idx[match[1]]
        #match2 = esutil.numpy_util.match(indices,d_indices)
        print('matching: done')

        data  =  load_hdf_files(fname,indices,columns,path='catalog/gold/')
        data['index']= indices
        data['tile'] = tile
        print('loading main data: done')

        save_hdf5_output(data,cat[w],outfile)
        data = 0
        print('outfile saved: %s'%(outfile))
    else:
        print('Error: empty tile')
    
    time_evolution[i+1] = time()
    print('partial time: %.2f min'%((time_evolution[i+1]-time_evolution[i])/60.))
    print('run time    : %.2f min'%((time_evolution[i+1]-time_evolution[0])/60.))
    print('\n')

total_time = (time_evolution[i+1]-time_evolution[0])/60.

print('It is done!')
print('Total time: %.2f hours \n'%total_time)
