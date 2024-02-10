#!/usr/bin/env python -W ignore::DeprecationWarning
# Modification feb 2024
# nohup python scripts/get_fields_per_column.py &> log.out 2> log.err &

print('Importing')
import matplotlib.pyplot as plt
import healpy as hp
import numpy as np
import os
from astropy.table import Table, vstack, join
from astropy.io.fits import getdata
import pandas as pd
import esutil
import h5py
from time import time

from helper import radec_pix

######### Setting the Code #########
####### Output Files
outdir = '/global/cfs/cdirs/des/jesteves/data/cardinal/v2/tracts/'
outfile_base = outdir+'Cardinal-3_v2.0_Y6a_all_raw_hpx8_{:06d}.hdf'
array_path = outdir+'tmp/'
#######

####### Input Files
path   = '/global/cfs/cdirs/des/chto/Cardinal/Cardinalv2/'
fname  =  path+'Cardinal-3_v2.0_Y6a_gold.h5'
fname_aux = path+'Cardinal-3_v2.0_Y6a_mastercat.h5'
fname_bpz = path+'Cardinal-3_v2.0_Y6a_bpz.h5'

path_halo = '/global/cfs/cdirs/des/jesteves/data/buzzard/v1.9.8/y3_rm/'
profile_output_fname = path_halo+'buzzard-1.9.8_3y3a_run_halos_profiles.fit'
halo_run_fname = path_halo + 'buzzard-1.9.8_3y3a_run_halos_lambda_chisq_mu_star.fit'

columns = ['haloid','coadd_object_id','ra','dec','rhalo']
columns+= ['mag_%s'%(ix) for ix in ['g','r','i','z']]
columns+= ['mag_err_%s'%(ix) for ix in ['g','r','i','z']]

bpz_columns = ['z','zmean_sof', 'zmc_sof', 'redshift_cos']
all_columns = columns+bpz_columns

######### Initiating Functions #########
def save_hdf5_output(gal,outfile):
    df  = gal.to_pandas()
    df.to_hdf(outfile, key='members', mode='w')

    gal = 0

def load_tmp_file(tile):
    mydict = dict().fromkeys(columns)
    indices = np.load('{}{:06d}_{}_arr.npy'.format(array_path, tile, 'indices'))
    mydict['id'] = indices
    for col in all_columns:
        mydict[col] = np.load('{}{:06d}_{}_arr.npy'.format(array_path, tile, col))
    return Table(mydict)

def read_halos(profile_output_fname, halo_run_fname):
    ### read in profile file
    data = Table(getdata(profile_output_fname))
    data_h = Table(getdata(halo_run_fname))
    
    #### discard bad halos ####
    cosi_all = data['cosi']
    pid_all = data['pid']
    redshift = data['redshift']
    
    select_good = (pid_all==-1)&(cosi_all>=0)&(cosi_all<=1)&(redshift < 0.70)
    select_good = select_good & ((redshift < 0.33)|(redshift > 0.37))
    
    print('all vs good',len(cosi_all), len(cosi_all[select_good]))
    data_h.rename_column('HALOID','haloid')
    table = join(data_h,data[select_good],keys=['haloid'])
    return table


######### Starting Code #########
print('Loading Master File')
t0 = time()
master = h5py.File(fname,'r')
indexes= h5py.File(fname_aux,'r')

select     = indexes['index/gold/select'][:]
mag_i      = master['catalog/gold/mag_i'][:][select]
maglim_idx = select[mag_i<=24.]
ra         = master['catalog/gold/ra'][:][maglim_idx]
dec        = master['catalog/gold/dec'][:][maglim_idx]
hpx32      = radec_pix(ra,dec,nside=32)

ra = dec = 0
select=mag_i=0
master.close()
indexes.close()
time_to_load = time()-t0
print('Time spent to load the hdf5 file: %.2f min'%(time_to_load/60.))

print('Healpix Map')
cat = read_halos(profile_output_fname, halo_run_fname)
rac = cat['RA']
decc = cat['DEC']

cluster_tile = radec_pix(rac,decc,nside=32)
myfields = np.unique(cluster_tile)#[:3]
print('Number of fields: %i'%len(myfields))

print('\n')
print('Starting Query')

print('a) Querying the Indices')
indices_list = []

t0 = time()
time_evolution = np.empty((len(myfields)+len(all_columns)+1,),dtype=np.float64)
time_evolution[0] = t0

for i, tile in enumerate(myfields):
    print('start tile: %i'%(tile))
    w      = esutil.numpy_util.where1(cluster_tile==tile)
    print('Number Of Clusters: %i'%(w.size))
    
    circles = np.append(hp.get_all_neighbours(32, tile, nest=True),tile)
    print('Circles: done')
    
    match  = esutil.numpy_util.match(circles,hpx32)
    indices= maglim_idx[match[1]]
    
    np.save('{}{:06d}_indices_arr.npy'.format(array_path, tile), indices)
    print('{}{:06d}_indices_arr.npy'.format(array_path, tile))

    time_evolution[i+1] = time()
    print('partial time: %.2f min'%((time_evolution[i+1]-time_evolution[i])/60.))
    print('run time    : %.2f min'%((time_evolution[i+1]-time_evolution[0])/60.))
    print('\n')
indices = hpx32 = 0

print('b) Loading the Columns')
master = h5py.File(fname,'r')
gold   = master['catalog/gold/']
i +=1
for col in columns:
    print(col)
    array = gold[col][:]
    for tile in myfields:
        indices = np.load('{}{:06d}_indices_arr.npy'.format(array_path, tile))
        sub_array = array[indices]
        np.save('{}{:06d}_{}_arr.npy'.format(array_path, tile, col), sub_array)
        print('{}{:06d}_{}_arr.npy'.format(array_path, tile, col))
        sub_array = 0
    array = 0
    
    time_evolution[i+1] = time()
    print('partial time: %.2f sec'%((time_evolution[i+1]-time_evolution[i])))
    print('run time    : %.2f min'%((time_evolution[i+1]-time_evolution[0])/60.))
    print('\n')
    i+= 1
master.close()

print('Retrieve Photoz Columns')
master = h5py.File(fname_bpz,'r')
gold   = master['catalog/bpz']
#gold.visititems(show_h5_dataset)
for col in bpz_columns:
    print(col)
    array = gold[col][:]
    for tile in myfields:
        indices = np.load('{}{:06d}_indices_arr.npy'.format(array_path, tile))
        sub_array = array[indices]
        np.save('{}{:06d}_{}_arr.npy'.format(array_path, tile, col), sub_array)
        print('{}{:06d}_{}_arr.npy'.format(array_path, tile, col))
        sub_array = 0
    array = 0
    time_evolution[i+1] = time()
    print('partial time: %.2f sec'%((time_evolution[i+1]-time_evolution[i])))
    print('run time    : %.2f min'%((time_evolution[i+1]-time_evolution[0])/60.))
    print('\n')
    i+= 1
master.close()

print('c) Saving the tmp arrays')
for i, tile in enumerate(myfields):
    #print('join tile: %i'%(tile))
    outfile = outfile_base.format(tile)
    gal = load_tmp_file(tile)
    gal.rename_column('zmean_sof','z_mean')
    save_hdf5_output(gal,outfile)

total_time = time()-time_evolution[0]
print('Total time    : %.1f min, %.2f hours'%(total_time/60.,total_time/3600))
print('Finished !')
