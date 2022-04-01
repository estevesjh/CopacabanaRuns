#!/usr/bin/env python -W ignore::DeprecationWarning
# This code is slow because it loads multiple times the same file but it prevents a memory issue
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
outdir = '/data/des61.a/data/johnny/Buzzard/Buzzard-3_v1.9.8_Y3a/catalog/y3/'
outfile_base = outdir+'buzzard_y3_v1.9.8_all_raw_hpx8_{:06d}.hdf'
array_path = outdir+'tmp/'
#######

####### Input Files
path   = '/data/des81.b/data/mariaeli/y3_buzz/Buzzard-3_v1.9.8_Y3a_mastercat/'
fname  =  path+'Buzzard_v1.9.8_Y3a_gold.h5'
fname_aux = path+'Buzzard-3_v1.9.8_Y3a_mastercat.h5'
fname_bpz = path+'Buzzard_v1.9.8_Y3a_bpz.h5'

path_halo = '/data/des61.a/data/johnny/Buzzard/Buzzard_v2.0.0/y3_rm/halos/'
profile_output_fname = path_halo+'buzzard-1.9.8_3y3a_run_halos_profiles.fit'
halo_run_fname = path_halo + 'buzzard-1.9.8_3y3a_run_halos_lambda_chisq_mu_star.fit'

columns = ['haloid','coadd_object_id','ra','dec','rhalo']
columns+= ['mag_%s'%(ix) for ix in ['g','r','i','z']]
columns+= ['mag_err_%s'%(ix) for ix in ['g','r','i','z']]

pz_columns = ['z_mean','z_sigma']
bpz_columns = ['z','redshift_cos']
pz_columns_out = [i+'_dnf' for i in pz_columns]
bpz_columns_out = [i+'_bpz' for i in bpz_columns]
all_columns = columns+bpz_columns_out+pz_columns_out
# columns = ['ra','dec','rhalo','mag_g','mag_err_g']

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
    
    select_good = (pid_all==-1)&(cosi_all>=0)&(cosi_all<=1)##&(self.lam_all > 3)
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
maglim_idx = select[mag_i<=23.5]
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
t0 = time()
time_evolution = np.empty((len(myfields)+len(columns)+1,),dtype=np.float64)
time_evolution[0] = t0

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
master = h5py.File(fname_aux,'r')
gold   = master['catalog/dnf/unsheared/']
for col in pz_columns:
    print(col)
    array = gold[col][:]
    for tile in myfields:
        indices = np.load('{}{:06d}_indices_arr.npy'.format(array_path, tile))
        sub_array = array[indices]
        np.save('{}{:06d}_{}_arr.npy'.format(array_path, tile, col+'_dnf'), sub_array)
        print('{}{:06d}_{}_arr.npy'.format(array_path, tile, col+'_dnf'))
        sub_array = 0
    array = 0
    time_evolution[i+1] = time()
    print('partial time: %.2f sec'%((time_evolution[i+1]-time_evolution[i])))
    print('run time    : %.2f min'%((time_evolution[i+1]-time_evolution[0])/60.))
    print('\n')
    i+= 1

master.close()
print('Retrieve Photoz-BPZ Columns')
master = h5py.File(fname_bpz,'r')
gold   = master['catalog/bpz']
#gold.visititems(show_h5_dataset)
for col in bpz_columns:
    print(col)
    array = gold[col][:]
    for tile in myfields:
        indices = np.load('{}{:06d}_indices_arr.npy'.format(array_path, tile))
        sub_array = array[indices]
        np.save('{}{:06d}_{}_arr.npy'.format(array_path, tile, col+'_bpz'), sub_array)
        print('{}{:06d}_{}_arr.npy'.format(array_path, tile, col+'_bpz'))
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
    gal.rename_column('z_bpz','z')
    save_hdf5_output(gal,outfile)

total_time = time()-time_evolution[0]
print('Total time    : %.1f min, %.2f hours'%(total_time/60.,total_time/3600))
print('Finished !')
