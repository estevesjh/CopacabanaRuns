################## README ##################
# The y3kp h5 file does not have DNF_ZSIGMA_SOF
# I created a full y3kp DNF file by retrieving the DNF sample from the ilinois des server
# Here I match the DNF files with the cutouts

import numpy as np
import os
import h5py
from astropy.table import Table, vstack
from astropy.io.fits import getdata
import matplotlib.pyplot as plt

import smatch
import esutil
from time import time

from helper import save_hdf5_output, upload_dataFrame, initiate_columns

def print_fractions(m,x1,x2):
    print('Fraction 1: %.3f'%(1.*m[0].size/len(x1)))
    print('Fraction 2: %.3f'%(1.*m[1].size/len(x2)))

path0 = '/data/des61.a/data/johnny/DESY3/desy3/data/photoz/dnf_gold_2_2/'
infile = path0+ 'dnf_mag23_5_{:06d}.fits'

path  = '/data/des81.b/data/mariaeli/y3_cats/full/'
fname = path+'Y3_GOLD_2_2.1_12_3_19.h5'
fname_aux = path+'Y3_GOLD_2_2.1_DNF_12_3_19.h5'

# path   = '/data/des81.b/data/mariaeli/y3_cats/subsampled/'
# fname  =     path+'Y3_GOLD_2_2.1_subsampled.h5'
# fname_aux  = path+'Y3_GOLD_2_2.1_DNF_subsampled.h5'

path1 = '/data/des61.a/data/johnny/emulatorPhotoZ/training_sample/'
infile1 = path1+'spec_y3_gold_2_2_27JUN19_photoz.fits'

# outfile = path0+'Y3_GOLD_2_2.1_DNF_subsampled_jesteves.h5'
outfile = path0+'Y3_GOLD_2_2.1_DNF_12_3_19_full_jesteves.h5'


cutouts = '/data/des61.a/data/johnny/DESY3/desy3/data/spt/cutouts/'
cluster_file = '/data/des61.a/data/johnny/DESY3/desy3/data/spt/join_2500d_sptecs.fits'
outfile_base = cutouts+'y3_gold_2.2.1_wide_sofcol_run_join_2500d_sptecs_hpx2_{:05d}.hdf5'

## loading dataset
# spec redshift sample
# tab = Table(getdata(infile1))

# # checking the h5 file
# master = h5py.File(fname,'r')
# mag_i      = master['catalog/gold/sof_cm_mag_corrected_i'][:][:]
# maglim_idx = np.where((mag_i<=23.5)&(mag_i>=0.))[0]

# # hpx16384   = master['catalog/gold/hpix_16384'][:][maglim_idx]
# # zndf        = master['catalog/gold/dnf_zmean_sof'][:][maglim_idx]
# cid        = master['catalog/gold/coadd_object_id'][:][maglim_idx]
# ra         = master['catalog/gold/ra'][:][maglim_idx]
# dec        = master['catalog/gold/dec'][:][maglim_idx]
# mag_i = 0.
# master.close()

# ## indexes
# indexes= h5py.File(fname_aux,'r')
# dnf    = indexes['catalog/unsheared']
# d_cid  = dnf['coadd_object_id'][:][maglim_idx]
# d_zmean= dnf['zmean_sof'][:][maglim_idx]
# d_sigma = 0.03*(1+d_zmean)
# indexes.close()

# ############################
# def match_sky(ra1,dec1,ra2,dec2,fname=None,nside=4096):
#     if os.path.isfile(fname):
#         m1 = np.loadtxt(fname).astype(int)
#     else:
#         t0 = time()
#         maxmatch=1 # return closest match
#         # ra,dec,radius in degrees
#         m = smatch.match(ra1, dec1, 1./3600, ra2, dec2,
#                         nside=nside, maxmatch=maxmatch)

#         m1 = [m['i1'],m['i2']]
#         if (fname is not None):
#             print('Saving Match')
#             np.savetxt(fname,np.vstack(m1))
#         print('matching time: %.2f s'%(time()-t0))
#     print_fractions(m1,ra1,ra2)
#     return m1

# def match_indices(id1,id2):
#     #id1, _ = np.unique(id1,return_index=True)
#     m2 = esutil.numpy_util.match(id1,id2)
#     m0 = [m2[0],m2[1]]
#     print_fractions(m0,id1,id2)
#     return m0
# #############################

# print('Load Data')
# print('Files:')
# table_list = []
# for i in range(17):
#     infilei = infile.format(i+1)
#     table_list.append(Table(getdata(infilei)))
#     print(infilei)

# data = vstack(table_list)
# print('\nSample Size: %.3f M'%(len(data) * 1e-6))

# print('Matching Spec Sample w/ DES Y3 Gold')
# ra2,dec2 = np.array(tab['RA']),np.array(tab['DEC'])
# match1 = match_sky(ra2,dec2,ra,dec,fname='match_spec_desy3_kp.npy',nside=int(2*4096))

# ztrue = -99.*np.ones_like(ra)
# ztrue[match1[1]] = tab['Z'][[match1[0]]]
# zcheck1 = -99.*np.ones_like(ra)
# zcheck1[match1[1]] = tab['DNF_ZMEAN_MOF'][[match1[0]]]

# print('Matching DES Y3 Gold w/ h5 files')
# match2 = match_indices(np.array(data['COADD_OBJECT_ID']),cid)
# # match2 = match_sky(ra1,dec1,ra,dec,fname='subsampled_desy3.npy',nside=50000)

# print('Save output')
# ## Save File
# z    = d_zmean
# zmc  = -99.*np.ones_like(ra)
# zerr = d_sigma
# #ztrue= -99.*np.ones_like(ra)
# zcheck2 = d_zmean

# z[match2[1]]     = data['DNF_ZMEAN_SOF'][match2[0]]
# zerr[match2[1]]  = data['DNF_ZSIGMA_SOF'][match2[0]]
# zmc[match2[1]]   = data['DNF_ZMC_SOF'][match2[0]]

# cols = ['indices','coadd_object_id','ra','dec','Z','DNF_ZMEAN_SOF','DNF_ZSIGMA_SOF','DNF_ZMC_SOF','DNF_ZMEAN_MOF','DNF_ZMEAN_SOF_OLD']
# data_list = [maglim_idx,cid,ra,dec,ztrue,z,zerr,zmc,zcheck1,zcheck2]

# print('outfile: %s'%outfile)
# master = h5py.File(outfile,'w')
# master.create_group('catalog')

# for col,di in zip(cols,data_list):
#     master.create_dataset('catalog/%s'%col, data=di)
# print(5*'---'+'\n')

print('Matching w/ cutouts data')
cat  = Table(getdata(cluster_file))
tiles = np.array(np.unique(cat['tile']))
#tiles = tiles[tiles>280]

indexes= h5py.File(outfile,'r')
dnf    = indexes['catalog']
d_cid  = dnf['coadd_object_id'][:][:]
d_z    = dnf[u'DNF_ZMEAN_SOF'][:][:]
d_zmc  = dnf[u'DNF_ZMC_SOF'][:][:]
d_zt   = dnf[u'Z'][:][:]
d_zerr = dnf[u'DNF_ZSIGMA_SOF'][:][:]
d_indices = dnf[u'indices'][:][:]
indexes.close()


for hi in tiles:
    outfile = outfile_base.format(hi)
    print(outfile)
    if os.path.isfile(outfile):
        gi = upload_dataFrame(outfile,keys='members')
        ci = upload_dataFrame(outfile,keys='cluster')

        #cid = np.array(gi['coadd_object_id'])
        idx = np.array(gi['index'])
        match = esutil.numpy_util.match(idx,d_indices)
        
        gi = initiate_columns(gi,columns=['z', 'z_mc_dnf','z_mean_dnf','z_sigma_dnf'])
        gi['z'][match[0]] = d_zt[match[1]]
        gi['z_mean_dnf'][match[0]] = d_z[match[1]]
        gi['z_sigma_dnf'][match[0]] = d_zerr[match[1]]
        gi['z_mc_dnf'][match[0]] = d_zmc[match[1]]

        fdnf = float(1.*match[0].size/len(gi))
        ci['fraction_dnf'] = fdnf
        print(fdnf)
        save_hdf5_output(gi,ci,outfile)