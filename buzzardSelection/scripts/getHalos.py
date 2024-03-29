#!/usr/bin/env python -W ignore::DeprecationWarning
# Purpose: Take 1000 galaxy clusters from the Buzzard v2.0 catalog
# Load files in Parallel only works for getdata()

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

from helper import *

######### Setting the Code #########

####### Input Variables
Nhalos   = 3000        ## choose the output sample size
nbins    = 25          ## choose how much bins you divide your whole sample
nprocess = 4

#######
rmax       = 4         ## maximum radius in Mpc
zmin, zmax = 0.1, 0.9  ## for GC
MassCut    = 1e12      ## the low Mass cut
Nside=1024
h=0.7
#######

####### Output Files
outdir = '/global/project/projectdirs/des/jesteves/buzzardSelection/v2.0.0/'

fileprefix      = 'buzzard_v2.0.0_{}.fits' 
file_cls_out_all= outdir+fileprefix.format('halos_hod')
file_cls_out    = outdir+fileprefix.format('%i_halos_cluster'%(Nhalos))
file_gal_out    = outdir+fileprefix.format('%i_halos_members'%(Nhalos))

hpx_file = outdir+'hpxmap_%i_buzzard_v2.0.0.fits'%Nside
#######


####### Input Files
indir         ='/global/project/projectdirs/des/jderose/Chinchilla/Herd/Chinchilla-3/v1.9.9/addgalspostprocess/'
files         = glob.glob(indir+'truth/Chinchilla-3_lensed_rs_shift_rs_scat_cam*')
files_truth   = glob.glob(indir+'halos/Chinchilla-3_halos*')

mag_file      = '/global/homes/j/jesteves/codes/buzzardAnalysis/buzzardSelection/scripts/files/annis_mags_04_Lcut.txt'

# print('\n'.join(files[:5]))

def split_text(infile):
    return int(infile.split('.')[-2])

## select a patch of the sky; 0deg < dec <30 deg and 0 deg< RA< 60 deg
rax= np.linspace(0,60.,200)
dex= np.linspace(0,30.,200)
ragrid, degrid = np.meshgrid(rax, dex)
pixels = np.unique(radec_pix(ragrid,degrid,nside=8))
healpix = np.array([split_text(infile) for infile in files])

w, = np.where(np.in1d(healpix,pixels))
files = [files[i] for i in w]
files.sort()

#######
#######
def masking_duplicates(hid,z):
    mask  = (hid < 1e8)&(z<0.34)
    mask |= (hid > 1e8)&(z>=0.34)
    mask |= (hid < 1e9)&(z<0.9)
    mask |= (hid > 1e9)&(z>0.9)
    return mask

# ####### only for tests purposes
#files = files[:25]

######## Starting the Code #########
print('loading dataset')
t0 = time()
colnames = ['ID', 'HALOID', 'RA', 'DEC', 'Z', 'OMAG', 'OMAGERR', 'AMAG', 'RHALO', 'CENTRAL','M200','R200','MAG_R']

### multiple files
print('Selecting Galaxy Clusters')
cluster_mask = 'CENTRAL > 0  && M200 >= %.2e &&  Z>= %.2f && Z<= %.2f'%(MassCut,zmin,zmax)

colnames = ['HALOID','RA','DEC','Z','M200','R200','MAG_R']
cin = loadfiles(files,columns=colnames,mask=cluster_mask)

### remove duplicates
mask = masking_duplicates(cin['HALOID'],cin['Z'])
c    = cin[mask]

ncls = len(c)
print('# clusters:', ncls)

## random selection of halos with mass less than 5 x E13 Msun
print('Selecting Silver Sample')
mask = c['M200']*h<5e13
cat_silver = get_random_selection(c[mask],Nsize=30500)

## uniform selection of halos with mass greater than 5 x E13 Msun
print('Selecting Golden Sample')
cat_golden = get_high_mass_selection(c,Nsize=Nhalos,nbins=20,h=h)

print('Mathching Sample')
## flag
cat_silver['sample'] = False
cat_golden['sample'] = True

print('Geting Critical Mass')
## catf is whole sample with M200 added
## catb is the halos data
print('Geting Critical Mass')
## catf is whole sample with M200 added
## catb is the halos data
catf,catb = get_critical_mass(catf,files_truth,MassCut=MassCut,zmin=zmin,zmax=zmax)

print('getting healpix map')
# if not os.path.isfile(hpx_file):
ra,dec = catf['RA'], catf['DEC']
hpx_map = make_hpx_map(ra,dec,Nside,hpx_file)
hpx_values = np.array(hpx_map['hpx_value'])

# print('Computing Area Fraction')
# catf['area_frac'] = compute_area_fraction(catf,hpx_values,rmax=8,nside=Nside)
print('Computing magLim')
catf = vstack([cat_golden,cat_silver])
keys, idx = np.unique(catf['HALOID'],return_index=True) ## unique keys
catf = catf[idx]

z = catf['Z']
magLim3 = getMagLimModel(mag_file,z,dm=2)
catf['magLim'] = magLim3[:,1]                ##i-band

print('Selectiong good halos')
x1=catf['M200_old']
x2=catf['M200']

fresidual = np.log10(x1/(x2+1e-3))
w = remove_outlier(fresidual,ns=1.5)

#catfo= catf
catf['bad']    = True
catf['bad'][w] = False
catf['hpx8_radec'] = radec_pix(catf['RA'],catf['DEC'],nside=8)

print('Golden cluster sample size: %i'%(np.count_nonzero(catf['sample'][w])))

## spliting sample
mask = catf['sample']
cat_g = catf[mask]
cat_s = catf[np.logical_not(mask)]

print('Saving cluster samples')
catf = vstack([cat_g,cat_s])

print('->',file_cls_out)
print('->',file_cls_out_all)

cat_g.write(file_cls_out,format='fits',overwrite=True)
catf.write(file_cls_out_all,format='fits',overwrite=True)

tf = (time()-t0)/60
print('Final time: %.1f'%(tf))

## testing plots
nbad = np.logical_not(catf['bad'])
plt.plot([5e12,3e15],[5e12,3e15],'k--')
plt.scatter(catf['M200_old'],catf['M200'],alpha=0.2)
plt.scatter(catf['M200_old'][nbad],catf['M200'][nbad],alpha=0.2,label='My Selection')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('M200m')
plt.ylabel('M200c')
plt.legend()
plt.savefig('getHalos_mass.png')

plt.clf()

## check healpix file
pixels0 = catf['hpx8']
pixels1 = catf['hpx8_radec']
plt.scatter(pixels0,pixels1)
plt.xlabel('healpix file')
plt.ylabel('healpix radec')
plt.savefig('check_healpix.png')
plt.clf()

## look at the map
npix = hp.nside2npix(8)
hpxmap = np.zeros(npix, dtype=np.float)
w, values = np.unique(pixels0,return_counts=True)
hpxmap[w] = values
hp.mollview(hpxmap, nest=True, title="NSIDE = 8 ")
plt.savefig('healpix_map_halos.png')
plt.clf()

### Distribution plot
nbad = np.logical_not(cat_g['bad'])
zcls = cat_g['Z'][nbad]
m200 = cat_g['M200'][nbad]*0.7

plot_scatter_hist(zcls,np.log10(m200),ylabel=r'$\log(M_{halo})$')
plt.savefig('halo_mass_redshift_distribution.png')


# # # plt.scatter(z,hid)