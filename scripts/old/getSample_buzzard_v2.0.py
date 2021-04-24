#!/usr/bin/env python -W ignore::DeprecationWarning
# Purpose: Take 1000 galaxy clusters from the Buzzard v2.0 catalog
# Load files in Parallel only works for getdata()

print('Importing')
import numpy as np
from astropy.table import Table, vstack
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

import glob

from time import time

import matplotlib 
import matplotlib.pyplot as plt
matplotlib.rcParams.update({'font.size': 22})

#from sklearn.model_selection import StratifiedShuffleSplit
try: from sklearn.model_selection import train_test_split
except: from sklearn.cross_validation import train_test_split


cosmo = FlatLambdaCDM(H0=70, Om0=0.283)
Mpc2cm = 3.086e+24
Msol = 1.98847e33

## Dask bug on NERSC cori
# __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"

def readFile(infile,zmin=0.1,zmax=1.,amagMax=25,colnames=None):
    if os.path.isfile(infile):
        galaxy_cat = getdata(infile)
        g0 = Table(galaxy_cat)
        g0 = g0[(g0['Z']>=zmin)&(g0['Z']<=zmax)&(g0['MAG_R']-5*np.log10(h)<=amagMax)]
        if colnames is not None:
            g0 = g0[colnames]
        return g0
    else:
        print('file not found: %s'%(infile))
        pass

def readFilesParallel(files,zmin=0.1,zmax=1.,nCores=2,colnames=None,amagMax=-18.0):
    nfiles = len(files)
    
    if nfiles>15: ## avoid buffer issue
        allData,results = [],[]
        for i in np.arange(0,nfiles,15,dtype=int):
            idxf = i+15
            results0 = [dask.delayed(readFile)(fi,zmin=zmin,zmax=zmax,colnames=colnames,magMax=amagMax) for fi in files[i:(idxf)]]
            results0 = dask.compute(*results0, scheduler='processes', num_workers=nCores)
            if (i+15)>nfiles:
                idxf=nfiles
            results.append(results0)
        allData = [data for ri in results for data in ri if data is not None]
        
    
    else:
        results = [dask.delayed(readFile)(fi,zmin=zmin,zmax=zmax,colnames=colnames) for fi in files]
        results = dask.compute(*results, scheduler='processes', num_workers=nCores)
        allData = [data for data in results if data is not None]

    g = vstack(allData)
    return g

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

def make_hpx_map(ra,dec,Nside,hpx_file):
    hpx_map = get_healpix_map(ra,dec,nside=Nside)
    hpx_indices = np.arange(1,len(hpx_map)+1,1,dtype=np.int)


    data = [hpx_indices,hpx_map]
    cols = ['hpx_pixel','hpx_value']

    data = Table(data,names=cols)
    data.write(hpx_file,format='fits',overwrite=True)
    print('healpix map, nside %i'%Nside)
    print('-> %s'%hpx_file)

    return data

def get_amag_cut(f,amagMax):
    ## geting a mask on Mr
    fits = fitsio.FITS(f)
    if amagMax is not None:
        amag_cut = amagMax - 5*np.log10(h)
        w = fits[1].where('AMAG[1] < %.2f'%(amag_cut))
    else: w = np.arange(fits[1].get_nrows(),dtype=np.int64)

    fits.close()
    return w

def loadfiles(filenames, columns=None, amagMax=None):
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
            w = get_amag_cut(f,amagMax) ## if amagMax None returns all the rows
            ## loading subset
            g0= Table(fitsio.read(f,rows=w,columns=columns))
            out.append(g0)
        i += 1
    mygals  = [data for data in out if data is not None]
    data       = vstack(mygals)
    return data

def get_critical_mass(cat2,files_truth):
    ### Truth table
    print('Load the halos files')
    cat= loadfiles(files_truth,columns=['HALOID','RA','DEC','Z','M200c','M200b'])
    #cat.rename_column('HOST_HALOID','HALOID')
    cat.rename_column('M200C','M200')

    print('\n Making cuts')
    print('clusters with M200<%.2E and z<%.2f'%(MassCut,zmax))
    cat = cat[(cat['Z']>=zmin)&(cat['Z']<=zmax)]
    cat = cat[cat['M200B']>=MassCut]

    ## get unique idx
    _, idx = np.unique(cat['HALOID'],return_index=True)
    cat = cat[idx]
    catb= cat

    print('matching halos truth files')
    cidx = cat2['HALOID']
    indices = list(chunks(catb['HALOID'],cidx))
    indices = np.array([int(ix[0]) for ix in indices if ix.size>0])
    
    cat = catb[indices]

    cat['sample'] = False
    for i,cid in enumerate(cidx):
        w, = np.where(cat['HALOID']==cid)
        if w.size>0:
            cat['RA'][w] = cat2['RA'][i]
            cat['DEC'][w] = cat2['DEC'][i]
            cat['sample'][w] = cat2['sample'][i]

    Z = cat['Z']
    M200c = cat['M200'][:]     ## [Msun/h] in physical units
    M200b = cat['M200B'][:]     ## [Msun/h] in physical units

    r200c = convertM200toR200(M200c*h,Z)/h  ## [Mpc/h] in physical units
    r200b = convertM200toR200(M200b*h,Z)/h  ## [Mpc/h] in physical units

    print('switching M200b and R200b to M200c and R200c')
    cat['R200B'] = r200b      ## [Mpc/h] mean, in comoving units 
    cat['R200'] = r200c

    return cat,catb

def get_random_selection(cat2,Nsize=3000):
    df = cat2.to_pandas()
    train_set, test_set = train_test_split(df, test_size=Nsize, random_state=42)

    cat3 = Table.from_pandas(test_set)
    return cat3

def get_high_mass_selection(cat2,Nsize=1000,nbins=25):
    Nsize+= 500

    M200 = cat2["M200"]
    Mh = M200[cat2["M200"]>=5e13]
    df = cat2.to_pandas()

    data = []
    mbins = np.logspace(np.log10(np.min(Mh)),np.log10(np.max(Mh)),nbins)
    for ml,mh in zip(mbins[:-1],mbins[1:]):
        w, = np.where((M200>=ml)&(M200<=mh))
        dfb= df.iloc[w]
        size=int(Nsize/nbins)
        if len(dfb)<size: size=len(dfb)-1
        if len(dfb)>0:
            print('bin size:',size)
            train_set, test_set = train_test_split(dfb, test_size=size, random_state=42)
            data.append(Table.from_pandas(test_set))

    cat_high_mass = vstack(data)
    return cat_high_mass

def computeNgals(g,keys,r200,mag_cut=-19.5,lcol='MAG_R'):
    ngals = []
    for idx,r2 in zip(keys,r200):
        w, = np.where((g['HALOID']==idx)&((g[lcol]-5*np.log10(h))<=mag_cut)&(g['RHALO']<=r2))
        # w, = np.where((g['HALOID']==idx)&((g[lcol]-5*np.log10(h))<=mag_cut))
        ni = w.size
        ngals.append(ni)
    return np.array(ngals)

def compute_ngals(cat,gcc,lcol='MAG_R'):
    gc3 = gcc[gcc["TRUE_MEMBERS"]==True]
    
    cidx = cat['HALOID']
    r200 = cat['R200'] ## critical radius

    cuts = [-19,-19.5,-20.,-20.5]
    labels = ["N190","N195","N200","N205"]
    
    for mc,li in zip(cuts,labels):
        n200 = computeNgals(gc3,cidx,r200,mag_cut=mc,lcol=lcol)
        cat[li] = n200
    return cat

def getIdx(g,idx,r200):
    # w, = np.where( (g['HALOID'] == int(idx) ) & (g['RHALO']<= r200) )
    w, = np.where( (g['HALOID'] == int(idx) ) )
    return w

def getTrueMembers(g,keys,r200):
    results = [dask.delayed(getIdx)(g,idx,r200[i]) for i,idx in enumerate(keys)]
    results = dask.compute(*results, scheduler='processes',num_workers=8)
    
    idxOut = np.empty(0,dtype=int)
    idxOut = [np.append(idxOut,w) for w in results if w.size>0]
    idxOut = np.concatenate(idxOut).ravel()
    return idxOut

def do_color_redshift_cut(zg1, mag, amag, crazy=[-1,4.], zrange=[0.01,1.],magMin=-18.):
    # gcut, rcut, icut, zcut = 24.33, 24.08, 23.44, 22.69 ## Y1 SNR_10 mag cut

    gr = mag[:,0]-mag[:,1]
    gi = mag[:,0]-mag[:,2]
    ri = mag[:,1]-mag[:,2]
    rz = mag[:,1]-mag[:,3]
    iz = mag[:,2]-mag[:,3]

    w, = np.where( (zg1>zrange[0]) & (zg1<zrange[1]) & 
                   (gr>crazy[0]) & (gr<crazy[1]) & (ri>crazy[0]) & (ri<crazy[1]) & (iz>crazy[0]) & (iz<crazy[1])& ((amag-np.log10(h)) <= magMin) )
                   #& (gi>crazy[0]) & (gi<crazy[1]) & (rz>crazy[0]) & (rz<crazy[1])) # ) #
    return w
    
def get_galaxy_cutouts(cat,g,Nside,rmax=8,zmin=0.,zmax=1.2):
    print('Healpix Pixel')
    hpx_gals = radec_pix(g['RA'],g['DEC'],nside=Nside)
    # hpx_cat = radec_pix(cat['RA'],cat['DEC'],nside=Nside)

    # print(g['healpix'][:5])
    print('\n Cut circles each galaxy cluster')
    DA = AngularDistance(np.array(cat['Z']))
    cat['rmax'] = 60*(float(rmax)/DA)*rad2deg ## arcmin
    # print(cat['rmax'][:5])

    healpix_list = get_healpix_list(cat,nside=Nside) ## healpix pixels within rmax from the GC center
    print(healpix_list)
    w, = np.where(np.in1d(hpx_gals,healpix_list))
    gc = g[w]
    print(len(w))

    print('\n do magnitude and crazy color cuts')
    w2 = do_color_redshift_cut(gc['Z'],gc['OMAG'],gc['AMAG'][:,1],zrange=[zmin,zmax])
    gcc = gc[w2]
    print('w2',len(w2))   

    print('geting the true members')
    gcc['TRUE_MEMBERS'] = False

    ## TRUE MEMBERS WITHIN R200b
    mask = gcc['RHALO']<=gcc['R200']
    w3 = getTrueMembers(gcc,cat['HALOID'],cat['R200'])
    gcc['TRUE_MEMBERS'][w3[mask[w3]]] = True

    return gcc

def get_galaxy_golden_sample(gcc,cat):
    print('switinching column names')
    z = gcc['Z']
    gcc['ZERR'] = 0.05*(1+z)

    magNames = ['G','R','I','Z']
    for i in range(4):
        magC_i = 'MAG_AUTO_%s'%(magNames[i])
        mag_i = gcc['OMAG'][:,i]

        magC_erri = 'MAGERR_AUTO_%s'%(magNames[i])
        mag_erri = gcc['OMAGERR'][:,i]
        
        gcc[magC_i] = mag_i
        gcc[magC_erri] = mag_erri

    gcc['FLAGS_GOLD'] = 0
    gcc['Mr'] = gcc['MAG_R']

    _, idx = np.unique(np.array(gcc['ID']),return_index=True)
    gcc = gcc[idx]

    gcc['healpix'] = radec_pix(gcc['RA'],gcc['DEC'],nside=1024)
    cat['healpix'] = radec_pix(cat['RA'],cat['DEC'],nside=1024)

    colnames = ['healpix','ID','HALOID', 'RA', 'DEC', 'Z', 'ZERR', 
                'MAG_AUTO_G','MAG_AUTO_R', 'MAG_AUTO_I', 'MAG_AUTO_Z', 
                'MAGERR_AUTO_G','MAGERR_AUTO_R', 'MAGERR_AUTO_I', 'MAGERR_AUTO_Z', 
                'FLAGS_GOLD','Mr', 'OMAG','AMAG', 'RHALO', 'CENTRAL','TRUE_MEMBERS']

    return gcc[colnames],cat

######### Setting the Code #########

####### Input Variables
Nhalos = 3000        ## choose the output sample size
nbins  = 25          ## choose how much bins you divide your whole sample
nprocess = 4

#######
rmax = 4               ## maximum radius in Mpc
zmin, zmax = 0.1, 1.0  ## for GC
MassCut = 1e12         ## the low Mass cut
Nside=1024
h=0.7
#######

####### Output Files
outdir = '/global/project/projectdirs/des/jesteves/buzzardSelection/v2.0.0/'

fileprefix      = 'buzzard_v2.0.0_{}.fits' 
file_cls_out_all= outdir+fileprefix.format('hod_halos')
file_cls_out    = outdir+fileprefix.format('%iHalos_cluster'%(Nhalos))
file_gal_out    = outdir+fileprefix.format('%iHalos_members'%(Nhalos))

hpx_file = outdir+'hpxmap_%i_buzzard_v2.0.0.fits'%Nside
#######


####### Input Files
indir         ='/global/project/projectdirs/des/jderose/Chinchilla/Herd/Chinchilla-3/v1.9.9/addgalspostprocess/'
files         = glob.glob(indir+'truth/Chinchilla-3_lensed_rs_shift_rs_scat_cam*')
files_truth   = glob.glob(indir+'halos/Chinchilla-3_halos*')

print('\n'.join(files[:5]))

def split(infile):
    return int(infile.split('.')[-2])

healpix = np.array([split(infile) for infile in files])
w, = np.where(healpix<60)

files = [files[i] for i in w]

files.sort()
#######

# ####### only for tests purposes
# files = files[:3]

######### Starting the Code #########
print('loading dataset')
t0 = time()
colnames = ['ID', 'HALOID', 'RA', 'DEC', 'Z', 'OMAG', 'OMAGERR', 'AMAG', 'RHALO', 'CENTRAL','M200','R200','MAG_R']

### multiple files
g = loadfiles(files,amagMax=-18,columns=colnames)

print('columns names')
print(g.colnames)

t_load = time()-t0
print('partial time: \n')
print(t_load,'\n')

print('getting healpix map')
if not os.path.isfile(hpx_file):
    ra,dec = g['RA'], g['DEC']
    hpx_map = make_hpx_map(ra,dec,Nside,hpx_file)

else:
    hpx_map = Table(getdata(hpx_file))

hpx_values = np.array(hpx_map['hpx_value'])

print('Selecting Galaxy Clusters')
central = (g['CENTRAL']==True)&(g['M200']>=MassCut)&(g['Z']>=zmin)&(g['Z']<=zmax)
c = g['HALOID','RA','DEC','Z','M200','R200','MAG_R'][central]

ngals = len(g)
ncls = len(c)

print('# galaxies:', ngals)
print('# clusters:', ncls)

## random selection of halos with mass less than 5 x E13 Msun
print('Selecting Silver Sample')
mask = c['M200']<5e13
cat_silver = get_random_selection(c[mask],Nsize=30000)

## uniform selection of halos with mass greater than 5 x E13 Msun
print('Selecting Golden Sample')
cat_golden = get_high_mass_selection(c,Nsize=Nhalos,nbins=nbins)

print('Mathching Sample')
## flag
cat_silver['sample'] = False
cat_golden['sample'] = True

catf = vstack([cat_golden,cat_silver])
keys, idx = np.unique(catf['HALOID'],return_index=True) ## unique keys
catf = catf[idx]

print('Geting Critical Mass')
## catf is whole sample with M200 added
## catb is the halos data
catf,catb = get_critical_mass(catf,files_truth)

print('Computing Area Fraction')
catf['area_frac'] = compute_area_fraction(catf,hpx_values,rmax=6,nside=Nside)

## spliting sample
mask = catf['sample']
cat_g = catf[mask]
cat_s = catf[np.logical_not(mask)]

print('Saving preliminar cluster samples')
catf = vstack([cat_g,cat_s])

print('->',file_cls_out)
print('->',file_cls_out_all)

cat_g.write(file_cls_out,format='fits',overwrite=True)
catf.write(file_cls_out_all,format='fits',overwrite=True)

## take all galaxies within 8Mpc from the cluster centers
## make redshift cuts and color cuts
## true members are galaxies with given HALOID and RHALO within R200b
print('Geting Galaxy Cutouts')
g2 = g[g['RHALO']<=1.5*g['R200']]

gal_g = get_galaxy_cutouts(cat_g,g,Nside,rmax=8,zmin=0.,zmax=4.)
g = 0.

gal_s = get_galaxy_cutouts(cat_s,g2,Nside,rmax=3,zmin=0.09,zmax=1.01)
g2 = 0.

## prepare galaxy file to copacabana; change columns names
print('Geting galaxy file')
gal_g, cat_g = get_galaxy_golden_sample(gal_g, cat_g)

## compute the number of galaxies (N190,N195,N200...) inside R200c for luminosity cut, -19, -19.5, -20., -20.5
print('Computing Ngals')
cat_g = compute_ngals(cat_g,gal_g,lcol='Mr')
cat_s = compute_ngals(cat_s,gal_s)

print('Saving output cluster samples')
catf = vstack([cat_g,cat_s])

print('->',file_cls_out)
print('->',file_cls_out_all)

cat_g.write(file_cls_out,format='fits',overwrite=True)
catf.write(file_cls_out_all,format='fits',overwrite=True)

print('Saving galaxy sample')
print('->',file_gal_out)
gal_g.write(file_gal_out,format='fits',overwrite=True)

