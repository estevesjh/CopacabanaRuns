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
matplotlib.rcParams.update({'font.size': 22})

import GCRCatalogs

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

def masking_duplicates(hid,z):
    mask  = (hid < 1e8)&(z<0.34)
    mask |= (hid > 1e8)&(z>=0.34)
    mask |= (hid < 1e9)&(z<0.9)
    mask |= (hid > 1e9)&(z>0.9)
    return mask

def get_critical_mass(cat2,files_truth,MassCut=1e12,zmin=0.,zmax=1.):
    ### Truth table
    print('Load the halos files')
    cat= loadfiles(files_truth,columns=['HALOID','Z','M200c','M200b','RVIR','Z_COS'])
    #cat.rename_column('HOST_HALOID','HALOID')
    cat.rename_column('M200C','M200')

    mask = masking_duplicates(cat['HALOID'],cat['Z'])
    cat  = cat[mask] 
    
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

def get_high_mass_selection(cat2,Nsize=1000,nbins=25,h=0.7):
    Nsize+= 1000

    M200 = cat2["M200"]*h
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

###########################################################################
##############################Galaxy Selection ############################
###########################################################################
    
class get_galaxy_sample:
    """ Get Galaxies around a cluster sample for the Buzzard v1.9.9 sample
    """
    def __init__(self,hpx,cat):
        self.hpx = hpx
        self.cat = cat
        
        self.indices  = np.where(cat['hpx8']==hpx)[0]
        self.hpx_list = self.cat['hpx8'][:]
        self.hpx_radec= self.cat['hpx8_radec'][:]

        self.t0 = time()
        print('\n')
        print(10*'---')
        print('Healpix %i'%(hpx))
        
    def get_healpix_neighbours(self,nside=8):
        _hpx_neighbours = get_healpix_list(self.cat[self.indices],nside=8)
        hpx_neighbours = _hpx_neighbours#np.array([self.match_healpix_to_file(hi) for hi in _hpx_neighbours])
        self.hpx_neighbours = hpx_neighbours[np.logical_not(np.isnan(hpx_neighbours))].astype(np.int64)
        print('hpx neighbours size: %i'%len(hpx_neighbours))
    
    def load_files(self,base,amagMax=-18.):
        #print('starting load_files()')
        self.files    = [base%(hi) for hi in self.hpx_neighbours]

        colnames      = ['ID', 'HALOID', 'RA', 'DEC', 'Z', 'OMAG', 'OMAGERR', 'AMAG', 'RHALO', 'CENTRAL','M200','R200','MAG_R']
        
        amag_cut = amagMax - 5*np.log10(h)
        mag_mask = 'AMAG[1] < %.2f'%(amag_cut)

        t0= time()
        g = loadfiles(self.files,mask=mag_mask,columns=colnames)
        load_time = (time()-t0)/60
        print('loading time: %.2f s \n'%(load_time))

        return g

    def load_files_gcr(self,catalog,columns,magMax=26):
        t0= time()
        data = catalog.get_quantities(columns,filters=['mag_i_lsst <= %.2f'%(magMax)], native_filters=[(lambda x: np.in1d(x, self.hpx_neighbours), 'healpix_pixel')]) 
        print('loading time: %.2f s \n'%(time()-t0))

        data = Table(data)
        data['hpx8'] = self.hpx
        data.rename_column('halo_id','HALOID')
        data.rename_column('ra','RA')
        data.rename_column('dec','DEC')
        data.rename_column('truth/RHALO','RHALO')

        return data

    def get_silver_sample(self,g,amagMax=-19.):
        #print('starting get_silver_sample()')
        hid_cls = np.unique(self.cat['HALOID'][self.indices])

        amag_cut = esutil.numpy_util.where1( g['Mag_true_r_des_z01']<= (amagMax - 5*np.log10(h)) )
        hid_gal = g['HALOID'][amag_cut]

        ### true members cut
        match = esutil.numpy_util.match(hid_cls,hid_gal)
        g_true_members = g[match[1]]
        #print('matching time: %.2f \n'%(time()-t0))

        return g_true_members
    
    def get_golden_sample(self,g,new_variables=False):
        #print('starting get_golden_sample()')
        t0 = time()

        c           = self.cat[self.indices]
        cat_golden  = c[c['sample']]

        ## make sky cutout
        ggold = make_healpix_cut(cat_golden,g,nside=1024,r_aper=8)

        ## define new variables
        if new_variables:
            ggold = get_galaxy_golden_sample(ggold,self.hpx)

        #print('time: %.2f s'%(time()-t0))
        return ggold
    
    def get_virial_mass(self,gsilver):
        gid, index = np.unique(gsilver['HALOID'][:],return_index=True)
        m200 = np.array(gsilver['halo_mass'][index])
        Z    = np.array(gsilver['redshift'][index])
        r200 = convertM200toR200(m200*h,Z)/h  ## [Mpc/h] in physical units

        match = esutil.numpy_util.match(gid,self.cat['HALOID'])
        self.cat['redshift'][match[1]] = Z[match[0]]
        self.cat['halo_mass'][match[1]]= m200[match[0]]

    def compute_cluster_richness(self,gsilver,lcol='Mag_true_r_des_z01'):
        #print('starting compute_cluster_richness()')
        #gsilver['Mr'] = gsilver['AMAG'][:,1]

        self.check_richness_columns(['redshift','halo_mass'])
        self.get_virial_mass(gsilver)
        
        res = compute_ngals(self.cat[self.indices],gsilver,lcol=lcol)
        columns = res.keys()
        self.check_richness_columns(columns)
        for col in columns:
            self.cat[col][self.indices] = res[col]

    
    def match_healpix_to_file(self,hpx):
        w, = np.where(self.hpx_radec==hpx)
        if w.size>0: 
            hpx_out = int(self.hpx_list[w][0])
        else:
            print('No healpix matched')
            hpx_out = np.nan
        return hpx_out
    
    def check_richness_columns(self,cols):
        columns = self.cat.columns
        nsize   = len(self.cat)
        
        for col in cols:
            if col not in columns:
                self.cat[col] = -1.*np.ones(nsize,dtype=np.float64)
    
def aper_match_healpix(rac,decc,redshift,rag,decg,nside=1024,radii  = 8):
    ## define healpix map
    hp1024 = healpixTools(nside,nest=True)
    
    # 1st step: center healpix list
    healpix_list = []
    for ra,dec,zcls in zip(rac,decc,redshift):
        hps_rad = hp1024._get_healpix_cutout_radec(ra,dec,zcls,radius=radii)
        healpix_list.append(hps_rad)
    healpix_list = np.unique(np.hstack(healpix_list))
    
    # 2nd step: galaxy healpix number
    healpix_gals = hp1024.radec_pix(rag,decg).astype(np.int64)

    # 3rd step: match
    mask = esutil.numpy_util.match(healpix_list,healpix_gals)
    gidx = mask[1]

    # gidx = np.empty(0,dtype=np.int64)
    # cidx = np.empty(0,dtype=np.int64)
    # for i,healpix_clus in enumerate(healpix_list):
    #     w, = np.where(np.in1d(healpix_gals,healpix_clus))
    #     gidx = np.append(gidx,w)
    #     cidx = np.append(cidx,np.full((w.size,),i,dtype=np.int64))
    
    return gidx

def make_healpix_cut(cdata,data,nside=1024,r_aper=8):
    """ Cut circles around each cluster using a healpix map
    """
    rac = cdata['RA'][:]
    decc = cdata['DEC'][:]
    redshift = cdata['Z'][:]
    
    rag = data['RA'][:]
    decg= data['DEC'][:]
    
    gidx = aper_match_healpix(rac,decc,redshift,rag,decg,nside=nside,radii  = 8)
    return data[gidx]

def computeNgals(g,keys,r200,mag_cut=-19.5,lcol='MAG_R'):
    ngals = []
    for idx,r2 in zip(keys,r200):
        #w, = np.where((g['HALOID']==idx)&((g[lcol]-5*np.log10(h))<=mag_cut)&(g['RHALO']<=r2))
        w = esutil.numpy_util.where1((g['HALOID']==idx)&((g[lcol]-5*np.log10(h))<=mag_cut)&(g['RHALO']<=r2))
        ni = w.size
        ngals.append(ni)
    return np.array(ngals)

def compute_ngals(cat,gcc,lcol='MAG_R'):
    cidx = cat['HALOID']
    r200 = cat['R200'] ## critical radius

    cuts = [-19,-19.5,-20.,-20.5]
    labels = ["N190","N195","N200","N205"]
    
    out = dict().fromkeys(labels)
    for mc,li in zip(cuts,labels):
        n200 = computeNgals(gcc,cidx,r200,mag_cut=mc,lcol=lcol)
        out[li] = n200
    return out

class healpixTools:
    """Healpix Operation Tools has a variety of functions which helps on the healpix operations for galaxy cluster science.
    """
    def __init__(self,nside,nest=False,h=0.7,Omega_m=0.283):
        """Starts a healpix map with a given nside and nest option.
        
        :param int nside: number of pixs of the map (usually a power of 2).
        :param bol nest : a boolean variable to order - TRUE (NESTED) or FALSE (RING).
        :param float h: hubble constant factor (H0 = h*100)
        :param float Omega_m: current omega matter density
        """
        self.nside = nside
        self.nest  = nest
        
        self.cosmo = FlatLambdaCDM(H0=100*h,Om0=Omega_m)
    
    def _get_healpix_cutout_radec(self,center_ra,center_dec,zcls,radius=1.):
        """ Get a healpix list which overlaps a circular cutout of a given radius centered on a given ra,dec coordinate. 

        :params float center_ra: ra coordinate in degrees.
        :params float center_dec: dec coordinate in degrees.
        :params float radius: aperture radius in Mpc.

        :returns: healpix list of the healpix values which overlaps the circular cutout.
        """
        center_ra_rad = np.radians(center_ra)
        center_dec_rad = np.radians(center_dec)

        center_vec = np.array([np.cos(center_dec_rad)*np.cos(center_ra_rad),
                            np.cos(center_dec_rad)*np.sin(center_ra_rad),
                            np.sin(center_dec_rad)])

        DA = float(self.AngularDistance(zcls))
        theta = (float(radius)/DA)*rad2deg    ## degrees

        healpix_list = healpy.query_disc(self.nside, center_vec, np.radians(theta), nest=self.nest, inclusive=True)

        return healpix_list

    def get_healpix_cutout_radec(self,ra_list,dec_list,redshift_list,radii=1):
        """ Get a healpix list which overlaps a circular cutout of a given radius centered on a given ra,dec coordinate. 
        
        :params ndarray ra_list: numpy array or list, ra coordinate in degrees.
        :params ndarray dec_list: dec coordinate in degrees.
        :params float radius: aperture radius in Mpc.
        
        :returns: healpix list of unique healpix values which overlaps all the circules defined by the list of center coordinates.
        """
        healpix_list = np.empty((0),dtype=int)
        for ra,dec,zcls in zip(ra_list,dec_list,redshift_list):
            hps_rad = self._get_healpix_cutout_radec(ra,dec,zcls,radius=radii)
            healpix_list = np.append(healpix_list,np.unique(hps_rad))
        return np.unique(healpix_list)

    def hpix2ang(self,pix):
        """ Convert pixels to astronomical coordinates ra and dec.
        
        :params pix: pixel values [int, ndarray or list]
        :returns: ra, dec [int, ndarray or list]
        """
        lon,lat = healpy.pix2ang(self.nside,pix,nest=self.nest)
        dec,ra=(90-(lon)*(180/np.pi)),(lat*(180/np.pi))
        return ra,dec

    def radec_pix(self,ra,dec):
        """ Convert astronomical coordinates ra and dec to pixels
        
        :params ra: ra [int, ndarray or list]
        :params dec: dec [int, ndarray or list]
        :returns: pixel values [int, ndarray or list]
        """
        return np.array(healpy.ang2pix(self.nside,np.radians(90-dec),np.radians(ra),nest=self.nest),dtype=np.int64)
    
    #@vectorize(signature="(),()->()")
    def AngularDistance(self,z):
        """Angular distance calculator
        :params float z: redshift
        :returns: angular distance in Mpc
        """
        DA = ( (self.cosmo.luminosity_distance(z)/(1+z)**2)/u.Mpc ) # em Mpc
        return DA
    
    def match_with_cat(self,df,hpx_clusters,radii=8):
        healpix_list = []
        for hpx in np.unique(hpx_clusters):
            w, = np.where(hpx_clusters==hpx)
            hp_list = self.get_healpix_cutout_radec(df['ra'].iloc[w],df['dec'].iloc[w],df['redshift'].iloc[w],radii=radii)
            healpix_list.append([hpx,hp_list])
        return healpix_list

def get_galaxy_golden_sample(gcc,hpx):
    print('switinching column names')
    magNames = ['G','R','I','Z']
    for i in range(4):
        magC_i = 'MAG_AUTO_%s'%(magNames[i])
        mag_i = gcc['OMAG'][:,i]

        magC_erri = 'MAGERR_AUTO_%s'%(magNames[i])
        mag_erri = gcc['OMAGERR'][:,i]
        
        gcc[magC_i] = mag_i
        gcc[magC_erri] = mag_erri

    gcc.rename_column('Z','z_true')

    gcc['FLAGS_GOLD'] = 0
    gcc['hpx8']       = hpx
    gcc['Mr']         = gcc['AMAG'][:,1]

    _, idx = np.unique(np.array(gcc['ID']),return_index=True)
    gcc = gcc[idx]

    colnames = ['hpx8','ID','HALOID', 'RA', 'DEC', 'z_true',
                'MAG_AUTO_G','MAG_AUTO_R', 'MAG_AUTO_I', 'MAG_AUTO_Z', 
                'MAGERR_AUTO_G','MAGERR_AUTO_R', 'MAGERR_AUTO_I', 'MAGERR_AUTO_Z', 
                'FLAGS_GOLD','Mr', 'RHALO', 'CENTRAL']

    return gcc[colnames]

def save_hdf5_output(gal,cat,outfile):
    df  = gal.to_pandas()
    df.to_hdf(outfile, key='members', mode='w')

    gal = 0

    dfc = cat.to_pandas()
    dfc.to_hdf(outfile, key='cluster', mode='a')

def getMagLimModel(auxfile,zvec,dm=0):
    '''
    Get the magnitude limit for 0.4L*
    
    file columns: z, mag_lim_i, (g-r), (r-i), (i-z)
    mag_lim = mag_lim + dm
    
    input: Galaxy clusters redshift
    return: mag. limit for each galaxy cluster and for the r,i,z bands
    output = [maglim_r(array),maglim_i(array),maglim_z(array)]
    '''
    annis=np.loadtxt(auxfile)
    jimz=[i[0] for i in annis]  ## redshift
    jimgi=[i[1] for i in annis] ## mag(i-band)
    jimgr=[i[2] for  i in annis]## (g-r) color
    jimri=[i[3] for i in annis] ## (r-i) color
    jimiz=[i[4] for i in annis] ## (i-z) color
    
    interp_magi=interp1d(jimz,jimgi,fill_value='extrapolate')
    interp_gr=interp1d(jimz,jimgr,fill_value='extrapolate')
    interp_ri=interp1d(jimz,jimri,fill_value='extrapolate')
    interp_iz=interp1d(jimz,jimiz,fill_value='extrapolate')

    mag_i,color_ri,color_iz = interp_magi(zvec),interp_ri(zvec),interp_iz(zvec)
    mag_r, mag_z = (color_ri+mag_i),(mag_i-color_iz)

    maglim_r, maglim_i, maglim_z = (mag_r+dm),(mag_i+dm),(mag_z+dm)

    magLim = np.array([maglim_r, maglim_i, maglim_z])
    
    return magLim.transpose()