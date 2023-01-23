## created on Jul 12th, 2022

import sys
import numpy as np
import pandas as pd

from astropy.table import Table, vstack
from astropy.io.fits import getdata
from collections import defaultdict
from time import time

sys.path.append("/home/s1/jesteves/git/ccopa/python/")
# sys.path.append('/home/s1/jesteves/git/buzzardAnalysis/mainAnalysis/selectionEffect/')
from main import copacabana
from fit_mor_evolution import mass_observable_relation, save_mor_results, load_mor_results

def get_mustar_log(x):
    x = np.where(x<1,1,x)
    return x

## Pivot points
Mp = 10.**15.5
Mup= 1./10**10

def load_datas(run, datas):
    cat0 = copa.load_copa_out('cluster',run)
    mask = (cat0['redshift']<0.65)&(cat0['MU']>1.)&(cat0['Ngals_true']>1.)
    cat = cat0[mask].copy()

    zcls = np.array(cat['redshift'])
    m200 = np.array(cat['M200_true']/Mp)

    mu    = get_mustar_log(cat['MU']/Mup)
    muerr = get_mustar_log(cat['MU_ERR_JK']/Mup)

    mut   = get_mustar_log(cat['MU_TRUE']/Mup)
    muterr= get_mustar_log(cat['MU_TRUE_ERR_JK']/Mup)

    ng    = np.array(cat['Ngals_flat'])
    ngt   = np.array(cat['Ngals_true'])
    
    datas[run]['x'] = m200
    datas[run]['z'] = zcls
    
    datas[run]['y1'] = mu
    datas[run]['y2'] = mut
    datas[run]['y1err'] = muerr
    datas[run]['y2err'] = muterr

    datas[run]['y3'] = ng
    datas[run]['y4'] = ngt
    return datas

def makeBin(variable, xedges):
    xbins = (xedges[1:]+xedges[:-1])/2
    indices = [ np.where((variable >= xedges[i]) & (variable <= xedges[i + 1]))[0] for i in range(len(xedges)-1)]
    return indices, xbins

## setup file
root = '/home/s1/jesteves/git/buzzardAnalysis/mainAnalysis/'
cfg = root+'config_buzzard_v2.yaml'
copa = copacabana(cfg, dataset='buzzard_v2')

# defining bins
nmbins = 11 # n+1
nzbins = 4 # n+1
outname = 'mor_uniform_matrix'
# outname = 'mor_uniform_pz_and_raper_variation_evol_nbins6'

# define runs
pz_files = ['gauss001', 'gauss003', 'gauss005']
z_widths = [0.01, 0.03, 0.05]
r_apers = [0.50, 0.75, 1.0]
runs_all = []
for zfile in pz_files:
    runs_all += ['{}_{}_{}_raper{:02d}'.format(zfile, 'rhod', '02Lstar', int(100*ri)) for ri in r_apers]

datas  = defaultdict(dict)
print('Loading Dataset')
for run in runs_all:
    print('run: %s'%run)
    datas = load_datas(run, datas)

print('Starting Mass Observable Evolution Fit')
total_time = 0.
mor_evol = mass_observable_relation()
# mor_evol = load_mor_results('mor_uniform_pz_and_raper_variation_evol_nbins6')

for run in runs_all:
    print(5*'---')
    print('run: %s'%run)
    zcls = datas[run]['z']
    mass = np.log10(np.array(datas[run]['x']))

    ## Bining the sample
    # zbins = np.percentile(zcls,np.linspace(0,100,11))
    zbins = np.linspace(0.1,0.65, nzbins)
    zkeys, zmed = makeBin(zcls,zbins)
    massBins = np.linspace(np.nanmin(mass), np.nanmax(mass), nmbins)
    binWidth = np.diff(massBins)[0]#/2.

    mor_evol.zmed = zmed
    zlabel = [r'%.3f < z < %.3f'%(zl,zh) for zl,zh in zip(zbins[:-1],zbins[1:])]
    i = 0
    for idx, zli in zip(zkeys, zlabel):
        print('zbin: %s'%zli)
        t0 = time()
        name = run+'_z%i'%i
        mor_evol.data[name]['zi'] = zmed[i]
        mor_evol.data[name]['nBins'] = len(zmed)

        mor_evol.add_dataset(name, datas[run]['x'][idx], datas[run]['y1'][idx])
        mor_evol.fit_kllr(name, is_log=True, bins=massBins, nbins=None, kernel_width=binWidth)

        mor_evol.add_dataset(name, datas[run]['x'][idx], datas[run]['y1'][idx], datas[run]['y1err'][idx])
        mor_evol.fit_linmix(name, is_log=True, nbins=nmbins, nchains=8, maxiter=100000)

        name = run+'_true_z%i'%i
        mor_evol.add_dataset(name, datas[run]['x'][idx], datas[run]['y2'][idx])
        mor_evol.fit_kllr(name, is_log=True, bins=massBins, nbins=None, kernel_width=binWidth)

        mor_evol.add_dataset(name, datas[run]['x'][idx], datas[run]['y2'][idx], datas[run]['y2err'][idx])
        mor_evol.fit_linmix(name, is_log=True, nbins=nmbins, nchains=8, maxiter=100000)
        
        partial_time = (time()-t0)/60.
        total_time += partial_time
        i += 1
        print('Partial time %.2f min'%partial_time)
        # print('\n')
    outname_tmp = outname+'_tmp'
    print('Saving tmp results: %s'%outname_tmp)
    save_mor_results(outname_tmp, mor_evol)
    
    print('Time elapsed %.2f min'%total_time)
    print('\n')

print('Saving results: %s'%outname)
save_mor_results(outname, mor_evol)

# # fiting
# print('Starting Mass Observable Relation Fit')
# total_time = 0.
# mor_all = mass_observable_relation()
# for run in runs_all:
#     print('run: %s'%run)
#     t0 = time()
#     mass = np.log10(np.array(datas[run]['x']))
#     massBins = np.linspace(np.nanmin(mass), np.nanmax(mass), 5)
#     binWidth = np.diff(massBins)[0]#/2.

#     mor_all.add_dataset(run, datas[run]['x'], datas[run]['y1'], datas[run]['y1err'])
#     mor_all.fit_kllr(run, is_log=True, nbins=5, kernel_width=0.35)
#     mor_all.fit_linmix(run, is_log=True, nbins=5, nchains=12, maxiter=100000)

#     mor_all.add_dataset(run+'wo_error', datas[run]['x'], datas[run]['y1'])
#     mor_all.fit_kllr(run+'wo_error', is_log=True, bins=massBins, nbins=None, kernel_width=binWidth)

#     mor_all.add_dataset(run+'_true', datas[run]['x'], datas[run]['y2'], datas[run]['y2err'])
#     mor_all.fit_kllr(run+'_true', is_log=True, nbins=5, kernel_width=0.35)
#     mor_all.fit_linmix(run+'_true', is_log=True, nbins=5, nchains=12, maxiter=100000)

#     mor_all.add_dataset(run+'_truewo_error', datas[run]['x'], datas[run]['y1'])
#     mor_all.fit_kllr(run+'_truewo_error', is_log=True, bins=massBins, nbins=None, kernel_width=binWidth)


#     partial_time = (time()-t0)/60.
#     total_time += partial_time
#     print('Partial time %.2f min'%partial_time)
#     print('Time elapsed %.2f min'%total_time)
#     print('\n')

# save_mor_results('mor_uniform_pz_and_raper_variation',mor_all)


# print('Starting Mass Observable Evolution Fit')
# total_time = 0.
# mor_evol = mass_observable_relation()

# for run in runs_all:
#     print(5*'---')
#     print('run: %s'%run)
#     zcls = datas[run]['z']
#     mass = np.log10(np.array(datas[run]['x']))

#     ## Bining the sample
#     # zbins = np.percentile(zcls,np.linspace(0,100,11))
#     zbins = np.linspace(0.1,0.65,13)
#     zkeys, zmed = makeBin(zcls,zbins)

#     mor_evol.zmed = zmed
#     zlabel = [r'%.3f < z < %.3f'%(zl,zh) for zl,zh in zip(zbins[:-1],zbins[1:])]
#     i = 0
#     for idx, zli in zip(zkeys, zlabel):
#         print('zbin: %s'%zli)
#         t0 = time()
#         name = run+'_z%i'%i
#         mor_evol.data[name]['zi'] = zmed[i]
#         mor_evol.data[name]['nBins'] = len(zmed)

#         mor_evol.add_dataset(name, datas[run]['x'][idx], datas[run]['y1'][idx], datas[run]['y1err'][idx])
#         mor_evol.fit_kllr(name, is_log=True, nbins=11, kernel_width=0.35)
#         mor_evol.fit_linmix(name, is_log=True, nbins=11, nchains=8, maxiter=100000)

#         name = run+'_true_z%i'%i
#         mor_evol.add_dataset(name, datas[run]['x'][idx], datas[run]['y2'][idx], datas[run]['y2err'][idx])
#         mor_evol.fit_kllr(name, is_log=True, nbins=11, kernel_width=0.35)
#         mor_evol.fit_linmix(name, is_log=True, nbins=11, nchains=8, maxiter=100000)
#         partial_time = (time()-t0)/60.
#         total_time += partial_time
#         i += 1
#         print('Partial time %.2f min'%partial_time)
#         # print('\n')
#     outname = run+'_uniform_evol'
#     print('Saving results: %s'%outname)
#     save_mor_results(outname, mor_evol)
#     print('Time elapsed %.2f min'%total_time)
#     print('\n')

# outname = 'mor_uniform_pz_and_raper_variation_evol_all'
# print('Saving results: %s'%outname)
# save_mor_results(outname, mor_evol)
