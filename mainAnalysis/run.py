import sys
import numpy as np
import h5py
from time import time


sys.path.append('/home/s1/jesteves/git/ccopa/python')
from main import copacabana
from make_input_files.make_photoz_pz import generate_photoz_models

t0 = time()

root = '/home/s1/jesteves/git/buzzardAnalysis/mainAnalysis/'
cfg  = root+'config_buzzard_rm_v2.yaml'
copa = copacabana(cfg,dataset='buzzard_v2')


##### Atention: this cell takes more than 6 hours to run. It only needs to be run one time.
#copa.run_bma_healpix(nCores=65,overwrite=True)

### nohup python run.py > log.out 2> log.err

## setup 
copa.kwargs['r_aper_model'] = 'rhod'
copa.kwargs['mag_selection']= '02Lstar'
## runs
pz_files = ['gauss001','gauss003','gauss005']
z_widths = [0.01,0.03,0.05]
runs = ['%s_%s_%s'%(pz,'rhod','02Lstar') for pz in pz_files]

pz_files += ['dnf_model','dnf']
z_widths += [0.03,0.03]
runs     += ['%s_%s_%s'%(pz,'rhod','02Lstar') for pz in pz_files[3:]]

for run, zfile, zw in zip(runs,pz_files,z_widths):
    print(5*'---')
    print('run     : %s'%run)
    print('zfile   : %s'%zfile)
    print('zwindow : %.2f'%zw)
    print(5*'---')
    
    copa.kwargs['z_window'] = zw
    copa.run_copa_healpix(run,   pz_file=zfile, nCores=60)
    copa.compute_muStar(  run, overwrite=True)
    print(5*'---')
    print('\n')

    
## setup 
copa.kwargs['r_aper_model'] = 'r200'
copa.kwargs['mag_selection']= '02Lstar'
## runs
pz_files = ['gauss001','gauss003','gauss005']
z_widths = [0.01,0.03,0.05]

runs = ['%s_%s_%s'%(pz,'r200','02Lstar') for pz in pz_files]
pz_files += ['dnf_model','dnf']
z_widths += [0.03,0.03]
runs     += ['%s_%s_%s'%(pz,'r200','02Lstar') for pz in pz_files[3:]]

for run, zfile, zw in zip(runs,pz_files,z_widths):
    print(5*'---')
    print('run     : %s'%run)
    print('zfile   : %s'%zfile)
    print('zwindow : %.2f'%zw)
    print(5*'---')
    
    copa.kwargs['z_window'] = zw
    copa.run_copa_healpix(run,   pz_file=zfile, nCores=60)
    copa.compute_muStar(  run, overwrite=True)
    print(5*'---')
    print('\n')


tf = (time()-t0)/60.
print('final time %.2f hours'%(tf/60.))
