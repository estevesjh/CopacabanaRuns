import sys
import numpy as np
import h5py
from time import time


sys.path.append('/home/s1/jesteves/git/ccopa/python')
from main import copacabana
from make_input_files.make_photoz_pz import generate_photoz_models

t0 = time()

root = '/data/des61.a/data/johnny/DESY3/projects/CopacabanaRuns/analysis/aux_files/'
cfg  = root+'config_desy3_rm.yaml'
copa = copacabana(cfg,dataset='des_y3')


##### Atention: this cell takes more than 6 hours to run. It only needs to be run one time.
#copa.run_bma_healpix(nCores=60,overwrite=False)

## setup 
copa.kwargs['r_aper_model'] = 'rhod'
copa.kwargs['mag_selection']= '02Lstar'

pz_files = ['dnf','dnf_model']
z_widths = [0.03,-1.]
runs = ['%s_%s_%s_test0'%(pz,'rhod','02Lstar') for pz in pz_files]

for run, zfile, zw in zip(runs,pz_files,z_widths):
    print('run     : %s'%run)
    print('zfile   : %s'%zfile)
    print('zwindow : %.2f'%zw)
    if zw==-1.:
        copa.kwargs['z_window'] = zw
        copa.run_copa_healpix(run,   pz_file=zfile, nCores=60)
        #copa.compute_muStar(  run, overwrite=True)
    print(5*'---')
    print('\n')

### nohup python run.py > log.out 2> log.err &
