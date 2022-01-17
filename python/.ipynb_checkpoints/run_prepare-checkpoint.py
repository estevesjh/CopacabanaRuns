import sys
import numpy as np
import h5py
from time import time

from magnitude_model import apply_02Lstar_cut

sys.path.append('/home/s1/jesteves/git/ccopa/python')
from main import copacabana
from make_input_files.make_photoz_pz import generate_photoz_models

t0 = time()

root = '/data/des61.a/data/johnny/DESY3/projects/CopacabanaRuns/analysis/aux_files/'
cfg  = root+'config_desy3_rm_full.yaml'
copa = copacabana(cfg,dataset='des_y3')

## Making input files
copa.make_input_file()

time_make_input = (time()-t0)/60.
print('Make Input Files: done')
print('Running time: %.2f min' % (time_make_input))

## Magnitude Model
for fname in copa.master_fname_tile_list:
    apply_02Lstar_cut(fname)

time_mag_model = (time()-t0)/60. - time_make_input
print('Magnitude Model: done')
print('Running time: %.2f min' % (time_mag_model))


## DNF Model
outfile1=root+'modelDNF_correction_z_gold_2_2.txt'
outfile2=root+'modelDNF_correction_z_gold_2_2.txt'

generate_photoz_models('bias',copa.master_fname_tile_list,0.03,
                       zwindow_file=outfile2,zerror_file=outfile1,
                       group_name='dnf_model',nCores=60)

time_photoz_model = (time()-t0)/60. - time_mag_model
print('Photo-z Model: done')
print('Running time: %.2f min' % (time_photoz_model))

### nohup python run_prepare.py > log.out 2> log.err &
