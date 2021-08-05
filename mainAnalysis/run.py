import sys
import numpy as np
import h5py
from time import time


sys.path.append('/home/s1/jesteves/git/ccopa/python')
from main import copacabana
from make_input_files.make_photoz_pz import generate_photoz_models

t0 = time()

root = '/home/s1/jesteves/git/buzzardAnalysis/mainAnalysis/'
cfg  = root+'config_buzzard_v2.yaml'

copa = copacabana(cfg,dataset='buzzard_v2')

##### Atention: this cell takes more than 6 hours to run. It only needs to be run one time.
copa.run_bma_healpix(nCores=60,overwrite=False)

### nohup python run.py > log.out 2> log.err