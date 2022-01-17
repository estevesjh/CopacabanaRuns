import sys
import numpy as np
import h5py

import sys
sys.path.append('/home/s1/jesteves/git/ccopa/python')
from make_input_files.make_input_datasets import read_hdf5_file_to_dict

def rm_char_luminosity(z):
    """ RedMaPPer Luminosity Cut, equivalent to 0.2L_{star}
    See: https://iopscience.iop.org/article/10.1088/0004-637X/785/2/104
    """
    lnz  = np.log(z)
    res0 = 22.44+3.35*lnz+0.273*lnz**2-0.0618*lnz**3-0.0227*lnz**4
    res1 = 22.94+3.08*lnz-11.22*lnz**2-27.11*lnz**3-18.02*lnz**4
    res = np.where(z>0.5,res1,res0)
    return res

def apply_02Lstar_cut(fname):
    print('loading data: %s'%fname)
    out = read_hdf5_file_to_dict(fname,cols=['redshift','mag'],path='members/main/')
    mag = out['mag'][:,2]  ## i-band
    zcls= out['redshift'][:]
    
    #print('applying mask')
    #cut = mag <= char_luminosity(zcls)+1.75
    cut = mag <= rm_char_luminosity(zcls)+1.75
    indices = np.where(cut)[0]
    
    print('writing indices selection: %s \n'%('members/indices/02Lstar'))
    write_indices_out(indices,fname,overwrite=False,col='02Lstar')

def write_indices_out(indices,fname,col='02Lstar',overwrite=False):
    fmaster = h5py.File(fname, 'a')
    if 'indices' not in fmaster['members/'].keys():
        fmaster.create_group('members/indices')

    group = fmaster['members/indices']
    
    try:
        group.create_dataset(col,data=indices)
    except:
        if overwrite:
            del group[col]
            group.create_dataset(col,data=indices)
        else:
            print('Error: indices already exists')
    
    fmaster.close()
    
### nohup python run_prepare.py > log.out 2> log.err &
