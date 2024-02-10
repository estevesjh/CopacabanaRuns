import sys
import numpy as np
import h5py
from time import time


sys.path.append('/home/s1/jesteves/git/ccopa/python')
from main import copacabana
from make_input_files.make_photoz_pz import generate_photoz_models
from make_input_files.make_input_datasets import read_hdf5_file_to_dict

t0 = time()

root = '/data/des61.a/data/johnny/DESY3/projects/buzzardAllHalos/CopacabanaRuns/analysis/'
cfg  = root+'cf_all.yaml'
copa = copacabana(cfg,dataset='buzzard_v2')

#copa.make_input_file()

# def write_indices_out(indices,fname,col='02Lstar',overwrite=False):
#     fmaster = h5py.File(fname, 'a')
#     if 'indices' not in fmaster['members/'].keys():
#         fmaster.create_group('members/indices')

#     group = fmaster['members/indices']
    
#     try:
#         group.create_dataset(col,data=indices)
#     except:
#         if overwrite:
#             del group[col]
#             group.create_dataset(col,data=indices)
#         else:
#             print('Error: indices already exists')
    
#     fmaster.close()

# def apply_02Lstar_cut(fname):
#     print('loading data: %s'%fname)
#     out = read_hdf5_file_to_dict(fname,cols=['redshift','mag'],path='members/main/')
#     mag = out['mag'][:,2]  ## i-band
#     zcls= out['redshift'][:]
    
#     #print('applying mask')
#     cut = mag <= char_luminosity(zcls)+1.75
#     indices = np.where(cut)[0]
    
#     print('writing indices selection: %s \n'%('members/indices/02Lstar'))

#     write_indices_out(indices,fname,overwrite=False,col='02Lstar')

# Z1 = [ 2.85703803, 23.35451485]
# Z2 = [ -0.13290108,   1.11453393,  -3.74351207,   6.96351527, -10.04310863,  12.60755443,  -7.85820367,   0.92175766]
# def char_luminosity(z):
#     return np.poly1d(Z1)(np.log(z)) + np.poly1d(Z2)(z)

# for fname in copa.master_fname_tile_list:
#     apply_02Lstar_cut(fname)

# ##### Atention: this cell takes more than 6 hours to run. It only needs to be run one time.
# copa.run_bma_healpix(nCores=63,overwrite=True)

### nohup python run.py > log.out 2> log.err &

# ## Photo-z
# generate_photoz_models('gaussian',copa.master_fname_tile_list,0.01,nCores=60)

# print('gauss003')
# generate_photoz_models('gaussian',copa.master_fname_tile_list,0.03,nCores=60)

# print('gauss05)
# generate_photoz_models('gaussian',copa.master_fname_tile_list,0.05,nCores=60)

# outfile1=None#root+'aux_files/modelDNF_correction_mag_buzzard.txt'
# outfile2=root+'aux_files/modelDNF_correction_z_buzzard.txt'

# generate_photoz_models('bias',copa.master_fname_tile_list,0.03,
#                        zwindow_file=outfile2,zerror_file=outfile1,
#                        group_name='dnf',nCores=60)

# # outfile1=root+'aux_files/modelDNF_correction_mag_buzzard.txt'
# # outfile2=root+'aux_files/modelDNF_correction_z_buzzard.txt'

# # generate_photoz_models('bias',copa.master_fname_tile_list,0.03,
# #                        zwindow_file=outfile2,zerror_file=outfile1,
# #                        group_name='dnf_model',nCores=60)

# ## Copa Run
# # setup 
copa.kwargs['r_aper_model'] = 'rhod'
copa.kwargs['mag_selection']= '02Lstar'
## runs
pz_files = ['gauss001','gauss003','gauss005']
z_widths = [0.01,0.03,0.05]
runs = ['%s_%s_%s'%(pz,'rhod','02Lstar') for pz in pz_files]

pz_files += ['dnf']
z_widths += [0.03]
runs     += ['%s_%s_%s'%(pz,'rhod','02Lstar') for pz in pz_files[3:]]

for run, zfile, zw in zip(runs,pz_files,z_widths):
    print(5*'---')
    print('run     : %s'%run)
    print('zfile   : %s'%zfile)
    print('zwindow : %.2f'%zw)
    print(5*'---')
    if zfile!='gauss001':
        copa.kwargs['z_window'] = zw
        copa.run_copa_healpix(run,   pz_file=zfile, nCores=60)
        #copa.compute_muStar(run, overwrite=True)
        print(5*'---')
        print('\n')

## setup 
copa.kwargs['r_aper_model'] = 'r200'
copa.kwargs['mag_selection']= '02Lstar'
## runs
pz_files = ['gauss001','gauss003','gauss005']
z_widths = [0.01,0.03,0.05]

runs = ['%s_%s_%s'%(pz,'r200','02Lstar') for pz in pz_files]
pz_files += ['dnf']
z_widths += [0.03]
runs     += ['%s_%s_%s'%(pz,'rhod','02Lstar') for pz in pz_files[3:]]

for run, zfile, zw in zip(runs,pz_files,z_widths):
    print(5*'---')
    print('run     : %s'%run)
    print('zfile   : %s'%zfile)
    print('zwindow : %.2f'%zw)
    print(5*'---')
    
    copa.kwargs['z_window'] = zw
    #copa.run_copa_healpix(run,   pz_file=zfile, nCores=60)
    #copa.compute_muStar(  run, overwrite=True)
    print(5*'---')
    print('\n')

tf = (time()-t0)/60.
print('final time %.2f hours'%(tf/60.))
