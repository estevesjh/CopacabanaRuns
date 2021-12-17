import os
import glob
import numpy as np
 
def get_healpix_list(file_list):
   healpix_list = []
   for infile in file_list:
       hpx = int(infile.split('_')[-1].split('.')[0])
       healpix_list.append(hpx)
   healpix_list = np.sort(np.array(healpix_list))
   return healpix_list
 
def rename_files(base_name,file_list):
   for infile in file_list:
       hpx = int(infile.split('_')[-1].split('.')[0])
       new_file = base_name.format(hpx)
       print((infile,new_file))
       os.rename(infile,new_file)
 
outdir = '/data/des61.a/data/johnny/DESY3/data/cutouts/'
files  = glob.glob(outdir+'y3_gold_2.2.1_wide_sofcol_run_redmapper_v6.4.22_wv1.2_subsampled_hpx8_*')
 
healpix_list = get_healpix_list(files)
 
print(healpix_list)

base_name = outdir+'y3_gold_2.2.1_wide_sofcol_run_redmapper_v6.4.22_wv1.2_subsampled_hpx8_{:05d}.hdf5'
rename_files(base_name,files)
