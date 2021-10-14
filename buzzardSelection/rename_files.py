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

#outdir = '/data/des61.a/data/johnny/Buzzard/Buzzard_v2.0.0/Buzzard_v2.0.0/y3/'
#files  = glob.glob(outdir+'buzzard_v2.0.0_*.hdf')

outdir = '/data/des61.a/data/johnny/Buzzard/Buzzard_v2.0.0/y3_rm/tiles/'
files  = glob.glob(outdir+'buzzard_y3_v2.0.0_redmapper_heidi_lgt20_copper_hpx8*.hdf')

 
healpix_list = get_healpix_list(files)
 
print(healpix_list)

base_name = '/data/des61.a/data/johnny/Buzzard/Buzzard_v2.0.0/y3_rm/input/buzzard_v2.0.0_lambda_gt20_{:05d}.hdf5'
rename_files(base_name,files)