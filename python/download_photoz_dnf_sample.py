#!/usr/bin/env python3
import os
import requests

job_id = 'ed237dab070d4d598370070381bb1964'
job_type = 'query'
username = 'jesteves'
#base_url = 'https://des.ncsa.illinois.edu/desaccess/api'
base_url = 'https://deslabs.ncsa.illinois.edu/files-desaccess'

def download_job_files(url, outdir):
      #os.makedirs(outdir, exist_ok=True)
      r = requests.get('{}/json'.format(url))
      for item in r.json():
         print('Starting item: %s'%item['name'])
         if item['type'] == 'directory':
            suburl = '{}/{}'.format(url, item['name'])
            subdir = '{}/{}'.format(outdir, item['name'])
            download_job_files(suburl, subdir)
         elif item['type'] == 'file':
            data = requests.get('{}/{}'.format(url, item['name']))
            with open('{}/{}'.format(outdir, item['name']), "wb") as file:
                  file.write(data.content)
            print('writing outfile: %s'%'{}/{}'.format(outdir, item['name']))
      return r.json()

job_url = '{}/{}/{}/{}'.format(base_url, username, job_type, job_id)
download_dir = '/data/des61.a/data/johnny/DESY3/data/photoz/dnf_gold_2_2'
download_job_files(job_url, download_dir)
print('Files for job "{}" downloaded to "{}"'.format(job_id, download_dir))