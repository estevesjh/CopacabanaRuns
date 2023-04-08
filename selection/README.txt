### SPT Clusters Cutouts ###

# 1) selectHalos.ipynb
I joined the spt ECS and 2500d samples. A redshift cut of 0.05 was applied. 

# 2) selectGalaxies.ipynb
I generated DES footprint cutouts around each cluster position. 
The galaxy sample is magnitude limited (i-band <23.5). 
There are some clusters that don't have galaxies in the DES footprint.
The tiles numbers should be retrieved from the file list.
In a tile is possible to have a cluster without galaxies. 
An analysis of the fraction of des galaxies in the footprint should be performed. 

# 3) cleanEmptyFields.ipynb
Set a flag if the cluster has a masked/empty region more than 20%. 

# 4) modelRedshiftBias.ipynb
 Model the galaxy distribution redshift peak for each cluster. 
Use the dnf_mc column. 
