link=https://pole.uchicago.edu/public/data/sptsz-clusters/sptecs_catalog_oct919.fits
link2=https://pole.uchicago.edu/public/data/sptsz-clusters/2500d_cluster_sample_Bocquet19.fits

echo "Download SPT Catalog"
echo "url:" $link

wget $link

echo "Download SPT2500d Catalog"
echo "url:" $link2

wget $link2
