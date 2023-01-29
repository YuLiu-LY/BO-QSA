# Download ShapeStacks dataset
cd data/ShapeStacks
wget http://shapestacks-file.robots.ox.ac.uk/static/download/v1/shapestacks-mjcf.tar.gz
wget http://shapestacks-file.robots.ox.ac.uk/static/download/v1/shapestacks-meta.tar.gz
wget http://shapestacks-file.robots.ox.ac.uk/static/download/v1/shapestacks-rgb.tar.gz
wget http://shapestacks-file.robots.ox.ac.uk/static/download/v1/shapestacks-iseg.tar.gz
tar xvzf shapestacks-meta.tar.gz
tar xvzf shapestacks-mjcf.tar.gz
tar xvzf shapestacks-rgb.tar.gz
tar xvzf shapestacks-iseg.tar.gz
# Download ObjectsRoom dataset
cd data/ObjectsRoom
gsutil -m cp -r \
  "gs://multi-object-datasets/objects_room" \
  .
# Download CLEVR-Tex dataset
cd data/CLEVR-Tex
wget https://thor.robots.ox.ac.uk/~vgg/data/clevrtex/clevrtex_packaged/clevrtex_full_part1.tar.gz
wget https://thor.robots.ox.ac.uk/~vgg/data/clevrtex/clevrtex_packaged/clevrtex_full_part2.tar.gz
wget https://thor.robots.ox.ac.uk/~vgg/data/clevrtex/clevrtex_packaged/clevrtex_full_part3.tar.gz
wget https://thor.robots.ox.ac.uk/~vgg/data/clevrtex/clevrtex_packaged/clevrtex_full_part4.tar.gz
wget https://thor.robots.ox.ac.uk/~vgg/data/clevrtex/clevrtex_packaged/clevrtex_full_part5.tar.gz
wget https://thor.robots.ox.ac.uk/~vgg/data/clevrtex/clevrtex_packaged/clevrtex_outd.tar.gz
wget https://thor.robots.ox.ac.uk/~vgg/data/clevrtex/clevrtex_packaged/clevrtex_camo.tar.gz
tar -zxvf clevrtex_full_part1.tar.gz
tar -zxvf clevrtex_full_part2.tar.gz
tar -zxvf clevrtex_full_part3.tar.gz
tar -zxvf clevrtex_full_part4.tar.gz
tar -zxvf clevrtex_full_part5.tar.gz
tar -zxvf clevrtex_outd.tar.gz
tar -zxvf clevrtex_camo.tar.gz
# Download Flowers dataset
cd data/Flowers
wget https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz
wget https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102segmentations.tgz
wget https://www.robots.ox.ac.uk/~vgg/data/flowers/102/setid.mat
tar -zxvf 102flowers.tgz
tar -zxvf 102segmentations.tgz
