# Geo lab

## Setup using conda

```sh
# download Miniconda from https://conda.io/
curl -OL https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
# execute intall file
bash Miniconda*.sh
# relogin to apply PATH changes
exit

# create new env
conda create --name lab8
# activate env
conda activate lab8
# install python
conda install python==3.7
# install meshplot
pip install git+git://github.com/skoch9/meshplot.git@0.4.0
# install jupyter and others
pip install jupyter numpy matplotlib pythreejs
# add envname kernel to jupyter
python -m ipykernel install --user --name=lab8
# install other libs 
conda install igl 
pip install pyyaml


# install pytorch
# instead of 10.1 use a current version of cuda, see nvidia-smi
conda install pytorch=1.7.0 cudatoolkit=10.1 -c pytorch
# check CUDA version
python -c 'import torch; print(torch.version.cuda)'
# chech torch GPU
# See question: https://stackoverflow.com/questions/48152674/how-to-check-if-pytorch-is-using-the-gpu
python -c 'import torch; print(torch.rand(2,3).cuda())'
# install pytorch_scatter
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.7.0+cu101.html
# install torch-geometric
pip install torch-sparse
# install torch-geometric
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.7.0+cu101.html
```

## use Jupyter remotely

```sh
# ssh port forwarding
ssh -L localhost:8888:localhost:8889 servername
# activate env
conda activate envname
# Map a jupyter process to port 8889 on server
jupyter notebook --no-browser --port=8889
```
