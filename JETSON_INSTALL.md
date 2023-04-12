
# Setup on Jeston Nano

The install instructions below have been tested on the 4 GB original (pre-Orin) Jetson Nano developer kit. The PyTorch installation instructions are adapted from [here](https://docs.ultralytics.com/yolov5/jetson_nano/#install-pytorch-and-torchvision).

```
# install latest pip
sudo apt-get install -y python3-pip
pip3 install --upgrade pip

# install torch v1.10.0
cd ~
sudo apt-get install -y libopenblas-base libopenmpi-dev
wget https://nvidia.box.com/shared/static/fjtbno0vpo676a25cgvuqc1wty0fkkg6.whl -O torch-1.10.0-cp36-cp36m-linux_aarch64.whl
pip3 install torch-1.10.0-cp36-cp36m-linux_aarch64.whl

# install torchvision v0.11.0 (takes 1 hr to build)
sudo apt install -y libjpeg-dev zlib1g-dev
git clone --branch v0.11.0 https://github.com/pytorch/vision torchvision
cd torchvision
pip3 install .

# install requirements (takes at least 1 hr to build opencv)
pip3 install -r jetson_nano_requirements.txt

# add Python executables to PATH to use gdown
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc 
source ~/.bashrc

# apply numpy architecture fix to avoid segfault
echo 'export OPENBLAS_CORETYPE=ARMV8' >> ~/.bashrc
source ~/.bashrc
```
