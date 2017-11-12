##### Tensorflow install cheat sheet
##### Alex Shah
##### NOT A WORKING SCRIPT, RUN LINES MANUALLY
#
# https://www.nvidia.com/en-us/data-center/gpu-accelerated-applications/tensorflow/
# https://www.tensorflow.org/install/install_linux
#
# files in tf_cuda8_cudnn6_install.zip:
#
# NVIDIA-Linux-x86_64-384.90.run (latest)
# NVIDIA-Linux-x86_64-375.66.run (reported by TF)
# cuda_8.0.61_375.26_linux.run (CUDA 8.0 Main file, contains 375.26 driver)
# cuda_8.0.61.2_linux.run (Patch 2)
# cudnn-8.0-linux-x64-v6.0.tgz (CUDnn 6.0 for CUDA 8.0 archive)
# libcudnn6_6.0.21-1+cuda8.0_amd64.deb (CUDnn 6.0 for CUDA 8.0 runtime)
# libcudnn6-dev_6.0.21-1+cuda8.0_amd64.deb (CUDnn 6.0 for CUDA 8.0 dev)
# 
# Steps might take minutes running blank, let them run
#
#####

# Assuming the zip is in your downloads

cd ~/Downloads
unzip tf_cuda8_cudnn6_install.zip
cd ~/Downloads/tf_cuda8_cudnn6_install/

# Prep and dependencies

sudo apt-get install libcupti-dev python3-pip python3-dev linux-headers-generic build-essential dkms

## OPTIONAL
# Remove old nvidia packages
sudo apt-get --purge remove nvidia*
sudo apt update && sudo apt autoremove

# Install tf
# permission error in pip solved by adding --user to the end of install command, or running with sudo

pip3 install --upgrade pip
pip3 install tensorflow-gpu
sudo pip3 install --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.3.0-cp35-cp35m-linux_x86_64.whl

# Install gpu drivers

sudo service lightdm stop
sudo killall Xorg
sudo chmod +x NVIDIA-Linux-x86_64-384.90.run
sudo sh NVIDIA-Linux-x86_64-384.90.run

##### GPU driver problems:
# 1) Error with nouveau and auto patch doesn't work try:
sudo apt-get --purge remove xserver-xorg-video-nouveau
# Or add "blacklist nouveau" to /etc/modprobe.d/blacklist.conf
# Backup initramfs
sudo mv /boot/initramfs-$(uname -r).img /boot/initramfs-$(uname -r).img.bak
# Rebuild it
sudo apt install dracut
sudo dracut -v /boot/initramfs-$(uname -r).img $(uname -r)
# Reboot
sudo init 3
# Try again

# 2) Error with kernel, download kernel headers and source, dkms
# Or try older kernel
#####

# Install cuda from extracted files
# Cuda v 8.0
# Select NO to install GPU drivers as we already have them

sudo sh cuda_8.0.61_375.26_linux.run

# Add paths to ~/.bashrc

echo "export PATH=/usr/local/cuda-8.0/bin${PATH:+:${PATH}}" >> ~/.bashrc
echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64/:usr/local/cuda/lib/:usr/local/cuda-8.0/lib64/:usr/local/cuda-8.0/lib/" >> ~/.bashrc
source ~/.bashrc

# Patch 2 file
cd ~/Downloads/tf_cuda8_cudnn6_install
sudo sh cuda_8.0.61.2_linux.run
# accept, enter

# Install cudnn
# Cudnn 6.0
# via deb
sudo dpkg -i libcudnn6_6.0.21-1+cuda8.0_amd64.deb
sudo dpkg -i libcudnn6-dev_6.0.21-1+cuda8.0_amd64.deb

# or manually from archive
tar -xzvf cudnn-8.0-linux-x64-v6.0.tgz
sudo cp cuda/include/cudnn.h /usr/local/cuda/include
sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64
sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*

#now that it's installed, get more stuff
sudo apt update
sudo apt install cuda
sudo apt install nvidia-cuda*
sudo apt install nvidia-modprobe

sudo reboot

##### DONE! #####
#Failures:

#choosing gpu, 0 or 1 or 0,1 whatever
export CUDA_VISIBLE_DEVICES=0

# CUDA failures after running
sudo apt install nvidia-modprobe
nvidia-cuda-mps-server

#Build from source
git clone https://github.com/tensorflow/tensorflow
sudo apt-get install python3-numpy python3-dev python3-pip python3-wheel

sudo apt-get install openjdk-8-jdk curl
echo "deb [arch=amd64] http://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list
curl https://bazel.build/bazel-release.pub.gpg | sudo apt-key add -
sudo apt-get update && sudo apt-get install bazel
sudo apt-get upgrade bazel

cd tensorflow
./configure

## config choices
Please specify the location of python. [Default is /usr/bin/python]: /usr/bin/python3.5
Found possible Python library paths:
  /usr/local/lib/python3.5/dist-packages
  /usr/lib/python3.5/dist-packages
Please input the desired Python library path to use.  Default is [/usr/lib/python3.5/dist-packages] /usr/lib/python3.5/dist-packages

Do you wish to build TensorFlow with MKL support? [y/N]
No MKL support will be enabled for TensorFlow
Please specify optimization flags to use during compilation when bazel option "--config=opt" is specified [Default is -march=native]:
Do you wish to use jemalloc as the malloc implementation? [Y/n]
jemalloc enabled
Do you wish to build TensorFlow with Google Cloud Platform support? [y/N]
No Google Cloud Platform support will be enabled for TensorFlow
Do you wish to build TensorFlow with Hadoop File System support? [y/N]
No Hadoop File System support will be enabled for TensorFlow
Do you wish to build TensorFlow with the XLA just-in-time compiler (experimental)? [y/N] N
No XLA support will be enabled for TensorFlow
Do you wish to build TensorFlow with VERBS support? [y/N]
No VERBS support will be enabled for TensorFlow
Do you wish to build TensorFlow with OpenCL support? [y/N]
No OpenCL support will be enabled for TensorFlow
Do you wish to build TensorFlow with CUDA support? [y/N] Y
CUDA support will be enabled for TensorFlow
Do you want to use clang as CUDA compiler? [y/N]
nvcc will be used as CUDA compiler
Please specify the Cuda SDK version you want to use, e.g. 7.0. [Leave empty to default to CUDA 8.0]: 8.0
Please specify the location where CUDA 8.0 toolkit is installed. Refer to README.md for more details. [Default is /usr/local/cuda]:
Please specify which gcc should be used by nvcc as the host compiler. [Default is /usr/bin/gcc]:
Please specify the cuDNN version you want to use. [Leave empty to default to cuDNN 6.0]: 6
Please specify the location where cuDNN 6 library is installed. Refer to README.md for more details. [Default is /usr/local/cuda]:
Please specify a list of comma-separated Cuda compute capabilities you want to build with.
You can find the compute capability of your device at: https://developer.nvidia.com/cuda-gpus.
Please note that each additional compute capability significantly increases your build time and binary size.
[Default is: "3.5,5.2"]: 3.5
Do you wish to build TensorFlow with MPI support? [y/N] 
MPI support will not be enabled for TensorFlow
Configuration finished

#then run
bazel build --config=opt --config=cuda //tensorflow/tools/pip_package:build_pip_package
bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
sudo pip3 install /tmp/tensorflow_pkg/tensorflowSomething.whl