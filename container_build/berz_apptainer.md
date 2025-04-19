Berzelius Apptainer Guide
1. Introduction
2. Apptainer on Berzelius
3. Downloading Apptainer Images
3.1 From Docker Hub
3.2 From NVIDIA NGC
4. Building Apptainer Images
5. Modifying Apptainer Images
5.1 Create a sandbox from an existing image
5.2 Enter the container and make modifications
5.3 Save the modified image
6. Binding Directories
7. Running Containers
7.1 Initializing a Shell
7.2 Executing Commands
8. Running Containers in Batch Jobs
9. Cheat Sheet
1. Introduction
Apptainer is an open-source containerization platform designed specifically for high-performance computing (HPC) and scientific computing environments. It focuses on delivering secure and efficient containerization for running applications and workflows, particularly within shared and multi-user HPC clusters.

In the context of containerization technologies like Apptainer, it’s important to distinguish between two closely related terms:

Container image: A static, standalone package that includes all the necessary files and configurations required to run an application or service.

Container: A running instance of a container image.

For AI and ML workloads on Berzelius—where complex environments and a high degree of user customizability are often required—NSC strongly recommends the use of container environments. Apptainer is the supported option, while Docker is not supported due to security considerations.

Using a container environment offers several key benefits, including:

Portability: Easily move workloads between different systems, including local laptops, Berzelius, and EuroHPC systems.

Reproducibility: Ensure consistent execution across platforms.

Flexibility: Choose your preferred operating system and environment, independent of the underlying host system, enabling a more familiar and user-friendly experience.

2. Apptainer on Berzelius
Apptainer is available on both login nodes and compute nodes.

[xuan@berzelius0 ~]$ apptainer --version
apptainer version 1.4.0-1.el8
You can check the available options and subcommands using --help:

apptainer --help
Apptainer image files can be quite large, and the /home/<your username> directory has a limited quota of 20 GB. To avoid issues, please store and run your Apptainer images directly from your project directory.

3. Downloading Apptainer Images
3.1 From Docker Hub
Apptainer is compatible with Docker images. Docker Hub is a cloud-based platform and registry service provided by Docker, serving as a central repository for container images. It allows developers to easily share, distribute, and collaborate on containerized applications and services.

Images hosted on Docker Hub can be easily pulled into Apptainer using a docker:// URL reference.

apptainer pull pytorch_2.0.1.sif docker://pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime
The image is stored locally as a .sif file (pytorch_2.0.1.sif, in this case).

3.2 From NVIDIA NGC
NVIDIA GPU Cloud (NGC) is a platform and repository offering a comprehensive collection of GPU-optimized containers, pre-trained deep learning models, and AI software. It is designed to accelerate and simplify AI and GPU-accelerated computing workflows.

apptainer pull tensorflow-20.03-tf2-py3.sif docker://nvcr.io/nvidia/tensorflow:20.03-tf2-py3
4. Building Apptainer Images
Running containers from publicly available images is not the only option. In many cases, it’s necessary to create a custom container image from scratch to meet specific requirements.

An Apptainer definition file is a configuration file that provides the instructions needed to build an Apptainer image. It defines the base environment, packages, dependencies, and custom setup steps.

Below is an example of an Apptainer definition file. For more details, please refer to the Apptainer User Guide for more details.

Let’s take a look at the key sections of an Apptainer definition file:

Header: The first two lines define the base image. In this example, the image cuda:11.7.1-cudnn8-devel-ubuntu22.04 from Docker Hub is used as the starting point.
%environment: This section is used to define environment variables that will be available inside the container at runtime.
%post: This section contains commands that will be executed during the build process, inside the container. It is typically used to install packages and set up the environment.
We set the environment variable PYTHONNOUSERSITE=1 to instruct Python inside the container to ignore the user-specific site-packages directory on the host system when searching for modules and packages.

This is particularly useful in a container environment, as it ensures that Python only uses packages installed within the container, preventing potential conflicts with user-installed packages on the host.

We first download the base image:

apptainer build cuda_11.7.1-cudnn8-devel-ubuntu22.04.sif  docker://nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04
Next, we build the image using the following Apptainer definition file:

Bootstrap: localimage
From: cuda_11.7.1-cudnn8-devel-ubuntu22.04.sif

%environment

export PATH=/opt/miniforge3/bin:$PATH
export PYTHONNOUSERSITE=True

%post

apt-get update && apt-get install -y --no-install-recommends \
git \
nano \
wget \
curl

# Install Miniforge3
cd /tmp
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh -fp /opt/miniforge3 -b
rm Miniforge3*sh

export PATH=/opt/miniforge3/bin:$PATH

mamba install python==3.10 pytorch==2.0.1 torchvision torchaudio torchdata torchtext pytorch-cuda=11.7 -c pytorch -c nvidia -y

# Pin packages
cat <<EOT > /opt/miniforge3/conda-meta/pinned
pytorch==2.0.1
EOT

mamba install matplotlib scipy pandas -y
Building an Apptainer image typically requires root access. However, on Berzelius, you can build images—within certain limitations—using the fakeroot feature.

The fakeroot feature allows an unprivileged user to act as a “fake root” inside the container by leveraging user namespace UID/GID mapping. This provides nearly the same administrative privileges as the real root user, but only within the container and its associated namespaces.

apptainer build --fakeroot pytorch_2.0.1.sif pytorch_2.0.1.def
You can build the image directly from a base image without downloading it from a registry. To do this, simply change the definition file header to:

Bootstrap: docker
From: nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04
5. Modifying Apptainer Images
You can modify an existing Apptainer image to suit your specific requirements using the --sandbox option. This creates a writable directory where you can make changes to the image contents.

5.1 Create a sandbox from an existing image
apptainer build --fakeroot --sandbox pytorch_2.0.1 pytorch_2.0.1.sif
This command creates a writable directory named pytorch_2.0.1/ based on the original .sif image.

5.2 Enter the container and make modifications
Use apptainer shell with the --writable flag to enter the container and install or modify components:

apptainer shell --fakeroot --writable pytorch_2.0.1/
Apptainer> mamba install jupyterlab -y 
Apptainer> apt update  
Apptainer> apt install vim -y
Apptainer> exit
5.3 Save the modified image
apptainer build pytorch_2.0.1_v2.sif pytorch_2.0.1
6. Binding Directories
Sometimes it’s necessary to access directories outside the default locations when running a container.

By default, Apptainer automatically binds the following paths into the container:

The user’s home directory ($HOME)
The current directory when the container is executed ($PWD)
System-defined paths: /tmp, /proj, etc.
You can bind additional directories using the --bind or -B flag. The syntax is:

apptainer shell -B /proj/<your project>/users/<your username>/data:/data pytorch_2.0.1.sif
In this example, the directory /proj/<your project>/users/<your username>/data on the host is made available as /data inside the container. The colon : separates the host path from the container path.

7. Running Containers
7.1 Initializing a Shell
The apptainer shell command initializes an interactive shell inside the container. Use the --nv flag to enable NVIDIA GPU support.

To exit the container, press exit or press Ctrl + D.

[xuan@node001 containers]$ apptainer shell --nv pytorch_2.0.1.sif
Apptainer> exit
exit
[xuan@node001 containers]$
Note: When you exit the container, all running processes are terminated. However, any changes made to bound directories are preserved. By default, changes made elsewhere in the container are not saved.

7.2 Executing Commands
The apptainer exec command allows you to run a single command inside the container:

[xuan@node001 containers]$ apptainer exec --nv pytorch_2.0.1.sif python -c "import torch; print('GPU Name: ' + torch.cuda.get_device_name(0))"
GPU Name: NVIDIA A100-SXM4-40GB
8. Running Containers in Batch Jobs
You can run Apptainer containers as part of a batch job by integrating them into a SLURM job submission script. Here is an example batch script:

 #!/bin/bash

#SBATCH -A your_proj_account
#SBATCH --gpus=1
#SBATCH --time 00:10:00

apptainer exec --nv -B /local_data_dir:/data apptainer_image.sif python some_script.py
Notes:

The --nv flag enables NVIDIA GPU support inside the container.

The -B flag binds /local_data_dir from the host to /data inside the container.

Replace apptainer_image.sif with the name of your container image.

Replace some_script.py with your script or command to run.

9. Cheat Sheet
Operation	Command
Downloading images	apptainer pull pytorch_2.0.1.sif docker://pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime
Building images	apptainer build –fakeroot apptainer_image.sif apptainer_image.def
Initializing a shell	apptainer shell –nv apptainer_image.sif
Executing commands	apptainer exec –nv -B /local_data_dir:/data apptainer_image.sif bash -c “python some_script.py”
