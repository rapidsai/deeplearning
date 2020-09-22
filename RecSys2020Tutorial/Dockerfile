ARG dev=false
FROM nvcr.io/nvidia/cuda:10.2-devel-ubuntu18.04 AS base

# install python and cudf
RUN apt-get update
RUN apt-get -y install graphviz git

ADD https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh /miniconda.sh
RUN sh /miniconda.sh -b -p /conda && /conda/bin/conda update -n base conda && /conda/bin/conda create --name nvtabular -c rapidsai -c nvidia -c numba -c conda-forge -c defaults pip cudf=0.15 python=3.7 cudatoolkit=10.2 dask-cudf nodejs>=10.0.0 ipython jupyterlab

ENV PATH=${PATH}:/conda/bin
SHELL ["/bin/bash", "-c"]

RUN source activate nvtabular && pip3 install matplotlib pydotplus sklearn torch dask_cuda graphviz xgboost
RUN source activate nvtabular && pip3 install git+https://github.com/NVIDIA/NVTabular.git

# Create working directory to add repo.
WORKDIR /dli

# Load contents into student working directory, excluding anything in .dockerignore
ADD . .

# Set the initial working directory for students.
WORKDIR /dli/task

# Jupyter listens on 8888.
EXPOSE 8888

ENTRYPOINT ["/dli/entrypoint.sh"]

