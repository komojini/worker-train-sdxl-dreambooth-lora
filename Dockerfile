# Base image
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

# Use bash shell with pipefail option
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Set the working directory
WORKDIR /workspace

COPY builder/setup.sh /workspace/