#!/bin/bash

# Update package list and install necessary system packages
sudo apt-get update -y
sudo apt-get install -y ffmpeg

# Install Python dependencies
pip3  install -U docarray
pip install -r requirements.txt
