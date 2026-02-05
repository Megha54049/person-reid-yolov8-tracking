#!/bin/bash
# Simple activation script for running scripts in the virtual environment

cd /home/meghaagrawal940
source reid_env/bin/activate
exec "$@"
