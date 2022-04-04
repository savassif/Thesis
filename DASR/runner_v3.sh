#!/bin/bash

for file in ../../Data/Sentinel-2_Images_Testing/*; do 
        echo $file 
        python3 quick_test.py --img_dir="$file" \
                     --scale='4' \
                     --resume=600 \
                     --blur_type='iso_gaussian'
done
