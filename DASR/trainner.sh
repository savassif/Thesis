#!/bin/bash
python3 main.py --dir_data='/home/savvas/Thesis/Scripts/DASR/Train_set/' \
               --model='blindsr' \
               --scale='6' \
               --blur_type='iso_gaussian' \
               --noise=25.0 \
               --lambda_min=0.2 \
               --lambda_max=4.0 \
	       --epochs_encoder=150 \
	       --epochs_sr=450
