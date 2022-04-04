from model.blindsr import BlindSR
import torch
import numpy as np
import imageio
import argparse
import os
import utility
import cv2
from my_utils import beautify
import rasterio as rio 


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', type=str, default='D:/LongguangWang/Data/test.png',
                        help='image directory')
    parser.add_argument('--scale', type=str, default='2',
                        help='super resolution scale')
    parser.add_argument('--resume', type=int, default=600,
                        help='resume from specific checkpoint')
    parser.add_argument('--blur_type', type=str, default='iso_gaussian',
                        help='blur types (iso_gaussian | aniso_gaussian)')
    return parser.parse_args()


def main():
    args = parse_args()
    if args.blur_type == 'iso_gaussian':
        dir = './experiment/blindsr_x' + str(int(args.scale[0])) + '_bicubic_iso'
    elif args.blur_type == 'aniso_gaussian':
        dir = './experiment/blindsr_x' + str(int(args.scale[0])) + '_bicubic_aniso'

    # path to save sr images
    save_dir = dir + '/results'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    DASR = BlindSR(args).cuda()
    DASR.load_state_dict(torch.load(dir + '/model/model_' + str(args.resume) + '.pt'), strict=False)
    DASR.eval()

    with rio.open(args.img_dir,'r') as f:
        img = f.read()
        img = img.astype(np.float32) / np.iinfo(img.dtype).max 
    lr = beautify(img)
    
    # inference
    if lr.shape[0]==4:
        lr = lr[:3,:,:]
        lr = np.ascontiguousarray(lr)
        lr = torch.from_numpy(lr).float().cuda().unsqueeze(0).unsqueeze(0)
        sr = DASR(lr[:, 0, ...])
        sr = utility.quantize(sr, 255.0)
          # save sr results
        img_name = args.img_dir.split('.tiff')[0].split('/')[-1]
        sr = np.array(sr.squeeze(0).permute(1, 2, 0).data.cpu())
        sr = sr[:, :, [2, 1, 0]]
        cv2.imwrite(save_dir + '/' + img_name + '_rgb_sr.png', sr)
    elif lr.shape[0]==2:
        return
    else :
        lr1 = lr[:3,:,:]
        lr2 = lr[3:,:,:]
        lr1 = np.ascontiguousarray(lr1)
        lr1 = torch.from_numpy(lr1).float().cuda().unsqueeze(0).unsqueeze(0)
        lr2 = np.ascontiguousarray(lr2)
        lr2 = torch.from_numpy(lr2).float().cuda().unsqueeze(0).unsqueeze(0)
        sr1 = DASR(lr1[:, 0, ...])
        sr2 = DASR(lr2[:, 0, ...])
        sr1 = utility.quantize(sr1, 255.0)
        sr2 = utility.quantize(sr2, 255.0)
        # save sr results
        img_name = args.img_dir.split('.tiff')[0].split('/')[-1]
        sr1 = np.array(sr1.squeeze(0).permute(1, 2, 0).data.cpu())
        sr1 = sr1[:, :, [2, 1, 0]]
        cv2.imwrite(save_dir + '/' + img_name + '_p1_sr.png', sr1)
          # save sr results
        img_name = args.img_dir.split('.tiff')[0].split('/')[-1]
        sr2 = np.array(sr2.squeeze(0).permute(1, 2, 0).data.cpu())
        sr2 = sr2[:, :, [2, 1, 0]]
        cv2.imwrite(save_dir + '/' + img_name + '_p2_sr.png', sr2)


if __name__ == '__main__':
    with torch.no_grad():
        main()