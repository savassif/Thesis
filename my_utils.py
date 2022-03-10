from glob import glob
from importlib.resources import path
import earthpy as et
import earthpy.spatial as es
import earthpy.plot as ep
import os 
from osgeo import gdal
from datetime import datetime
import matplotlib.pyplot as plt
import rasterio as rio
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from skimage import exposure
import plotly.graph_objects as go

np.seterr(divide='ignore', invalid='ignore')

def beautify(img):
    stretched=_stretch_im(img,str_clip=1)
    stretched_uint8=es.bytescale(stretched)
    return stretched_uint8

def tiff2png(path_to_files,path_to_save):
    if path_to_files[-1]=='/':
        path_to_files = path_to_files[:-1]
    if not os.path.exists(str(path_to_save)):
        os.mkdir(path_to_save)
    for file in glob(path_to_files+'/*.tiff'):
        with rio.open(file,'r') as f:
            img = f.read()
            file_name = file.split('/')[-1].split('.')[0]
        img = beautify(img)
        img = img.astype(np.float32) / np.iinfo(img.dtype).max
        img = np.transpose(img,(1,2,0))
        print(file_name)
        print(img.shape[-1])
        if img.shape[-1] == 2 :
            plt.imsave(path_to_save+file_name+'_band1.png',img[:,:,0],cmap='gray')
            plt.imsave(path_to_save+file_name+'_band2.png',img[:,:,1],cmap='gray')
        else :
            plt.imsave(path_to_save+file_name+'_p1.png',img[:,:,:3])
            if img.shape[-1] == 6 :
                plt.imsave(path_to_save+file_name+'_p2.png',img[:,:,3:])
            else :
                plt.imsave(path_to_save+file_name+'_p2.png',img[:,:,3],cmap='gray')
                
def tiff2jpeg(path_to_files,path_to_save):
    if path_to_files[-1]=='/':
        path_to_files = path_to_files[:-1]
    if not os.path.exists(str(path_to_save)):
        os.mkdir(path_to_save)
    for file in glob(path_to_files+'/*.tiff'):
        with rio.open(file,'r') as f:
            img = f.read()
            file_name = file.split('/')[-1].split('.')[0]
        img = beautify(img)
        img = img.astype(np.float32) / np.iinfo(img.dtype).max
        img = np.transpose(img,(1,2,0))
        print(file_name)
        print(img.shape[-1])
        if img.shape[-1] == 2 :
            plt.imsave(path_to_save+file_name+'_band1.jpeg',img[:,:,0],cmap='gray')
            plt.imsave(path_to_save+file_name+'_band2.jpeg',img[:,:,1],cmap='gray')
        else :
            plt.imsave(path_to_save+file_name+'_p1.jpeg',img[:,:,:3])
            if img.shape[-1] == 6 :
                plt.imsave(path_to_save+file_name+'_p2.jpeg',img[:,:,3:])
            else :
                plt.imsave(path_to_save+file_name+'_p2.jpeg',img[:,:,3],cmap='gray')

def save_image(img,channels,path):
    with rio.open(
        path+'.tif',
        'w',
        driver='GTiff',
        height=img.shape[1],
        width=img.shape[2],
        count=channels,
        dtype=img.dtype,

    ) as dst:
        print(img.shape)
        dst.write(img)

def get_data2(path_to_directory,paths_to_imgs=False,imgs=True,norm=True):
    if path_to_directory[-1]=='/':
        path_to_directory = path_to_directory[:-1]
    path_dict = {}
    img_dict = {}
    for file in glob(path_to_directory+'/*.tiff'):
        if paths_to_imgs:
            if file.split('_')[-2] in path_dict:
                path_dict[file.split('_')[-2]].append(file)
            else : 
                path_dict[file.split('_')[-2]] = [file]
        if imgs:
            if file.split('_')[-2] in img_dict:
                f = gdal.Open(file)
                img = f.ReadAsArray()
                if norm:
                    img = _stretch_im(img,2)
                    img = img.astype(np.float32) / np.iinfo(img.dtype).max
                else:
                    img = img.astype(np.float32)
                img_dict[file.split('_')[-2]].append(img)
            else : 
                f = gdal.Open(file)
                img = f.ReadAsArray()
                if norm:
                    img = _stretch_im(img,2)
                    img = img.astype(np.float32) / np.iinfo(img.dtype).max
                else:
                    img = img.astype(np.float32)
                img_dict[file.split('_')[-2]] = [img]
                    
    if paths_to_imgs and imgs :
        return path_dict,img_dict
    elif paths_to_imgs:
        return path_dict
    elif imgs:
        return img_dict

def get_data(path_to_directory,paths_to_imgs=False,imgs=True,norm=True):
    if path_to_directory[-1]=='/':
        path_to_directory = path_to_directory[:-1]
    path_dict = {}
    img_dict = {}
    for file in glob(path_to_directory+'/*.tiff'):
        if paths_to_imgs:
            if file.split('_')[-2] in path_dict:
                path_dict[file.split('_')[-2]].append(file)
            else : 
                path_dict[file.split('_')[-2]] = [file]
        if imgs:
            if file.split('_')[-2] in img_dict:
                with rio.open(file,'r') as f:
                    img = f.read()
                    if norm:
                        img = img.astype(np.float32) / np.iinfo(img.dtype).max
                    else:
                        img = img.astype(np.float32)
                    img_dict[file.split('_')[-2]].append(img)
            else : 
                with rio.open(file,'r') as f:
                    img = f.read()
                    if norm:
                        img = img.astype(np.float32) / np.iinfo(img.dtype).max
                    else:
                        img = img.astype(np.float32)
                    img_dict[file.split('_')[-2]] = [img]
                    
    if paths_to_imgs and imgs :
        return path_dict,img_dict
    elif paths_to_imgs:
        return path_dict
    elif imgs:
        return img_dict
    
def get_patches(path_to_directory,type=None,image_num=None):
    if path_to_directory[-1]=='/':
        path_to_directory = path_to_directory[:-1]
    path_dict = {}
    img_dict = {}
    if type==None:
        full_path = path_to_directory+f'/image_{image_num}*.tif'
    else :
        full_path = path_to_directory+f'/*{type}.tif'
    for file in glob(full_path):
        if file.split('_')[-4].split('/')[-1] in img_dict:
            with rio.open(file,'r') as f:
                img = f.read()
                # img = img.astype(np.float32) / np.iinfo(img.dtype).max
                img_dict[file.split('_')[-4].split('/')[-1]].append(img)
        else : 
            with rio.open(file,'r') as f:
                img = f.read()
                # img = img.astype(np.float32) / np.iinfo(img.dtype).max
                img_dict[file.split('_')[-4].split('/')[-1]] = [img]

    return img_dict

def get_subimages(img,sub_images,channels_first=True):
    if channels_first :
        c,w,h = img.shape
    else :
        w,h,c = img.shape 
    size = w//sub_images
    assert w%sub_images==0, f'Cannot create {sub_images**2} non-overlapping subimages from image of shape : {img.shape}'
    simgs = []
    index_w = 0
    index_h = 0 
    while index_h+size<=h:
        if channels_first :
            for index_w in range(0,w,size):
                simgs.append(img[:,index_h:index_h+size,index_w:index_w+size])
            index_h +=size
        else :
            for index_w in range(0,w,size):
                simgs.append(img[index_h:index_h+size,index_w:index_w+size,:])
            index_h +=size
    return np.asarray(simgs)

def reconstruct_image(path_to_patches=None,patches=None,type=None,image_num=None):
    # Get patches in a dict 
    if path_to_patches :
        patches = get_patches(path_to_patches,type=type,image_num=image_num)
    # Get patches per dimension
    ppd = np.sqrt(len(patches))
    #Initialize a list for the rows of the final image 
    rows = []
    patch = 1 
    while patch <=len(patches):
        if patch == 1 :
            tmp = patches[f'{patch}'][0]
            patch+=1
            continue
        
        if patch % ppd == 0 :
            tmp = np.dstack((tmp,patches[f'{patch}'][0]))
            rows.append(tmp)
            if patch<len(patches): patch+=1
            tmp = patches[f'{patch}'][0]
        else:
            tmp = np.dstack((tmp,patches[f'{patch}'][0]))
        patch+=1

    return np.hstack(rows) 
        
def read_bands(img_path,resolution=None):
    assert (resolution==10 or resolution==20 or resolution==60 or resolution==None),"Supported Sentinel-2 resolutions are 10m, 20m and 60m" 
    resolutions = ['/R10m/*B?*.jp2','/R20m/*B?*.jp2','/R60m/*B?*.jp2']
    if resolution :
        if resolution == 10 :
            path = img_path+resolutions[0]
        elif resolution == 20 :
            path = img_path+resolutions[1]
        elif resolution == 60 :
            path = img_path+resolutions[2]
        S_sentinel_bands = glob(path)
        S_sentinel_bands.sort()
        l = []
        for i in S_sentinel_bands:
            with rio.open(i, 'r') as f:
                l.append(f.read(1))

        bands = np.stack(l)
        return bands,S_sentinel_bands
    else :
        S_sentinel_bands_dict = {}
        bands_dict = {}
        for r_path in resolutions:
            l=[]
            r = r_path.split('/')[1][1:]
            path = img_path+r_path
            tmp = glob(path)
            tmp.sort()
            S_sentinel_bands_dict[r] = tmp
            for i in S_sentinel_bands_dict[r]:
                with rio.open(i, 'r') as f:
                    l.append(f.read(1))
            bands_dict[r] = np.stack(l)
        return bands_dict,S_sentinel_bands_dict

def compose_rgb(S_sentinel_bands,dtype='uint16',channel_first=False,stretch=True,str_clip=0.2):
    rgb,info = es.stack(S_sentinel_bands[2::-1])
    assert dtype == 'uint16' or dtype == 'uint8','dytpe must be either uint16 or uint8'
    
    if dtype == 'uint8':
        rgb=_stretch_im(rgb,str_clip=str_clip)
        rgb=es.bytescale(rgb)
        
    if channel_first:
        return rgb,info
    else :
        return np.moveaxis(rgb, 0, -1),info
    
def _stretch_im(arr, str_clip):
#     Implementation of earthpy.plot
    """Stretch an image in numpy ndarray format using a specified clip value.

    Parameters
    ----------
    arr: numpy array
        N-dimensional array in rasterio band order (bands, rows, columns)
    str_clip: int
        The % of clip to apply to the stretch. Default = 2 (2 and 98)

    Returns
    ----------
    arr: numpy array with values stretched to the specified clip %

    """
    s_min = str_clip
    s_max = 100 - str_clip
    arr_rescaled = np.zeros_like(arr)
    for ii, band in enumerate(arr):
        lower, upper = np.nanpercentile(band, (s_min, s_max))
        arr_rescaled[ii] = exposure.rescale_intensity(
            band, in_range=(lower, upper)
        )
    return arr_rescaled.copy()

def adjusted_stretch_im(arr, str_clip):
#     Based on the _stretch_im Implementation of earthpy.plot
    """Stretch an image in numpy ndarray format using a specified clip value.

    Parameters
    ----------
    arr: numpy array
        N-dimensional array in rasterio band order (bands, rows, columns)
    str_clip: int
        The % of clip to apply to the stretch. Default = 2 (2 and 98)

    Returns
    ----------
    arr: numpy array with values stretched to the specified clip %

    """
    if arr.shape[0]>6:
        arr = np.moveaxis(arr,-1,0)
    s_min = str_clip
    s_max = 100 - str_clip
    arr_rescaled = np.zeros_like(arr)
    for ii, band in enumerate(arr):
        lower, upper = np.nanpercentile(band, (s_min, s_max))
        arr_rescaled[ii] = exposure.rescale_intensity(
            band, in_range=(lower, upper)
        )
    if arr_rescaled.shape[-1]>6:
        arr_rescaled = np.moveaxis(arr_rescaled,0,-1)
    return arr_rescaled.copy()

def create_folder(factor,path='/home/savvas/Thesis/Results/'):
    today = datetime.now()
    folder_name = path + f'SR_x{factor}_' +today.strftime('%d%m%Y')+'_'+ today.strftime("%H:%M:%S")
    os.mkdir(folder_name)
    return folder_name
