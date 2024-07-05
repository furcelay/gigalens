from astropy.io import fits
from skimage.measure import block_reduce
import json
import h5py
import numpy as np


binning = 4

image, hdr = fits.getdata('model_data/f140w_img.fits', header=True)

psf = np.load('model_data/f140_psf.npy')

with h5py.File('model_data/masks.h5', 'r') as hf:
    mask = hf['full_mask'][:]
    mask |= hf['sim_mask_1'][:]
    mask |= hf['sim_mask_3'][:]
    mask |= hf['sim_mask_4'][:]
    mask |= hf['sim_mask_5'][:]
    # mask |= hf['sim_mask_6'][:]
    # mask |= hf['sim_mask_7_extended'][:]

bkg_rms = hdr['BKGRMS'] / np.sqrt(binning)
exp_time = hdr['EXPTIME'] * binning

image = block_reduce(image, (binning, binning), np.mean)
psf = block_reduce(psf, (binning, binning), np.mean)
mask = block_reduce(mask, (binning, binning), np.min)

positions_dict = json.load(open('model_data/sources_pos_dict.json', 'r'))
centroids_x = [
    positions_dict['1']['x'],
    positions_dict['3']['x'],
    positions_dict['4']['x'],
    positions_dict['5']['x'],
    positions_dict['3.2']['x'],
    positions_dict['3.3']['x'],
    positions_dict['4.2']['x'],
    positions_dict['4.3']['x'],
    positions_dict['5.2']['x'],
    positions_dict['5.3']['x'],
]
centroids_y = [
    positions_dict['1']['y'],
    positions_dict['3']['y'],
    positions_dict['4']['y'],
    positions_dict['5']['y'],
    positions_dict['3.2']['y'],
    positions_dict['3.3']['y'],
    positions_dict['4.2']['y'],
    positions_dict['4.3']['y'],
    positions_dict['5.2']['y'],
    positions_dict['5.3']['y'],
]
centroids_err = [
    positions_dict['1']['err'],
    positions_dict['3']['err'],
    positions_dict['4']['err'],
    positions_dict['5']['err'],
    positions_dict['3.2']['err'],
    positions_dict['3.3']['err'],
    positions_dict['4.2']['err'],
    positions_dict['4.3']['err'],
    positions_dict['5.2']['err'],
    positions_dict['5.3']['err'],
]