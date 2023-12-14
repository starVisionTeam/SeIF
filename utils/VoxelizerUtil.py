from __future__ import print_function, division
import os
import numpy as np
import math
import scipy.spatial
import scipy.io as sio
from subprocess import call
from scipy import ndimage

def resize_volume(volume, dim_x, dim_y, dim_z):
    new_volume = np.zeros((dim_x, dim_y, dim_z), dtype=np.uint8)
    scale_x = volume.shape[0]/dim_x
    scale_y = volume.shape[1]/dim_y
    scale_z = volume.shape[2]/dim_z

    print(scale_x, scale_y, scale_z)

    for xx in range(dim_x):
        for yy in range(dim_y):
            for zz in range(dim_z):
                xxx = int(round((xx+0.5) * scale_x - 0.5))
                yyy = int(round((yy+0.5) * scale_y - 0.5))
                zzz = int(round((zz+0.5) * scale_z - 0.5))
                new_volume[xx, yy, zz] = volume[xxx, yyy, zzz]

    return new_volume


def get_volume_from_points(points, dim_x, dim_y, dim_z, voxel_size):
    dim_x_half, dim_y_half, dim_z_half = dim_x / 2, dim_y / 2, dim_z / 2
    new_volume = np.zeros((dim_x, dim_y, dim_z), dtype=np.uint8)
    for p in points:
        xx = int(round(p[0]/voxel_size - 0.5 + dim_x_half))
        yy = int(round(p[1]/voxel_size - 0.5 + dim_y_half))
        zz = int(round(p[2]/voxel_size - 0.5 + dim_z_half))
        xx = min(max(0, xx), dim_x-1)
        yy = min(max(0, yy), dim_y-1)
        zz = min(max(0, zz), dim_z-1)
        new_volume[xx, yy, zz] = 1
    return new_volume

def save_volume_doubleIdx(volume, fname):

    x_dim, y_dim, z_dim = volume.shape[0], volume.shape[1], volume.shape[2]
    with open(fname, 'wb') as fp:
        for xx in range(x_dim):
            for yy in range(y_dim):
                for zz in range(z_dim):
                    if volume[xx, yy, zz] > 0:
                        pt = np.array([(xx*2),
                                       (yy*2),
                                       (zz*2)])
                        fp.write('v %f %f %f\n' % (pt[0], pt[1], pt[2]))

def save_volume(volume, fname, dim_h, dim_w, voxel_size):
    dim_h_half = dim_h / 2
    dim_w_half = dim_w / 2
    sigma = 0.05 * 0.05

    x_dim, y_dim, z_dim = volume.shape[0], volume.shape[1], volume.shape[2]
    with open(fname, 'w') as fp:
        for xx in range(x_dim):
            for yy in range(y_dim):
                for zz in range(z_dim):
                    if volume[xx, yy, zz] > 0:
                        pt = np.array([(xx - dim_w_half + 0.5) * voxel_size,
                                       (yy - dim_h_half + 0.5) * voxel_size,
                                       (zz - dim_w_half + 0.5) * voxel_size])
                        fp.write('v %f %f %f\n' % (pt[0], pt[1], pt[2]))


def save_v_volume(v_volume, fname, dim_h, dim_w, voxel_size):
    dim_h_half = dim_h / 2
    dim_w_half = dim_w / 2
    sigma = 0.05 * 0.05

    x_dim, y_dim, z_dim = v_volume.shape[0], v_volume.shape[1], v_volume.shape[2]
    with open(fname, 'w') as fp:
        for xx in range(x_dim):
            for yy in range(y_dim):
                for zz in range(z_dim):
                    if (v_volume[xx, yy, zz, :] != np.zeros((3,), dtype=np.float32)).any():
                        pt = np.array([(xx - dim_w_half + 0.5) * voxel_size,
                                       (yy - dim_h_half + 0.5) * voxel_size,
                                       (zz - dim_w_half + 0.5) * voxel_size])
                        fp.write('v %f %f %f %f %f %f\n' %
                                 (pt[0], pt[1], pt[2], v_volume[xx, yy, zz, 0], v_volume[xx, yy, zz, 1], v_volume[xx, yy, zz, 2]))


def save_volume_soft(volume, fname, dim_h, dim_w, voxel_size, thres):
    dim_h_half = dim_h / 2
    dim_w_half = dim_w / 2
    sigma = 0.05 * 0.05

    x_dim, y_dim, z_dim = volume.shape[0], volume.shape[1], volume.shape[2]
    with open(fname, 'wb') as fp:
        for xx in range(x_dim):
            for yy in range(y_dim):
                for zz in range(z_dim):
                    if volume[xx, yy, zz] > thres:
                        pt = np.array([(xx - dim_w_half + 0.5) * voxel_size,
                                       (yy - dim_h_half + 0.5) * voxel_size,
                                       (zz - dim_w_half + 0.5) * voxel_size])
                        fp.write('v %f %f %f\n' % (pt[0], pt[1], pt[2]))


def load_volume_from_mat(fname):
    return sio.loadmat(fname)


def rotate_volume(volume, view_id):

    new_volume = volume

    # canonize as viewed from left-2-right
    if view_id == 1:    # z-->x, (-x)-->z

        # print("Need to visually check each step?")
        # pdb.set_trace()
        if len(new_volume.shape) == 3:
            new_volume = np.transpose(new_volume, (2, 1, 0))
        elif len(new_volume.shape) == 4:
            new_volume = np.transpose(new_volume, (2, 1, 0, 3))
        new_volume = np.flip(new_volume, axis=2)

    # canonize as viewed from back-2-front
    elif view_id == 2: # (-x)-->x, (-z)-->z

        # print("Need to visually check each step?")
        # pdb.set_trace()
        new_volume = np.flip(new_volume, axis=0)
        new_volume = np.flip(new_volume, axis=2)

    # canonize as viewed from right-2-left
    elif view_id == 3: # (-z)-->x, x-->z

        # print("Need to visually check each step?")
        # pdb.set_trace()
        if len(new_volume.shape) == 3:
            new_volume = np.transpose(new_volume, (2, 1, 0))
        elif len(new_volume.shape) == 4:
            new_volume = np.transpose(new_volume, (2, 1, 0, 3))
        new_volume = np.flip(new_volume, axis=0)

    """
    if view-1: canonize as viewed from right-2-left
    if view-2: canonize as viewed from back-2-front
    if view-3: canonize as viewed from left-2-right
    """
    return new_volume


def binary_fill_from_corner_3D(input, structure=None, output=None, origin=0):

    # now True means outside, False means inside
    mask = np.logical_not(input)

    # mark 8 corners as True
    tmp = np.zeros(mask.shape, bool)
    for xi in [0, tmp.shape[0]-1]:
        for yi in [0, tmp.shape[1]-1]:
            for zi in [0, tmp.shape[2]-1]:
                tmp[xi, yi, zi] = True

    # find connected regions from the 8 corners, to remove empty holes inside the voxels
    inplace = isinstance(output, np.ndarray)
    if inplace:
        ndimage.binary_dilation(tmp, structure=structure, iterations=-1,
                                mask=mask, output=output, border_value=0,
                                origin=origin)
        np.logical_not(output, output)
    else:
        output = ndimage.binary_dilation(tmp,structure=structure,iterations=-1,mask=mask,border_value=0,origin=origin)
        np.logical_not(output, output) # now 1 means inside, 0 means outside

        return output

