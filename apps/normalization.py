import numpy as np
import trimesh
import copy
import os
from pymeshfix import PyTMesh
from ObjIO import load_obj_data,save_obj_data_with_mlt
from utils.cam_util import *

mesh_obj_root = '/media/amax/4C76448F76447C28/Dataset/COMA_crop/people14'
savepath = '/media/amax/4C76448F76447C28/Dataset/COMA_norm/people14'   # save obj_new normalization path;
i = 1
for filename in os.listdir(mesh_obj_root):
    qianzhui , houzhui = os.path.splitext(filename)  # qianzhui = bareteeth.000001 ; houzhui = obj, mtl, png
    if(houzhui == ".obj"):   # if end of is obj
        print(i)
        i = i +1
        meshPathTmp = os.path.join(mesh_obj_root, filename)  # COMA_crop/people1/bareteeth.0000001.obj
        mesh_crop = load_obj_data(meshPathTmp)
        mesh_normalization = mesh_crop.copy()  # new mesh which = mesh_crop

        # normalization vertices of obj;
        min_xyz = np.min(mesh_crop['v'], axis=0, keepdims=True)
        max_xyz = np.max(mesh_crop['v'], axis=0, keepdims=True)
        mesh_normalization['v'] = mesh_crop['v'] - (min_xyz + max_xyz) * 0.5
        scale_inv = np.max(max_xyz - min_xyz)
        scale = 1.0 / scale_inv * (0.75 + np.random.rand() * 0.15)
        mesh_normalization['v'] *= scale

        # people13和14需要rotate
        R = make_rotate(np.deg2rad(45), 0, np.deg2rad(0))
        mesh_normalization['v'] = np.matmul(mesh_normalization['v'], R)
        R = make_rotate(np.deg2rad(0), np.deg2rad(45), np.deg2rad(0))
        mesh_normalization['v'] = np.matmul(mesh_normalization['v'], R)

        # save new obj of normalization;
        saveTemp = os.path.join(savepath,filename)  # ./COMA_norm/people1/bareteeth.000001.obj
        save_obj_data_with_mlt(mesh_normalization, saveTemp, os.path.basename(meshPathTmp).replace('.obj', '.obj.mtl'))
        os.system('cp %s %s' % (meshPathTmp.replace('.obj', '.obj.mtl'), saveTemp.replace('.obj', '.obj.mtl')))   # copy mtl to new file
        os.system('cp %s %s' % (meshPathTmp.replace('.obj', '.png'), saveTemp.replace('.obj', '.png')))

        # after normalization , fill hole use this obj_normliazation
        mix = PyTMesh(False)
        mix.load_array(mesh_normalization['v'], mesh_normalization['f'])
        mix.fill_small_boundaries(refine=True)
        filename = saveTemp.replace("COMA_norm", "COMA_fill")
        mix.save_file(filename)





