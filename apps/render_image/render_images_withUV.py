import numpy as np
import os
import cv2 as cv
import glob
import math
import random
# import pyexr
from tqdm import tqdm
import scipy.io as sio
import prt.sh_util as sh_util
import copy
from Render.camera import Camera
from Render.mesh import load_obj_mesh, compute_tangent, compute_normal, load_obj_mesh_mtl

from utils.cam_util import *

view_num = 360
cam_f = 5000
cam_dist = 10
cam_t = np.array([0, 0, cam_dist])
img_resw = 512
img_resh = 512


def read_data(item, prt_file):
    """reads data """
    mesh_filename = item + '.obj'  # r'D:\FullData\facescape_norm\1\1_neural.obj'
    text_filename = item + '.jpg'  # assumes one .jpg file
    # print(mesh_filename)
    assert os.path.exists(mesh_filename) and os.path.exists(text_filename)
    vertices, faces, normals, faces_normals, textures, face_textures \
        = load_obj_mesh(mesh_filename, with_normal=True, with_texture=True)
    texture_image = cv.imread(text_filename)
    texture_image = cv.cvtColor(texture_image, cv.COLOR_BGR2RGB)

    # print(item)


    # father_path = os.path.abspath(os.path.dirname(mesh_filename) + os.path.sep + ".")  # ./data/data_obj
    # father_path = os.path.abspath(os.path.dirname(father_path) + os.path.sep + ".")  # ./data

    _, houmian = os.path.split(item)  # houmian = bareteeth.0000001

    # prt_file = r'D:\FullData\PRT\1'
    prt_filename = os.path.join(prt_file, houmian)  # D:\Dataset\PRT\people1\bareteeth.000001
    prt_filename = prt_filename + '.mat'  # D:\Dataset\PRT\people1\bareteeth.000001.mat
    assert os.path.exists(prt_filename)
    prt_data = sio.loadmat(prt_filename)
    prt, face_prt = prt_data['bounce0'], prt_data['face']
    return vertices, faces, normals, faces_normals, textures, face_textures, texture_image, prt, face_prt
    # import pdb
    # pdb.set_trace()


import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--index', type=int)
args = parser.parse_args()
index = args.index

def main():
    data_root = r'E:\FCH_Test\normalization'
    prt_root = r'E:\FCH_Test\PRT'
    save_data_file_root = r'E:\FCH_Test\Render'
    for file in np.arange(16, 17):
        file = 'people'+str(file)
        # file = '53_' + file

        data_item = os.path.join(data_root, file)  # r'D:\FullData\facescape_norm\53_1'
        prt_file = os.path.join(prt_root, file)  # 53_1
        save_data_file = os.path.join(save_data_file_root, file)  # D:\Dataset\Render\53_1

        if os.path.exists(data_item):  # if this path exits we can render it ;   r'D:\FullData\facescape_norm\1'
            if not os.path.exists(save_data_file):  # if render is exits ; we will pass this process ;

                for file_name in os.listdir(data_item):  #
                    qianzhui, houzhui = os.path.splitext(file_name)  # qianzhui = bareteeth.000001

                    if (houzhui == ".obj"):     # bareteeth.000001.obj
                        file_obj_path = os.path.join(data_item,
                                                     qianzhui)  # ； pl D:\FullData\facescape_norm\1\1_neural

                        from Render.gl.prt_render import PRTRender
                        rndr = PRTRender(width=img_resw, height=img_resh, ms_rate=1.0)
                        rndr_uv = PRTRender(width=img_resw, height=img_resh, uv_mode=True)
                        vertices, faces, normals, faces_normals, textures, face_textures, \
                        texture_image, prt, face_prt = read_data(file_obj_path,
                                                                 prt_file)  # D:/Dataset/COMA_norm/people1/bareteeth.000001 ;

                        '''
                        min_xyz = np.min(vertices, axis=0, keepdims=True)
                        max_xyz = np.max(vertices, axis=0, keepdims=True)
                        vertices = vertices - (min_xyz + max_xyz) * 0.5
                        scale_inv = np.max(max_xyz - min_xyz)
                        scale = 1.0 / scale_inv * (0.75 + np.random.rand() * 0.15)
                        vertices *= scale
                        '''
                        cam = Camera(width=img_resw, height=img_resh, focal=cam_f, near=0.1, far=40, camera_type='p')
                        cam.sanity_check()
                        vertsMean = 0
                        scaleMin = 1
                        rndr.set_norm_mat(scaleMin, vertsMean)
                        tan, bitan = compute_tangent(vertices, faces, normals, textures, face_textures)
                        rndr.set_mesh(vertices, faces, normals, faces_normals, textures, face_textures, prt, face_prt,
                                      tan,
                                      bitan)
                        rndr.set_albedo(texture_image)
                        rndr_uv.set_mesh(vertices, faces, normals, faces_normals, textures, face_textures, prt,
                                         face_prt,
                                         tan,
                                         bitan)
                        rndr_uv.set_albedo(texture_image)
                        cam.center = cam_t

                        # save_data_file = os.path.join(save_data_file_root, file)  #  D:\Dataset\Render\people1

                        os.makedirs(os.path.join(save_data_file, file_name, 'jpg'),
                                    exist_ok=True)  #   D:\Dataset\Render\people1\bareteeth.00001.obj\jpg
                        os.makedirs(os.path.join(save_data_file, file_name, 'mask_jpg'),
                                    exist_ok=True)  #   D:\Dataset\Render\people1\bareteeth.00001.obj\mask_jpg
                        new_path_jpg = os.path.join(save_data_file, file_name,
                                                    'jpg')  # D:\Dataset\Render\people1\bareteeth.00001.obj\jpg
                        new_path_mask_jpg = os.path.join(save_data_file, file_name,
                                                         'mask_jpg')  # D:\Dataset\Render\people1\bareteeth.00001.obj\mask_jpg

                        # for view_id in tqdm(range(0, view_num)):
                        #     R = make_rotate(0, view_id * np.pi / 180, 0)
                        #     rndr.rot_matrix = R.T
                        #     cam.sanity_check()
                        #     rndr.set_camera(cam)
                        #     rndr.display()
                        #     out_all_f = rndr.get_color(0)
                        #     out_mask = out_all_f[:, :, 3]
                        #     out_all_f = cv.cvtColor(out_all_f, cv.COLOR_RGBA2BGR)
                        #
                        #     # this_path = os.path.join(new_path_jpg, '%04d.jpg' % view_id)  # D:\Dataset\COMA_norm\people1\bareteeth.00001.obj\1.jpg
                        #     cv.imwrite(os.path.join(new_path_jpg, '%04d.jpg' % view_id), np.uint8(out_all_f * 255))
                        #     cv.imwrite(os.path.join(new_path_mask_jpg, '%04d_mask.jpg' % view_id),
                        #                np.uint8(out_mask * 255))
                        for view_id in tqdm(range(0, view_num)):
                            if view_id<50 or view_id>310:
                                R = make_rotate(0, view_id * np.pi / 180, 0)
                                rndr.rot_matrix = R.T
                                cam.sanity_check()
                                rndr.set_camera(cam)
                                rndr.display()
                                out_all_f = rndr.get_color(0)
                                out_mask = out_all_f[:, :, 3]
                                out_all_f = cv.cvtColor(out_all_f, cv.COLOR_RGBA2BGR)

                                # this_path = os.path.join(new_path_jpg, '%04d.jpg' % view_id)  # D:\Dataset\COMA_norm\people1\bareteeth.00001.obj\1.jpg
                                cv.imwrite(os.path.join(new_path_jpg, '%04d.jpg' % view_id), np.uint8(out_all_f * 255))
                                cv.imwrite(os.path.join(new_path_mask_jpg, '%04d_mask.jpg' % view_id),
                                           np.uint8(out_mask * 255))


            print('file = ', file)

    # data_item=r'D:\FullData\facescape_norm\1'
    # i =1
    # for file_name in os.listdir(data_item):   # bareteeth.000001.obj, bareteeth.000001.mtl, bareteeth.000001.png
    #     qianzhui,houzhui = os.path.splitext(file_name)  # qianzhui = bareteeth.000001
    #
    #     if(houzhui == ".obj"):  # bareteeth.000001.obj
    #         file_obj_path = os.path.join(data_item, qianzhui)   pl D:/Dataset/COMA_norm/people1/bareteeth.000001 ;
    #         print(i)
    #         from Render.gl.prt_render import PRTRender
    #         rndr = PRTRender(width=img_resw, height=img_resh, ms_rate=1.0)
    #         rndr_uv = PRTRender(width=img_resw, height=img_resh, uv_mode=True)
    #         vertices, faces, normals, faces_normals, textures, face_textures, \
    #         texture_image, prt, face_prt = read_data(file_obj_path)  # D:/Dataset/COMA_norm/people1/bareteeth.000001 ;
    #         #
    #         '''
    #         min_xyz = np.min(vertices, axis=0, keepdims=True)
    #         max_xyz = np.max(vertices, axis=0, keepdims=True)
    #         vertices = vertices - (min_xyz + max_xyz) * 0.5
    #         scale_inv = np.max(max_xyz - min_xyz)
    #         scale = 1.0 / scale_inv * (0.75 + np.random.rand() * 0.15)
    #         vertices *= scale
    #         '''
    #         cam = Camera(width=img_resw, height=img_resh, focal=cam_f, near=0.1, far=40, camera_type='p')
    #         cam.sanity_check()
    #         vertsMean = 0
    #         scaleMin = 1
    #         rndr.set_norm_mat(scaleMin, vertsMean)
    #         tan, bitan = compute_tangent(vertices, faces, normals, textures, face_textures)
    #         rndr.set_mesh(vertices, faces, normals, faces_normals, textures, face_textures, prt, face_prt, tan, bitan)
    #         rndr.set_albedo(texture_image)
    #         rndr_uv.set_mesh(vertices, faces, normals, faces_normals, textures, face_textures, prt, face_prt, tan,
    #                          bitan)
    #         rndr_uv.set_albedo(texture_image)
    #         cam.center = cam_t
    #
    #
    #
    #         save_data_file = r'D:\FullData\Render\1'
    #
    #         os.makedirs(os.path.join(save_data_file, file_name, 'jpg'), exist_ok=True)   D:\Dataset\Render\people1\bareteeth.00001.obj\jpg
    #         os.makedirs(os.path.join(save_data_file, file_name, 'mask_jpg'), exist_ok=True)  #  D:\Dataset\Render\people1\bareteeth.00001.obj\mask_jpg
    #         new_path_jpg = os.path.join(save_data_file,file_name, 'jpg')  #  D:\Dataset\Render\people1\bareteeth.00001.obj\jpg  ；
    #         new_path_mask_jpg = os.path.join(save_data_file, file_name, 'mask_jpg')  # D:\Dataset\Render\people1\bareteeth.00001.obj\mask_jpg  ；
    #
    #
    #         for view_id in tqdm(range(0, view_num)):
    #             R = make_rotate(0, view_id * np.pi / 180, 0)
    #             rndr.rot_matrix = R.T
    #             cam.sanity_check()
    #             rndr.set_camera(cam)
    #             rndr.display()
    #             out_all_f = rndr.get_color(0)
    #             out_mask = out_all_f[:, :, 3]
    #             out_all_f = cv.cvtColor(out_all_f, cv.COLOR_RGBA2BGR)
    #
    #             # this_path = os.path.join(new_path_jpg, '%04d.jpg' % view_id)  # D:\Dataset\COMA_norm\people1\bareteeth.00001.obj\1.jpg
    #             cv.imwrite(os.path.join(new_path_jpg, '%04d.jpg' % view_id), np.uint8(out_all_f * 255))
    #             cv.imwrite(os.path.join(new_path_mask_jpg, '%04d_mask.jpg' % view_id), np.uint8(out_mask * 255))
    #
    #         i = i+1
    #
    #


if __name__ == '__main__':
    main()