from lib.mesh_util import *
from lib.sample_util import *
from lib.train_util import *
from lib.model import *
import scipy
import scipy.io as sci
from ObjIO import load_obj_data
import os

'''
Ensure that the obj model and FLAME model are spatially aligned before manipulation.
The Flame model was derived from DECA during training. (Paper: Learning an Animatable Detailed 3D Face Model from In-The-Wild Images )

Can get all Semantic information of each obj;
'''

template_Flame_path='/media/amax/4C76448F76447C28/FullData/template/template_normalization.obj'
template_Flame_obj = load_obj_data(template_Flame_path)
nose_tip = 29641
nose_tip_xyz = template_Flame_obj['v'][nose_tip]

# face_idx = np.where(np.linalg.norm(template_Flame_obj['v']-nose_tip_xyz,axis=-1)<0.33)[0].tolist()

#########################################################################
point_index = []
face_region_model_path = '/media/amax/4C76448F76447C28/FullData/template/region/face_and_ear_region/face_and_ear.obj'
# face_region_model_path ='/media/amax/4C76448F76447C28/SePifu_trainDate/template/region/face_region/face_region.obj'
face_region_mesh = load_obj_data(face_region_model_path)
# face_region_mesh = Mesh.load(face_region_model_path)

T_model_path = '/media/amax/4C76448F76447C28/FullData/template/template_normalization.obj'
T_model_mesh = load_obj_data(T_model_path)
# T_model_mesh = Mesh.load(T_model_path)

for i in np.arange(0,face_region_mesh['v'].shape[0]):
    mask_sphere = np.where(np.linalg.norm(T_model_mesh['v'] - face_region_mesh['v'][i], axis=-1) <= 0.0001)
    point_index.append(mask_sphere[0][0])

face_idx = point_index
###########################################################################

face_vtx_std=template_Flame_obj['v'][face_idx]
# normalization of it location ;
min_x = np.min(face_vtx_std[:, 0])
max_x = np.max(face_vtx_std[:, 0])
min_y = np.min(face_vtx_std[:, 1])
max_y = np.max(face_vtx_std[:, 1])
min_z = np.min(face_vtx_std[:, 2])
max_z = np.max(face_vtx_std[:, 2])
face_vtx_std[:, 0] = (face_vtx_std[:, 0] - min_x) / (max_x - min_x)
face_vtx_std[:, 1] = (face_vtx_std[:, 1] - min_y) / (max_y - min_y)
face_vtx_std[:, 2] = (face_vtx_std[:, 2] - min_z) / (max_z - min_z)
face_vtx_std_vertex_code = np.zeros_like(template_Flame_obj['v'])
# face_vtx_std_vertex_code = np.ones_like(template_Flame_obj['v'])-0.3   # only visualize use!
# this region_semantic == the location of this region point process normalization!
face_vtx_std_vertex_code[face_idx,:] = np.float32(np.copy(face_vtx_std))
save_obj_mesh_with_color('/home/amax/Desktop/template_Flame_sematic.obj',template_Flame_obj['v'],template_Flame_obj['f'][:,::-1],face_vtx_std_vertex_code)


KNN_K = 4
SIGMA_SQUARE = 0.05*0.05
def get_semantic_mesh(mesh_path, FLAME_path):
    # assert (os.path.exists(smpl_path))
    Flame_obj = load_obj_data(FLAME_path)
    mesh_obj = load_obj_data(mesh_path)
    kd_tree_smpl_v = scipy.spatial.KDTree(Flame_obj['v'])  # create KD-Tree from 6890 SMPL vertices
    # KNN searching from mesh-vertex to SMPL-vertex
    dist_list, id_list = kd_tree_smpl_v.query(mesh_obj['v'], k=KNN_K)  # (N,k), (N,k)
    # compute semantic label for each mesh vertex
    weight_list = np.exp(-np.square(dist_list) / SIGMA_SQUARE)  # (N,k)
    weight_sum = np.zeros((weight_list.shape[0], 1))  # (N,1)
    mesh_vertex_label = np.zeros((weight_list.shape[0], 3))  # (N,3)
    mesh_vertex_label_keepRatio = np.zeros((weight_list.shape[0], 3))  # (N,3)
    for ni in range(KNN_K):
        weight_sum[:, 0] += weight_list[:, ni]
        mesh_vertex_label += weight_list[:, ni:(ni + 1)] * face_vtx_std_vertex_code[id_list[:, ni], :]
        mesh_vertex_label_keepRatio += weight_list[:, ni:(ni + 1)] * face_vtx_std_vertex_code[id_list[:, ni], :]
    mesh_vertex_label /= weight_sum  # this is add weight and calculate average!
    mesh_vertex_label_keepRatio /= weight_sum

    # clip into [0,1]
    _ = np.clip(a=mesh_vertex_label, a_min=0, a_max=1, out=mesh_vertex_label)
    _ = np.clip(a=mesh_vertex_label_keepRatio, a_min=0, a_max=1, out=mesh_vertex_label_keepRatio)

    save_visualize = False
    if save_visualize:
        savepath = mesh_path.split(os.path.basename(mesh_path))[0]
        savepath = savepath.replace('facescape_fill', 'Visualize')
        if not os.path.exists(savepath):
            os.makedirs(savepath,exist_ok=True)
        savepath = savepath+os.path.basename(mesh_path)
        save_obj_mesh_with_color(savepath.replace('.obj','_semantic.obj'),mesh_obj['v'],mesh_obj['f'][:,::-1], mesh_vertex_label)
    return mesh_vertex_label

GT_head_path_root = '/media/amax/4C76448F76447C28/FullData/facescape_fill'
GT_Flame_path_root = '/media/amax/4C76448F76447C28/FullData/3DMMGT_Flame'
occ_path_root = '/media/amax/4C76448F76447C28/FullData/OCC'
occ_save_root = '/media/amax/4C76448F76447C28/FullData/OCC_Semantic'
for people_name in os.listdir(GT_head_path_root):
    people_path = os.path.join(GT_head_path_root,people_name)
    for biaoqing in os.listdir(people_path):
        biaoqing_path = os.path.join(people_path,biaoqing)
        biaoqing_Flame_path = os.path.join(GT_Flame_path_root, people_name, biaoqing, 'DECA_detail.obj')
        occ_path = os.path.join(occ_path_root, people_name, biaoqing.replace('.obj', '.mat'))
        occ_semantic_save = os.path.join(occ_save_root,people_name)
        if not os.path.exists(occ_semantic_save):
            os.makedirs(occ_semantic_save,exist_ok=True)
        occ_semantic_save_path = os.path.join(occ_semantic_save,biaoqing.replace('.obj','_semantic.mat'))

        # GT_head_path='/media/amax/4C76448F76447C28/刘旭/SePifu_trainDate/facescape_fill/1/2_smile.obj'
        # GT_Flame_path='/media/amax/4C76448F76447C28/刘旭/SePifu_trainDate/3DMMGT_Flame/1/2_smile.obj/DECA_detail.obj'
        GT_head_path = biaoqing_path
        GT_Flame_path = biaoqing_Flame_path
        print(GT_head_path)
        print(GT_Flame_path)
        print(occ_path)
        print(occ_semantic_save_path)

        GT_head_obj = load_obj_data(GT_head_path)
        #  get gtHead_model semantic information ;  from FLAME model ;
        mesh_vertex_label = get_semantic_mesh(GT_head_path, GT_Flame_path)


        # take semantic from head_model to OCC!
        # occ_path = '/media/amax/4C76448F76447C28/SePifu_trainDate/OCC/1/2_smile.mat'
        # occ_semantic_path = '/media/amax/4C76448F76447C28/SePifu_trainDate/OCC/1/2_smile_semantic.mat'
        occ_semantic_path = occ_semantic_save_path
        occupancy =  sci.loadmat(occ_path, verify_compressed_data_integrity=False)
        surface_points_inside = occupancy['surface_points_inside']
        surface_points_outside = occupancy['surface_points_outside']
        uniform_points_inside = occupancy['uniform_points_inside']
        uniform_points_outside = occupancy['uniform_points_outside']

        kd_tree_GT_head_obj = scipy.spatial.KDTree(GT_head_obj['v'])  # create KD-Tree from GT_head model ;
        # KNN searching from GT_headmodel to OCC!
        dist_list, id_list = kd_tree_GT_head_obj.query(surface_points_inside, k=1)
        surface_points_weight_list = np.exp(-np.square(dist_list) / 0.0025)
        surface_points_inside_semantic_list = surface_points_weight_list[:,None]*mesh_vertex_label[id_list,:]

        dist_list, id_list = kd_tree_GT_head_obj.query(uniform_points_inside, k=1)  # (N,k), (N,k)
        uniform_points_weight_list = np.exp(-np.square(dist_list) / 0.0025)
        uniform_points_inside_semantic_list = uniform_points_weight_list[:,None]*mesh_vertex_label[id_list,:]


        dist_list, id_list = kd_tree_GT_head_obj.query(uniform_points_outside, k=1)  # (N,k), (N,k)
        uniform_points_weight_list = np.exp(-np.square(dist_list) / 0.0025)
        uniform_points_outside_semantic_list = uniform_points_weight_list[:,None]*mesh_vertex_label[id_list,:]


        dist_list, id_list = kd_tree_GT_head_obj.query(surface_points_outside, k=1)  # (N,k), (N,k)
        surface_points_weight_list = np.exp(-np.square(dist_list) / 0.0025)
        surface_points_outside_semantic_list = surface_points_weight_list[:,None]*mesh_vertex_label[id_list,:]

        sci.savemat(occ_semantic_path,
                    {
                        'surface_points_inside': surface_points_inside_semantic_list,
                        'surface_points_outside': surface_points_outside_semantic_list,
                        'uniform_points_inside': uniform_points_inside_semantic_list,
                        'uniform_points_outside': uniform_points_outside_semantic_list
                    }, do_compression=True)



        # whether visualize and save point of OCC_Semantic!
        visualize = False
        if visualize:
            fname = '/media/amax/4C76448F76447C28/FullData/Visualize/samples_all.ply'
            points1 = surface_points_inside  # [N, 3]
            points2 = surface_points_outside
            points3 = uniform_points_inside
            points4 = uniform_points_outside
            points = np.concatenate([points1, points2, points3, points4], axis=0)
            # points = points4

            color1 = surface_points_inside_semantic_list
            color2 = surface_points_outside_semantic_list
            color3 = uniform_points_inside_semantic_list
            color4 = uniform_points_outside_semantic_list
            color = np.concatenate([color1, color2, color3, color4], axis=0)
            # color = color4

            to_save = np.concatenate([points, color * 255], axis=-1)  # (N, 6)  ply color must is [0-255] not as obj is [0-1]
            np.savetxt(fname, to_save, fmt='%.6f %.6f %.6f %d %d %d', comments='', header=(
                'ply\nformat ascii 1.0\nelement vertex {:d}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header').format(
                points.shape[0])
                       )

# import matplotlib.pyplot as plt
# import numpy as np
#
# X=np.linspace(0,uniform_points_inside.shape[0],uniform_points_inside.shape[0])
# Y =uniform_points_inside
# # plt.plot(X,Y,lable="$sin(X)$",color="red",linewidth=2)
# plt.figure(figsize=(8,6))
# plt.xlabel("x(s)")
# plt.ylabel("dist")
# plt.title("Example")
# plt.plot(X,Y)
# plt.show()
#
# im
# temp1=torch.tensor(1.) -torch.tanh(torch.from_numpy(dist_list))
