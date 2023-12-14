import sys
import os
import scipy.io as sio
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from lib.train_util import *
import glob
import copy
import pdb # pdb.set_trace()
this_file_path_abs       = os.path.dirname(__file__)
target_dir_path_relative = os.path.join(this_file_path_abs, '../..')
target_dir_path_abs      = os.path.abspath(target_dir_path_relative)
sys.path.insert(0, target_dir_path_abs)
from Constants import consts
target_dir_path_relative = os.path.join(this_file_path_abs, '../../DataUtil')
target_dir_path_abs      = os.path.abspath(target_dir_path_relative)
sys.path.insert(0, target_dir_path_abs)

import trimesh
import logging
log = logging.getLogger('trimesh')
log.setLevel(40)

# global consts
B_MIN           = np.array([-0.5, -0.5, -0.5])
B_MAX           = np.array([ 0.5,  0.5,  0.5])
SENSEI_DATA_DIR = "/trainman-mount/trainman-storage-d5c0a121-bb5d-4afb-8020-c53f096d2a5c"

def get_training_test_indices(args, shuffle):

    # sanity check for args.totalNumFrame
    assert(os.path.exists(args.datasetDir))   # /mnt/GeoRender
    totalNumFrameTrue = len(glob.glob(args.datasetDir+"/config/*.json"))
    assert((args.totalNumFrame == totalNumFrameTrue) or (args.totalNumFrame == totalNumFrameTrue+len(consts.black_list_images)//4))

    max_idx = args.totalNumFrame # total data number: N*M'*4 = 6795*4*4 = 108720
    indices = np.asarray(range(max_idx))
    assert(len(indices)%4 == 0)

    testing_flag = (indices >= args.trainingDataRatio*max_idx)
    testing_inds = indices[testing_flag] # 21744 testing indices: array of [86976, ..., 108719]
    testing_inds = testing_inds.tolist()
    if shuffle: np.random.shuffle(testing_inds)
    assert(len(testing_inds) % 4 == 0)

    training_inds = indices[np.logical_not(testing_flag)] # 86976 training indices: array of [0, ..., 86975]
    training_inds = training_inds.tolist()
    if shuffle: np.random.shuffle(training_inds)
    assert(len(training_inds) % 4 == 0)

    return training_inds, testing_inds

def compute_split_range(testing_inds, args):
    """
    determine split range, for multi-process running
    """

    dataNum = len(testing_inds)
    splitLen = int(np.ceil(1.*dataNum/args.splitNum))
    splitRange = [args.splitIdx*splitLen, min((args.splitIdx+1)*splitLen, dataNum)]

    testIdxList = []
    for eachTestIdx in testing_inds[splitRange[0]:splitRange[1]]:
        if ("%06d"%(eachTestIdx)) in consts.black_list_images: continue
        print("checking %06d-%06d-%06d..." % (testing_inds[splitRange[0]], eachTestIdx, testing_inds[splitRange[1]-1]+1))

        # check existance
        configPath = "%s/config/%06d.json" % (args.datasetDir, eachTestIdx)
        assert(os.path.exists(configPath)) # config file
        
        # save idx
        testIdxList.append([configPath])

    return testIdxList

def inverseRotateY(points,angle):
    """
    Rotate the points by a specified angle., LEFT hand rotation
    """

    angle = np.radians(angle)
    ry = np.array([ [ np.cos(angle), 0., np.sin(angle)],
                    [            0., 1.,            0.],
                    [-np.sin(angle), 0., np.cos(angle)] ]) # (3,3)
    return np.dot(points, ry) # (N,3)

def voxelization_normalization(verts,useMean=True,useScaling=True):
    """
    normalize the mesh into H [-0.5,0.5]*(1-margin), W/D [-0.333,0.333]*(1-margin)
    """

    vertsVoxelNorm = copy.deepcopy(verts)
    vertsMean, scaleMin = None, None

    if useMean:
        vertsMean = np.mean(vertsVoxelNorm,axis=0,keepdims=True) # (1, 3)
        vertsVoxelNorm -= vertsMean

    xyzMin = np.min(vertsVoxelNorm, axis=0); assert(np.all(xyzMin < 0))
    xyzMax = np.max(vertsVoxelNorm, axis=0); assert(np.all(xyzMax > 0))

    if useScaling:
        scaleArr = np.array([consts.threshWD/abs(xyzMin[0]), consts.threshH/abs(xyzMin[1]),consts.threshWD/abs(xyzMin[2]), consts.threshWD/xyzMax[0], consts.threshH/xyzMax[1], consts.threshWD/xyzMax[2]])
        scaleMin = np.min(scaleArr)
        vertsVoxelNorm *= scaleMin

    return vertsVoxelNorm, vertsMean, scaleMin

def save_samples_truncted_prob(fname, points, prob):
    '''
    Save the visualization of sampling to a ply file.
    Red points represent positive predictions.
    Green points represent negative predictions.
    :param fname: File name to save
    :param points: [N, 3] array of points
    :param prob: [N, 1] array of predictions in the range [0~1]
    :return:
    '''
    r = (prob > 0.5).reshape([-1, 1]) * 255 # (5000, 1) Red-inside
    g = (prob < 0.5).reshape([-1, 1]) * 255 # (5000, 1) green-outside
    b = np.zeros(r.shape) # (5000, 1)

    to_save = np.concatenate([points, r, g, b], axis=-1) # (5000, 6)
    return np.savetxt(fname,
                      to_save,
                      fmt='%.6f %.6f %.6f %d %d %d',
                      comments='',
                      header=(
                          'ply\nformat ascii 1.0\nelement vertex {:d}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header').format(
                          points.shape[0])
                      )

# this is the current sample ;
def process_one_data_item(data_item, output_data_dir):  # /media/amax/4C76448F76447C28/FullData/facescape_fill/1/2_smile.obj
    num_sample_surface = 400000
    num_sample_uniform = 25000
    sigma = 0.025
    sigma_small = 0.01
    curv_thresh = 0.004


    # output_data_dir='/media/amax/4C76448F76447C28/FullData/OCC/1'
    _, item_name = os.path.split(data_item)  # bareteeth.000001.obj
    item_name,_ = os.path.splitext(item_name)  # bareteeth.000001
    item_name = item_name+'.mat'  # # bareteeth.000001.mat
    # print(item_name)
    output_fd = os.path.join(output_data_dir, item_name)  # /media/amax/4C76448F76447C28/Dataset/OCC/people1/bareteeth.000001.mat
    # os.makedirs(output_fd, exist_ok=True)
    # os.makedirs(os.path.join(output_fd, 'sample'), exist_ok=True)

    mesh = trimesh.load(data_item)
    mesh_bbox_min = np.min(mesh.vertices, axis=0, keepdims=True)
    mesh_bbox_max = np.max(mesh.vertices, axis=0, keepdims=True)
    mesh_bbox_size = mesh_bbox_max - mesh_bbox_min

    surface_points, _ = trimesh.sample.sample_surface(mesh, num_sample_surface)
    curvs = trimesh.curvature.discrete_gaussian_curvature_measure(mesh, surface_points, 0.004)

    curvs = abs(curvs)
    curvs = curvs / max(curvs)  # normalize curvature
    sigmas = np.zeros(curvs.shape)
    sigmas[curvs <= curv_thresh] = sigma
    sigmas[curvs > curv_thresh] = sigma_small
    random_shifts = np.random.randn(surface_points.shape[0], surface_points.shape[1])
    random_shifts *= np.expand_dims(sigmas, axis=-1)
    surface_points = surface_points + random_shifts
    inside = mesh.contains(surface_points)

    surface_points_inside = surface_points[inside]
    surface_points_outside = surface_points[np.logical_not(inside)]

    uniform_points1 = np.random.rand(num_sample_uniform * 2, 3) * mesh_bbox_size + mesh_bbox_min
    uniform_points2 = np.random.rand(num_sample_uniform, 3) * 1.0 - 0.5
    inside1 = mesh.contains(uniform_points1)
    inside2 = mesh.contains(uniform_points2)
    uniform_points_inside = uniform_points1[inside1]
    uniform_points_outside = uniform_points2[np.logical_not(inside2)]
    if len(uniform_points_inside) > num_sample_uniform // 2:
        uniform_points_inside = uniform_points_inside[:(num_sample_uniform // 2)]
        uniform_points_outside = uniform_points_outside[:(num_sample_uniform // 2)]
    else:
        uniform_points_outside = uniform_points_outside[:(num_sample_uniform - len(uniform_points_inside))]

    sio.savemat(output_fd,
                {
                    'surface_points_inside': surface_points_inside,
                    'surface_points_outside': surface_points_outside,
                    'uniform_points_inside': uniform_points_inside,
                    'uniform_points_outside': uniform_points_outside
                }, do_compression=True)
    # sio.savemat(os.path.join(output_fd, 'sample', 'meta.mat'),
    #             {
    #                 'sigma': sigma,
    #                 'sigma_small': sigma_small,
    #                 'curv_thresh': curv_thresh,
# })
# def main(args):
#     """
#     For each frame will save the following items for example, about 0.115 MB
#
#         occu_sigma3.5_pts5k/088046_ep000_inPts.npy            0.0573 MB, np.float64, (2500, 3)
#         occu_sigma3.5_pts5k/088046_ep000_outPts.npy           0.0573 MB, np.float64, (2500, 3)
#
#     In total, (86976 frames) * (15 epochs) * (0.115 MB) / 1024. = 146.5 G
#     """
#
#     # init.
#     visualCheck_0 = False
#
#     # create dirs for saving query pts
#     # saveQueryDir = "%s/%s" % (args.datasetDir, args.sampleType)
#     saveQueryDir = r"/home/amax/python_code/geopifu_new/test1/%s_split%02d_%02d" % (args.sampleType, args.splitNum, args.splitIdx)   # 采样方法为sigma3.5_pts5k; splitNum=8; splitIdx=0;
#     # print(saveQueryDir)  # ./sigma3.5_pts5k_split08_00
#     if not os.path.exists(saveQueryDir):
#         os.makedirs(saveQueryDir)
#
#     # start query pts sampling
#
#     count = 0
#     previousMeshPath, mesh, meshVN, meshFN, meshV, randomRot = None, None, None, None, None, np.zeros((3,3))
#     timeStart = time.time()
#     t0 = time.time()
#
#
#     meshPathTmp = r'/home/amax/python_code/geopifu_new/fill_hole/here._hole.obj'
#     mesh = trimesh.load(meshPathTmp)
#     meshVN = copy.deepcopy(mesh.vertex_normals)
#     meshFN = copy.deepcopy(mesh.face_normals)
#     meshV = copy.deepcopy(mesh.vertices)  # np.float64
#
#
#     min_xyz = np.min(meshV, axis=0, keepdims=True)
#     max_xyz = np.max(meshV, axis=0, keepdims=True)
#     vertices = meshV - (min_xyz + max_xyz) * 0.5
#     scale_inv = np.max(max_xyz - min_xyz)
#     scale = 1.0 / scale_inv * (0.75 + np.random.rand() * 0.15)
#     meshV *= scale
#
#
#     mesh.vertices = meshV
#     # print(mesh.face_normals)
#
#     # normalize into volumes of X~[+-0.333], Y~[+-0.5], Z~[+-0.333]
#
#
#     '''
#     randomRot = np.array(dataConfig["randomRot"], np.float32)  # by random R
#     mesh.vertex_normals = np.dot(mesh.vertex_normals, np.transpose(randomRot))
#     mesh.face_normals = np.dot(mesh.face_normals, np.transpose(randomRot))
#     mesh.vertices, _, _ = voxelization_normalization(np.dot(mesh.vertices, np.transpose(randomRot)))
#     '''
#
#
#     t_sample_pts, t_save_pts, t_move_files = 0., 0., 0.
#     for epochId in range(args.epoch_range[0], args.epoch_range[1]):   # 0，  15
#
#         # uniformly sample points on mesh surface
#         surface_points, _ = trimesh.sample.sample_surface(mesh, 4 * args.num_sample_inout)  # (N1,3)
#         # add gausian noise to surface points
#         sample_points = surface_points + np.random.normal(scale=1. * args.sigma / consts.dim_h,
#                                                           size=surface_points.shape)  # (N1, 3)
#
#         # uniformly sample inside the 128x192x128 volume, surface-points : volume-points ~ 16:1
#         print(B_MAX)
#         length = B_MAX - B_MIN
#         # print(length)
#         random_points = np.random.rand(args.num_sample_inout // 4, 3) * length + B_MIN  # (N2, 3)
#         sample_points = np.concatenate([sample_points, random_points], 0)  # (N1+N2, 3)
#         np.random.shuffle(sample_points)  # (N1+N2, 3)
#         # print(sample_points)  # 23250 ， 3
#
#         # determine {1, 0} occupancy ground-truth
#         inside = mesh.contains(sample_points)
#         inside_points = sample_points[inside]
#         outside_points = sample_points[np.logical_not(inside)]
#
#         # constrain (n_in + n_out) <= self.num_sample_inout
#         nin = inside_points.shape[0]
#         # inside_points  =  inside_points[:self.num_sample_inout//2] if nin > self.num_sample_inout//2 else inside_points
#         # outside_points = outside_points[:self.num_sample_inout//2] if nin > self.num_sample_inout//2 else outside_points[:(self.num_sample_inout - nin)]
#         if nin > args.num_sample_inout // 2:
#             inside_points = inside_points[:args.num_sample_inout // 2]
#             outside_points = outside_points[:args.num_sample_inout // 2]
#         else:
#             inside_points = inside_points
#             if outside_points.shape[0] < (args.num_sample_inout - nin):
#                 print("Error: outside_points.shape[0] {} < (args.num_sample_inout - nin) {}!".format(
#                     outside_points.shape[0], (args.num_sample_inout - nin)))
#                 pdb.set_trace()
#             outside_points = outside_points[:(args.num_sample_inout - nin)]
#
#         # save query pts
#         inside_path = "%s/%06d_ep_inPts.npy" % (saveQueryDir, epochId)
#         outside_path = "%s/%06d_ep_outPts.npy" % (saveQueryDir, epochId)
#         # inside_path  = "./sample_images/%06d_ep%03d_inPts.npy"  % (frameIdx[1], epochId)
#         # outside_path = "./sample_images/%06d_ep%03d_outPts.npy" % (frameIdx[1], epochId)
#         np.save(inside_path, inside_points)
#         np.save(outside_path, outside_points)
#         print("ok")
#
#     # yici keshihua yige
#     if visualCheck_0:
#
#         print("visualCheck_0: see if query samples are inside the volume...")
#
#
#         samples = np.concatenate([inside_points], 0).T # (3, n_in)
#         labels  = np.concatenate([np.ones((1, inside_points.shape[0]))], 1) # (1, n_in)
#         save_samples_truncted_prob("/home/amax/python_code/geopifu_new/sample_images/samples_inside.ply", samples.T, labels.T)
#
#
#         samples = np.concatenate([outside_points], 0).T # (3, n_in)
#         labels  = np.concatenate([np.zeros((1, outside_points.shape[0]))], 1) # (1, n_in)
#         save_samples_truncted_prob("/home/amax/python_code/geopifu_new/sample_images/samples_outside.ply", samples.T, labels.T)
#
#
#         gtMesh = {"v": mesh.vertices, "vn": mesh.vertex_normals, "vc": mesh.visual.vertex_colors, "f": mesh.faces, "fn": mesh.face_normals}
#         ObjIO.save_obj_data_color(gtMesh, "/home/amax/python_code/geopifu_new/sample_images/meshGT.obj")
#
#         # pdb.set_trace()
#
#
#     # un-normalize the mesh
#     '''
#     mesh.vertex_normals = meshVN
#     mesh.face_normals   = meshFN
#     mesh.vertices       = meshV
#     '''


if __name__ == '__main__':
    # calculate OCC of obj model
    root = '/media/amax/4C76448F76447C28/FullData_2/facescape_fill'
    root_occ = '/media/amax/4C76448F76447C28/FullData_2/OCC'
    for file in os.listdir(root):
        Path_fillhole_obj = os.path.join(root, file)  # '/media/amax/4C76448F76447C28/FullData/facescape_fill/1'
        output_data_dir = os.path.join(root_occ, file)  # '/media/amax/4C76448F76447C28/FullData/OCC/1'   can baozheng not change !

        if not os.path.exists(output_data_dir):
            os.makedirs(output_data_dir, exist_ok=True)
        print(output_data_dir)
        print(Path_fillhole_obj)
        for filename in os.listdir(Path_fillhole_obj):
            path_save = os.path.join(Path_fillhole_obj, filename)  # /media/amax/4C76448F76447C28/FullData/facescape_fill/1/2_smile.obj
            process_one_data_item(path_save, output_data_dir)

        print('file = ', file)












