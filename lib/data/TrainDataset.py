from torch.utils.data import Dataset
import numpy as np
import os
import random
import torchvision.transforms as transforms
from PIL import Image, ImageOps
import cv2
import torch
from PIL.ImageFilter import GaussianBlur
import trimesh
import logging

import pdb # pdb.set_trace()
import glob
import sys
import json
import copy
import scipy.io as sci
import scipy.io as sio
SENSEI_DATA_DIR          = "/trainman-mount/trainman-storage-d5c0a121-bb5d-4afb-8020-c53f096d2a5c"
this_file_path_abs       = os.path.dirname(__file__)
target_dir_path_relative = os.path.join(this_file_path_abs, '../../..')
target_dir_path_abs      = os.path.abspath(target_dir_path_relative)
sys.path.insert(0, target_dir_path_abs)
from Constants import consts

log = logging.getLogger('trimesh')
log.setLevel(40)

def load_trimesh(root_dir):
    """
    XYZ direction
        X-right, Y-up, Z-outwards. All meshes face outwards along +Z.
    return
        a dict of ALL meshes, indexed by mesh-names
    """
    folders = os.listdir(root_dir)
    meshs = {}
    for i, f in enumerate(folders):
        sub_name = f
        meshs[sub_name] = trimesh.load(os.path.join(root_dir, f, '%s_100k.obj' % sub_name))

    return meshs


def load_trimesh_iccv(meshPathsList):
    """
    XYZ direction
        X-right, Y-down, Z-inwards. All meshes face inwards along +Z.
    return
        a dict of ALL meshes, indexed by mesh-names
    """

    meshs = {}
    count = 0
    for meshPath in meshPathsList:
        print("Loading mesh %d/%d..." % (count, len(meshPathsList)))
        count += 1
        meshs[meshPath] = trimesh.load(meshPath)

    print("Sizes of meshs dict: %.3f MB." % (sys.getsizeof(meshs)/1024.)); pdb.set_trace()

    return meshs


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

class TrainDatasetICCV(Dataset):

    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def __init__(self, opt, phase='train', allow_aug=True):

        self.opt = opt
        self.projection_mode = 'orthogonal'

        # Path setup
        # self.root = self.opt.dataroot
        # self.meshDirSearch = opt.meshDirSearch

        # this may need change
        # self.B_MIN = np.array([-consts.real_w/2., -consts.real_h/2., -consts.real_w/2.])
        # self.B_MAX = np.array([ consts.real_w/2.,  consts.real_h/2.,  consts.real_w/2.])
        self.B_MIN = np.array([-0.5, -0.5, -0.5])   #
        self.B_MAX = np.array([0.5, 0.5, 0.5])

        self.is_train = (phase == 'train')
        self.load_size = self.opt.loadSize  # photo size 512
        self.allow_aug = allow_aug

        self.num_views = self.opt.num_views  # 1

        self.num_sample_inout = self.opt.num_sample_inout  # 10000

        # PIL to tensor
        self.to_tensor = transforms.Compose([
            transforms.Resize(self.load_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        # PIL to tensor, for the target view if use view rendering losses
        if opt.use_view_pred_loss: 
            self.to_tensor_target_view = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

        # augmentation
        self.aug_trans = transforms.Compose([
            transforms.ColorJitter(brightness=opt.aug_bri, contrast=opt.aug_con, saturation=opt.aug_sat,
                                   hue=opt.aug_hue)
        ])


        self.training_inds, self.testing_inds = self.get_training_test_indices(args=self.opt, shuffle=False)

        # record epoch idx for offline query sampling
        self.epochIdx = 0


    def get_training_test_indices(self, args, shuffle=False):  # get index of datasetDir

        # sanity check for args.totalNumFrame
        self.tolal=[]
        totalPeople = glob.glob(args.datasetDir+'/*')  # /media/amax/4C76448F76447C28/Dataset/Render/people1
        for people in totalPeople:
            self.tolal+=glob.glob(people+'/*')  #'/media/amax/4C76448F76447C28/Dataset/Render/people1/bareteeth.000006.obj'
        max_idx = len(self.tolal) # total data number: all of obj
        indices = np.asarray(range(max_idx))  # [1,2,3,,,,55]


        testing_flag = (indices >= args.trainingDataRatio*max_idx)
        testing_inds = indices[testing_flag] # testing indices: array of [44 45 46 47 48 49 50 51 52 53 54]  ;
        testing_inds = testing_inds.tolist()

        training_inds = indices[np.logical_not(testing_flag)]  #  training indices: array of [0, ..., 44]
        training_inds = training_inds.tolist()

        return training_inds, testing_inds  # get train_obj.file number;

    def __len__(self):

        # return len(self.training_inds)*60 if self.is_train else len(self.testing_inds)*60
        return len(self.training_inds) * 100 if self.is_train else len(self.testing_inds) * 100

    def rotateY_by_view(self, view_id):
        """
        input
            view_id: 0-front, 1-right, 2-back, 3-left
        """

        angle = np.radians(-view_id)
        ry = np.array([ [ np.cos(angle), 0., np.sin(angle)],
                        [            0., 1.,            0.],
                        [-np.sin(angle), 0., np.cos(angle)]]) # (3,3)
        ry = np.transpose(ry)

        return ry

    def get_render_iccv(self, args, num_views,  random_sample, volume_id, view_id):
        '''
        Return the render data
        :param subject: subject name
        :param num_views: how many views to return
        :param view_id: the first view_id. If None, select a random one.
        :return:
            'img': [num_views, C, W, H] for 3x512x512 images
            'calib': [num_views, 4, 4] calibration matrix
            'extrinsic': [num_views, 4, 4] extrinsic matrix
            'mask': [num_views, 1, W, H] for 1x512x512 masks
        '''

        # init.
        render_list = []
        calib_list = []
        extrinsic_list = []
        mask_list = []
        normal_list = []
        # for each view
        # ----- load mask Part-0 -----
        if True:

            # set path
            mask_path = "%s/mask_jpg/%04d_mask.jpg" % (self.tolal[volume_id], view_id)
            if not os.path.exists(mask_path):
                print("Can not find %s!!!" % (mask_path))
                pdb.set_trace()

            # {read, discretize} data, values only within {0., 1.}
            mask_data = np.round((cv2.imread(mask_path)[:,:,0]).astype(np.float32)/255.) # (1536, 1024)
            mask_data_padded = np.zeros((max(mask_data.shape), max(mask_data.shape)), np.float32) # (1536, 1536)
            mask_data_padded[:,mask_data_padded.shape[0]//2-min(mask_data.shape)//2:mask_data_padded.shape[0]//2+min(mask_data.shape)//2] = mask_data # (1536, 1536)

            # NN resize to (512, 512)
            mask_data_padded = cv2.resize(mask_data_padded, (self.opt.loadSize,self.opt.loadSize), interpolation=cv2.INTER_NEAREST)
            mask_data_padded = Image.fromarray(mask_data_padded)

        # ----- load image Part-0 -----
        if True:

            # set paths
            image_path = '%s/jpg/%04d.jpg' % (self.tolal[volume_id],view_id)  # this photo is in where obj and where item
            normal_path = '%s/normal_jpg/%04d_normal.jpg' % (self.tolal[volume_id], view_id)
            if not os.path.exists(image_path):
                print("Can not find %s!!!" % (image_path))
                pdb.set_trace()
            if not os.path.exists(normal_path):
                print("Can not find %s!!!" % (normal_path))
                pdb.set_trace()

            # read data BGR -> RGB, np.uint8
            image = cv2.imread(image_path)[:,:,::-1] # (1536, 1024, 3), np.uint8, {0,...,255}
            normal_image = cv2.imread(normal_path)[:, :, ::-1]

            image_padded = np.zeros((max(image.shape), max(image.shape), 3), np.uint8) # (1536, 1536, 3)
            normal_image_padded = np.zeros((max(normal_image.shape), max(normal_image.shape), 3), np.uint8)  # (1536, 1536, 3)

            image_padded[:,image_padded.shape[0]//2-min(image.shape[:2])//2:image_padded.shape[0]//2+min(image.shape[:2])//2,:] = image # (1536, 1536, 3)
            normal_image_padded[:, normal_image_padded.shape[0] // 2 - min(normal_image.shape[:2]) // 2:normal_image_padded.shape[0] // 2 + min(
                normal_image.shape[:2]) // 2, :] = normal_image  # (1536, 1536, 3)

            # resize to (512, 512, 3), np.uint8
            image_padded = cv2.resize(image_padded, (self.opt.loadSize, self.opt.loadSize))
            normal_image_padded = cv2.resize(normal_image_padded, (self.opt.loadSize, self.opt.loadSize))

            image_padded = Image.fromarray(image_padded)
            normal_image_padded = Image.fromarray(normal_image_padded)

        # ----- load calib and extrinsic Part-0 -----
        if True:

            # intrinsic matrix: ortho. proj. cam. model
            trans_intrinsic = np.identity(4) # trans intrinsic
            scale_intrinsic = np.identity(4) # ortho. proj. focal length
            scale_intrinsic[0, 0] = 1. / consts.h_normalize_half  # const ==  2.
            scale_intrinsic[1, 1] = -1. / consts.h_normalize_half  # const ==  2.
            scale_intrinsic[2, 2] = -1. / consts.h_normalize_half  # const == -2.

            # extrinsic: model to cam R|t
            extrinsic        = np.identity(4)
            # randomRot        = np.array(dataConfig["randomRot"], np.float32) # by random R
            viewRot          = self.rotateY_by_view(view_id=view_id) # by view direction R
            # extrinsic[:3,:3] = np.dot(viewRot, randomRot)
            extrinsic[:3,:3] = viewRot.T

        # ----- training data augmentation -----
        if self.is_train and self.allow_aug:

            # Pad images
            pad_size         = int(0.1 * self.load_size)
            image_padded     = ImageOps.expand(image_padded, pad_size, fill=0)
            normal_image_padded = ImageOps.expand(normal_image_padded, pad_size, fill=0)

            mask_data_padded = ImageOps.expand(mask_data_padded, pad_size, fill=0)
            w, h   = image_padded.size
            w1 , h1 = normal_image_padded.size

            th, tw = self.load_size, self.load_size

            # random flip
            if self.opt.random_flip and np.random.rand() > 0.5:  # False
                scale_intrinsic[0, 0] *= -1
                image_padded     = transforms.RandomHorizontalFlip(p=1.0)(image_padded)
                normal_image_padded = transforms.RandomHorizontalFlip(p=1.0)(normal_image_padded)

                mask_data_padded = transforms.RandomHorizontalFlip(p=1.0)(mask_data_padded)

            # random scale
            if self.opt.random_scale:  # false
                rand_scale = random.uniform(0.9, 1.1)
                w = int(rand_scale * w)
                h = int(rand_scale * h)
                w1 = int(rand_scale * w1)
                h1 = int(rand_scale * h1)

                image_padded     = image_padded.resize((w, h), Image.BILINEAR)
                normal_image_padded = normal_image_padded.resize((w1, h1), Image.BILINEAR)

                mask_data_padded = mask_data_padded.resize((w, h), Image.NEAREST)
                scale_intrinsic *= rand_scale
                scale_intrinsic[3, 3] = 1

            # random translate in the pixel space
            if self.opt.random_trans:  # False
                dx = random.randint(-int(round((w - tw) / 10.)),
                                    int(round((w - tw) / 10.)))
                dx1 = random.randint(-int(round((w1 - tw) / 10.)),
                                    int(round((w1 - tw) / 10.)))

                dy = random.randint(-int(round((h - th) / 10.)),
                                    int(round((h - th) / 10.)))
                dy1 = random.randint(-int(round((h1 - th) / 10.)),
                                    int(round((h1 - th) / 10.)))
            else:
                dx = 0
                dy = 0
                dx1 = 0
                dy1 = 0

            trans_intrinsic[0, 3] = -dx / float(self.opt.loadSize // 2)
            trans_intrinsic[1, 3] = -dy / float(self.opt.loadSize // 2)

            x1 = int(round((w - tw) / 2.)) + dx
            y1 = int(round((h - th) / 2.)) + dy
            x1_normal = int(round((w1 - tw) / 2.)) + dx1
            y1_normal = int(round((h1 - th) / 2.)) + dy1

            image_padded = image_padded.crop((x1, y1, x1 + tw, y1 + th))
            normal_image_padded = normal_image_padded.crop((x1_normal, y1_normal, x1_normal + tw, y1_normal + th))

            mask_data_padded = mask_data_padded.crop((x1, y1, x1 + tw, y1 + th))

            # color space augmentation
            image_padded = self.aug_trans(image_padded)
            normal_image_padded = self.aug_trans(normal_image_padded)

            # random blur
            if self.opt.aug_blur > 0.00001:  # False
                blur = GaussianBlur(np.random.uniform(0, self.opt.aug_blur))
                image_padded = image_padded.filter(blur)
                normal_image_padded = normal_image_padded.filter(blur)

        # ----- load mask Part-1 -----
        if True:

            # convert to (1, 512, 512) tensors, float, 1-fg, 0-bg
            mask_data_padded = transforms.ToTensor()(mask_data_padded).float() # 1. inside, 0. outside
            mask_list.append(mask_data_padded)

        # ----- load image Part-1 -----
        if True:

            # convert to (3, 512, 512) tensors, RGB, float, -1 ~ 1. note that bg is 0 not -1.
            image_padded = self.to_tensor(image_padded) # (3, 512, 512), float -1 ~ 1
            image_padded = mask_data_padded.expand_as(image_padded) * image_padded
            render_list.append(image_padded)

        if True:
            normal_image_padded = self.to_tensor(normal_image_padded)
            normal_image_padded = mask_data_padded.expand_as(normal_image_padded) * normal_image_padded
            normal_list.append(normal_image_padded)



        # ----- load calib and extrinsic Part-1 -----
        if True:

            # obtain the final calib and save calib/extrinsic
            intrinsic = np.matmul(trans_intrinsic, scale_intrinsic)
            calib     = torch.Tensor(np.matmul(intrinsic, extrinsic)).float()

            # write by myself ; change calib to ensure project!!!

            # rot = calib[:3, :3]
            # rot[0, 2] *= -1
            # rot[1, 1] *= -1
            # calib[:3, :3] = rot

            extrinsic = torch.Tensor(extrinsic).float()
            calib_list.append(calib)         # save calib
            extrinsic_list.append(extrinsic) # save extrinsic

        return {'img'      : torch.stack(render_list   , dim=0),
                'normal'      : torch.stack(normal_list   , dim=0),
                'calib'    : torch.stack(calib_list    , dim=0), # model will be transformed into a XY-plane-center-aligned-2x2x2-volume of the cam. coord.
                'extrinsic': torch.stack(extrinsic_list, dim=0),
                'mask'     : torch.stack(mask_list     , dim=0)
               }

    def select_sampling_method_iccv_offline(self,volume_obj):  # /media/amax/4C76448F76447C28/Dataset/Render/people1/bareteeth.000004.obj

        # get inside and outside points
        mat_file=volume_obj.replace('Render','OCC').replace('.obj','.mat')  # /media/amax/4C76448F76447C28/Dataset/OCC/people1/bareteeth.000004.mat
        assert(os.path.exists(mat_file) )
        pts_data=sci.loadmat(mat_file,verify_compressed_data_integrity=False)

        pts_adp_idp = np.int32(np.random.rand(self.num_sample_inout // 2) * len(pts_data['surface_points_inside']))
        pts_adp_idn = np.int32(np.random.rand(self.num_sample_inout // 2) * len(pts_data['surface_points_outside']))
        pts_uni_idp = np.int32(np.random.rand(self.num_sample_inout // 32) * len(pts_data['uniform_points_inside']))
        pts_uni_idn = np.int32(np.random.rand(self.num_sample_inout // 32) * len(pts_data['uniform_points_outside']))

        pts_adp_p = pts_data['surface_points_inside'][pts_adp_idp]
        pts_adp_n = pts_data['surface_points_outside'][pts_adp_idn]
        pts_uni_p = pts_data['uniform_points_inside'][pts_uni_idp]
        pts_uni_n = pts_data['uniform_points_outside'][pts_uni_idn]

        ################
        # read occ semantic information , write by 2022/12/10
        mat_file_semantic=volume_obj.replace('Render','OCC_Semantic').replace('.obj','_semantic.mat')  #_semantic /media/amax/4C76448F76447C28/Dataset/OCC/people1/bareteeth.000004.mat
        pts_data_semantic = sci.loadmat(mat_file_semantic, verify_compressed_data_integrity=False)
        pts_adp_p_semantic = pts_data_semantic['surface_points_inside'][pts_adp_idp]
        pts_adp_n_semantic = pts_data_semantic['surface_points_outside'][pts_adp_idn]
        pts_uni_p_semantic = pts_data_semantic['uniform_points_inside'][pts_uni_idp]
        pts_uni_n_semantic = pts_data_semantic['uniform_points_outside'][pts_uni_idn]

        '''
          according to the visualize ply file and compare with read data ,
           we can se that every point have corresponding right Semantic information;
        '''
        ##############

        pts = np.concatenate([pts_adp_p, pts_adp_n, pts_uni_p, pts_uni_n], axis=0).T
        pts_semantic = np.concatenate([pts_adp_p_semantic, pts_adp_n_semantic, pts_uni_p_semantic, pts_uni_n_semantic], axis=0).T
        pts_ov = np.concatenate([
            np.ones([len(pts_adp_p), 1]), np.zeros([len(pts_adp_n), 1]),
            np.ones([len(pts_uni_p), 1]), np.zeros([len(pts_uni_n), 1]),
        ], axis=0).T

        pts = pts.astype(np.float32)
        pts_ov = pts_ov.astype(np.float32)
        pts_semantic = pts_semantic.astype(np.float32)

        samples = torch.Tensor(pts).float() # convert np.array to torch.Tensor
        labels  = torch.Tensor(pts_ov).float()
        semantic  = torch.Tensor(pts_semantic).float()

        return {'samples': samples, # (3, n_in + n_out),
                'labels' : labels,   # (1, n_in + n_out), 1.0-inside, 0.0-outside
                'semantic':semantic  # (3 , n_in + n_out) ; sample's semantic label!
               }


    def get_item(self, index):   # total read len photo, each photo is diffrent and from 100/obj

        visualCheck_0 = False

        if not self.is_train: index += len(self.training_inds*100)  #  so we shoule skip train dataset



        volume_id = index // 100

        # view_id    = ((index %60-30)+360)%360
        # view_id    = index % 360
        view_id = ((index % 100 - 50) + 360) % 360   # get [0,49]&&[310,359]!


        #11111 ----- load "name", "view_id" -----
        res = {"name"         : self.tolal[volume_id],
               "view_id"        : view_id,
              }


        #22222 ----- load "img", "calib", "extrinsic", "mask" -----
        """
        render_data
            'img'      : [num_views, C, H, W] RGB, 3x512x512 images, float -1. ~ 1., bg is all ZEROS not -1.
            'normal'   : the same as img ; 
            'calib'    : [num_views, 4, 4] calibration matrix
            'extrinsic': [num_views, 4, 4] extrinsic matrix
            'mask'     : [num_views, 1, H, W] for 1x512x512 masks, float 1.0-inside, 0.0-outside
        """

        render_data = self.get_render_iccv(args=self.opt, num_views=self.num_views, random_sample=self.opt.random_multiview,volume_id=volume_id, view_id=view_id)
        res.update(render_data)  # update dict ;


        #33333 ----- load "samples", "labels" if needed -----
        if self.opt.num_sample_inout:  # default: 10000

            """
            sample_data is a dict.
                "samples" : (3, n_in + n_out), float XYZ coords are inside the 3d-volume of [self.B_MIN, self.B_MAX]
                "labels"  : (1, n_in + n_out), float 1.0-inside, 0.0-outside
            """
            sample_data = self.select_sampling_method_iccv_offline(self.tolal[volume_id])
            res.update(sample_data)

            #  rbg_jpg == mask_jpg == sample.ply  ==normal_image
            if visualCheck_0:
                print("visualCheck_0: see if 'samples' can be properly projected to the 'img' by 'calib'...")

                # image RGB
                img = np.uint8((np.transpose(res['img'][0].numpy(), (1, 2, 0)) * 0.5 + 0.5)[:, :, ::-1] * 255.0) # HWC, BGR, float 0 ~ 255, de-normalized by mean 0.5 and std 0.5
                cv2.imwrite("./sample_images/%06d_img.png"%(index), img)

                # mask
                mask = np.uint8(res['mask'][0,0].numpy() * 255.0) # (512, 512), 255 inside, 0 outside
                cv2.imwrite("./sample_images/%06d_mask.png"%(index), mask)

                # orthographic projection
                rot   = res['calib'][0,:3, :3]  # R_norm(-1,1)Cam_model, assuming that ortho. proj.: cam_f==256, c_x==c_y==256, img.shape==(512,512,3)
                trans = res['calib'][0,:3, 3:4] # T_norm(-1,1)Cam_model
                pts = torch.addmm(trans, rot, res['samples'][:, res['labels'][0] > 0.5])  # (3, 2500)
                pts = 0.5 * (pts.numpy().T + 1.0) * 512 # (2500,3), ortho. proj.: cam_f==256, c_x==c_y==256, img.shape==(512,512,3)
                imgProj = cv2.UMat(img)
                for p in pts: cv2.circle(imgProj, (int(p[0]), int(p[1])), 2, (0,255,0), -1)
                cv2.imwrite("./sample_images/%06d_img_ptsProj.png"%(index), imgProj.get())     

                # save points in 3d
                samples_roted = torch.addmm(trans, rot, res['samples']) # (3, N)
                # samples_roted[2,:] *= -1
                save_samples_truncted_prob("./sample_images/%06d_samples.ply"%(index), samples_roted.T, res["labels"].T)


        # return a data point of dict. structure
        return res

    def __getitem__(self, index):  # index=(0-len)
        return self.get_item(index)

