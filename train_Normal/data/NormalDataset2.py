from torch.utils.data import Dataset
import os
import sys
import os.path as osp
import numpy as np
from PIL import Image
import torch
import pdb
import cv2
import torchvision.transforms as transforms
import glob
from Constants import consts
from train_Normal.lib.model.options import BaseOptions


class NormalDataset():
    # @staticmethod
    # def modify_commandline_options(parser, is_train):
    #     return parser

    def __init__(self, opt, phase='train', allow_aug=True):

        self.opt = opt
        self.projection_mode = 'orthogonal'

        # Path setup
        self.is_train = (phase == 'train')
        self.load_size = self.opt.loadSize
        self.allow_aug = allow_aug

        self.aug_trans = transforms.Compose([
            transforms.ColorJitter(brightness=opt.aug_bri, contrast=opt.aug_con, saturation=opt.aug_sat,
                                   hue=opt.aug_hue)
        ])
        # PIL to tensor
        self.to_tensor = transforms.Compose([
            transforms.Resize(self.load_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        self.training_inds, self.testing_inds = self.get_training_test_indices(args=self.opt, shuffle=False)

        self.epochIdx = 0

    def get_training_test_indices(self, args, shuffle=False):
        # sanity check for args.totalNumFrame
        self.tolal = []
        totalPeople = glob.glob(
            args.datasetDir + '/*')  # /media/amax/4C76448F76447C28/Dataset/Render/people1
        for people in totalPeople:
            self.tolal += glob.glob(
                people + '/*')  #  '/media/amax/4C76448F76447C28/Dataset/Render/people1/bareteeth.000006.obj'
        max_idx = len(self.tolal)  # total data number: all of obj
        indices = np.asarray(range(max_idx))  # [1,2,3,,,,55]

        # 这里更改下traning_data_ration
        testing_flag = (
                indices >= args.trainingDataRatio * max_idx)  #
        testing_inds = indices[
            testing_flag]  # testing indices: array of [44 45 46 47 48 49 50 51 52 53 54]
        testing_inds = testing_inds.tolist()

        training_inds = indices[
            np.logical_not(testing_flag)]  # training indices: array of [0, ..., 44]
        training_inds = training_inds.tolist()

        return training_inds, testing_inds

    def len(self):
        return len(self.training_inds)*180 if self.is_train else len(self.testing_inds)*180
        # total have 10000obj, each obj have 180 photo for trainning
      # totoal have 1800000 photo ; and then we will use gongshi to get each diffrent photo ,so we could get dataitem_of_each_ photo;


    def get_item(self, index):  # get this index+1 photo and return a dataitem ; total get 1000000 item '
        render={}
        if not self.is_train: index += len(self.training_inds)*180

        volume_id = index // 180
        view_id    = ((index %180-90)+360)%360

        print("index = %d 时代表取第%d 张照片，这个照片在第%d 个obj中，编号是%d"%(index, index+1, volume_id,view_id))

        # ------load mask-------
        mask_path = "%s/mask_jpg/%04d_mask.jpg" % (self.tolal[volume_id], view_id)
        if not os.path.exists(mask_path):
            print("Can not find %s!!!" % (mask_path))
            pdb.set_trace()

        mask_data = np.round((cv2.imread(mask_path)[:,:,0]).astype(np.float32)/255.) # (1536, 1024)
        mask_data_padded = np.zeros((max(mask_data.shape), max(mask_data.shape)), np.float32) # (1536, 1536)
        mask_data_padded[:,mask_data_padded.shape[0]//2-min(mask_data.shape)//2:mask_data_padded.shape[0]//2+min(mask_data.shape)//2] = mask_data
        mask_data_padded = cv2.resize(mask_data_padded, (self.opt.loadSize,self.opt.loadSize), interpolation=cv2.INTER_NEAREST)
        mask_data_padded = Image.fromarray(mask_data_padded)
        mask_data_padded = transforms.ToTensor()(mask_data_padded).float()
        render['mask'] = mask_data_padded

        # ------- load image -------
        if True:
            image_path = '%s/jpg/%04d.jpg' % (self.tolal[volume_id], view_id)
            print(image_path)
            if not os.path.exists(image_path):
                print("Can not find %s!!!" % (image_path))
                pdb.set_trace()

            image = cv2.imread(image_path)[:,:,::-1] # (1536, 1024, 3), np.uint8, {0,...,255}
            image_padded = np.zeros((max(image.shape), max(image.shape), 3), np.uint8) # (1536, 1536, 3)
            image_padded[:,image_padded.shape[0]//2-min(image.shape[:2])//2:image_padded.shape[0]//2+min(image.shape[:2])//2,:] = image # (1536, 1536, 3)
            # resize to (512, 512, 3), np.uint8
            image_padded = cv2.resize(image_padded, (self.opt.loadSize, self.opt.loadSize))
            image_padded = Image.fromarray(image_padded)
            image_padded=self.to_tensor(image_padded)
            image_padded = mask_data_padded.expand_as(image_padded) * image_padded
            render['img']=image_padded

        # ------ load normal_image --------
        if True:
            # set paths
            normal_path = '%s/normal_jpg/%04d_normal.jpg' % (self.tolal[volume_id], view_id)
            if not os.path.exists(normal_path):
                print("Can not find %s!!!" % (normal_path))
                pdb.set_trace()

            # read data BGR -> RGB, np.uint8
            normal = cv2.imread(normal_path)[:,:,::-1] # (1536, 1024, 3), np.uint8, {0,...,255}
            normal_padded = np.zeros((max(normal.shape), max(normal.shape), 3), np.uint8) # (1536, 1536, 3)
            normal_padded[:,normal_padded.shape[0]//2-min(normal.shape[:2])//2:normal_padded.shape[0]//2+min(normal.shape[:2])//2,:] = normal # (1536, 1536, 3)
            # resize to (512, 512, 3), np.uint8
            normal_padded = cv2.resize(normal_padded, (self.opt.loadSize, self.opt.loadSize))
            normal_padded = Image.fromarray(normal_padded)
            normal_padded= self.to_tensor(normal_padded)
            normal_padded = mask_data_padded.expand_as(normal_padded) * normal_padded
            render['normal_F']=normal_padded

        img = np.uint8((np.transpose(render['img'].numpy(), (1, 2, 0)) * 0.5 + 0.5)[:, :,
                       ::-1] * 255.0)  # HWC, BGR, float 0 ~ 255, de-normalized by mean 0.5 and std 0.5
        cv2.imwrite(r"/home/amax/Python_code/train_Normal/sample_result/%06d_img.png" % (index), img)

        img = np.uint8((np.transpose(render['normal_F'].numpy(), (1, 2, 0)) * 0.5 + 0.5)[:, :,
                       ::-1] * 255.0)  # HWC, BGR, float 0 ~ 255, de-normalized by mean 0.5 and std 0.5
        cv2.imwrite(r"/home/amax/Python_code/train_Normal/sample_result/%06d_normal_img.png" % (index), img)



        return render

    def getitem(self, index):
        return self.get_item(index)

if __name__ == '__main__':
    opt = BaseOptions().parse()
    dataset1 = NormalDataset(opt, phase='train')
    # for i in np.arange(0, 360):
    #     dataset1.get_item(i)
    dataset1.get_item(800000)


