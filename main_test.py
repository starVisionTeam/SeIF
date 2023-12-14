import os

import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch
from lib.options import BaseOptions
from lib.options2 import BaseOptions as BaseOptions2
from lib.model import *
from lib.train_util import *
from Constants import consts
from lib.model2 import HGPIFuNet as SemanticNet
import torchvision.transforms as transforms


opt = BaseOptions().parse()
opt2 = BaseOptions2().parse()

def rotateY_by_view(view_id):
    """
    input
        view_id: 0-front, 1-right, 2-back, 3-left
    """

    angle = np.radians(-view_id)
    ry = np.array([[np.cos(angle), 0., np.sin(angle)],
                   [0., 1., 0.],
                   [-np.sin(angle), 0., np.cos(angle)]])  # (3,3)
    ry = np.transpose(ry)

    return ry
to_tensor = transforms.Compose([
            transforms.Resize(opt.loadSize),  # 512 , 512
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])



def main(opt):
    image_path = 'demo/1.png'  # input image
    image_mask_path = 'demo/1_mask.png'  # The mask corresponding to the picture
    normal_path = 'demo/1_normal.png'  # image normal
    save_file = 'demo/1.obj'  # final result path

    # load data path
    cuda = torch.device('cuda') if len(opt.gpu_ids) > 1 else torch.device('cuda:%d' % opt.gpu_id)
    projection_mode = 'orthogonal'

    with torch.no_grad():
        # load model only once time is ok!
        netG_SeIF = HGPIFuNet(opt, projection_mode)
        netG_SeIF.to(cuda)
        netG_Semantic = SemanticNet(opt2, projection_mode)
        netG_Semantic.to(cuda)

        # can exchange net with each other!
        model_semantic_path = r'checkpoints/model_Semantic/netG_epoch_6_293299'
        netG_Semantic.load_state_dict(torch.load(model_semantic_path, map_location=cuda))
        netG_Semantic.eval()

        model_SeIF_path = 'checkpoints/model_SeIF/netG_epoch_3_293566'
        netG_SeIF.load_state_dict(torch.load(model_SeIF_path, map_location=cuda))
        netG_SeIF.eval()

        # produce a mesh without color
        # gen_mesh_iccv(opt, netG.module if len(opt.gpu_ids) > 1 else netG, cuda, data, save_path)
        # acoording to the test we can find that the muti_demo is the same as this single demo ;0
        # paths = ['209']
        # for path in paths:
        if True:
            projection_mode = 'orthogonal'
            cuda = torch.device('cuda') if len(opt.gpu_ids) > 1 else torch.device('cuda:%d' % opt.gpu_id)
            # print(cuda)
            if os.path.exists(image_path):
                print(normal_path)
                if not os.path.exists(save_file):
                    os.makedirs(os.path.dirname(save_file), exist_ok=True)
                save_path = save_file

                # compute calib and img of data
                render_list = []
                normal_list = []
                calib_list = []
                extrinsic_list = []
                mask_list = []

                # load mask
                mask_data = np.round((cv2.imread(image_mask_path)[:, :, 0]).astype(np.float32) / 255.)  # (1536, 1024)
                mask_data_padded = np.zeros((max(mask_data.shape), max(mask_data.shape)), np.float32)  # (1536, 1536)
                mask_data_padded[:,
                mask_data_padded.shape[0] // 2 - min(mask_data.shape) // 2:mask_data_padded.shape[0] // 2 + min(
                    mask_data.shape) // 2] = mask_data  # (1536, 1536)

                # NN resize to (512, 512)
                mask_data_padded = cv2.resize(mask_data_padded, (opt.loadSize, opt.loadSize),
                                              interpolation=cv2.INTER_NEAREST)
                mask_data_padded = Image.fromarray(mask_data_padded)

                # load image

                image = cv2.imread(image_path)[:, :, ::-1]  # (1536, 1024, 3), np.uint8, {0,...,255}
                normal_image = cv2.imread(normal_path)[:, :, ::-1]

                image_padded = np.zeros((max(image.shape), max(image.shape), 3), np.uint8)  # (1536, 1536, 3)
                normal_image_padded = np.zeros((max(normal_image.shape), max(normal_image.shape), 3), np.uint8)
                image_padded[:,
                image_padded.shape[0] // 2 - min(image.shape[:2]) // 2:image_padded.shape[0] // 2 + min(
                    image.shape[:2]) // 2,
                :] = image  # (1536, 1536, 3)
                normal_image_padded[:,
                normal_image_padded.shape[0] // 2 - min(normal_image.shape[:2]) // 2:normal_image_padded.shape[
                                                                                         0] // 2 + min(
                    normal_image.shape[:2]) // 2, :] = normal_image  # (1536, 1536, 3)

                # resize to (512, 512, 3), np.uint8
                image_padded = cv2.resize(image_padded, (opt.loadSize, opt.loadSize))
                normal_image_padded = cv2.resize(normal_image_padded, (opt.loadSize, opt.loadSize))
                image_padded = Image.fromarray(image_padded)
                normal_image_padded = Image.fromarray(normal_image_padded)

                # load calib and intrinsic

                trans_intrinsic = np.identity(4)  # trans intrinsic
                scale_intrinsic = np.identity(4)  # ortho. proj. focal length
                scale_intrinsic[0, 0] = 1. / consts.h_normalize_half  # const ==  2.
                scale_intrinsic[1, 1] = -1. / consts.h_normalize_half  # const ==  2.
                scale_intrinsic[2, 2] = -1. / consts.h_normalize_half  # const == -2.
                # extrinsic: model to cam R|t
                extrinsic = np.identity(4)
                # randomRot        = np.array(dataConfig["randomRot"], np.float32) # by random R
                viewRot = rotateY_by_view(view_id=0)  # by view direction R
                # extrinsic[:3,:3] = np.dot(viewRot, randomRot)
                extrinsic[:3, :3] = viewRot.T

                mask_data_padded = transforms.ToTensor()(mask_data_padded).float()  # 1. inside, 0. outside
                mask_list.append(mask_data_padded)

                image_padded = to_tensor(image_padded)  # (3, 512, 512), float -1 ~ 1
                normal_image_padded = to_tensor(normal_image_padded)  # (3, 512, 512), float -1 ~ 1
                image_padded = mask_data_padded.expand_as(image_padded) * image_padded
                normal_image_padded = mask_data_padded.expand_as(normal_image_padded) * normal_image_padded
                render_list.append(image_padded)
                normal_list.append(normal_image_padded)

                intrinsic = np.matmul(trans_intrinsic, scale_intrinsic)
                calib = torch.Tensor(np.matmul(intrinsic, extrinsic)).float()

                # write by myself ; change calib（旋转矩阵） to ensure project!!!
                # rot = calib[:3, :3]
                # rot[0, 2] *= -1
                # rot[1, 1] *= -1
                # calib[:3, :3] = rot

                extrinsic = torch.Tensor(extrinsic).float()
                calib_list.append(calib)  # save calib
                extrinsic_list.append(extrinsic)  # save extrinsic
                #####

                data = {'img': torch.stack(render_list, dim=0),
                        'normal': torch.stack(normal_list, dim=0),
                        'calib': torch.stack(calib_list, dim=0),
                        # model will be transformed into a XY-plane-center-aligned-2x2x2-volume of the cam. coord.
                        'extrinsic': torch.stack(extrinsic_list, dim=0),
                        'mask': torch.stack(mask_list, dim=0)
                        }
                # every time we do a new encoder photo! and we get a net image feature!
                gen_mesh(opt, netG_SeIF.module if len(opt.gpu_ids) > 1 else netG_SeIF, netG_Semantic, cuda, data, save_path)

if __name__ == '__main__':
    main(opt)
