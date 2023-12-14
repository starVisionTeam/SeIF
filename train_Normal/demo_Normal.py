import numpy as np
import cv2
import torch
from data.NormalNet import NormalNet
import torchvision.transforms as transforms
from PIL import Image
from train_Normal.lib.model.options import BaseOptions
opt = BaseOptions().parse()

to_tensor = transforms.Compose([
            transforms.Resize(opt.loadSize),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

cuda = torch.device('cuda:%d' % opt.gpu_id)

def main(opt):
    #  input img, mask; and we can get normal_image of input_image
    image_path = '/media/lx/4A42-E0B2/code_github/SePIFU_08_15into/demo_result/2.png'
    mask_path = '/media/lx/4A42-E0B2/code_github/SePIFU_08_15into/demo_result/2_mask.png'
    save_path = '/media/lx/4A42-E0B2/code_github/SePIFU_08_15into/demo_result/2_normal.png'


    render = {}
    render_list = []
    normal_list = []
    mask_data = np.round((cv2.imread(mask_path)[:, :, 0]).astype(np.float32) / 255.)  # (1536, 1024)
    mask_data_padded = np.zeros((max(mask_data.shape), max(mask_data.shape)), np.float32)  # (1536, 1536)
    mask_data_padded[:, mask_data_padded.shape[0] // 2 - min(mask_data.shape) // 2:mask_data_padded.shape[0] // 2 + min(
        mask_data.shape) // 2] = mask_data
    mask_data_padded = cv2.resize(mask_data_padded, (opt.loadSize, opt.loadSize), interpolation=cv2.INTER_NEAREST)
    mask_data_padded = Image.fromarray(mask_data_padded)
    mask_data_padded = transforms.ToTensor()(mask_data_padded).float()
    render['mask'] = mask_data_padded

    #  load image ------
    image = cv2.imread(image_path)[:, :, ::-1]  # (1536, 1024, 3), np.uint8, {0,...,255}
    image_padded = np.zeros((max(image.shape), max(image.shape), 3), np.uint8)  # (1536, 1536, 3)
    image_padded[:,
    image_padded.shape[0] // 2 - min(image.shape[:2]) // 2:image_padded.shape[0] // 2 + min(image.shape[:2]) // 2,
    :] = image  # (1536, 1536, 3)
    # resize to (512, 512, 3), np.uint
    image_padded = cv2.resize(image_padded, (opt.loadSize, opt.loadSize))
    image_padded = Image.fromarray(image_padded)
    image_padded = to_tensor(image_padded)  # float -1,1 , rgb, 3 ,512,512

    # use mask to recove image , so the mask must be same as image
    image_padded = mask_data_padded.expand_as(image_padded) * image_padded
    render_list.append(image_padded)  # [1, 3 , h ,w] , float -1,1

    render['img'] = torch.stack(render_list, dim=0)
    render['img'] = render['img'].to(cuda)

    normal_list.append(image_padded)
    render['normal_F'] = torch.stack(normal_list, dim=0)
    render['normal_F'] = render['normal_F'].to(cuda)

    #   load model
    with torch.no_grad():
        netG = NormalNet(opt)
        netG.to(cuda)
        # 2023/05/06 we get the 27_1984
        model_path = '/media/lx/4A42-E0B2/code_github/SePIFU_08_15into/checkpoints/model_Normal-Net/Train_Normal_epoch_27_1984'
        netG.load_state_dict(torch.load(model_path, map_location=cuda))
        netG.eval()
        # the return is a normal image!
        res, error = netG.forward(render)  # res = [1, c, h ,w ] a model result and the data is    in float -1,1, rgb
        # print(res[0].cpu())  # 1-1nei ;
        img = np.uint8((np.transpose(res[0].cpu().numpy(), (1, 2, 0)) * 0.5 + 0.5)[:, :,
                       ::-1] * 255.0)  # HWC, BGR, from -1,1 to float 0 ~ 255, de-normalized by mean 0.5 and std 0.5

        cv2.imwrite(save_path, img)
        # print(img.shape())

        # save_img_list=[]
        # for v in range(res.shape[0]):
        #     save_img = (np.transpose(res[v].detach().cpu().numpy(), (1, 2, 0)) * 0.5 + 0.5)[:, :,
        #                ::-1] * 255.0  # RGB -> BGR, (3,512,512), [0, 255]
        #     save_img_list.append(save_img)
        # save_img = np.concatenate(save_img_list, axis=1)
        # Image.fromarray(np.uint8(save_img[:, :, ::-1])).save('/home/lx/文档/python_code/train_Normal/result/033_1.png')


if __name__ == '__main__':
    main(opt)







