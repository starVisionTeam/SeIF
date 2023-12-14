
import torch
import torch.nn as nn
import sys, os


from train_Normal.lib.model.BasePIFuNet import BasePIFuNet
from train_Normal.lib.model.HGFilters import *
from train_Normal.lib.net_util import init_net
from train_Normal.lib.net_util import VGGLoss
from train_Normal.lib.model.FBNet import define_G

class NormalNet(BasePIFuNet):
    def __init__(self, opt, error_term=nn.SmoothL1Loss()):

        super(NormalNet, self).__init__(error_term=error_term)
        self.l1_loss = nn.SmoothL1Loss()  # define a L1 LOSS ;
        self.opt = opt
        self.name= "pix2pixHD"
        if self.training:
            print('self.training is true ')
            self.vgg_loss = [VGGLoss()]

        self.netF = None
        if True:
            self.in_nmlF_dim = 3

            self.netF = define_G(self.in_nmlF_dim, 3, 64, "global", 4, 9, 1, 3,
                                 "instance")
        # initialize network with normal
        init_net(self)
    def filter(self, images):
        '''
        apply a fully convolutional network to images.
        the resulting feature will be stored.
        args:
            images: [B, C, H, W]
        '''
        # if you wish to train jointly, remove detach etc.
        if self.netF is not None:
            self.nmlF = self.netF.forward(images)
            # nmls.append(self.nmlF)

        mask = (images.abs().sum(dim=1, keepdim=True) !=
                0.0).detach().float()  # torch.Size([2, 1, 512, 512])
        self.nmlF = self.nmlF * mask  #  torch.Size([2, 3, 512, 512])

    def forward(self, input_data):
        self.filter(input_data['img'])   # [b , 3, 512, 512 ]
        # output: float_arr [-1,1] with [B, C, H, W]
        error=self.get_norm_error(input_data['normal_F'])  # 【b, 3, 512, 512]

        return self.nmlF, error


    def get_norm_error(self, tgt_F):
        """calculate normal loss
        Args:
            self.nmlf (torch.tensor): [B, 3, 512, 512]
            tagt (torch.tensor): [B, 3, 512, 512]
        """
        l1_F_loss = self.l1_loss(self.nmlF, tgt_F)
        # this also can change into a houbei NOrmal predict! ! tow net are different weight!

        # from icon we can see, caculate vgg_lose doesnt need gradient;

        # with torch.no_grad():
        #     vgg_F_loss = self.vgg_loss[0](self.nmlF, tgt_F)

        # total_loss = 5.0 * l1_F_loss + vgg_F_loss   # do it fellow some weight ;
        total_loss = 5.0 * l1_F_loss
        return total_loss
