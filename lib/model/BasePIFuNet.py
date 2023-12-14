import torch
import torch.nn as nn
import torch.nn.functional as F

from ..geometry import index, index_3d, multiRanges_deepVoxels_sampling, orthogonal, perspective

class BasePIFuNet(nn.Module):
    def __init__(self,
                 projection_mode='orthogonal',
                 error_term=nn.MSELoss(),
                 error_term_semantic=nn.MSELoss(),
                 ):

        super(BasePIFuNet, self).__init__()
        self.name = 'base'

        self.error_term = error_term
        self.error_term_semantic = error_term_semantic

        self.index = index
        self.index_3d = index_3d
        self.projection = orthogonal if projection_mode == 'orthogonal' else perspective
        self.multiRanges_deepVoxels_sampling = multiRanges_deepVoxels_sampling

        self.preds = None
        self.labels = None

    def forward(self, points, images, calibs, transforms=None):
        '''
        :param points: [B, 3, N] world space coordinates of points
        :param images: [B, C, H, W] input images
        :param calibs: [B, 3, 4] calibration matrices for each image
        :param transforms: Optional [B, 2, 3] image space coordinate transforms
        :return: [B, Res, N] predictions for each point
        '''
        self.filter(images)
        self.query(points, calibs, transforms)
        return self.get_preds()

    def filter(self, images):

        None

    def query(self, points, calibs, transforms=None, labels=None):

        None

    def get_preds(self):

        return self.preds

    def get_error(self):

        return self.error_term(self.preds, self.labels)
