import numpy as np
import math
import copy
def make_rotate(rx, ry, rz):
    sinX = np.sin(rx)
    sinY = np.sin(ry)
    sinZ = np.sin(rz)

    cosX = np.cos(rx)
    cosY = np.cos(ry)
    cosZ = np.cos(rz)

    Rx = np.zeros((3,3))
    Rx[0, 0] = 1.0
    Rx[1, 1] = cosX
    Rx[1, 2] = -sinX
    Rx[2, 1] = sinX
    Rx[2, 2] = cosX

    Ry = np.zeros((3,3))
    Ry[0, 0] = cosY
    Ry[0, 2] = sinY
    Ry[1, 1] = 1.0
    Ry[2, 0] = -sinY
    Ry[2, 2] = cosY

    Rz = np.zeros((3,3))
    Rz[0, 0] = cosZ
    Rz[0, 1] = -sinZ
    Rz[1, 0] = sinZ
    Rz[1, 1] = cosZ
    Rz[2, 2] = 1.0

    R = np.matmul(np.matmul(Rz,Ry),Rx)
    return R
#
#
def generate_cameras(dist=2, view_num=360):
    cams = []
    target = [0, 0, 0]
    up = [0, 1, 0]
    for view_idx in range(view_num):
        angle = (math.pi * 2 / view_num) * view_idx
        eye = np.asarray([dist * math.sin(angle), 0, dist * math.cos(angle)])

        fwd = np.asarray(target, np.float64) - eye
        fwd /= np.linalg.norm(fwd)
        right = np.cross(fwd, up)
        right /= np.linalg.norm(right)
        down = np.cross(fwd, right)

        cams.append(
            {
                'center': eye,
                'direction': fwd,
                'right': right,
                'up': -down,
            }
        )

    return cams
def voxelization_normalization(verts, useMean=True, useScaling=True,threshWD=0.3333,threshH=0.5):
    """
    normalize the mesh into H [-0.5,0.5]*(1-margin), W/D [-0.333,0.333]*(1-margin)
    """

    vertsVoxelNorm = copy.deepcopy(verts)
    vertsMean, scaleMin = None, None

    if useMean:
        vertsMean = np.mean(vertsVoxelNorm, axis=0, keepdims=True)  # (1, 3)
        vertsVoxelNorm -= vertsMean

    xyzMin = np.min(vertsVoxelNorm, axis=0)
    assert (np.all(xyzMin < 0))
    xyzMax = np.max(vertsVoxelNorm, axis=0)
    assert (np.all(xyzMax > 0))

    if useScaling:
        scaleArr = np.array(
            [threshWD / abs(xyzMin[0]), threshH / abs(xyzMin[1]), threshWD / abs(xyzMin[2]),
             threshWD / xyzMax[0], threshH / xyzMax[1], threshWD / xyzMax[2]])
        scaleMin = np.min(scaleArr)
        vertsVoxelNorm *= scaleMin

    return vertsVoxelNorm, vertsMean, scaleMin