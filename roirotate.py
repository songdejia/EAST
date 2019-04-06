import torch.nn.functional as F
import numpy as np
import cv2
import torch
import visdom
"""
box: 1 * num (in a img) * 4 * 2
img: 1 * 3 * 512 * 512
feature: 1 * 32 * 128 *128   1/4 img 

output:
rotated_feature: num * w * h * 32

Args:
        theta (Tensor): input batch of affine matrices (:math:`N \times 2 \times 3`)
        size (torch.Size): the target output image size (:math:`N \times C \times H \times W`)
                           Example: torch.Size((32, 3, 24, 24))
"""
"""
Args:
        input (Tensor): input of shape :math:`(N, C, H_\text{in}, W_\text{in})` (4-D case)
                        or :math:`(N, C, D_\text{in}, H_\text{in}, W_\text{in})` (5-D case)
        grid (Tensor): flow-field of shape :math:`(N, H_\text{out}, W_\text{out}, 2)` (4-D case)
                       or :math:`(N, D_\text{out}, H_\text{out}, W_\text{out}, 3)` (5-D case)
        mode (str): interpolation mode to calculate output values
            'bilinear' | 'nearest'. Default: 'bilinear'
        padding_mode (str): padding mode for outside grid values
            'zeros' | 'border' | 'reflection'. Default: 'zeros'

    Returns:
        output (Tensor): output Tensor
"""
vis = visdom.Visdom()
def rotate(img, rectangle, feature, ht = 8):
    #resize_img = torch.from_numpy(cv2.resize(img[0].permute(1,2,0).cpu().numpy(), (128,128))).permute(2,0,1).cuda()
    #vis.image(img[0])
    #vis.image(resize_img)
    vis.heatmap(feature[0][0])

    rectangle = rectangle.numpy()
    w_max = 0
    l = [0, 0, 1]
    scale = feature.shape[2] / img.shape[2] #if w=h
    for rect in rectangle[0]:
        rect = rect * scale   # position on feature_map, it is the same for computing on image, cause the pts1 is identical through normolized to -1,1
        w = rect[1][0] - rect[0][0]
        h = rect[2][1] - rect[1][1]
        s = ht/h
        wt = s*w
        if wt > w_max:
            w_max = wt
        ###########normalized coordinates correspond
        pts1 = np.float32([[rect[0][0]/(256* scale)-1, rect[0][1]/(256* scale)-1], [rect[1][0]/(256* scale)-1, rect[1][1]/(256* scale)-1], [rect[2][0]/(256* scale)-1, rect[2][1]/(256* scale)-1]])  #顺时针
        pts2 = np.float32([[-1, -1], [1, -1], [1, 1]])   #pts1和pts2的点位置要对应
        M = cv2.getAffineTransform(pts1, pts2)
        print(M)

        ###########inverse M to adopt affine_grid
        param = np.vstack((M, l))
        param_inv = np.linalg.inv(param)
        M_inv = np.delete(param_inv, (2), axis=0)
        print(M_inv)

        grid = F.affine_grid(torch.from_numpy(M_inv[None]), torch.Size((1, feature.shape[1], ht, int(wt) + 1)))
        rotated_feature = F.grid_sample(feature, grid.float().cuda())
        #grid = F.affine_grid(torch.from_numpy(M_inv[None]), torch.Size((1, feature.shape[1], int(h), int(w))))  #before resized to ht = 8
        #crop_img = F.grid_sample(img, grid.float().cuda())
        #resized = F.grid_sample(torch.unsqueeze(resize_img, 0), grid.float().cuda())
        #vis.image(crop_img[0])
        #vis.image(resized[0])
        vis.heatmap(rotated_feature[0][0])
    w_max = int(w_max) + 1
    return rotated_feature
