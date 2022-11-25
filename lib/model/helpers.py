import torch
import math
import cv2
import numpy as np
import torch.nn.functional as F
from torch import einsum
import pytorch3d.ops as ops



def query_weights_smpl(x, smpl_verts, smpl_weights):
    
    distance_batch, index_batch, neighbor_points  = ops.knn_points(x,smpl_verts,K=1,return_nn=True)

    index_batch = index_batch[0]

    skinning_weights = smpl_weights[:,index_batch][:,:,0,:]

    return skinning_weights

def create_voxel_grid(d, h, w, device='cpu'):
    x_range = (torch.linspace(-1,1,steps=w,device=device)).view(1, 1, 1, w).expand(1, d, h, w)  # [1, H, W, D]
    y_range = (torch.linspace(-1,1,steps=h,device=device)).view(1, 1, h, 1).expand(1, d, h, w)  # [1, H, W, D]
    z_range = (torch.linspace(-1,1,steps=d,device=device)).view(1, d, 1, 1).expand(1, d, h, w)  # [1, H, W, D]
    grid = torch.cat((x_range, y_range, z_range), dim=0).reshape(1, 3,-1).permute(0,2,1)

    return grid

def bmv(m, v):
    return (m*v.transpose(-1,-2).expand(-1,3,-1)).sum(-1,keepdim=True)

def fast_inverse(T):

    shape = T.shape

    T = T.reshape(-1,4,4)
    R = T[:, :3,:3]
    t = T[:, :3,3].unsqueeze(-1)

    R_inv = R.transpose(1,2)
    t_inv = -bmv(R_inv,t)

    T_inv = T
    T_inv[:,:3,:3] = R_inv
    T_inv[:,:3,3] = t_inv.squeeze(-1)
    
    return T_inv.reshape(shape)


def skinning(x, w, tfs, inverse=False):
    """Linear blend skinning

    Args:
        x (tensor): canonical points. shape: [B, N, D]
        w (tensor): conditional input. [B, N, J]
        tfs (tensor): bone transformation matrices. shape: [B, J, D+1, D+1]
    Returns:
        x (tensor): skinned points. shape: [B, N, D]
    """
    x_h = F.pad(x, (0, 1), value=1.0)
    b,p,n = w.shape

    if inverse:
        # p:n_point, n:n_bone, i,k: n_dim+1
        w_tf = einsum("bpn,bnij->bpij", w, tfs)

        x_h = x_h.view(b,p,1,4).expand(b,p,4,4)
        x_h = ((w_tf.inverse())*x_h).sum(-1)

    else:
        w_tf = einsum("bpn,bnij->bpij", w, tfs)

        x_h = x_h.view(b,p,1,4).expand(b,p,4,4)
        x_h = (w_tf*x_h).sum(-1)

    return x_h[:, :, :3]


def masked_softmax(vec, mask, dim=-1, mode='softmax', soft_blend=1):
    if mode == 'softmax':

        vec = torch.distributions.Bernoulli(logits=vec).probs

        masked_exps = torch.exp(soft_blend*vec) * mask.float()
        masked_exps_sum = masked_exps.sum(dim)

        output = torch.zeros_like(vec)
        output[masked_exps_sum>0,:] = masked_exps[masked_exps_sum>0,:]/ masked_exps_sum[masked_exps_sum>0].unsqueeze(-1)

        output = (output * vec).sum(dim, keepdim=True)

        output = torch.distributions.Bernoulli(probs=output).logits

    elif mode == 'max':
        vec[~mask] = -15.94
        output = torch.max(vec, dim, keepdim=True)[0]

    return output


''' Hierarchical softmax following the kinematic tree of the human body. Imporves convergence speed'''
def hierarchical_softmax(x):
    def softmax(x):
        return torch.nn.functional.softmax(x, dim=-1)

    def sigmoid(x):
        return torch.sigmoid(x)

    n_batch, n_point, n_dim = x.shape
    x = x.flatten(0,1)

    prob_all = torch.ones(n_batch * n_point, 24, device=x.device)

    prob_all[:, [1, 2, 3]] = prob_all[:, [0]] * sigmoid(x[:, [0]]) * softmax(x[:, [1, 2, 3]])
    prob_all[:, [0]] = prob_all[:, [0]] * (1 - sigmoid(x[:, [0]]))

    prob_all[:, [4, 5, 6]] = prob_all[:, [1, 2, 3]] * (sigmoid(x[:, [4, 5, 6]]))
    prob_all[:, [1, 2, 3]] = prob_all[:, [1, 2, 3]] * (1 - sigmoid(x[:, [4, 5, 6]]))

    prob_all[:, [7, 8, 9]] = prob_all[:, [4, 5, 6]] * (sigmoid(x[:, [7, 8, 9]]))
    prob_all[:, [4, 5, 6]] = prob_all[:, [4, 5, 6]] * (1 - sigmoid(x[:, [7, 8, 9]]))

    prob_all[:, [10, 11]] = prob_all[:, [7, 8]] * (sigmoid(x[:, [10, 11]]))
    prob_all[:, [7, 8]] = prob_all[:, [7, 8]] * (1 - sigmoid(x[:, [10, 11]]))

    prob_all[:, [12, 13, 14]] = prob_all[:, [9]] * sigmoid(x[:, [24]]) * softmax(x[:, [12, 13, 14]])
    prob_all[:, [9]] = prob_all[:, [9]] * (1 - sigmoid(x[:, [24]]))

    prob_all[:, [15]] = prob_all[:, [12]] * (sigmoid(x[:, [15]]))
    prob_all[:, [12]] = prob_all[:, [12]] * (1 - sigmoid(x[:, [15]]))

    prob_all[:, [16, 17]] = prob_all[:, [13, 14]] * (sigmoid(x[:, [16, 17]]))
    prob_all[:, [13, 14]] = prob_all[:, [13, 14]] * (1 - sigmoid(x[:, [16, 17]]))

    prob_all[:, [18, 19]] = prob_all[:, [16, 17]] * (sigmoid(x[:, [18, 19]]))
    prob_all[:, [16, 17]] = prob_all[:, [16, 17]] * (1 - sigmoid(x[:, [18, 19]]))

    prob_all[:, [20, 21]] = prob_all[:, [18, 19]] * (sigmoid(x[:, [20, 21]]))
    prob_all[:, [18, 19]] = prob_all[:, [18, 19]] * (1 - sigmoid(x[:, [20, 21]]))

    prob_all[:, [22, 23]] = prob_all[:, [20, 21]] * (sigmoid(x[:, [22, 23]]))
    prob_all[:, [20, 21]] = prob_all[:, [20, 21]] * (1 - sigmoid(x[:, [22, 23]]))

    prob_all = prob_all.reshape(n_batch, n_point, prob_all.shape[-1])
    return prob_all

def rectify_pose(pose, root_abs):
    """
    Rectify AMASS pose in global coord adapted from https://github.com/akanazawa/hmr/issues/50.
 
    Args:
        pose (72,): Pose.

    Returns:
        Rotated pose.
    """
    pose = pose.copy()
    R_abs = cv2.Rodrigues(root_abs)[0]
    R_root = cv2.Rodrigues(pose[:3])[0]
    new_root = np.linalg.inv(R_abs).dot(R_root)
    pose[:3] = cv2.Rodrigues(new_root)[0].reshape(3)
    return pose

def tv_loss(vox,l2=True):
    """
    Compute total variation loss.
    Inputs:
    - img: PyTorch Variable of shape (1, 3, H, W) holding an input image.
    - tv_weight: Scalar giving the weight w_t to use for the TV loss.
    Returns:
    - loss: PyTorch Variable holding a scalar giving the total variation loss
      for img weighted by tv_weight.
    """
    b,c,d,w,h = vox.shape
    d_variance = vox[:,:,:,:,:-1] - vox[:,:,:,:,1:]
    w_variance = vox[:,:,:,:-1,:] - vox[:,:,:,1:,:]
    h_variance = vox[:,:,:-1,:,:] - vox[:,:,1:,:,:]

    if l2:
        loss = (h_variance.pow(2).sum() + w_variance.pow(2).sum() + d_variance.pow(2).sum())/ (d*w*h)
    else:
        loss = (h_variance.abs().sum() + w_variance.abs().sum() + d_variance.abs().sum())/ (d*w*h)
    return loss