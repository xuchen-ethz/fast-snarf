import os
import torch
import pytorch_lightning as pl

from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import glob
import hydra

import pickle

from lib.model.sample import PointInSpace
from pytorch3d.io.obj_io import load_obj,load_objs_as_meshes
import trimesh
from pysdf import SDF
from lib.model.smpl import SMPLServer
import pandas
from pytorch3d.transforms import rotation_conversions

import numpy as np
import trimesh
from PIL import Image


    
class MeshDataset(Dataset):

    def __init__(self, 
                    obj_path,
                    param_path,
                    with_texture=False):

        # init point sampler
        self.sampler = PointInSpace(global_sigma=1, local_sigma=0.01)

        # load smpl params
        smpl_file = pandas.read_pickle(param_path)
        smpl_params = torch.zeros(86)
        smpl_params[0] = 1

        # format of SMPLX conversion tool
        if isinstance(smpl_file['global_orient'], torch.Tensor):
            smpl_params[4:7] = rotation_conversions.matrix_to_axis_angle(smpl_file['global_orient']).flatten()
            smpl_params[7:-10] = rotation_conversions.matrix_to_axis_angle(smpl_file['body_pose']).flatten()
            smpl_params[-10:] = smpl_file['betas'][0,:10]
            smpl_scale = 1
        # THhuman format
        else:
            smpl_params[4:7] = torch.tensor(smpl_file['global_orient']).float().flatten()
            smpl_params[7:-10] = torch.tensor(smpl_file['body_pose']).float().flatten()
            smpl_params[-10:] = torch.tensor(smpl_file['betas']).float()[0,:10]
            smpl_file['transl'] = torch.tensor(smpl_file['transl']).float()
            smpl_scale = smpl_file['scale'][0]
        # load scan obj
        print('loading scan...')
        meshes = trimesh.load(obj_path, process=False)
        scan_verts = (meshes.vertices - smpl_file['transl'].data.cpu().numpy())/smpl_scale
        scan_faces = meshes.faces

        print('computing SDF...')
        sdf_gt, pts_d = self.prepare_data(scan_verts, scan_faces)

        self.smpl_params = smpl_params.cuda()
        self.pts_d_all = torch.tensor(pts_d).cuda().float()
        self.sdf_gt_all = torch.tensor(sdf_gt).cuda().float()

        print('dataset loaded...')


    def prepare_data(self, verts, faces, n_samples=int(1e7)):

        n_verts, n_dim = verts.shape
        verts_cuda = torch.tensor(verts).cuda().float()
        rand_idx = torch.randint(0, n_verts, [n_samples, 1], device=verts_cuda.device)
        rand_pts = torch.gather(verts_cuda, 0, rand_idx.expand(-1, n_dim)).float()
        pts_d = self.sampler.get_points(rand_pts.unsqueeze(0)).data.cpu().float()[0]

        sdf_obj = SDF(verts, faces)
        sdf_gt = sdf_obj(pts_d)

        return sdf_gt, pts_d

    def __getitem__(self, index):

        data = {}
        data['smpl_params'] = self.smpl_params

        rand_idx = torch.randint(0, self.pts_d_all.shape[0], [int(1e5), 1], device=self.pts_d_all.device)
        data['pts_d'] = torch.gather(self.pts_d_all, 0, rand_idx.expand(-1, 3)).float()
        data['sdf_gt']= torch.gather(self.sdf_gt_all, 0, rand_idx.squeeze(-1)).float()

        return data

    def __len__(self):
        return 2000 # large number for psuedo-infinite loop


