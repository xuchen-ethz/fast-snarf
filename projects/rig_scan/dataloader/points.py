import os
import torch
import pytorch_lightning as pl

from torch.utils.data import DataLoader, Dataset
import glob
import hydra

import pickle

from lib.model.sample import PointInSpace
import trimesh
from lib.model.smpl import SMPLServer
import pandas
from pytorch3d.transforms import rotation_conversions
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.io.obj_io import load_objs_as_meshes

import numpy as np
import trimesh
from PIL import Image

# im = Image.open("Lmobl/texture.png")
# mesh = trimesh.load('Lmobl/raw_model.obj',process=False)
# tex = trimesh.visual.TextureVisuals(image=im)
# mesh.visual.texture = tex
# mesh.show()

    
class PointsDataset(Dataset):

    def __init__(self, 
                obj_path,
                param_path,
                tex_path=None):

        self.with_tex = tex_path is not None
        # load smpl params
        smpl_file = pandas.read_pickle(param_path)
        smpl_params = torch.zeros(86)
        smpl_params[0] = 1
        smpl_params[4:7] = rotation_conversions.matrix_to_axis_angle(smpl_file['global_orient']).flatten()
        smpl_params[7:-10] = rotation_conversions.matrix_to_axis_angle(smpl_file['body_pose']).flatten()
        smpl_params[-10:] = smpl_file['betas'][0,:10]

        # load scan obj
        print('loading scan...')
        mesh = trimesh.load(obj_path, process=False)
        if self.with_tex:
            print('loading texture...')
            meshes = load_objs_as_meshes([obj_path],device=torch.device("cuda:0"), load_textures=True)
            pts_d, tex = sample_points_from_meshes(meshes, num_samples=int(1e7), return_textures=True, return_normals=False)
            pts_d = pts_d.data.cpu().numpy()[0]
            tex = tex.data.cpu().numpy()[0]
        else:
            print('sampling points...')
            pts_d = mesh.sample(int(1e6))

        pts_d = pts_d - smpl_file['transl'].data.cpu().numpy()
  
        self.smpl_params = smpl_params.cuda()
        self.pts_d_all = torch.tensor(pts_d).cuda().float()
        self.sdf_gt_all = torch.zeros_like(self.pts_d_all)[...,0]
        if self.with_tex:
            self.tex_all = torch.tensor(tex[...,:3]).cuda().float()
        print('dataset loaded...')

    def __getitem__(self, index):

        data = {}

        rand_idx = torch.randint(0, self.pts_d_all.shape[0], [int(1e5), 1], device=self.pts_d_all.device)
        data['pts_d'] = torch.gather(self.pts_d_all, 0, rand_idx.expand(-1, 3))
        data['sdf_gt']= torch.gather(self.sdf_gt_all, 0, rand_idx.squeeze(-1))
        if self.with_tex:
            data['tex_gt']= torch.gather(self.tex_all, 0, rand_idx.expand(-1, 3))

        data['smpl_params'] = self.smpl_params
        
        return data

    def __len__(self):
        return 2000 # large number for psuedo-infinite loop


