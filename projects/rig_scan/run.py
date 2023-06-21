import os
import yaml
import torch
import imageio
import numpy as np
from munch import Munch
from tqdm import trange, tqdm
import torch.nn.functional as F

import sys
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(os.path.dirname(current))
sys.path.append(parent)

from lib.model.smpl import SMPLServer
from lib.model.avatar import AvatarModel
from lib.model.helpers import load_motion_sequence, gradient
from dataloader.mesh import MeshDataset
from dataloader.points import PointsDataset

# read argument from command line
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--obj_path', type=str, default='data/CustomHuman/00070_3_0001/mesh-f00002.obj')
parser.add_argument('--param_path', type=str, default='data/CustomHuman/00070_3_0001/mesh-f00002_smpl.pkl')
parser.add_argument('--tex_path', type=str, default='data/CustomHuman/00070_3_0001/mesh-f00002.png')
parser.add_argument('--with_tex', action='store_true')
parser.add_argument('--max_iter', type=int, default=2000)
parser.add_argument('--loose_cloth', action='store_true')
parser.add_argument('--self_contact', action='store_true')

args = parser.parse_args()

with_tex = args.with_tex
loose_cloth = args.loose_cloth
self_contact = args.self_contact

dataset = MeshDataset(obj_path=args.obj_path, param_path=args.param_path)
dataloder = torch.utils.data.DataLoader(dataset, batch_size=1)
if with_tex:
    tex_dataset = PointsDataset(obj_path=args.obj_path, param_path=args.param_path, tex_path=args.tex_path)
    tex_dataloder = torch.utils.data.DataLoader(tex_dataset, batch_size=1)
smpl_betas = next(iter(dataloder))['smpl_params'][:,-10:]

# Define model
opt = Munch(yaml.load(open('config/deformer/fast_snarf.yaml'))['model']['deformer']['opt'])
opt.skinning_mode = 'preset'
opt.cvg = 1e-4
opt.dvg = 1
if loose_cloth:
    opt.skinning_mode = 'preset_smooth'
    opt.gloabl_scale = 1.5
    opt.n_iters = 15
avatar_model = AvatarModel(opt, gender='neutral', smpl_betas=smpl_betas)
avatar_model.pretrain() # intialize shape with SMPL params

# Optimize
optimizer = torch.optim.Adam(avatar_model.parameters(), lr=1e-3)

cur_iter = 0
for cur_iter in trange(args.max_iter):

    # sdf
    sample = next(iter(dataloder))
    
    avatar_model.set_pose(sample['smpl_params'])
    res_pd = avatar_model.predict_posed(sample['pts_d'])
    loss = F.l1_loss(res_pd[...,0], sample['sdf_gt'])

    # pts_c_rand = torch.rand( (1, int(1e5), 3)).cuda()
    # pts_c_rand.requires_grad_()
    # occ_rand = avatar_model.predict_cano(pts_c_rand)[...,0]
    # grad_eikonal = gradient(pts_c_rand, occ_rand)
    # grad_loss = ((grad_eikonal.norm(2, dim=-1) - 1) ** 2).mean()
    # loss = loss + 0.002*grad_loss + 1*torch.exp(-100*occ_rand.abs()).mean()

    # texture
    if with_tex:
        sample_tex = next(iter(tex_dataloder))

        avatar_model.set_pose(sample_tex['smpl_params'])
        res_pd = avatar_model.predict_posed(sample_tex['pts_d'])
        loss = loss + F.l1_loss(res_pd[...,1:], sample_tex['tex_gt'])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# load motion sequence
smpl_params_all = load_motion_sequence('data/aist_demo/seqs', skip=2)
# smpl_params_all = load_motion_sequence('data/ACCAD/Male2MartialArtsExtended_c3d/Extended 1_poses.npz', skip=5)[:450]
images = []
for i in trange(smpl_params_all.shape[0]):
    
    avatar_model.set_pose(smpl_params_all[[i]])
    img = avatar_model.to_image(res_up=3, with_tex=with_tex, use_mise=False)

    images.append(img)
    imageio.imsave('img.png', img) 
imageio.mimsave('anim.mp4', images)
