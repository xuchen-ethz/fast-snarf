import os
import numpy as np
os.chdir('/local/home/xuchen/fast-snarf')
print(os.getcwd())

import torch
from tqdm import trange, tqdm
import yaml
from munch import Munch
from lib.utils.check_sign import check_sign
from lib.model.smpl import SMPLServer
from lib.model.network_ngp import ImplicitNetwork
from lib.model.fast_snarf import ForwardDeformer
from lib.model.helpers import masked_softmax
from lib.utils.meshing import generate_mesh
from lib.model.sample import PointInSpace
from pytorch3d.io.obj_io import load_obj,load_objs_as_meshes
import trimesh
from pysdf import SDF
import pandas
from pytorch3d.ops import sample_points_from_meshes

with_texture = True
def occ_func(x):
    batch_points = 200000000
    acc = []
    for pts_d_split in torch.split(x, batch_points, dim=1):
        
        pts_d_split = (pts_d_split + deformer.offset) * deformer.scale
        pts_d_split = (pts_d_split+1)/2
        
        occ = network(pts_d_split, {'smpl': smpl_params[:,7:-10]/np.pi})[...,:1]
    
        mask = torch.logical_or(pts_d_split.max(-1).values >= 1, pts_d_split.min(-1).values <= 0)
        occ[mask] = -100

        acc.append(occ)
    acc = torch.cat(acc,dim=1).reshape(-1, 1)
    return acc

def prepare_data(scan_verts, scan_faces):

    _, num_verts, num_dim = scan_verts.shape
    random_idx = torch.randint(0, num_verts, [1, int(1e8), 1], device=scan_verts.device)
    random_pts = torch.gather(scan_verts, 1, random_idx.expand(-1, -1, num_dim)).float()
    pts_d = sampler.get_points(random_pts)


    mesh = trimesh.Trimesh(vertices=scan_verts[0].cpu().numpy(), faces=scan_faces.cpu().numpy())
    sdf_obj = SDF(mesh.vertices, mesh.faces)
    sdf_gt = sdf_obj(pts_d[0].cpu().numpy())
    sdf_gt = torch.tensor(sdf_gt).cuda().float().unsqueeze(0)
    # sdf_gt = sdf_gt.clamp(-1,1)
    return sdf_gt, pts_d


# Prepare the data


smpl_params = torch.zeros(86).cuda().float()[None]
smpl_params[0,0] = 1

gender      = 'male'
betas       = smpl_params[:,-10:]
smpl_server = SMPLServer(gender=gender, betas=betas).cuda()
sampler = PointInSpace(global_sigma=1.8, local_sigma=0.01)


smpl_outputs = smpl_server.forward(smpl_params)
smpl_verts = smpl_outputs['smpl_verts']
smpl_verts_cano = smpl_server.verts_c
smpl_faces = torch.tensor(smpl_server.smpl.faces.astype('int')).cuda()
num_dim = 3


smpl_tfs = smpl_outputs['smpl_tfs']
cond = {'smpl': smpl_params[:,7:-10].cuda()/np.pi, 'smpl_params': smpl_params.cuda()}

sdf_gt_smpl, pts_d_smpl = prepare_data(smpl_verts_cano, smpl_faces)

# Canonical model
network = ImplicitNetwork(3, 1, 64, 3).cuda()

# Deformer
opt = Munch(yaml.load(open('config/deformer/fast_snarf.yaml'))['model']['deformer']['opt'])
opt.skinning_mode = 'preset'
opt.cvg = float(opt.cvg)
opt.dvg = float(opt.dvg)
deformer = ForwardDeformer(opt,smpl_server).cuda()

# Optimizer
optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)

for iter in trange(1000):
    optimizer.zero_grad()

    random_idx = torch.randint(0, pts_d_smpl.shape[1], [1, int(1e5), 1], device=pts_d_smpl.device)
    pts_c = torch.gather(pts_d_smpl, 1, random_idx.expand(-1, -1, num_dim)).float()
    sdf_gt_cur = torch.gather(sdf_gt_smpl, 1, random_idx.squeeze(-1)).float()

    pts_c = ((pts_c + deformer.offset) * deformer.scale+1)/2
    occ_pd = network(pts_c, cond)[:,:,:1]

    loss = torch.nn.functional.l1_loss(occ_pd, sdf_gt_cur.unsqueeze(-1))
    loss.backward()
    optimizer.step()
    if iter%100==0:
        print(loss.item())

