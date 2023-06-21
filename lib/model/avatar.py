import torch
import numpy as np
from tqdm import trange
from lib.model.network_ngp import ImplicitNetwork
from lib.model.fast_snarf import ForwardDeformer
from lib.utils.meshing import generate_mesh
from lib.model.smpl import SMPLServer
from lib.utils.render import render_trimesh
from lib.model.implicit_renderer import DepthModule
import pytorch3d.ops as ops
from lib.model.sample import PointInSpace
from pysdf import SDF

class AvatarModel(torch.nn.Module):

    def __init__(self, opt, gender='neutral', smpl_betas=None):
        super().__init__()

        self.opt = opt

        self.smpl_server = SMPLServer(gender=gender, cano_pose='T', betas=smpl_betas).cuda()

        self.deformer = ForwardDeformer(opt, self.smpl_server)

        self.cano_model = ImplicitNetwork(3, 4, 64, 3).cuda()

        self.smpl_betas = torch.zeros(1, 10).cuda() if smpl_betas is None else smpl_betas

        self.renderer = DepthModule(n_secant_steps=8,
                                    max_points=500000,
                                    check_cube_intersection=False,
                                    depth_range=[0,2],
                                    n_steps=[128,129])

    def set_pose(self, smpl_params):

        self.smpl_params = smpl_params
        self.smpl_params[:, 76:] = self.smpl_betas
        self.smpl_outputs = self.smpl_server.forward(self.smpl_params)
        self.smpl_verts = self.smpl_outputs['smpl_verts']
        self.smpl_tfs = torch.einsum('bnij,njk->bnik', self.smpl_outputs['smpl_tfs'], self.smpl_server.tfs_c_inv) 
        self.smpl_thetas = self.smpl_params[:, 4:76]

        self.device = self.smpl_params.device

        self.cond = {'smpl': self.smpl_params[:,7:-10]/np.pi}

    def pretrain(self):
        
        sampler = PointInSpace(global_sigma=1, local_sigma=0.01)
        optimizer = torch.optim.Adam(self.cano_model.parameters(), lr=1e-3)

        self.set_pose(self.smpl_server.param_canonical)

        verts = self.smpl_server.verts_c[0]
        n_verts, n_dim = verts.shape
        rand_idx = torch.randint(0, n_verts, [int(1e7), 1], device=verts.device)
        rand_pts = torch.gather(verts, 0, rand_idx.expand(-1, n_dim))
        pts_c = sampler.get_points(rand_pts.unsqueeze(0))

        sdf_obj = SDF(verts.data.cpu().numpy(), self.smpl_server.smpl.faces)
        sdf_gt = sdf_obj(pts_c.data.cpu().numpy()[0])

        sdf_gt = torch.tensor(sdf_gt).cuda().float()[None,...]

        for iter in trange(1000):
            optimizer.zero_grad()

            random_idx = torch.randint(0, pts_c.shape[1], [1, int(1e5), 1], device=pts_c.device)
            pts_c_cur = torch.gather(pts_c, 1, random_idx.expand(-1, -1, n_dim))
            sdf_gt_cur = torch.gather(sdf_gt, 1, random_idx.squeeze(-1))

            loss = torch.nn.functional.l1_loss(self.predict_cano(pts_c_cur)[...,0], sdf_gt_cur)
            loss.backward()
            optimizer.step()

    def normalize_cano(self, pts):
    
        pts = (pts + self.deformer.offset) * self.deformer.scale
        pts = (pts + 1) / 2

        return pts

    def predict_cano(self, pts, mask=None):

        pts = self.normalize_cano(pts)

        mask_oob = ((pts.max(-1)[0] <1) & (pts.min(-1)[0] > 0)).flatten() 
        mask = mask_oob if mask is None else mask & mask_oob

        res = self.cano_model(pts, self.cond, mask=mask)
        
        return res

    def predict_posed(self, pts, mask=None):

        pts_c, others = self.deformer(pts, self.cond, self.smpl_tfs, mask=mask, eval_mode=True)
        mask = others['valid_ids']

        n_batch, n_point, n_init, n_dim = pts_c.shape
        pts_c = pts_c.reshape(n_batch, n_point * n_init, n_dim)

        res = self.predict_cano(pts_c, mask=mask.flatten())
        res = res.reshape(n_batch, n_point, n_init, res.shape[-1])

        res[...,0][~mask] = -100
        _, idx = torch.max(res[...,0], -1, keepdim=True)

        res = torch.gather(res, 2, idx.unsqueeze(-1).expand(-1,-1, 1, res.shape[-1])).squeeze(2)

        return res


    def to_mesh(self, res_up=3, with_tex=False, cano=False, use_mise=False):

        func = self.predict_cano if cano else self.predict_posed

        occ_func = lambda x: func(x)[...,0].reshape(-1, 1)
        mesh = generate_mesh(occ_func, self.smpl_verts[0], res_up=res_up, use_mise=use_mise)

        if with_tex:
            v  = torch.tensor(mesh.vertices).type_as(self.smpl_verts).unsqueeze(0)
            color = func(v)[...,1:].clamp(0,1)*255
            mesh.visual.vertex_colors = (color.data.cpu().numpy()[0]).astype(np.uint8)

        return mesh

    def to_image(self, res_up=3, with_tex=False, cano=False, use_mise=False):

        mesh = self.to_mesh(res_up=res_up, with_tex=with_tex, cano=cano, use_mise=use_mise)
        img = render_trimesh(mesh, mode='a' if with_tex else 'n')
        return img

    def to_image_implicit(self, res=512, with_tex=False, cano=False):

        yv, xv = torch.meshgrid([torch.linspace(-1, 1, res), torch.linspace(-1, 1, res)])
        ray_origins = torch.stack([xv, yv], dim=-1).type_as(self.smpl_tfs)
        ray_origins = ray_origins.reshape(1,res*res,2)

        ray_origins = torch.stack([ray_origins[...,0], -ray_origins[...,1] - 0.3, torch.zeros_like(ray_origins[...,0]) + 1], dim=-1)
        ray_dirs = torch.zeros_like(ray_origins)
        ray_dirs[...,-1] = -1

        func = self.predict_cano if cano else self.predict_posed
        def occ_func(x, mask=None):
            res = func(x)[...,0]
            if mask is None:
                return res.reshape(1, -1, 1)
            else:
                return res[mask].reshape(-1,1)

        depth = self.renderer(ray_origins, ray_dirs, occ_func)
        coords = ray_origins + depth.unsqueeze(-1) * ray_dirs

        depth_mask = depth.isfinite()

        colors = func(coords, mask=depth_mask.unsqueeze(-1))[...,1:].clamp(0,1)*255
        colors[~depth_mask] = 255

        img = colors.reshape(res,res,3).data.cpu().numpy().astype(np.uint8)
        return img