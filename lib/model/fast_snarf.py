import torch
import torch.nn.functional as F


from lib.model.network import ImplicitNetwork
from lib.model.helpers import hierarchical_softmax, skinning, bmv, create_voxel_grid, query_weights_smpl
from torch.utils.cpp_extension import load
import os

cuda_dir = os.path.join(os.path.dirname(__file__), "../cuda")
fuse_kernel = load(name='fuse_cuda',
                   extra_cuda_cflags=[],
                   sources=[f'{cuda_dir}/fuse_kernel/fuse_cuda.cpp',
                            f'{cuda_dir}/fuse_kernel/fuse_cuda_kernel.cu'])
filter_cuda = load(name='filter',
                   sources=[f'{cuda_dir}/filter/filter.cpp',
                            f'{cuda_dir}/filter/filter.cu'])
precompute_cuda = load(name='precompute',
                   sources=[f'{cuda_dir}/precompute/precompute.cpp',
                            f'{cuda_dir}/precompute/precompute.cu'])


class ForwardDeformer(torch.nn.Module):
    """
    Tensor shape abbreviation:
        B: batch size
        N: number of points
        J: number of bones
        I: number of init
        D: space dimension
    """

    def __init__(self, opt, smpl_server=None):
        super().__init__()

        self.opt = opt

        self.init_bones = [0, 1, 2, 4, 5, 16, 17, 18, 19] 

        self.init_bones_cuda = torch.tensor(self.init_bones).cuda().int()

        # convert to voxel grid
        smpl_verts = smpl_server.verts_c
        device = smpl_server.verts_c.device

        d, h, w = self.opt.res//self.opt.z_ratio, self.opt.res, self.opt.res
        grid = create_voxel_grid(d, h, w, device=device)
        
        gt_bbox = torch.cat([smpl_verts.min(dim=1).values, 
                             smpl_verts.max(dim=1).values], dim=0)
        self.offset = -(gt_bbox[0] + gt_bbox[1])[None,None,:] / 2

        self.scale = torch.zeros_like(self.offset)
        self.scale[...] = 1./((gt_bbox[1] - gt_bbox[0]).max()/2 * self.opt.global_scale)
        self.scale[:,:,-1] = self.scale[:,:,-1] * self.opt.z_ratio

        self.grid_denorm = grid/self.scale - self.offset

        if self.opt.skinning_mode == 'preset':
            self.lbs_voxel_final = query_weights_smpl(self.grid_denorm, smpl_verts, smpl_server.weights_c)
            self.lbs_voxel_final = self.lbs_voxel_final.permute(0,2,1).reshape(1,-1,d,h,w)

        elif self.opt.skinning_mode == 'voxel':
            lbs_voxel = 0.001 * torch.ones((1, 24, d, h, w), dtype=self.grid_denorm.dtype, device=self.grid_denorm.device)
            self.register_parameter('lbs_voxel', torch.nn.Parameter(lbs_voxel,requires_grad=True))

        elif self.opt.skinning_mode == 'mlp':
            self.lbs_network = ImplicitNetwork(**self.opt.network)

        else:
            raise NotImplementedError('Unsupported Deformer.')

    def forward(self, xd, cond, tfs, eval_mode=False):
        """Given deformed point return its caonical correspondence

        Args:
            xd (tensor): deformed points in batch. shape: [B, N, D]
            cond (dict): conditional input.
            tfs (tensor): bone transformation matrices. shape: [B, J, D+1, D+1]

        Returns:
            xc (tensor): canonical correspondences. shape: [B, N, I, D]
            others (dict): other useful outputs.
        """

        xc_opt, others = self.search(xd, cond, tfs, eval_mode=eval_mode)

        if eval_mode or self.opt.skinning_mode == 'preset':
            return xc_opt, others

        # do not back-prop through broyden
        xc_opt = xc_opt.detach()

        n_batch, n_point, n_init, n_dim = xc_opt.shape

        xd_opt = self.forward_skinning(xc_opt.reshape((n_batch, n_point * n_init, n_dim)), cond, tfs, mask=others['valid_ids'].flatten(1,2))

        if not self.opt.use_slow_grad:
            grad_inv = others['J_inv'].reshape(n_batch, n_point * n_init, 3,3)
        else:
            grad_inv = self.gradient(xc_opt.reshape((n_batch, n_point * n_init, n_dim)), cond, tfs).inverse().detach()

        correction = xd_opt - xd_opt.detach()
        correction = bmv(-grad_inv.flatten(0,1), correction.unsqueeze(-1).flatten(0,1)).squeeze(-1).reshape(xc_opt.shape)

        xc = xc_opt + correction
        xc = xc.reshape(n_batch, n_point, n_init, n_dim)

        return xc, others

    def precompute(self, tfs=None, recompute_skinning=True):

        if recompute_skinning or not hasattr(self,"lbs_voxel_final"):

            if self.opt.skinning_mode == 'mlp':
                self.mlp_to_voxel()
            
            elif self.opt.skinning_mode == 'voxel':
                self.voxel_to_voxel()

        b, c, d, h, w = tfs.shape[0], 3, self.opt.res//self.opt.z_ratio, self.opt.res, self.opt.res

        voxel_d = torch.zeros( (b,3,d,h,w), device=tfs.device)
        voxel_J = torch.zeros( (b,12,d,h,w), device=tfs.device)
        
        precompute_cuda.precompute(self.lbs_voxel_final, tfs, voxel_d, voxel_J, self.offset, self.scale)

        return voxel_d, voxel_J

    def search(self, xd, cond, tfs, eval_mode=False):
        """Search correspondences.

        Args:
            xd (tensor): deformed points in batch. shape: [B, N, D]
            xc_init (tensor): deformed points in batch. shape: [B, N, I, D]
            cond (dict): conditional input.
            tfs (tensor): bone transformation matrices. shape: [B, J, D+1, D+1]

        Returns:
            xc_opt (tensor): canonoical correspondences of xd. shape: [B, N, I, D]
            valid_ids (tensor): identifiers of converged points. [B, N, I]
        """

        voxel_d, voxel_J = self.precompute(tfs, recompute_skinning=not eval_mode)

        # run broyden without grad
        with torch.no_grad():
            result = self.broyden_cuda(xd, voxel_d, voxel_J, tfs, 
                                    cvg_thresh=self.opt.cvg,
                                    dvg_thresh=self.opt.dvg)

        return result['result'], result


    def broyden_cuda(self,
                    xd_tgt,
                    voxel,
                    voxel_J_inv,
                    tfs,
                    cvg_thresh=1e-4,
                    dvg_thresh=1):
        """
        Args:
            g:     f: (N, 3, 1) -> (N, 3, 1)
            x:     (N, 3, 1)
            J_inv: (N, 3, 3)
        """
        b, n, _ = xd_tgt.shape

        n_init = self.init_bones_cuda.shape[0]

        xc = torch.zeros((b,n,n_init,3),device=xd_tgt.device,dtype=torch.float)

        J_inv = torch.zeros((b,n,n_init,3,3),device=xd_tgt.device,dtype=torch.float)

        is_valid = torch.zeros((b,n,n_init),device=xd_tgt.device,dtype=torch.bool)

        fuse_kernel.fuse_broyden(xc, xd_tgt, voxel, voxel_J_inv, tfs, self.init_bones_cuda, self.opt.align_corners, J_inv, is_valid, self.offset, self.scale, cvg_thresh, dvg_thresh)

        mask = filter_cuda.filter(xc, is_valid)

        return {"result": xc, 'valid_ids': mask, 'J_inv': J_inv}


    def forward_skinning(self, xc, cond, tfs, mask=None):
        """Canonical point -> deformed point

        Args:
            xc (tensor): canonoical points in batch. shape: [B, N, D]
            cond (dict): conditional input.
            tfs (tensor): bone transformation matrices. shape: [B, J, D+1, D+1]

        Returns:
            xd (tensor): deformed point. shape: [B, N, D]
        """
        if mask is None:
            w = self.query_weights(xc, cond)
            xd = skinning(xc, w, tfs, inverse=False)
        else:
            w = self.query_weights(xc, cond, mask=mask.flatten(0,1))
            xd = skinning(xc,w, tfs, inverse=False)

        return xd

    def mlp_to_voxel(self):

        d, h, w = self.opt.res//self.opt.z_ratio, self.opt.res, self.opt.res

        lbs_voxel_final = self.lbs_network(self.grid_denorm, {}, None)
        lbs_voxel_final = self.opt.soft_blend * lbs_voxel_final

        if self.opt.softmax_mode == "hierarchical":
            lbs_voxel_final = hierarchical_softmax(lbs_voxel_final)
        else:
            lbs_voxel_final = F.softmax(lbs_voxel_final, dim=-1)

        self.lbs_voxel_final = lbs_voxel_final.permute(0,2,1).reshape(1,24,d,h,w)

    def voxel_to_voxel(self):

        lbs_voxel_final = self.lbs_voxel*self.opt.soft_blend

        self.lbs_voxel_final = F.softmax(lbs_voxel_final, dim=1)

    def query_weights(self, xc, cond=None, mask=None, mode='bilinear'):

        if not hasattr(self,"lbs_voxel_final"):
            if self.opt.skinning_mode == 'mlp':
                self.mlp_to_voxel()
            elif self.opt.skinning_mode == 'voxel':
                self.voxel_to_voxel()

        xc_norm = (xc + self.offset) * self.scale
        
        w = F.grid_sample(self.lbs_voxel_final.expand(xc.shape[0],-1,-1,-1,-1), xc_norm.unsqueeze(2).unsqueeze(2), align_corners=self.opt.align_corners, mode=mode, padding_mode='zeros')
        
        w = w.squeeze(-1).squeeze(-1).permute(0,2,1)
        
        return w
