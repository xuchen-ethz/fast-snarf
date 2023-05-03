import hydra
import torch
import numpy as np
import pytorch_lightning as pl

from lib.model.smpl import SMPLServer
from lib.model.sample import PointOnBones
from lib.model.network import ImplicitNetwork
from lib.model.metrics import calculate_iou
from lib.utils.meshing import generate_mesh, generate_sdf
from lib.model.helpers import masked_softmax, tv_loss, skinning
from lib.utils.render import render_trimesh, render_joint, weights2colors

class SNARFModel(pl.LightningModule):

    def __init__(self, opt, meta_info, data_processor=None):
        super().__init__()

        self.opt = opt

        gender      = str(meta_info['gender'])
        betas       = meta_info['betas'] if 'betas' in meta_info else None
        v_template  = meta_info['v_template'] if 'v_template' in meta_info else None

        self.smpl_server = SMPLServer(gender=gender, betas=betas, v_template=v_template)
        self.smpl_faces  = torch.tensor(self.smpl_server.smpl.faces.astype('int')).unsqueeze(0)
        self.sampler_bone = PointOnBones(self.smpl_server.bone_ids)

        self.network = ImplicitNetwork(**opt.network)
        self.deformer = hydra.utils.instantiate(opt.deformer, smpl_server=self.smpl_server)

        print(self.network)
        print(self.deformer)

        self.data_processor = data_processor

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.parameters(), lr=self.opt.optim.lr)
        
        return optimizer


    def forward(self, pts_d, smpl_tfs, smpl_params, eval_mode=True, calc_time=False):

        # rectify rest pose 
        smpl_tfs = torch.einsum('bnij,njk->bnik', smpl_tfs, self.smpl_server.tfs_c_inv)

        cond = {'smpl': smpl_params[:,7:-10]/np.pi, 'smpl_params': smpl_params}
        
        if calc_time: time = {}
        
        batch_points = 200000

        accum_pred = []
        # split to prevent out of memory
        for pts_d_split in torch.split(pts_d, batch_points, dim=1):

            if calc_time:
                start_tot = torch.cuda.Event(enable_timing=True)
                end_tot = torch.cuda.Event(enable_timing=True)
                start_tot.record()

                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
            # compute canonical correspondences
            pts_c, intermediates = self.deformer(pts_d_split, cond, smpl_tfs, eval_mode=eval_mode)

            if calc_time:
                end.record()
                torch.cuda.synchronize()
                time['time_deformer'] = start.elapsed_time(end)
        
            mask = intermediates['valid_ids']

            # query occuancy in canonical space
            num_batch, num_point, num_init, num_dim = pts_c.shape
            pts_c = pts_c.reshape(num_batch, num_point * num_init, num_dim)

            
            if calc_time:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                time['n_corres_bbox'] = mask[:,:num_point//2].float().sum().item()/(num_batch*num_point/2)
                time['n_corres_surf'] = mask[:,num_point//2:].float().sum().item()/(num_batch*num_point/2)

            occ_pd = self.network(pts_c, cond, mask=mask.reshape(num_batch*num_point * num_init)).reshape(num_batch, num_point, num_init)

            if calc_time:
                end.record()
                torch.cuda.synchronize()
                time['time_shape'] = start.elapsed_time(end)

            if calc_time:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()

            mode = 'softmax' if not eval_mode and self.opt.softmax else 'max'
            occ_pd = masked_softmax(occ_pd, mask, dim=-1, mode=mode, soft_blend=self.opt.soft_blend)
            
            if calc_time:
                end.record()
                torch.cuda.synchronize()
                time['time_agg'] = start.elapsed_time(end)

                end_tot.record()
                torch.cuda.synchronize()
                time['time_tot'] = start_tot.elapsed_time(end_tot)

            accum_pred.append(occ_pd)

        accum_pred = torch.cat(accum_pred, 1)   

        if calc_time:
            return accum_pred, time
        else:
            return accum_pred


    def training_step(self, data, data_idx):

        # Data prep
        if self.data_processor is not None:
            data = self.data_processor.process(data)

        # BCE loss
        occ_pd = self.forward(data['pts_d'], data['smpl_tfs'], data['smpl_params'], eval_mode=False)

        loss = 0

        loss_bce = torch.nn.functional.binary_cross_entropy_with_logits(occ_pd, data['occ_gt'])
        self.log('train_bce', loss_bce)
        loss = loss + loss_bce

        # Bootstrapping
        num_batch = data['pts_d'].shape[0]
        cond = {'smpl': data['smpl_thetas'][:,3:]/np.pi, 'smpl_params': data['smpl_params']}

        if self.deformer.opt.skinning_mode=='voxel':
            loss_tv = self.opt.lambda_tv*tv_loss(self.deformer.lbs_voxel,l2=True)*((self.deformer.opt.res//32)**3)
            self.log('loss_tv', loss_tv)
            loss = loss + loss_tv

        # # Bone occupancy loss
        if self.current_epoch < self.opt.nepochs_pretrain:
            if self.opt.lambda_bone_occ > 0:

                pts_c, occ_gt = self.sampler_bone.get_points(self.smpl_server.joints_c.expand(num_batch, -1, -1))
                occ_pd = self.network(pts_c, cond)

                loss_bone_occ = torch.nn.functional.binary_cross_entropy_with_logits(occ_pd, occ_gt.unsqueeze(-1))

                loss = loss + self.opt.lambda_bone_occ * loss_bone_occ
                self.log('train_bone_occ', loss_bone_occ)

            if self.opt.lambda_bone_w > 0:

                pts_c, w_gt = self.sampler_bone.get_joints(self.smpl_server.joints_c.expand(num_batch, -1, -1))
                w_pd = self.deformer.query_weights(pts_c, cond)
                loss_bone_w = torch.nn.functional.mse_loss(w_pd, w_gt)

                loss = loss + self.opt.lambda_bone_w * loss_bone_w
                self.log('train_bone_w', loss_bone_w)

        return loss
    
    def validation_step(self, data, data_idx):

        if self.data_processor is not None:
            data = self.data_processor.process(data)

        with torch.no_grad():
            if data_idx == 0:
                
                smpl_params = torch.zeros((1, 86),dtype=torch.float32).cuda()
                smpl_params[0, 0] = 1
                smpl_params[0, 6+16*3] = -np.pi / 3
                smpl_params[0, 6+17*3] = np.pi / 3
                
                smpl_plot = self.smpl_server(smpl_params, absolute=True)
                smpl_plot['smpl_thetas'] = smpl_params[:, 4:76]

                smpl_plot['smpl_params'] = smpl_params

                img_all = self.plot(smpl_plot)['img_all']
                self.logger.log_image(key='vis', images=[img_all])

            # return
            
            occ_pd = self.forward(data['pts_d'], data['smpl_tfs'], data['smpl_params'], eval_mode=True)

            _, num_point, _ = data['occ_gt'].shape
            bbox_iou = calculate_iou(data['occ_gt'][:,:num_point//2]>0.5, occ_pd[:,:num_point//2]>0)
            surf_iou = calculate_iou(data['occ_gt'][:,num_point//2:]>0.5, occ_pd[:,num_point//2:]>0)

        return {'bbox_iou':bbox_iou, 'surf_iou':surf_iou}

    def validation_epoch_end(self, validation_step_outputs):

        bbox_ious, surf_ious = [], []
        for output in validation_step_outputs:
            bbox_ious.append(output['bbox_iou'])
            surf_ious.append(output['surf_iou'])
        
        self.log('valid_bbox_iou', torch.stack(bbox_ious).mean())
        self.log('valid_surf_iou', torch.stack(surf_ious).mean())

    def test_step(self, data, data_idx):
            
        with torch.no_grad():

            occ_pd, time_metrics = self.forward(data['pts_d'], data['smpl_tfs'], data['smpl_params'], eval_mode=True, calc_time=True)

            _, num_point, _ = data['occ_gt'].shape
            bbox_iou = calculate_iou(data['occ_gt'][:,:num_point//2]>0.5, occ_pd[:,:num_point//2]>0)
            surf_iou = calculate_iou(data['occ_gt'][:,num_point//2:]>0.5, occ_pd[:,num_point//2:]>0)            
            metrics = {'bbox_iou': bbox_iou.data.cpu().numpy(), 'surf_iou':surf_iou.data.cpu().numpy()}

            metrics.update(time_metrics)

        return metrics
            
    def test_epoch_end(self, test_step_outputs):

        metrics_tot = {}
        
        for output in test_step_outputs:
            for key in output:
                if key not in metrics_tot:
                    metrics_tot[key] = []
                metrics_tot[key].append(output[key])
        
        for key in metrics_tot:
            self.log(key, np.stack(metrics_tot[key]).mean())


    def plot(self, data, res=256, verbose=True, fast_mode=False):


        res_up = np.log2(res//32)

        if verbose:
            surf_pred_cano = self.extract_mesh(self.smpl_server.verts_c, data['smpl_tfs'][[0]], data['smpl_params'][[0]], res_up=res_up, canonical=True, with_weights=True)
            surf_pred_def = self.extract_mesh(data['smpl_verts'][[0]], data['smpl_tfs'][[0]], data['smpl_params'][[0]], res_up=res_up, canonical=False, with_weights=False)

            img_pred_cano = render_trimesh(surf_pred_cano)
            img_pred_def  = render_trimesh(surf_pred_def)
            
            img_joint = render_joint(data['smpl_jnts'].data.cpu().numpy()[0],self.smpl_server.bone_ids)
            img_pred_def[1024:,:,:3] = 255
            img_pred_def[1024:-512,:, :3] = img_joint
            img_pred_def[1024:-512,:, -1] = 255

            results = {
                'img_all': np.concatenate([img_pred_cano, img_pred_def], axis=1),
                'mesh_cano': surf_pred_cano,
                'mesh_def' : surf_pred_def
            }
        else:
            smpl_verts = self.smpl_server.verts_c if fast_mode else data['smpl_verts'][[0]]

            surf_pred_def = self.extract_mesh(smpl_verts, data['smpl_tfs'][[0]], data['smpl_params'][[0]], res_up=res_up, canonical=False, with_weights=False, fast_mode=fast_mode)

            img_pred_def  = render_trimesh(surf_pred_def, mode='p')
            results = {
                'img_all': img_pred_def,
                'mesh_def' : surf_pred_def
            }
        

        return results
    
    def sdf(self, data, res=256, res_up=0):
        smpl_verts = data['smpl_verts'][[0]]
        occ_func = lambda x: self.forward(x, data['smpl_tfs'][[0]], data['smpl_params'][[0]], eval_mode=True).reshape(-1, 1)
        points, values, res = generate_sdf(occ_func, smpl_verts.squeeze(0), res_init=res, res_up=res_up)
        results = {
            'res': res,
            'points': points,
            'values': values,
        }
        
        return results

    def extract_mesh(self, smpl_verts, smpl_tfs, smpl_params, canonical=False, with_weights=False, res_up=2, fast_mode=False):
        '''
        In fast mode, we extract canonical mesh and then forward skin it to posed space.
        This is faster as it bypasses root finding.
        However, it's not deforming the continuous field, but the discrete mesh.
        '''
        if canonical or fast_mode:
            def occ_func(x):
                batch_points = 200000
                acc = []
                for pts_d_split in torch.split(x, batch_points, dim=1):
                    occ = self.network(pts_d_split, {'smpl': smpl_params[:,7:-10]/np.pi})
                    acc.append(occ)
                acc = torch.cat(acc,dim=1).reshape(-1, 1)
                return acc
        else:
            occ_func = lambda x: self.forward(x, smpl_tfs, smpl_params, eval_mode=True).reshape(-1, 1)
            
        mesh = generate_mesh(occ_func, smpl_verts.squeeze(0),res_up=res_up)

        cond = {'smpl': smpl_params[:,7:-10]/np.pi,
                'smpl_params': smpl_params}

        if fast_mode:
            verts  = torch.tensor(mesh.vertices).type_as(smpl_verts)
            weights = self.deformer.query_weights(verts[None], cond).clamp(0,1)[0]

            smpl_tfs = torch.einsum('bnij,njk->bnik', smpl_tfs, self.smpl_server.tfs_c_inv)
            
            verts_mesh_deformed = skinning(verts.unsqueeze(0), weights.unsqueeze(0), smpl_tfs).data.cpu().numpy()[0]
            mesh.vertices = verts_mesh_deformed

        if with_weights:
            verts  = torch.tensor(mesh.vertices).cuda().float()
            weights = self.deformer.query_weights(verts[None], cond).clamp(0,1)[0]*0.999
            mesh.visual.vertex_colors = weights2colors(weights.data.cpu().numpy())

        return mesh
