import numpy as np
import torch
from skimage import measure
from lib.libmise import mise
import trimesh

''' Code adapted from NASA https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/projects/nasa/lib/utils.py'''
def get_mesh_extractor(func, level_set, res_init, res_up, scale, gt_scale, gt_center):
    mesh_extractor = mise.MISE(res_init, res_up, level_set)
    points = mesh_extractor.query()
    
    # query occupancy grid
    with torch.no_grad():
        while points.shape[0] != 0:
            
            orig_points = points
            points = points.astype(np.float32)
            points = (points / mesh_extractor.resolution - 0.5) * scale
            points = points * gt_scale + gt_center
            points = torch.tensor(points).float().cuda()

            values = func(points.unsqueeze(0))[:,0]
            values = values.data.cpu().numpy().astype(np.float64)

            mesh_extractor.update(orig_points, values)
            
            points = mesh_extractor.query()

    return mesh_extractor


def generate_mesh(func, verts, level_set=0, res_init=32, res_up=3):
    scale = 1.1  # Scale of the padded bbox regarding the tight one.

    verts = verts.data.cpu().numpy()
    gt_bbox = np.stack([verts.min(axis=0), verts.max(axis=0)], axis=0)
    gt_center = (gt_bbox[0] + gt_bbox[1]) * 0.5
    gt_scale = (gt_bbox[1] - gt_bbox[0]).max()

    mesh_extractor = get_mesh_extractor(func, level_set, res_init, res_up, scale, gt_scale, gt_center)

    value_grid = mesh_extractor.to_dense()
    # value_grid = np.pad(value_grid, 1, "constant", constant_values=-1e6)

    # marching cube
    verts, faces, normals, values = measure.marching_cubes_lewiner(
                                                volume=value_grid,
                                                gradient_direction='ascent',
                                                level=min(level_set, value_grid.max()))

    verts = (verts / mesh_extractor.resolution - 0.5) * scale
    verts = verts * gt_scale + gt_center

    meshexport = trimesh.Trimesh(verts, faces, normals, vertex_colors=values)

    # remove disconnect part
    connected_comp = meshexport.split(only_watertight=False)
    max_area = 0
    max_comp = None
    for comp in connected_comp:
        if comp.area > max_area:
            max_area = comp.area
            max_comp = comp
    meshexport = max_comp

    return meshexport


def generate_sdf(func, verts, level_set=0, res_init=32, res_up=3):
    scale = 1.1  # Scale of the padded bbox regarding the tight one.

    # Compute sparse grid.
    verts = verts.data.cpu().numpy()
    gt_bbox = np.stack([verts.min(axis=0), verts.max(axis=0)], axis=0)
    gt_center = (gt_bbox[0] + gt_bbox[1]) * 0.5
    gt_scale = (gt_bbox[1] - gt_bbox[0]).max()

    mesh_extractor = get_mesh_extractor(func, level_set, res_init, res_up, scale, gt_scale, gt_center)
    
    # Compute size of the grid.
    res = res_init * (1 << res_up)
    size = res + 1

    # Get sparse points.
    points, values = mesh_extractor.get_points()
    assert np.all(points <= 1024)
    
    # Pack point coordinates in a single int32
    points = points.astype(np.int32)
    points = points[:, 2] * (size * size) + points[:, 1] * size + points[:, 0]

    # Convert SDF values to float32.
    values = values.astype(np.float32)

    return points, values, res