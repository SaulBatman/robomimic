"""
A collection of utilities for working with observation dictionaries and
different kinds of modalities such as images.
"""
import numpy as np
from copy import deepcopy
from collections import OrderedDict
from scipy.spatial.transform import Rotation
import re
import open3d as o3d
import cv2
import copy 

import torch
import torch.nn.functional as F
from pytorch3d.ops import sample_farthest_points


import robomimic.utils.tensor_utils as TU
from robosuite.utils.camera_utils import get_real_depth_map, get_camera_extrinsic_matrix, get_camera_intrinsic_matrix

# MACRO FOR VALID IMAGE CHANNEL SIZES
VALID_IMAGE_CHANNEL_DIMS = {1, 3}       # depth, rgb

# DO NOT MODIFY THIS!
# This keeps track of observation types (modalities) - and is populated on call to @initialize_obs_utils_with_obs_specs.
# This will be a dictionary that maps observation modality (e.g. low_dim, rgb) to a list of observation
# keys under that observation modality.
OBS_MODALITIES_TO_KEYS = None

# DO NOT MODIFY THIS!
# This keeps track of observation types (modalities) - and is populated on call to @initialize_obs_utils_with_obs_specs.
# This will be a dictionary that maps observation keys to their corresponding observation modality
# (e.g. low_dim, rgb)
OBS_KEYS_TO_MODALITIES = None

# DO NOT MODIFY THIS
# This holds the default encoder kwargs that will be used if none are passed at runtime for any given network
DEFAULT_ENCODER_KWARGS = None

# DO NOT MODIFY THIS
# This holds the registered observation modality classes
OBS_MODALITY_CLASSES = {}

# DO NOT MODIFY THIS
# This global dict stores mapping from observation encoder / randomizer network name to class.
# We keep track of these registries to enable automated class inference at runtime, allowing
# users to simply extend our base encoder / randomizer class and refer to that class in string form
# in their config, without having to manually register their class internally.
# This also future-proofs us for any additional encoder / randomizer classes we would
# like to add ourselves.
OBS_ENCODER_CORES = {"None": None}          # Include default None
OBS_RANDOMIZERS = {"None": None}            # Include default None

DEPTH_MINMAX = {'birdview_depth': [1.180, 2.480],
                'agentview_depth': [0.1, 1.1],
                'sideview_depth': [1.0, 2.0],
                'robot0_eye_in_hand_depth': [0., 1.0],
                'sideview2_depth': [0.8, 2.2],
                'backview_depth': [0.6, 1.6],
                'frontview_depth': [1.2, 2.2],
                'spaceview_depth': [0.45, 1.45],
                'spaceview_depth_no_robot': [0.45, 1.45],
                'farspaceview_depth': [0.58, 1.58],
                }
WORKSPACE_MAX = {
                'spaceview': 1.40,
                }

center = np.array([0, 0, 0.7])
WS_SIZE = 0.6
VOXEL_RESO = 64
LOCAL_VOXEL_RESO = 64
BBOX_SIZE_M = 0.3
WORKSPACE = np.array([
    [center[0] - WS_SIZE/2, center[0] + WS_SIZE/2],
    [center[1] - WS_SIZE/2, center[1] + WS_SIZE/2],
    [center[2], center[2] + WS_SIZE]
])
# this is for roughly cropping the point cloud
# we need this because we want inhand voxel to access larger workspace
CLIP_SIZE = 1.0
CLIPSPACE = np.array([
    [center[0] - CLIP_SIZE/2, center[0] + CLIP_SIZE/2],
    [center[1] - CLIP_SIZE/2, center[1] + CLIP_SIZE/2],
    [center[2], center[2] + CLIP_SIZE]
])

def get_clipspace(task_name, table_offset):
    if task_name.startswith('HammerCleanup') or task_name.startswith('Kitchen'):
        ws_size = 0.7 + 0.2
    elif task_name.startswith('PickPlace_'):
        ws_size = 1.1 + 0.2
    else:
        ws_size = CLIP_SIZE
    clipspace =np.array([
                [table_offset[0] - ws_size/2, table_offset[0] + ws_size/2],
                [table_offset[1] - ws_size/2, table_offset[1] + ws_size/2],
                [table_offset[2], table_offset[2] + ws_size]
            ])
    return clipspace

def get_workspace(task_name, table_offset):
    if task_name.startswith('HammerCleanup') or task_name.startswith('Kitchen'):
        ws_size = 0.7
    elif task_name.startswith('PickPlace_'):
        ws_size = 1.1
    else:
        ws_size = WS_SIZE
    workspace = np.array([
                [table_offset[0] - ws_size/2, table_offset[0] + ws_size/2],
                [table_offset[1] - ws_size/2, table_offset[1] + ws_size/2],
                [table_offset[2], table_offset[2] + ws_size]
            ])
    return workspace

def get_pcd_z_min(task_name, table_offset):
    if task_name.startswith('Nut') or task_name.startswith('Square'):
        pcd_z_min = 0.835
    elif task_name.startswith('Kitchen'):
        pcd_z_min = table_offset[2] 
    elif task_name.startswith('HammerCleanup'):
        pcd_z_min = 0.92
    else:
        pcd_z_min = 0.82
    return pcd_z_min

def enlarge_mask(binary_mask, kernel_size, iterations=1):
    """
    Enlarges a binary mask using morphological dilation.

    Returns:
        numpy.ndarray: The enlarged binary mask.
    """

    # --- Parameters for the Mask ---
    image_height, image_width = binary_mask.shape
    binary_mask = binary_mask.astype(np.uint8)

    # 2. Define the structuring element (kernel) for dilation
    # For a square kernel:
    # kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # For a circular kernel (often preferred for smoother enlargement):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    # 3. Perform dilation
    enlarged_mask = cv2.dilate(binary_mask, kernel, iterations=iterations)
    return enlarged_mask.astype(bool)

def voxel_to_rgbd(
    vox: np.ndarray,
    front_is_z0: bool = False,   
) -> np.ndarray:
    """
    Compress a (4, H, W, D) voxel grid (occupancy, R, G, B) into a (4, H, W) RGB-D image.
    The first three channels are the RGB composite and the fourth is the depth.
    Linear interpolation is applied along depth, where the depth value is computed as
    the weighted average of depth layer indices.
    
    Args:
        vox: np.ndarray of shape (4, H, W, D), channels=(occupancy, R, G, B).
             Occupancy is interpreted as per-voxel opacity in [0,1] (will be clipped).
        method:
            - "alpha": front-to-back alpha compositing with occupancy as alpha.
                       Equivalent to rendering onto a black background.
            - "occ_mean": occupancy-weighted average of RGB along depth.
            - "occ_argmax": take RGB at the depth with max occupancy.
        front_is_z0: if True, depth index 0 corresponds to the front; otherwise, reverse.
        normalize_by_coverage (alpha only): if True, divides by the sum of weights so that
            colors are not darkened when total occupancy is small.
        eps: small constant to avoid division by zero.
    
    Returns:
        (4, H, W) float32 RGB-D image, where the first three channels are RGB and the fourth channel
        is the interpolated depth value (in voxel-layer index space).
    """
    if vox.ndim != 4 or vox.shape[0] != 4:
        raise ValueError("Expected input of shape (4, H, W, D).")
    
    _, H, W, D = vox.shape
    occ = np.clip(vox[0], 0.0, 1.0)           # (H, W, D)
    rgb = vox[1:4].astype(np.uint8)         # (3, H, W, D)
    rgb[:,occ == 0] = 255
    
    # If the front is not at index 0, reverse the depth order
    if not front_is_z0:
        occ = occ[..., ::-1]
        rgb = rgb[..., ::-1]
    
    # Initialize accumulators before processing depth layers
    composite_rgb = np.ones((H, W, 3), dtype=np.uint8)*255
    composite_depth = np.ones((H, W), dtype=np.uint8)*255

    # Loop over each depth layer to composite color and interpolate depth
    for d in range(D):
        # Get the per-pixel occupancy for the current depth layer
        curr_occ = occ[..., d].astype(np.uint8)
        curr_occ_large = enlarge_mask((curr_occ==1), kernel_size=7)
        # Move the RGB slice from (3,H,W) to (H,W,3) for proper broadcast
        curr_rgb = np.moveaxis(rgb[..., d], 0, -1)
        # interpolate the non-occupied pixels in rgb_d
        # Inpaint the image using the mask.
        # cv2.INPAINT_NS (Navier-Stokes) or cv2.INPAINT_TELEA (Fast Marching Method) are options.
        if d < 220:
            inpaint_mask = curr_occ_large.astype(np.uint8)*255-curr_occ.astype(np.uint8)*255
            curr_rgb = cv2.inpaint(curr_rgb, inpaint_mask, 7, cv2.INPAINT_NS) # 3 is inpaint radius.
            curr_occ[inpaint_mask.astype(bool)] = D-d
            mask = curr_occ_large > 0
        else:
            mask = curr_occ > 0
        # If there is any occupancy, overwrite the pixel's RGB and depth
        
        composite_rgb[mask] = curr_rgb[mask]
        composite_depth[mask] = D-d

    # Return a (4, H, W) image: three RGB channels and one depth channel
    return np.concatenate([composite_rgb, composite_depth[...,None]], axis=-1).astype(np.uint8)
    
def preprocess_pcd(np_pcd):
    # delete table points
    table_mask = (np_pcd[:, 2] > 0.82)
    ws_mask = (np_pcd[:, 0] > WORKSPACE[0, 0]) * (np_pcd[:, 0] < WORKSPACE[0, 1]) * (np_pcd[:, 1] > WORKSPACE[1, 0]) * (np_pcd[:, 1] < WORKSPACE[1, 1]) * (np_pcd[:, 2] > WORKSPACE[2, 0]) * (np_pcd[:, 2] < WORKSPACE[2, 1])
    # padd fake table points. x and y are grid points in workspace
    num_points = 10000
    fake_table_points = np.zeros((num_points, 6))
    grid_size = int(np.ceil(np.sqrt(num_points)))
    enlarge = 0.8
    grid_x = np.linspace(WORKSPACE[0, 0]-enlarge, WORKSPACE[0, 1]+enlarge, grid_size)
    grid_y = np.linspace(WORKSPACE[1, 0]-enlarge, WORKSPACE[1, 1]+enlarge, grid_size)
    gx, gy = np.meshgrid(grid_x, grid_y)
    gx = gx.flatten()[:num_points]
    gy = gy.flatten()[:num_points]
    fake_table_points[:, 0] = gx
    fake_table_points[:, 1] = gy
    fake_table_points[:, 2] = 0.82
    fake_table_points[:, 3:] = 1.0

    np_pcd = np.concatenate([np_pcd[ws_mask*table_mask], fake_table_points], axis=0)
    return np_pcd

def np2o3d(pcd, color=None):
    # pcd: (n, 3)
    # color: (n, 3)
    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = o3d.utility.Vector3dVector(pcd)
    if color is not None and color.shape[0] > 0:
        assert pcd.shape[0] == color.shape[0]
        assert color.max() <= 1
        assert color.min() >= 0
        pcd_o3d.colors = o3d.utility.Vector3dVector(color)
    return pcd_o3d

def o3d2np(pcd_o3d, num_samples=4412):
    # pcd: (n, 3)
    # color: (n, 3)
    xyz = np.asarray(pcd_o3d.points)
    rgb = np.asarray(pcd_o3d.colors)
    num_points = xyz.shape[0]

    pcd_np = np.concatenate([xyz, rgb], axis=1)

    return pcd_np

def depth2fgpcd(depth, mask, cam_params):
    # depth: (h, w)
    # fgpcd: (n, 3)
    # mask: (h, w)
    h, w = depth.shape
    mask = np.logical_and(mask, depth > 0)
    # mask = (depth <= 0.599/0.8)
    fgpcd = np.zeros((mask.sum(), 3))
    fx, fy, cx, cy = cam_params
    pos_x, pos_y = np.meshgrid(np.arange(w), np.arange(h))
    pos_x = pos_x[mask]
    pos_y = pos_y[mask]
    fgpcd[:, 0] = (pos_x - cx) * depth[mask] / fx
    fgpcd[:, 1] = (pos_y - cy) * depth[mask] / fy
    fgpcd[:, 2] = depth[mask]
    return fgpcd

def rgbd2pcd(rgb, depth, intrinsics, extrinsics):
    cam_param = [intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]]
    mask = np.ones_like(depth, dtype=bool)
    pcd = depth2fgpcd(depth, mask, cam_param)                
    trans_pcd = np.einsum('ij,jk->ik', extrinsics, np.concatenate([pcd.T, np.ones((1, pcd.shape[0]))], axis=0))
    trans_pcd = trans_pcd[:3, :].T

    mask = (trans_pcd[:, 0] > WORKSPACE[0, 0]) * (trans_pcd[:, 0] < WORKSPACE[0, 1]) * (trans_pcd[:, 1] > WORKSPACE[1, 0]) * (trans_pcd[:, 1] < WORKSPACE[1, 1]) * (trans_pcd[:, 2] > WORKSPACE[2, 0]) * (trans_pcd[:, 2] < WORKSPACE[2, 1])
    pcd_WRT_world = np.concatenate([trans_pcd[mask], rgb.reshape(-1, 3)[mask].astype(np.float64) / 255], axis=1)
    return pcd_WRT_world

def clip_depth(raw_depth, depth_key):
    depth_min, depth_max = DEPTH_MINMAX[depth_key]
    if isinstance(raw_depth, torch.Tensor):
        raw_depth = torch.clip(raw_depth, depth_min, depth_max)
    elif isinstance(raw_depth, np.ndarray):
        raw_depth = np.clip(raw_depth, depth_min, depth_max)
    else:
        NotImplementedError
    return raw_depth

def normalize_depth(depth, camera_name):
    depth_min, depth_max = DEPTH_MINMAX[camera_name]
    depth = np.clip(depth, depth_min, depth_max)
    return (depth - depth_min) / (depth_max - depth_min)

def unnormalize_depth(depth, camera_name):
    depth_min, depth_max = DEPTH_MINMAX[camera_name]
    return depth * (depth_max - depth_min) + depth_min

def convert_rgbd_to_pcd_batch(rgbd_images, camera_intrinsic, camera_extrinsic, gripper_centric):
    """
    Convert a batch of RGBD images to point cloud data (PCD).

    Parameters:
    - rgbd_images: array of shape (batch_size, height, width, 4), where the last dimension is [R, G, B, D]
    - camera_intrinsic: 3x3 list representing camera intrinsic matrix
    - camera_extrinsic: 4x4 list representing camera extrinsic matrix
    - gripper_centric: boolean, if True, convert to gripper-centric coordinates

    Returns:
    - pcd: array of shape (batch_size, height * width, 6), point cloud data
    """
    batch_size, _, height, width = rgbd_images.shape
    rgb_images = rgbd_images[:, :3]
    depth_images = rgbd_images[:, 3]

    # Create meshgrid for pixel coordinates
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    x = x.reshape(-1)
    y = y.reshape(-1)

    # Convert depth to point cloud
    fx, fy = camera_intrinsic[0][0], camera_intrinsic[1][1]
    cx, cy = camera_intrinsic[0][2], camera_intrinsic[1][2]

    pcd = np.zeros((batch_size, height * width, 6))
    for i in range(batch_size):
        depth = depth_images[i].reshape(-1)
        pcd[i, :, 0] = (x - cx) * depth / fx
        pcd[i, :, 1] = (y - cy) * depth / fy
        pcd[i, :, 2] = depth_images[i].reshape(-1)
        pcd[i, :, 3:] = rgb_images[i].reshape(-1)

    # Apply extrinsic transformation
    if gripper_centric:
        pcd_homogeneous = np.concatenate([pcd, np.ones((batch_size, height * width, 1))], axis=-1)
        pcd_transformed = np.einsum('ij,bkj->bki', camera_extrinsic, pcd_homogeneous)
        pcd = pcd_transformed[..., :3] / pcd_transformed[..., 3:]

    return pcd

def match_multi_camera_name(pattern: str) -> bool:
    # assuming that all multicam depths are given names like "camera{NUM}_depth"
    # Regular expression to match the pattern "camera{some int number}_depth"
    regex = r'^camera\d+_depth$'
    return bool(re.match(regex, pattern))

def discretize_depth(depth: np.array, cam_name: str, depth_minmax=None):
    # input a true depth and discretize to [0,255]
    # depths.shape = (N, H, W, C)
    if depth_minmax == None:
        if cam_name in DEPTH_MINMAX.keys():
            depth_minmax = DEPTH_MINMAX[cam_name]
        elif match_multi_camera_name(cam_name): 
            depth_minmax = [depth.min(), depth.max()]
        else:
            assert True, "you need add depth_minmax in obs_utils.py"
    minmax_range = depth_minmax[1] - depth_minmax[0]
    ndepths =(np.clip(depth, depth_minmax[0], depth_minmax[1]) - depth_minmax[0]) / minmax_range * 255
    return ndepths.astype(np.uint8)

def undiscretize_depth(depth, cam_name, depth_minmax=None):
    # input a true depth and discretize to [0,255]
    # depths.shape = (N, H, W, C)
    if depth_minmax == None:
        assert cam_name in DEPTH_MINMAX.keys(), "you need add depth_minmax in env_robosuite.py"
        depth_minmax = DEPTH_MINMAX[cam_name]
    minmax_range = depth_minmax[1] - depth_minmax[0]
    ndepths = depth / 255. * minmax_range + depth_minmax[0]
    return ndepths.astype(np.float64)


def xyz_to_bbox_center_batch(xyz_points, camera_intrinsic, camera_extrinsic):
    """
    Convert xyz points to bbox centers given camera extrinsic matrix.
    
    Parameters:
    - xyz_points: array of shape (batch_size, x, y, z) in world frame
    - camera_intrinsic: 3x3 list representing camera intrinsic in world frame
    - camera_extrinsic: 4x4 list representing camera extrinsic in world frame
    
    Returns:
    - bb_centers: array of bounding box centers (x_center, y_center) for each image
    """
    camera_intrinsic, camera_extrinsic = np.asarray(camera_intrinsic).astype(np.float64), np.asarray(camera_extrinsic).astype(np.float64)
    camera_extrinsic_inv, camera_extrinsic_rot, camera_extrinsic_tran = np.eye(4), camera_extrinsic[:3,:3], camera_extrinsic[:3,-1]
    camera_extrinsic_inv[:3,:3] = camera_extrinsic_rot.T
    camera_extrinsic_inv[:3,-1] = -camera_extrinsic_rot.T @ camera_extrinsic_tran
    xyz_world = np.concatenate([xyz_points, np.ones(xyz_points.shape[0])[...,None]], axis=1)
    xyz_camera = camera_extrinsic_inv @ xyz_world.T
    xyz_camera = xyz_camera.T
    xyz_camera = xyz_camera[:,:3] / xyz_camera[:,3:4]
    uv_camera = camera_intrinsic @ xyz_camera.T
    uv_camera = uv_camera.T
    bb_centers = uv_camera[:,:2] / uv_camera[:,2:3]
    return bb_centers.astype(int)

def crop_and_pad_batch(images, bb_centers, output_size):
    """
    Crop and pad a batch of images based on given coordinates.
    
    Parameters:
    - images: array of shape (batch_size, height, width, channels), row is y and column is x
    - bb_centers: array of bounding box centers (x_center, y_center) for each image
    - output_size: Tuple (out_height, out_width), the desired size of the output crop
    
    Returns:
    - Tensor of shape (batch_size, channels, out_height, out_width)
    """
    batch_size, height, width, channels = images.shape
    if channels == 4:
        pad_with = (((0, 255, 0, 255), (0, 255, 0, 255)), ((0, 255, 0, 255), (0, 255, 0, 255)), ((0, 0))) # pad with green and 255 depth
    else:
        pad_with = (((0, 255, 0), (0, 255, 0)), ((0, 255, 0), (0, 255, 0)), ((0, 0))) # pad with green and 255 depth
    out_height, out_width = output_size
    image_pad_height = out_height // 2
    image_pad_width = out_width // 2

    output_size_half = np.array(output_size)[None, ...]//2
    output_size_half = output_size_half
    top_lefts = bb_centers - output_size_half
    bottom_rights = bb_centers + output_size_half

    cropped_images = []
    
    for img, top_left, bottom_right in zip(images, top_lefts, bottom_rights):
        # pad_with = ((0, 255), (0, 255), (0, 255))
        padded_img = np.pad(img, ((image_pad_height,image_pad_height),(image_pad_width,image_pad_width),(0,0)), mode='constant',constant_values=np.array(pad_with, dtype=object))

        x1, y1 = top_left + image_pad_height
        x2, y2 = bottom_right + image_pad_width
        
        # Crop the image
        cropped_img = padded_img[y1:y2, x1:x2, :]
        
        
        cropped_images.append(cropped_img)
    
    return np.stack(cropped_images)

def mask_batch(images, bb_centers, output_size):
    """
    Mask out the areas that are away from the gripper on a batch of images based on given coordinates.
    
    Parameters:
    - images: array of shape (batch_size, height, width, channels), row is y and column is x
    - bb_centers: array of bounding box centers (x_center, y_center) for each image
    - output_size: Tuple (out_height, out_width), the desired size of the output crop
    
    Returns:
    - Tensor of shape (batch_size, channels, out_height, out_width)
    """
    batch_size, height, width, channels = images.shape
    out_height, out_width = output_size
    image_pad_height = out_height // 2
    image_pad_width = out_width // 2

    output_size_half = np.array(output_size)[None, ...]//2
    output_size_half = output_size_half
    top_lefts = bb_centers - output_size_half
    bottom_rights = bb_centers + output_size_half

    masked_images = []
    
    for img, top_left, bottom_right in zip(images, top_lefts, bottom_rights):
        mask = np.zeros([height, width])
        padded_mask = np.pad(mask, ((image_pad_height,image_pad_height),(image_pad_width,image_pad_width)))

        x1, y1 = top_left + image_pad_height
        x2, y2 = bottom_right + image_pad_width
        
        # assign ones to the pixels around the gripper
        padded_mask[y1:y2, x1:x2] = 1
        mask = padded_mask[image_pad_height:-image_pad_height, image_pad_width:-image_pad_width].astype(bool)

        img[~mask, :3] = np.array([0, 255, 0])
        if channels == 4:
            img[~mask, 3] = 1
        masked_images.append(img)

    return np.stack(masked_images)


def clip_depth_alone_gripper_x_batch(goal_key, rgbd_images, gripper_pose, camera_intrinsic, camera_extrinsic, depth_normalization_around_gripper, x_range=0.30):
    """
    Clip depths to be centered at gripper x axis for a batch of raw RGBD images.
    
    Parameters:
    - rgbds: array of uint8 images in shape (batch_size, height, width, channels), row is y and column is x
    - gripper_pose: array of 3d positions of shape (batch_size, 3) in world frame
    - camera_extrinsic: list of a 4x4 transformation matrix of the camera pose in world frame
    - x_range: the depth range for clipping depth on x-axis. Default 0.3m. 
    
    Returns:
    - array of images of shape (batch_size, height, width, channels), value range is [0, x_range]
    """
    camera_extrinsic = np.asarray(camera_extrinsic)
    camera_intrinsic = np.asarray(camera_intrinsic)
    rgbd_images = rgbd_images.astype(float)
    x_range_half = x_range / 2
    batch_size, height, width, channels = rgbd_images.shape
    
    pcd_rt_cam = depth2fgpcd(rgbd_images[...,-1], camera_intrinsic)
    pcd_rt_cam = np.concatenate([pcd_rt_cam, np.ones([batch_size, pcd_rt_cam.shape[1], 1])], axis=2)
    pcd_rt_world = np.einsum('ij,jkl->ikl', camera_extrinsic, pcd_rt_cam.T).T
    pcd_rt_world = pcd_rt_world[...,:3] / pcd_rt_world[...,-1:]

    clip_near_gripper=False
    if clip_near_gripper:
        pcd_mask_far = pcd_rt_world[..., 0] > (gripper_pose[..., 0:1] - x_range_half)
        pcd_mask_close = pcd_rt_world[..., 0] < (gripper_pose[..., 0:1] + x_range_half)
        # pcd_mask = pcd_mask_far * pcd_mask_close
        pcd_mask = pcd_mask_far
    else:
        pcd_mask = pcd_rt_world[..., 0] < WORKSPACE_MAX[goal_key.split('_')[0]]

    pcd_mask = pcd_mask.reshape(batch_size, height, width)
    rgbd_images[~pcd_mask,:3] = np.array([0.,255.,0.])

    if depth_normalization_around_gripper:
        pcd_mask_close, pcd_mask_far = pcd_mask_close.reshape(batch_size, height, width), pcd_mask_far.reshape(batch_size, height, width)
        camera_tilt_angle = Rotation.from_matrix(camera_extrinsic[:3,:3]).as_euler('ZYX')[2] + np.pi / 2
        # center depth at 
        camera_clip_range = x_range_half / np.cos(camera_tilt_angle)
        rgbd_images[~pcd_mask_close, 3] = -camera_clip_range
        rgbd_images[~pcd_mask_far, 3] = camera_clip_range
        x_distance_camera_gripper = camera_extrinsic[0, 3] - gripper_pose[..., 0]
        rgbd_images[..., 3] = rgbd_images[..., 3] - x_distance_camera_gripper[:,None,None] / np.cos(camera_tilt_angle)
        depth_min, depth_max = -camera_clip_range, camera_clip_range
        rgbd_images[...,3] = (rgbd_images[...,3] - depth_min) / (depth_max - depth_min) * 255
    else:
        rgbd_images[~pcd_mask, 3] = DEPTH_MINMAX[goal_key.split('_')[0]+'_depth'][1]
        rgbd_images[...,3] = discretize_depth(rgbd_images[...,3], goal_key.split('_')[0]+'_depth')

    
    # x_distance_camera_gripper = camera_extrinsic[0, 3] - gripper_pose[..., 0]
    # rgbd_images[..., -1] = rgbd_images[..., -1] - x_distance_camera_gripper[:, None, None] # center depth at gripper x
    # image_mask = (rgbd_images[..., -1] > -x_range_half) * (rgbd_images[..., -1] < x_range_half)
    # rgbd_images[~image_mask, :3] = np.array([0.,0.,0.])
    # rgbd_images[..., 3] = np.clip(rgbd_images[...,3], -x_range_half, x_range_half)
    return rgbd_images

def convert_sideview_to_gripper_batch(sim, images, goal_key, robot0_eef_pos, camera_info=None, bbox_size = (40, 40), centering_method='crop', silence=False):
    """
    Clip depths to be centered at gripper x axis for a batch of raw RGBD images.
    
    Parameters:
    - sim: robosuite env class, for getting camera intrinsic and extrinsic
    - images: array of uint8 images in shape (batch_size, height, width, channels), row is y and column is x
    - goal_key: image key, e.g., frontview_gripper_image or frontview_gripper_rgbd
    - robot0_eef_pos: array of 3d positions of shape (batch_size, 3) in world frame
    - camera_info: dict contains 'intrinsic' and 'extrinsic' which is lists of 4x4 matrices
    
    Returns:
    - array of images of shape (batch_size, height, width, channels), np.uint8
    """
    camera_name = goal_key.split('_')[0]
    b, h, w, c = images.shape
    assert h == 128 and w == 128, f"GRIPPER OBS CONVERSION ERROR: image HW {images.shape} is incorrect"
    assert c == 3 or c == 4, f"GRIPPER OBS CONVERSION ERROR: image  {images.shape} is incorrect"
    if camera_info is None:
        assert sim is not None
        extrinsic = get_camera_extrinsic_matrix(sim, camera_name).tolist()
        intrinsic = get_camera_intrinsic_matrix(sim, camera_name, h, w).tolist()
    else:
        extrinsic = camera_info['extrinsic']
        intrinsic = camera_info['intrinsic']

    depth_range = 0.35
    depth_normalization_around_gripper = False
    raw_image = images.copy().astype(float)

    
    # clip the pixels that are behind the gripper
    if 'rgbd' in goal_key:
        raw_image[...,3] = undiscretize_depth(raw_image[...,3], goal_key.split('_')[0]+'_depth')
    if 'rgbd' in goal_key:
        raw_image = clip_depth_alone_gripper_x_batch(goal_key, raw_image, robot0_eef_pos, intrinsic, extrinsic, depth_normalization_around_gripper, x_range=depth_range)
    
    # convert eef pos in world coor to pixel locations in image coor
    bbox_centers = xyz_to_bbox_center_batch(robot0_eef_pos, intrinsic, extrinsic)
    bbox_center_in_image = np.clip(bbox_centers, 0, np.array([h,w]))
    if not (bbox_center_in_image == bbox_centers).all() and not silence:
        print("WARNING: End-effector center is not in observation. Trying clipping.")

    # extracting gripper centric images
    if centering_method == 'crop':
        gripper_centered_current_obs = crop_and_pad_batch(raw_image, bbox_center_in_image, output_size=bbox_size)
    elif centering_method == 'mask':
        gripper_centered_current_obs = mask_batch(raw_image, bbox_center_in_image, output_size=bbox_size)
    else:
        assert centering_method in ['crop', 'mask'], f"Only support [crop, mask], you are trying to use {centering_method}"

    gripper_centered_current_obs = F.interpolate(torch.from_numpy(gripper_centered_current_obs.astype(np.uint8)).permute(0, 3, 1, 2), (h, w), mode='bilinear').permute(0, 2, 3, 1).numpy()
    return gripper_centered_current_obs, bbox_center_in_image

def populate_point_num(pcd, point_num):
    if pcd.shape[0] == 0:
        pcd = np.zeros((point_num, 6))
    elif 0 < pcd.shape[0] < point_num:
        extra_choice = np.random.choice(pcd.shape[0], point_num-pcd.shape[0], replace=True)
        pcd = np.concatenate([pcd, pcd[extra_choice]], axis=0)
    else:
        shuffle_idx = np.random.permutation(pcd.shape[0])[:point_num]
        pcd = pcd[shuffle_idx]
    return pcd

# def pcd_to_voxel(pcds: np.ndarray, gripper_crop: float=None, voxel_size: float = 0.01):
#     assert pcds.shape[2] == 6, "PCD CONVERSION ERROR: pcd shape is incorrect"
#     assert (pcds[0, :,3:6] <= 1.).all(), "PCD CONVERSION ERROR: pcd color is incorrect"
#     # pcd: (n, 6)
#     voxels = []
#     if gripper_crop is None:
#         voxel_bound = WORKSPACE.T
#     else:
#         voxel_bound = np.array([
#             [-gripper_crop, gripper_crop],
#             [-gripper_crop, gripper_crop],
#             [-gripper_crop, gripper_crop]
#         ]).T
        
#     for pcd in pcds:
#         p = o3d.geometry.PointCloud()
#         p.points = o3d.utility.Vector3dVector(pcd[:,:3])
#         p.colors = o3d.utility.Vector3dVector(pcd[:,3:6])
#         voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud_within_bounds(p, voxel_size=voxel_size, min_bound=voxel_bound[0], max_bound=voxel_bound[1])
#         voxel = voxel_grid.get_voxels()  # returns list of voxels
#         if len(voxel) == 0:
#             np_voxels = np.zeros([4, VOXEL_RESO, VOXEL_RESO, VOXEL_RESO], dtype=np.float32)
#         else:
#             indices = np.stack(list(vx.grid_index for vx in voxel))
#             colors = np.stack(list(vx.color for vx in voxel))

#             mask = (indices > 0) * (indices < VOXEL_RESO)
#             indices = indices[mask.all(axis=1)]
#             colors = colors[mask.all(axis=1)]

#             np_voxels = np.zeros([4, VOXEL_RESO, VOXEL_RESO, VOXEL_RESO], dtype=np.float32)
#             np_voxels[0, indices[:, 0], indices[:, 1], indices[:, 2]] = 1
#             np_voxels[1:, indices[:, 0], indices[:, 1], indices[:, 2]] = colors.T

#             # np_voxels = np.moveaxis(np_voxels, [0, 1, 2, 3], [0, 3, 2, 1])
#             # np_voxels = np.flip(np_voxels, (1, 2))
#         voxels.append(np_voxels)
#     return np.stack(voxels)

def sample_pcd(pcd: np.ndarray, num_points: int = 1024):
    if pcd.shape[0] > num_points:
        pcd_o3d = np2o3d(pcd[:, :3], pcd[:, 3:6])
        pcd_o3d = pcd_o3d.farthest_point_down_sample(num_samples=num_points)
        xyz = np.asarray(pcd_o3d.points)
        rgb = np.asarray(pcd_o3d.colors)
        pcd_np = np.concatenate([xyz, rgb], axis=1)
    else:
        pcd_np = populate_point_num(pcd, num_points)
    return pcd_np

def crop_pcd(pcd: np.ndarray, input_type: str = 'absolute', num_points: int = 1024):
    assert pcd.shape[1] == 6, "PCD CONVERSION ERROR: pcd shape is incorrect"
    assert (pcd[0, 3:6] <= 1.).all(), "PCD CONVERSION ERROR: pcd color is incorrect"
    assert input_type in ['absolute', 'relative', 'gripper'], "PCD CONVERSION ERROR: input_type should be absolute or gripper_crop"

    # Define voxel bounds
    if input_type == 'gripper':
        local_size = BBOX_SIZE_M / 2
        x_offset = 0.
        # z_offset = 0.05 if input_type=='gripper_se3' else 0.# because we dont need to see the entire gripper in the voxel grid
        z_offset = 0.
        bound = np.array([
            [-local_size+x_offset, local_size+x_offset],
            [-local_size, local_size],
            [-local_size+z_offset, local_size+z_offset]
        ])
    elif input_type == 'relative':
        ws_size = WS_SIZE * 2
        bound = np.array([
            [-ws_size/2, ws_size/2],
            [-ws_size/2, ws_size/2],
            [-ws_size/2, ws_size/2]
        ])
    else:
        bound = WORKSPACE

    pcd = copy.deepcopy(pcd)
    mask = (pcd[:, 0] >= bound[0, 0]) * (pcd[:, 0] <= bound[0, 1]) * (pcd[:, 1] >= bound[1, 0]) * (pcd[:, 1] <= bound[1, 1]) * (pcd[:, 2] >= bound[2, 0]) * (pcd[:, 2] <= bound[2, 1])
    pcd = pcd[mask]
    return sample_pcd(pcd, num_points)

def pcd_to_voxel(pcds: np.ndarray, input_type: str = 'absolute'):
    assert pcds.shape[2] == 6, "PCD CONVERSION ERROR: pcd shape is incorrect"
    assert (pcds[0, :, 3:6] <= 1.).all(), "PCD CONVERSION ERROR: pcd color is incorrect"
    assert input_type in ['absolute', 'relative', 'gripper'], "PCD CONVERSION ERROR: input_type should be absolute or gripper_crop"

    # Define voxel bounds
    if input_type == 'gripper':
        local_size = BBOX_SIZE_M / 2
        x_offset = 0.
        # z_offset = 0.05 if input_type=='gripper_se3' else 0.# because we dont need to see the entire gripper in the voxel grid
        z_offset = 0.
        voxel_bound = np.array([
            [-local_size+x_offset, local_size+x_offset],
            [-local_size, local_size],
            [-local_size+z_offset, local_size+z_offset]
        ]).T
        voxel_size = BBOX_SIZE_M/LOCAL_VOXEL_RESO
    elif input_type == 'gripper_se3':
        local_size = BBOX_SIZE_M / 2
        x_offset = 0.
        z_offset = 0.08 # because we dont need to see the entire gripper in the voxel grid
        bound = np.array([
            [-local_size+x_offset, local_size+x_offset],
            [-local_size, local_size],
            [-local_size+z_offset, local_size+z_offset]
        ])
    elif input_type == 'relative':
        ws_size = WS_SIZE * 2
        voxel_bound = np.array([
            [-ws_size/2, ws_size/2],
            [-ws_size/2, ws_size/2],
            [-ws_size/2, ws_size/2]
        ]).T
        voxel_size = ws_size / VOXEL_RESO
    else:
        voxel_bound = WORKSPACE.T
        voxel_size = WS_SIZE / VOXEL_RESO

    # Precompute voxel grid dimensions
    grid_min = voxel_bound[0]
    grid_max = voxel_bound[1]
    grid_size = ((grid_max - grid_min) / voxel_size  + 1e-4).astype(int)
    voxel_reso = grid_size[0]
    grid_size = np.clip(grid_size, 0, VOXEL_RESO)
    assert voxel_reso in [32, 64, 84, 128], f"Voxel resolution {voxel_reso} is not supported. Supported resolutions are 32, 64, and 128."

    # # Preallocate voxel array
    # batch_voxels = np.zeros((len(pcds), 4, VOXEL_RESO, VOXEL_RESO, VOXEL_RESO), dtype=np.float32)

    # for i, pcd in enumerate(pcds):
    #     # Filter points within bounds
    #     mask = np.all((pcd[:, :3] >= grid_min) & (pcd[:, :3] < grid_max), axis=1)
    #     pcd = pcd[mask]

    #     if len(pcd) == 0:
    #         continue

    #     # Compute voxel indices
    #     indices = ((pcd[:, :3] - grid_min) / voxel_size).astype(int)
    #     indices = np.clip(indices, 0, VOXEL_RESO - 1)

    #     # Aggregate voxel occupancy and color
    #     batch_voxels[i, 0, indices[:, 0], indices[:, 1], indices[:, 2]] = 1  # Occupancy
    #     batch_voxels[i, 1:, indices[:, 0], indices[:, 1], indices[:, 2]] = pcd[:, 3:6]  # Colors

    batch_voxels= []
    for i, pcd in enumerate(pcds):
        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud_within_bounds(np2o3d(pcd[:,:3], pcd[:,3:]), voxel_size=voxel_size+1e-4, min_bound=voxel_bound[0], max_bound=voxel_bound[1])
        voxels = voxel_grid.get_voxels()  # returns list of voxels
        if len(voxels) == 0:
            np_voxels = np.zeros([4, voxel_reso, voxel_reso, voxel_reso], dtype=np.uint8)
        else:
            indices = np.stack(list(vx.grid_index for vx in voxels))
            colors = np.stack(list(vx.color for vx in voxels))

            mask = (indices > 0) * (indices < voxel_reso)
            indices = indices[mask.all(axis=1)]
            colors = colors[mask.all(axis=1)]

            np_voxels = np.zeros([4, voxel_reso, voxel_reso, voxel_reso], dtype=np.uint8)
            np_voxels[0, indices[:, 0], indices[:, 1], indices[:, 2]] = 1
            np_voxels[1:, indices[:, 0], indices[:, 1], indices[:, 2]] = colors.T * 255
        
        batch_voxels.append(np_voxels)
    batch_voxels = np.stack(batch_voxels)
    return batch_voxels

def transform_pcd_to_ee_frame(points_world, T_W_e_xyzqxyzw, local_type):
    """
    Transform a set of points from the world coordinate system to the 'e' coordinate system.
    
    Args:
        points_world (numpy.ndarray): An (N, 3) array of points in the world coordinate system.
        T_W_e (numpy.ndarray): A (7,) transformation matrix that transforms points 
                               from frame e to the world frame (i.e., {}^W T_e).
    
    Returns:
        numpy.ndarray: An (N, 3) array of points expressed in the 'e' coordinate system.
    """
    assert local_type in ['xyz', 'xyz_goal', 'se3'] 
    # Construct T_W_e mat
    T_W_e = np.eye(4)
    if local_type == 'se3':
        T_W_e[:3,:3] = Rotation.from_quat(T_W_e_xyzqxyzw[3:]).as_matrix()
        T_W_e[:3, 3] = T_W_e_xyzqxyzw[:3]
        # Convert points from world coordinates to homogeneous coordinates
        num_points = points_world.shape[0]
        points_homogeneous = np.hstack([points_world, np.ones((num_points, 1))])

        # Apply the inverse transformation: np.linalg.inv(T_W_e)
        transformed_homogeneous = (np.linalg.inv(T_W_e) @ points_homogeneous.T).T

        # Convert back to Cartesian coordinates (drop the homogeneous coordinate)
        points_e = transformed_homogeneous[:, :3]
        return points_e
    elif 'xyz' in local_type:
        points_e = deepcopy(points_world)
        points_e = points_e - T_W_e_xyzqxyzw[:3]
        return points_e
    else:
        raise ValueError(f"Unknown local_type: {local_type}. Expected 'xyz' or 'se3'.")

def localize_pcd_batch(pcds, previous_eef_poses, local_type='xyz'):
    """
    Clip depths to be centered at gripper x axis for a batch of raw RGBD images.
    
    Parameters:
    - sim: robosuite env class, for getting camera intrinsic and extrinsic
    - pcd: array of float point cloud in shape (batch_size, height, width, channels), row is y and column is x
    - goal_key: image key, e.g., frontview_gripper_image or frontview_gripper_rgbd
    - robot0_eef_poss: array of 3d positions of shape (batch_size, 3) in world frame
    
    Returns:
    - array of images of shape (batch_size, height, width, channels), np.uint8
    """
    pcds = copy.deepcopy(pcds)
    is_batch = False
    if len(pcds.shape) == 4:
        is_batch = True
        n_envs, n_pcd, num_points, dim = pcds.shape
        pcds = pcds.reshape(n_envs * n_pcd, num_points, dim)
        repeat_poses = np.repeat(previous_eef_poses, n_pcd, axis=0)
    else:
        n_pcd, num_points, dim = pcds.shape
        repeat_poses = np.repeat(previous_eef_poses[None], n_pcd, axis=0)

    assert pcds.shape[-1] == 6 and len(pcds.shape) == 3, f"PCD CONVERSION ERROR: pcd shape {pcds.shape} is incorrect"

    for i in range(pcds.shape[0]):
        pcds[i,:,:3] = transform_pcd_to_ee_frame(pcds[i,:,:3], repeat_poses[i], local_type)
    
    if is_batch:
        pcds = pcds.reshape(n_envs, n_pcd, num_points, dim)
    return pcds.astype(np.float32)

def populate_pcd_batch(pcds, point_num):
    processed_pcds = []
    for i, pcd in enumerate(pcds):
        p = populate_point_num(pcd, point_num) 
        processed_pcds.append(p)
    return np.stack(processed_pcds)

def crop_local_pcd(pcd, gripper_pos, bbox_size_m, fix_point_num=1024, crop_method='cube'):
    if crop_method == 'sphere':
        distances = np.linalg.norm(pcd[:, :3] - gripper_pos[:3], axis=1)
        mask = distances <= bbox_size_m / 2
    elif crop_method == 'cube':
        mask_x = (pcd[..., 0] > (gripper_pos[..., 0] - bbox_size_m / 2)) & (pcd[..., 0] < (gripper_pos[..., 0] + bbox_size_m / 2))
        mask_y = (pcd[..., 1] > (gripper_pos[..., 1] - bbox_size_m / 2)) & (pcd[..., 1] < (gripper_pos[..., 1] + bbox_size_m / 2))
        mask_z = (pcd[..., 2] > (gripper_pos[..., 2] - bbox_size_m / 2)) & (pcd[..., 2] < (gripper_pos[..., 2] + bbox_size_m / 2))
        mask = mask_x & mask_y & mask_z 
        # mask_table = (pcd[..., 2] > 0.815) 
        # mask = mask & mask_table
    else:
        raise ValueError(f"Unknown crop method: {crop_method}")
    
    if mask.sum() > 50:
        p = populate_point_num(pcd[mask], fix_point_num) 
        is_empty = False
    else: 
        p = np.repeat(np.zeros([1,6]), fix_point_num, axis=0) 
        is_empty = True
    return p, is_empty

def crop_local_pcd_batch(pcds, gripper_pose, bbox_size_m=0.3, fix_point_num=1024, crop_method='cube'):
    """
    Clip depths to be centered at gripper x axis for a batch of raw RGBD images.
    
    Parameters:
    - pcd: array of float point cloud in shape (batch_size, height, width, channels), row is y and column is x
    - robot0_eef_poss: array of 3d positions of shape (batch_size, 3) in world frame
    
    Returns:
    - array of images of shape (batch_size, height, width, channels), np.uint8
    """
    is_batch = False
    if len(pcds.shape) == 4:
        is_batch = True
        n_envs, n_pcd, num_points, dim = pcds.shape
        pcds = pcds.reshape(n_envs * n_pcd, num_points, dim)
        gripper_pose = np.repeat(gripper_pose[:,None], n_pcd, axis=1)
        gripper_pose = gripper_pose.reshape(n_envs * n_pcd, -1)
    else:
        n_pcd, num_points, dim = pcds.shape

    assert pcds.shape[-1] == 6 and len(pcds.shape) == 3, f"PCD CONVERSION ERROR: pcd shape {pcds.shape} is incorrect"
    
    if gripper_pose.shape[0] == 1:
        gripper_pose = np.repeat(gripper_pose, pcds.shape[0], axis=0)
    
    processed_pcds = []
    is_emptys = []
    for i, (pcd, pos) in enumerate(zip(pcds, gripper_pose)):
        p, is_emp = crop_local_pcd(pcd, pos, bbox_size_m, fix_point_num, crop_method=crop_method)
        processed_pcds.append(p)
        is_emptys.append(is_emp)

    pcd = np.stack(processed_pcds)
    is_emptys = np.array(is_emptys)
    if is_batch:
        pcd = pcd.reshape(n_envs, n_pcd, fix_point_num, dim)
        is_emptys = is_emptys.reshape(n_envs, n_pcd)
    return pcd, is_emptys

def get_temporal_obs_and_goal_pcd(obs, goal):
    current_pcd = np.concatenate([obs, np.zeros((*obs.shape[:-1],) +(1,))], axis=-1)
    goal_pcd = np.concatenate([goal, np.ones((*goal.shape[:-1],) +(1,))], axis=-1)
    axis_ind = 0 if len(obs.shape) == 3 else 1
    return np.concatenate([current_pcd, goal_pcd], axis=axis_ind)

def register_obs_key(target_class):
    assert target_class not in OBS_MODALITY_CLASSES, f"Already registered modality {target_class}!"
    OBS_MODALITY_CLASSES[target_class.name] = target_class


def register_encoder_core(target_class):
    assert target_class not in OBS_ENCODER_CORES, f"Already registered obs encoder core {target_class}!"
    OBS_ENCODER_CORES[target_class.__name__] = target_class


def register_randomizer(target_class):
    assert target_class not in OBS_RANDOMIZERS, f"Already registered obs randomizer {target_class}!"
    OBS_RANDOMIZERS[target_class.__name__] = target_class


class ObservationKeyToModalityDict(dict):
    """
    Custom dictionary class with the sole additional purpose of automatically registering new "keys" at runtime
    without breaking. This is mainly for backwards compatibility, where certain keys such as "latent", "actions", etc.
    are used automatically by certain models (e.g.: VAEs) but were never specified by the user externally in their
    config. Thus, this dictionary will automatically handle those keys by implicitly associating them with the low_dim
    modality.
    """
    def __getitem__(self, item):
        # If a key doesn't already exist, warn the user and add default mapping
        if item not in self.keys():
            print(f"ObservationKeyToModalityDict: {item} not found,"
                  f" adding {item} to mapping with assumed low_dim modality!")
            self.__setitem__(item, "low_dim")
        return super(ObservationKeyToModalityDict, self).__getitem__(item)


def obs_encoder_kwargs_from_config(obs_encoder_config):
    """
    Generate a set of args used to create visual backbones for networks
    from the observation encoder config.

    Args:
        obs_encoder_config (Config): Config object containing relevant encoder information. Should be equivalent to
            config.observation.encoder

    Returns:
        dict: Processed encoder kwargs
    """
    # Loop over each obs modality
    # Unlock encoder config
    obs_encoder_config.unlock()
    for obs_modality, encoder_kwargs in obs_encoder_config.items():
        # First run some sanity checks and store the classes
        for cls_name, cores in zip(("core", "obs_randomizer"), (OBS_ENCODER_CORES, OBS_RANDOMIZERS)):
            # Make sure the requested encoder for each obs_modality exists
            cfg_cls = encoder_kwargs[f"{cls_name}_class"]
            if cfg_cls is not None:
                assert cfg_cls in cores, f"No {cls_name} class with name {cfg_cls} found, must register this class before" \
                    f"creating model!"
                # encoder_kwargs[f"{cls_name}_class"] = cores[cfg_cls]

        # Process core and randomizer kwargs
        encoder_kwargs.core_kwargs = dict() if encoder_kwargs.core_kwargs is None else \
            deepcopy(encoder_kwargs.core_kwargs)
        encoder_kwargs.obs_randomizer_kwargs = dict() if encoder_kwargs.obs_randomizer_kwargs is None else \
            deepcopy(encoder_kwargs.obs_randomizer_kwargs)

    # Re-lock keys
    obs_encoder_config.lock()

    return dict(obs_encoder_config)


def initialize_obs_modality_mapping_from_dict(modality_mapping):
    """
    This function is an alternative to @initialize_obs_utils_with_obs_specs, that allows manually setting of modalities.
    NOTE: Only one of these should be called at runtime -- not both! (Note that all training scripts that use a config)
        automatically handle obs modality mapping, so using this function is usually unnecessary)

    Args:
        modality_mapping (dict): Maps modality string names (e.g.: rgb, low_dim, etc.) to a list of observation
            keys that should belong to that modality
    """
    global OBS_KEYS_TO_MODALITIES, OBS_MODALITIES_TO_KEYS

    OBS_KEYS_TO_MODALITIES = ObservationKeyToModalityDict()
    OBS_MODALITIES_TO_KEYS = dict()

    for mod, keys in modality_mapping.items():
        OBS_MODALITIES_TO_KEYS[mod] = deepcopy(keys)
        OBS_KEYS_TO_MODALITIES.update({k: mod for k in keys})


def initialize_obs_utils_with_obs_specs(obs_modality_specs):
    """
    This function should be called before using any observation key-specific
    functions in this file, in order to make sure that all utility
    functions are aware of the observation modalities (e.g. which ones
    are low-dimensional, which ones are rgb, etc.).

    It constructs two dictionaries: (1) that map observation modality (e.g. low_dim, rgb) to
    a list of observation keys under that modality, and (2) that maps the inverse, specific
    observation keys to their corresponding observation modality.

    Input should be a nested dictionary (or list of such dicts) with the following structure:

        obs_variant (str):
            obs_modality (str): observation keys (list)
            ...
        ...

    Example:
        {
            "obs": {
                "low_dim": ["robot0_eef_pos", "robot0_eef_quat"],
                "rgb": ["agentview_image", "robot0_eye_in_hand"],
            }
            "goal": {
                "low_dim": ["robot0_eef_pos"],
                "rgb": ["agentview_image"]
            }
        }

    In the example, raw observations consist of low-dim and rgb modalities, with
    the robot end effector pose under low-dim, and the agentview and wrist camera
    images under rgb, while goal observations also consist of low-dim and rgb modalities,
    with a subset of the raw observation keys per modality.

    Args:
        obs_modality_specs (dict or list): A nested dictionary (see docstring above for an example)
            or a list of nested dictionaries. Accepting a list as input makes it convenient for
            situations where multiple modules may each have their own modality spec.
    """
    global OBS_KEYS_TO_MODALITIES, OBS_MODALITIES_TO_KEYS

    OBS_KEYS_TO_MODALITIES = ObservationKeyToModalityDict()

    # accept one or more spec dictionaries - if it's just one, account for this
    if isinstance(obs_modality_specs, dict):
        obs_modality_spec_list = [obs_modality_specs]
    else:
        obs_modality_spec_list = obs_modality_specs

    # iterates over observation specs
    obs_modality_mapping = {}
    for obs_modality_spec in obs_modality_spec_list:
        # iterates over observation variants (e.g. observations, goals, subgoals)
        for obs_modalities in obs_modality_spec.values():
            for obs_modality, obs_keys in obs_modalities.items():
                # add all keys for each obs modality to the corresponding list in obs_modality_mapping
                if obs_modality not in obs_modality_mapping:
                    obs_modality_mapping[obs_modality] = []
                obs_modality_mapping[obs_modality] += obs_keys
                # loop over each modality, and add to global dict if it doesn't exist yet
                for obs_key in obs_keys:
                    if obs_key not in OBS_KEYS_TO_MODALITIES:
                        OBS_KEYS_TO_MODALITIES[obs_key] = obs_modality
                    # otherwise, run sanity check to make sure we don't have conflicting, duplicate entries
                    else:
                        assert OBS_KEYS_TO_MODALITIES[obs_key] == obs_modality, \
                            f"Cannot register obs key {obs_key} with modality {obs_modality}; " \
                            f"already exists with corresponding modality {OBS_KEYS_TO_MODALITIES[obs_key]}"

    # remove duplicate entries and store in global mapping
    OBS_MODALITIES_TO_KEYS = { obs_modality : list(set(obs_modality_mapping[obs_modality])) for obs_modality in obs_modality_mapping }

    print("\n============= Initialized Observation Utils with Obs Spec =============\n")
    for obs_modality, obs_keys in OBS_MODALITIES_TO_KEYS.items():
        print("using obs modality: {} with keys: {}".format(obs_modality, obs_keys))


def initialize_default_obs_encoder(obs_encoder_config):
    """
    Initializes the default observation encoder kwarg information to be used by all networks if no values are manually
    specified at runtime.

    Args:
        obs_encoder_config (Config): Observation encoder config to use.
            Should be equivalent to config.observation.encoder
    """
    global DEFAULT_ENCODER_KWARGS
    DEFAULT_ENCODER_KWARGS = obs_encoder_kwargs_from_config(obs_encoder_config)


def initialize_obs_utils_with_config(config):
    """
    Utility function to parse config and call @initialize_obs_utils_with_obs_specs and
    @initialize_default_obs_encoder_kwargs with the correct arguments.

    Args:
        config (BaseConfig instance): config object
    """
    if config.algo_name == "hbc":
        obs_modality_specs = [
            config.observation.planner.modalities, 
            config.observation.actor.modalities,
        ]
        obs_encoder_config = config.observation.actor.encoder
    elif config.algo_name == "iris":
        obs_modality_specs = [
            config.observation.value_planner.planner.modalities, 
            config.observation.value_planner.value.modalities, 
            config.observation.actor.modalities,
        ]
        obs_encoder_config = config.observation.actor.encoder
    else:
        obs_modality_specs = [config.observation.modalities]
        obs_encoder_config = config.observation.encoder
    initialize_obs_utils_with_obs_specs(obs_modality_specs=obs_modality_specs)
    initialize_default_obs_encoder(obs_encoder_config=obs_encoder_config)


def key_is_obs_modality(key, obs_modality):
    """
    Check if observation key corresponds to modality @obs_modality.

    Args:
        key (str): obs key name to check
        obs_modality (str): observation modality - e.g.: "low_dim", "rgb"
    """
    assert OBS_KEYS_TO_MODALITIES is not None, "error: must call ObsUtils.initialize_obs_utils_with_obs_config first"
    return OBS_KEYS_TO_MODALITIES[key] == obs_modality


def center_crop(im, t_h, t_w):
    """
    Takes a center crop of an image.

    Args:
        im (np.array or torch.Tensor): image of shape (..., height, width, channel)
        t_h (int): height of crop
        t_w (int): width of crop

    Returns:
        im (np.array or torch.Tensor): center cropped image
    """
    assert(im.shape[-3] >= t_h and im.shape[-2] >= t_w)
    assert(im.shape[-1] in [1, 3])
    crop_h = int((im.shape[-3] - t_h) / 2)
    crop_w = int((im.shape[-2] - t_w) / 2)
    return im[..., crop_h:crop_h + t_h, crop_w:crop_w + t_w, :]


def batch_image_hwc_to_chw(im):
    """
    Channel swap for images - useful for preparing images for
    torch training.

    Args:
        im (np.array or torch.Tensor): image of shape (batch, height, width, channel)
            or (height, width, channel)

    Returns:
        im (np.array or torch.Tensor): image of shape (batch, channel, height, width)
            or (channel, height, width)
    """
    start_dims = np.arange(len(im.shape) - 3).tolist()
    s = start_dims[-1] if len(start_dims) > 0 else -1
    if isinstance(im, np.ndarray):
        return im.transpose(start_dims + [s + 3, s + 1, s + 2])
    else:
        return im.permute(start_dims + [s + 3, s + 1, s + 2])


def batch_image_chw_to_hwc(im):
    """
    Inverse of channel swap in @batch_image_hwc_to_chw.

    Args:
        im (np.array or torch.Tensor): image of shape (batch, channel, height, width)
            or (channel, height, width)

    Returns:
        im (np.array or torch.Tensor): image of shape (batch, height, width, channel)
            or (height, width, channel)
    """
    start_dims = np.arange(len(im.shape) - 3).tolist()
    s = start_dims[-1] if len(start_dims) > 0 else -1
    if isinstance(im, np.ndarray):
        return im.transpose(start_dims + [s + 2, s + 3, s + 1])
    else:
        return im.permute(start_dims + [s + 2, s + 3, s + 1])


def process_obs(obs, obs_modality=None, obs_key=None):
    """
    Process observation @obs corresponding to @obs_modality modality (or implicitly inferred from @obs_key)
    to prepare for network input.

    Note that either obs_modality OR obs_key must be specified!

    If both are specified, obs_key will override obs_modality

    Args:
        obs (np.array or torch.Tensor): Observation to process. Leading batch dimension is optional
        obs_modality (str): Observation modality (e.g.: depth, image, low_dim, etc.)
        obs_key (str): Name of observation from which to infer @obs_modality

    Returns:
        processed_obs (np.array or torch.Tensor): processed observation
    """
    assert obs_modality is not None or obs_key is not None, "Either obs_modality or obs_key must be specified!"
    if obs_key is not None:
        obs_modality = OBS_KEYS_TO_MODALITIES[obs_key]
    return OBS_MODALITY_CLASSES[obs_modality].process_obs(obs)


def process_obs_dict(obs_dict):
    """
    Process observations in observation dictionary to prepare for network input.

    Args:
        obs_dict (dict): dictionary mapping observation keys to np.array or
            torch.Tensor. Leading batch dimensions are optional.

    Returns:
        new_dict (dict): dictionary where observation keys have been processed by their corresponding processors
    """
    return { k : process_obs(obs=obs, obs_key=k) for k, obs in obs_dict.items() } # shallow copy


def process_frame(frame, channel_dim, scale):
    """
    Given frame fetched from dataset, process for network input. Converts array
    to float (from uint8), normalizes pixels from range [0, @scale] to [0, 1], and channel swaps
    from (H, W, C) to (C, H, W).

    Args:
        frame (np.array or torch.Tensor): frame array
        channel_dim (int): Number of channels to sanity check for
        scale (float or None): Value to normalize inputs by

    Returns:
        processed_frame (np.array or torch.Tensor): processed frame
    """
    # Channel size should either be 3 (RGB) or 1 (depth)
    assert (frame.shape[-1] == channel_dim)
    frame = TU.to_float(frame)
    if scale is not None:
        frame = frame / scale
        frame = frame.clip(0.0, 1.0)
    frame = batch_image_hwc_to_chw(frame)

    return frame


def unprocess_obs(obs, obs_modality=None, obs_key=None):
    """
    Prepare observation @obs corresponding to @obs_modality modality (or implicitly inferred from @obs_key)
    to prepare for deployment.

    Note that either obs_modality OR obs_key must be specified!

    If both are specified, obs_key will override obs_modality

    Args:
        obs (np.array or torch.Tensor): Observation to unprocess. Leading batch dimension is optional
        obs_modality (str): Observation modality (e.g.: depth, image, low_dim, etc.)
        obs_key (str): Name of observation from which to infer @obs_modality

    Returns:
        unprocessed_obs (np.array or torch.Tensor): unprocessed observation
    """
    assert obs_modality is not None or obs_key is not None, "Either obs_modality or obs_key must be specified!"
    if obs_key is not None:
        obs_modality = OBS_KEYS_TO_MODALITIES[obs_key]
    return OBS_MODALITY_CLASSES[obs_modality].unprocess_obs(obs)


def unprocess_obs_dict(obs_dict):
    """
    Prepare processed observation dictionary for saving to dataset. Inverse of
    @process_obs.

    Args:
        obs_dict (dict): dictionary mapping observation keys to np.array or
            torch.Tensor. Leading batch dimensions are optional.

    Returns:
        new_dict (dict): dictionary where observation keys have been unprocessed by
            their respective unprocessor methods
    """
    return { k : unprocess_obs(obs=obs, obs_key=k) for k, obs in obs_dict.items() } # shallow copy


def unprocess_frame(frame, channel_dim, scale):
    """
    Given frame prepared for network input, prepare for saving to dataset.
    Inverse of @process_frame.

    Args:
        frame (np.array or torch.Tensor): frame array
        channel_dim (int): What channel dimension should be (used for sanity check)
        scale (float or None): Scaling factor to apply during denormalization

    Returns:
        unprocessed_frame (np.array or torch.Tensor): frame passed through
            inverse operation of @process_frame
    """
    assert frame.shape[-3] == channel_dim # check for channel dimension
    frame = batch_image_chw_to_hwc(frame)
    if scale is not None:
        frame = scale * frame
    return frame


def get_processed_shape(obs_modality, input_shape):
    """
    Given observation modality @obs_modality and expected inputs of shape @input_shape (excluding batch dimension), return the
    expected processed observation shape resulting from process_{obs_modality}.

    Args:
        obs_modality (str): Observation modality to use (e.g.: low_dim, rgb, depth, etc...)
        input_shape (list of int): Expected input dimensions, excluding the batch dimension

    Returns:
        list of int: expected processed input shape
    """
    return list(process_obs(obs=np.zeros(input_shape), obs_modality=obs_modality).shape)


def normalize_obs(obs_dict, obs_normalization_stats):
    """
    Normalize observations using the provided "mean" and "std" entries 
    for each observation key. The observation dictionary will be
    modified in-place.

    Args:
        obs_dict (dict): dictionary mapping observation key to np.array or
            torch.Tensor. Can have any number of leading batch dimensions.

        obs_normalization_stats (dict): this should map observation keys to dicts
            with a "mean" and "std" of shape (1, ...) where ... is the default
            shape for the observation.

    Returns:
        obs_dict (dict): obs dict with normalized observation arrays
    """

    # ensure we have statistics for each modality key in the observation
    assert set(obs_dict.keys()).issubset(obs_normalization_stats)

    for m in obs_dict:
        # get rid of extra dimension - we will pad for broadcasting later
        mean = obs_normalization_stats[m]["mean"][0]
        std = obs_normalization_stats[m]["std"][0]

        # shape consistency checks
        m_num_dims = len(mean.shape)
        shape_len_diff = len(obs_dict[m].shape) - m_num_dims
        assert shape_len_diff >= 0, "shape length mismatch in @normalize_obs"
        assert obs_dict[m].shape[-m_num_dims:] == mean.shape, "shape mismatch in @normalize_obs"

        # Obs can have one or more leading batch dims - prepare for broadcasting.
        # 
        # As an example, if the obs has shape [B, T, D] and our mean / std stats are shape [D]
        # then we should pad the stats to shape [1, 1, D].
        reshape_padding = tuple([1] * shape_len_diff)
        mean = mean.reshape(reshape_padding + tuple(mean.shape))
        std = std.reshape(reshape_padding + tuple(std.shape))

        obs_dict[m] = (obs_dict[m] - mean) / std

    return obs_dict


def has_modality(modality, obs_keys):
    """
    Returns True if @modality is present in the list of observation keys @obs_keys.

    Args:
        modality (str): modality to check for, e.g.: rgb, depth, etc.
        obs_keys (list): list of observation keys
    """
    for k in obs_keys:
        if key_is_obs_modality(k, obs_modality=modality):
            return True
    return False


def repeat_and_stack_observation(obs_dict, n):
    """
    Given an observation dictionary and a desired repeat value @n,
    this function will return a new observation dictionary where
    each modality is repeated @n times and the copies are
    stacked in the first dimension. 

    For example, if a batch of 3 observations comes in, and n is 2,
    the output will look like [ob1; ob1; ob2; ob2; ob3; ob3] in
    each modality.

    Args:
        obs_dict (dict): dictionary mapping observation key to np.array or
            torch.Tensor. Leading batch dimensions are optional.

        n (int): number to repeat by

    Returns:
        repeat_obs_dict (dict): repeated obs dict
    """
    return TU.repeat_by_expand_at(obs_dict, repeats=n, dim=0)


def crop_image_from_indices(images, crop_indices, crop_height, crop_width):
    """
    Crops images at the locations specified by @crop_indices. Crops will be 
    taken across all channels.

    Args:
        images (torch.Tensor): batch of images of shape [..., C, H, W]

        crop_indices (torch.Tensor): batch of indices of shape [..., N, 2] where
            N is the number of crops to take per image and each entry corresponds
            to the pixel height and width of where to take the crop. Note that
            the indices can also be of shape [..., 2] if only 1 crop should
            be taken per image. Leading dimensions must be consistent with
            @images argument. Each index specifies the top left of the crop.
            Values must be in range [0, H - CH - 1] x [0, W - CW - 1] where
            H and W are the height and width of @images and CH and CW are
            @crop_height and @crop_width.

        crop_height (int): height of crop to take

        crop_width (int): width of crop to take

    Returns:
        crops (torch.Tesnor): cropped images of shape [..., C, @crop_height, @crop_width]
    """

    # make sure length of input shapes is consistent
    assert crop_indices.shape[-1] == 2
    ndim_im_shape = len(images.shape)
    ndim_indices_shape = len(crop_indices.shape)
    assert (ndim_im_shape == ndim_indices_shape + 1) or (ndim_im_shape == ndim_indices_shape + 2)

    # maybe pad so that @crop_indices is shape [..., N, 2]
    is_padded = False
    if ndim_im_shape == ndim_indices_shape + 2:
        crop_indices = crop_indices.unsqueeze(-2)
        is_padded = True

    # make sure leading dimensions between images and indices are consistent
    assert images.shape[:-3] == crop_indices.shape[:-2]

    device = images.device
    image_c, image_h, image_w = images.shape[-3:]
    num_crops = crop_indices.shape[-2]

    # make sure @crop_indices are in valid range
    assert (crop_indices[..., 0] >= 0).all().item()
    assert (crop_indices[..., 0] < (image_h - crop_height)).all().item()
    assert (crop_indices[..., 1] >= 0).all().item()
    assert (crop_indices[..., 1] < (image_w - crop_width)).all().item()

    # convert each crop index (ch, cw) into a list of pixel indices that correspond to the entire window.

    # 2D index array with columns [0, 1, ..., CH - 1] and shape [CH, CW]
    crop_ind_grid_h = torch.arange(crop_height).to(device)
    crop_ind_grid_h = TU.unsqueeze_expand_at(crop_ind_grid_h, size=crop_width, dim=-1)
    # 2D index array with rows [0, 1, ..., CW - 1] and shape [CH, CW]
    crop_ind_grid_w = torch.arange(crop_width).to(device)
    crop_ind_grid_w = TU.unsqueeze_expand_at(crop_ind_grid_w, size=crop_height, dim=0)
    # combine into shape [CH, CW, 2]
    crop_in_grid = torch.cat((crop_ind_grid_h.unsqueeze(-1), crop_ind_grid_w.unsqueeze(-1)), dim=-1)

    # Add above grid with the offset index of each sampled crop to get 2d indices for each crop.
    # After broadcasting, this will be shape [..., N, CH, CW, 2] and each crop has a [CH, CW, 2]
    # shape array that tells us which pixels from the corresponding source image to grab.
    grid_reshape = [1] * len(crop_indices.shape[:-1]) + [crop_height, crop_width, 2]
    all_crop_inds = crop_indices.unsqueeze(-2).unsqueeze(-2) + crop_in_grid.reshape(grid_reshape)

    # For using @torch.gather, convert to flat indices from 2D indices, and also
    # repeat across the channel dimension. To get flat index of each pixel to grab for 
    # each sampled crop, we just use the mapping: ind = h_ind * @image_w + w_ind
    all_crop_inds = all_crop_inds[..., 0] * image_w + all_crop_inds[..., 1] # shape [..., N, CH, CW]
    all_crop_inds = TU.unsqueeze_expand_at(all_crop_inds, size=image_c, dim=-3) # shape [..., N, C, CH, CW]
    all_crop_inds = TU.flatten(all_crop_inds, begin_axis=-2) # shape [..., N, C, CH * CW]

    # Repeat and flatten the source images -> [..., N, C, H * W] and then use gather to index with crop pixel inds
    images_to_crop = TU.unsqueeze_expand_at(images, size=num_crops, dim=-4)
    images_to_crop = TU.flatten(images_to_crop, begin_axis=-2)
    crops = torch.gather(images_to_crop, dim=-1, index=all_crop_inds)
    # [..., N, C, CH * CW] -> [..., N, C, CH, CW]
    reshape_axis = len(crops.shape) - 1
    crops = TU.reshape_dimensions(crops, begin_axis=reshape_axis, end_axis=reshape_axis, 
                    target_dims=(crop_height, crop_width))

    if is_padded:
        # undo padding -> [..., C, CH, CW]
        crops = crops.squeeze(-4)
    return crops


def sample_random_image_crops(images, crop_height, crop_width, num_crops, pos_enc=False):
    """
    For each image, randomly sample @num_crops crops of size (@crop_height, @crop_width), from
    @images.

    Args:
        images (torch.Tensor): batch of images of shape [..., C, H, W]

        crop_height (int): height of crop to take
        
        crop_width (int): width of crop to take

        num_crops (n): number of crops to sample

        pos_enc (bool): if True, also add 2 channels to the outputs that gives a spatial 
            encoding of the original source pixel locations. This means that the
            output crops will contain information about where in the source image 
            it was sampled from.

    Returns:
        crops (torch.Tensor): crops of shape (..., @num_crops, C, @crop_height, @crop_width) 
            if @pos_enc is False, otherwise (..., @num_crops, C + 2, @crop_height, @crop_width)

        crop_inds (torch.Tensor): sampled crop indices of shape (..., N, 2)
    """
    device = images.device

    # maybe add 2 channels of spatial encoding to the source image
    source_im = images
    if pos_enc:
        # spatial encoding [y, x] in [0, 1]
        h, w = source_im.shape[-2:]
        pos_y, pos_x = torch.meshgrid(torch.arange(h), torch.arange(w))
        pos_y = pos_y.float().to(device) / float(h)
        pos_x = pos_x.float().to(device) / float(w)
        position_enc = torch.stack((pos_y, pos_x)) # shape [C, H, W]

        # unsqueeze and expand to match leading dimensions -> shape [..., C, H, W]
        leading_shape = source_im.shape[:-3]
        position_enc = position_enc[(None,) * len(leading_shape)]
        position_enc = position_enc.expand(*leading_shape, -1, -1, -1)

        # concat across channel dimension with input
        source_im = torch.cat((source_im, position_enc), dim=-3)

    # make sure sample boundaries ensure crops are fully within the images
    image_c, image_h, image_w = source_im.shape[-3:]
    max_sample_h = image_h - crop_height
    max_sample_w = image_w - crop_width

    # Sample crop locations for all tensor dimensions up to the last 3, which are [C, H, W].
    # Each gets @num_crops samples - typically this will just be the batch dimension (B), so 
    # we will sample [B, N] indices, but this supports having more than one leading dimension,
    # or possibly no leading dimension.
    #
    # Trick: sample in [0, 1) with rand, then re-scale to [0, M) and convert to long to get sampled ints
    crop_inds_h = (max_sample_h * torch.rand(*source_im.shape[:-3], num_crops).to(device)).long()
    crop_inds_w = (max_sample_w * torch.rand(*source_im.shape[:-3], num_crops).to(device)).long()
    crop_inds = torch.cat((crop_inds_h.unsqueeze(-1), crop_inds_w.unsqueeze(-1)), dim=-1) # shape [..., N, 2]

    crops = crop_image_from_indices(
        images=source_im, 
        crop_indices=crop_inds, 
        crop_height=crop_height, 
        crop_width=crop_width, 
    )

    return crops, crop_inds


class Modality:
    """
    Observation Modality class to encapsulate necessary functions needed to
    process observations of this modality
    """
    # observation keys to associate with this modality
    keys = set()

    # Custom processing function that should prepare raw observations of this modality for training
    _custom_obs_processor = None

    # Custom unprocessing function that should prepare observations of this modality used during training for deployment
    _custom_obs_unprocessor = None

    # Name of this modality -- must be set by subclass!
    name = None

    def __init_subclass__(cls, **kwargs):
        """
        Hook method to automatically register all valid subclasses so we can keep track of valid modalities
        """
        assert cls.name is not None, f"Name of modality {cls.__name__} must be specified!"
        register_obs_key(cls)

    @classmethod
    def set_keys(cls, keys):
        """
        Sets the observation keys associated with this modality.

        Args:
            keys (list or set): observation keys to associate with this modality
        """
        cls.keys = {k for k in keys}

    @classmethod
    def add_keys(cls, keys):
        """
        Adds the observation @keys associated with this modality to the current set of keys.

        Args:
            keys (list or set): observation keys to add to associate with this modality
        """
        for key in keys:
            cls.keys.add(key)

    @classmethod
    def set_obs_processor(cls, processor=None):
        """
        Sets the processor for this observation modality. If @processor is set to None, then
        the obs processor will use the default one (self.process_obs(...)). Otherwise, @processor
        should be a function to process this corresponding observation modality.

        Args:
            processor (function or None): If not None, should be function that takes in either a
                np.array or torch.Tensor and output the processed array / tensor. If None, will reset
                to the default processor (self.process_obs(...))
        """
        cls._custom_obs_processor = processor

    @classmethod
    def set_obs_unprocessor(cls, unprocessor=None):
        """
        Sets the unprocessor for this observation modality. If @unprocessor is set to None, then
        the obs unprocessor will use the default one (self.unprocess_obs(...)). Otherwise, @unprocessor
        should be a function to process this corresponding observation modality.

        Args:
            unprocessor (function or None): If not None, should be function that takes in either a
                np.array or torch.Tensor and output the unprocessed array / tensor. If None, will reset
                to the default unprocessor (self.unprocess_obs(...))
        """
        cls._custom_obs_unprocessor = unprocessor

    @classmethod
    def _default_obs_processor(cls, obs):
        """
        Default processing function for this obs modality.

        Note that this function is overridden by self.custom_obs_processor (a function with identical inputs / outputs)
        if it is not None.

        Args:
            obs (np.array or torch.Tensor): raw observation, which may include a leading batch dimension

        Returns:
            np.array or torch.Tensor: processed observation
        """
        raise NotImplementedError

    @classmethod
    def _default_obs_unprocessor(cls, obs):
        """
        Default unprocessing function for this obs modality.

        Note that this function is overridden by self.custom_obs_unprocessor
        (a function with identical inputs / outputs) if it is not None.

        Args:
            obs (np.array or torch.Tensor): processed observation, which may include a leading batch dimension

        Returns:
            np.array or torch.Tensor: unprocessed observation
        """
        raise NotImplementedError

    @classmethod
    def process_obs(cls, obs):
        """
        Prepares an observation @obs of this modality for network input.

        Args:
            obs (np.array or torch.Tensor): raw observation, which may include a leading batch dimension

        Returns:
            np.array or torch.Tensor: processed observation
        """
        processor = cls._custom_obs_processor if \
            cls._custom_obs_processor is not None else cls._default_obs_processor
        return processor(obs)

    @classmethod
    def unprocess_obs(cls, obs):
        """
        Prepares an observation @obs of this modality for deployment.

        Args:
            obs (np.array or torch.Tensor): processed observation, which may include a leading batch dimension

        Returns:
            np.array or torch.Tensor: unprocessed observation
        """
        unprocessor = cls._custom_obs_unprocessor if \
            cls._custom_obs_unprocessor is not None else cls._default_obs_unprocessor
        return unprocessor(obs)

    @classmethod
    def process_obs_from_dict(cls, obs_dict, inplace=True):
        """
        Receives a dictionary of keyword mapped observations @obs_dict, and processes the observations with keys
        corresponding to this modality. A copy will be made of the received dictionary unless @inplace is True

        Args:
            obs_dict (dict): Dictionary mapping observation keys to observations
            inplace (bool): If True, will modify @obs_dict in place, otherwise, will create a copy

        Returns:
            dict: observation dictionary with processed observations corresponding to this modality
        """
        if inplace:
            obs_dict = deepcopy(obs_dict)
        # Loop over all keys and process the ones corresponding to this modality
        for key, obs in obs_dict.values():
            if key in cls.keys:
                obs_dict[key] = cls.process_obs(obs)

        return obs_dict


class ImageModality(Modality):
    """
    Modality for RGB image observations
    """
    name = "rgb"

    @classmethod
    def _default_obs_processor(cls, obs):
        """
        Given image fetched from dataset, process for network input. Converts array
        to float (from uint8), normalizes pixels from range [0, 255] to [0, 1], and channel swaps
        from (H, W, C) to (C, H, W).

        Args:
            obs (np.array or torch.Tensor): image array

        Returns:
            processed_obs (np.array or torch.Tensor): processed image
        """
        return process_frame(frame=obs, channel_dim=3, scale=255.)

    @classmethod
    def _default_obs_unprocessor(cls, obs):
        """
        Given image prepared for network input, prepare for saving to dataset.
        Inverse of @process_frame.

        Args:
            obs (np.array or torch.Tensor): image array

        Returns:
            unprocessed_obs (np.array or torch.Tensor): image passed through
                inverse operation of @process_frame
        """
        return TU.to_uint8(unprocess_frame(frame=obs, channel_dim=3, scale=255.))


class DepthModality(Modality):
    """
    Modality for depth observations
    """
    name = "depth"

    @classmethod
    def _default_obs_processor(cls, obs):
        """
        Given depth fetched from dataset, process for network input. Converts array
        to float (from uint8), normalizes pixels from range [0, 1] to [0, 1], and channel swaps
        from (H, W, C) to (C, H, W).

        Args:
            obs (np.array or torch.Tensor): depth array

        Returns:
            processed_obs (np.array or torch.Tensor): processed depth
        """
        # assume a [0,255] discretized depth input
        scale = 255.
        return process_frame(frame=obs, channel_dim=1, scale=scale)

    @classmethod
    def _default_obs_unprocessor(cls, obs):
        """
        Given depth prepared for network input, prepare for saving to dataset.
        Inverse of @process_depth.

        Args:
            obs (np.array or torch.Tensor): depth array

        Returns:
            unprocessed_obs (np.array or torch.Tensor): depth passed through
                inverse operation of @process_depth
        """
        # assume a [0,255] discretized depth input
        scale = 255.
        return unprocess_frame(frame=obs, channel_dim=1, scale=scale)


class ScanModality(Modality):
    """
    Modality for scan observations
    """
    name = "scan"

    @classmethod
    def _default_obs_processor(cls, obs):
        # Channel swaps ([...,] L, C) --> ([...,] C, L)
        
        # First, add extra dimension at 2nd to last index to treat this as a frame
        shape = obs.shape
        new_shape = [*shape[:-2], 1, *shape[-2:]]
        obs = obs.reshape(new_shape)
        
        # Convert shape
        obs = batch_image_hwc_to_chw(obs)
        
        # Remove extra dimension (it's the second from last dimension)
        obs = obs.squeeze(-2)
        return obs

    @classmethod
    def _default_obs_unprocessor(cls, obs):
        # Channel swaps ([B,] C, L) --> ([B,] L, C)
        
        # First, add extra dimension at 1st index to treat this as a frame
        shape = obs.shape
        new_shape = [*shape[:-2], 1, *shape[-2:]]
        obs = obs.reshape(new_shape)

        # Convert shape
        obs = batch_image_chw_to_hwc(obs)

        # Remove extra dimension (it's the second from last dimension)
        obs = obs.squeeze(-2)
        return obs


class LowDimModality(Modality):
    """
    Modality for low dimensional observations
    """
    name = "low_dim"

    @classmethod
    def _default_obs_processor(cls, obs):
        return obs

    @classmethod
    def _default_obs_unprocessor(cls, obs):
        return obs
