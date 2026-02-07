"""
This file contains the robosuite environment wrapper that is used
to provide a standardized environment API for training policies and interacting
with metadata present in datasets.
"""
import os
import json
import numpy as np
from copy import deepcopy
import open3d as o3d
import torch
import torch.nn.functional as F

import robosuite

import robosuite.utils.transform_utils as T
from robosuite.utils.camera_utils import get_real_depth_map, get_camera_extrinsic_matrix, get_camera_intrinsic_matrix
try:
    # this is needed for ensuring robosuite can find the additional mimicgen environments (see https://mimicgen.github.io)
    import mimicgen
except ImportError:
    pass

import robomimic.utils.obs_utils as ObsUtils
import robomimic.envs.env_base as EB

# protect against missing mujoco-py module, since robosuite might be using mujoco-py or DM backend
try:
    import mujoco_py
    MUJOCO_EXCEPTIONS = [mujoco_py.builder.MujocoException]
except ImportError:
    MUJOCO_EXCEPTIONS = []

from robomimic.utils.obs_utils import (depth2fgpcd, np2o3d, o3d2np, pcd_to_voxel, localize_pcd_batch, 
                                       enlarge_mask, crop_pcd, get_clipspace, get_workspace, get_pcd_z_min)


import cv2
import copy
import numpy as np
import matplotlib.pyplot as plt # For displaying images in environments like Jupyter
from scipy.spatial.transform import Rotation as R

def visualize_voxel(np_voxels):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    voxel_reso = np_voxels.shape[-1]

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    indices = np.argwhere(np_voxels[0] != 0)
    colors = np_voxels[1:, indices[:, 0], indices[:, 1], indices[:, 2]].T

    ax.scatter(indices[:, 0], indices[:, 1], indices[:, 2], color=colors/255., marker='s')

    # Set labels and show the plot
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    ax.set_xlim(0, voxel_reso)
    ax.set_ylim(0, voxel_reso)
    ax.set_zlim(0, voxel_reso)
    plt.show(block=False)

def visualize_pcds(points: list, mode='color'):
    assert mode in ['color', 'xyz'], "Mode must be 'color' or 'xyz'"
    
    pcds = []
    for p in points:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(p[:,:3])
    
        if mode == 'color':
            assert p.shape[1] >= 6
            pcd.colors = o3d.utility.Vector3dVector(p[:,3:6])
        
        pcds.append(pcd)
    
    origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([*pcds, origin])

def apply_se3_pcd_transform(points, transform):
    """
    Apply SE3 transformation to a set of points.
    points: a (N, 6) array where N is number of points and the last dimension is (x,y,z,r,g,b)
    """
    new_points = copy.deepcopy(points)
    points_homogeneous = np.hstack([new_points[:, :3], np.ones((new_points.shape[0], 1))])
    transformed_points = (transform @ points_homogeneous.T).T
    new_points[:, :3] = transformed_points[:, :3]
    return new_points

def populate_point_num(pcd, point_num):
    if pcd.shape[0] < point_num:
        extra_choice = np.random.choice(pcd.shape[0], point_num-pcd.shape[0], replace=True)
        pcd = np.concatenate([pcd, pcd[extra_choice]], axis=0)
    else:
        shuffle_idx = np.random.permutation(pcd.shape[0])[:point_num]
        pcd = pcd[shuffle_idx]
    return pcd

def generate_sphere_pcd(is_rot_colorful=True):
    num_points = 896  # adjust the number of points as needed
    r = 0.05  # sphere radius, modify r if needed
    indices = np.arange(num_points, dtype=float) + 0.5
    phi = np.arccos(1 - 2 * indices / num_points)
    theta = np.pi * (1 + 5**0.5) * indices
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    colors = np.zeros((num_points, 3), dtype=np.float32)
    mask1 = (x >= 0) & (y >= 0) & (z >= 0)
    mask2 = (x < 0) & (y >= 0) & (z >= 0)
    mask3 = (x < 0) & (y < 0) & (z >= 0)
    mask4 = (x >= 0) & (y < 0) & (z >= 0)
    mask5 = (x >= 0) & (y >= 0) & (z < 0)
    mask6 = (x < 0) & (y >= 0) & (z < 0)
    mask7 = (x < 0) & (y < 0) & (z < 0)
    mask8 = (x >= 0) & (y < 0) & (z < 0)

    if is_rot_colorful:
        colors[mask1] = [1.0, 0.0, 0.0]   # red      : (+x, +y, +z)
        colors[mask2] = [0.0, 1.0, 0.0]   # green    : (-x, +y, +z)
        colors[mask3] = [0.0, 0.0, 1.0]   # blue     : (-x, -y, +z)
        colors[mask4] = [1.0, 1.0, 0.0]   # yellow   : (+x, -y, +z)
        colors[mask5] = [1.0, 0.0, 1.0]   # magenta  : (+x, +y, -z)
        colors[mask6] = [0.0, 1.0, 1.0]   # cyan     : (-x, +y, -z)
        colors[mask7] = [0.5, 0.5, 0.5]   # grey     : (-x, -y, -z)
        colors[mask8] = [1.0, 0.5, 0.0]   # orange   : (+x, -y, -z)
        # colors = np.tile(np.array([0., 1., 0.]), (num_points, 1)) 
        # colors = np.where(y[:, None] < 0, np.array([0., 1., 0.]), np.array([0., 0., 1.]))
    return np.concatenate([np.stack([x, y, z], axis=1), colors], axis=1)

# def generate_sphere_pcd():
#     # the sphere is all black except the two sides are black
#     num_points = 896
#     r = 0.05  # sphere radius
#     indices = np.arange(num_points, dtype=float) + 0.5
#     phi = np.arccos(1 - 2 * indices / num_points)
#     theta = np.pi * (1 + 5**0.5) * indices

#     x = r * np.sin(phi) * np.cos(theta)
#     y = r * np.sin(phi) * np.sin(theta)
#     z = r * np.cos(phi)

#     # start with all points gray
#     colors = np.ones((num_points, 3), dtype=np.float32) * 0.5  # gray color

#     # Mark two "sides" (points near the extreme left and right) as black.
#     # Here, we set points with |x| greater than 90% of the radius to black.
#     side_thresh = r * 0.8
#     mask = np.abs(x) >= side_thresh
#     colors[mask] = [0.0, 1.0, 0.0]
    
#     mask = z <= -side_thresh
#     colors[mask] = [1.0, 0.0, 0.0]

#     return np.concatenate([np.stack([x, y, z], axis=1), colors], axis=1)

SPHERE = generate_sphere_pcd(is_rot_colorful=True)
SPHERE_NO_COLOR = generate_sphere_pcd(is_rot_colorful=False)

def get_gripper_pcd(is_grasp, model_type):
    # make a sphere r=0.01. It turns black when the gripper is closed else white
    # is_grasp: bool, if True, return a sphere pcd with color black, else return a sphere pcd with color gray
    num_points = 128
    r = 0.01  # sphere radius, modify r if needed
    indices = np.arange(num_points, dtype=float) + 0.5
    phi = np.arccos(1 - 2 * indices / num_points)
    theta = np.pi * (1 + 5**0.5) * indices
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    colors = np.zeros((num_points, 3), dtype=np.float32) if is_grasp else np.ones((num_points, 3), dtype=np.float32) * 0.5
    SPHERE_GRIP = np.concatenate([np.stack([x, y, z], axis=1), colors], axis=1)
    if model_type == 'color_sphere':
        return np.concatenate([copy.deepcopy(SPHERE), SPHERE_GRIP], axis=0)
    elif model_type == 'grip_sphere':
        return SPHERE_GRIP
    elif model_type == 'sphere':
        return np.concatenate([copy.deepcopy(SPHERE_NO_COLOR), SPHERE_GRIP], axis=0)
    else:
        raise NotImplementedError(f"model type {model_type} not implemented")

def render_pcd_from_pose(ee_pose, gripper_qpos, fix_point_num=4412, model_type='color_sphere'):
    """
    Render the gripper point cloud at the given end effector pose.
    ee_pose has a shpae of (N, 7) where 7 means (x, y, z, qx, qy, qz, qw)
    is_add_noisy is used to add noise to the point cloud.
    """
    assert gripper_qpos.shape[-1] in [2, 6], "gripper_qpos should have a shape of (N, 2)"
    gripper_qpos = gripper_qpos.reshape(-1, gripper_qpos.shape[-1])
        
        

    B = list(ee_pose.shape[:-1])
        
    ee_pose = ee_pose.reshape(-1, 7)
    batch_size, ndim = ee_pose.shape
    pcds = []
    for i in range(batch_size):
        if gripper_qpos.shape[-1] == 2:
            is_grasp = (np.abs(gripper_qpos[i, :2]) < 0.021).any()
        elif gripper_qpos.shape[-1] == 6:
            is_grasp = gripper_qpos[i][0]>0.04
        gripper_pcd = get_gripper_pcd(is_grasp, model_type=model_type)
        tran_mat = np.eye(4)
        tran_mat[:3, 3] = ee_pose[i, :3] 
        tran_mat[:3, :3] = R.from_quat(ee_pose[i, 3:7]).as_matrix()
        pcd = apply_se3_pcd_transform(gripper_pcd, tran_mat)
        pcds.append(populate_point_num(pcd, fix_point_num))

    return np.stack(pcds).reshape([*B, fix_point_num, -1])





class EnvRobosuite(EB.EnvBase):
    """Wrapper class for robosuite environments (https://github.com/ARISE-Initiative/robosuite)"""
    def __init__(
        self, 
        env_name, 
        render=False, 
        render_offscreen=False, 
        use_image_obs=False, 
        use_depth_obs=False, 
        postprocess_visual_obs=True, 
        **kwargs,
    ):
        """
        Args:
            env_name (str): name of environment. Only needs to be provided if making a different
                environment from the one in @env_meta.

            render (bool): if True, environment supports on-screen rendering

            render_offscreen (bool): if True, environment supports off-screen rendering. This
                is forced to be True if @env_meta["use_images"] is True.

            use_image_obs (bool): if True, environment is expected to render rgb image observations
                on every env.step call. Set this to False for efficiency reasons, if image
                observations are not required.

            use_depth_obs (bool): if True, environment is expected to render depth image observations
                on every env.step call. Set this to False for efficiency reasons, if depth
                observations are not required.

            postprocess_visual_obs (bool): if True, postprocess image observations
                to prepare for learning. This should only be False when extracting observations
                for saving to a dataset (to save space on RGB images for example).
        """
        self.postprocess_visual_obs = postprocess_visual_obs
        self.use_depth_obs = use_depth_obs

        # robosuite version check
        self._is_v1 = (robosuite.__version__.split(".")[0] == "1")
        if self._is_v1:
            assert (int(robosuite.__version__.split(".")[1]) >= 2), "only support robosuite v0.3 and v1.2+"

        kwargs = deepcopy(kwargs)
        # self.main_camera = kwargs['main_camera']
        # kwargs.pop("main_camera", None)

        # update kwargs based on passed arguments
        update_kwargs = dict(
            has_renderer=render,
            has_offscreen_renderer=(render_offscreen or use_image_obs),
            ignore_done=True,
            use_object_obs=True,
            use_camera_obs=use_image_obs,
            camera_depths=True,
            camera_segmentations='instance',  # always use instance segmentation
        )
        kwargs.update(update_kwargs)

        if self._is_v1:
            if kwargs["has_offscreen_renderer"]:
                # ensure that we select the correct GPU device for rendering by testing for EGL rendering
                # NOTE: this package should be installed from this link (https://github.com/StanfordVL/egl_probe)
                import egl_probe
                valid_gpu_devices = egl_probe.get_available_devices()
                if len(valid_gpu_devices) > 0:
                    kwargs["render_gpu_device_id"] = valid_gpu_devices[0]
        else:
            # make sure gripper visualization is turned off (we almost always want this for learning)
            kwargs["gripper_visualization"] = False
            del kwargs["camera_depths"]
            kwargs["camera_depth"] = use_depth_obs # rename kwarg

        self._env_name = env_name
        self._init_kwargs = deepcopy(kwargs)
        self.env = robosuite.make(self._env_name, **kwargs)

        
        current_file_path = os.path.dirname(os.path.abspath(__file__))
        #self.SPACEVIEW_RGB_BACKGROUND = np.load(os.path.join(current_file_path, "spaceview_image_background.npy"))
        #self.SPACEVIEW_DEPTH_BACKGROUND = np.load(os.path.join(current_file_path, "spaceview_depth_background.npy"))

        if self._is_v1:
            # Make sure joint position observations and eef vel observations are active
            for ob_name in self.env.observation_names:
                if ("joint_pos" in ob_name) or ("eef_vel" in ob_name):
                    self.env.modify_observable(observable_name=ob_name, attribute="active", modifier=True)

    def step(self, action):
        """
        Step in the environment with an action.

        Args:
            action (np.array): action to take

        Returns:
            observation (dict): new observation dictionary
            reward (float): reward for this step
            done (bool): whether the task is done
            info (dict): extra information
        """
        obs, r, done, info = self.env.step(action)
        obs = self.get_observation(obs)
        return obs, r, self.is_done(), info

    def reset(self):
        """
        Reset environment.

        Returns:
            observation (dict): initial observation dictionary.
        """
        di = self.env.reset()
        return self.get_observation(di)

    def reset_to(self, state):
        """
        Reset to a specific simulator state.

        Args:
            state (dict): current simulator state that contains one or more of:
                - states (np.ndarray): initial state of the mujoco environment
                - model (str): mujoco scene xml
        
        Returns:
            observation (dict): observation dictionary after setting the simulator state (only
                if "states" is in @state)
        """
        should_ret = False
        if "model" in state:
            self.reset()
            robosuite_version_id = int(robosuite.__version__.split(".")[1])
            if robosuite_version_id <= 3:
                from robosuite.utils.mjcf_utils import postprocess_model_xml
                xml = postprocess_model_xml(state["model"])
            else:
                # v1.4 and above use the class-based edit_model_xml function
                xml = self.env.edit_model_xml(state["model"])
            self.env.reset_from_xml_string(xml)
            self.env.sim.reset()
            if not self._is_v1:
                # hide teleop visualization after restoring from model
                self.env.sim.model.site_rgba[self.env.eef_site_id] = np.array([0., 0., 0., 0.])
                self.env.sim.model.site_rgba[self.env.eef_cylinder_id] = np.array([0., 0., 0., 0.])
        if "states" in state:
            self.env.sim.set_state_from_flattened(state["states"])
            self.env.sim.forward()
            should_ret = True

        if "goal" in state:
            self.set_goal(**state["goal"])
        if should_ret:
            # only return obs if we've done a forward call - otherwise the observations will be garbage
            return self.get_observation()
        return None

    def render(self, mode="human", height=None, width=None, camera_name="agentview"):
        """
        Render from simulation to either an on-screen window or off-screen to RGB array.

        Args:
            mode (str): pass "human" for on-screen rendering or "rgb_array" for off-screen rendering
            height (int): height of image to render - only used if mode is "rgb_array"
            width (int): width of image to render - only used if mode is "rgb_array"
            camera_name (str): camera name to use for rendering
        """
        if mode == "human":
            cam_id = self.env.sim.model.camera_name2id(camera_name)
            self.env.viewer.set_camera(cam_id)
            return self.env.render()
        elif mode == "rgb_array":
            im = self.env.sim.render(height=height, width=width, camera_name=camera_name)
            if self.use_depth_obs:
                # render() returns a tuple when self.use_depth_obs=True
                return im[0][::-1]
            return im[::-1]
        else:
            raise NotImplementedError("mode={} is not implemented".format(mode))

    def get_observation(self, di=None):
        """
        Get current environment observation dictionary.

        Args:
            di (dict): current raw observation dictionary from robosuite to wrap and provide 
                as a dictionary. If not provided, will be queried from robosuite.
        """
        if di is None:
            di = self.env._get_observations(force_update=True) if self._is_v1 else self.env._get_observation()
        ret = {}
        for k in di:
            if (k in ObsUtils.OBS_KEYS_TO_MODALITIES) and ObsUtils.key_is_obs_modality(key=k, obs_modality="rgb"):
                # by default images from mujoco are flipped in height
                ret[k] = di[k][::-1]
                if self.postprocess_visual_obs:
                    ret[k] = ObsUtils.process_obs(obs=ret[k], obs_key=k)
            if (k in ObsUtils.OBS_KEYS_TO_MODALITIES) and ObsUtils.key_is_obs_modality(key=k, obs_modality="depth"):
                ret[k] = di[k][::-1]
                ret[k] = get_real_depth_map(self.env.sim, ret[k])
                if self.postprocess_visual_obs:
                    ret[k] = ObsUtils.process_obs(obs=ret[k], obs_key=k)
                    # ret[k] = clip_depth(ret[k])
            if (k in ObsUtils.OBS_KEYS_TO_MODALITIES) and ObsUtils.key_is_obs_modality(key=k, obs_modality="low_dim"):
                ret[k] = di[k].astype(np.float32)

        # "object" key contains object information
        ret["object"] = np.array(di["object-state"])
        axis = 0 if self.postprocess_visual_obs else 2
        # print(ret.keys())
        
        # if self.env.use_camera_obs:
        #     # add rgbd into obs
        #     for cam in self.env.camera_names:
        #         ret[f"{cam}_rgbd"] = np.concatenate([ret[f"{cam}_image"], ret[f"{cam}_depth"]], axis=axis)
            
            # # add gripper obs into obs
            # for goal_key in [f"{self.main_camera}_gripper_rgbd"]:
            #     robot0_eef_poss = di['robot0_eef_pos'][None, ...]
            #     raw_image = ret[goal_key.replace('_gripper', '')].copy()[None, ...]
            #     # raw_image = np.transpose(raw_image, (1,2,0))
            #     gripper_obs = convert_sideview_to_gripper_batch(self.env.sim, raw_image, goal_key, robot0_eef_poss)[0]
            #     # ret[goal_key] = np.transpose(gripper_obs[0], (2,0,1))
            #     ret[goal_key] = gripper_obs
            #     # generate fake next_gripper obs
            #     next_goal_key = goal_key.replace('gripper', 'next_gripper')
            #     ret[next_goal_key] = np.zeros_like(ret[goal_key])
            

        if self.env.use_camera_obs:

            # voxel_bound = np.array([
            #     [center[0] - ws_size/2, center[1] - ws_size/2, center[2] - 0.05],
            #     [center[0] + ws_size/2, center[1] + ws_size/2, center[2] - 0.05 + ws_size],
            # ])
            table_offset = self.env.table_offset if hasattr(self.env, 'table_offset') else [0, 0, 0.7]
            clipspace = get_clipspace(self._env_name, table_offset)
            workspace = get_workspace(self._env_name, table_offset)

            all_pcds = o3d.geometry.PointCloud()
            all_pcds_no_robot = o3d.geometry.PointCloud()
            all_rgb_no_robot_dict = dict()
            for cam_idx, camera_name in enumerate(self.env.camera_names):
                if "eye_in_hand" in camera_name:
                    continue
                cam_height = self.env.camera_heights[cam_idx]
                cam_width = self.env.camera_widths[cam_idx]
                ext_mat = get_camera_extrinsic_matrix(self.env.sim, camera_name)
                int_mat = get_camera_intrinsic_matrix(self.env.sim, camera_name, cam_height, cam_width)
                depth = di[f'{camera_name}_depth'][::-1]
                depth = get_real_depth_map(self.env.sim, depth)
                depth = depth[:, :, 0]
                color = di[f'{camera_name}_image'][::-1]
                
                #----------- get raw pcd-----------------
                cam_param = [int_mat[0, 0], int_mat[1, 1], int_mat[0, 2], int_mat[1, 2]]
                mask = np.ones_like(depth, dtype=bool)
                pcd = depth2fgpcd(depth, mask, cam_param)
                # pose = np.linalg.inv(ext_mat)
                pose = ext_mat
                # trans_pcd = pose @ np.concatenate([pcd.T, np.ones((1, pcd.shape[0]))], axis=0)
                trans_pcd = np.einsum('ij,jk->ik', pose, np.concatenate([pcd.T, np.ones((1, pcd.shape[0]))], axis=0))
                trans_pcd = trans_pcd[:3, :].T
                mask = (trans_pcd[:, 0] > clipspace[0, 0]) * (trans_pcd[:, 0] < clipspace[0, 1]) * (trans_pcd[:, 1] > clipspace[1, 0]) * (trans_pcd[:, 1] < clipspace[1, 1]) * (trans_pcd[:, 2] > clipspace[2, 0]) * (trans_pcd[:, 2] < clipspace[2, 1])
                pcd_o3d = np2o3d(trans_pcd[mask], color.reshape(-1, 3)[mask].astype(np.float64) / 255)
                if "eye_in_hand" not in camera_name:
                    all_pcds += pcd_o3d

                #----------- get raw pcd without robot-----------------
                seg = di[f'{camera_name}_segmentation_instance'][::-1][...,-1]
                robot_id = [seg.max(), seg.max()-1, seg.max()-2]  # robot is always the last 3 ids
                robot = (seg == robot_id[0]) | enlarge_mask(seg == robot_id[1], kernel_size=3) | (seg == robot_id[2])
                # robot = ((seg == robot_id[0]) | (seg == robot_id[1]) | (seg == robot_id[2]))

                depth_no_robot = depth.copy()
                rgb_no_robot = color.copy()
                rgb_no_robot[64:] = 255
                all_rgb_no_robot_dict[camera_name] = rgb_no_robot
                # if "spaceview" in camera_name:
                #     #resize SPACEVIEW_RGB_BACKGROUND
                #     self.SPACEVIEW_RGB_BACKGROUND = cv2.resize(self.SPACEVIEW_RGB_BACKGROUND, (cam_width, cam_height), interpolation=cv2.INTER_LINEAR)
                #     self.SPACEVIEW_DEPTH_BACKGROUND = cv2.resize(self.SPACEVIEW_DEPTH_BACKGROUND, (cam_width, cam_height), interpolation=cv2.INTER_LINEAR)
                #     rgb_no_robot[robot] = self.SPACEVIEW_RGB_BACKGROUND[robot] 
                #     depth_no_robot[robot] = self.SPACEVIEW_DEPTH_BACKGROUND[robot] 
                #     ret['spaceview_image_no_robot'] = rgb_no_robot.copy()
                #     ret['spaceview_depth_no_robot'] = depth_no_robot[...,None].copy()
                # else:
                #     depth_no_robot = depth.copy()
                #     depth_no_robot[robot] = 5.0  # set robot pixels to a far distance

                depth_no_robot = depth.copy()
                depth_no_robot[robot] = 5.0  # set robot pixels to a far distance

                mask = np.ones_like(depth_no_robot, dtype=bool)
                pcd = depth2fgpcd(depth_no_robot, mask, cam_param)
                trans_pcd = np.einsum('ij,jk->ik', pose, np.concatenate([pcd.T, np.ones((1, pcd.shape[0]))], axis=0))
                trans_pcd = trans_pcd[:3, :].T
                mask = (trans_pcd[:, 0] > workspace[0, 0]) * (trans_pcd[:, 0] < workspace[0, 1]) * (trans_pcd[:, 1] > workspace[1, 0]) * (trans_pcd[:, 1] < workspace[1, 1]) * (trans_pcd[:, 2] > workspace[2, 0]) * (trans_pcd[:, 2] < workspace[2, 1])
                pcd_o3d = np2o3d(trans_pcd[mask], color.reshape(-1, 3)[mask].astype(np.float64) / 255)
                if "eye_in_hand" not in camera_name:
                    all_pcds_no_robot += pcd_o3d

            # get raw pcd with robot
            np_pcd = o3d2np(all_pcds)
            eef_pos = np.concatenate([di['robot0_eef_pos'], di['robot0_eef_quat']], axis=-1)

            use_voxel = False
            if use_voxel:
                np_pcd = np_pcd[np_pcd[:,2]>0.82]
                # np_pcd = sample_pcd(np_pcd, 1024)
                np_pcd_se3_rel = localize_pcd_batch(np_pcd[None,...], eef_pos, local_type='xyz')[0]
                local_voxel = pcd_to_voxel(np_pcd_se3_rel[None,...], 'gripper')[0]

                ret['voxels'] = pcd_to_voxel(np_pcd[None,...])[0]
                ret['rel_voxels'] = pcd_to_voxel(np_pcd_se3_rel[None,...], 'relative')[0]
                ret['local_voxel'] = local_voxel

                # # get render pcd
                # np_pcd_no_robot = o3d2np(all_pcds_no_robot)
                # color_geco = render_pcd_from_pose(eef_pos, di['robot0_gripper_qpos'], 1024, 'color_sphere')
                # pcd_render = np.concatenate([np_pcd_no_robot, color_geco], axis=0)

                # geco = render_pcd_from_pose(eef_pos, di['robot0_gripper_qpos'], 1024, 'grip_sphere')
                # np_pcd_no_robot = preprocess_pcd(np_pcd_no_robot)
                # pcd_render_geco = np.concatenate([np_pcd_no_robot, geco], axis=0)
                # local_pcd_renders = localize_pcd_batch(pcd_render_geco[None,...], eef_pos, local_type='xyz')
                # local_voxel_render = pcd_to_voxel(local_pcd_renders, 'gripper')[0]

                # ret['render_voxels'] = pcd_to_voxel(pcd_render[None,...])[0]
                # ret['rel_render_voxels'] = pcd_to_voxel(local_pcd_renders, 'relative')[0]
                # ret['local_render_voxel'] = local_voxel_render
            else:
                pcd_z_min = get_pcd_z_min(self._env_name, table_offset)
                np_pcd = np_pcd[np_pcd[:,2]>pcd_z_min]
                np_pcd_se3_rel = localize_pcd_batch(np_pcd[None,...], eef_pos, local_type='xyz')[0]
                ret['pcd'] = crop_pcd(np_pcd, input_type='absolute')
                # ret['is_contact'] = np.array([False]) # placeholder
                # ret['pcd_t3'] = crop_pcd(np_pcd_se3_rel, input_type='relative')
                # ret['local_pcd_t3'] = crop_pcd(np_pcd_se3_rel, input_type='gripper')
                # ret['local_pcd_se3'] = crop_pcd(localize_pcd_batch(np_pcd[None,...], eef_pos, local_type='se3')[0], input_type='gripper_se3')

                # color_geco = render_pcd_from_pose(eef_pos, di['robot0_gripper_qpos'], 1024, 'color_sphere')
                # np_pcd_no_robot = o3d2np(all_pcds_no_robot)
                # pcd_render = np.concatenate([np_pcd_no_robot, color_geco], axis=0)
                # ret['render_pcd'] = crop_pcd(pcd_render, input_type='absolute')


            

        if self._is_v1:
            for robot in self.env.robots:
                # add all robot-arm-specific observations. Note the (k not in ret) check
                # ensures that we don't accidentally add robot wrist images a second time
                pf = robot.robot_model.naming_prefix
                for k in di:
                    if k.startswith(pf) and (k not in ret) and \
                            (not k.endswith("proprio-state")):
                        ret[k] = np.array(di[k])
        else:
            # minimal proprioception for older versions of robosuite
            ret["proprio"] = np.array(di["robot-state"])
            ret["eef_pos"] = np.array(di["eef_pos"])
            ret["eef_quat"] = np.array(di["eef_quat"])
            ret["gripper_qpos"] = np.array(di["gripper_qpos"])
        return ret

    def get_real_depth_map(self, depth_map):
        """
        Reproduced from https://github.com/ARISE-Initiative/robosuite/blob/c57e282553a4f42378f2635b9a3cbc4afba270fd/robosuite/utils/camera_utils.py#L106
        since older versions of robosuite do not have this conversion from normalized depth values returned by MuJoCo
        to real depth values.
        """
        # Make sure that depth values are normalized
        assert np.all(depth_map >= 0.0) and np.all(depth_map <= 1.0)
        extent = self.env.sim.model.stat.extent
        far = self.env.sim.model.vis.map.zfar * extent
        near = self.env.sim.model.vis.map.znear * extent
        return near / (1.0 - depth_map * (1.0 - near / far))

    def get_camera_intrinsic_matrix(self, camera_name, camera_height, camera_width):
        """
        Obtains camera intrinsic matrix.
        Args:
            camera_name (str): name of camera
            camera_height (int): height of camera images in pixels
            camera_width (int): width of camera images in pixels
        Return:
            K (np.array): 3x3 camera matrix
        """
        cam_id = self.env.sim.model.camera_name2id(camera_name)
        fovy = self.env.sim.model.cam_fovy[cam_id]
        f = 0.5 * camera_height / np.tan(fovy * np.pi / 360)
        K = np.array([[f, 0, camera_width / 2], [0, f, camera_height / 2], [0, 0, 1]])
        return K

    def get_camera_extrinsic_matrix(self, camera_name):
        """
        Returns a 4x4 homogenous matrix corresponding to the camera pose in the
        world frame. MuJoCo has a weird convention for how it sets up the
        camera body axis, so we also apply a correction so that the x and y
        axis are along the camera view and the z axis points along the
        viewpoint.
        Normal camera convention: https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html
        Args:
            camera_name (str): name of camera
        Return:
            R (np.array): 4x4 camera extrinsic matrix
        """
        cam_id = self.env.sim.model.camera_name2id(camera_name)
        camera_pos = self.env.sim.data.cam_xpos[cam_id]
        camera_rot = self.env.sim.data.cam_xmat[cam_id].reshape(3, 3)
        R = T.make_pose(camera_pos, camera_rot)

        # IMPORTANT! This is a correction so that the camera axis is set up along the viewpoint correctly.
        camera_axis_correction = np.array(
            [[1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
        )
        R = R @ camera_axis_correction
        return R

    def get_camera_transform_matrix(self, camera_name, camera_height, camera_width):
        """
        Camera transform matrix to project from world coordinates to pixel coordinates.
        Args:
            camera_name (str): name of camera
            camera_height (int): height of camera images in pixels
            camera_width (int): width of camera images in pixels
        Return:
            K (np.array): 4x4 camera matrix to project from world coordinates to pixel coordinates
        """
        R = self.get_camera_extrinsic_matrix(camera_name=camera_name)
        K = self.get_camera_intrinsic_matrix(
            camera_name=camera_name, camera_height=camera_height, camera_width=camera_width
        )
        K_exp = np.eye(4)
        K_exp[:3, :3] = K

        # Takes a point in world, transforms to camera frame, and then projects onto image plane.
        return K_exp @ T.pose_inv(R)

    def get_state(self):
        """
        Get current environment simulator state as a dictionary. Should be compatible with @reset_to.
        """
        xml = self.env.sim.model.get_xml() # model xml file
        state = np.array(self.env.sim.get_state().flatten()) # simulator state
        return dict(model=xml, states=state)

    def get_reward(self):
        """
        Get current reward.
        """
        return self.env.reward()

    def get_goal(self):
        """
        Get goal observation. Not all environments support this.
        """
        return self.get_observation(self.env._get_goal())

    def set_goal(self, **kwargs):
        """
        Set goal observation with external specification. Not all environments support this.
        """
        return self.env.set_goal(**kwargs)

    def is_done(self):
        """
        Check if the task is done (not necessarily successful).
        """

        # Robosuite envs always rollout to fixed horizon.
        return False

    def is_success(self):
        """
        Check if the task condition(s) is reached. Should return a dictionary
        { str: bool } with at least a "task" key for the overall task success,
        and additional optional keys corresponding to other task criteria.
        """
        succ = self.env._check_success()
        if isinstance(succ, dict):
            assert "task" in succ
            return succ
        return { "task" : succ }

    @property
    def action_dimension(self):
        """
        Returns dimension of actions (int).
        """
        return self.env.action_spec[0].shape[0]

    @property
    def name(self):
        """
        Returns name of environment name (str).
        """
        return self._env_name

    @property
    def type(self):
        """
        Returns environment type (int) for this kind of environment.
        This helps identify this env class.
        """
        return EB.EnvType.ROBOSUITE_TYPE

    @property
    def version(self):
        """
        Returns version of robosuite used for this environment, eg. 1.2.0
        """
        return robosuite.__version__

    def serialize(self):
        """
        Save all information needed to re-instantiate this environment in a dictionary.
        This is the same as @env_meta - environment metadata stored in hdf5 datasets,
        and used in utils/env_utils.py.
        """
        return dict(
            env_name=self.name,
            env_version=self.version,
            type=self.type,
            env_kwargs=deepcopy(self._init_kwargs)
        )

    @classmethod
    def create_for_data_processing(
        cls, 
        env_name, 
        camera_names, 
        camera_height, 
        camera_width, 
        reward_shaping, 
        render=None, 
        render_offscreen=None, 
        use_image_obs=None, 
        use_depth_obs=None, 
        **kwargs,
    ):
        """
        Create environment for processing datasets, which includes extracting
        observations, labeling dense / sparse rewards, and annotating dones in
        transitions. 

        Args:
            env_name (str): name of environment
            camera_names (list of str): list of camera names that correspond to image observations
            camera_height (int): camera height for all cameras
            camera_width (int): camera width for all cameras
            reward_shaping (bool): if True, use shaped environment rewards, else use sparse task completion rewards
            render (bool or None): optionally override rendering behavior. Defaults to False.
            render_offscreen (bool or None): optionally override rendering behavior. The default value is True if
                @camera_names is non-empty, False otherwise.
            use_image_obs (bool or None): optionally override rendering behavior. The default value is True if
                @camera_names is non-empty, False otherwise.
            use_depth_obs (bool): if True, use depth observations
        """
        is_v1 = (robosuite.__version__.split(".")[0] == "1")
        has_camera = (len(camera_names) > 0)

        new_kwargs = {
            "reward_shaping": reward_shaping,
        }

        if has_camera:
            if is_v1:
                new_kwargs["camera_names"] = list(camera_names)
                new_kwargs["camera_heights"] = camera_height
                new_kwargs["camera_widths"] = camera_width
            else:
                assert len(camera_names) == 1
                if has_camera:
                    new_kwargs["camera_name"] = camera_names[0]
                    new_kwargs["camera_height"] = camera_height
                    new_kwargs["camera_width"] = camera_width

        kwargs.update(new_kwargs)

        # also initialize obs utils so it knows which modalities are image modalities
        image_modalities = list(camera_names)
        depth_modalities = list(camera_names)
        if is_v1:
            image_modalities = ["{}_image".format(cn) for cn in camera_names]
            depth_modalities = ["{}_depth".format(cn) for cn in camera_names]
        elif has_camera:
            # v0.3 only had support for one image, and it was named "image"
            assert len(image_modalities) == 1
            image_modalities = ["image"]
            depth_modalities = ["depth"]
        obs_modality_specs = {
            "obs": {
                "low_dim": [], # technically unused, so we don't have to specify all of them
                "rgb": image_modalities,
                "depth": depth_modalities,
            }
        }
        if use_depth_obs:
            obs_modality_specs["obs"]["depth"] = depth_modalities
        ObsUtils.initialize_obs_utils_with_obs_specs(obs_modality_specs)

        # note that @postprocess_visual_obs is False since this env's images will be written to a dataset
        return cls(
            env_name=env_name,
            render=(False if render is None else render), 
            render_offscreen=(has_camera if render_offscreen is None else render_offscreen), 
            use_image_obs=(has_camera if use_image_obs is None else use_image_obs), 
            use_depth_obs=use_depth_obs,
            postprocess_visual_obs=False,
            **kwargs,
        )

    @property
    def rollout_exceptions(self):
        """
        Return tuple of exceptions to except when doing rollouts. This is useful to ensure
        that the entire training run doesn't crash because of a bad policy that causes unstable
        simulation computations.
        """
        return tuple(MUJOCO_EXCEPTIONS)

    @property
    def base_env(self):
        """
        Grabs base simulation environment.
        """
        return self.env

    def __repr__(self):
        """
        Pretty-print env description.
        """
        return self.name + "\n" + json.dumps(self._init_kwargs, sort_keys=True, indent=4)
    

    def get_camera_info(
        self,
        camera_names=None, 
        camera_height=84, 
        camera_width=84,
    ):
        """
        Helper function to get camera intrinsics and extrinsics for cameras being used for observations.
        """

        # TODO: make this function more general than just robosuite environments

        if camera_names is None:
            return None

        camera_info = dict()
        for cam_name in camera_names:
            K = get_camera_intrinsic_matrix(self.env.sim, camera_name=cam_name, camera_height=camera_height, camera_width=camera_width)
            R = get_camera_extrinsic_matrix(self.env.sim, camera_name=cam_name) # camera pose in world frame
            if "eye_in_hand" in cam_name:
                # convert extrinsic matrix to be relative to robot eef control frame
                assert cam_name.startswith("robot0")
                eef_site_name = self.env.robots[0].controller.eef_name
                eef_pos = np.array(self.env.sim.data.site_xpos[self.env.sim.model.site_name2id(eef_site_name)])
                eef_rot = np.array(self.env.sim.data.site_xmat[self.env.sim.model.site_name2id(eef_site_name)].reshape([3, 3]))
                eef_pose = np.zeros((4, 4)) # eef pose in world frame
                eef_pose[:3, :3] = eef_rot
                eef_pose[:3, 3] = eef_pos
                eef_pose[3, 3] = 1.0
                eef_pose_inv = np.zeros((4, 4))
                eef_pose_inv[:3, :3] = eef_pose[:3, :3].T
                eef_pose_inv[:3, 3] = -eef_pose_inv[:3, :3].dot(eef_pose[:3, 3])
                eef_pose_inv[3, 3] = 1.0
                R = R.dot(eef_pose_inv) # T_E^W * T_W^C = T_E^C
            camera_info[cam_name] = dict(
                intrinsics=K.tolist(),
                extrinsics=R.tolist(),
            )
        return camera_info


