# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym import LEGGED_GYM_ROOT_DIR
import copy
import os

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger
from legged_gym.utils.helpers import PolicyExporterHIM

import numpy as np
import torch


class OnnxPolicyWrapper:
    def __init__(self, policy_path, device):
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("onnxruntime is required to load ONNX policies.") from exc
        preferred_providers = ['CUDAExecutionProvider', 'ROCMExecutionProvider', 'CPUExecutionProvider']
        available = ort.get_available_providers()
        providers = [p for p in preferred_providers if p in available]
        if not providers:
            providers = available
        self.session = ort.InferenceSession(policy_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.device = device

    def __call__(self, obs_tensor):
        obs_np = obs_tensor.detach().to('cpu').numpy().astype(np.float32)
        actions = self.session.run([self.output_name], {self.input_name: obs_np})[0]
        return torch.from_numpy(actions).to(self.device)


def export_policy_as_onnx(actor_critic, path, example_obs):
    if example_obs is None:
        raise ValueError("An example observation tensor is required to export the policy to ONNX.")
    os.makedirs(path, exist_ok=True)
    onnx_path = os.path.join(path, 'policy.onnx')
    if hasattr(actor_critic, 'estimator'):
        exporter = PolicyExporterHIM(actor_critic)
    else:
        exporter = copy.deepcopy(actor_critic.actor)
    exporter = exporter.to('cpu').eval()
    dummy_input = example_obs[:1].detach().to('cpu')
    torch.onnx.export(
        exporter,
        dummy_input,
        onnx_path,
        input_names=['obs_history'],
        output_names=['actions'],
        dynamic_axes={'obs_history': {0: 'batch'}, 'actions': {0: 'batch'}},
        opset_version=14
    )
    print('Exported policy as onnx to: ', onnx_path)


def _apply_symmetric_limit(range_pair, limit):
    if limit is None:
        return list(range_pair)
    limit = abs(limit)
    return [max(range_pair[0], -limit), min(range_pair[1], limit)]


def _get_latest_run_dir(train_cfg):
    log_root = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name)
    runs = [d for d in os.listdir(log_root) if os.path.isdir(os.path.join(log_root, d)) and d != 'exported']
    if not runs:
        raise FileNotFoundError(f"No runs found under {log_root}")
    runs.sort()
    latest_run = runs[-1]
    return os.path.join(log_root, latest_run)


def play(args,
         load_jit_policy=False,
         load_onnx_policy=False,
         onnx_policy_path=None,
         lin_vel_x_limit=None,
         lin_vel_y_limit=None,
         ang_vel_yaw_limit=None):
    if load_jit_policy and load_onnx_policy:
        raise ValueError("Cannot load JIT and ONNX policies at the same time.")
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 50)
    env_cfg.terrain.num_rows = 10
    env_cfg.terrain.num_cols = 8
    env_cfg.terrain.curriculum = True
    env_cfg.terrain.max_init_terrain_level = 9
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.disturbance = False
    env_cfg.domain_rand.randomize_payload_mass = False
    env_cfg.commands.heading_command = False
    # env_cfg.terrain.mesh_type = 'plane'
    env_cfg.commands.ranges.lin_vel_x = _apply_symmetric_limit(env_cfg.commands.ranges.lin_vel_x, lin_vel_x_limit)
    env_cfg.commands.ranges.lin_vel_y = _apply_symmetric_limit(env_cfg.commands.ranges.lin_vel_y, lin_vel_y_limit)
    env_cfg.commands.ranges.ang_vel_yaw = _apply_symmetric_limit(env_cfg.commands.ranges.ang_vel_yaw, ang_vel_yaw_limit)
    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    # initial command
    env.commands[:, 0] = 0.
    env.commands[:, 1] = 0.
    env.commands[:, 2] = 0.

    obs = env.get_observations()
    if load_jit_policy:
        policy_path = os.path.join(_get_latest_run_dir(train_cfg), 'policy.pt')
        print(f"Loading JIT policy from: {policy_path}")
        policy = torch.jit.load(policy_path, map_location=env.device)
    elif load_onnx_policy:
        if onnx_policy_path is None:
            policy_path = os.path.join(_get_latest_run_dir(train_cfg), 'exported', 'policies', 'policy.onnx')
        else:
            policy_path = onnx_policy_path
        if not os.path.isfile(policy_path):
            raise FileNotFoundError(f"ONNX policy not found at {policy_path}")
        print(f"Loading ONNX policy from: {policy_path}")
        policy = OnnxPolicyWrapper(policy_path, env.device)
    else:
        # load policy
        train_cfg.runner.resume = True
        ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
        policy = ppo_runner.get_inference_policy(device=env.device)
        # export policy as a jit module (used to run it from C++)
        if EXPORT_POLICY:
            path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
            export_policy_as_jit(ppo_runner.alg.actor_critic, path)
            print('Exported policy as jit script to: ', path)
            export_policy_as_onnx(ppo_runner.alg.actor_critic, path, obs[:1])

    logger = Logger(env.dt)
    robot_index = 0 # which robot is used for logging
    joint_index = 1 # which joint is used for logging
    stop_state_log = 100 # number of steps before plotting states
    stop_rew_log = env.max_episode_length + 1 # number of steps before print average episode rewards
    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    camera_vel = np.array([1., 1., 0.])
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    img_idx = 0

    resampling_time = env_cfg.commands.resampling_time
    resampling_step = int(resampling_time / env.dt)
    lin_vel_x_range = env_cfg.commands.ranges.lin_vel_x
    lin_vel_y_range = env_cfg.commands.ranges.lin_vel_y
    heading_range = getattr(env_cfg.commands.ranges, "heading", [-np.pi, np.pi])
    ang_vel_yaw_range = env_cfg.commands.ranges.ang_vel_yaw

    def resample_commands(env_ids):
        if env_ids.numel() == 0:
            return
        env_ids = env_ids.to(device=env.device, dtype=torch.long)
        count = env_ids.shape[0]
        env.commands[env_ids, 0] = torch.empty(count, device=env.device).uniform_(lin_vel_x_range[0], lin_vel_x_range[1])
        env.commands[env_ids, 1] = torch.empty(count, device=env.device).uniform_(lin_vel_y_range[0], lin_vel_y_range[1])
        if env_cfg.commands.heading_command:
            env.commands[env_ids, 3] = torch.empty(count, device=env.device).uniform_(heading_range[0], heading_range[1])
            env.commands[env_ids, 2] = 0.
        else:
            env.commands[env_ids, 2] = torch.empty(count, device=env.device).uniform_(ang_vel_yaw_range[0], ang_vel_yaw_range[1])

    full_env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    resample_commands(full_env_ids)

    for i in range(10*int(env.max_episode_length)):
        if i % resampling_step == 0:
            resample_commands(full_env_ids)
        actions = policy(obs.detach())
        obs, _, rews, dones, infos, _, _ = env.step(actions.detach())
        reset_ids = torch.nonzero(dones, as_tuple=False).flatten()
        if reset_ids.numel() > 0:
            resample_commands(reset_ids)

        if RECORD_FRAMES:
            if i % 2:
                filename = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'frames', f"{img_idx}.png")
                env.gym.write_viewer_image_to_file(env.viewer, filename)
                img_idx += 1 
        if MOVE_CAMERA:
            camera_position += camera_vel * env.dt
            env.set_camera(camera_position, camera_position + camera_direction)

        if i < stop_state_log:
            logger.log_states(
                {
                    'dof_pos_target': actions[robot_index, joint_index].item() * env.cfg.control.action_scale + env.default_dof_pos[robot_index, joint_index].item(),
                    'dof_pos': env.dof_pos[robot_index, joint_index].item(),
                    'dof_vel': env.dof_vel[robot_index, joint_index].item(),
                    'dof_torque': env.torques[robot_index, joint_index].item(),
                    'command_x': env.commands[robot_index, 0].item(),
                    'command_y': env.commands[robot_index, 1].item(),
                    'command_yaw': env.commands[robot_index, 2].item(),
                    'base_vel_x': env.base_lin_vel[robot_index, 0].item(),
                    'base_vel_y': env.base_lin_vel[robot_index, 1].item(),
                    'base_vel_z': env.base_lin_vel[robot_index, 2].item(),
                    'base_vel_yaw': env.base_ang_vel[robot_index, 2].item(),
                    'contact_forces_z': env.contact_forces[robot_index, env.feet_indices, 2].cpu().numpy()
                }
            )
        elif i==stop_state_log:
            logger.plot_states()
        if  0 < i < stop_rew_log:
            if infos["episode"]:
                num_episodes = torch.sum(env.reset_buf).item()
                if num_episodes>0:
                    logger.log_rewards(infos["episode"], num_episodes)
        elif i==stop_rew_log:
            logger.print_rewards()

if __name__ == '__main__':
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    LOAD_JIT_POLICY = False
    LOAD_ONNX_POLICY = True# False
    ONNX_POLICY_PATH = "/home/lch/HIMLOCO_GO2/legged_gym/logs/rough_go2/exported/policies/policy.onnx"
    LIN_VEL_X_LIMIT = 1
    LIN_VEL_Y_LIMIT = 0
    ANG_VEL_YAW_LIMIT = 0
    args = get_args()
    play(
        args,
        load_jit_policy=LOAD_JIT_POLICY,
        load_onnx_policy=LOAD_ONNX_POLICY,
        onnx_policy_path=ONNX_POLICY_PATH,
        lin_vel_x_limit=LIN_VEL_X_LIMIT,
        lin_vel_y_limit=LIN_VEL_Y_LIMIT,
        ang_vel_yaw_limit=ANG_VEL_YAW_LIMIT,
    )
