# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import RewardTermCfg as RewTerm
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns

import numpy as np


##
# Pre-defined configs
##
from isaaclab_assets.robots.eigenbot import EIGENBOT_CFG 


@configclass 
class EigenbotRoughEnvCfg(LocomotionVelocityRoughEnvCfg): # eigen TODO: change specific properties in the parent class, eg. reward function, panelty/ileagal contacts, any body_names
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # switch robot to anymal-d
        self.scene.robot = EIGENBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.init_state.joint_pos={
            'bendy_joint_M1_S1': -np.pi/4,   # [rad]
            'bendy_joint_M2_S2': 0.0,   # [rad]
            'bendy_joint_M3_S3': np.pi/4,
            'bendy_joint_M4_S4': -np.pi/4,
            'bendy_joint_M5_S5': 0.0,
            'bendy_joint_M6_S6': np.pi/4,
            'bendy_joint_M7_S7': np.pi/4,
            'bendy_joint_M8_S8': np.pi/4,
            'bendy_joint_M9_S9': np.pi/4,
            'bendy_joint_M10_S10': np.pi/4,
            'bendy_joint_M11_S11': np.pi/4,
            'bendy_joint_M12_S12': np.pi/4,
            'bendy_joint_M13_S13': np.pi/4,
            'bendy_joint_M14_S14': np.pi/4,
            'bendy_joint_M15_S15': np.pi/4,
            'bendy_joint_M16_S16': np.pi/4,
            'bendy_joint_M17_S17': np.pi/4,
            'bendy_joint_M18_S18': np.pi/4,
        }
        # self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/eigenbody_M0_S0"
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base_link"
        # self.scene.height_scanner.offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0))

        

        # scale down the terrains because the robot is small
        self.scene.terrain.terrain_generator.sub_terrains["boxes"].grid_height_range = (0.01, 0.1)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_range = (0.01, 0.1)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_step = 0.01

        self.scene.terrain.terrain_generator.sub_terrains["hf_pyramid_slope"].slope_range = (0.0, 0.1)
        self.scene.terrain.terrain_generator.sub_terrains["hf_pyramid_slope_inv"].slope_range = (0.0, 0.1)

        # reduce action scale
        self.actions.joint_pos.scale = 0.25

        # event
        # self.events.push_robot = None
        self.events.add_base_mass.params["mass_distribution_params"] = (-1.0, 1.0) # eigen TODO what should this be?
        # self.events.add_base_mass.params["asset_cfg"].body_names = "eigenbody_M0_S0"
        # self.events.base_external_force_torque.params["asset_cfg"].body_names = "eigenbody_M0_S0"
        self.events.add_base_mass.params["asset_cfg"].body_names = "base_link"
        self.events.base_external_force_torque.params["asset_cfg"].body_names = "base_link"

        self.events.reset_robot_joints.params["position_range"] = (0.9, 1.1) # eigen TODO what should this be?
        self.events.reset_base.params = {
            "pose_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.0, 0.0),
                "y": (-0.0, 0.0),
                "z": (-0.0, 0.0),
                "roll": (-0.0, 0.0),
                "pitch": (-0.0, 0.0),
                "yaw": (0,0),
            },
        }
        

        # rewards
        foot_contact_threshold = 0.5
        undesired_contact_threshold = 0.5
        self.rewards.flat_orientation_l2.weight = -0.2#-0.5#-5.0
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = "foot_input_.*"
        self.rewards.feet_air_time.weight = 0.5 #0.6 # 0.87
        # self.rewards.undesired_contacts = None
        # interlimb rules
        self.rewards.rule3 = RewTerm(func=mdp.reward_rule_3, weight=0.2,params={"sensor_cfg": 
                                                                            SceneEntityCfg("contact_forces", 
                                                                            body_names="foot_input_.*"), 
                                                                            "threshold": foot_contact_threshold})

        self.rewards.rule1 = RewTerm(func=mdp.reward_rule_1, weight=0.35,params={"sensor_cfg": 
                                                                            SceneEntityCfg("contact_forces", 
                                                                            body_names="foot_input_.*"), 
                                                                            "threshold": foot_contact_threshold})

        self.rewards.dof_torques_l2.weight = -0.00001#-0.0002#-0.0001 previously -0.0002 causes nan values in back prop

        self.rewards.tracking_goal_vel.weight = 2.5
        self.rewards.delta_yaw.weight = 1.2#2.0

        self.rewards.dof_acc_l2.weight = -2.5e-7
        self.rewards.action_rate_l2.weight = -0.01#-0.1
        self.rewards.lin_vel_z_l2.weight = -0.5#-1.5
        # self.rewards.energy_penalty = RewTerm(func=mdp.penalty_energy, weight=-0.0005,params={"command_name": "base_velocity", "threshold": 0.1})

        self.rewards.undesired_contacts.params["sensor_cfg"] = SceneEntityCfg(
            "contact_forces", body_names=["bendy_input_.*","bendy_output_.*"]
        )
        self.rewards.undesired_contacts.params["threshold"] = undesired_contact_threshold
        self.rewards.undesired_contacts.weight = -1.0
        self.rewards.base_height_l2 = RewTerm(
            func=mdp.base_height_l2,
            weight= -0.25,
            # weight= -0.4,#-0.25,
            params={"sensor_cfg": SceneEntityCfg("height_scanner"), "target_height": 0.2},#0.25},
        )

        
        self.rewards.reward_stand_still = RewTerm(
            func=mdp.reward_stand_still, weight=-0.5, params={"command_name": "base_velocity", "threshold": 0.1}
        )
        # self.rewards.reward_slide = RewTerm(
        #     func=mdp.feet_slide, weight=-1.0, params={"sensor_cfg": SceneEntityCfg(
        #     "contact_forces", body_names="foot_input_.*"), "asset_cfg": SceneEntityCfg(
        #     "robot", body_names="foot_input_.*"), "threshold": 0.5}
        # )
        # def feet_slide(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), threshold: float = 1.0) -> torch.Tensor:



        # termination
        self.terminations.base_contact.params["sensor_cfg"] = SceneEntityCfg("contact_forces", body_names=["base_link"])#, "eigenbody_M0_S0"])
            # "bendy_input_M1_S1","bendy_input_M2_S2","bendy_input_M3_S3","bendy_input_M4_S4","bendy_input_M5_S5","bendy_input_M6_S6"]),
            # "bendy_output_M1_S1","bendy_output_M2_S2","bendy_output_M3_S3","bendy_output_M4_S4","bendy_output_M5_S5","bendy_output_M6_S6"])
        self.terminations.base_contact.params["threshold"] = 1
        self.termination_panelty = RewTerm(
            func=mdp.is_terminated_term,
            # weight=-0.5,
            weight=-1,
            # params={"term_keys": "object_out_of_reach"},
        )
        # commands
        self.commands.base_velocity.ranges.lin_vel_x = (0.2, 0.5)#(0.35, 0.7) #(0.2,0.7)
        # rel_heading_envs
        self.commands.base_velocity.rel_heading_envs = 1.0
@configclass
class EigenbotRoughEnvCfg_PLAY(EigenbotRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 15
        self.scene.env_spacing = 2.5
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None
