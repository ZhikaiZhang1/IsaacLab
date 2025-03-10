# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import RewardTermCfg as RewTerm
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp


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
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/eigenbody_M0_S0"
        

        # scale down the terrains because the robot is small
        self.scene.terrain.terrain_generator.sub_terrains["boxes"].grid_height_range = (0.01, 0.06)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_range = (0.005, 0.03)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_step = 0.005

        # reduce action scale
        self.actions.joint_pos.scale = 0.25

        # event
        # self.events.push_robot = None
        self.events.add_base_mass.params["mass_distribution_params"] = (-1.0, 1.0) # eigen TODO what should this be?
        self.events.add_base_mass.params["asset_cfg"].body_names = "eigenbody_M0_S0"
        self.events.base_external_force_torque.params["asset_cfg"].body_names = "eigenbody_M0_S0"
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0) # eigen TODO what should this be?
        self.events.reset_base.params = {
            "pose_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0), "yaw": (-1.2, 1.2)},#"yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        }

        # rewards
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = "foot_input_.*"
        self.rewards.feet_air_time.weight = 0.87
        # self.rewards.undesired_contacts = None
        self.rewards.dof_torques_l2.weight = -0.0002#-0.0001 previously -0.0002 causes nan values in back prop
        self.rewards.track_lin_vel_xy_exp.weight = 2.0
        self.rewards.track_ang_vel_z_exp.weight = 0.5
        self.rewards.dof_acc_l2.weight = -2.5e-7
        self.rewards.action_rate_l2.weight = -0.1
        self.rewards.lin_vel_z_l2.weight = -1.0

        self.rewards.flat_orientation_l2.weight = -0.3


        self.rewards.undesired_contacts.params["sensor_cfg"] = SceneEntityCfg(
            "contact_forces", body_names=["bendy_input_M18_S18","bendy_input_M17_S17","bendy_input_M16_S16",
                                          "bendy_input_M15_S15","bendy_input_M14_S14","bendy_input_M13_S13",
                                          "bendy_input_M12_S12","bendy_input_M11_S11","bendy_input_M10_S10",
                                          "bendy_input_M9_S9","bendy_input_M8_S8","bendy_input_M7_S7",
                                          "bendy_input_M6_S6","bendy_input_M5_S5","bendy_input_M4_S4",
                                          "bendy_input_M3_S3","bendy_input_M2_S2","bendy_input_M1_S1"]
        )
        self.rewards.undesired_contacts.params["threshold"] = 0.1
        self.rewards.base_height_l2 = RewTerm(
            func=mdp.base_height_l2,
            weight=-0.15,
            params={"sensor_cfg": SceneEntityCfg("height_scanner"), "target_height": 0.15},
        )
        self.rewards.reward_stand_still = RewTerm(
            func=mdp.reward_stand_still, weight=-0.5, params={"command_name": "base_velocity", "threshold": 0.1}
        )


        # termination
        self.terminations.base_contact.params["sensor_cfg"] = SceneEntityCfg("contact_forces", body_names=["base_link", "eigenbody_M0_S0"])

        # commands
        self.commands.base_velocity.ranges.lin_vel_x = (0.2, 0.7)
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
