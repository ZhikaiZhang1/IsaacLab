# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

from isaaclab.utils import configclass


@configclass
# eigen TODO: change based on eigenbot 
class EigenbotRoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 5000
    save_interval = 50
    experiment_name = "eigenbot_rough"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    load_opt = True
    # value_loss_coef = 1.0
    #     use_clipped_value_loss = True
    #     clip_param = 0.2
    #     entropy_coef = 0.01
    #     num_learning_epochs = 5
    #     num_mini_batches = 4 # mini batch size = num_envs*nsteps / nminibatches
    #     learning_rate = 1.e-3 #5.e-4
    #     schedule = 'adaptive' # could be adaptive, fixed
    #     gamma = 0.99
    #     lam = 0.95
    #     desired_kl = 0.01
    #     max_grad_norm = 1.
    clip_actions = None#3.14
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2, # 0.2
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        init_weights=[False,False],
        init_weight_scales_policy = [0.01,0.01,0.01,0.01],
        init_weight_scales_value = [1.0,1.0,1.0,1.0],
        optimizer_choice = "Adam",
        # weight_decay_val = 1e-5,

    )
    # to try out: 1. reward, 2. even smaller learning rate, 3. layernorm, 4. # ppo updates, 5. max_grad_norm 6. train from scratch


@configclass
class EigenbotFlatPPORunnerCfg(EigenbotRoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 5000#300
        self.experiment_name = "eigenbot_flat"
        # self.policy.actor_hidden_dims = [128, 128, 128]
        # self.policy.critic_hidden_dims = [128, 128, 128]
