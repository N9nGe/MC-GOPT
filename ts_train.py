""" 
refer:
- https://github.com/albertcity/OCARL
- https://github.com/pioneer-innovation/Real-3D-Embodied-Dataset

"""
import sys
import os
curr_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(curr_path)  
sys.path.append(parent_path) 
import warnings
warnings.filterwarnings("ignore")

import time
import pprint
import shutil
import random

import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR, ExponentialLR
import tianshou as ts
from tianshou.utils import TensorboardLogger, LazyLogger
from tianshou.data import VectorReplayBuffer
from tianshou.utils.net.common import ActorCritic, DataParallelNet
from tianshou.trainer import onpolicy_trainer

import model
import arguments
from tools import *
from masked_ppo import MaskedPPOPolicy
from masked_a2c import MaskedA2CPolicy
from mycollector import PackCollector
from envs.Packing.multiSizeWrapper import MultiSizeWrapper
from curriculum import CurriculumScheduler
from envs.Packing.curriculumBoxCreator import CurriculumBoxCreator, set_global_difficulty
 

def make_envs(args):
    # Check if multi-size training is enabled (Objective 2)
    use_multi_size = args.env.get('bin_sizes') is not None
    bin_sizes = args.env.bin_sizes if use_multi_size else None

    # Use first bin size as base for environment creation in multi-size mode
    base_container_size = bin_sizes[0] if use_multi_size else args.env.container_size

    def make_train_env():
        env = gym.make(args.env.id,
                      container_size=base_container_size,
                      enable_rotation=args.env.rot,
                      data_type=args.env.box_type,
                      item_set=args.env.box_size_set,
                      reward_type=args.train.reward_type,
                      action_scheme=args.env.scheme,
                      k_placement=args.env.k_placement)
        if use_multi_size:
            env = MultiSizeWrapper(env, bin_sizes)
        return env

    def make_test_env():
        env = gym.make(args.env.id,
                      container_size=base_container_size,
                      enable_rotation=args.env.rot,
                      data_type=args.env.box_type,
                      item_set=args.env.box_size_set,
                      reward_type=args.train.reward_type,
                      action_scheme=args.env.scheme,
                      k_placement=args.env.k_placement)
        if use_multi_size:
            env = MultiSizeWrapper(env, bin_sizes)
        return env

    train_envs = ts.env.SubprocVectorEnv(
        [make_train_env for _ in range(args.train.num_processes)]
    )
    test_envs = ts.env.SubprocVectorEnv(
        [make_test_env for _ in range(1)]
    )
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)

    return train_envs, test_envs


def build_net(args, device):
    # For multi-size training, use max bin size for network architecture
    use_multi_size = args.env.get('bin_sizes') is not None
    if use_multi_size:
        max_bin_size = [max(b[i] for b in args.env.bin_sizes) for i in range(3)]
        container_size = max_bin_size
    else:
        container_size = args.env.container_size

    feature_net = model.ShareNet(
        k_placement=args.env.k_placement,
        box_max_size=args.env.box_big,
        container_size=container_size,
        embed_size=args.model.embed_dim,
        num_layers=args.model.num_layers,
        forward_expansion=args.model.forward_expansion,
        heads=args.model.heads,
        dropout=args.model.dropout,
        device=device,
        place_gen=args.env.scheme
    )

    actor = model.ActorHead(
        preprocess_net=feature_net, 
        embed_size=args.model.embed_dim, 
        padding_mask=args.model.padding_mask,
        device=device, 
    ).to(device)

    critic = model.CriticHead(
        preprocess_net=feature_net, 
        k_placement=args.env.k_placement,
        embed_size=args.model.embed_dim,
        padding_mask=args.model.padding_mask,
        device=device, 
    ).to(device)

    return actor, critic


def train(args):

    date = time.strftime(r'%Y.%m.%d-%H-%M-%S', time.localtime(time.time()))

    # Allow custom run name via config
    if args.get('run_name') is not None:
        time_str = args.run_name + "_" + date
    else:
        # Default naming scheme
        time_str = args.env.id + "_" + \
            str(args.env.container_size[0]) + "-" + str(args.env.container_size[1]) + "-" + str(args.env.container_size[2]) + "_" + \
            args.env.scheme + "_" + str(args.env.k_placement) + "_" +\
            args.env.box_type + "_" + \
            args.train.algo  + '_' \
            'seed' + str(args.seed) + "_" + \
            args.opt.optimizer + "_" \
            + date

    if args.cuda and torch.cuda.is_available():
        device = torch.device("cuda", args.device)
    else:
        device = torch.device("cpu")

    set_seed(args.seed, args.cuda, args.cuda_deterministic)

    # environments 
    train_envs, test_envs = make_envs(args)  # make envs and set random seed

    # network
    actor, critic = build_net(args, device)
    actor_critic = ActorCritic(actor, critic)

    if args.opt.optimizer == 'Adam':
        optim = torch.optim.Adam(actor_critic.parameters(), lr=args.opt.lr, eps=args.opt.eps)
    elif args.opt.optimizer == 'RMSprop':
        optim = torch.optim.RMSprop(
            actor_critic.parameters(),
            lr=args.opt.lr,
            eps=args.opt.eps,
            alpha=args.opt.alpha,
        )
    else:
        raise NotImplementedError

    lr_scheduler = None
    if args.opt.lr_decay:
        # decay learning rate to 0 linearly
        max_update_num = np.ceil(args.train.step_per_epoch / args.train.step_per_collect) * args.train.epoch
        lr_scheduler = LambdaLR(optim, lr_lambda=lambda epoch: 1 - epoch / max_update_num)


    # RL agent 
    dist = CategoricalMasked
    if args.train.algo == 'PPO':
        policy = MaskedPPOPolicy(
            actor=actor,
            critic=critic,
            optim=optim,
            dist_fn=dist,
            discount_factor=args.train.gamma,
            eps_clip=args.train.clip_param,
            advantage_normalization=False,
            vf_coef=args.loss.value,
            ent_coef=args.loss.entropy,
            gae_lambda=args.train.gae_lambda,
            lr_scheduler=lr_scheduler
        )
    elif args.algo == 'A2C':    
        policy = MaskedA2CPolicy(
            actor,
            critic,
            optim,
            dist,
            discount_factor=args.train.gamma,
            vf_coef=args.loss.value,
            ent_coef=args.loss.entropy,
            gae_lambda=args.train.gae_lambda,
            lr_scheduler=lr_scheduler
        )
    else:
        raise NotImplementedError

    log_path = './logs/' + time_str
    
    is_debug = True if sys.gettrace() else False
    if not is_debug:
        writer = SummaryWriter(log_path)
        logger = TensorboardLogger(
            writer=writer,
            train_interval=args.log_interval,
            update_interval=args.log_interval
        )
        # backup the config file, os.path.join(,)
        shutil.copy(args.config, log_path)  # config file
        shutil.copy("model.py", log_path)  # network
        shutil.copy("arguments.py", log_path)  # network
    else:
        logger = LazyLogger()

    # ======== Curriculum Learning Setup (Objective 3) =========
    use_curriculum = args.train.get('use_curriculum', False)
    curriculum_scheduler = None

    if use_curriculum:
        curriculum_scheduler = CurriculumScheduler(
            initial_difficulty=args.train.get('curriculum_initial', 0.2),
            final_difficulty=args.train.get('curriculum_final', 0.8),
            curriculum_epochs=args.train.get('curriculum_epochs', 400),
            schedule_type=args.train.get('curriculum_schedule', 'linear')
        )
        print(f"\nCurriculum learning enabled!")
        print(f"  Schedule: {args.train.curriculum_schedule}")
        print(f"  Difficulty: {args.train.curriculum_initial:.2f} → {args.train.curriculum_final:.2f}")
        print(f"  Ramp-up epochs: {args.train.curriculum_epochs}\n")

    # ======== callback functions used during training =========
    def train_fn(epoch, env_step):
        # Update curriculum difficulty if enabled
        if use_curriculum and curriculum_scheduler is not None:
            target_difficulty = curriculum_scheduler.get_target_difficulty(epoch)
            set_global_difficulty(target_difficulty)

            # Log curriculum info
            if not is_debug and epoch % args.log_interval == 0:
                writer.add_scalar('curriculum/difficulty', target_difficulty, epoch)
                writer.add_scalar('curriculum/progress',
                                min(1.0, epoch / args.train.curriculum_epochs), epoch)

            # Print curriculum status every 50 epochs
            if epoch % 50 == 0:
                print(f"Epoch {epoch}: Curriculum difficulty = {target_difficulty:.3f}")

        # monitor leraning rate in tensorboard
        # writer.add_scalar('train/lr', optim.param_groups[0]["lr"], env_step)
        pass

    def save_best_fn(policy):
        if not is_debug:
            torch.save(policy.state_dict(), os.path.join(log_path, 'policy_step_best.pth'))
        else:
            pass

    def final_save_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, 'policy_step_final.pth'))

    def save_checkpoint_fn(epoch, env_step, gradient_step):
        if not is_debug:
            # see also: https://pytorch.org/tutorials/beginner/saving_loading_models.html
            ckpt_path = os.path.join(log_path, "checkpoint.pth")
            # Example: saving by epoch num
            # ckpt_path = os.path.join(log_path, f"checkpoint_{epoch}.pth")
            torch.save({"model": policy.state_dict(), "optim": optim.state_dict()}, ckpt_path)
            return ckpt_path
        else:
            return None
    
    def watch(train_info):
        print("Setup test envs ...")
        policy.eval()
        test_envs.seed(args.seed)
        print("Testing agent ...")
        test_collector.reset()
        result = test_collector.collect(n_episode=1000)
        ratio = result["ratio"]
        ratio_std = result["ratio_std"]
        total = result["num"]
        print(f"The result (over {result['n/ep']} episodes): ratio={ratio}, ratio_std={ratio_std}, total={total}")
        with open(os.path.join(log_path, f"{ratio:.4f}_{ratio_std:.4f}_{total}.txt"), "w") as file:
            file.write(str(train_info).replace("{", "").replace("}", "").replace(", ", "\n"))

    buffer = VectorReplayBuffer(total_size=10000, buffer_num=len(train_envs))
    train_collector = PackCollector(policy, train_envs, buffer)
    test_collector = PackCollector(policy, test_envs)
    
    # trainer
    result = onpolicy_trainer(
        policy,
        train_collector,
        test_collector,
        max_epoch=args.train.epoch,
        step_per_epoch=args.train.step_per_epoch,
        repeat_per_collect=args.train.repeat_per_collect,
        episode_per_test=10, # args.test_num,
        batch_size=args.train.batch_size,
        step_per_collect=args.train.step_per_collect,
        # episode_per_collect=args.episode_per_collect,
        train_fn=train_fn,
        save_best_fn=save_best_fn,
        save_checkpoint_fn=save_checkpoint_fn,
        logger=logger,
        test_in_train=False
    )

    final_save_fn(policy)
    pprint.pprint(f'Finished training! \n{result}')
    watch(result)


if __name__ == '__main__':
    registration_envs()
    args = arguments.get_args()
    args.train.algo = args.train.algo.upper()
    args.train.step_per_collect = args.train.num_processes * args.train.num_steps  

    train(args)
