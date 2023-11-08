import copy
import random
from collections import deque
from pathlib import Path
from typing import List, Union

import gymnasium as gym
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn, optim
from torch.utils import tensorboard as tb
from tqdm import tqdm, trange

plt.switch_backend("agg")


"""
TODO: 添加动作噪声Gauss/epsilon
TODO: 评估方式
"""


class ReplayBuffer:
    def __init__(self, capicity: int) -> None:
        self.capicity = capicity
        self.buffer = deque(maxlen=self.capicity)

    @property
    def size(self):
        return len(self.buffer)

    def push(self, s, a, r, next_s, t):
        if self.size == self.capicity:
            self.buffer.popleft()
        self.buffer.append([s, a, r, next_s, t])

    def is_full(self):
        return self.size == self.capicity

    def sample(self, N: int, device: str):
        """采样数据并打包"""
        assert N <= self.size, "batch is too big"
        samples = random.sample(self.buffer, N)
        states, actions, rewards, next_states, terminated = zip(*samples)
        return (
            torch.from_numpy(np.vstack(states)).float().to(device),
            torch.from_numpy(np.vstack(actions)).float().to(device),
            torch.from_numpy(np.vstack(rewards)).float().to(device),
            torch.from_numpy(np.vstack(next_states)).float().to(device),
            torch.from_numpy(np.vstack(terminated)).float().to(device),
        )


class Actor(nn.Module):
    """actor网络"""

    def __init__(self, state_dim, hidden_dim, action_bound: np.ndarray):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()
        # tanh将输出限制在(-1,+1)之间
        self.fc_ess = nn.Linear(hidden_dim, 1)
        self.tanh = nn.Tanh()
        self.fc_hvac = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
        # action_bound是环境可以接受的动作最大值
        self.action_bound = torch.from_numpy(action_bound).float()

    def forward(self, state_tensor):
        x = self.fc1(state_tensor)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        output_ess = self.fc_ess(x)
        output_ess = self.action_bound[0] * self.tanh(output_ess)
        output_hvac = self.fc_hvac(x)
        output_hvac = self.action_bound[1] * self.sigmoid(output_hvac)
        y = torch.cat([output_ess, output_hvac], dim=-1)
        return y


class Critic(nn.Module):
    """Q网络: (s,a)-->q"""

    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, state_tensor, action_tensor):
        """网络输入是状态和动作, 因此需要cat在一起"""
        x = torch.cat([state_tensor, action_tensor], dim=-1)  # 拼接状态和动作
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x


class DDPG:
    def __init__(
        self,
        env: Union[gym.Env, str],
        env_index: int,
        heter: np.ndarray = np.array([0.5, 0.5, 0.5]),
        lr: float = 1e-3,
        tau: float = 0.005,
        gamma: float = 0.98,
        hidden_dim: list = [400, 400],
        buffer_capicity: int = 10000,
        buffer_init_ratio: float = 0.30,
        batch_size: int = 64,
        train_batchs: int = None,
        device: str = "cpu",
        save_dir: str = None,
        **kwargs,
    ):
        self.env = gym.make(env, heter=heter)
        self.env_name = self.env.spec.id
        self.env_index = env_index
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]
        action_bound = self.env.action_space.high
        self.actor = Actor(state_dim, hidden_dim, action_bound).to(device)
        self.critic = Critic(state_dim, hidden_dim, action_dim).to(device)
        self.actor_target = Actor(state_dim, hidden_dim, action_bound).to(device)
        self.critic_target = Critic(state_dim, hidden_dim, action_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr / 10)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr)
        self.tau = tau
        self.gamma = gamma
        self.device = device
        self.replay_buffer = ReplayBuffer(buffer_capicity)
        self.buffer_init_ratio = buffer_init_ratio
        self.batch_size = batch_size
        # 训练时使用
        self.episode = 0
        self.episode_reward = 0
        self.episode_reward_list = []
        self.episode_len = 0
        self.global_step = 0
        self.total_train_batchs = train_batchs
        assert save_dir is not None
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True)
        self.logger = tb.SummaryWriter(self.save_dir / "log")
        self.collect_exp_before_train()
        # 开始训练
        self.state, _ = self.env.reset()

    @torch.no_grad()
    def get_action(self, s: Union[np.array, torch.Tensor], eps: float = 0.0):
        """在训练时得到含噪声的连续动作"""
        if isinstance(s, np.ndarray) is True:
            s = torch.from_numpy(s).float().to(self.device)
        if np.random.uniform() < eps:
            a = self.env.action_space.sample()
        else:
            s_ = self.env.unwrapped.normalize_state(s)
            a = self.actor(s_)
            a = a.cpu().numpy()
        s = s.cpu().numpy()
        # ---------------------------------------------------------------------------
        p_ess, p_hvac = a
        p_solar, p_load, ess_level, temp_outdoor, temp_indoor, price, _ = s
        if p_ess >= 0:
            p_ess = np.clip(
                p_ess,
                0,
                min(
                    (self.env.unwrapped.ess_level_max - ess_level) / self.env.unwrapped.eta_ess,
                    self.env.unwrapped.p_ess_max,
                ),
            )
        else:
            p_ess = -np.clip(
                -p_ess,
                0,
                min(
                    (ess_level - self.env.unwrapped.ess_level_min) * self.env.unwrapped.eta_ess,
                    self.env.unwrapped.p_ess_max,
                ),
            )
        if temp_indoor <= self.env.unwrapped.T_min:
            p_hvac = 0
        if temp_indoor > self.env.unwrapped.T_max:
            p_hvac = np.clip(p_hvac, 0.1, self.env.unwrapped.p_hvac_max)

        return np.array([p_ess, p_hvac])

    def collect_exp_before_train(self):
        """开启训练之前预先往buffer里面存入一定数量的经验"""
        assert 0 < self.buffer_init_ratio < 1
        num = self.buffer_init_ratio * self.replay_buffer.capicity
        bar = tqdm(range(int(num)), leave=False, ncols=80)
        bar.set_description_str(f"env:{self.env_index}->")
        s, _ = self.env.reset()
        while self.replay_buffer.size < num:
            a = self.get_action(s, 1.0)
            ns, r, t1, t2, _ = self.env.step(a)
            self.replay_buffer.push(
                self.env.unwrapped.normalize_state(s),
                a,
                r,
                self.env.unwrapped.normalize_state(ns),
                t1,
            )
            s = ns if not t2 else self.env.reset()[0]
            bar.update()

    def soft_sync_target(self):
        """软更新参数到target"""
        net_groups = [(self.actor, self.actor_target), (self.critic, self.critic_target)]
        for net, net_ in net_groups:
            for p, p_ in zip(net.parameters(), net_.parameters()):
                p_.data.copy_(p.data * self.tau + p_.data * (1 - self.tau))

    @property
    def epsilon(self):
        """得到递减的epsilon"""
        prog = self.global_step / self.total_train_batchs
        assert 0 <= prog <= 1.0
        return max(np.exp(-4 * prog), 0.10)

    def train_one_batch(self):
        # 从buffer中取出一批数据
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size, self.device)
        # 计算critic_loss并更新
        with torch.no_grad():
            td_targets = rewards + self.gamma * (1 - dones) * self.critic_target(
                next_states, self.actor_target(next_states)
            )
        td_errors = td_targets - self.critic(states, actions)
        critic_loss = torch.pow(td_errors, 2).mean()
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        # 计算actor_loss并更新
        actor_loss = -self.critic(states, self.actor(states)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        # 软更新target
        self.soft_sync_target()
        return actor_loss.detach(), critic_loss.detach()

    def train(self, train_batchs: int, disable_prog_bar: bool = True):
        # 开始训练
        for _ in trange(train_batchs, disable=disable_prog_bar, ncols=80, leave=False):
            a = self.get_action(self.state, self.epsilon)
            ns, r, t1, t2, _ = self.env.step(a)
            self.episode_reward += r
            self.replay_buffer.push(
                self.env.unwrapped.normalize_state(self.state),
                a,
                r,
                self.env.unwrapped.normalize_state(ns),
                t1,
            )
            if t1 or t2:
                self.log_info_per_episode()
                self.state, _ = self.env.reset()
            else:
                self.state = ns
            actor_loss, critic_loss = self.train_one_batch()
            self.log_info_per_batch(actor_loss, critic_loss)

    def log_info_per_episode(self):
        self.logger.add_scalar("Train/episode_reward", self.episode_reward, self.episode)
        self.logger.add_scalar("Train/buffer_size", self.replay_buffer.size, self.episode)
        self.logger.add_scalar("Episode/episode_len", self.episode_len, self.episode)
        self.episode_reward_list.append(self.episode_reward)
        self.episode += 1
        self.episode_len = 0
        self.episode_reward = 0

    def log_info_per_batch(self, actor_loss, critic_loss):
        self.logger.add_scalar("Loss/actor_loss", actor_loss, self.global_step)
        self.logger.add_scalar("Loss/critic_loss", critic_loss, self.global_step)
        self.logger.add_scalar("Train/epsilon", self.epsilon, self.global_step)
        self.global_step += 1
        self.episode_len += 1

    def save(self, save_path: str):
        params = {"actor": self.actor.state_dict(), "critic": self.critic.state_dict()}
        torch.save(params, save_path)


class Server:
    """server角色"""

    def __init__(self, points: List[DDPG], device: str = "cpu") -> None:
        """为保护用户隐私, 除了神经网络参数之外, 不能从节点读取任何数据"""
        self.points = points
        self.device = device
        self.actor = copy.deepcopy(self.points[0].actor).to(self.device)
        self.actor_target = copy.deepcopy(self.points[0].actor_target).to(self.device)
        self.critic = copy.deepcopy(self.points[0].critic).to(self.device)
        self.critic_target = copy.deepcopy(self.points[0].critic_target).to(self.device)

    def merge_params(self, merge_target: bool = False) -> None:
        """合并/分发参数"""
        for name, param in self.actor.state_dict().items():
            avg_param = torch.stack([p.actor.state_dict()[name] for p in self.points]).mean(dim=0)
            param.data.copy_(avg_param.data)
        for name, param in self.critic.state_dict().items():
            avg_param = torch.stack([p.critic.state_dict()[name] for p in self.points]).mean(dim=0)
            param.data.copy_(avg_param.data)
        if merge_target is True:
            for name, param in self.actor_target.state_dict().items():
                avg_param = torch.stack([p.actor_target.state_dict()[name] for p in self.points]).mean(dim=0)
                param.data.copy_(avg_param.data)
            for name, param in self.critic_target.state_dict().items():
                avg_param = torch.stack([p.critic_target.state_dict()[name] for p in self.points]).mean(dim=0)
                param.data.copy_(avg_param.data)
        for p in self.points:
            p.actor.load_state_dict(self.actor.state_dict())
            p.critic.load_state_dict(self.critic.state_dict())
            if merge_target is True:
                p.actor_target.load_state_dict(self.actor_target.state_dict())
                p.critic_target.load_state_dict(self.critic_target.state_dict())


class FedDDPG:
    def __init__(
        self,
        point_configs: List[dict],
        merge_num: int,
        merge_interval: int,
        merge_target: bool,
        episode_num_eval: int,
        save_dir: str = None,
        device: str = "cpu",
    ) -> None:
        assert save_dir is not None, "save_dir can't be empty"
        self.device = device
        self.point_configs = point_configs
        self.merge_num = merge_num
        self.merge_interval = merge_interval
        self.merge_target = merge_target
        self.episode_num_eval = episode_num_eval
        self.save_dir = save_dir

        self.points = [DDPG(**c) for c in point_configs]
        self.server = Server(self.points, device=self.device)
        self.logger = tb.SummaryWriter(self.save_dir / "global" / "log")

    def train(self):
        """总共合并训练self.merge_num次"""
        with trange(self.merge_num, ncols=80) as prog_bar:
            for m in range(self.merge_num):
                for p in tqdm(self.points, leave=False, disable=True):
                    p.train(self.merge_interval)
                self.server.merge_params(self.merge_target)
                self.save(self.save_dir / "server" / f"aggre_{self.merge_idx}.pt")
                avg_merge_episode_reward = self.evaluate_avg_reward()
                self.logger.add_scalar("aggregate/reward", avg_merge_episode_reward, global_step=m)
                prog_bar.set_description_str(f"reward->{int(avg_merge_episode_reward):3d}|")
                prog_bar.update()
        self.summarize_point_reward()
        for p in self.points:
            p.logger.close()
        self.logger.close()

    def train_baseline(self):
        """训练baseline用于对照"""
        with trange(self.merge_num, ncols=80) as prog_bar:
            for m in range(self.merge_num):
                for p in self.points:
                    p.train(self.merge_interval, disable_prog_bar=True)
                avg_episode_reward = self.evaluate_avg_reward()
                self.logger.add_scalar("Average/reward", avg_episode_reward, global_step=m)
                prog_bar.set_description_str(f"reward->{int(avg_episode_reward):3d}")
                prog_bar.update()
        self.summarize_point_reward()
        for p in self.points:
            p.save(p.save_dir / "latest.pt")
            p.logger.close()
        self.logger.close()

    def evaluate_point_reward(self, point: DDPG):
        """传入一个节点, 评估奖励(不改变环境状态)"""
        env = copy.deepcopy(point.env)
        point_r = 0
        for _ in range(self.episode_num_eval):
            s, _ = env.reset()
            while True:
                a = point.get_action(s)
                ns, r, t1, t2, _ = env.step(a)
                point_r += r
                s = ns
                if t1 or t2:
                    break
        return point_r / self.episode_num_eval

    def evaluate_avg_reward(self):
        """评估每个节点的奖励并取平均"""
        reward_list = []
        for p in self.points:
            point_r = self.evaluate_point_reward(p)
            reward_list.append(point_r)
        return sum(reward_list) / len(reward_list)

    def summarize_point_reward(self):
        """统计每个point在训练过程中已经完成的episode的奖励, 并按最短的长度取平均"""
        min_length = min([len(p.episode_reward_list) for p in self.points])
        table = []
        for p in self.points:
            table.append(p.episode_reward_list[:min_length])
            np.save(p.save_dir / "episode_reward_list.npy", np.array(p.episode_reward_list))
        avg_episode_reward = np.array(table).mean(0)
        plt.plot(range(min_length), avg_episode_reward), plt.grid(), plt.title("average episode reward")
        plt.savefig(self.save_dir / "global" / "average_episode_reward.svg")
        plt.close()

    def save(self, save_path):
        """保存权重"""
        Path(save_path).parent.mkdir(exist_ok=True)
        params = {
            "weights": [
                {"actor": self.server.actor.state_dict()},
                {"critic": self.server.critic.state_dict()},
            ]
        }
        torch.save(params, save_path)
