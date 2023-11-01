import random
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import *
import gymnasium as gym
import numpy as np
import torch
from torch import nn, optim
from torch.utils import tensorboard as tb
from tqdm import tqdm


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

    def __init__(self, state_dim, hidden_dim, action_dim, action_bound: np.ndarray):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, action_dim)
        # tanh将输出限制在(-1,+1)之间
        self.tanh = nn.Tanh()
        # action_bound是环境可以接受的动作最大值
        self.action_bound = torch.from_numpy(action_bound).float()

    def forward(self, state_tensor):
        x = self.fc1(state_tensor)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.tanh(x)
        return x * self.action_bound


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
        heter: np.ndarray = np.array([0.5]),
        lr: float = 1e-3,
        sigma: float = 0.10,
        tau: float = 0.005,
        gamma: float = 0.98,
        hidden_dim: int = 400,
        buffer_capicity: int = 10000,
        buffer_init_ratio: float = 0.30,
        batch_size: int = 64,
        device: str = "cpu",
        **kwargs,
    ):
        self.init_params = self.get_hyperparams(locals())
        if isinstance(env, str):
            self.env = gym.make(env, heter=heter)
            self.env_name = env
        else:
            self.env = env
            self.env_name = self.env.spec.id

        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]
        action_bound = self.env.action_space.high
        self.actor = Actor(state_dim, hidden_dim, action_dim, action_bound).to(device)
        self.critic = Critic(state_dim, hidden_dim, action_dim).to(device)
        self.actor_target = Actor(state_dim, hidden_dim, action_dim, action_bound).to(device)
        self.critic_target = Critic(state_dim, hidden_dim, action_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr / 10)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr)
        self.sigma = sigma
        self.tau = tau
        self.gamma = gamma
        self.device = device
        self.replay_buffer = ReplayBuffer(buffer_capicity)
        self.buffer_init_ratio = buffer_init_ratio
        self.batch_size = batch_size

        # 训练时使用
        self.save_dir = Path("result") / f"DDPG-{self.env_name}-{datetime.now().strftime(r'%Y-%m-%d-%H-%M-%S')}"
        self.episode = 0
        self.episode_reward = 0
        self.episode_len = 0
        self.global_step = 0
        self.logger = None
        self.bar = None
        self.evaluation_interval = None

    def get_hyperparams(self, dic: dict):
        """获取init中传入的超参数, 方便后续保存"""
        return {k: v for k, v in dic.items() if k not in ["self", "env", "kwargs"]}

    def get_action(self, s):
        """在训练时得到含噪声的连续动作"""
        if isinstance(s, np.ndarray) is True:
            s = torch.from_numpy(s).float().to(self.device)
        with torch.no_grad():
            a = self.actor(s)
            a += torch.normal(0.0, self.sigma, a.shape)
        return a.cpu().numpy()

    def collect_exp_before_train(self):
        """开启训练之前预先往buffer里面存入一定数量的经验"""
        assert 0 < self.buffer_init_ratio < 1
        num = self.buffer_init_ratio * self.replay_buffer.capicity
        s, _ = self.env.reset()
        while self.replay_buffer.size < num:
            a = self.env.action_space.sample()
            ns, r, t1, t2, _ = self.env.step(a)
            d = t1 or t2
            self.replay_buffer.push(s, a, r, ns, d)
            s = ns if not t1 else self.env.reset()[0]

    def update_target(self):
        """定期同步权重参数到target"""
        net_groups = [(self.actor, self.actor_target), (self.critic, self.critic_target)]
        for net, net_ in net_groups:
            for p, p_ in zip(net.parameters(), net_.parameters()):
                p_.data.copy_(p.data * self.tau + p_.data * (1 - self.tau))

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
        self.update_target()
        return actor_loss.detach(), critic_loss.detach()

    def log_info_per_episode(self):
        self.logger.add_scalar("Train/episode_reward", self.episode_reward, self.episode)
        self.logger.add_scalar("Train/buffer_size", self.replay_buffer.size, self.episode)
        self.logger.add_scalar("Episode/episode_len", self.episode_len, self.episode)
        self.bar.set_description_str(f"episode->{self.episode+1:0>5d}|")
        self.bar.set_postfix_str(f"reward->{self.episode_reward:8.1f}")
        if self.episode == 0:
            self.max_episode_reward = self.episode_reward
        if self.episode_reward >= self.max_episode_reward:
            self.max_episode_reward = self.episode_reward
            self.save(self.save_dir / "best.pt")
        self.episode += 1
        self.episode_len = 0
        self.episode_reward = 0

    def log_info_per_batch(self, actor_loss, critic_loss):
        self.logger.add_scalar("Loss/actor_loss", actor_loss, self.global_step)
        self.logger.add_scalar("Loss/critic_loss", critic_loss, self.global_step)
        if self.global_step % self.evaluation_interval == 0:
            evaluation_reward = self.test(is_render=False, print_log=False)
            self.logger.add_scalar("Evaluation/episode_reward", evaluation_reward, self.global_step)
            self.save(self.save_dir / f"episode-{self.episode}-reward-{self.episode_reward:.2f}.pt")
        self.global_step += 1
        self.episode_len += 1
        self.bar.update()

    def train(self, total_time_steps: int, evaluation_interval: int):
        # 训练前准备
        self.save_dir.mkdir(parents=True)
        self.logger = tb.SummaryWriter(self.save_dir / "train")
        self.evaluation_interval = evaluation_interval
        self.collect_exp_before_train()
        self.bar = tqdm(range(total_time_steps), ncols=80)
        # 开始训练
        s, _ = self.env.reset()
        while self.global_step < total_time_steps:
            a = self.get_action(s)
            ns, r, t1, t2, _ = self.env.step(a)
            self.episode_reward += r
            self.replay_buffer.push(s, a, r, ns, t1)
            if t1 or t2:
                self.log_info_per_episode()
                s, _ = self.env.reset()
            else:
                s = ns
            actor_loss, critic_loss = self.train_one_batch()
            self.log_info_per_batch(actor_loss, critic_loss)

    def save(self, save_path: str, save_hyper: bool = True):
        """保存模型权重/超参数"""
        weights = {"actor": self.actor.state_dict(), "critic": self.critic.state_dict()}
        if not save_hyper:
            torch.save(weights, save_path)
        else:
            hypers = self.init_params
            hypers.update({"env": self.env_name})
            weights.update(hypers=hypers)
            torch.save(weights, save_path)

    @staticmethod
    def load(path):
        """加载DDPG模型"""
        weights = torch.load(path)
        hypers = weights.get("hypers")
        if hypers.get("env").startswith("envs:"):
            model = DDPG(**hypers)
        else:
            model = DDPG(gym.make(hypers.pop("env")), **hypers)
        model.actor.load_state_dict(weights.get("actor"))
        model.critic.load_state_dict(weights.get("critic"))
        return model

    @torch.no_grad()
    def predict(self, state_tensor):
        """输入状态输出动作"""
        state_tensor = state_tensor.to(self.device)
        action = self.actor(state_tensor).cpu().numpy()
        return action

    def test(
        self,
        device: str = None,
        is_render: bool = True,
        truncated_exit=True,
        print_log: bool = True,
    ):
        """模型测试, 支持多设备"""
        self.device = device if device is not None else self.device
        self.actor.to(self.device)
        self.critic.to(self.device)
        if is_render and self.env.render_mode is None:
            env = gym.make(self.env_name, render_mode="human")
        else:
            env = gym.make(self.env_name)

        total_reward = 0.0
        s, _ = env.reset()
        iter_num = 0
        while True:
            a = self.predict(torch.from_numpy(s).float().to(self.device))
            s, r, t1, t2, _ = env.step(a)
            total_reward += r
            iter_num += 1
            if is_render:
                env.render()
            if print_log:
                print(iter_num, r, s)
            if (truncated_exit and t2) or t1:
                break
        return total_reward
