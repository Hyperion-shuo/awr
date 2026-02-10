import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions as distributions
import numpy as np
import gym
import enum
import abc
from typing import List, Dict, Tuple, Union, Optional, Any

# 尝试导入 jaxtyping，如果不存在则定义占位符
try:
    from jaxtyping import Float, Int, Array
    from torch import Tensor
except ImportError:
    # Fallback for typing if jaxtyping is not installed
    class Placeholder:
        def __getitem__(self, item):
            return Any
    Float = Placeholder()
    Int = Placeholder()
    Array = np.ndarray
    Tensor = torch.Tensor

# ==========================================
# Util Classes (RLPath, Terminate)
# ==========================================

class Terminate(enum.Enum):
    Null = 0
    Fail = 1

class RLPath(object):
    def __init__(self):
        self.states: List[np.ndarray] = []
        self.actions: List[np.ndarray] = []
        self.logps: List[float] = []
        self.rewards: List[float] = []
        self.terminate: Terminate = Terminate.Null
        self.clear()

    def pathlength(self) -> int:
        return len(self.actions)

    def is_valid(self) -> bool:
        valid = True
        l = self.pathlength()
        valid &= len(self.states) == l + 1
        valid &= len(self.actions) == l
        valid &= len(self.logps) == l
        valid &= len(self.rewards) == l
        valid |= (l == 0)
        return valid

    def check_vals(self) -> bool:
        for key, vals in vars(self).items():
            if type(vals) is list and len(vals) > 0:
                for v in vals:
                    if isinstance(v, (np.ndarray, float, int)):
                         if not np.isfinite(v).all():
                            return False
        return True

    def clear(self):
        for key, vals in vars(self).items():
            if type(vals) is list:
                vals.clear()
        self.terminate = Terminate.Null

    def calc_return(self) -> float:
        return sum(self.rewards)

# ==========================================
# Replay Buffer (Full Implementation)
# ==========================================

INVALID_IDX = -1

class ReplayBuffer(object):
    """
    完整还原源代码的 Replay Buffer 实现。
    支持循环缓冲区、路径存储、采样等功能。
    """
    TERMINATE_KEY = "terminate"
    PATH_START_KEY = "path_start"
    PATH_END_KEY = "path_end"

    def __init__(self, buffer_size: int):
        assert buffer_size > 0
        self.buffer_size = buffer_size
        self.total_count = 0
        self.buffer_head = 0
        self.buffer_tail = INVALID_IDX
        self.num_paths = 0
        self.buffers: Optional[Dict[str, np.ndarray]] = None
        self.clear()

    def sample(self, n: int, filter_end: bool = True) -> Int[np.ndarray, "n"]:
        """
        从缓冲区中采样 n 个索引。
        """
        curr_size = self.get_current_size()
        assert curr_size > 0

        if filter_end:
            idx = np.empty(n, dtype=int)
            # makes sure that the end states are not sampled
            for i in range(n):
                while True:
                    curr_idx = np.random.randint(0, curr_size, size=1)[0]
                    curr_idx += self.buffer_tail
                    curr_idx = np.mod(curr_idx, self.buffer_size)

                    if not self.is_path_end(curr_idx):
                        break
                idx[i] = curr_idx
        else:
            idx = np.random.randint(0, curr_size, size=n)
            idx += self.buffer_tail
            idx = np.mod(idx, self.buffer_size)

        return idx

    def get(self, key: str, idx: Union[int, np.ndarray]) -> np.ndarray:
        return self.buffers[key][idx]

    def get_all(self, key: str) -> np.ndarray:
        return self.buffers[key]

    def get_unrolled_indices(self) -> List[int]:
        indices = None
        if self.buffer_tail == INVALID_IDX:
            indices = []
        elif self.buffer_tail < self.buffer_head:
            indices = list(range(self.buffer_tail, self.buffer_head))
        else:
            indices = list(range(self.buffer_tail, self.buffer_size))
            indices += list(range(0, self.buffer_head))
        return indices
    
    def get_path_start(self, idx: Union[int, np.ndarray]) -> Union[int, np.ndarray]:
        return self.buffers[self.PATH_START_KEY][idx]

    def get_path_end(self, idx: Union[int, np.ndarray]) -> Union[int, np.ndarray]:
        return self.buffers[self.PATH_END_KEY][idx]

    def get_pathlen(self, idx: Union[int, np.ndarray, List[int]]) -> Union[int, np.ndarray]:
        is_array = isinstance(idx, np.ndarray) or isinstance(idx, list)
        if not is_array:
            idx = [idx]

        n = len(idx)
        start_idx = self.get_path_start(idx)
        end_idx = self.get_path_end(idx)
        pathlen = np.empty(n, dtype=int)

        for i in range(n):
            curr_start = start_idx[i]
            curr_end = end_idx[i]
            if curr_start < curr_end:
                curr_len = curr_end - curr_start
            else:
                curr_len = self.buffer_size - curr_start + curr_end
            pathlen[i] = curr_len

        if not is_array:
            pathlen = pathlen[0]

        return pathlen

    def is_valid_path(self, idx: int) -> bool:
        start_idx = self.get_path_start(idx)
        valid = start_idx != INVALID_IDX
        return valid

    def store(self, path: RLPath) -> int:
        start_idx = INVALID_IDX
        n = path.pathlength()
        
        if (n > 0):
            assert path.is_valid()

            if path.check_vals():
                if self.buffers is None:
                    self._init_buffers(path)

                idx = self._request_idx(n + 1)
                self._store_path(path, idx)

                self.num_paths += 1
                self.total_count += n + 1
                start_idx = idx[0]
            else:
                print("Invalid path data value detected")

        return start_idx

    def clear(self):
        self.buffer_head = 0
        self.buffer_tail = INVALID_IDX
        self.num_paths = 0
        return
    
    def check_terminal_flag(self, idx: Union[int, np.ndarray], flag: Terminate) -> Union[bool, np.ndarray]:
        terminate_flags = self.buffers[self.TERMINATE_KEY][idx]
        terminate = (terminate_flags == flag.value)
        return terminate

    def is_path_start(self, idx: Union[int, np.ndarray]) -> Union[bool, np.ndarray]:
        is_end = self.buffers[self.PATH_START_KEY][idx] == idx
        return is_end

    def is_path_end(self, idx: Union[int, np.ndarray]) -> Union[bool, np.ndarray]:
        is_end = self.buffers[self.PATH_END_KEY][idx] == idx
        return is_end

    def get_current_size(self) -> int:
        if self.buffer_tail == INVALID_IDX:
            return 0
        elif self.buffer_tail < self.buffer_head:
            return self.buffer_head - self.buffer_tail
        else:
            return self.buffer_size - self.buffer_tail + self.buffer_head

    def _init_buffers(self, path: RLPath):
        self.buffers = dict()
        self.buffers[self.PATH_START_KEY] = INVALID_IDX * np.ones(self.buffer_size, dtype=int)
        self.buffers[self.PATH_END_KEY] = INVALID_IDX * np.ones(self.buffer_size, dtype=int)
        self.buffers[self.TERMINATE_KEY] = np.zeros(shape=[self.buffer_size], dtype=int)

        for key, val in vars(path).items():
            if type(val) is list:
                if len(val) == 0: continue
                val_type = type(val[0])
                is_array = val_type == np.ndarray
                if is_array:
                    shape = [self.buffer_size, val[0].shape[0]]
                    dtype = val[0].dtype
                else:
                    shape = [self.buffer_size]
                    dtype = val_type
                    
                self.buffers[key] = np.zeros(shape, dtype=dtype)

    def _request_idx(self, n: int) -> List[int]:
        assert n + 1 < self.buffer_size # bad things can happen if path is too long

        remainder = n
        idx = []

        start_idx = self.buffer_head
        while remainder > 0:
            end_idx = np.minimum(start_idx + remainder, self.buffer_size)
            remainder -= (end_idx - start_idx)

            free_idx = list(range(start_idx, end_idx))
            self._free_idx(free_idx)
            idx += free_idx
            start_idx = 0

        self.buffer_head = (self.buffer_head + n) % self.buffer_size
        return idx

    def _free_idx(self, idx: List[int]):
        assert(idx[0] <= idx[-1])
        n = len(idx)
        if self.buffer_tail != INVALID_IDX:
            update_tail = idx[0] <= idx[-1] and idx[0] <= self.buffer_tail and idx[-1] >= self.buffer_tail
            update_tail |= idx[0] > idx[-1] and (idx[0] <= self.buffer_tail or idx[-1] >= self.buffer_tail)
            
            if update_tail:
                i = 0
                while i < n:
                    curr_idx = idx[i]
                    if self.is_valid_path(curr_idx):
                        start_idx = self.get_path_start(curr_idx)
                        end_idx = self.get_path_end(curr_idx)
                        pathlen = self.get_pathlen(curr_idx)

                        if start_idx < end_idx:
                            self.buffers[self.PATH_START_KEY][start_idx:end_idx + 1] = INVALID_IDX
                        else:
                            self.buffers[self.PATH_START_KEY][start_idx:self.buffer_size] = INVALID_IDX
                            self.buffers[self.PATH_START_KEY][0:end_idx + 1] = INVALID_IDX
                        
                        self.num_paths -= 1
                        i += pathlen + 1
                        self.buffer_tail = (end_idx + 1) % self.buffer_size
                    else:
                        i += 1
        else:
            self.buffer_tail = idx[0]

    def _store_path(self, path: RLPath, idx: List[int]):
        n = path.pathlength()
        idx_arr = np.array(idx)
        for key, data in self.buffers.items():
            if key != self.PATH_START_KEY and key != self.PATH_END_KEY and key != self.TERMINATE_KEY:
                val = getattr(path, key)
                val_len = len(val)
                assert val_len == n or val_len == n + 1
                data[idx_arr[:val_len]] = val

        self.buffers[self.TERMINATE_KEY][idx_arr] = path.terminate.value
        self.buffers[self.PATH_START_KEY][idx_arr] = idx[0]
        self.buffers[self.PATH_END_KEY][idx_arr] = idx[-1]

# ==========================================
# Normalizer
# ==========================================

class Normalizer(nn.Module):
    """
    用于标准化观测值和动作的工具类。
    还原了 util.normalizer 的核心逻辑，适配 PyTorch。
    """
    def __init__(self, size: int, init_mean: np.ndarray = None, init_std: np.ndarray = None, epsilon: float = 1e-4):
        super().__init__()
        self.size = size
        self.epsilon = epsilon
        
        if init_mean is None: init_mean = np.zeros(size)
        if init_std is None: init_std = np.ones(size)
        
        # 使用 register_buffer 保存状态，但不作为模型参数更新
        self.register_buffer("mean", torch.FloatTensor(init_mean))
        self.register_buffer("std", torch.FloatTensor(init_std))
        self.register_buffer("count", torch.FloatTensor([epsilon]))
        
        # 用于计算增量更新的统计量
        self.local_sum = np.zeros(size)
        self.local_sumsq = np.zeros(size)
        self.local_count = 0

    def update(self):
        if self.local_count > 0:
            batch_mean = self.local_sum / self.local_count
            batch_var = (self.local_sumsq / self.local_count) - np.square(batch_mean)
            batch_var = np.maximum(batch_var, 0) # 防止数值误差导致负方差
            
            self._update_from_moments(batch_mean, batch_var, self.local_count)
            
            self.local_sum[:] = 0
            self.local_sumsq[:] = 0
            self.local_count = 0

    def _update_from_moments(self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: int):
        delta = batch_mean - self.mean.cpu().numpy()
        tot_count = self.count.item() + batch_count

        new_mean = self.mean.cpu().numpy() + delta * batch_count / tot_count
        m_a = np.square(self.std.cpu().numpy()) * self.count.item()
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count.item() * batch_count / tot_count
        new_var = M2 / tot_count
        new_std = np.sqrt(new_var)

        self.mean.copy_(torch.FloatTensor(new_mean))
        self.std.copy_(torch.FloatTensor(new_std))
        self.count.fill_(tot_count)

    def record(self, x: np.ndarray):
        if x.ndim == 1:
            x = x.reshape(1, -1)
        self.local_sum += np.sum(x, axis=0)
        self.local_sumsq += np.sum(np.square(x), axis=0)
        self.local_count += x.shape[0]

    def normalize(self, x: Union[np.ndarray, Tensor]) -> Union[np.ndarray, Tensor]:
        if isinstance(x, torch.Tensor):
            return (x - self.mean.to(x.device)) / (self.std.to(x.device) + 1e-8)
        return (x - self.mean.cpu().numpy()) / (self.std.cpu().numpy() + 1e-8)

    def unnormalize(self, x: Union[np.ndarray, Tensor]) -> Union[np.ndarray, Tensor]:
        if isinstance(x, torch.Tensor):
            return x * (self.std.to(x.device) + 1e-8) + self.mean.to(x.device)
        return x * (self.std.cpu().numpy() + 1e-8) + self.mean.cpu().numpy()

# ==========================================
# Network Components
# ==========================================

class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int, activation=nn.ReLU):
        super(MLP, self).__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(activation())
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)

# ==========================================
# AWR Agent
# ==========================================

class AWRAgent(nn.Module):
    """
    AWR (Advantage Weighted Regression) Agent 的 PyTorch 实现。
    完整还原了源代码的逻辑，包括动作空间标准化、TD-Lambda 计算、Loss 计算等。
    """
    ADV_EPS = 1e-5

    def __init__(self, 
                 env: gym.Env,
                 actor_hidden_sizes: List[int] = [128, 64],
                 critic_hidden_sizes: List[int] = [128, 64],
                 actor_lr: float = 5e-5,
                 critic_lr: float = 1e-2,
                 gamma: float = 0.99,
                 td_lambda: float = 0.95,
                 samples_per_iter: int = 2048,
                 batch_size: int = 256,
                 actor_steps: int = 1000,
                 critic_steps: int = 500,
                 action_std: float = 0.2,
                 temp: float = 1.0,
                 weight_clip: float = 20.0,
                 actor_init_output_scale: float = 0.01,
                 device: str = 'cpu'):
        super(AWRAgent, self).__init__()
        
        self.env = env
        self.state_dim = np.prod(env.observation_space.shape)
        
        # 动作空间处理
        self.is_discrete = isinstance(env.action_space, gym.spaces.Discrete)
        if self.is_discrete:
            self.action_dim = env.action_space.n
        else:
            self.action_dim = np.prod(env.action_space.shape)
            self.action_bound_min = env.action_space.low
            self.action_bound_max = env.action_space.high

        self.gamma = gamma
        self.td_lambda = td_lambda
        self.batch_size = batch_size
        self.actor_steps = actor_steps
        self.critic_steps = critic_steps
        self.action_std = action_std
        self.temp = temp
        self.weight_clip = weight_clip
        self.samples_per_iter = samples_per_iter
        self.device = device
        self.actor_init_output_scale = actor_init_output_scale

        # 初始化 Normalizers
        self._build_normalizers()
        
        # 构建网络
        self._build_nets(actor_hidden_sizes, critic_hidden_sizes)

        # 优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

    def _build_normalizers(self):
        # State Normalizer
        high = self.env.observation_space.high.copy()
        low = self.env.observation_space.low.copy()
        # 处理无穷大边界
        inf_mask = np.logical_or((high >= np.finfo(np.float32).max), (low <= np.finfo(np.float32).min))
        high[inf_mask] = 1.0
        low[inf_mask] = -1.0
        mean = 0.5 * (high + low)
        std = 0.5 * (high - low)
        self.s_norm = Normalizer(self.state_dim, init_mean=mean, init_std=std).to(self.device)

        # Action Normalizer (仅用于连续动作)
        if not self.is_discrete:
            high = self.action_bound_max.copy()
            low = self.action_bound_min.copy()
            inf_mask = np.logical_or((high >= np.finfo(np.float32).max), (low <= np.finfo(np.float32).min))
            high[inf_mask] = 1.0
            low[inf_mask] = -1.0
            mean = 0.5 * (high + low)
            std = 0.5 * (high - low)
            self.a_norm = Normalizer(self.action_dim, init_mean=mean, init_std=std).to(self.device)
        else:
            self.a_norm = None

        # Value Normalizer
        val_mean = 0.0
        val_std = 1.0 / (1.0 - self.gamma)
        self.val_norm = Normalizer(1, init_mean=np.array([val_mean]), init_std=np.array([val_std])).to(self.device)

    def _build_nets(self, actor_hidden_sizes, critic_hidden_sizes):
        # Actor: 输出 mean (连续) 或 logits (离散)
        # 注意：Actor 输出的是标准化后的动作分布参数
        self.actor = MLP(self.state_dim, actor_hidden_sizes, self.action_dim).to(self.device)
        
        # 初始化 Actor 最后一层权重，使其输出较小
        nn.init.uniform_(self.actor.net[-1].weight, -self.actor_init_output_scale, self.actor_init_output_scale)
        nn.init.zeros_(self.actor.net[-1].bias)

        # Critic: 输出标准化后的 Value
        self.critic = MLP(self.state_dim, critic_hidden_sizes, 1).to(self.device)

        # 连续动作的 LogStd (固定参数)
        if not self.is_discrete:
            logstd_bias_init = np.log(self.action_std) * np.ones(self.action_dim)
            self.log_std = nn.Parameter(torch.FloatTensor(logstd_bias_init).to(self.device), requires_grad=False)

    def get_action(self, state: np.ndarray, test: bool = False) -> Tuple[np.ndarray, float]:
        """
        根据当前状态采样动作。
        输入: state [state_dim]
        输出: action [action_dim], log_prob [1]
        """
        # 1. 状态标准化
        state_tensor = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        norm_state_tensor = self.s_norm.normalize(state_tensor)

        with torch.no_grad():
            if self.is_discrete:
                logits = self.actor(norm_state_tensor)
                dist = distributions.Categorical(logits=logits)
                if test:
                    action = torch.argmax(logits, dim=1)
                else:
                    action = dist.sample()
                log_prob = dist.log_prob(action)
                
                # 离散动作不需要反归一化
                final_action = action.cpu().numpy()[0]
                
            else:
                mean = self.actor(norm_state_tensor)
                std = torch.exp(self.log_std)
                dist = distributions.Normal(mean, std)
                
                if test:
                    sample_norm_action = mean
                else:
                    sample_norm_action = dist.sample()
                
                log_prob = dist.log_prob(sample_norm_action).sum(dim=-1)
                
                # 动作反归一化
                final_action = self.a_norm.unnormalize(sample_norm_action).cpu().numpy()[0]

        return final_action, log_prob.cpu().item()

    def update(self, replay_buffer: ReplayBuffer, new_sample_count: int):
        """
        核心更新逻辑
        """
        # 获取所有数据索引
        idx = np.array(replay_buffer.get_unrolled_indices())
        
        # 过滤掉路径结束点 (因为没有 next state 用于某些计算，虽然 AWR 主要用 MC/TD(lambda))
        # 原代码逻辑：
        end_mask = replay_buffer.is_path_end(idx)
        valid_mask = np.logical_not(end_mask)
        valid_idx = idx[valid_mask]
        # 这里的 valid_idx 处理有点奇怪，原代码是 valid_idx = np.column_stack([valid_idx, np.nonzero(valid_mask)[0]])
        # 实际上是为了保留原始 buffer index 和 mask index 的对应关系。
        # 我们简化处理，直接提取数据。
        
        # 1. 计算 Value Estimates (用于 Advantage 计算)
        # 需要对所有状态计算 Value，包括结束状态
        all_states = replay_buffer.get("states", idx)
        all_vals = self._compute_batch_vals(all_states) # 未标准化的 Value
        
        # 2. 计算 TD(lambda) Returns (New Values)
        new_vals = self._compute_batch_new_vals(replay_buffer, idx, all_vals)
        
        # 3. 更新 Critic
        # 动态调整步数
        critic_steps = int(np.ceil(self.critic_steps * new_sample_count / self.samples_per_iter))
        self._update_critic(critic_steps, idx, new_vals) # 使用所有数据更新 Critic

        # 4. 计算 Advantage
        # 仅在非结束状态上计算 Advantage 用于 Actor 更新
        # (因为结束状态没有动作？或者说我们只关心采取了动作的状态)
        # 原代码中 Actor 更新使用了 valid_idx (非结束状态)
        
        vals_valid = all_vals[valid_mask]
        new_vals_valid = new_vals[valid_mask]
        
        adv = new_vals_valid - vals_valid
        adv_mean = np.mean(adv)
        adv_std = np.std(adv)
        norm_adv = (adv - adv_mean) / (adv_std + self.ADV_EPS)
        
        # 计算权重
        weights = np.exp(norm_adv / self.temp)
        weights = np.minimum(weights, self.weight_clip)
        
        # 5. 更新 Actor
        actor_steps = int(np.ceil(self.actor_steps * new_sample_count / self.samples_per_iter))
        self._update_actor(actor_steps, valid_idx, weights, replay_buffer)

    def _compute_batch_vals(self, states: np.ndarray) -> np.ndarray:
        """
        计算状态价值 V(s)
        """
        # 分批计算以防显存溢出
        vals = []
        batch_size = 1024
        for i in range(0, len(states), batch_size):
            batch_states = states[i:i+batch_size]
            state_tensor = torch.FloatTensor(batch_states).to(self.device)
            norm_state_tensor = self.s_norm.normalize(state_tensor)
            
            with torch.no_grad():
                norm_vals = self.critic(norm_state_tensor)
                batch_vals = self.val_norm.unnormalize(norm_vals)
            vals.append(batch_vals.cpu().numpy())
            
        return np.concatenate(vals, axis=0).flatten()

    def _compute_batch_new_vals(self, buffer: ReplayBuffer, idx: np.ndarray, val_buffer: np.ndarray) -> np.ndarray:
        """
        使用 TD(lambda) 计算新的价值目标
        """
        new_vals = np.zeros_like(val_buffer)
        n = len(idx)
        
        start_i = 0
        while start_i < n:
            start_idx = idx[start_i]
            path_len = buffer.get_pathlen(start_idx)
            end_i = start_i + path_len
            
            # 提取当前路径的数据
            path_indices = idx[start_i : end_i + 1] # 包括结束状态
            
            # 获取奖励 (注意：rewards 长度通常比 states 少 1，或者对齐方式不同)
            # 在 ReplayBuffer 中，rewards 存储在 step t，对应 s_t -> s_{t+1}
            # path_indices 长度为 path_len + 1 (states)
            # rewards 长度为 path_len
            r = buffer.get("rewards", path_indices[:-1])
            v = val_buffer[start_i : end_i + 1]
            
            # 处理失败终止 (Fail) 的情况，Value 应为 0
            # 原代码逻辑：
            # is_fail = self._replay_buffer.check_terminal_flag(idx, rl_path.Terminate.Fail)
            # vals[is_fail] = 0.0
            # 这里我们在 _compute_return 中处理，或者预处理 v
            
            # 检查路径结束是否是 Fail
            end_idx = path_indices[-1]
            if buffer.check_terminal_flag(end_idx, Terminate.Fail):
                v[-1] = 0.0

            path_returns = self._compute_return(r, self.gamma, self.td_lambda, v)
            new_vals[start_i : end_i] = path_returns
            
            # 最后一个状态的 return 就是它的 value (或者 0 如果 fail)
            new_vals[end_i] = v[-1] 
            
            start_i = end_i + 1
            
        return new_vals

    def _compute_return(self, rewards: np.ndarray, discount: float, td_lambda: float, val_t: np.ndarray) -> np.ndarray:
        """
        计算单条路径的 TD(lambda) Return
        """
        path_len = len(rewards)
        assert len(val_t) == path_len + 1
        
        return_t = np.zeros(path_len)
        last_val = rewards[-1] + discount * val_t[-1]
        return_t[-1] = last_val
        
        for i in reversed(range(0, path_len - 1)):
            curr_r = rewards[i]
            next_ret = return_t[i + 1]
            curr_val = curr_r + discount * ((1.0 - td_lambda) * val_t[i + 1] + td_lambda * next_ret)
            return_t[i] = curr_val
            
        return return_t

    def _update_critic(self, steps: int, idx: np.ndarray, targets: np.ndarray):
        dataset_size = len(idx)
        indices = np.arange(dataset_size)
        
        for _ in range(steps):
            np.random.shuffle(indices)
            for start in range(0, dataset_size, self.batch_size):
                end = start + self.batch_size
                batch_idx = indices[start:end]
                
                # 获取数据
                buffer_idx = idx[batch_idx]
                batch_states = replay_buffer.get("states", buffer_idx)
                batch_targets = targets[batch_idx]
                
                # 转 Tensor
                s_tensor = torch.FloatTensor(batch_states).to(self.device)
                t_tensor = torch.FloatTensor(batch_targets).to(self.device).unsqueeze(1)
                
                # Normalize Inputs/Targets
                norm_s = self.s_norm.normalize(s_tensor)
                norm_t = self.val_norm.normalize(t_tensor)
                
                # Forward
                pred_norm_val = self.critic(norm_s)
                
                # Loss
                loss = 0.5 * F.mse_loss(pred_norm_val, norm_t)
                
                self.critic_optimizer.zero_grad()
                loss.backward()
                self.critic_optimizer.step()

    def _update_actor(self, steps: int, idx: np.ndarray, weights: np.ndarray, buffer: ReplayBuffer):
        dataset_size = len(idx)
        indices = np.arange(dataset_size)
        
        for _ in range(steps):
            np.random.shuffle(indices)
            for start in range(0, dataset_size, self.batch_size):
                end = start + self.batch_size
                batch_idx = indices[start:end]
                
                # 获取数据
                buffer_idx = idx[batch_idx]
                batch_states = buffer.get("states", buffer_idx)
                batch_actions = buffer.get("actions", buffer_idx)
                batch_weights = weights[batch_idx]
                
                # 转 Tensor
                s_tensor = torch.FloatTensor(batch_states).to(self.device)
                a_tensor = torch.FloatTensor(batch_actions).to(self.device)
                w_tensor = torch.FloatTensor(batch_weights).to(self.device)
                
                # Normalize State
                norm_s = self.s_norm.normalize(s_tensor)
                
                # 计算 Log Prob
                if self.is_discrete:
                    logits = self.actor(norm_s)
                    dist = distributions.Categorical(logits=logits)
                    log_prob = dist.log_prob(a_tensor)
                else:
                    # 连续动作：Actor 输出的是标准化后的 Mean
                    mean = self.actor(norm_s)
                    std = torch.exp(self.log_std)
                    dist = distributions.Normal(mean, std)
                    
                    # 这里的 batch_actions 是原始动作，需要标准化
                    norm_a = self.a_norm.normalize(a_tensor)
                    
                    log_prob = dist.log_prob(norm_a).sum(dim=-1)
                
                # Loss = - mean( weight * log_prob )
                actor_loss = -torch.mean(w_tensor * log_prob)
                
                # Action Bound Loss (仅连续)
                if not self.is_discrete:
                    bound_loss = self._action_bound_loss(mean)
                    actor_loss += 10.0 * bound_loss # 10.0 是原代码权重
                
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

    def _action_bound_loss(self, mean: Tensor) -> Tensor:
        """
        计算动作边界损失，防止动作均值超出标准化后的边界。
        """
        # 获取标准化后的边界
        bound_min = torch.FloatTensor(self.action_bound_min).to(self.device)
        bound_max = torch.FloatTensor(self.action_bound_max).to(self.device)
        
        norm_bound_min = self.a_norm.normalize(bound_min)
        norm_bound_max = self.a_norm.normalize(bound_max)
        
        violation_min = torch.clamp(mean - norm_bound_min, max=0) # 应该小于 min 的部分 (负数) -> 实际上是 mean < min -> mean - min < 0. 
        # 等等，原代码：violation_min = tf.minimum(val - norm_a_bound_min, 0)
        # 如果 val < min, val - min < 0. minimum 选它。
        # 然后 square。
        # 所以是惩罚小于 min 的部分。
        
        violation_max = torch.clamp(mean - norm_bound_max, min=0) # 应该大于 max 的部分
        # 原代码：violation_max = tf.maximum(val - norm_a_bound_max, 0)
        
        violation = torch.sum(violation_min**2, dim=-1) + torch.sum(violation_max**2, dim=-1)
        return 0.5 * torch.mean(violation)

    def record_normalizers(self, path: RLPath):
        """
        记录路径数据以更新 Normalizer
        """
        states = np.array(path.states)
        # 注意：path.states 包含最后一个状态，通常也包含在内
        self.s_norm.record(states)
        
    def update_normalizers(self):
        """
        执行 Normalizer 的参数更新
        """
        self.s_norm.update()

# ==========================================
# Main Example
# ==========================================

if __name__ == "__main__":
    # 简单的测试环境
    # env_name = "CartPole-v1" # 离散
    env_name = "Pendulum-v1" # 连续
    
    env = gym.make(env_name)
    
    # 实例化 Agent
    agent = AWRAgent(env, device='cpu')
    
    # 实例化 Buffer
    buffer = ReplayBuffer(buffer_size=10000)
    
    print(f"Start training on {env_name}...")
    
    for i in range(20):
        # 1. 收集数据
        path = RLPath()
        s = env.reset()
        path.states.append(s)
        
        done = False
        while not done:
            a, logp = agent.get_action(s)
            next_s, r, done, _ = env.step(a)
            
            path.states.append(next_s)
            path.actions.append(a)
            path.rewards.append(r)
            path.logps.append(logp)
            
            s = next_s
            
        # 2. 存储路径
        buffer.store(path)
        
        # 3. 记录并更新 Normalizer (通常在训练初期进行)
        if i < 5:
            agent.record_normalizers(path)
            agent.update_normalizers()
        
        # 4. 更新 Agent
        if buffer.num_paths >= 2: # 至少有几条路径再更新
            print(f"Iter {i}: Updating agent... Return: {path.calc_return():.2f}")
            agent.update(buffer, new_sample_count=path.pathlength())
            buffer.clear() # On-Policy 变体
            
    print("Done.")