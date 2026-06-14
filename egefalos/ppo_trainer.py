"""PPO Trainer — Proximal Policy Optimization for arithmetic self-play.

Connects MathGymEnv (with curriculum RL) to the MathTransformer
policy network with its existing value head.

Usage (via train_specialist.py --rl):
    python3 train_specialist.py add --rl --rl-steps 2000
"""

import math
import random
import time
from collections import deque
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from tabula_rasa.config import Config
from tabula_rasa.tokenizer import MathTokenizer
from tabula_rasa.model import MathTransformer


def ppo_loss(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    values: torch.Tensor,
    returns: torch.Tensor,
    clip_epsilon: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
) -> dict:
    """Compute PPO loss components.

    Args:
        log_probs: New action log probs. Shape (batch,).
        old_log_probs: Old action log probs. Shape (batch,).
        advantages: GAE advantages. Shape (batch,).
        values: Value predictions. Shape (batch,).
        returns: Discounted returns. Shape (batch,).
        clip_epsilon: PPO clip range.
        value_coef: Value loss coefficient.
        entropy_coef: Entropy bonus coefficient.

    Returns:
        dict with total_loss, policy_loss, value_loss, entropy, approx_kl
    """
    ratio = torch.exp(log_probs - old_log_probs)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()

    # Value loss (clipped)
    value_pred_clipped = values + (values - returns).clamp(-clip_epsilon, clip_epsilon)
    value_losses = (values - returns).pow(2)
    value_losses_clipped = (value_pred_clipped - returns).pow(2)
    value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()

    # Entropy bonus
    entropy = -(torch.exp(log_probs) * log_probs).mean()

    approx_kl = (ratio - 1 - torch.log(ratio)).mean()

    total_loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

    return {
        "total_loss": total_loss,
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "entropy": entropy,
        "approx_kl": approx_kl,
    }


def compute_gae(
    rewards: list[float],
    values: list[float],
    gamma: float = 0.99,
    lam: float = 0.95,
) -> tuple:
    """Compute Generalized Advantage Estimation.

    Args:
        rewards: Episode rewards, terminal reward last.
        values: Value predictions for each step (including terminal bootstrap).
        gamma: Discount factor.
        lam: GAE lambda.

    Returns:
        (advantages, returns) lists.
    """
    gae = 0
    advantages = []
    returns = []
    next_value = 0.0  # terminal state has value 0
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * lam * gae
        advantages.insert(0, gae)
        returns.insert(0, gae + values[t])
        next_value = values[t]
    return advantages, returns


class RolloutBuffer:
    """Stores episode transitions for PPO update."""

    def __init__(self):
        self.states: list[torch.Tensor] = []
        self.actions: list[int] = []
        self.log_probs: list[float] = []
        self.rewards: list[float] = []
        self.values: list[float] = []
        self.dones: list[bool] = []

    def add(self, state, action, log_prob, reward, value, done):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()

    def __len__(self):
        return len(self.states)


class PPOTrainer:
    """PPO training loop for MathTransformer with MathGymEnv.

    Args:
        model: MathTransformer with use_value_head=True.
        tokenizer: MathTokenizer.
        env: MathGymEnv instance (with curriculum=True for auto-advance).
        lr: Learning rate for both policy and value.
        clip_epsilon: PPO clip range.
        gamma: Discount factor.
        gae_lambda: GAE lambda.
        value_coef: Value loss weight.
        entropy_coef: Entropy bonus weight.
        update_epochs: Number of epochs per PPO update.
        batch_size: Minibatch size for PPO update.
        horizon: Steps collected before each PPO update.
        max_grad_norm: Gradient clipping norm.
        device: Torch device.
    """

    def __init__(
        self,
        model: MathTransformer,
        tokenizer: MathTokenizer,
        env,
        lr: float = 3e-4,
        clip_epsilon: float = 0.2,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        update_epochs: int = 4,
        batch_size: int = 64,
        horizon: int = 512,
        max_grad_norm: float = 0.5,
        device: torch.device = torch.device("cpu"),
    ):
        self.model = model
        self.tok = tokenizer
        self.env = env
        self.device = device

        self.clip_epsilon = clip_epsilon
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.update_epochs = update_epochs
        self.batch_size = batch_size
        self.horizon = horizon
        self.max_grad_norm = max_grad_norm

        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
        self.buffer = RolloutBuffer()

        # Check value head exists
        if model.value_head is None:
            raise ValueError(
                "PPO requires use_value_head=True in model config. "
                "Add cfg.use_value_head = True before creating the model."
            )

    @torch.no_grad()
    def _get_action_and_value(self, prompt_ids: torch.Tensor) -> tuple:
        """Get action from policy and value from value head.

        Args:
            prompt_ids: Tensor (1, seq_len) of token IDs.

        Returns:
            (action_id, log_prob, value) tuple.
        """
        prompt_ids = prompt_ids.to(self.device)
        self.model.eval()

        # Forward pass to get logits and value
        logits, _, value = self.model(prompt_ids)
        last_logits = logits[0, -1, :]  # (vocab_size,)
        value = value[0, 0].item()  # scalar

        # Sample action from policy
        probs = F.softmax(last_logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action).item()
        action_id = action.item()

        return action_id, log_prob, value

    def _collect_rollout(self, max_steps: int = 512) -> dict:
        """Collect one rollout trajectory using current policy.

        Args:
            max_steps: Max steps for this rollout.

        Returns:
            dict with episode stats.
        """
        self.buffer.clear()
        obs, info = self.env.reset()
        prompt_ids = obs[0].unsqueeze(0)  # (1, seq_len)

        ep_reward = 0.0
        ep_steps = 0
        expr = info["expression"]

        for step in range(max_steps):
            action_id, log_prob, value = self._get_action_and_value(prompt_ids)
            obs, reward, terminated, truncated, info = self.env.step(action_id)

            self.buffer.add(
                state=prompt_ids.clone(),
                action=action_id,
                log_prob=log_prob,
                reward=reward,
                value=value,
                done=terminated or truncated,
            )

            ep_reward += reward
            ep_steps += 1
            prompt_ids = obs[0].unsqueeze(0)

            if terminated or truncated:
                break

        return {
            "expression": expr,
            "reward": ep_reward,
            "steps": ep_steps,
            "correct": ep_reward > 0,
        }

    def _update(self) -> dict:
        """Perform one PPO update on collected rollout buffer.

        Returns:
            dict with loss components.
        """
        if len(self.buffer) < 4:
            return {"policy_loss": 0, "value_loss": 0, "entropy": 0, "approx_kl": 0}

        # Convert buffer to tensors
        states = torch.cat(self.buffer.states, dim=0).to(self.device)  # (T, seq_len)
        actions = torch.tensor(self.buffer.actions, device=self.device)
        old_log_probs = torch.tensor(self.buffer.log_probs, device=self.device)
        rewards = self.buffer.rewards
        values = self.buffer.values

        # Compute GAE
        advantages, returns = compute_gae(rewards, values, self.gamma, self.gae_lambda)
        advantages = torch.tensor(advantages, device=self.device)
        returns = torch.tensor(returns, device=self.device)
        # Normalize advantages
        if advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO update epochs
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_kl = 0.0
        n_updates = 0

        for epoch in range(self.update_epochs):
            # Mini-batch indices
            indices = torch.randperm(len(self.buffer))
            for start in range(0, len(indices), self.batch_size):
                batch_idx = indices[start:start + self.batch_size]

                batch_states = states[batch_idx]
                batch_actions = actions[batch_idx]
                batch_old_logp = old_log_probs[batch_idx]
                batch_adv = advantages[batch_idx]
                batch_ret = returns[batch_idx]

                # Get new log probs and values
                batch_states = batch_states.to(self.device)
                logits, _, values_pred = self.model(batch_states)
                # Get log probs for the taken actions
                # logits shape: (B, seq_len, vocab), we want last action logits
                action_logits = logits[:, -1, :]  # (B, vocab)
                probs = F.softmax(action_logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                new_log_probs = dist.log_prob(batch_actions)
                new_values = values_pred[:, 0]  # (B,)

                # PPO loss
                losses = ppo_loss(
                    new_log_probs, batch_old_logp, batch_adv,
                    new_values, batch_ret,
                    self.clip_epsilon, self.value_coef, self.entropy_coef,
                )

                self.optimizer.zero_grad()
                losses["total_loss"].backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_policy_loss += losses["policy_loss"].item()
                total_value_loss += losses["value_loss"].item()
                total_entropy += losses["entropy"].item()
                total_kl += losses["approx_kl"].item()
                n_updates += 1

        return {
            "policy_loss": total_policy_loss / max(1, n_updates),
            "value_loss": total_value_loss / max(1, n_updates),
            "entropy": total_entropy / max(1, n_updates),
            "approx_kl": total_kl / max(1, n_updates),
        }

    def train(
        self,
        total_steps: int = 5000,
        log_interval: int = 10,
        eval_interval: int = 50,
    ) -> dict:
        """Run PPO training loop.

        Args:
            total_steps: Total environment steps.
            log_interval: Log every N episodes.
            eval_interval: Evaluate every N episodes.

        Returns:
            dict with training stats.
        """
        self.model.train()
        episode = 0
        total_env_steps = 0
        reward_history = deque(maxlen=20)
        correct_history = deque(maxlen=20)
        start_time = time.time()

        print(f"\n{'='*60}")
        print(f"  PPO Training — {total_steps} env steps")
        print(f"  gamma={self.gamma} clip={self.clip_epsilon} lr={self.optimizer.param_groups[0]['lr']}")
        print(f"  Device: {self.device}")
        print(f"{'='*60}")

        while total_env_steps < total_steps:
            # Collect rollout
            rollout = self._collect_rollout()
            total_env_steps += rollout["steps"]
            episode += 1
            reward_history.append(rollout["reward"])
            correct_history.append(1 if rollout["correct"] else 0)

            # PPO update every horizon steps
            if len(self.buffer) >= self.horizon or total_env_steps >= total_steps:
                loss_info = self._update()

            # Logging
            if episode % log_interval == 0:
                avg_reward = sum(reward_history) / max(1, len(reward_history))
                avg_correct = sum(correct_history) / max(1, len(correct_history)) * 100
                elapsed = time.time() - start_time
                cur_digits = getattr(self.env, '_curriculum_current_digits', '?')

                print(
                    f"  Ep {episode:>4d} | steps={total_env_steps:>5d}/{total_steps} | "
                    f"reward={avg_reward:.2f} | correct={avg_correct:.0f}% | "
                    f"digits={cur_digits} | {elapsed:.0f}s",
                    flush=True,
                )

            # Evaluation
            if episode % eval_interval == 0:
                self.model.eval()
                eval_correct = 0
                for _ in range(20):
                    r = self._collect_rollout(max_steps=16)
                    if r["correct"]:
                        eval_correct += 1
                eval_acc = eval_correct / 20 * 100
                print(f"    [Eval] {eval_acc:.0f}% ({eval_correct}/20)", flush=True)
                self.model.train()

        elapsed = time.time() - start_time
        avg_reward = sum(reward_history) / max(1, len(reward_history))
        avg_correct = sum(correct_history) / max(1, len(correct_history)) * 100

        print(f"\n{'='*60}")
        print(f"  PPO Training Complete")
        print(f"  Episodes: {episode} | Steps: {total_env_steps} | Time: {elapsed:.0f}s")
        print(f"  Avg reward: {avg_reward:.2f} | Avg correct: {avg_correct:.0f}%")
        print(f"{'='*60}")

        return {
            "episodes": episode,
            "env_steps": total_env_steps,
            "time_s": elapsed,
            "avg_reward": avg_reward,
            "avg_correct_pct": avg_correct,
        }
