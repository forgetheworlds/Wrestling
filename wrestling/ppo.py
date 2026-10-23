"""MLP actor-critic and a clipped PPO update with GAE (JAX)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax


class PolicyParams(NamedTuple):
    w1: jnp.ndarray
    b1: jnp.ndarray
    w2: jnp.ndarray
    b2: jnp.ndarray
    w_mu: jnp.ndarray
    b_mu: jnp.ndarray
    w_v: jnp.ndarray
    b_v: jnp.ndarray
    log_std: jnp.ndarray


def _xavier(key, fan_in, fan_out, scale=1.0):
    lim = scale * np.sqrt(6.0 / (fan_in + fan_out))
    return jax.random.uniform(key, (fan_in, fan_out), minval=-lim, maxval=lim)


class ActorCritic:
    """Two tanh hidden layers shared by a Gaussian policy head (tanh mean) and a value head."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256):
        self.obs_dim, self.act_dim, self.hidden = obs_dim, act_dim, hidden

    def init(self, key: jax.Array) -> PolicyParams:
        k1, k2, k3, k4 = jax.random.split(key, 4)
        h = self.hidden
        return PolicyParams(
            w1=_xavier(k1, self.obs_dim, h),
            b1=jnp.zeros((h,)),
            w2=_xavier(k2, h, h),
            b2=jnp.zeros((h,)),
            w_mu=_xavier(k3, h, self.act_dim, scale=0.01),
            b_mu=jnp.zeros((self.act_dim,)),
            w_v=_xavier(k4, h, 1),
            b_v=jnp.zeros((1,)),
            log_std=jnp.full((self.act_dim,), -0.5),
        )

    @staticmethod
    def forward(params: PolicyParams, obs: jnp.ndarray):
        x = jnp.tanh(obs @ params.w1 + params.b1)
        x = jnp.tanh(x @ params.w2 + params.b2)
        mu = jnp.tanh(x @ params.w_mu + params.b_mu)
        v = (x @ params.w_v + params.b_v).squeeze(-1)
        return mu, params.log_std, v


def gaussian_log_prob(mu, log_std, actions):
    var = jnp.exp(2.0 * log_std)
    return jnp.sum(-0.5 * (((actions - mu) ** 2) / var + 2.0 * log_std + jnp.log(2.0 * jnp.pi)), axis=-1)


def gaussian_entropy(log_std):
    return jnp.sum(log_std + 0.5 * jnp.log(2.0 * jnp.pi * jnp.e), axis=-1)


@dataclass
class PPOConfig:
    lr: float = 3e-4
    clip_eps: float = 0.2
    gamma: float = 0.99
    gae_lambda: float = 0.95
    vf_coef: float = 0.5
    ent_coef: float = 0.0
    max_grad_norm: float = 1.0
    n_epochs: int = 5
    minibatch_size: int = 512


class RolloutBatch(NamedTuple):
    obs: jnp.ndarray
    actions: jnp.ndarray
    logp: jnp.ndarray
    advantages: jnp.ndarray
    returns: jnp.ndarray
    values: jnp.ndarray


def compute_gae(rews, values, dones, last_value, gamma, lam):
    T = rews.shape[0]

    def body(carry, t):
        gae, next_value = carry
        i = T - 1 - t
        nonterminal = 1.0 - dones[i]
        delta = rews[i] + gamma * next_value * nonterminal - values[i]
        gae = delta + gamma * lam * nonterminal * gae
        return (gae, values[i]), gae

    _, adv_rev = jax.lax.scan(body, (0.0, last_value), jnp.arange(T))
    advantages = adv_rev[::-1]
    return advantages, advantages + values


def ppo_loss(params: PolicyParams, batch: RolloutBatch, cfg: PPOConfig):
    mu, log_std, v_pred = ActorCritic.forward(params, batch.obs)
    ratio = jnp.exp(gaussian_log_prob(mu, log_std, batch.actions) - batch.logp)
    adv = (batch.advantages - batch.advantages.mean()) / (batch.advantages.std() + 1e-8)
    pg_loss = -jnp.mean(
        jnp.minimum(ratio * adv, jnp.clip(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * adv)
    )
    v_loss = jnp.mean((v_pred - batch.returns) ** 2)
    ent = jnp.mean(gaussian_entropy(log_std))
    loss = pg_loss + cfg.vf_coef * v_loss - cfg.ent_coef * ent
    return loss, {"pg_loss": pg_loss, "v_loss": v_loss}


def ppo_update(params, opt_state, batch: RolloutBatch, cfg: PPOConfig, key, optimizer):
    """n_epochs of shuffled minibatch updates over one rollout batch."""
    n, mb = batch.obs.shape[0], cfg.minibatch_size

    def epoch(carry, _):
        params, opt_state, key = carry
        key, sk = jax.random.split(key)
        perm = jax.random.permutation(sk, n)

        def minibatch(carry2, start):
            params, opt_state = carry2
            idx = jax.lax.dynamic_slice(perm, (start,), (mb,))
            mbatch = RolloutBatch(*(f[idx] for f in batch))
            (_, stats), grads = jax.value_and_grad(ppo_loss, has_aux=True)(params, mbatch, cfg)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            return (optax.apply_updates(params, updates), opt_state), stats

        (params, opt_state), stats = jax.lax.scan(minibatch, (params, opt_state), jnp.arange(n // mb) * mb)
        return (params, opt_state, key), jax.tree.map(lambda x: x.mean(), stats)

    (params, opt_state, key), stats = jax.lax.scan(epoch, (params, opt_state, key), jnp.arange(cfg.n_epochs))
    return params, opt_state, jax.tree.map(lambda x: x.mean(), stats), key
