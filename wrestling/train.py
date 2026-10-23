"""Train an imitation policy with PPO on CPU worker processes.

    python -m wrestling.train configs/skills/double_leg.json --out runs/double_leg
    python -m wrestling.train configs/skills/double_leg_smooth.json --init runs/double_leg/best.npz --out runs/double_leg_smooth

Each iteration collects `horizon` steps from every worker, updates the Lagrange multipliers
of the costs from the batch cost means, and runs a clipped PPO update. Every `eval_every`
iterations the deterministic policy is evaluated from frame 0; best.npz keeps the policy with
the highest success rate, then progress, then lowest joint error.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from .evaluate import rollout
from .policy import forward, load_checkpoint, save_checkpoint
from .skills import load_skill

_ENV = None


def _worker_init(skill: dict, seed: int):
    global _ENV
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[k] = "1"
    from .env import ImitationEnv

    _ENV = ImitationEnv(skill, seed=seed)
    _ENV.train_noise = (_ENV.cfg.rsi_joint_noise, _ENV.cfg.rsi_root_noise)
    _ENV.obs_now = _ENV.reset()
    _ENV.ep = [0.0, 0]


def _collect(args):
    p, horizon, seed, lam, gamma = args
    env = _ENV
    rng = np.random.default_rng(seed)
    env.cfg.lagrange = tuple(lam)
    od, ad = env.obs_dim, env.act_dim
    T = {
        "obs": np.zeros((horizon, od), np.float32),
        "act": np.zeros((horizon, ad), np.float32),
        "logp": np.zeros(horizon, np.float32),
        "rew": np.zeros(horizon, np.float32),
        "val": np.zeros(horizon, np.float32),
        "done": np.zeros(horizon, np.float32),
    }
    costs, ep_rets, ep_lens, falls = [], [], [], 0
    obs = env.obs_now
    for t in range(horizon):
        mu, ls, v = forward(p, obs)
        std = np.exp(ls)
        a = np.clip(mu + std * rng.normal(size=mu.shape), -1, 1)
        nobs, r, d, info = env.step(a)
        if d and info["term"] == "timeout":
            r += gamma * forward(p, nobs)[2]  # time limits are truncations, not terminal states
        T["obs"][t], T["act"][t], T["val"][t], T["done"][t] = obs, a, v, float(d)
        T["logp"][t] = -0.5 * np.sum(((a - mu) ** 2) / std**2 + 2 * ls + np.log(2 * np.pi))
        T["rew"][t] = r
        costs.append(info["costs"])
        env.ep[0] += r
        env.ep[1] += 1
        if d:
            ep_rets.append(env.ep[0])
            ep_lens.append(env.ep[1])
            falls += info["term"] == "fall"
            env.ep = [0.0, 0]
            nobs = env.reset(seed=int(rng.integers(1 << 30)))
        obs = nobs
    env.obs_now = obs
    T.update(
        last_val=forward(p, obs)[2],
        cost_mean=np.mean(costs, axis=0),
        ep_rets=ep_rets,
        ep_lens=ep_lens,
        falls=falls,
    )
    return T


def _evaluate(args):
    p, clip, seed = args
    env = _ENV
    r = rollout(env, p, clip, seed)
    env.cfg.rsi_joint_noise, env.cfg.rsi_root_noise = env.train_noise
    env.obs_now = env.reset()
    env.ep = [0.0, 0]
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", help="skill config JSON")
    ap.add_argument("--out", required=True)
    ap.add_argument("--init", default="", help="checkpoint to start from")
    ap.add_argument("--minutes", type=float, help="override the config's training time budget")
    ap.add_argument("--workers", type=int, help="override the config's worker count")
    ap.add_argument(
        "--iters", type=int, default=0, help="stop after this many iterations (0: time budget only)"
    )
    ap.add_argument("--seed", type=int)
    a = ap.parse_args()

    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    import jax
    import jax.numpy as jnp
    import optax
    from .ppo import ActorCritic, PolicyParams, PPOConfig, RolloutBatch, compute_gae, ppo_update

    skill = load_skill(a.config)
    tr = dict(skill["train"])
    for k in ("minutes", "workers", "seed"):
        if getattr(a, k) is not None:
            tr[k] = getattr(a, k)
    lagrange = skill["lagrange"]
    budget = np.asarray(lagrange["budgets"], float)
    lam = np.zeros(len(budget))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(
        json.dumps({"config": str(a.config), "init": a.init, "train": tr}, indent=1)
    )

    from .env import ImitationEnv

    probe = ImitationEnv(skill)
    od, ad = probe.obs_dim, probe.act_dim
    del probe
    ac = ActorCritic(od, ad, hidden=256)
    key = jax.random.PRNGKey(tr["seed"])
    key, k0 = jax.random.split(key)
    params = ac.init(k0)._replace(log_std=jnp.full((ad,), tr["init_log_std"]))
    if a.init:
        p0, _ = load_checkpoint(a.init)
        params = PolicyParams(**{k: jnp.asarray(v) for k, v in p0.items()})
        params = params._replace(log_std=jnp.minimum(params.log_std, tr["init_log_std"]))
    cfg = PPOConfig(
        lr=tr["lr"],
        gamma=tr["gamma"],
        gae_lambda=tr["gae_lambda"],
        n_epochs=tr["epochs"],
        minibatch_size=tr["minibatch"],
    )
    opt = optax.chain(
        optax.clip_by_global_norm(cfg.max_grad_norm),
        optax.inject_hyperparams(optax.adam)(learning_rate=tr["lr"]),
    )
    opt_state = opt.init(params)
    update = jax.jit(lambda p, s, b, k: ppo_update(p, s, b, cfg, k, opt))

    @jax.jit
    def kl_div(p0, p1, obs):
        m0, l0, _ = ac.forward(p0, obs)
        m1, l1, _ = ac.forward(p1, obs)
        return jnp.mean(
            jnp.sum(l1 - l0 + (jnp.exp(2 * l0) + (m0 - m1) ** 2) / (2 * jnp.exp(2 * l1)) - 0.5, axis=-1)
        )

    W = int(tr["workers"])
    pool = get_context("spawn").Pool(W, initializer=_worker_init, initargs=(skill, tr["seed"]))
    eval_jobs = [(k, s) for k in range(len(skill["clips"])) for s in range(int(tr["n_eval"]))]
    log = open(out / "train.jsonl", "w")
    best, lr_kl, active, t_last = None, tr["lr"], 0.0, time.time()
    n_iters = a.iters or 10**9
    print(f"{skill['name']}: obs {od}, act {ad}, {W} workers, {tr['minutes']} min", flush=True)
    for it in range(n_iters):
        frac = max(it / max(1, n_iters - 1), active / 60.0 / tr["minutes"])
        p_np = {k: np.asarray(v) for k, v in params._asdict().items()}
        res = pool.map(
            _collect,
            [
                (p_np, tr["horizon"], tr["seed"] * 1000 + it * 37 + w, lam.tolist(), cfg.gamma)
                for w in range(W)
            ],
        )
        cost = np.mean([R["cost_mean"] for R in res], axis=0)
        lam = np.clip(lam + lagrange["lambda_lr"] * (cost - budget), 0.0, lagrange["eta_max"])

        advs, rets = [], []
        for R in res:
            adv, ret = compute_gae(
                jnp.asarray(R["rew"]),
                jnp.asarray(R["val"]),
                jnp.asarray(R["done"]),
                R["last_val"],
                cfg.gamma,
                cfg.gae_lambda,
            )
            advs.append(np.asarray(adv))
            rets.append(np.asarray(ret))
        cat = lambda k: jnp.asarray(np.concatenate([R[k] for R in res]))
        batch = RolloutBatch(
            obs=cat("obs"),
            actions=cat("act"),
            logp=cat("logp"),
            advantages=jnp.asarray(np.concatenate(advs)),
            returns=jnp.asarray(np.concatenate(rets)),
            values=cat("val"),
        )

        lr = tr["lr"] * (1.0 - (1.0 - tr["lr_final_frac"]) * min(1.0, frac))
        if tr["target_kl"] > 0:
            lr = min(lr, lr_kl)
        opt_state[1].hyperparams["learning_rate"] = jnp.asarray(lr, jnp.float32)
        p_old = params
        params, opt_state, stats, key = update(params, opt_state, batch, key)
        if tr["max_log_std"] is not None:
            params = params._replace(log_std=jnp.minimum(params.log_std, tr["max_log_std"]))
        kl = float(kl_div(p_old, params, batch.obs[::4]))
        if tr["target_kl"] > 0:
            if kl > 2 * tr["target_kl"]:
                lr_kl = max(lr / 1.5, tr["lr"] * 0.02)
            elif kl < tr["target_kl"] / 2:
                lr_kl = min(lr_kl * 1.5, tr["lr"])

        ep_rets = sum((R["ep_rets"] for R in res), [])
        ep_lens = sum((R["ep_lens"] for R in res), [])
        row = {
            "iter": it,
            "minutes": round(active / 60, 2),
            "lr": lr,
            "kl": kl,
            "mean_reward": float(np.concatenate([R["rew"] for R in res]).mean()),
            "episode_return": float(np.mean(ep_rets)) if ep_rets else None,
            "episode_len": float(np.mean(ep_lens)) if ep_lens else None,
            "falls": int(sum(R["falls"] for R in res)),
            "std": float(np.exp(np.asarray(params.log_std)).mean()),
            "costs": np.round(cost, 4).tolist(),
            "lambda": np.round(lam, 4).tolist(),
            **{k: float(v) for k, v in stats.items()},
        }

        if it % tr["eval_every"] == 0:
            p_np = {k: np.asarray(v) for k, v in params._asdict().items()}
            ev = pool.map(_evaluate, [(p_np, k, s) for k, s in eval_jobs])
            row["eval"] = {
                "success": float(np.mean([e["success"] for e in ev])),
                "progress": float(np.mean([e["progress"] for e in ev])),
                "joint_mse": float(np.mean([e["joint_mse"] for e in ev])),
                "jitter": float(np.mean([e["jitter"] for e in ev])),
            }
            meta = {"skill": skill["name"], "iter": it, "obs_dim": od, "act_dim": ad, "eval": row["eval"]}
            save_checkpoint(out / "last.npz", p_np, meta)
            if tr["save_every"] and it % tr["save_every"] == 0:
                save_checkpoint(out / f"iter{it}.npz", p_np, meta)
            e = row["eval"]
            score = (round(e["success"], 6), round(e["progress"], 6), -e["joint_mse"])
            if best is None or score > best:
                best = score
                save_checkpoint(out / "best.npz", p_np, meta)
                row["best"] = True

        log.write(json.dumps(row) + "\n")
        log.flush()
        print(json.dumps(row), flush=True)
        now = time.time()
        active += min(now - t_last, 600.0)  # ignore time the machine spent suspended
        t_last = now
        if active / 60 > tr["minutes"]:
            break
    pool.close()
    print("best", best, flush=True)


if __name__ == "__main__":
    main()
