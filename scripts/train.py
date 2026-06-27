"""Training entrypoint for 2D AI Dodgeball.

Curriculum loop (Phase 1 -> 4), self-play with opponent pool + ELO,
checkpointing every snapshot_interval steps.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict, deque

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnvWrapper, VecNormalize

from dodgeball.env.dodgeball_parallel import DodgeBall2DParallelEnv
from dodgeball.wrappers import make_vec_env
from dodgeball.selfplay.opponent_pool import OpponentPool
from dodgeball.selfplay.elo import ELORater


class TensorboardCallback(BaseCallback):
    def __init__(self, log_every: int = 2048):
        super().__init__()
        self.log_every = log_every
        self.episode_returns = deque(maxlen=200)
        self.episode_lengths = deque(maxlen=200)
        self.episode_winner = deque(maxlen=200)  # -1 draw, 0/1
        self.last_log_step = 0

    def _on_step(self):
        # VecEnv returns dones per sub-env in infos at "episode" key
        infos = self.model.ep_info_buffer
        for info in infos:
            self.episode_returns.append(info.get("r", 0.0))
            self.episode_lengths.append(info.get("l", 0))
        if self.num_timesteps - self.last_log_step >= self.log_every:
            self.last_log_step = self.num_timesteps
            if self.episode_returns:
                self.logger.record("rollout/mean_ep_return",
                                   float(np.mean(list(self.episode_returns)[-50:])))
                self.logger.record("rollout/mean_ep_len",
                                   float(np.mean(list(self.episode_lengths)[-50:])))
        return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--total-timesteps", type=int, default=None,
                    help="Override total_timesteps from config")
    ap.add_argument("--num-envs", type=int, default=None)
    ap.add_argument("--mode", choices=["selfplay", "vs_bot"], default="selfplay")
    ap.add_argument("--opponent", default="stationary",
                    help="For vs_bot mode: stationary/random/heuristic/heuristic_dodger")
    ap.add_argument("--phase", type=int, default=None,
                    help="Skip curriculum, train this phase only")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--save-path", default="models/dodgeball_final")
    ap.add_argument("--resume-from", default=None,
                    help="Path to a saved PPO model to resume from. Loads weights + VecNormalize stats.")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    ppo_kwargs = dict(cfg["ppo"])
    if "policy_kwargs" in ppo_kwargs:
        ppo_kwargs["policy_kwargs"] = dict(ppo_kwargs["policy_kwargs"])
        act_name = ppo_kwargs["policy_kwargs"].pop("activation_fn", None)
        if act_name == "torch.nn.Tanh":
            import torch.nn as nn
            ppo_kwargs["policy_kwargs"]["activation_fn"] = nn.Tanh
    train_cfg = cfg["training"]
    curr_cfg = cfg["curriculum"]

    total_steps = args.total_timesteps or train_cfg["total_timesteps"]
    num_envs = args.num_envs or train_cfg["n_envs"]
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    os.makedirs("models", exist_ok=True)
    os.makedirs("models/snapshots", exist_ok=True)

    # Initial phase
    initial_phase = args.phase if args.phase else curr_cfg["initial_phase"]
    phases = [initial_phase] if args.phase else curr_cfg["phases"]

    # Self-play setup
    opponent_pool = OpponentPool(
        root="models/snapshots",
        max_size=curr_cfg["opponent_pool_max"],
        latest_prob=curr_cfg["latest_opponent_prob"],
    )
    elo = ELORater()
    snap_interval = curr_cfg["snapshot_interval"]
    chunk_steps = curr_cfg["chunk_steps"]

    # Build initial env
    print(f"Building vec env (mode={args.mode}, phase={initial_phase}, "
          f"num_envs={num_envs}) ...")
    env = make_vec_env(num_envs=num_envs,
                       mode=args.mode,
                       curriculum_phase=initial_phase,
                       opponent=args.opponent if args.mode == "vs_bot" else "self")

    if args.resume_from:
        print(f"Resuming from {args.resume_from}")
        model = PPO.load(args.resume_from, env=env, device=args.device,
                           tensorboard_log=train_cfg.get("tensorboard_log", "runs/"))
        # Load VecNormalize stats if present
        vn_path = args.resume_from + "_vecnorm.pkl"
        if os.path.exists(vn_path):
            env = VecNormalize.load(vn_path, env.venv)
            env.training = True
            env.norm_reward = False
            print(f"Loaded VecNormalize stats from {vn_path}")
        print(f"  resumed at step {model.num_timesteps}")
    else:
        model = PPO(
            "MlpPolicy", env, verbose=1,
            seed=seed,
            device=args.device,
            tensorboard_log=train_cfg.get("tensorboard_log", "runs/"),
            **ppo_kwargs,
        )

    cb = TensorboardCallback()

    # Train through phases
    steps_remaining = total_steps
    cur_step = model.num_timesteps if args.resume_from else 0
    if args.resume_from:
        print(f"Resuming at cur_step={cur_step}; will train until {total_steps}")
        steps_remaining = max(0, total_steps - cur_step)
    t0 = time.time()
    for phase in phases:
        env.venv.env_method("set_curriculum_phase", phase)
        env.venv.env_method("set_opponent_kind",
                            args.opponent if args.mode == "vs_bot" else "self")
        print(f"=== Phase {phase} ===")

        while steps_remaining > 0:
            this_chunk = min(chunk_steps, steps_remaining)
            cb.episode_returns.clear()
            cb.episode_lengths.clear()

            # self-play: rotate opponent in the envs (for vs_bot mode the
            # scripted bot handles itself; for selfplay we want every chunk
            # to sample a snapshot from the pool)
            if args.mode == "selfplay" and len(opponent_pool) > 0:
                snap_path = opponent_pool.sample(cur_step)
                if snap_path is not None:
                    opp_model = PPO.load(snap_path, device="cpu")
                    def _opp_act(obs_p0, obs_p1):
                        # Model trained on [obs_p0, obs_p1] (34-dim).
                        # obs_p1 is egocentric (mirrored frame);
                        # swap slot order so the model takes player_1
                        # perspective as input, and MIRROR its output back
                        # so the resulting action makes sense in player_1
                        # coordinates (x components flipped).
                        swap = np.concatenate([obs_p1, obs_p0], axis=0)
                        a, _ = opp_model.predict(swap, deterministic=False)
                        a[0] = -a[0]
                        a[2] = -a[2]
                        return a
                    env.venv.env_method("set_opponent_policy", _opp_act)
                else:
                    env.venv.env_method("set_opponent_policy", None)
            else:
                env.venv.env_method("set_opponent_policy", None)

            model.learn(total_timesteps=this_chunk,
                        reset_num_timesteps=False,
                        callback=cb)
            cur_step += this_chunk
            steps_remaining -= this_chunk

            # Save snapshot
            if len(opponent_pool) == 0 or (cur_step % snap_interval == 0):
                snap = opponent_pool.add(model, cur_step)
                print(f"  saved snapshot @ {cur_step}: {snap.path}")
                elo.get(f"snap_{cur_step}")

            elapsed = time.time() - t0
            print(f"  step={cur_step}/{total_steps}  elapsed={elapsed/60:.1f}min")

            # When --phase is given, we train a single phase to total_timesteps,
            # not break after one chunk. The break used to short-circuit at
            # one chunk — that was a bug. Now we just keep going until
            # steps_remaining == 0 naturally.

    model.save(args.save_path)
    env.save(f"{args.save_path}_vecnorm.pkl")
    print(f"Saved final model to {args.save_path}")
    print(f"ELO snapshot: {elo.snapshot()}")
    with open("models/elo.json", "w", encoding="utf-8") as f:
        json.dump(elo.snapshot(), f, indent=2)

    env.close()


if __name__ == "__main__":
    main()