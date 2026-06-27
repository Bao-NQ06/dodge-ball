"""Export a trained PPO policy to ONNX for use in Godot (or any ONNX runtime).

The PPO ActorCriticPolicy exposes two MLPs:
  - policy.mlp_extractor.policy_net  -> features
  - policy.action_net                -> mean (5-dim)
We bake the deterministic mean into a single forward pass. Sampling is
deterministic-by-default (mean); if you want exploration, add Gaussian noise
client-side. SB3 stores the policy under `model.policy`.

VecNormalize stats are exported separately to `model.onnx_vecnorm.json`
(mean and var per feature, plus epsilon) so the Godot side can normalize
obs the same way Python did.

Usage:
    python scripts/export_onnx.py \
        --model models/dodgeball_final.zip \
        --out models/dodgeball_policy.onnx
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from dodgeball.env.dodgeball_parallel import OBS_DIM, ACT_DIM
from dodgeball.wrappers import _PettingZooToGymWrapper, make_vec_env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="Path to .zip (without .zip suffix)")
    ap.add_argument("--out", default="models/dodgeball_policy.onnx")
    ap.add_argument("--obs-dim", type=int, default=34)
    ap.add_argument("--act-dim", type=int, default=5)
    args = ap.parse_args()

    # Load the policy net only (no env needed for the export itself,
    # but we want the VecNormalize stats).
    model = PPO.load(args.model, device="cpu")

    # --- export VecNormalize stats ---
    vn_path = args.model + "_vecnorm.pkl"
    vecnorm_stats = {}
    if os.path.exists(vn_path):
        # Build a dummy inner env matching what training used (single env,
        # no special mode).  VecNormalize.load needs the env shape.
        env = make_vec_env(num_envs=1, mode="selfplay",
                           curriculum_phase=4, opponent="self")
        vn = VecNormalize.load(vn_path, env.venv)
        vn.training = False
        # `obs_rms` is a RunningMeanStd with .mean, .var, .count
        rms = vn.obs_rms
        vecnorm_stats = {
            "mean": rms.mean.tolist(),
            "var": rms.var.tolist(),
            "epsilon": float(vn.epsilon),
            "clip_obs": float(vn.clip_obs),
        }
        env.close()

    # --- export policy MLP to ONNX ---
    # The policy net is: obs -> features -> mean (5-dim)
    # We need a single nn.Module that maps (B, 34) -> (B, 5).
    # SB3's ActorCriticPolicy already does this in forward(); the relevant
    # path is policy.mlp_extractor.policy_net followed by policy.action_net.
    class _PolicyForward(torch.nn.Module):
        def __init__(self, sb3_model):
            super().__init__()
            self.policy_net = sb3_model.policy.mlp_extractor.policy_net
            self.action_net = sb3_model.policy.action_net

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            features = self.policy_net(x)
            mean = self.action_net(features)
            return mean

    wrapper = _PolicyForward(model)
    wrapper.eval()

    dummy = torch.zeros(1, args.obs_dim, dtype=torch.float32)
    out_path = args.out
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    torch.onnx.export(
        wrapper,
        dummy,
        out_path,
        input_names=["obs"],
        output_names=["action_mean"],
        dynamic_axes={"obs": {0: "batch"}, "action_mean": {0: "batch"}},
        opset_version=17,
    )
    print(f"Exported policy -> {out_path}")

    # VecNormalize stats (if present)
    if vecnorm_stats:
        vn_json = out_path + ".vecnorm.json"
        with open(vn_json, "w", encoding="utf-8") as f:
            json.dump(vecnorm_stats, f, indent=2)
        print(f"Exported VecNormalize stats -> {vn_json}")

    # Smoke test: load ONNX and verify output shape
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])
        out = sess.run(None, {"obs": dummy.numpy()})
        print(f"ONNX smoke test OK: input={dummy.shape} -> output={out[0].shape}")
    except ImportError:
        print("(skipping smoke test - install onnxruntime to verify)")


if __name__ == "__main__":
    main()