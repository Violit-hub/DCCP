# DCCP

**Decision-Centric Counterfactual Preference Optimization for Vision-Language-Action Post-Training**

DCCP extends the WMPO imagined-rollout pipeline with local action preferences. It mines decision-sensitive states using progress curvature and action-token entropy, compares short counterfactual branches under the same prefix, and combines a local DPO-style preference loss with trajectory-level GRPO.

This release is organized from the existing server working tree. The primary training entry is **`dccp/run_train_dccp_full.sh`**. Current ready-to-configure assets and launch defaults target MimicGen Coffee. Model weights and full demonstration datasets are supplied separately.

## Files

```text
dccp/
  run_train_dccp_full.sh       # Main training launcher
  run_train_dccp_smoke.sh      # Reduced real-pipeline check
  verl/utils/dccp_*.py        # Mining, branching, scoring, preferences
  verl/workers/actor/dp_rob.py # GRPO + local preference loss
  reward_model/lrm_server/    # Completion and progress adapters
  scripts/merge_lora_adapter.py
  experiments/               # State-selection and branch-validity audits
  dependencies/              # OpenSora and OpenVLA-OFT source
Large-Reward-Models/          # Backend and Coffee reference frames
assets/                      # Initial images, dataset config, WM config
docs/environments/           # Observed server package versions
```

## Setup

Use separate training and LRM environments. The server training environment uses Python 3.11, PyTorch 2.5.1, PEFT 0.11.0 and the OpenVLA-OFT Transformers fork. Exact observed package versions are in [docs/environments](docs/environments). They are environment records, not a tested installation lockfile.

In the training environment:

```bash
cd dccp
pip install -r requirements.txt
# Install FlashAttention after the compatible PyTorch/CUDA build is ready:
pip install --no-build-isolation -r requirements-flash-attn.txt
bash install.sh
cd ..
```

`install.sh` follows the existing project setup, including simulator dependencies and system packages. In the separate LRM environment, install `Large-Reward-Models/vlm_reward/requirements.txt`; match the recorded LRM environment when using the retained Qwen3-VL backend.

## Run

Copy `.env.example` to `.env.local`, set the real policy, world-model, VAE, LRM and HDF5 dataset paths, then load it in each terminal:

```bash
source .env.local
python scripts/prepare_coffee.py --dataset "$DCCP_DATASET_HDF5"
```

Start the services in separate terminals using the LRM environment:

```bash
bash dccp/reward_model/lrm_server/start_completion_server.sh
bash dccp/reward_model/lrm_server/start_progress_server.sh
```

In the training environment:

```bash
source .env.local
bash dccp/run_train_dccp_full.sh
# Optional reduced real-pipeline check:
# bash dccp/run_train_dccp_smoke.sh
```

The launcher defaults to `G=8`, two selected states, eight candidates including the nominal action, three action chunks per branch, margins `0.10`, `beta=0.1`, and preference weight `0.3`. Set `CUDA_VISIBLE_DEVICES`, `N_GPUS`, `TOTAL_EPOCHS`, and other environment variables as needed. Additional Hydra overrides can be passed after the script name.

The default one-GPU, two-epoch configuration is an engineering launcher; it does not enforce the paper's 1280-rollout budget or reproduce the full eight-H100 protocol. The smoke configuration uses fewer states/candidates and shorter branches.

## Checkpoints and evaluation

The launcher writes checkpoints under `dccp/checkpoints/coffee_dccp` and logs under `dccp/logs`. LoRA adapters can be merged using `dccp/scripts/merge_lora_adapter.py --help`.

Retained evaluation code includes the OpenVLA-OFT LIBERO evaluator and the Coffee state-selection / branch-validity audits. Their README files describe the protocols. Historical evaluation and audit examples contain `/path/to/...` placeholders that must be configured before GPU runs. `verl/trainer/main_eval.py` is inherited text-task evaluation code, not the robot evaluation entry point.

## Release status

See [VALIDATION.md](VALIDATION.md) for checks run on this package. Full GPU training and the complete paper benchmark results have not been rerun during packaging. Custom LIBERO-BDDL task assets, paper-specific checkpoints and full multi-task experiment settings still need to be supplied for complete paper reproduction.

Existing Apache-2.0 and MIT notices are retained. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for component scope. Add the final paper citation after the author details are settled.

Repository: [Violit-hub/DCCP](https://github.com/Violit-hub/DCCP).
