# Backup and restore (node wipe, Wed 7 Oct 2026 morning)

The node is wiped clean on Wednesday 7 October in the morning. Code is on GitHub
(`lukaskrl/foundation_model_testing`, branch `probes-and-window-arm`). Everything below is
outside git and has to be copied off and put back at the **same absolute paths**: about
900 configs hard-code `/home/lukas/projects/foundation_model_testing/...` and
`/home/lukas/data/...` (weights, dataset root, upsampler checkpoint, volume store).

## 1. What to copy

| priority | path | size | why |
|---|---|---|---|
| must | `/home/lukas/projects/foundation_model_testing/unified/runs/` | 23 GB | every trained checkpoint: the interface heads (test evals, resume of the SAM-Med3D trilinear control), the upsamplers (`runs/upsampler/*/best.pt`, incl. the one every head uses: `dino3d_label/best.pt`), queue scripts and logs |
| must | `/home/lukas/projects/foundation_model_testing/weights/` | 16 GB | pretrained encoder weights; several are gated or slow to fetch |
| must | `/home/lukas/.claude/projects/-home-lukas-projects-foundation-model-testing/` | small | Claude Code memory and session history for this project |
| must | `/home/lukas/.netrc`, `.gitconfig`, `.ssh/`, `.hf-cli/`, `.claude.json` | small | wandb / git / Hugging Face credentials, Claude Code config (or log in again) |
| should | `/home/lukas/data/TotalSegmentatorDataset/` | 32 GB | the dataset; re-downloadable, but slowly |
| should | `/home/lukas/data/cache/TotalSeg/3e6024a06c45/` | 136 GB | the 1.5 mm preprocessing cache every interface run reads; without it the first epoch of the first run rebuilds it |
| optional | `/home/lukas/data/cache/upsampler/` | 31 GB | probe volume store (`vol15`, 19 GB) and feature banks (`banks`, 12 GB); both are rebuilt automatically (`scripts/upsampler_prepare.py`, then the probe builds missing banks) in roughly 1–2 h |
| optional | `/home/lukas/data/model_checkpoints/` | 11 MB | |
| optional | `/tmp/claude-1007/-home-lukas-projects-foundation-model-testing/` | 290 MB | Claude's scratch space: literature notes from the prior-art searches (`*/scratchpad/lit*_NOTES.md`) and the papers they cite, for the related-work section |
| skip | `/home/lukas/data/cache/TotalSeg/833d9d046402/` | 42 GB | the old 2.25 mm cache (pre-spacing-fix runs only) |
| skip | `/home/lukas/data/cache/ctfm_lighter/` | 136 GB | CT-FM original-pipeline arm only; rebuilt on use |
| skip | `../env/` | 5 GB | rebuilt from `requirements.lock.txt` |
| skip | `/home/lukas/data/TotalSegmentatorDataset_ctfm/` | 148 KB | symlink view, rebuilt by `scripts/setup_ctfm_original.py` |

Not covered here: other things in the home directory (`~/3Ddinov3`, `~/env`, ...).

Copy `runs/interface/samMed3d_i1t_frz_pt_f100/` last: it is the SAM-Med3D trilinear control,
still training on the backup day. `runs/upsampler/stop_i1t_ep90.sh` stops it right after its
epoch-90 checkpoint is written (expected Tue 6 Oct ~19:00) and logs the stop, with a load check of
`epoch_0090.pt`, to `runs/upsampler/gpu0_oct2.log`. Copy the folder after that line appears.

## 2. Restore

```bash
cd /home/lukas/projects
git clone https://github.com/lukaskrl/foundation_model_testing.git
cd foundation_model_testing
git checkout probes-and-window-arm
git submodule update --init            # the eleven upstream repos, pinned
python3.10 -m venv env
env/bin/pip install -r unified/requirements.lock.txt
# put back: unified/runs/, weights/, /home/lukas/data/..., ~/.claude/projects/..., dotfiles
wandb login                            # unless ~/.netrc was restored
cd unified
../env/bin/python scripts/verify_setup.py --load-weights   # all encoders load
../env/bin/python -m scripts.test_upsampler                # upsampler unit tests
nvidia-smi
```

The CT-FM original-pipeline arm (not needed for the upsampler paper) also needs its own venv
and `../CT-FM/.venv/bin/python scripts/setup_ctfm_original.py` (lighter patch + dataset view).

## 3. Relaunch

GPU launches need the user's go-ahead on a named GPU.

| what | command (from `unified/`) | then |
|---|---|---|
| SAM-Med3D I1 + trilinear control, resume to 250 epochs (~2 days left) | `GPU=<n> EPOCHS=250 bash scripts/run_interface.sh configs/interface/samMed3d_i1t_frz_pt_f100.yaml` — resumes from the newest `epoch_*.pt` | `python -m scripts.upsampler_test_eval --run samMed3d_i1t_frz_pt_f100` |
| any test eval left unfinished | `python -m scripts.upsampler_test_eval --run <run>` — skips the cases already in `results/upsampler/test_heads/<run>_test.json` | |
| any probe left unfinished | rerun the same `scripts.upsampler_probe` command — skips methods already in the JSON | |

What ran before the wipe, and its state at backup time, is in `UPSAMPLER_PLAN.md`
("Results of the 2026-10-05 round") and in `runs/upsampler/gpu0_oct2.log`.
