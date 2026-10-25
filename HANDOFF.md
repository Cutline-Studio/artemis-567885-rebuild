# Artemis 567885 — rebuild handoff

This repo (`lpittman-blip/artemis-567885-rebuild`) is a full copy of
[`lpittman-blip/artemis-567885`](https://github.com/lpittman-blip/artemis-567885), made on
2026-10-24 (ET) so Artemis can be rebuilt without undoing the original.
**The original repo was not modified** — no pushes, branch changes, PR edits or merges. Every
branch and tag was copied with its full history; the ref SHAs here match the original exactly
at copy time (14 branches, 1 tag).

Every number below is copied from files in this repo (`PROGRESS.md`, `PLAN.md`, `docs/`,
`checks/README.md`, `config.json`, `runs/`) or from the original repo's PR descriptions, and the
file is named. Anything not in the repo is marked as such.

## Important: `main` is only the baseline

`main` holds just the original four files (`README.md`, `moe.py`, `transformer.py`,
`sandbox.py`) plus a README edit. **All later work lives on the stacked stage branches**, none of
which was ever merged into `main`. The most complete code is the tip of **`stage-8-routing`**
(commit `1b007ed`), which contains stages 1–8 and the repo checks. The tests, training commands
and module list below refer to that branch:

```bash
git checkout stage-8-routing
```

## What Artemis is

A tokenization-free, **byte-level mixture-of-experts (MoE) language model**, CPU-first, written
in PyTorch. Raw UTF-8 bytes (0..255) go into `nn.Embedding(256, d)`, through N causal MoE
transformer blocks (top-k sparse expert routing with an auxiliary load-balancing loss and
optional capacity limits), a final LayerNorm and `Linear(d, 256)`, predicting the next byte.
Quality is measured in held-out bits per byte (bpb) against a unigram baseline. It is one model;
the "experts" are sub-networks inside each MoE layer, not separate LLMs. Alongside the LM there
is an auto-growing "expand engine" (adds experts/layers when loss stalls) and a separate physics
(PINN) sandbox.

## Repo layout (branch `stage-8-routing`)

| Path | What it is |
|---|---|
| `moe.py` | `CapacitySparseMoELayer`: top-k router, aux balance loss, capacity limits (`capacity_mode="global"` default, causal-safe `"sequence"`), loop and vectorized dispatch, `gate_weighting` (`renorm`/`switch`), router z-loss, `routing` (`token_choice`/`expert_choice`), `expert_load_entropy`. `legacy=True` reproduces the original code bit-for-bit. |
| `transformer.py` | Multi-head attention + `MoETransformerBlock` (causal mask option). |
| `positional.py` | Positional encodings: `none`, `sinusoidal`, `learned`. |
| `model.py` | `DynamicMultiBlockTransformer` (N blocks, `add_layer` identity init, `add_expert` copy+noise) and `ArtemisModel` (byte LM), plus `next_byte_loss`, `bits_per_byte`. |
| `train_bytes.py` | Stage-6+ byte-level CPU training loop: held-out bpb, unigram baseline, sampling, curve/summary logging. |
| `expand.py`, `supervisor.py`, `config.json` | Expand engine (X → sin(X) regression, growth rules and caps) and the self-fixing supervisor (backup, ast check, subprocess timeout, shape auto-correct, rollback). |
| `bench.py`, `benchmarks/` | Loop vs vectorized dispatch benchmark (parity-gated) and results. |
| `sandbox.py` | Original verification harness; `--mode legacy` / `fixed` / `both`. |
| `physics_simulation.py`, `physics_solver.py`, `pilot/` | Physics PINN sandbox, FD solver, SIREN frequency/capacity study. |
| `bytelm/` | Earlier byte-LM corpus benchmark, corpus builder (`build_corpus.py`) and `corpus/MANIFEST.md` (sha256 of every book). |
| `checks/` | Five pre-integration checks with raw consoles (`checks/README.md`). |
| `tests/` | pytest suite. |
| `tools/` | Stage evaluation/sweep scripts (`stage6_full_eval.py`, `stage7_sweep.py`, `stage8_sweep.py`, `stage8_gate_grad.py`, `investigate_baseline.py`). |
| `docs/` | `REVIEW.md` (verified findings), `STAGE7_SWEEP.md`, `STAGE8_ROUTING.md`, results note, the user's concept specs, `reference/clf_stream_original.py`. |
| `runs/` | Measured run logs (stage6/7/8 curves and summaries, expand/supervisor logs). Checkpoints `runs/*.pt` are git-ignored. |
| `PLAN.md`, `PROGRESS.md` | Staged roadmap and the iteration log. |

## Install, test, train (verified in this copy)

Environment used for verification: Python 3.13.5, torch 2.14.1+cpu, pytest 9.1.1, Linux, CPU only.

```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch
pip install pytest numpy

git checkout stage-8-routing
python3 -m pytest -q tests checks        # full suite
python3 sandbox.py --mode legacy         # original numbers
python3 sandbox.py --mode fixed
```

Measured in this copy on 2026-10-24 (ET), branch `stage-8-routing` @ `1b007ed`:

- `python3 -m pytest -q tests checks -p no:cacheprovider` → **274 passed, 1 xfailed in 46.01s**
  (matches the count in PROGRESS.md iteration 6). The xfail is the strict float32 tolerance
  case from check 3.
- `python3 sandbox.py --mode legacy` → Total Loss 1.5083, dropped 6, ∇x 0.1581, ∇W_g 0.0546.
- `python3 sandbox.py --mode fixed` → Total Loss 0.3415.
- Branch `expand-loop` @ `80ffa76`: `python3 -m pytest -q tests checks` → 134 passed, 1 xfailed.
- `main`: no tests (pytest collects nothing).

Training needs the Gutenberg corpus, which is **not in git**. Build it with
`bytelm/build_corpus.py` (it reads `raw/pg<id>.txt` for the Gutenberg ids listed in the script and
writes `corpus/{train,val,test}`), then check every file's sha256 against
`bytelm/corpus/MANIFEST.md`. `train_bytes.py` looks for `--corpus DIR`, else `$ARTEMIS_CORPUS`,
else `/workspace/artemis-corpus-bench/corpus`.

```bash
# Stage 6 run exactly as recorded in PROGRESS.md (900 s CPU budget, seed 567885 by default):
python3 train_bytes.py --run-name gutenberg_cpu_900s --prompt "It was a " --prompt "The " --save-checkpoint

# Quick smoke run (verified working in this copy: 20 steps, 0.3 s):
python3 train_bytes.py --max-steps 20 --eval-every 10 --d-model 64 --layers 2 --experts 4 \
  --seq-len 64 --batch 8 --eval-windows 16 --out-dir /tmp/artemis-smoke --run-name smoke

# Sweeps
python3 tools/stage7_sweep.py      # --report to aggregate only
python3 tools/stage8_sweep.py      # --report to aggregate only

# Expand engine
python3 expand.py init
python3 supervisor.py --cycles 10 --timeout 300
```

Stage-8 training flags: `--gate-weighting switch`, `--z-loss COEF`, `--routing expert_choice`
(leaky), `--capacity-factor CF --capacity-mode sequence`.

## Current state

### Roadmap stages done (PLAN.md)

1 ✅ tests + review fixes behind `legacy` · 2 ✅ expand engine + supervisor + multi-block model +
physics sandbox · 3 ✅ causal mask + positional encodings · 4 ✅ vectorized dispatch + bench ·
5 ✅ `ArtemisModel` · 6 ✅ byte-level CPU training · 7 ✅ alpha × CF sweep with causal-safe capacity ·
8 ✅ Switch weighting, z-loss, expert-choice routing. Also done: repo checks (5 checks), physics
pilot, frequency/capacity study, byte-LM corpus benchmark.
Not done: **8b** (physics sandbox follow-ups) and **9(a)–(e)** (Fourier features, web byte
streaming, Continuous-Language-Field prototype, hypernetwork, serving — 9(e) deferred).
The SaaS layer is deferred until stage 9c has held-out bpb results.

### Pull requests in the original repo

Open PRs do not carry over to a copied repo; they stay in the original. All are open except #4
(merged into the `stage-physics-frequency-study` branch, not into `main`).

Stacking order (by git ancestry): **#1 → #2 → #3 (+#4) → #6 → #7 → #8 → #9 → #10 → #11 → #12**.
#5 sits directly on `main`. Several PRs declare `main` as their base on GitHub (#7, #9, #10, #12),
so their diffs include all earlier unmerged stages; merge in stack order.

| PR | Branch | GitHub base | Adds |
|---|---|---|---|
| [#6](https://github.com/lpittman-blip/artemis-567885/pull/6) | `repo-checks` | `stage-physics-frequency-study` | Five pre-integration checks (historical spec, checkpoint reload, MoE loop vs vectorized dispatch equivalence, FD solver + PINN held-out error, causal attention + MoE leakage) with raw consoles; causal mask. |
| [#7](https://github.com/lpittman-blip/artemis-567885/pull/7) | `stage-3-positional` | `main` (stacked on #6) | `positional.py` (none / sinusoidal / learned), `pos_encoding=` on `DynamicMultiBlockTransformer`, causal + permutation tests. |
| [#8](https://github.com/lpittman-blip/artemis-567885/pull/8) | `stage-4-dispatch-bench` | `stage-3-positional` | `bench.py` loop vs vectorized dispatch (parity first, median/p90, 1 and 8 threads) + results. |
| [#9](https://github.com/lpittman-blip/artemis-567885/pull/9) | `stage-5-artemis-model` | `main` (stacked on #8) | `ArtemisModel` byte-level MoE LM, shape/grad/causal tests. |
| [#10](https://github.com/lpittman-blip/artemis-567885/pull/10) | `stage-6-byte-training` | `main` (stacked on #9) | `train_bytes.py` CPU byte training on Gutenberg, held-out bpb vs unigram, sampling, measured 900 s run. |
| [#11](https://github.com/lpittman-blip/artemis-567885/pull/11) | `stage-7-alpha-cf-sweep` | `stage-6-byte-training` | `capacity_mode="sequence"`, drop-rate / load-entropy metrics, 20-run alpha × CF sweep. |
| [#12](https://github.com/lpittman-blip/artemis-567885/pull/12) | `stage-8-routing` | `main` (stacked on #11) | Switch gate weighting, router z-loss, expert-choice routing (leaky), 8-run comparison; PROGRESS iteration 6. |

Earlier PRs, also open: [#1](https://github.com/lpittman-blip/artemis-567885/pull/1) stage 1,
[#2](https://github.com/lpittman-blip/artemis-567885/pull/2) stage 2,
[#3](https://github.com/lpittman-blip/artemis-567885/pull/3) frequency study,
[#5](https://github.com/lpittman-blip/artemis-567885/pull/5) byte-LM corpus results (seeds 42/1/2).

### `expand-loop` branch

Branched from `repo-checks`. Latest commit `80ffa76` (2026-10-24 20:54 ET), "expand: cycles
50-59 via supervisor (measured; no rollbacks; num_layers cap 8 reached)". From `config.json` and
`runs/expand_log.jsonl` on that branch:

- 60 cycles logged (0–59); next cycle 60. No failures or rollbacks (PROGRESS.md).
- 8 layers; experts per layer **[6, 9, 5, 8, 5, 5, 5, 6]** (49 total); top-2; d_model 16.
- capacity_factor **1.85**; alpha 0.1.
- Best eval loss **0.0255** (0.025525 at cycle 56); last cycle 0.026348.
- Caps: `num_layers_max` **8 (reached)**, `capacity_factor_max` **2.0**, `num_experts_max` 32,
  `alpha_max` 1.0. With the layer cap reached, the engine can only add experts per layer.

A scheduled loop on the original repo's own clone keeps extending `expand-loop`; this copy is a
snapshot as of the commit above.

### Key measured results

- **Stage 6** (PROGRESS.md iter. 4, `runs/stage6/gutenberg_cpu_900s/`): 5,399,296 params,
  2593 steps in 900.9 s on CPU. Held-out Austen (author unseen in training) **1.747 bpb** on the
  256 eval windows vs unigram **4.580**; full held-out pass 1.7605 vs 4.5814; test split (Twain)
  1.9746 vs 4.7076. Still falling at the end of the run.
- **Stage 7** (`docs/STAGE7_SWEEP.md`, `runs/stage7/`): 1,058,304 params, 2000 steps, seed
  567885, alpha ∈ {0, 0.01, 0.1, 0.3} × CF ∈ {0.8, 1.0, 1.25, 1.5, dropless}, all causal-safe.
  Best alpha 0.3 / CF 1.5 **1.9809** bpb; best dropless alpha 0.1 1.9849. CF dominates: mean bpb
  2.140 (CF 0.8, ~19.6% drops) / 2.023 (1.0) / 1.991 (1.25) / 1.986 (1.5) / 1.990 (dropless).
  Alpha moves bpb ≤ 0.005 on average but lowers drops and evens load. Single seed.
- **Stage 8** (`docs/STAGE8_ROUTING.md`, `runs/stage8/`): same setup, dropless, alpha 0.01.
  Top-2 baseline **1.9866**; top-2 + z-loss 1e-3 **1.9767** (−0.010, **single seed, within
  noise**); top-1 renorm 2.1087; top-1 Switch 1.9936 (gap to top-2 0.122 → 0.007); top-2 Switch
  1.9845; z 1e-2 1.9932; top-1 Switch + z 1e-3 1.9977; expert choice 1.9849 (LEAKY).

### Known caveats

- **Expert-choice routing is leaky** (not causal): a future byte can change which past tokens an
  expert picks. It is labelled (`causal_safe=False`, `leaky: true`), not restricted, and its
  result is not a valid byte-LM number.
- `capacity_mode="global"` (library default) also leaks future bytes across the flattened batch
  (check 5); use dropless or `"sequence"` for the LM.
- **Sandbox scores**: `sandbox.py` fixed Total Loss **0.3415**, legacy **1.5083** (re-measured
  in this copy). The two modes use different targets (fixed regresses sin(X); legacy keeps the
  original zeros target), so the numbers are not comparable to each other.
- All stage 7/8 comparisons are single-seed; ~0.01 bpb differences are not significant.
- `checks/test_check1_historical.py` has shown an occasional non-reproducing failure (bit-exact
  float compare), per PROGRESS.md iterations 4 and 6. It passed in this copy.
- `docs/CONTINUOUS_LANGUAGE_FIELD.md` is truncated before its code section.

## Open decisions

- **Expand-engine layer cap: keep 8 or raise to 12.** The cap is 8 in `expand.py` and
  `config.json`, and it has been reached. The 12 option comes from the project conversation, not
  from any repo file. `model.py` itself does not enforce a layer limit; the cap lives only in
  `expand.py` defaults and `config.json` (PLAN.md and README.md also describe it as ≤ 8).
- **GPU training run is waiting on Brev billing.** Not recorded in the repo. Per the project
  conversation, the approved Brev NVIDIA L4 instance cannot be created until a payment card is
  connected on Brev. Everything in this repo so far was measured on CPU (no cloud charges).

## Suggested rebuild starting points

1. **Start from `stage-8-routing`**, not `main`. It has every stage plus the checks, with 274
   tests passing. Either merge the stack into `main` here in order (#1 → … → #12 equivalents), or
   make `stage-8-routing` the new `main` for the rebuild.
2. **Keep `legacy=True` and the test suite as the safety net.** Re-run
   `pytest -q tests checks` and `sandbox.py --mode legacy` (1.5083) after every change.
3. **Fold the stage-8 findings into the defaults** once they hold up over 2–3 seeds: Switch
   weighting (needed for top-1) and z-loss 1e-3 are the candidates. Use causal-safe capacity
   (`sequence`) or dropless routing; don't use expert choice for the LM unless a causal variant is
   built.
4. **Scale the stage-6 model** (it was still improving at 900 s) once a GPU is available; the
   training loop is CPU-oriented today.
5. **Then stages 8b and 9(a)–(d)** per PLAN.md, and the control experiment from the Codex review
   (fixed architecture vs auto-growing expand run at matched compute) before claiming growth helps.

## What did not carry over

- Pull requests (#1–#12), their review comments and discussions stay in the original repo.
- GitHub release `artemis-100m-step5250-preview` (on the original repo) and its assets
  (`artemis-100m-step5250-bf16.pt`, `tokenizer.json`, `SHA256SUMS`). The **git tag** was copied;
  the release object and its files were not. Per the release notes, that checkpoint comes from a
  separate Kaggle training run, not from code in this repo.
- Git-ignored files: checkpoints (`runs/*.pt`), `.backup_checkpoint/`, and the Gutenberg corpus.
- Repo settings, secrets, collaborators and Actions configuration (none copied).

## The original is untouched

`lpittman-blip/artemis-567885` was only read (clone and PR reads). Nothing was pushed, no branch
was changed, no PR was edited or merged, and the running expansion loop's clone on the box was
not touched.
