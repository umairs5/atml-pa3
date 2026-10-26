# ATML PA3

Code for the three PA3 experiments (EE-5102/CS-6304, Fall 2026):

1. Fine-tuning Qwen3-0.6B on MetaMathQA with full FT, L2-SP, LoRA and OPLoRA
2. A SigLIP + Qwen3 vision-language model and visual grounding tests
3. Test-time compute on the supplied Qwen2.5 rollouts

GPU work was run on Google Colab (T4). Task 3's offline analysis runs on CPU.

## Setup

```
pip install -r requirements.txt
```

All scripts are run from the repo root as modules (`python -m task1.train ...`).
Raw datasets and checkpoints are not in the repo; the scripts download what they need from the Hugging Face hub.

## Task 1

| Step | Command |
| --- | --- |
| Training data (first 40k eligible MetaMathQA examples) | `python -m task1.prepare_data --out DATA` |
| Probe set for activation diagnostics | `python -m task1.probe_set` (already saved in `splits/`) |
| Train one run | `python -m task1.train --config task1/configs/<run>.yaml --out_root RUNS --data DATA/metamath_40k.npz` |
| Build an eval model (merge adapters / fp16 export) | `python -m task1.merge --run_dir RUNS/<run> --update 1250 --out EVAL/<condition>` |
| Benchmarks | `python -m task1.evaluate --model EVAL/<condition> --name <condition> --out RESULTS/lm_eval` |
| Results table | `python -m task1.summarize_eval --lm_eval_dir RESULTS/lm_eval --out RESULTS` |
| Weight / activation diagnostics | `python -m task1.diagnostics --runs_dir RUNS --out RESULTS/diagnostics` |
| Qualitative responses | `python -m task1.qualitative --eval_dir EVAL --out RESULTS/qualitative` |
| Learning curves | `python -m task1.plot_curves --runs_dir RUNS --out RESULTS/figures` |

`notebooks/task1_colab.ipynb` runs all of the above on Colab with outputs on Google Drive.

Run configs are in `task1/configs/`. Training resumes from the latest checkpoint if the same command is run again.

LoRA and OPLoRA are implemented in `common/lora.py` (not peft), so that LoRA-16 and OPLoRA-16 start from identical factors and the projected OPLoRA factors can be exported directly. `tests/test_lora.py` checks the projection constraints and the merge.

## External code and data

- `data/first_plot_questions.yaml` is taken unchanged from the Emergent Misalignment repository (https://github.com/emergent-misalignment/emergent-misalignment).
- The HellaSwag context preprocessing in `task1/probe_set.py` is copied from lm-evaluation-harness (`lm_eval/tasks/hellaswag/utils.py`) so the probe prompts match the benchmark.
- Benchmarks are run with lm-evaluation-harness 0.4.13.
