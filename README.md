# ATML PA3

Code for the three PA3 experiments (EE-5102/CS-6304, Fall 2026):

1. Fine-tuning Qwen3-0.6B on MetaMathQA with full FT, L2-SP, LoRA and OPLoRA
2. A SigLIP + Qwen3-0.6B vision-language model and visual grounding tests
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
| Paired tests (bootstrap + McNemar), BoolQ yes-rate | `python -m task1.significance --lm_eval_dir RESULTS/lm_eval --out RESULTS` |
| Weight / activation diagnostics | `python -m task1.diagnostics --runs_dir RUNS --out RESULTS/diagnostics` |
| Qualitative responses | `python -m task1.qualitative --eval_dir EVAL --out RESULTS/qualitative` |
| Learning curves | `python -m task1.plot_curves --runs_dir RUNS --out RESULTS/figures` |

`notebooks/task1_colab.ipynb` runs all of the above on Colab with outputs on Google Drive.

Run configs are in `task1/configs/`. Training resumes from the latest checkpoint if the same command is run again.

LoRA and OPLoRA are implemented in `common/lora.py` (not peft), so that LoRA-16 and OPLoRA-16 start from identical factors and the projected OPLoRA factors can be exported directly. `tests/test_lora.py` checks the projection constraints and the merge.

## Task 2

| Step | Command |
| --- | --- |
| Select splits (COCO 2014 captions/instances + VQAv2 complementary pairs, seed 6304) | `python -m task2.prepare_data --raw RAW --out splits/task2` |
| Download the selected images | `python -m task2.download_images --out IMAGES` |
| Cache SigLIP patch features (fp16, 196 x 768 per image) | `python -m task2.cache_features --images IMAGES --out FEATS` |
| Interface checks (shapes, frozen sets, gradients, split isolation) | `python -m task2.checks --features FEATS --out RESULTS/checks.json` |
| Train all three stages (resumable) | `python -m task2.train --config task2/configs/vlm196.yaml --features FEATS --out_root RUNS` |
| Evaluate a stage under the four image conditions | `python -m task2.evaluate --run_dir RUNS/vlm196 --stage 3 --features FEATS --out RESULTS/vlm196/test_stage3` |
| Tables, bootstrap intervals, curves, examples | `python -m task2.analyze --res RESULTS/vlm196 --images IMAGES` |
| 49 vs 196 tokens on the same test pairs | `python -m task2.compare_runs --a RESULTS/vlm196 --b RESULTS/vlm49 --out RESULTS/comparison_49_vs_196.csv` |

`notebooks/task2_colab.ipynb` runs these steps on Colab. `splits/task2/` holds every selected image, caption, question and pair id, the generated grounding pairs and the fixed mismatched-image mapping; `selection_info.json` lists the filtering rules and counts. `task2/configs/vlm49.yaml` is the optional 49-token run.

`tests/test_vlm.py` checks the prompt layout, the 2x2 pooling, that gradients reach the connector while Qwen stays frozen, and that the cached greedy decoding loop matches `generate` with and without left padding.

## Results

`results/task1/` and `results/task2/` hold the outputs of the runs reported in the paper (tables, logs, per-question scores, figures).
The raw lm-eval sample logs are not committed because of their size; `results/task1/per_question.csv` has the per-question correctness for every condition, which is all the significance tests use.

## External code and data

- `data/first_plot_questions.yaml` is taken unchanged from the Emergent Misalignment repository (https://github.com/emergent-misalignment/emergent-misalignment).
- The HellaSwag context preprocessing in `task1/probe_set.py` is copied from lm-evaluation-harness (`lm_eval/tasks/hellaswag/utils.py`) so the probe prompts match the benchmark.
- Benchmarks are run with lm-evaluation-harness 0.4.13.
- The VQA answer normalization in `task2/vqa_norm.py` (contractions, number words, articles, punctuation rules) is adapted from the official VQA evaluation script (https://github.com/GT-Vision-Lab/VQA, `vqaEval.py`).
- Task 2 uses COCO 2014 captions and instance annotations, VQAv2 questions/annotations/complementary pairs, and the `google/siglip-base-patch16-224` encoder.
