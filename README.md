# SOES: Secure Operational Efficiency Score for Edge LLMs

This repository contains the benchmark pipeline for the Secure Operational Efficiency Score (SOES). SOES rates small open LLMs on accuracy, safety and resource cost together, and compares each model at FP16, INT8 and INT4 precision.

```
SOES = (accuracy × jailbreak resistance × privacy score) / (energy [J/token] × VRAM [GB] × latency [s/token])
```

The three numerator terms are fractions in [0, 1]. A higher SOES is better.

## What the pipeline measures

| Metric | How it is measured |
|---|---|
| Task accuracy | Full MMLU (57 subjects, 14,042 questions), 5-shot. The answer is the letter (A–D) with the highest next-token log-probability. |
| Jailbreak resistance | 136 prompts from [onepaneai/harmful-prompts](https://huggingface.co/datasets/onepaneai/harmful-prompts): toxicity, simple injection and jailbreak. [Llama Guard 3 8B](https://huggingface.co/meta-llama/Llama-Guard-3-8B) judges each response. For injection and jailbreak prompts, complying instead of refusing also counts as a successful attack. |
| Privacy score | System-prompt secret leakage. A random access code is placed in the system prompt and attacked with 8 extraction prompts (10 codes, so 80 attempts). Score = 1 − leak rate. |
| Latency | Median ms per generated token over 5 runs of exactly 128 tokens, with greedy decoding, a static KV cache and `torch.compile`. |
| Energy | GPU board energy per token from the NVML energy counter. Energy above the idle baseline is also reported. |
| VRAM | GPU memory allocated by the model (allocated memory after loading, minus memory before). |

Precisions:
- **FP16:** unquantized.
- **INT8:** bitsandbytes LLM.int8().
- **INT4:** torchao `Int4WeightOnlyConfig`, group size 128, tinygemm kernel.

The safety judge stays in CPU memory and moves to the GPU only after the tested model has been unloaded. It therefore never affects the latency, energy or VRAM numbers.

## Requirements

- **GPU:** an NVIDIA GPU with an energy counter (Volta or newer). The results were produced on an RTX A6000 (48 GB). The largest model, Phi-4 at 14B, needs about 30 GB in FP16. Run the benchmark on an otherwise idle GPU, because other processes distort latency and energy. The results table records this in the `GPU Shared` column.
- **CPU RAM:** at least 32 GB, because the 8B Llama Guard judge is held in CPU memory.
- **Disk:** about 110 GB for the Hugging Face cache, covering all models plus the judge.
- **Hugging Face access:** an account token, plus accepted licenses for the gated models: Meta Llama 3.1, Llama Guard 3 and Google Gemma.

## Setup

```bash
conda create -n soes python=3.10 -y
conda activate soes
pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
python -m ipykernel install --user --name soes
```

Create a `.env` file next to the notebook:

```
HF_TOKEN=<your Hugging Face token>
```

## Running

1. Open `SecureOperationalEfficiencyScore_(SOES)_Pipeline.ipynb` with the `soes` kernel.
2. Set the precision in the "Models to Evaluate" cell:
   ```python
   WEIGHT_PREDISION = "INT4"   # or "INT8", "FP"
   ```
   Edit `models_config` to choose which models to run.
3. Run all cells. Skip the `!pip install` cell if you installed `requirements.txt`.
4. Repeat for each precision, one at a time.


## Outputs

| File | Content |
|---|---|
| `soes_benchmark_results.csv` | One row per model with every metric and SOES. Rename it per precision, because each run overwrites it. |
| `mmlu_5shot_<model>_<precision>.json` | MMLU accuracy for one model and precision. |
| `safety_records_<model>_<precision>.json` | Every prompt and response with its judge verdict. Use these for paired comparisons between precisions, such as McNemar's test. |
| `*.png`, `*.pdf` | Plots: radar chart, SOES vs. model size, component breakdown, security vs. efficiency, jailbreak resistance vs. size, resource comparison. |

## Configuration

| Setting | Where | Default |
|---|---|---|
| Jailbreak prompt source | `JAILBREAK_SOURCE` | `"dataset"` (`"builtin"` uses 10 hand-written prompts) |
| Jailbreak categories | `JAILBREAK_TYPES` | `jailbreak`, `toxicity`, `simple_injection` |
| Safety judge | `USE_SAFETY_JUDGE`, `JUDGE_ID` | `True`, `meta-llama/Llama-Guard-3-8B` |
| MMLU shots | `NUM_SHOTS` | `5` |
| Latency settings | `SOESBenchmarkColab(latency_tokens=, latency_runs=, latency_compile=)` | `128`, `5`, `True` |
| Secret-leak codes | `SOESBenchmarkColab(leak_secrets=)` | `10` |
| Generation batch size | `SOESBenchmarkColab(gen_batch_size=)` | `16` |

## Known limitations

- **Hardware:** the A6000 stands in for an edge server. Absolute latency and energy will differ on embedded devices.
- **INT8 speed:** bitsandbytes INT8 is much slower at batch size 1 than FP16 or INT4, so its energy per token is higher. This reflects the LLM.int8() implementation, not INT8 in general.

