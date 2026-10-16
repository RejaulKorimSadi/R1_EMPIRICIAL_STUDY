# Study 1: removing attention at matched data and scale

Pythia (attention) and Mamba (selective state space) at 2.8B and 1.4B parameters, all trained on the Pile with the GPT-NeoX tokenizer. Probe 1a uses the 2.8B pair (`EleutherAI/pythia-2.8b`, `state-spaces/mamba-2.8b-hf`). Probe 1b uses both pairs and Qwen2.5-1.5B as a reference.

## 1a: free recall

| File | Description |
|---|---|
| `data/questions.json` | 200 SimpleQA questions (first 200 after shuffling the test set with seed 42) |
| `data/outputs_pythia_2_8b.json`, `data/outputs_mamba_2_8b.json` | Raw model outputs, greedy decoding, 64 new tokens, three-shot Q/A prompt |
| `data/scoresheet_blind.csv` | Shuffled, de-identified sheet given to annotators |
| `data/model_key.csv` | Response-to-model mapping, opened after scoring |
| `data/annotator_A.csv`, `data/annotator_B.csv` | Independent labels: 0 correct, 1 factual error, 2 abstention, 3 degenerate |
| `code/run_free_recall.py` | Runs a model over the questions and builds the blind sheet |
| `code/analyze_study1a.py` | Agreement, label counts and paired McNemar tests (Table 1) |

Reproduce the reported numbers:

```
cd study1_architecture
python code/analyze_study1a.py
```

Regenerate model outputs (one 8 GB GPU is sufficient):

```
python code/run_free_recall.py run --model EleutherAI/pythia-2.8b --pool data/questions.json --out data/outputs_pythia_2_8b.json
python code/run_free_recall.py run --model state-spaces/mamba-2.8b-hf --pool data/questions.json --out data/outputs_mamba_2_8b.json
python code/run_free_recall.py blind --inputs data/outputs_pythia_2_8b.json data/outputs_mamba_2_8b.json --out data/scoresheet_blind.csv --key data/model_key.csv
```

## 1b: context conflict

Files are in `study1b/`. See `study1b/README.md`.
