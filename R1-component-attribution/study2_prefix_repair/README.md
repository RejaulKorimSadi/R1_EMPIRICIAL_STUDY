# Study 2: repairing the first error

Three instruction-tuned models (GPT-OSS-120B, Qwen3.6-27B, ALLaM-2-7B), served through the Groq API at temperature 0.

| File | Description |
|---|---|
| `data/questions.json` | 200 SimpleQA questions, simple random sample with seed 20260819 |
| `data/baseline_*.json` | Raw baseline answers, one file per model (600 in total) |
| `data/qualifying_items.json` | 128 candidate items: divergence offset, correct and wrong replacements, status (`authored` or `dropped_noncommit`) |
| `data/qualifying_items_review.csv` | The same items in spreadsheet form |
| `data/annotator_A.csv`, `data/annotator_B.csv` | Independent scores for all 444 rows (111 items x 4 conditions) |
| `data/kappa_results.json` | Inter-annotator agreement |
| `code/build_pool.py` | Draws the question sample |
| `code/run_conditions.py` | Queries the models for the baseline and the four conditions |
| `code/cohens_kappa.py` | Agreement for a scored sheet |
| `code/analyze_study2.py` | Corrected analysis reported in the paper (Table 2, Appendix B) |
| `code/analyze_study2_original.py` | First analysis, which keyed items by question only (Section 5.1, deviation 5) |

Conditions: `a_baseline` (original answer), `b_substituted` (gold answer inserted at the divergence point), `c_isolated` (question asked again with no prior answer), `d_control` (a different wrong answer inserted).

Score columns: `n_downstream_failing` (failing claims after the divergence point), `isolated_query_correct` (condition C), `spontaneous_correction` (condition A), `abstain` (conditions B and D).

Reproduce the reported numbers:

```
cd study2_prefix_repair
python code/analyze_study2.py                          # annotator A, main text
python code/analyze_study2.py data/annotator_B.csv     # annotator B, Appendix B
python code/analyze_study2_original.py                 # first analysis, 90 items
```

Regenerate model outputs (requires an OpenAI-compatible endpoint):

```
export LLM_ENDPOINT="https://api.groq.com/openai/v1"
export LLM_API_KEY="..."
python code/build_pool.py --n 200 --seed 20260819 --out data/questions.json
python code/run_conditions.py --stage screen --models MODEL_NAME --pool data/questions.json --n 200 --out screened.json
python code/run_conditions.py --stage generate --items data/qualifying_items.json --out responses.csv
```

Qwen3.6-27B was run with `--max-tokens 2048`.
