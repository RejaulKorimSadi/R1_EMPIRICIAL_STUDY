# Companion study (anonymised)

Pre-registered study of five open-weight models under six prompt conditions and three identity-probe sets (5,620 responses, dual-scored). Section 6 of the paper cites two of its results: error rates on niche-domain versus general-domain questions, and the share of incorrect answers stated without hedging.

| Folder | Contents |
|---|---|
| `data/prompts/`, `data/probes/` | Prompt and probe sets |
| `data/raw_outputs/` | Raw model outputs |
| `data/scoring/` | Blind scoring sheets from two raters, the blinding key and the agreement report |
| `docs/` | Pre-registration, scoring rubric and prompt taxonomy |
| `code/` | Sampling, collection, agreement and analysis scripts |

Reproduce the cited numbers (lines `H3` and `H2` of the output):

```
cd companion_study
pip install -r requirements.txt
python code/run_analysis.py
python code/compute_kappa.py
```
