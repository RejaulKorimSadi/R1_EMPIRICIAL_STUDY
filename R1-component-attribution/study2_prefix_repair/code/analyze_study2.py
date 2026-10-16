"""Study 2: corrected analysis of the prefix-substitution experiment.

Items are keyed by (item_id, model); 17 questions produced qualifying errors
from two models, so keying by item_id alone merges distinct items.
Refusals under conditions B and D are treated as missing.

Usage: python analyze_study2.py [scored_csv]   (default: data/annotator_A.csv)
"""
import csv
import collections
import statistics
import sys

from scipy.stats import wilcoxon
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportion_confint, proportions_ztest

path = sys.argv[1] if len(sys.argv) > 1 else "data/annotator_A.csv"
rows = list(csv.DictReader(open(path, encoding="utf-8")))
items = collections.defaultdict(dict)
for r in rows:
    items[(r["item_id"], r["model"])][r["condition"]] = r


def refused(r):
    return r["abstain"].strip() in ("1", "1.0")


def n_fail(r):
    return float(r["n_downstream_failing"])


def has_error(r):
    if r["condition"] in ("b_substituted", "d_control") and refused(r):
        return None
    return int(n_fail(r) > 0)


def paired(pairs):
    t = collections.Counter(pairs)
    table = [[t[(1, 1)], t[(1, 0)]], [t[(0, 1)], t[(0, 0)]]]
    return t, sum(t.values()), mcnemar(table, exact=True).pvalue


def h1(subset):
    pairs = [(has_error(v["a_baseline"]), has_error(v["b_substituted"])) for v in subset]
    t, n, p = paired([x for x in pairs if x[1] is not None])
    return n, (t[(1, 1)] + t[(1, 0)]) / n, (t[(1, 1)] + t[(0, 1)]) / n, t[(1, 0)], t[(0, 1)], p


print(f"items: {len(items)}  file: {path}")
by_model = collections.Counter(k[1] for k in items)
print("items per model:", dict(by_model))

n, before, after, fixed, broken, p_h1 = h1(items.values())
print(f"H1: n={n}  error rate {before:.1%} -> {after:.1%}  discordant {fixed} vs {broken}  exact McNemar p={p_h1:.2e}")
for m in sorted(by_model):
    n_m, b_m, a_m, f_m, br_m, p_m = h1([v for k, v in items.items() if k[1] == m])
    print(f"   {m}: n={n_m}  {b_m:.1%} -> {a_m:.1%}  discordant {f_m} vs {br_m}  p={p_m:.2g}")

a_cnt, b_cnt = zip(*[(n_fail(v["a_baseline"]), n_fail(v["b_substituted"]))
                     for v in items.values() if not refused(v["b_substituted"])])
print(f"   failing claims per answer: {statistics.mean(a_cnt):.2f} -> {statistics.mean(b_cnt):.2f}  "
      f"Wilcoxon p={wilcoxon(a_cnt, b_cnt).pvalue:.1e}")

succeeded = [v for v in items.values()
             if has_error(v["a_baseline"]) == 1 and has_error(v["b_substituted"]) == 0]
known = sum(1 for v in succeeded if float(v["c_isolated"]["isolated_query_correct"] or 0) == 1)
lo, hi = proportion_confint(known, len(succeeded), method="wilson")
print(f"H2: correct in isolation {known}/{len(succeeded)} = {known / len(succeeded):.1%}  Wilson 95% CI [{lo:.1%}, {hi:.1%}]")
total_known = sum(1 for v in items.values() if float(v["c_isolated"]["isolated_query_correct"] or 0) == 1)
print(f"    correct in isolation, all items: {total_known}/{len(items)}")

bd = [(has_error(v["b_substituted"]), has_error(v["d_control"])) for v in items.values()]
bd = [x for x in bd if None not in x]
t, n, p_h3 = paired(bd)
b_free, d_free = (t[(0, 0)] + t[(0, 1)]), (t[(0, 0)] + t[(1, 0)])
z, p_z = proportions_ztest([b_free, d_free], [n, n])
print(f"H3: n={n}  error-free B {b_free / n:.1%} vs D {d_free / n:.1%}  "
      f"B better {t[(0, 1)]} / D better {t[(1, 0)]}  exact McNemar p={p_h3:.2f}  (two-proportion z={z:.2f}, p={p_z:.2f})")
b_c, d_c = zip(*[(n_fail(v["b_substituted"]), n_fail(v["d_control"])) for v in items.values()
                 if not refused(v["b_substituted"]) and not refused(v["d_control"])])
print(f"    failing claims per answer: B {statistics.mean(b_c):.2f} vs D {statistics.mean(d_c):.2f}  "
      f"Wilcoxon p={wilcoxon(b_c, d_c).pvalue:.2f}")

q = multipletests([p_h1, p_h3], method="fdr_bh")[1]
print(f"Benjamini-Hochberg q: H1={q[0]:.1e}  H3={q[1]:.2f}")

ref = collections.Counter((refused(v["b_substituted"]), refused(v["d_control"])) for v in items.values())
p_ref = mcnemar([[ref[(True, True)], ref[(True, False)]], [ref[(False, True)], ref[(False, False)]]], exact=True).pvalue
print(f"refusals: B={ref[(True, True)] + ref[(True, False)]}  D={ref[(True, True)] + ref[(False, True)]}  "
      f"discordant {ref[(True, False)]} vs {ref[(False, True)]}  exact McNemar p={p_ref:.3f}")
print("self-correction in baseline:", sum(1 for v in items.values()
                                         if float(v["a_baseline"]["spontaneous_correction"] or 0) == 1))

lengths = collections.defaultdict(list)
for v in items.values():
    a = v["a_baseline"]
    lengths["A"].append(len(a["response_text"][int(float(a["divergence_char"] or 0)):].split()))
    lengths["B"].append(len(v["b_substituted"]["response_text"].split()))
    lengths["D"].append(len(v["d_control"]["response_text"].split()))
print("median words after divergence:", {k: statistics.median(x) for k, x in lengths.items()})
