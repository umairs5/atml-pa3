"""Builds the report's LaTeX tables from the saved result files and copies the figures.

python report/make_tables.py
"""
import json
import os
import shutil

import pandas as pd

R, G, F = "results", "report/gen", "report/figures"
os.makedirs(G, exist_ok=True)
os.makedirs(F, exist_ok=True)
PEND = r"\pend{--}"
NAMES = {"baseline": "Baseline", "full_mid": "Full FT (mid)", "full_final": "Full FT", "l2sp_final": "L2-SP",
         "lora4_final": "LoRA-4", "lora16_final": "LoRA-16", "oplora16_final": "OPLoRA-16"}
ORDER = list(NAMES)


def write(name, text):
    with open(os.path.join(G, name), "w", encoding="utf-8") as f:
        f.write(text)


def load(path):
    with open(os.path.join(R, path), encoding="utf-8") as f:
        return json.load(f)


def signed(x, d=2):
    return f"{x:+.{d}f}"


def t1_benchmark():
    t = pd.read_csv(f"{R}/task1/benchmark_table.csv").set_index("condition")
    cols = ["gsm8k_flex", "hellaswag_acc_norm", "boolq_acc", "arc_easy_acc_norm"]
    rows = []
    for c in ORDER:
        cells = [f"{t.loc[c, k]:.2f}" if c == "baseline" else f"{t.loc[c, k]:.2f} ({signed(t.loc[c, k + '_delta_pp'])})" for k in cols]
        rows.append(f"{NAMES[c]} & " + " & ".join(cells) + r" \\")
    write("t1_benchmark.tex", "\n".join(rows))


def t1_tests():
    s = pd.read_csv(f"{R}/task1/significance.csv")
    keep = [("l2sp_final", "full_final"), ("lora16_final", "lora4_final"), ("oplora16_final", "lora16_final"), ("lora16_final", "full_final")]
    label = {("l2sp_final", "full_final"): "L2-SP $-$ Full FT", ("lora16_final", "lora4_final"): "LoRA-16 $-$ LoRA-4",
             ("oplora16_final", "lora16_final"): "OPLoRA-16 $-$ LoRA-16", ("lora16_final", "full_final"): "LoRA-16 $-$ Full FT"}
    rows = []
    for k in keep:
        cells = []
        for task in ["gsm8k", "hellaswag", "boolq", "arc_easy"]:
            r = s[(s.task == task) & (s.condition == k[0]) & (s.reference == k[1])].iloc[0]
            mark = "$^{*}$" if r.holm_p < 0.05 else ""
            cells.append(f"{signed(r.diff_pp, 1)} [{signed(r.ci_low_pp, 1)}, {signed(r.ci_high_pp, 1)}]{mark}")
        rows.append(label[k] + " & " + " & ".join(cells) + r" \\")
    write("t1_tests.tex", "\n".join(rows))


def t1_resources():
    sizes = pd.read_csv(f"{R}/task1/checkpoint_sizes.csv").set_index("run")
    names = {"full_ft": "Full FT", "full_ft_l2sp": "L2-SP", "lora_r4": "LoRA-4", "lora_r16": "LoRA-16", "oplora_r16": "OPLoRA-16"}
    rows = []
    for run, name in names.items():
        p = f"{R}/task1/run_info/{run}/resources.json"
        if os.path.exists(p):
            with open(p) as f:
                r = json.load(f)
            cells = [f"{r['trainable_params']/1e6:.2f}", f"{r['train_seconds']/60:.0f}", f"{r['peak_mem_allocated_gb']:.2f}"]
        else:
            cells = [PEND, PEND, PEND]
        rows.append(f"{name} & " + " & ".join(cells) + f" & {sizes.loc[run, 'final_checkpoint']} " + r"\\")
    write("t1_resources.tex", "\n".join(rows))


def t1_diag():
    w = pd.read_csv(f"{R}/task1/diagnostics/weight_diagnostics.csv")
    a = pd.read_csv(f"{R}/task1/diagnostics/activation_summary.csv")
    rows = []
    for c in ORDER[1:]:
        wc, ac = w[w.condition == c].set_index("layer"), a[a.condition == c].set_index("layer")
        cells = [f"{wc.loc[l, 'rho_k']:.3f}" for l in [0, 13, 27]] + [f"{wc.loc[l, 'eta']:.4f}" for l in [0, 13, 27]]
        cells += [f"{ac.loc[l, 'mean_math'] / ac.loc[l, 'mean_non_math']:.1f}" for l in [0, 13, 27]]
        rows.append(f"{NAMES[c]} & " + " & ".join(cells) + r" \\")
    write("t1_diag.tex", "\n".join(rows))


def t1_annotation():
    s = pd.read_csv(f"{R}/task1/qualitative/annotation_summary.csv").set_index("condition")
    rows = []
    for c in ORDER:
        r = s.loc[c]
        rows.append(f"{NAMES[c]} & {int(r.format_ok)} & {int(r.coherent)}+{int(r.partly_coherent)} & {int(r.repetition)} & {int(r.math_framing)} & {int(r.inappropriate)} & {int(r.truncated)} " + r"\\")
    write("t1_annotation.tex", "\n".join(rows))


def t2_final():
    a = load("task2/vlm196/analysis/final_results.json")
    b = load("task2/vlm49/analysis/final_results.json")
    names = {"correct": "Correct", "complementary": "Complementary", "mismatched": "Mismatched", "none": "No image"}
    rows = []
    for suite, title in [("grounding_test", "Grounding (64 pairs)"), ("vqa_test", "VQAv2 (96 pairs)")]:
        rows.append(rf"\multicolumn{{7}}{{l}}{{\emph{{{title}}}}} \\")
        for c, name in names.items():
            r, s = a[suite]["all"][c], b[suite]["all"][c]
            acc = r["accuracy"]
            rows.append(f"{name} & {100*acc[0]:.1f} [{100*acc[1]:.1f}, {100*acc[2]:.1f}] & {100*r['pair_correct'][0]:.1f} & {100*r['identical'][0]:.1f} & "
                        f"{100*r['follows_partner'][0]:.1f} & {r['nll'][0]:.2f} & {100*s['accuracy'][0]:.1f} " + r"\\")
        adv, adv49 = a[suite]["all"]["advantage_vs_complementary"], b[suite]["all"]["advantage_vs_complementary"]
        rows.append(rf"Correct $-$ compl. & {signed(100*adv['accuracy'][0],1)} [{signed(100*adv['accuracy'][1],1)}, {signed(100*adv['accuracy'][2],1)}] & & & & "
                    f"{signed(adv['nll'][0])} & {signed(100*adv49['accuracy'][0],1)} " + r"\\")
    write("t2_final.tex", "\n".join(rows))


def t2_types():
    a = load("task2/vlm196/analysis/final_results.json")
    label = {"presence": "Presence", "counting": "Counting", "yes/no": "Yes/no", "number": "Number", "other": "Other"}
    rows = []
    for suite, types in [("grounding_test", ["presence", "counting"]), ("vqa_test", ["yes/no", "number", "other"])]:
        for k in types:
            d = a[suite][k]
            cells = [f"{100*d[c]['accuracy'][0]:.1f}" for c in ["correct", "complementary", "mismatched", "none"]]
            adv = d["advantage_vs_complementary"]["accuracy"]
            rows.append(f"{label[k]} & {d['n_pairs']} & " + " & ".join(cells) + f" & {signed(100*adv[0],1)} [{signed(100*adv[1],1)}, {signed(100*adv[2],1)}] " + r"\\")
    write("t2_types.tex", "\n".join(rows))


def t3_methods():
    show = [("first", "FirstAns"), ("self_consistency", "Self-consistency"), ("top_logp", "Top log-prob"), ("top_certainty", "Top self-certainty"),
            ("rwc_certainty", "Self-certainty + RWC"), ("shortest", "Shortest (control)"), ("top_vision", "Top vision"), ("rwc_vision", "Vision + RWC"),
            ("top_visconf", "Top VisConf"), ("rwc_visconf", "VisConf + RWC"), ("top_tokenwise", "Top tokenwise"), ("rwc_tokenwise", "Tokenwise + RWC")]
    cur = {d: pd.read_csv(f"{R}/task3/{d}/method_curves.csv").query("split == 'eval'") for d in ["math", "mathvista"]}
    cell = lambda d, m, n: "--" if m not in set(cur[d].method) else f"{100*cur[d][(cur[d].method == m) & (cur[d].n == n)].accuracy.iloc[0]:.1f}"
    rows = [f"{label} & " + " & ".join(cell("math", m, n) for n in [4, 8, 32]) + " & " + " & ".join(cell("mathvista", m, n) for n in [4, 8, 32]) + r" \\" for m, label in show]
    sc = lambda d, n, col: cur[d][(cur[d].method == "self_consistency") & (cur[d].n == n)][col].iloc[0]
    rows.append(r"\midrule")
    rows.append("Oracle coverage & " + " & ".join(f"{100*sc('math', n, 'covered'):.1f}" for n in [4, 8, 32]) + " & " + " & ".join(f"{100*sc('mathvista', n, 'covered'):.1f}" for n in [4, 8, 32]) + r" \\")
    rows.append("Distinct answers & " + " & ".join(f"{sc('math', n, 'n_groups'):.2f}" for n in [4, 8, 32]) + " & " + " & ".join(f"{sc('mathvista', n, 'n_groups'):.2f}" for n in [4, 8, 32]) + r" \\")
    rows.append("Tokens per question & " + " & ".join(f"{sc('math', n, 'cost'):,.0f}" for n in [4, 8, 32]) + " & " + " & ".join(f"{sc('mathvista', n, 'cost'):,.0f}" for n in [4, 8, 32]) + r" \\")
    write("t3_methods.tex", "\n".join(rows))


def t3_failures():
    rows = []
    for d, name, quality in [("math", "MATH", "rwc_certainty"), ("mathvista", "MathVista", "rwc_visconf")]:
        c = pd.read_csv(f"{R}/task3/{d}/method_curves.csv").query("split == 'eval'")
        for n in [1, 4, 32]:
            s = c[(c.n == n) & (c.method == "self_consistency")].iloc[0]
            q = c[(c.n == n) & (c.method == quality)].iloc[0]
            rows.append(f"{name} & {n} & {100*s.not_generated:.1f} & {100*s.missed:.1f} & {100*q.missed:.1f} " + r"\\")
    write("t3_failures.tex", "\n".join(rows))


def t3_signals():
    label = {"mean_logp": "Mean log-prob", "self_certainty": "Self-certainty", "shortness": "Shortness", "vision": "Vision share", "visconf": "VisConf", "tokenwise": "Tokenwise"}
    rows = []
    for d, name in [("math", "MATH"), ("mathvista", "MathVista")]:
        s = pd.read_csv(f"{R}/task3/{d}/signal_associations.csv")
        for _, r in s.iterrows():
            rows.append(f"{name} & {label[r.score]} & {r.spearman_with_length:+.2f} & {r.corr_with_correct:+.3f} & {signed(r.top_minus_bottom_pp,1)} [{signed(r.ci_low_pp,1)}, {signed(r.ci_high_pp,1)}] " + r"\\")
    write("t3_signals.tex", "\n".join(rows))


def t3_adaptive():
    rows = []
    for d, name in [("math", "MATH"), ("mathvista", "MathVista")]:
        a = load(f"task3/{d}/adaptive_results.json")
        fixed = {r["n"]: r for r in a["fixed_self_consistency"]}
        for n in [16, 32]:
            rows.append(f"{name} & Fixed $N={n}$ & -- & {fixed[n]['accuracy']:.1f} & {fixed[n]['mean_tokens']:,.0f} & {n} & -- " + r"\\")
        for fam, label in [("agreement", "Agreement"), ("quality", "Quality-aware")]:
            r, s = a["held_out"][fam], a["selected_on_dev"][fam]
            g = f", {s['gamma']:g}" if s["gamma"] is not None else ""
            dist = " / ".join(f"{100*r['stop_distribution'][str(k)]:.0f}" for k in [4, 8, 16, 32])
            rows.append(f"{name} & {label} & {s['tau']:g}{g} & {r['accuracy']:.1f} & {r['mean_tokens']:,.0f} & {r['mean_rollouts']:.1f} & {dist} " + r"\\")
    write("t3_adaptive.tex", "\n".join(rows))


def t3_live():
    p = f"{R}/task3/live/summary.csv"
    names = {"single_long": "Single long", "parallel": "Parallel + vote", "refinement": "Refinement", "initial_only": "Initial only"}
    rows = []
    if os.path.exists(p):
        s = pd.read_csv(p).set_index("strategy")
        for k, n in names.items():
            r = s.loc[k]
            rows.append(f"{n} & {r.accuracy:.0f} & {r.output_tokens:,.0f} & {r.input_tokens:,.0f} & {r.seconds:.1f} & {int(r.truncated_calls)}/{int(r.calls)} & {int(r.extraction_failures)}/{int(r.calls)} " + r"\\")
    else:
        rows = [f"{n} & " + " & ".join([PEND] * 6) + r" \\" for n in names.values()]
    write("t3_live.tex", "\n".join(rows))


def figures():
    for src, dst in [("task1/figures/task1_learning_curves.pdf", "task1_curves.pdf"), ("task2/vlm196/analysis/task2_learning_curves.pdf", "task2_curves.pdf"),
                     ("task2/vlm196/analysis/task2_examples.pdf", "task2_examples.pdf"), ("task3/task3_accuracy_curves.pdf", "task3_curves.pdf"),
                     ("task3/task3_adaptive.pdf", "task3_adaptive.pdf"), ("task3/task3_trajectories.pdf", "task3_trajectories.pdf")]:
        shutil.copy(os.path.join(R, src), os.path.join(F, dst))


if __name__ == "__main__":
    for fn in [t1_benchmark, t1_tests, t1_resources, t1_diag, t1_annotation, t2_final, t2_types, t3_methods, t3_failures, t3_signals, t3_adaptive, t3_live, figures]:
        fn()
    print("tables in", G)
