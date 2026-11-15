"""Paper figures from the frozen analysis JSONs (no model output is read).

F1 paper/figures/f1_accuracy_vs_tokens.pdf: METAR all-fields exact and MedCalc accuracy vs mean output tokens per item,
   every Part D-C/F system (results/analysis/partd_f.json: scores, cost_per_item).
F2 paper/figures/f2_library_vs_alternatives.pdf: 4B LoRA library minus each alternative on core/para/mixed with 95% CIs
   (results/analysis/partb_test_embed-lr.json, Part B frozen analysis).

usage: .venv-paper/bin/python scripts/paper/figures.py
Palette: dataviz reference categorical slots 1-3 (validated --pairs all, light); identity also carried by marker shape
and direct labels (relief rule for the aqua slot's sub-3:1 contrast).
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402,F401

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(ROOT, "paper", "figures")
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "pdf.fonttype": 42, "legend.frameon": False,
})


def load(name):
    with open(os.path.join(ROOT, "results", "analysis", name)) as f:
        return json.load(f)


def kind(system):
    if "+" in system and ("metar" in system or "medcalc" in system):
        return "adapter"
    return "thinking" if "thinking" in system else "prompted"


STYLE = {"adapter": (S1, "o", "4B/1.7B + LoRA adapter"), "prompted": (S2, "s", "untouched, no thinking"),
         "thinking": (S3, "^", "untouched, thinking")}
# (task, system) -> (label, x offset pt, y offset pt); unlisted points stay unlabelled (values are in Table 4)
LABEL = {
    ("metar", "4B+metar"): ("4B and 1.7B + adapter", 5, -2), ("metar", "32B thinking"): ("32B think", -8, 4),
    ("metar", "8B few-shot+thinking"): ("8B fs+think", 5, 2), ("metar", "8B thinking"): ("8B think", 5, -9),
    ("metar", "4B few-shot+thinking"): ("4B fs+think", 5, -3), ("metar", "8B few-shot"): ("8B fs", 5, 0),
    ("metar", "32B zero-shot"): ("32B", 5, -4), ("metar", "8B zero-shot"): ("8B", 5, -3),
    ("medcalc", "4B+medcalc"): ("4B+adapter", 5, -1), ("medcalc", "1.7B+medcalc"): ("1.7B+adapter", 5, -3),
    ("medcalc", "8B few-shot+thinking"): ("8B fs+think", 5, -1), ("medcalc", "4B few-shot+thinking"): ("4B fs+think", -6, 4),
    ("medcalc", "8B few-shot"): ("8B, 4B fs", -4, 5), ("medcalc", "32B thinking"): ("32B think", 5, -2),
    ("medcalc", "8B zero-shot"): ("8B", 5, -3), ("medcalc", "32B zero-shot"): ("32B", 5, -2),
}


def f1():
    f = load("partd_f.json")
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.7))
    for ax, task, title in zip(axes, ("metar", "medcalc"), ("METAR decoding (all 14 fields exact)", "MedCalc-Bench")):
        for sysname, sc in f["scores"].items():
            if task not in sc:
                continue
            if kind(sysname) == "adapter" and task not in sysname:
                continue  # an adapter outside its own task is not a system for that task
            c, m, _ = STYLE[kind(sysname)]
            x = f["cost_per_item"][sysname][task]["out_tokens"]
            y = sc[task]
            ax.scatter([x], [y], s=26, c=c, marker=m, edgecolors="white", linewidths=1.0, zorder=3)
            if (task, sysname) in LABEL:
                lab, dx, dy = LABEL[(task, sysname)]
                ax.annotate(lab, (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=6.5, color=INK2,
                            ha="right" if dx < 0 else "left")
        ax.set_xscale("log")
        ax.set_xlim(100, 6000)
        ax.set_xticks([100, 300, 1000, 3000])
        ax.set_xticklabels(["100", "300", "1,000", "3,000"])
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xlabel("mean output tokens per item (log)")
        ax.set_ylabel("accuracy (%)")
        ax.set_ylim(-3, 103)
        ax.set_title(title, fontsize=8, color=INK, loc="left")
        ax.grid(True, color=GRID, linewidth=0.5, zorder=0)
    handles = [plt.Line2D([], [], marker=m, color=c, linestyle="", markersize=5, label=lab)
               for c, m, lab in STYLE.values()]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=7, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(os.path.join(OUT, "f1_accuracy_vs_tokens.pdf"))
    plt.close(fig)


def f2():
    b = load("partb_test_embed-lr.json")
    want = {"base": "base", "pooled": "pooled", "distilled": "distill", "Qwen3-8B": "8B"}
    sets = [("core", S1, "o"), ("para", S2, "s"), ("mixed", S3, "^")]
    rows = list(want)
    fig, ax = plt.subplots(figsize=(4.6, 2.5))
    for j, (st, c, m) in enumerate(sets):
        for i, r in enumerate(rows):
            t = next(t for t in b["tests"] if t["test"].split(" [")[0].split()[:2] == ["4B", "lora"]
                     and t["test"].split(" [")[0].split()[3] == want[r] and t["test"].endswith(f"[{st}]"))
            y = i + (j - 1) * 0.22
            ax.plot([t["lo"], t["hi"]], [y, y], color=c, linewidth=1.6, solid_capstyle="round", zorder=2)
            ax.scatter([t["diff"]], [y], s=24, c=c, marker=m, edgecolors="white", linewidths=1.0, zorder=3,
                       label=st if i == 0 else None)
    ax.axvline(0, color=INK2, linewidth=0.7, zorder=1)
    ax.axvline(-2, color=INK2, linewidth=0.7, linestyle=(0, (3, 3)), zorder=1)
    ax.text(-2.1, -0.55, "non-inferiority bar (-2)", fontsize=6.5, color=INK2, ha="right", va="bottom")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"vs {r}" for r in rows])
    ax.invert_yaxis()
    ax.set_xlabel("4B library minus alternative (points, 95% CI)")
    ax.grid(True, axis="x", color=GRID, linewidth=0.5, zorder=0)
    ax.legend(loc="center left", fontsize=7, title="prompt set", title_fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "f2_library_vs_alternatives.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    f1()
    f2()
    print("wrote", sorted(os.listdir(OUT)))
