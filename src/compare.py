"""
Этап 4: сводное сравнение всех методов + графики для НИРСа.

Запуск:
    python src/compare.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

RESULTS_DIR = Path(__file__).parent.parent / "results"
FIGURES_DIR = Path(__file__).parent.parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

CLASS_COLS = ["f1_legit", "f1_sqli", "f1_obfuscated", "f1_privilege", "f1_exfiltration"]
CLASS_NAMES = ["legit", "sqli", "obfuscated", "privilege", "exfiltration"]

METHOD_ORDER = [
    "Regex",
    "TF-IDF + LogReg",
    "TF-IDF + LinearSVC",
    "TF-IDF + RandomForest",
    "LLM Llama-3.3-70B (zero_shot)",
    "LLM Llama-3.3-70B (few_shot)",
    "LLM zai-org-GLM-5.1 (few_shot)",
    "LLM openai-gpt-oss-120b (few_shot)",
]

METHOD_COLORS = {
    "Regex":                               "#e07b54",
    "TF-IDF + LogReg":                     "#5b8db8",
    "TF-IDF + LinearSVC":                  "#3a6ea5",
    "TF-IDF + RandomForest":               "#2a4e7c",
    "LLM Llama-3.3-70B (zero_shot)":       "#6dbf67",
    "LLM Llama-3.3-70B (few_shot)":        "#3d9e36",
    "LLM zai-org-GLM-5.1 (few_shot)":      "#f0a500",
    "LLM openai-gpt-oss-120b (few_shot)":  "#c0392b",
}

GROUP_COLORS = {
    "Regex": "#e07b54",
    "TF-IDF": "#3a6ea5",
    "LLM": "#3d9e36",
}


def load_results() -> pd.DataFrame:
    frames = [pd.read_csv(RESULTS_DIR / "baseline_results.csv")]

    for f in RESULTS_DIR.glob("llm_results*.csv"):
        frames.append(pd.read_csv(f))

    df = pd.concat(frames, ignore_index=True)

    # оставляем только методы из METHOD_ORDER
    known = set(METHOD_ORDER)
    df = df[df["method"].isin(known)]

    df["method"] = pd.Categorical(df["method"], categories=METHOD_ORDER, ordered=True)
    df = df.sort_values("method").reset_index(drop=True)
    return df


# ── График 1: F1 macro по методам ─────────────────────────────────────────

def plot_f1_macro(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))

    colors = [METHOD_COLORS.get(m, "#888") for m in df["method"]]
    bars = ax.bar(df["method"], df["f1_macro"], color=colors, edgecolor="white", width=0.6)

    for bar, val in zip(bars, df["f1_macro"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_ylim(0, 1.1)
    ax.set_ylabel("F1-macro", fontsize=12)
    ax.set_title("Comparison of detection methods: F1-macro score", fontsize=13, fontweight="bold")
    ax.set_xticklabels(df["method"], rotation=20, ha="right", fontsize=10)
    ax.axhline(0.9, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)

    legend = [
        mpatches.Patch(color="#e07b54", label="Rule-based (Regex)"),
        mpatches.Patch(color="#3a6ea5", label="ML (TF-IDF)"),
        mpatches.Patch(color="#3d9e36", label="LLM (Llama 3.1 8B)"),
    ]
    ax.legend(handles=legend, fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    out = FIGURES_DIR / "f1_macro_comparison.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[+] Saved: {out}")


# ── График 2: Heatmap F1 по классам ───────────────────────────────────────

def plot_f1_heatmap(df: pd.DataFrame) -> None:
    heat = df.set_index("method")[CLASS_COLS].copy()
    heat.columns = CLASS_NAMES
    heat.index = [str(m) for m in heat.index]

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.heatmap(
        heat.astype(float),
        annot=True, fmt=".2f", cmap="RdYlGn",
        vmin=0, vmax=1, linewidths=0.5,
        ax=ax, cbar_kws={"label": "F1-score"},
    )
    ax.set_title("F1-score by threat class and method", fontsize=13, fontweight="bold")
    ax.set_xlabel("Threat class", fontsize=11)
    ax.set_ylabel("Method", fontsize=11)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    fig.tight_layout()
    out = FIGURES_DIR / "f1_heatmap.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[+] Saved: {out}")


# ── График 3: Latency ─────────────────────────────────────────────────────

def plot_latency(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))

    colors = [METHOD_COLORS.get(m, "#888") for m in df["method"]]
    bars = ax.barh(df["method"].astype(str)[::-1],
                   df["latency_ms_per_sample"][::-1],
                   color=colors[::-1], edgecolor="white", height=0.6)

    for bar, val in zip(bars, df["latency_ms_per_sample"][::-1]):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f} ms", va="center", fontsize=9)

    ax.set_xlabel("Latency (ms per sample)", fontsize=11)
    ax.set_title("Detection latency per sample", fontsize=13, fontweight="bold")
    ax.set_xlim(0, df["latency_ms_per_sample"].max() * 1.3)
    ax.grid(axis="x", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    out = FIGURES_DIR / "latency_comparison.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[+] Saved: {out}")


# ── График 4: Radar (паутина) precision/recall/f1 ─────────────────────────

def plot_radar(df: pd.DataFrame) -> None:
    metrics = ["f1_macro", "precision_macro", "recall_macro"]
    labels  = ["F1-macro", "Precision", "Recall"]
    N = len(metrics)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"polar": True})

    for _, row in df.iterrows():
        values = [row[m] for m in metrics]
        values += values[:1]
        color = METHOD_COLORS.get(row["method"], "#888")
        ax.plot(angles, values, "o-", linewidth=2, color=color, label=str(row["method"]))
        ax.fill(angles, values, alpha=0.07, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8)
    ax.set_title("Precision / Recall / F1 radar", fontsize=13, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out = FIGURES_DIR / "radar_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[+] Saved: {out}")


# ── Итоговая таблица ──────────────────────────────────────────────────────

def print_summary(df: pd.DataFrame) -> None:
    cols = ["method", "f1_macro", "f1_weighted", "precision_macro",
            "recall_macro", "latency_ms_per_sample"]
    summary = df[cols].copy()
    summary.columns = ["Method", "F1-macro", "F1-weighted", "Precision", "Recall", "Latency(ms)"]
    for c in ["F1-macro", "F1-weighted", "Precision", "Recall"]:
        summary[c] = summary[c].map("{:.3f}".format)
    summary["Latency(ms)"] = summary["Latency(ms)"].map("{:.2f}".format)

    print("\n" + "="*75)
    print("  FINAL COMPARISON TABLE")
    print("="*75)
    print(summary.to_string(index=False))
    print("="*75)

    out = RESULTS_DIR / "summary_table.csv"
    summary.to_csv(out, index=False, encoding="utf-8")
    print(f"[+] Saved: {out}")


def main() -> None:
    df = load_results()
    print_summary(df)
    plot_f1_macro(df)
    plot_f1_heatmap(df)
    plot_latency(df)
    plot_radar(df)
    print(f"\n[+] All figures saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
