"""Keşif analizi: dağılımlar, kişilik-DASS korelasyonları, şiddete göre kişilik
profilleri ve cinsiyet karşılaştırması. Grafikler figures/ altına kaydedilir.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parent.parent
IN = ROOT / "data" / "processed" / "scored.csv"
FIG = ROOT / "figures"

DASS = ["depression", "anxiety", "stress"]
TRAITS = ["extraversion", "agreeableness", "conscientiousness", "emotional_stability", "openness"]
SEVERITY = ["normal", "mild", "moderate", "severe", "extremely severe"]
GENDERS = {1: "Erkek", 2: "Kadın", 3: "Diğer"}

TR = {
    "depression": "Depresyon", "anxiety": "Kaygı", "stress": "Stres",
    "extraversion": "Dışadönüklük", "agreeableness": "Uyumluluk",
    "conscientiousness": "Sorumluluk", "emotional_stability": "Duygusal denge",
    "openness": "Deneyime açıklık",
    "normal": "Normal", "mild": "Hafif", "moderate": "Orta", "severe": "Şiddetli",
    "extremely severe": "Çok şiddetli",
}

# Renkler: dataviz referans paleti (kategorik sıra sabit, ıraksak = mavi <-> gri <-> kırmızı)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, INK_2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
DIVERGING = LinearSegmentedColormap.from_list("div", ["#184f95", "#6da7ec", "#f0efec", "#ec8a89", "#b52e2e"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 10,
    "text.color": INK, "axes.labelcolor": INK_2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": "#c3c2b7", "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
})


def save(fig, name: str) -> None:
    fig.savefig(FIG / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> figures/{name}")


def plot_distributions(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), sharey=True)
    for ax, scale in zip(axes, DASS):
        ax.hist(df[scale], bins=np.arange(0, 44, 2), color=SERIES[0], edgecolor=SURFACE, linewidth=1)
        ax.axvline(df[scale].median(), color=INK_2, linewidth=1, linestyle="--")
        ax.text(df[scale].median() + 0.8, ax.get_ylim()[1] * 0.92, f"medyan {df[scale].median():.0f}",
                color=INK_2, fontsize=9)
        ax.set_title(TR[scale])
        ax.set_xlabel("Puan (0-42)")
        ax.grid(axis="y", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Kişi sayısı")
    fig.suptitle("DASS alt ölçek dağılımları", x=0.01, y=1.04, ha="left", fontweight="bold")
    save(fig, "01_distributions.png")


def plot_correlations(df: pd.DataFrame) -> pd.DataFrame:
    corr = df[TRAITS + DASS].corr().loc[TRAITS, DASS]
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    im = ax.imshow(corr.values, cmap=DIVERGING, vmin=-0.7, vmax=0.7)
    for i in range(len(TRAITS)):
        for j in range(len(DASS)):
            v = corr.values[i, j]
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=10,
                    color="#ffffff" if abs(v) > 0.45 else INK)
    ax.set_xticks(range(len(DASS)), [TR[c] for c in DASS])
    ax.set_yticks(range(len(TRAITS)), [TR[t] for t in TRAITS])
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.outline.set_visible(False)
    cbar.set_label("Pearson r", color=INK_2)
    ax.set_title("Kişilik ve DASS korelasyonları")
    save(fig, "02_correlations.png")
    return corr


def plot_profiles(df: pd.DataFrame) -> pd.DataFrame:
    means = df.groupby("depression_severity", observed=True)[TRAITS].mean().loc[SEVERITY]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = np.arange(len(SEVERITY))
    for color, trait in zip(SERIES, TRAITS):
        ax.plot(x, means[trait], color=color, linewidth=2, marker="o", markersize=7,
                markeredgecolor=SURFACE, markeredgewidth=2)

    # Sağ uçtaki doğrudan etiketler: birbirine çok yakın olanları aralarında en az 0.2 kalacak şekilde it
    ends = means.iloc[-1].sort_values()
    label_y, prev = {}, -np.inf
    for trait, y in ends.items():
        label_y[trait] = prev = max(y, prev + 0.2)
    for trait in TRAITS:
        ax.text(x[-1] + 0.12, label_y[trait], TR[trait], color=INK_2, va="center", fontsize=9)
    ax.set_xticks(x, [TR[s] for s in SEVERITY])
    ax.set_xlim(-0.2, len(SEVERITY) - 0.2)
    ax.set_ylim(1, 7)
    ax.set_xlabel("Depresyon şiddet düzeyi")
    ax.set_ylabel("Ortalama TIPI puanı (1-7)")
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_title("Depresyon şiddetine göre kişilik profili")
    save(fig, "03_profiles_by_severity.png")
    return means


def cohens_d(a: pd.Series, b: pd.Series) -> float:
    pooled = np.sqrt(((len(a) - 1) * a.var() + (len(b) - 1) * b.var()) / (len(a) + len(b) - 2))
    return (a.mean() - b.mean()) / pooled


def gender_comparison(df: pd.DataFrame) -> pd.DataFrame:
    g = df[df["gender"].isin(GENDERS)].assign(group=lambda d: d["gender"].map(GENDERS))
    cols = DASS + TRAITS
    summary = g.groupby("group")[cols].agg(["mean", "std", "count"])

    men, women = g[g.group == "Erkek"], g[g.group == "Kadın"]
    d = pd.Series({c: cohens_d(women[c], men[c]) for c in cols}, name="d (kadın - erkek)")

    # Grafik: her alt ölçek için grup ortalaması ve %95 güven aralığı
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
    order = list(GENDERS.values())
    for ax, scale in zip(axes, DASS):
        for k, (group, color) in enumerate(zip(order, SERIES)):
            vals = g.loc[g.group == group, scale]
            ci = 1.96 * vals.std() / np.sqrt(len(vals))
            ax.errorbar(k, vals.mean(), yerr=ci, fmt="o", color=color, markersize=8,
                        markeredgecolor=SURFACE, markeredgewidth=2, elinewidth=2, capsize=0)
            ax.text(k + 0.12, vals.mean(), f"{vals.mean():.1f}", va="center", color=INK_2, fontsize=9)
        ax.set_xticks(range(len(order)), [f"{o}\n(n={(g.group == o).sum():,})" for o in order])
        ax.set_xlim(-0.5, len(order) - 0.3)
        ax.set_title(f"{TR[scale]}   d = {d[scale]:+.2f}")
        ax.grid(axis="y", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Ortalama puan (0-42)")
    axes[0].set_ylim(0, 42)
    fig.suptitle("Cinsiyete göre DASS ortalamaları (%95 GA; d = kadın - erkek, Cohen's d)",
                 x=0.01, y=1.04, ha="left", fontweight="bold")
    save(fig, "04_gender.png")

    means = summary.xs("mean", axis=1, level=1).T[order].round(2)
    return means.join(d.round(2))


def main() -> None:
    FIG.mkdir(exist_ok=True)
    df = pd.read_csv(IN)
    df["depression_severity"] = pd.Categorical(df["depression_severity"], SEVERITY, ordered=True)

    print("Grafikler:")
    plot_distributions(df)
    corr = plot_correlations(df)
    profiles = plot_profiles(df)
    gender = gender_comparison(df)

    pd.set_option("display.width", 120)
    print("\nKorelasyonlar (kişilik x DASS):\n", corr.round(2).to_string())
    print("\nDepresyon şiddetine göre kişilik ortalamaları:\n", profiles.round(2).to_string())
    print("\nCinsiyete göre ortalamalar ve etki büyüklüğü:\n", gender.to_string())


if __name__ == "__main__":
    main()
