"""Temiz veriden DASS-42 alt ölçek puanlarını ve TIPI kişilik skorlarını hesaplar.

Çıktı: data/processed/scored.csv (demografi + skorlar)
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
IN = ROOT / "data" / "processed" / "clean.csv"
OUT = ROOT / "data" / "processed" / "scored.csv"

# DASS-42 puanlama anahtarı (Lovibond & Lovibond, 1995)
DASS_SCALES = {
    "depression": [3, 5, 10, 13, 16, 17, 21, 24, 26, 31, 34, 37, 38, 42],
    "anxiety": [2, 4, 7, 9, 15, 19, 20, 23, 25, 28, 30, 36, 40, 41],
    "stress": [1, 6, 8, 11, 12, 14, 18, 22, 27, 29, 32, 33, 35, 39],
}

# Şiddet düzeyleri: her eşik bir sonraki kategorinin alt sınırı
SEVERITY_LABELS = ["normal", "mild", "moderate", "severe", "extremely severe"]
SEVERITY_CUTOFFS = {
    "depression": [10, 14, 21, 28],
    "anxiety": [8, 10, 15, 20],
    "stress": [15, 19, 26, 34],
}

# TIPI (Gosling vd., 2003): (düz madde, ters madde)
TIPI_TRAITS = {
    "extraversion": (1, 6),
    "agreeableness": (7, 2),
    "conscientiousness": (3, 8),
    "emotional_stability": (9, 4),
    "openness": (5, 10),
}


def cronbach_alpha(items: pd.DataFrame) -> float:
    k = items.shape[1]
    item_var = items.var(axis=0, ddof=1).sum()
    total_var = items.sum(axis=1).var(ddof=1)
    return k / (k - 1) * (1 - item_var / total_var)


def score(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    out = df[["age", "gender", "education", "urban", "married", "country"]].copy()
    alphas = {}

    for scale, items in DASS_SCALES.items():
        cols = df[[f"Q{i}A" for i in items]] - 1  # 1-4 kodlamasını orijinal 0-3'e çevir
        out[scale] = cols.sum(axis=1)
        bins = [-np.inf, *SEVERITY_CUTOFFS[scale], np.inf]
        out[f"{scale}_severity"] = pd.cut(out[scale], bins, right=False, labels=SEVERITY_LABELS)
        alphas[scale] = cronbach_alpha(cols)

    for trait, (straight, reverse) in TIPI_TRAITS.items():
        out[trait] = (df[f"TIPI{straight}"] + (8 - df[f"TIPI{reverse}"])) / 2

    return out, alphas


def main() -> None:
    df = pd.read_csv(IN)
    out, alphas = score(df)
    out.to_csv(OUT, index=False)

    print("Cronbach alfa (DASS):")
    for scale, a in alphas.items():
        print(f"  {scale:<12}{a:.3f}")

    print("\nŞiddet dağılımı (%):")
    dist = pd.DataFrame({s: out[f"{s}_severity"].value_counts(normalize=True) for s in DASS_SCALES})
    print((dist.loc[SEVERITY_LABELS] * 100).round(1).to_string())

    print("\nSkor özetleri:")
    cols = list(DASS_SCALES) + list(TIPI_TRAITS)
    print(out[cols].describe().loc[["mean", "std", "min", "max"]].round(2).T.to_string())
    print(f"\nKaydedildi: {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
