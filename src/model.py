"""Kişilik (+ yaş, cinsiyet) ile DASS alt ölçeklerini tahmin eden doğrusal regresyon.

- R², 5 katlı çapraz doğrulamayla (modelin görmediği kişiler üzerinde) ölçülür.
- Katsayılar, sürekli girdiler ve hedef z-skoruna çevrildikten sonra tüm veride tahmin edilir
  (standartlaştırılmış beta): "1 SD artış, DASS'ta kaç SD değişim?" Cinsiyet 0/1 kalır.
- Duygusal denge çıkarılarak kurulan ikinci model döngüselliği kontrol eder
  (TIPI4 = "kaygılı, kolay üzülen").
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score

from explore import DASS, FIG, GRID, INK_2, SERIES, SURFACE, TR, TRAITS, save

ROOT = Path(__file__).resolve().parent.parent
IN = ROOT / "data" / "processed" / "scored.csv"

TR.update({"age": "Yaş", "female": "Kadın (vs erkek)", "other_gender": "Diğer (vs erkek)"})
CV = KFold(n_splits=5, shuffle=True, random_state=42)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["gender"].isin([1, 2, 3])].copy()
    df["female"] = (df["gender"] == 2).astype(int)
    df["other_gender"] = (df["gender"] == 3).astype(int)
    return df


def zscore(frame: pd.DataFrame) -> pd.DataFrame:
    return (frame - frame.mean()) / frame.std()


def cv_r2(X: pd.DataFrame, y: pd.Series) -> float:
    return cross_val_score(LinearRegression(), X, y, cv=CV, scoring="r2").mean()


def fit_betas(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    # 0/1 değişkenler olduğu gibi kalır: katsayı = referans gruba göre kaç SD fark.
    # (Nadir bir grubu z-skorlamak katsayısını yapay olarak küçültürdü.)
    binary = [c for c in X if set(X[c].unique()) <= {0, 1}]
    X = X.copy()
    cont = X.columns.difference(binary)
    X[cont] = zscore(X[cont])
    res = sm.OLS(zscore(y), sm.add_constant(X)).fit()
    ci = res.conf_int().drop(index="const")
    return pd.DataFrame({"beta": res.params.drop("const"), "lo": ci[0], "hi": ci[1]})


def plot_betas(betas: dict[str, pd.DataFrame]) -> None:
    predictors = list(next(iter(betas.values())).index)[::-1]
    y = np.arange(len(predictors))
    offsets = np.linspace(-0.22, 0.22, len(DASS))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axvline(0, color="#c3c2b7", linewidth=1)
    for color, off, scale in zip(SERIES, offsets, DASS):
        b = betas[scale].loc[predictors]
        ax.errorbar(b["beta"], y + off, xerr=[b["beta"] - b["lo"], b["hi"] - b["beta"]], fmt="o",
                    color=color, markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.5,
                    elinewidth=1.5, capsize=0, label=TR[scale])
    ax.set_yticks(y, [TR[p] for p in predictors])
    ax.set_xlabel("Standartlaştırılmış beta (%95 GA); cinsiyet: erkeklere göre SD farkı")
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1), labelcolor=INK_2)
    ax.set_title("Kişilik ve demografinin DASS'a özgün katkısı")
    save(fig, "05_regression_betas.png")


def main() -> None:
    df = prepare(pd.read_csv(IN))
    demo = ["age", "female", "other_gender"]
    specs = {
        "Sadece demografi": demo,
        "Kişilik + demografi": TRAITS + demo,
        "Duygusal denge hariç": [t for t in TRAITS if t != "emotional_stability"] + demo,
    }

    r2 = pd.DataFrame({name: {TR[s]: cv_r2(df[cols], df[s]) for s in DASS} for name, cols in specs.items()})
    print(f"n = {len(df):,}\n")
    print("Çapraz doğrulamalı R² (5 kat):")
    print(r2.round(3).to_string())

    betas = {s: fit_betas(df[specs["Kişilik + demografi"]], df[s]) for s in DASS}
    table = pd.DataFrame({TR[s]: b["beta"] for s, b in betas.items()}).rename(index=TR)
    print("\nStandartlaştırılmış betalar (Kişilik + demografi):")
    print(table.round(3).to_string())

    FIG.mkdir(exist_ok=True)
    print("\nGrafik:")
    plot_betas(betas)


if __name__ == "__main__":
    main()
