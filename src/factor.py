"""DASS-42 gerçekten 3 ayrı boyut mu ölçüyor? Faktör analizi.

1. Uygunluk: KMO ve Bartlett testleri
2. Kaç faktör: özdeğerler, scree plot ve paralel analiz
3. Açımlayıcı faktör analizi (3 faktör, eğik/oblimin rotasyon): maddeler resmi alt ölçeklerine mi yükleniyor?
4. Schmid-Leiman dönüşümüyle bifaktör yapısı: omega hiyerarşik ve ECV
   ("genel sıkıntı" faktörü toplam puanın ne kadarını belirliyor?)

Not: Maddeler 4 basamaklı olduğu için ideal olan polikorik korelasyondur; basitlik için
Pearson korelasyonu kullanıldı. Bu, yükleri bir miktar düşük tahmin edebilir.
"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.utils
import factor_analyzer.factor_analyzer as fa_module
from factor_analyzer import FactorAnalyzer, calculate_bartlett_sphericity, calculate_kmo

from explore import DIVERGING, FIG, GRID, INK, INK_2, MUTED, SERIES, SURFACE, TR, save
from score import DASS_SCALES

# factor_analyzer 0.5.1, scikit-learn >= 1.6'da adı değişen bir parametreyi kullanıyor
# (force_all_finite -> ensure_all_finite). Kütüphane güncellenene kadar küçük bir uyum katmanı.
fa_module.check_array = lambda X, force_all_finite=True, **kw: sklearn.utils.check_array(
    X, ensure_all_finite=force_all_finite, **kw
)

ROOT = Path(__file__).resolve().parent.parent
IN = ROOT / "data" / "processed" / "clean.csv"

ITEMS = [f"Q{i}A" for i in range(1, 43)]
ITEM_SCALE = {f"Q{i}A": scale for scale, nums in DASS_SCALES.items() for i in nums}
N_PARALLEL = 20


def parallel_analysis(n: int, p: int, rng: np.random.Generator) -> np.ndarray:
    """Aynı boyutta rastgele verinin özdeğerlerinin 95. yüzdeliği."""
    eig = [np.linalg.eigvalsh(np.corrcoef(rng.standard_normal((n, p)), rowvar=False))[::-1]
           for _ in range(N_PARALLEL)]
    return np.percentile(eig, 95, axis=0)


def plot_scree(real: np.ndarray, random: np.ndarray, n_keep: int) -> None:
    k = 10
    x = np.arange(1, k + 1)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x, real[:k], color=SERIES[0], linewidth=2, marker="o", markersize=8,
            markeredgecolor=SURFACE, markeredgewidth=2)
    ax.plot(x, random[:k], color=MUTED, linewidth=2, linestyle="--")
    ax.text(2.25, real[1] + 0.5, "Gerçek veri", color=INK_2, fontsize=9)
    ax.text(7, random[6] + 0.6, "Rastgele veri (paralel analiz)", color=INK_2, fontsize=9)
    ax.annotate(f"İlk faktör: {real[0]:.1f}", (1, real[0]), xytext=(1.6, real[0] - 1),
                color=INK, fontsize=9)
    ax.set_xticks(x)
    ax.set_xlim(0.5, k + 0.5)
    ax.set_xlabel("Faktör")
    ax.set_ylabel("Özdeğer")
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_title(f"Scree plot: paralel analiz {n_keep} faktör öneriyor")
    save(fig, "06_scree.png")


def match_factors(loadings: pd.DataFrame) -> dict[int, str]:
    """Her faktörü, maddeleri ona en yüksek ortalama yükü veren alt ölçekle eşleştirir."""
    by_scale = loadings.groupby(ITEM_SCALE).mean()
    return {f: by_scale[f].idxmax() for f in loadings.columns}


def plot_loadings(loadings: pd.DataFrame, misfits: list[str]) -> None:
    order = [i for scale in DASS_SCALES for i in sorted(DASS_SCALES[scale])]
    rows = [f"Q{i}A" for i in order]
    L = loadings.loc[rows, list(DASS_SCALES)]

    fig, ax = plt.subplots(figsize=(4.8, 11))
    im = ax.imshow(L.values, cmap=DIVERGING, vmin=-1, vmax=1, aspect="auto")
    for r, item in enumerate(rows):
        for c in range(L.shape[1]):
            v = L.values[r, c]
            ax.text(c, r, f"{v:.2f}", ha="center", va="center", fontsize=7.5,
                    color="#ffffff" if abs(v) > 0.6 else INK)
    labels = [f"{'⚠ ' if item in misfits else ''}{item[:-1]}  ({TR[ITEM_SCALE[item]][:4]}.)" for item in rows]
    ax.set_yticks(range(len(rows)), labels, fontsize=8)
    ax.set_xticks(range(3), [f"Faktör:\n{TR[s]}" for s in DASS_SCALES], fontsize=9)
    ax.xaxis.tick_top()
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    # Alt ölçek sınırları
    for b in (14, 28):
        ax.axhline(b - 0.5, color=SURFACE, linewidth=3)
    cbar = fig.colorbar(im, ax=ax, shrink=0.4, pad=0.03)
    cbar.outline.set_visible(False)
    cbar.set_label("Faktör yükü", color=INK_2)
    ax.set_title("3 faktörlü çözüm: madde yükleri (oblimin)", pad=36)
    save(fig, "07_loadings.png")


def schmid_leiman(loadings: pd.DataFrame, phi: np.ndarray, R: np.ndarray) -> dict[str, float]:
    """3 eğik faktörden genel + özgül faktörlere geçiş; omega hiyerarşik, omega toplam ve ECV.

    3 faktörlü bir korelasyon matrisinde tek ikinci düzey faktörün yükleri kapalı formda
    çözülür: g_i = sqrt(r_ij * r_ik / r_jk).
    """
    g = np.array([np.sqrt(phi[i, j] * phi[i, k] / phi[j, k])
                  for i, j, k in [(0, 1, 2), (1, 0, 2), (2, 0, 1)]])
    lam = loadings.values
    general = lam @ g
    specific = lam * np.sqrt(1 - g**2)
    total_var = R.sum()  # toplam puanın varyansı (standartlaştırılmış maddelerle)
    uniq = 1 - (general**2 + (specific**2).sum(axis=1))
    return {
        "second_order": dict(zip(loadings.columns, g.round(3).tolist())),
        "omega_total": 1 - uniq.sum() / total_var,
        "omega_h": general.sum() ** 2 / total_var,
        "ecv": (general**2).sum() / ((general**2).sum() + (specific**2).sum()),
    }


def main() -> None:
    X = pd.read_csv(IN)[ITEMS] - 1
    R = X.corr().values
    rng = np.random.default_rng(42)

    # 1. Uygunluk
    chi2, p = calculate_bartlett_sphericity(X)
    with warnings.catch_warnings():
        # 42 madde birbiriyle çok ilişkili olduğundan KMO'da matris tersi yerine
        # genelleştirilmiş ters kullanılıyor; sonuç için sorun değil.
        warnings.simplefilter("ignore", UserWarning)
        _, kmo = calculate_kmo(X)
    print(f"n = {len(X):,}, madde = {X.shape[1]}")
    print(f"KMO = {kmo:.3f}   (0,90+ 'mükemmel')")
    print(f"Bartlett chi2 = {chi2:,.0f}, p = {p:.3g}")

    # 2. Kaç faktör?
    real = np.linalg.eigvalsh(R)[::-1]
    random = parallel_analysis(*X.shape, rng)
    n_parallel = int(np.argmax(real <= random))
    print("\nİlk 6 özdeğer (gerçek):  ", real[:6].round(2))
    print("İlk 6 özdeğer (rastgele):", random[:6].round(2))
    print(f"Kaiser (>1): {(real > 1).sum()} faktör | Paralel analiz: {n_parallel} faktör")
    print(f"İlk faktörün açıkladığı varyans: %{real[0] / len(real) * 100:.1f}")
    plot_scree(real, random, n_parallel)

    # 3. EFA, 3 faktör
    fa = FactorAnalyzer(n_factors=3, rotation="oblimin", method="minres")
    fa.fit(X)
    loadings = pd.DataFrame(fa.loadings_, index=ITEMS)
    mapping = match_factors(loadings)
    loadings = loadings.rename(columns=mapping)[list(DASS_SCALES)]
    order = [list(mapping.values()).index(s) for s in DASS_SCALES]
    phi = fa.phi_[np.ix_(order, order)]

    print("\nFaktör korelasyonları:")
    print(pd.DataFrame(phi, index=[TR[s] for s in DASS_SCALES], columns=[TR[s] for s in DASS_SCALES]).round(2).to_string())

    primary = loadings.abs().idxmax(axis=1)
    misfits = [i for i in ITEMS if primary[i] != ITEM_SCALE[i]]
    weak = [i for i in ITEMS if loadings.loc[i].abs().max() < 0.4]
    cross = [i for i in ITEMS if (loadings.loc[i].abs() >= 0.3).sum() > 1]
    print(f"\nKendi alt ölçeğine en yüksek yüklenen madde: {42 - len(misfits)}/42")
    for i in misfits:
        print(f"  {i}: resmi = {TR[ITEM_SCALE[i]]}, en yüksek yük = {TR[primary[i]]} "
              f"({loadings.loc[i].round(2).to_dict()})")
    print(f"Zayıf maddeler (en yüksek yük < 0,40): {weak or 'yok'}")
    print(f"Çapraz yüklenen maddeler (2+ faktörde >= 0,30): {cross or 'yok'}")
    plot_loadings(loadings, misfits)

    # 4. Bifaktör (Schmid-Leiman)
    sl = schmid_leiman(loadings, phi, R)
    print("\nİkinci düzey (genel faktör) yükleri:", {TR[k]: v for k, v in sl["second_order"].items()})
    print(f"Omega toplam      = {sl['omega_total']:.3f}")
    print(f"Omega hiyerarşik  = {sl['omega_h']:.3f}   (toplam puanın genel faktörce belirlenen kısmı)")
    print(f"ECV               = {sl['ecv']:.3f}   (ortak varyansın genel faktöre düşen payı)")


if __name__ == "__main__":
    main()
