"""Ham DASS verisini temizler ve data/processed/clean.csv olarak kaydeder.

Her filtre adımında kaç kişinin elendiği yazdırılır, böylece hangi kararın
örneklemi ne kadar etkilediği görülebilir.
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "data.csv"
OUT = ROOT / "data" / "processed" / "clean.csv"

DASS_ITEMS = [f"Q{i}A" for i in range(1, 43)]
TIPI_ITEMS = [f"TIPI{i}" for i in range(1, 11)]
FAKE_WORDS = ["VCL6", "VCL9", "VCL12"]  # cuivocal, florted, verdid
DEMOGRAPHICS = ["age", "gender", "education", "urban", "married", "country"]

MIN_AGE, MAX_AGE = 18, 80
MIN_TEST_SECONDS = 60  # 42 madde için soru başına ~1.4 sn


def clean(df: pd.DataFrame) -> pd.DataFrame:
    filters = [
        ("DASS yanıtı eksik değil", ((df[DASS_ITEMS] >= 1) & (df[DASS_ITEMS] <= 4)).all(axis=1)),
        ("TIPI yanıtı eksik değil", ((df[TIPI_ITEMS] >= 1) & (df[TIPI_ITEMS] <= 7)).all(axis=1)),
        ("Uydurma kelime işaretlememiş", df[FAKE_WORDS].sum(axis=1) == 0),
        (f"Yaş {MIN_AGE}-{MAX_AGE}", df["age"].between(MIN_AGE, MAX_AGE)),
        (f"Test süresi >= {MIN_TEST_SECONDS} sn", df["testelapse"] >= MIN_TEST_SECONDS),
    ]

    keep = pd.Series(True, index=df.index)
    print(f"{'Başlangıç':<32}{len(df):>8,}")
    for name, mask in filters:
        before = keep.sum()
        keep &= mask
        print(f"{name:<32}{keep.sum():>8,}   (-{before - keep.sum():,})")

    return df.loc[keep, DEMOGRAPHICS + DASS_ITEMS + TIPI_ITEMS].reset_index(drop=True)


def main() -> None:
    df = pd.read_csv(RAW, sep="\t")
    out = clean(df)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"\nKaydedildi: {OUT.relative_to(ROOT)}  ({len(out):,} satır, %{len(out) / len(df) * 100:.1f} korundu)")


if __name__ == "__main__":
    main()
