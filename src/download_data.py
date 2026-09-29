"""Open Psychometrics'ten DASS ham verisini indirip data/raw/ altına açar."""

import io
import urllib.request
import zipfile
from pathlib import Path

URL = "https://openpsychometrics.org/_rawdata/DASS_data_21.02.19.zip"
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"İndiriliyor: {URL}")
    with urllib.request.urlopen(URL) as resp:
        payload = resp.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        for member in zf.infolist():
            if member.is_dir():
                continue
            # Zip içindeki klasör yapısını düzleştir: data/raw/data.csv, data/raw/codebook.txt
            target = RAW_DIR / Path(member.filename).name
            target.write_bytes(zf.read(member))
            print(f"  -> {target.relative_to(RAW_DIR.parent.parent)}")


if __name__ == "__main__":
    main()
