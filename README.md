# dass-personality

Kişilik özellikleri (Big Five / TIPI) ile depresyon, kaygı ve stres (DASS-42) arasındaki ilişkiyi inceleyen küçük bir veri bilimi projesi.

Veri: [Open Psychometrics](https://openpsychometrics.org/_rawdata/) — DASS çevrimiçi test verisi (~39 bin katılımcı).

## Kurulum

```bash
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python src/download_data.py
```
