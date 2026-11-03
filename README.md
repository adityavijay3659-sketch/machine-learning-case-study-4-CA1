# Case Study 4: Sentiment Analysis

Fine-tunes pretrained BERT-Mini and compares it with an LSTM on Amazon review sentiments. The first run needs internet to download the pretrained model; training may take time.

## Source and attribution

Copied from [adityasinghin01-hash/case-study-4](https://github.com/adityasinghin01-hash/case-study-4) at commit `1356240bf2161f3d86b9ef9733fb743a710c1959`.

The original `code.py`, `dataset.csv`, and `output.pdf` are preserved byte for byte. Original author, dataset, and model attribution in the script is retained. The included PDF is the source project's existing report; it has not been regenerated locally.

## Run

Use Python 3.11 or 3.12 in a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python code.py
```

The script reads `dataset.csv` beside itself and writes `output.pdf` in the same directory.

## Verification

All three copied files were checked against the source Git blob hashes, and Python syntax was checked. Dataset columns and row counts were verified. Model training was not run locally.
