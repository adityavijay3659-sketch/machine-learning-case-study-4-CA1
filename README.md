# Case Study 4: Sentiment Analysis

Compares a bidirectional LSTM trained from scratch with a fine-tuned pretrained BERT-Mini model (`google/bert_uncased_L-4_H-256_A-4`) on Amazon product reviews.

## Files

- `code.py`: trains the LSTM for eight epochs and BERT for three epochs, then prints classification metrics and confusion matrices.
- `dataset.csv`: 1,800 reviews, with 600 each labeled negative, neutral, and positive. Ratings 1–2 map to negative, 3 to neutral, and 4–5 to positive.
- `output.txt`: the supplied console output for this version. It replaces the previous PDF report and was not regenerated locally.
- `requirements.txt`: Python dependencies.

The script makes a stratified 80/20 training/test split with `random_state=42` (1,440 training and 360 test reviews). Its LSTM vocabulary is built from training reviews.

## Run

Use Python 3.11 or 3.12 in a virtual environment. Run these commands from the repository directory so the script can find `dataset.csv`:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python code.py
```

Internet access is required for the first pretrained model and tokenizer download. The script chooses MPS, CUDA, or CPU according to availability. Training prints to the console; to capture a new log instead, run:

```powershell
python -u code.py > output.txt 2>&1
```

## Source and attribution

The current `code.py`, `dataset.csv`, and `output.txt` were supplied by the repository owner and copied without modification. The earlier repository version was imported from [adityasinghin01-hash/case-study-4](https://github.com/adityasinghin01-hash/case-study-4) at commit `1356240bf2161f3d86b9ef9733fb743a710c1959`.

## Verification

Python syntax, dataset columns, nonempty and unique review text, class counts, and rating-to-sentiment labels were checked. The three replacement files were verified byte for byte against the supplied files. Model training was not run locally.
