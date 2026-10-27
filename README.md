[README.md](https://github.com/user-attachments/files/34001373/README.md)
# AI-Based Banking Prediction and Fraud Detection Using Dataset and Weight Optimization

## Data
By default the script downloads both datasets itself with `kagglehub`:
- `janiobachmann/bank-marketing-dataset` (bank.csv, target `deposit`, ~11k rows)
- `mlg-ulb/creditcardfraud` (creditcard.csv, target `Class`)

If Kaggle asks for credentials, set `KAGGLE_USERNAME` and `KAGGLE_KEY` (or run `kagglehub.login()` once).
To use your own files instead, pass `--bank` / `--fraud`:
- **Bank Marketing**: UCI "Bank Marketing" -> `bank-full.csv` (semicolon-separated, target column `y`). Kaggle variants with a `deposit` target also work.
- **Credit Card Fraud**: ULB / Kaggle "Credit Card Fraud Detection" -> `creditcard.csv` (target column `Class`).

## Run
```bash
pip install -r requirements.txt
python banking_ml_project.py --out results                      # auto-download
python banking_ml_project.py --bank bank-full.csv --fraud creditcard.csv --out results   # local files
# optional
#   --quick           fewer trees/folds for a fast dry run (noisier numbers)
#   --drop-duration   remove the leaky 'duration' column from the bank data
#   --ratings-json '{"Bank Marketing": {"Problem relevance": 9}}'   override judgement ratings
```
Expect several minutes on the full fraud file (Random Forest, 200 trees, cross-validation).

## Outputs (in `results/`)
`REPORT.md` (full report), `figures/` (EDA, ROC/PR, confusion matrices, importance, score charts),
`models/*.joblib` (6 trained pipelines), and CSVs for metrics, dataset ratings/scores and model scores.

## Rating types
Every rating in the report is tagged **rule-derived** (formula from measured data/results) or
**JUDGEMENT** (problem relevance, real-world impact - yours to set). Nothing in the report is
hard-coded as a result; the example ratings in the earlier `dataset_auction_scoring.py` are superseded.
