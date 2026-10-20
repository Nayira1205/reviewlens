# ReviewLens

**Sentiment analysis of Flipkart electronics product reviews**
CSE4039 Advanced Text and Social Media Analytics, lab project (VIT-AP)

ReviewLens reads a customer review and predicts whether it is **positive, neutral or negative**. It is trained on over 67,000 Flipkart electronics reviews, using the customer's star rating as the label.

> **Status:** in progress. Data preparation and the baseline model are done. See the [roadmap](#roadmap).

---

## The problem in one minute

Shoppers leave thousands of reviews. Reading them all is impossible, but the overall mood (and what people like or dislike, for example battery or camera) matters to buyers and sellers. We teach a model to read review text and label its sentiment.

## How it works

```
 raw reviews (CSV)
      |
      v
 1. Prepare data      remove duplicates, build labels from star ratings,
      |               keep only electronics products
      v
 2. Clean the text    lowercase, remove noise, KEEP negations ("not good")
      |
      v
 3. Text -> numbers   TF-IDF (words and word pairs)
      |
      v
 4. Train a model     Logistic Regression (baseline)
      |
      v
 5. Evaluate          precision / recall / macro-F1, error analysis
```

## Folder structure

```
reviewlens/
├── README.md                  <- you are here
├── requirements.txt           <- Python libraries needed
├── notebooks/
│   ├── 01_data_preparation.ipynb   <- load, check, label, filter, clean the data
│   ├── 02_baseline_model.ipynb     <- split, TF-IDF, train, evaluate, error analysis
│   └── 03_model_comparison.ipynb   <- Naive Bayes vs SVM vs Logistic Regression, tuning
├── src/                       <- reusable code used by the notebooks
│   ├── preprocess.py               <- text cleaning (keeps negations)
│   └── electronics_filter.py       <- keyword rules to select electronics products
├── docs/
│   ├── GLOSSARY.md                 <- every ML term used, in plain English
│   └── VIVA_QA.md                  <- likely questions and answers about this project
├── data/
│   ├── raw/                        <- put the downloaded CSV here (not in Git)
│   └── processed/                  <- generated files (not in Git)
├── models/                         <- saved models (not in Git)
└── reports/                        <- charts and results for the report
```

**Start with the notebooks, in order.** Each one explains every step in plain language next to the code.

## How to run it

1. **Set up Python**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   python -m nltk.downloader stopwords wordnet
   ```
   (On macOS, if the NLTK download shows an SSL certificate error, run `pip install certifi` and `export SSL_CERT_FILE=$(python -c "import certifi; print(certifi.where())")` first.)

2. **Get the data.** Download the Kaggle dataset *Flipkart Product Customer Reviews Dataset* and save the CSV as `data/raw/Dataset-SA 2.csv` (or change `RAW_PATH` in notebook 01).

3. **Run the notebooks in order:** `jupyter notebook`, then open `notebooks/01_data_preparation.ipynb`, then `02_baseline_model.ipynb`, then `03_model_comparison.ipynb`, running each cell from top to bottom.

## Dataset

Flipkart product reviews (Kaggle). Columns: `product_name`, `product_price`, `Rate` (1 to 5 stars), `Review` (title), `Summary` (review text), `Sentiment` (VADER label, not used for training).

| Step | Rows |
|---|---|
| Raw file | about 205,000 |
| After removing duplicates and missing text | about 170,600 |
| Electronics products only | about 67,400 |

## Key design decisions (and why)

| Decision | Why |
|---|---|
| Labels come from **star ratings** (4-5 positive, 3 neutral, 1-2 negative), not the dataset's VADER column | VADER agreed with stars on most reviews but only about 22% of 3-star reviews. Training on VADER would teach the model to copy VADER. |
| Use `Summary` as input, ignore `Review` | `Review` is a short title that often states the sentiment outright (label leakage). |
| Remove duplicate rows **before** splitting | Identical rows in train and test inflate scores. |
| Build an **electronics filter** and hand-check it | The raw data mixes many product categories and has no category column. |
| **Keep negations** during cleaning | Standard stop-word lists delete "not", turning "not good" into "good". |
| Report **macro-F1**, not only accuracy | About 78% of reviews are positive, so an "always positive" model gets about 78% accuracy. |
| Add a **strict test set** | Shows the model is not just memorising repeated texts like "good". |

## Results so far (baseline: TF-IDF + Logistic Regression)

| Test set | Accuracy | Macro-F1 | Neutral F1 |
|---|---|---|---|
| Normal (13,481 reviews) | 0.791 | 0.659 | 0.285 |
| Strict (7,621 reviews) | 0.801 | 0.660 | 0.301 |

- Positive and negative reviews are handled well (F1 about 0.8 to 0.9).
- **Neutral is the hardest class**: 3-star reviews are often worded politely, and identical text receives different star ratings from different people (label noise).

## Roadmap

- [x] Project setup and repository
- [x] Data exploration, cleaning and labelling
- [x] Electronics filter (validated by hand on random samples)
- [x] Text preprocessing that keeps negations
- [x] Baseline model and evaluation
- [ ] Compare Naive Bayes, Linear SVM and Logistic Regression
- [ ] Improve the neutral class
- [ ] Transformer model (DistilBERT) and comparison
- [ ] Aspect-based sentiment (battery, camera, price, ...)
- [ ] Streamlit app: paste a review, see the prediction
- [ ] Same pipeline on food and agriculture products (second domain)

## Limitations

- Star ratings are an imperfect stand-in for true sentiment, especially for 3-star reviews.
- The electronics filter uses keywords, so a few non-electronics or missed electronics products remain.
- Cleaning removes emojis and non-English text; Hinglish reviews are not handled specially.

## Tools

Python, pandas, NumPy, scikit-learn, NLTK, matplotlib, seaborn, Jupyter.