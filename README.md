# TFM — Reddit & news sentiment for stock-movement prediction

Sentiment analysis of Reddit (r/WallStreetBets) comments and financial news (FNSPID)
to study predictability of market movements for 10 S&P 500 companies
(**AAPL, AMD, AMZN, DIS, GOOGL, META, MSFT, NFLX, NVDA, TSLA**), 2022–2023.
Primary sentiment model: RoBERTa StockTwits (+ 4 transformer models:
FinBERT, RoBERTa Social, DistilRoBERTa Financial, BERT Sentiment).

## Repository structure

```
.
├── manuscript.Rmd                    # full thesis: text, figures, result + descriptive tables
├── references.bib                    # bibliography
├── images/temporal_windows.png               # figure used by the manuscript
├── COMPANIES/
│   ├── extract_reddit.py             # Reddit extraction (Arctic Shift + PRAW)
│   ├── analyze_comments.Rmd          # comments analysis  -> results/
│   ├── analyze_news.Rmd              # news analysis      -> results/
│   ├── results/helper_comments.rds             # analyze output read by the manuscript
│   ├── results/helper_news.rds
│   └── <TICKER>/                     # one folder per company
│       ├── extract_<TICKER>.Rmd      # build clean comments + news from raw sources
│       ├── sentiment_comments_<TICKER>.py
│       ├── sentiment_news_<TICKER>.py        (8 companies with news)
│       ├── comments_<t>_clean.csv
│       ├── comments_<t>_all_sentiments.csv
│       ├── news_<T>.csv                       (8 with news)
│       └── news_<t>_sentiments_S1/S2.csv      (8 with news)
└── README.md
```

## Pipeline

```
 FNSPID news (external, ~22 GB)         extract_reddit.py
                │                               │
                │                               ▼
                │                       WSB comment dumps
                │                               │
                └───────────────┬───────────────┘
                                ▼
                      extract_<TICKER>.Rmd   (per company)
                                │
                ┌───────────────┴───────────────┐
                ▼                                ▼
     comments_<t>_clean.csv               news_<T>.csv
                ▼                                ▼
    sentiment_comments_<T>.py         sentiment_news_<T>.py
                ▼                                ▼
 comments_<t>_all_sentiments.csv  news_<t>_sentiments_S1/S2.csv
                ▼                                ▼
      analyze_comments.Rmd              analyze_news.Rmd
                ▼                                ▼
   helper_comments.rds          helper_news.rds
                └───────────────┬───────────────┘
                                ▼
                      manuscript.Rmd  →  PDF
                 (result + descriptive tables, figures, text)
```

FNSPID is the only true external input; the WSB comment dumps are produced by
`extract_reddit.py` (in the repo) and are just too large to commit. The
`analyze_*.Rmd` pull live market data (OHLC + volume) via `quantmod::getSymbols`
(Yahoo) and write their results to `helper_*.rds`; `manuscript.Rmd` reads
those (committed) to render the final tables, figures and text.

## Scripts

| File | Role |
|---|---|
| `COMPANIES/extract_reddit.py` | Reddit extraction (Arctic Shift API + PRAW). One run writes three files to the working directory: `<sub>_Discussion_Threads_<start>_to_<end>_posts.csv`, `_comments.csv` and `_posts_FILTRADO.csv` (intermediate). Only the `_comments.csv` dumps are used downstream |
| `COMPANIES/<T>/extract_<T>.Rmd` | Per-company extract: filters FNSPID news + WSB dumps by company keywords, applies temporal/weekend rules → writes `comments_<t>_clean.csv` and `news_<T>.csv` |
| `COMPANIES/<T>/sentiment_comments_<T>.py` | Per-company comments-sentiment (5 transformer models): reads `comments_<t>_clean.csv` → writes `comments_<t>_all_sentiments.csv` |
| `COMPANIES/<T>/sentiment_news_<T>.py` | Per-company news-sentiment (strategies S1 title-only / S2 smart-weighted); S2 weights by a company-specific term list |
| `COMPANIES/analyze_comments.Rmd` | Final comments analysis over all companies (models M1/M2/M3: net overnight, AH+PM decomposition, pos/neg by window). OLS + Newey-West HAC. Pulls market data live via `quantmod` |
| `COMPANIES/analyze_news.Rmd` | Final news analysis (Model A net shift / Model B pos-neg, strategies S1 & S2, mandatory 1-day lag). OLS + Newey-West HAC; market data live via `quantmod` |
| `manuscript.Rmd` | Full thesis document. Reads `helper_*.rds` + the committed sentiment CSVs to render the final result tables, the descriptive tables (per-company comments/news) and the figures. Knits to PDF (needs LaTeX + `references.bib`) |

## External data (not versioned — too large)

- **Stock/index market data** (OHLC + volume): pulled live inside the `analyze_*` Rmd via `quantmod::getSymbols` (Yahoo Finance). Nothing to download manually.
- **Financial news**: FNSPID dataset (`nasdaq_exteral_data.csv` / `All_external.csv`, ~22GB). Download from Hugging Face: [Zihan1004/FNSPID](https://huggingface.co/datasets/Zihan1004/FNSPID) (paper: [arXiv:2402.06698](https://arxiv.org/abs/2402.06698)). Place it at the repo root; `extract_<T>.Rmd` reads it as `../../nasdaq_exteral_data.csv`.
- **Reddit**: raw WSB Discussion-Thread dumps produced by `extract_reddit.py`. The script is interactive; the values used in the study are subreddit `wallstreetbets` and `comment_limit = 50`, run once per year (2022 and 2023) — those are the prompt defaults (press Enter), only the date range is entered each run. Each run produces `<sub>_Discussion_Threads_<start>_to_<end>_{posts,comments}.csv`. Put the two `_comments.csv` files (2022 and 2023) in a folder `extraccion_reddit_v.1.1/` at the repo root: `extract_<T>.Rmd` reads them from there and ignores the posts files.

The per-company CSVs committed here (`*_clean.csv`, `*_all_sentiments.csv`, `news_*`, `news_*_sentiments_S1/S2.csv`) let you run from the sentiment/analysis stage **without** the large external files.

## Reddit API credentials

`extract_reddit.py` needs PRAW credentials, read from the environment
(`REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET`, `REDDIT_USER_AGENT`). Create a
"script" app at <https://www.reddit.com/prefs/apps>, then either set the
variables in your shell or copy `.env.example` to `.env` and fill them in
(`.env` is git-ignored; the script auto-loads it if `python-dotenv` is
installed). Only the comment-extraction step uses them; everything downstream
runs from the committed CSVs without credentials.

## How to run

1. **From scratch** (needs the external data above): run `COMPANIES/<T>/extract_<T>.Rmd` per company.
2. **From the committed CSVs**: the `*_all_sentiments.csv` and `news_*_sentiments_S1/S2.csv` are already in the repo, so you can knit `analyze_comments.Rmd` / `analyze_news.Rmd` directly **from the `COMPANIES/` directory** (they use `getwd()` as base, loop over all companies, and need internet for `quantmod`). To regenerate the sentiments from the clean CSVs instead, run `sentiment_comments_<T>.py` / `sentiment_news_<T>.py` first (each reads/writes inside its own company folder).
3. **The manuscript**: `helper_comments.rds` and `helper_news.rds` are committed, so `manuscript.Rmd` knits straight to PDF (needs LaTeX + internet for the figures). Re-running the `analyze_*.Rmd` overwrites those `.rds` (note: since the analysis code is in English, regenerated tables would carry English labels).

## Notes

- **News**: META and NFLX have no usable news in FNSPID for the period → excluded from the news analysis (8 companies: AAPL, AMD, AMZN, DIS, GOOGL, MSFT, NVDA, TSLA). `analyze_news.Rmd` skips any company whose news CSVs are absent.
- **Time zone**: raw Reddit timestamps are in Madrid local time; the analysis converts `force_tz(Europe/Madrid) → with_tz(America/New_York)` to classify afterhours (≥16:00) / premarket (<09:30) windows.
- **Manuscript**: the descriptive tables/figures and the "wide" result-table layout live only in `manuscript.Rmd`; the `analyze_*.Rmd` provide the regression numbers it consumes (via the `.rds`). The cover page (`\includepdf{Portada.pdf}`) is commented out in the public version.
