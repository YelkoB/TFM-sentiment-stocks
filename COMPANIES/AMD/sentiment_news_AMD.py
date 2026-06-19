"""
sentiment_news_AMD.py - Financial-news sentiment for AMD (AMD)

Two title/summary weighting strategies, five transformer models (FinBERT,
RoBERTa Social, RoBERTa StockTwits, DistilRoBERTa Financial, BERT Sentiment):
    S1 (title-only):     sentiment of the article title only
    S2 (smart-weighted): 100% title, unless the title omits the company but the
                         summary mentions it >= 2 times -> 30% title + 70% summary

Input:  news_AMD.csv  (columns: Date, Article_title, Lsa_summary, Stock_symbol)
Output: news_amd_sentiments_S1_title_only.csv
        news_amd_sentiments_S2_smart_weighted.csv
        Per model/strategy: s{1,2}_{model}_label (positive|negative|neutral),
                            _score (-1|0|1), _confidence [0, 1]
"""

import os
import re
import warnings

import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ============================== Configuration ==============================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(SCRIPT_DIR, "news_AMD.csv")
OUTPUT_FILE_BASE = os.path.join(SCRIPT_DIR, "news_amd_sentiments")
BATCH_SIZE = 32

# Company terms for contextual detection. PRIMARY (ticker + company name) is
# used for explicit-mention checks; the full list drives relevance/sentences.
TERMS = [
    'amd', 'advanced micro devices',
    'ryzen', 'threadripper',
    'epyc',
    'radeon', 'instinct',
    'mi300', 'mi325', 'mi350',
    'fpga', 'xilinx',
    'lisa su',
    'data center', 'ai chip', 'ai accelerator',
    'santa clara'
]
PRIMARY_TERMS = TERMS[:2]

print("=" * 70)
print("SENTIMENT ANALYSIS - NEWS AMD (2 STRATEGIES)")
print("=" * 70)
print(f"\nInput:  {INPUT_FILE}")
print(f"Output: {OUTPUT_FILE_BASE}_S*.csv\n")

if not os.path.exists(INPUT_FILE):
    print(f"ERROR: file not found: {INPUT_FILE}")
    print(f"\n.csv files in {SCRIPT_DIR}:")
    for f in [f for f in os.listdir(SCRIPT_DIR) if f.endswith('.csv')]:
        print(f"    - {f}")
    exit(1)

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
    import torch
    print("Transformers: OK")
except ImportError:
    print("ERROR: transformers not available. Install: pip install transformers torch")
    exit(1)

device = 0 if torch.cuda.is_available() else -1
print(f"Device: {'GPU (CUDA)' if device == 0 else 'CPU'}")

# ============================== Load models ==============================

print("\n" + "=" * 70)
print("LOADING MODELS (2-5 minutes)")
print("=" * 70 + "\n")

MODEL_IDS = {
    'finbert':           "ProsusAI/finbert",
    'roberta_social':    "cardiffnlp/twitter-roberta-base-sentiment-latest",
    'roberta_stocks':    "zhayunduo/roberta-base-stocktwits-finetuned",
    'distilroberta_fin': "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis",
    'bert_sentiment':    "nlptown/bert-base-multilingual-uncased-sentiment",
}

models = {}
for i, (key, model_id) in enumerate(MODEL_IDS.items(), start=1):
    print(f"[{i}/5] {key} ({model_id})...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSequenceClassification.from_pretrained(model_id)
        models[key] = pipeline(
            "sentiment-analysis", model=model, tokenizer=tokenizer,
            device=device, max_length=512, truncation=True,
        )
        print("       OK")
    except Exception as e:
        print(f"       Error: {e}")
        models[key] = None

n_available = sum(1 for v in models.values() if v is not None)
print(f"\nModels loaded: {n_available}/5")
if n_available == 0:
    print("ERROR: no model available")
    exit(1)

MODEL_LIST = list(MODEL_IDS.keys())

# ============================== Text helpers ==============================

def count_term_mentions(text):
    """Count total occurrences of any company term."""
    if pd.isna(text):
        return 0
    text_lower = str(text).lower()
    return sum(text_lower.count(term) for term in TERMS)


def is_ticker_mentioned(text):
    """True if the ticker or company name appears explicitly."""
    if pd.isna(text):
        return False
    text_lower = str(text).lower()
    return any(p in text_lower for p in PRIMARY_TERMS)


def extract_term_sentences(text):
    """Keep only the sentences that mention a company term."""
    if pd.isna(text):
        return ""
    text = str(text).replace('U.S.', 'US').replace('U.K.', 'UK')
    sentences = re.split(r'[.!?]+', text)
    keep = [s.strip() for s in sentences if any(t in s.lower() for t in TERMS)]
    return ' '.join(keep)


def term_relevance(text):
    """How central the company is in the text (0-1): term ratio, capped."""
    if pd.isna(text):
        return 0.0
    words = str(text).lower().split()
    if not words:
        return 0.0
    hits = sum(str(text).lower().count(term) for term in TERMS)
    return min(hits / len(words) * 20, 1.0)


# ============================== Sentiment helpers ==============================

def normalize_label(label):
    """Map heterogeneous model labels to positive | negative | neutral."""
    label = str(label).lower()
    if any(x in label for x in ['pos', 'label_2', '5 star', '4 star']):
        return 'positive'
    elif any(x in label for x in ['neg', 'label_0', '1 star', '2 star']):
        return 'negative'
    return 'neutral'


def label_to_score(label):
    """Map a categorical label to a numeric score (-1, 0, 1)."""
    return {'positive': 1, 'neutral': 0, 'negative': -1}.get(label, 0)


def process_transformer(text, model_key):
    """Score a single text with one model."""
    if not models.get(model_key) or pd.isna(text) or str(text).strip() == "":
        return {'label': 'neutral', 'score': 0.33, 'confidence': 0.33}
    try:
        result = models[model_key](str(text)[:512])[0]
        return {
            'label_raw': result['label'],
            'label': normalize_label(result['label']),
            'score_raw': result['score'],
            'confidence': result['score'],
        }
    except Exception:
        return {'label': 'neutral', 'score': 0.33, 'confidence': 0.33}


def process_transformer_batch(texts, pipeline_obj, batch_size=BATCH_SIZE):
    """Run a transformer pipeline over texts in batches."""
    results = []
    for i in tqdm(range(0, len(texts), batch_size), desc="    Processing", leave=False):
        batch = texts[i:i + batch_size]
        try:
            results.extend(pipeline_obj(batch))
        except Exception:
            results.extend([{'label': 'neutral', 'score': 0.33}] * len(batch))
    return results


def weighted_sentiment(sent1, sent2, weight1=0.5):
    """Combine two sentiment dicts with weights (weight2 = 1 - weight1)."""
    weight2 = 1 - weight1
    combined_score = sent1['score'] * weight1 + sent2['score'] * weight2
    if combined_score > 0.05:
        combined_label = 'positive'
    elif combined_score < -0.05:
        combined_label = 'negative'
    else:
        combined_label = 'neutral'
    combined_conf = sent1['confidence'] * weight1 + sent2['confidence'] * weight2
    return {'label': combined_label, 'score': combined_score, 'confidence': combined_conf}


# ============================== Load data ==============================

print("\n" + "=" * 70)
print("LOADING DATA")
print("=" * 70 + "\n")

for enc in ('utf-8', 'latin-1', 'ISO-8859-1'):
    try:
        df = pd.read_csv(INPUT_FILE, encoding=enc)
        break
    except Exception:
        continue

print(f"News loaded: {len(df)}")

required_cols = ['Article_title', 'Lsa_summary']
missing = [c for c in required_cols if c not in df.columns]
if missing:
    print(f"ERROR: missing columns: {missing}")
    print(f"Available columns: {list(df.columns)}")
    exit(1)

df['Article_title'] = df['Article_title'].fillna("").astype(str)
df['Lsa_summary'] = df['Lsa_summary'].fillna("").astype(str)

# ============================== Context analysis ==============================

print("\n" + "=" * 70)
print("CONTEXT ANALYSIS")
print("=" * 70 + "\n")

df['title_has_ticker'] = df['Article_title'].apply(is_ticker_mentioned)
df['summary_term_mentions'] = df['Lsa_summary'].apply(count_term_mentions)
df['summary_term_relevance'] = df['Lsa_summary'].apply(term_relevance)
df['summary_term_sentences'] = df['Lsa_summary'].apply(extract_term_sentences)

print(f"Titles with explicit mention: {df['title_has_ticker'].sum()} ({df['title_has_ticker'].mean()*100:.1f}%)")
print(f"Mean mentions in summary: {df['summary_term_mentions'].mean():.2f}")
print(f"Mean relevance in summary: {df['summary_term_relevance'].mean():.3f}")
print(f"Summaries with >= 2 mentions: {(df['summary_term_mentions'] >= 2).sum()} ({(df['summary_term_mentions'] >= 2).mean()*100:.1f}%)")

# ============================== Strategy 1: title-only ==============================

print("\n" + "=" * 70)
print("STRATEGY 1: TITLE-ONLY")
print("=" * 70 + "\n")

texts = df['Article_title'].tolist()

for idx, model_key in enumerate(MODEL_LIST, 1):
    if models.get(model_key):
        print(f"[{idx}/5] {model_key}...")
        results = process_transformer_batch(texts, models[model_key])
        df[f's1_{model_key}_label_raw'] = [r['label'] for r in results]
        df[f's1_{model_key}_confidence'] = [r['score'] for r in results]
        df[f's1_{model_key}_label'] = df[f's1_{model_key}_label_raw'].apply(normalize_label)
        df[f's1_{model_key}_score'] = df[f's1_{model_key}_label'].apply(label_to_score)
        print("       Done")
    else:
        df[f's1_{model_key}_label'] = np.nan
        df[f's1_{model_key}_score'] = np.nan
        df[f's1_{model_key}_confidence'] = np.nan

print("\nStrategy 1 done")

# ============================== Strategy 2: smart-weighted ==============================

print("\n" + "=" * 70)
print("STRATEGY 2: SMART-WEIGHTED")
print("=" * 70 + "\n")

print("Weighting logic:")
print("    - ticker in title: 100% title")
print("    - not in title but >= 2 summary mentions: 30% title + 70% summary sentences")
print("    - otherwise: 100% title\n")


def get_sentiment_for_model(text, model_key):
    """Sentiment of a text for one model, with label/score normalized."""
    result = process_transformer(text, model_key)
    if 'label' not in result:
        result['label'] = normalize_label(result.get('label_raw', 'neutral'))
    result['score'] = label_to_score(result['label'])
    return result


def apply_smart_weighted_strategy(row, model_key):
    """Apply strategy 2 for one model and return (label, score, conf, method)."""
    title_sent = {
        'label': row[f's1_{model_key}_label'],
        'score': row[f's1_{model_key}_score'],
        'confidence': row[f's1_{model_key}_confidence'],
    }

    if row['title_has_ticker']:
        return title_sent['label'], title_sent['score'], title_sent['confidence'], 'title_only'

    if row['summary_term_mentions'] >= 2:
        term_text = row['summary_term_sentences']
        if term_text.strip() == "":
            return title_sent['label'], title_sent['score'], title_sent['confidence'], 'title_only'
        summary_sent = get_sentiment_for_model(term_text, model_key)
        combined = weighted_sentiment(title_sent, summary_sent, weight1=0.3)
        return combined['label'], combined['score'], combined['confidence'], 'weighted_30_70'

    return title_sent['label'], title_sent['score'], title_sent['confidence'], 'title_only'


for model_key in MODEL_LIST:
    if f's1_{model_key}_label' in df.columns and df[f's1_{model_key}_label'].notna().any():
        print(f"    Applying strategy for {model_key}...")
        results = df.apply(lambda row: apply_smart_weighted_strategy(row, model_key), axis=1)
        df[f's2_{model_key}_label'] = [r[0] for r in results]
        df[f's2_{model_key}_score'] = [r[1] for r in results]
        df[f's2_{model_key}_confidence'] = [r[2] for r in results]
        df[f's2_{model_key}_method'] = [r[3] for r in results]

print("\nStrategy 2 done")

if 's2_finbert_method' in df.columns:
    print("\nS2 method distribution:")
    for method, count in df['s2_finbert_method'].value_counts().items():
        print(f"    {method}: {count} ({count/len(df)*100:.1f}%)")

# ============================== Comparative statistics ==============================

print("\n" + "=" * 70)
print("COMPARATIVE STATISTICS BY STRATEGY")
print("=" * 70 + "\n")


def strategy_stats(strategy_num, model_key):
    """Return distribution/score stats for one strategy, or None."""
    label_col = f's{strategy_num}_{model_key}_label'
    score_col = f's{strategy_num}_{model_key}_score'
    if label_col not in df.columns or df[label_col].isna().all():
        return None
    dist = df[label_col].value_counts()
    total = dist.sum()
    stats = {f'{lbl}_pct': (dist.get(lbl, 0) / total * 100) if total > 0 else 0
             for lbl in ['positive', 'negative', 'neutral']}
    stats['mean_score'] = df[score_col].mean()
    stats['std_score'] = df[score_col].std()
    return stats


comparison_data = []
for model_key in ['finbert', 'roberta_social', 'distilroberta_fin']:
    if f's1_{model_key}_label' in df.columns:
        print(f"--- {model_key.upper()} ---")
        for strat_num in [1, 2]:
            stats = strategy_stats(strat_num, model_key)
            if stats:
                print(f"\n  Strategy {strat_num}:")
                print(f"      Positive: {stats['positive_pct']:.1f}%")
                print(f"      Negative: {stats['negative_pct']:.1f}%")
                print(f"      Neutral:  {stats['neutral_pct']:.1f}%")
                print(f"      Mean score: {stats['mean_score']:+.3f} (std: {stats['std_score']:.3f})")
                comparison_data.append({
                    'Model': model_key, 'Strategy': f'S{strat_num}',
                    'Positive%': stats['positive_pct'], 'Negative%': stats['negative_pct'],
                    'Neutral%': stats['neutral_pct'], 'Mean_Score': stats['mean_score'],
                    'Std_Score': stats['std_score'],
                })
        print()

# ============================== Save ==============================

print("=" * 70)
print("SAVING RESULTS (2 separate CSVs)")
print("=" * 70 + "\n")

original_cols = ['Date', 'Article_title', 'Lsa_summary', 'Stock_symbol']
for col in df.columns:
    if not col.startswith(('s1_', 's2_', 's3_', 'title_', 'summary_')) and col not in original_cols:
        original_cols.append(col)

context_cols = [c for c in ['title_has_ticker', 'summary_term_mentions',
                            'summary_term_relevance', 'summary_term_sentences'] if c in df.columns]

s1_cols = [c for c in df.columns if c.startswith('s1_')]
df_s1 = df[[c for c in original_cols if c in df.columns] + context_cols + s1_cols].copy()
output_s1 = OUTPUT_FILE_BASE + '_S1_title_only.csv'
df_s1.to_csv(output_s1, index=False)
print(f"S1 saved: {output_s1}  ({len(df_s1.columns)} cols, {len(df_s1)} rows)")

s2_cols = [c for c in df.columns if c.startswith('s2_')]
df_s2 = df[[c for c in original_cols if c in df.columns] + context_cols + s2_cols].copy()
output_s2 = OUTPUT_FILE_BASE + '_S2_smart_weighted.csv'
df_s2.to_csv(output_s2, index=False)
print(f"S2 saved: {output_s2}  ({len(df_s2.columns)} cols, {len(df_s2)} rows)")

if comparison_data:
    output_comparison = OUTPUT_FILE_BASE + '_comparison.csv'
    pd.DataFrame(comparison_data).to_csv(output_comparison, index=False)
    print(f"Comparison saved: {output_comparison}")

print("\n" + "=" * 70)
print("DONE")
print("=" * 70 + "\n")

print("Included models:")
for model_key in MODEL_LIST:
    print(f"    - {model_key}: {'OK' if models.get(model_key) else 'not available'}")
