"""
sentiment_comments_MSFT.py - Reddit comment sentiment for Microsoft (MSFT)

Scores cleaned Reddit comments with five transformer models:
    1. FinBERT (ProsusAI)             - financial text
    2. RoBERTa Social Media           - Twitter
    3. RoBERTa StockTwits             - trading forums
    4. DistilRoBERTa Financial        - financial news (lightweight)
    5. BERT Sentiment (nlptown)       - multilingual

Input:  comments_msft_clean.csv
Output: comments_msft_all_sentiments.csv
        Per model: {model}_label (positive|negative|neutral),
                   {model}_score (-1|0|1), {model}_confidence [0, 1]
"""

import os
import warnings

import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ============================== Configuration ==============================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(SCRIPT_DIR, "comments_msft_clean.csv")
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "comments_msft_all_sentiments.csv")
BATCH_SIZE = 32

print("=" * 70)
print("SENTIMENT ANALYSIS - REDDIT COMMENTS MSFT")
print("=" * 70)
print(f"\nInput:  {INPUT_FILE}")
print(f"Output: {OUTPUT_FILE}\n")

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

# ============================== Helpers ==============================

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


# ============================== Load data ==============================

print("\n" + "=" * 70)
print("LOADING DATA")
print("=" * 70 + "\n")

df = pd.read_csv(INPUT_FILE)
print(f"Comments loaded: {len(df)}")

df['body_clean'] = df['body'].fillna("").astype(str)
df['text_length'] = df['body_clean'].str.len()

# Drop very short comments (< 5 chars)
df_to_process = df[df['text_length'] >= 5].copy()
print(f"Comments to process (>= 5 chars): {len(df_to_process)}")

texts = df_to_process['body_clean'].tolist()

# ============================== Score with each model ==============================

print("\n" + "=" * 70)
print("SCORING SENTIMENT")
print("=" * 70 + "\n")

MODEL_NAMES = [
    ('finbert', 'FinBERT'),
    ('roberta_social', 'RoBERTa Social'),
    ('roberta_stocks', 'RoBERTa StockTwits'),
    ('distilroberta_fin', 'DistilRoBERTa Financial'),
    ('bert_sentiment', 'BERT Sentiment'),
]

for i, (key, name) in enumerate(MODEL_NAMES, start=1):
    if models.get(key):
        print(f"[{i}/5] {name}...")
        res = process_transformer_batch(texts, models[key])
        df_to_process[f'{key}_label_raw'] = [r['label'] for r in res]
        df_to_process[f'{key}_confidence'] = [r['score'] for r in res]
        df_to_process[f'{key}_label'] = df_to_process[f'{key}_label_raw'].apply(normalize_label)
        df_to_process[f'{key}_score'] = df_to_process[f'{key}_label'].apply(label_to_score)
        print("       Done")
    else:
        df_to_process[f'{key}_label'] = np.nan
        df_to_process[f'{key}_score'] = np.nan
        df_to_process[f'{key}_confidence'] = np.nan

# ============================== Merge back ==============================

print("\nMerging results with the original dataframe...")

result_columns = [col for col in df_to_process.columns if col not in df.columns]
if 'body' not in result_columns:
    result_columns = result_columns + ['body']

df_final = df.merge(df_to_process[result_columns], on='body', how='left')
print(f"Final records: {len(df_final)}")

# ============================== Statistics ==============================

print("\n" + "=" * 70)
print("STATISTICS PER MODEL")
print("=" * 70 + "\n")

models_stats = {}
for key, name in MODEL_NAMES:
    label_col, score_col, conf_col = f'{key}_label', f'{key}_score', f'{key}_confidence'
    if label_col in df_final.columns and df_final[label_col].notna().any():
        print(f"{name}:")
        dist = df_final[label_col].value_counts()
        total = dist.sum()
        print("    Distribution:")
        for label, count in dist.items():
            print(f"        {label:10s}: {count:6d} ({count / total * 100:5.1f}%)")
        mean_score = df_final[score_col].mean()
        std_score = df_final[score_col].std()
        mean_conf = df_final[conf_col].mean()
        print(f"    Mean score: {mean_score:+.3f} (std: {std_score:.3f})")
        print(f"    Mean confidence: {mean_conf:.3f}\n")
        models_stats[name] = {
            'positive_pct': (dist.get('positive', 0) / total * 100) if total > 0 else 0,
            'negative_pct': (dist.get('negative', 0) / total * 100) if total > 0 else 0,
            'neutral_pct': (dist.get('neutral', 0) / total * 100) if total > 0 else 0,
            'mean_score': mean_score,
            'std_score': std_score,
            'mean_confidence': mean_conf,
        }

# ============================== Comparison table ==============================

print("=" * 70)
print("COMPARISON TABLE")
print("=" * 70 + "\n")

comparison_df = None
if models_stats:
    comparison_df = pd.DataFrame(models_stats).T.round(3)
    print(comparison_df.to_string())
    print()

# ============================== Save ==============================

print("=" * 70)
print("SAVING RESULTS")
print("=" * 70 + "\n")

df_final.to_csv(OUTPUT_FILE, index=False)
print(f"CSV saved: {OUTPUT_FILE}")

comparison_file = None
if comparison_df is not None:
    comparison_file = OUTPUT_FILE.replace('.csv', '_comparison.csv')
    comparison_df.to_csv(comparison_file)
    print(f"Comparison saved: {comparison_file}")

print("\n" + "=" * 70)
print("DONE")
print("=" * 70 + "\n")

print("Generated files:")
print(f"    1. {os.path.basename(OUTPUT_FILE)}")
if comparison_file:
    print(f"    2. {os.path.basename(comparison_file)}")

print("\nIncluded models:")
for key, name in MODEL_NAMES:
    print(f"    - {name}: {'OK' if models.get(key) else 'not available'}")
