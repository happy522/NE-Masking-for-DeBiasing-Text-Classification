# ============================================================
# HATE SPEECH CLASSIFICATION - ZERO-SHOT LEARNING
# GERMAN DATASETS: GERMEVAL, HASOC, GAHD
#
# Model: mistral-large-3-675b-instruct-2512 via SAIA API
# (OpenAI-compatible endpoint, https://chat-ai.academiccloud.de/v1)
#
# PARALLEL VERSION:
#   - Uses 2 API keys round-robin to roughly double throughput
#   - Classifies documents concurrently with a thread pool
#     (this workload is I/O-bound: almost all the wall-clock
#     time is spent waiting on the API, not on local CPU, so
#     threads work fine here - no need for multiprocessing)
#   - Batches spaCy NER with nlp.pipe instead of per-row .apply
#   - Runs on the FULL dataset (MAX_SAMPLES_PER_DATASET = None)
#
# No training happens here. Every document in every dataset is
# masked according to one of the 7 strategies (or left unmasked)
# and sent straight to the LLM with a zero-shot prompt.
#
# Install once:
#   pip install openai spacy pandas scikit-learn emoji
#   python -m spacy download de_core_news_lg
# ============================================================

import os
import re
import json
import time
import random
import concurrent.futures

import emoji
import numpy as np
import pandas as pd
import spacy

from openai import OpenAI

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)

# ============================================================
# CONFIG
# ============================================================

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Put your keys in environment variables instead of hardcoding them:
#   export SAIA_API_KEY_1="your_first_key_here"
#   export SAIA_API_KEY_2="your_second_key_here"
# (Your original script had one key hardcoded as a fallback - avoid
# committing real keys to source, even for internal/research code.)
API_KEY_1 = os.environ.get("SAIA_API_KEY_1", "256a34825e9fbf26a3d9e31c096a0c86")
API_KEY_2 = os.environ.get("SAIA_API_KEY_2", "57445dacfe7afa67232c1b82ea596532")

API_BASE_URL = "https://openrouter.ai/api/v1"
MODEL_NAME = "mistralai/mistral-large"

_raw_keys = [API_KEY_1, API_KEY_2]
API_KEYS = [k for k in _raw_keys if k]

if len(API_KEYS) == 0:
    raise RuntimeError("No API keys found. Set SAIA_API_KEY_1 (and optionally SAIA_API_KEY_2).")

CLIENTS = [OpenAI(api_key=k, base_url=API_BASE_URL) for k in API_KEYS]
print(f"Using {len(CLIENTS)} API key(s) for classification.")

# Set an integer (e.g. 200) while testing so you don't burn time/quota
# on the full datasets. Set to None to run everything.
MAX_SAMPLES_PER_DATASET = 200  # <- full dataset, as requested

# Seconds to wait between API calls submitted by a single worker.
# Usually 0 is fine once you're parallelizing; raise it if you see
# rate-limit errors.
REQUEST_DELAY = 0.0

# Retries per document if the API call fails or the answer is unparseable
MAX_RETRIES = 5

# How many concurrent requests to run PER API KEY. Total concurrency
# = WORKERS_PER_KEY * len(CLIENTS). Start conservative (4-8) and raise
# it if the API tolerates more without throttling/erroring.
WORKERS_PER_KEY = 1
MAX_WORKERS = WORKERS_PER_KEY * len(CLIENTS)

OUTPUT_DIR = "zero_shot_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# DATASET URLS
# ============================================================

DATA_BASE_URL = (
    "https://raw.githubusercontent.com/"
    "happy522/NE-Masking-for-DeBiasing-Text-Classification/"
    "refs/heads/main/Dataset"
)

DATASET_FILES = {
    "GERMEVAL": f"{DATA_BASE_URL}/germeval2018.csv",
    "HASOC":    f"{DATA_BASE_URL}/HASOC.csv",
    "GAHD":     f"{DATA_BASE_URL}/GAHD.csv",
}

# ============================================================
# TEXT CLEANING (unchanged)
# ============================================================

URL_PATTERN = re.compile(r"http\S+|www\.\S+")
MULTISPACE_PATTERN = re.compile(r"\s+")


def clean_text(text):

    if pd.isna(text):
        return ""

    text = str(text)
    text = URL_PATTERN.sub("", text)
    text = text.replace("@", "")
    text = text.replace("#", "")
    text = emoji.replace_emoji(text, replace="")
    text = MULTISPACE_PATTERN.sub(" ", text)

    return text.strip()


def clean_dataframe(df):

    df = df.copy()

    if "text" not in df.columns:
        raise ValueError("Dataset must contain a 'text' column.")
    if "label" not in df.columns:
        raise ValueError("Dataset must contain a 'label' column.")

    df["text"] = df["text"].astype(str).apply(clean_text)
    df["label"] = df["label"].astype(int)
    df = df[df["text"].str.len() > 0]
    df = df.reset_index(drop=True)

    return df


# ============================================================
# SPACY / ENTITY EXTRACTION (batched for speed)
# ============================================================

nlp = spacy.load(
    "de_core_news_lg",
    disable=["parser", "tagger", "lemmatizer"],
)

PERSON_LABELS = {"PER", "PERSON"}
ORG_LABELS = {"ORG"}
LOC_LABELS = {"LOC", "GPE"}
TARGET_LABELS = PERSON_LABELS | ORG_LABELS | LOC_LABELS

# spaCy batch size for nlp.pipe - higher batches amortize per-call
# overhead. 128 is a reasonable default for short social media texts.
SPACY_BATCH_SIZE = 128
SPACY_N_PROCESS = 1  # bump if you have spare CPU cores and want multiprocessing in spaCy


def extract_entities_batch(texts):
    """
    Runs NER over a list of texts using nlp.pipe, which is
    substantially faster than calling nlp(text) once per row via
    df.apply, especially on the full dataset.
    """

    all_entities = []

    for doc in nlp.pipe(texts, batch_size=SPACY_BATCH_SIZE, n_process=SPACY_N_PROCESS):
        entities = []
        for ent in doc.ents:
            entities.append({
                "start": ent.start_char,
                "end": ent.end_char,
                "text": ent.text,
                "label": ent.label_,
            })
        all_entities.append(entities)

    return all_entities


# ============================================================
# ENTITY POOLS (unchanged)
#
# In the original CV setup these were built from the training
# fold only. There is no train/test split in zero-shot, so we
# build one pool per dataset from that dataset's own documents.
# ============================================================

def build_entity_pools(df):

    pools = {"PER": [], "ORG": [], "LOC": []}
    seen = {"PER": set(), "ORG": set(), "LOC": set()}

    for entity_list in df["entities"]:
        for ent in entity_list:

            text = ent["text"].strip()
            if not text:
                continue

            label = ent["label"]

            if label in PERSON_LABELS:
                key = "PER"
            elif label in ORG_LABELS:
                key = "ORG"
            elif label in LOC_LABELS:
                key = "LOC"
            else:
                continue

            if text not in seen[key]:
                seen[key].add(text)
                pools[key].append(text)

    return pools


def sample_entity(original, pool, rng):

    if len(pool) == 0:
        return original

    if len(pool) == 1:
        return pool[0]

    while True:
        candidate = rng.choice(pool)
        if candidate != original:
            return candidate


# ============================================================
# MASKING STRATEGIES - EXACTLY THE SAME 7 STRATEGIES (unchanged)
# ============================================================

MASKING_STRATEGIES = [
    "PER_ORG_LOC_GENERIC_ENTITY",
    "PER_ONLY",
    "ORG_ONLY",
    "LOC_ONLY",
    "PER_ORG_LOC_TYPED",
    "X_LENGTH",
    "RANDOM_SUBSTITUTION",
]


def mask_text(text, entities, strategy, rng, entity_pools):

    if strategy is None:
        return text

    replacements = []

    for ent in entities:

        label = ent["label"]
        replacement = None

        if strategy == "PER_ORG_LOC_GENERIC_ENTITY":
            if label in TARGET_LABELS:
                replacement = "[ENTITY]"

        elif strategy == "PER_ONLY":
            if label in PERSON_LABELS:
                replacement = "[PER]"

        elif strategy == "ORG_ONLY":
            if label in ORG_LABELS:
                replacement = "[ORG]"

        elif strategy == "LOC_ONLY":
            if label in LOC_LABELS:
                replacement = "[LOC]"

        elif strategy == "PER_ORG_LOC_TYPED":
            if label in PERSON_LABELS:
                replacement = "[PER]"
            elif label in ORG_LABELS:
                replacement = "[ORG]"
            elif label in LOC_LABELS:
                replacement = "[LOC]"

        elif strategy == "X_LENGTH":
            if label in TARGET_LABELS:
                replacement = "X" * len(ent["text"])

        elif strategy == "RANDOM_SUBSTITUTION":
            if label in PERSON_LABELS:
                replacement = sample_entity(ent["text"], entity_pools["PER"], rng)
            elif label in ORG_LABELS:
                replacement = sample_entity(ent["text"], entity_pools["ORG"], rng)
            elif label in LOC_LABELS:
                replacement = sample_entity(ent["text"], entity_pools["LOC"], rng)

        if replacement is not None:
            replacements.append((ent["start"], ent["end"], replacement))

    if len(replacements) == 0:
        return text

    replacements.sort(key=lambda x: x[0], reverse=True)

    result = text
    for start, end, replacement in replacements:
        result = result[:start] + replacement + result[end:]

    return result


def mask_dataframe(df, strategy, seed, entity_pools):

    rng = random.Random(seed)
    masked_texts = []

    for text, entities in zip(df["text"], df["entities"]):
        masked_texts.append(
            mask_text(
                text=text,
                entities=entities,
                strategy=strategy,
                rng=rng,
                entity_pools=entity_pools,
            )
        )

    masked_df = df.copy()
    masked_df["text"] = masked_texts

    return masked_df


# ============================================================
# ZERO-SHOT PROMPT + API CALL
# ============================================================

SYSTEM_PROMPT = (
    "You are a strict text classifier for German social media text. "
    "Decide whether the text is hate speech or not hate speech. "
    "Reply with exactly one word: 'hate' or 'nothate'. "
    "Do not explain your answer, do not add punctuation."
)


def build_user_prompt(text):
    return f'Text: "{text}"\n\nLabel (hate or nothate):'


def parse_label(response_text):
    """
    Turn the model's free-text reply into 0 (not hate) or 1 (hate).
    Returns None if it can't be parsed, so the caller can retry.
    """

    if response_text is None:
        return None

    answer = response_text.strip().lower()
    answer = re.sub(r"[^a-z]", "", answer)  # strip punctuation/whitespace

    if "nothate" in answer or answer.startswith("not"):
        return 0
    if "hate" in answer:
        return 1

    return None


def classify_text(text, client):
    """
    Calls the SAIA API for one document using the given client
    (so different threads can be pinned to different API keys).
    Retries on failure or on an unparseable answer. Falls back to
    label 0 if nothing works.
    """

    for attempt in range(MAX_RETRIES):

        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_prompt(text)},
                ],
                max_tokens=5,
                temperature=0.0,
            )

            raw_answer = response.choices[0].message.content
            label = parse_label(raw_answer)

            if label is not None:
                return label, raw_answer

        except Exception as e:
            print(f"  API error (attempt {attempt + 1}/{MAX_RETRIES}): {e}")
            time.sleep(2 * (attempt + 1))

    return 0, "PARSE_FAILED"


def classify_batch_parallel(texts, clients, max_workers):
    """
    Classifies a list of texts concurrently using a thread pool.
    Each task is assigned a client round-robin across the provided
    API keys/clients, so load is split roughly evenly between keys.

    Returns a list of (label, raw_answer) tuples in the SAME ORDER
    as the input texts (order is preserved even though completion
    order is not).
    """

    n = len(texts)
    results = [None] * n

    def _worker(idx):
        client = clients[idx % len(clients)]
        label, raw_answer = classify_text(texts[idx], client)
        if REQUEST_DELAY > 0:
            time.sleep(REQUEST_DELAY)
        return idx, label, raw_answer

    completed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_worker, i) for i in range(n)]

        for future in concurrent.futures.as_completed(futures):
            idx, label, raw_answer = future.result()
            results[idx] = (label, raw_answer)

            completed += 1
            if completed % 50 == 0 or completed == n:
                print(f"  {completed}/{n} documents classified")

    return results


# ============================================================
# METRICS (unchanged)
# ============================================================

def compute_metrics(y_true, y_pred):

    report = classification_report(
        y_true, y_pred, output_dict=True, zero_division=0,
    )

    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "Recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "F1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "Weighted_F1": report["weighted avg"]["f1-score"],
        "Class0_Precision": report.get("0", {}).get("precision", 0),
        "Class0_Recall": report.get("0", {}).get("recall", 0),
        "Class0_F1": report.get("0", {}).get("f1-score", 0),
        "Class0_Support": report.get("0", {}).get("support", 0),
        "Class1_Precision": report.get("1", {}).get("precision", 0),
        "Class1_Recall": report.get("1", {}).get("recall", 0),
        "Class1_F1": report.get("1", {}).get("f1-score", 0),
        "Class1_Support": report.get("1", {}).get("support", 0),
    }


# ============================================================
# LOAD DATASETS
# ============================================================

def load_and_prepare_datasets():

    datasets = {}

    for name, path in DATASET_FILES.items():

        print(f"Loading {name}...")
        df = pd.read_csv(path)
        df = clean_dataframe(df)

        if MAX_SAMPLES_PER_DATASET is not None:
            df = df.sample(
                n=min(MAX_SAMPLES_PER_DATASET, len(df)),
                random_state=SEED,
            ).reset_index(drop=True)

        print(f"Extracting named entities for {name} ({len(df)} docs)...")
        df["entities"] = extract_entities_batch(df["text"].tolist())

        datasets[name] = df
        print(f"{name}: {len(df)} documents ready\n")

    return datasets


# ============================================================
# MAIN ZERO-SHOT LOOP
# ============================================================

def run_zero_shot():

    datasets = load_and_prepare_datasets()

    all_predictions = []
    all_metrics = []

    for dataset_name, df in datasets.items():

        # Built from the whole dataset since there's no train/test split
        entity_pools = build_entity_pools(df)

        for strategy in MASKING_STRATEGIES:

            print("=" * 70)
            print(f"{dataset_name} | Strategy: {strategy} | N={len(df)} | workers={MAX_WORKERS}")
            print("=" * 70)

            masked_df = mask_dataframe(
                df,
                strategy,
                seed=SEED,
                entity_pools=entity_pools,
            )

            texts = masked_df["text"].tolist()
            results = classify_batch_parallel(texts, CLIENTS, MAX_WORKERS)

            y_true = masked_df["label"].tolist()
            y_pred = [label for label, _ in results]

            for i, row in enumerate(masked_df.itertuples(index=False)):
                label, raw_answer = results[i]
                all_predictions.append({
                    "Dataset": dataset_name,
                    "Strategy": strategy,
                    "Index": i,
                    "Text": row.text,
                    "TrueLabel": row.label,
                    "PredictedLabel": label,
                    "RawModelAnswer": raw_answer,
                })

            metrics = compute_metrics(y_true, y_pred)

            row_result = {
                "Dataset": dataset_name,
                "Strategy": strategy,
                "N": len(masked_df),
            }
            row_result.update(metrics)
            all_metrics.append(row_result)

            print(f"Accuracy: {metrics['Accuracy']:.4f} | Macro F1: {metrics['F1_macro']:.4f}\n")

    return all_predictions, all_metrics


# ============================================================
# SAVE RESULTS (unchanged)
# ============================================================

def save_results(all_predictions, all_metrics):

    predictions_df = pd.DataFrame(all_predictions)
    predictions_path = os.path.join(OUTPUT_DIR, "PREDICTIONS.csv")
    predictions_df.to_csv(predictions_path, index=False)
    print(f"Saved {predictions_path}")

    metrics_df = pd.DataFrame(all_metrics)
    metrics_path = os.path.join(OUTPUT_DIR, "RESULTS.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Saved {metrics_path}")

    for dataset_name in metrics_df["Dataset"].unique():
        dataset_metrics = metrics_df[metrics_df["Dataset"] == dataset_name]
        out_path = os.path.join(OUTPUT_DIR, f"{dataset_name}_RESULTS.csv")
        dataset_metrics.to_csv(out_path, index=False)
        print(f"Saved {out_path}")

    ranking = metrics_df.sort_values("F1_macro", ascending=False)
    ranking_path = os.path.join(OUTPUT_DIR, "RANKING_BY_F1.csv")
    ranking.to_csv(ranking_path, index=False)
    print(f"Saved {ranking_path}")

    print("\nTop configurations by Macro F1:")
    print(ranking[["Dataset", "Strategy", "F1_macro", "Accuracy"]].head(10).to_string(index=False))

    return metrics_df


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    predictions, metrics = run_zero_shot()
    save_results(predictions, metrics)

    print("\nZero-shot evaluation finished.")