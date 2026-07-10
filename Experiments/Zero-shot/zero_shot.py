# ============================================================
# HATE SPEECH CLASSIFICATION - ZERO-SHOT LEARNING
# GERMAN DATASETS: GERMEVAL, HASOC, GAHD
#
# Model: joeddav/xlm-roberta-large-xnli (local, via HF `transformers`
# zero-shot-classification pipeline, running on GPU)
#
# LOCAL-MODEL VERSION (replaces the previous SAIA-API version):
#   - No API keys / network calls for classification anymore -
#     the model runs locally, so all the API-key / round-robin /
#     thread-pool machinery has been removed.
#   - Classification is instead batched: all texts for a given
#     (dataset, strategy) combo are handed to the HF pipeline at
#     once, which internally batches them onto the GPU
#     (see BATCH_SIZE below). This is the "parallelism" now -
#     there's no I/O wait to hide behind threads, so threading
#     wouldn't help here.
#   - Batches spaCy NER with nlp.pipe instead of per-row .apply
#     (unchanged from before).
#   - Adds an "UNMASKED" strategy (alongside the 7 masking
#     strategies) that sends the raw, unmasked text - useful as
#     a baseline to compare all the masking strategies against.
#   - Runs on the FULL dataset when MAX_SAMPLES_PER_DATASET = None.
#
# No training happens here. Every document in every dataset is
# masked according to one of the 7 masking strategies (or left
# unmasked) and classified zero-shot against two candidate labels.
#
# Install once:
#   pip install transformers torch spacy pandas scikit-learn emoji
#   python -m spacy download de_core_news_lg
#
# NOTE: joeddav/xlm-roberta-large-xnli is a large (~2.2GB) model.
# Make sure you have a CUDA-capable GPU with enough free VRAM
# (a few GB is generally enough for inference at modest batch
# sizes). The script falls back to CPU automatically if no GPU
# is found, but that will be much slower.
# ============================================================

import os
import re
import random

import emoji
import numpy as np
import pandas as pd
import spacy
import torch

from transformers import pipeline

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

MODEL_NAME = "joeddav/xlm-roberta-large-xnli"

# German candidate labels + hypothesis template. xlm-roberta-large-xnli
# is multilingual, so we phrase the hypothesis in German to match the
# domain (German social media text) as closely as possible.
CANDIDATE_LABELS = ["Hassrede", "keine Hassrede"]
HYPOTHESIS_TEMPLATE = "Dieser Text ist {}."

# Set an integer (e.g. 200) while testing so you don't burn time
# on the full datasets. Set to None to run everything.
MAX_SAMPLES_PER_DATASET = None  # <- set to None for the full dataset

# How many texts the HF pipeline sends to the GPU at once. Raise
# this if you have spare VRAM (speeds things up); lower it if you
# hit out-of-memory errors.
BATCH_SIZE = 16

OUTPUT_DIR = "zero_shot_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# DEVICE SETUP + PIPELINE (runs on GPU if available)
# ============================================================

DEVICE = 0 if torch.cuda.is_available() else -1

if DEVICE == 0:
    print(f"CUDA available - using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA not available - falling back to CPU (this will be slow).")

print(f"Loading zero-shot classification model: {MODEL_NAME} ...")
classifier = pipeline(
    "zero-shot-classification",
    model=MODEL_NAME,
    device=DEVICE,
)
print("Model loaded.\n")

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
# SPACY / ENTITY EXTRACTION (batched for speed, unchanged)
# ============================================================

nlp = spacy.load(
    "de_core_news_lg",
    disable=["parser", "tagger", "lemmatizer"],
)

PERSON_LABELS = {"PER", "PERSON"}
ORG_LABELS = {"ORG"}
LOC_LABELS = {"LOC", "GPE"}
TARGET_LABELS = PERSON_LABELS | ORG_LABELS | LOC_LABELS

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
# MASKING STRATEGIES
#
# The 7 original masking strategies, PLUS a new "UNMASKED"
# strategy that sends the raw text through untouched. This acts
# as the baseline every masking strategy gets compared against.
# ============================================================

MASKING_STRATEGIES = [
    "UNMASKED",
    "PER_ORG_LOC_GENERIC_ENTITY",
    "PER_ONLY",
    "ORG_ONLY",
    "LOC_ONLY",
    "PER_ORG_LOC_TYPED",
    "X_LENGTH",
    "RANDOM_SUBSTITUTION",
]


def mask_text(text, entities, strategy, rng, entity_pools):

    # "UNMASKED" (or None) means: leave the text exactly as-is.
    if strategy is None or strategy == "UNMASKED":
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
# ZERO-SHOT CLASSIFICATION (local model, GPU-batched)
# ============================================================

def classify_batch_zero_shot(texts, batch_size=BATCH_SIZE):
    """
    Runs the local zero-shot-classification pipeline over a list
    of texts. The HF pipeline batches internally (via `batch_size`)
    when given a list, so the whole list is handed over in one call
    and the GPU (if available) does the batching/parallel work.

    Returns a list of (label, top_label_text, top_score) tuples in
    the SAME ORDER as the input texts, where `label` is 1 if the
    top-scoring candidate label is "Hassrede" (hate) and 0 otherwise.
    """

    if len(texts) == 0:
        return []

    raw_results = classifier(
        texts,
        candidate_labels=CANDIDATE_LABELS,
        hypothesis_template=HYPOTHESIS_TEMPLATE,
        batch_size=batch_size,
        multi_label=False,
    )

    # The pipeline returns a single dict if given a single string,
    # but we always pass a list, so this should already be a list -
    # this guard just makes the function robust either way.
    if isinstance(raw_results, dict):
        raw_results = [raw_results]

    parsed = []
    for res in raw_results:
        top_label = res["labels"][0]
        top_score = res["scores"][0]
        label = 1 if top_label == "Hassrede" else 0
        parsed.append((label, top_label, top_score))

    return parsed


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
            print(f"{dataset_name} | Strategy: {strategy} | N={len(df)} | batch_size={BATCH_SIZE}")
            print("=" * 70)

            masked_df = mask_dataframe(
                df,
                strategy,
                seed=SEED,
                entity_pools=entity_pools,
            )

            texts = masked_df["text"].tolist()
            results = classify_batch_zero_shot(texts, batch_size=BATCH_SIZE)

            y_true = masked_df["label"].tolist()
            y_pred = [label for label, _, _ in results]

            for i, row in enumerate(masked_df.itertuples(index=False)):
                label, top_label_text, top_score = results[i]
                all_predictions.append({
                    "Dataset": dataset_name,
                    "Strategy": strategy,
                    "Index": i,
                    "Text": row.text,
                    "TrueLabel": row.label,
                    "PredictedLabel": label,
                    "PredictedLabelText": top_label_text,
                    "Score": top_score,
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