import gc
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

# Every model here is run through the FULL (dataset x strategy)
# evaluation, one at a time, in this order. Add/remove entries to
# change which models get compared. Keys are short display names
# used in the results tables; values are the HF model IDs.
MODELS_TO_RUN = {
    "xlm-roberta-large-xnli": "joeddav/xlm-roberta-large-xnli",
    "deberta-v3-nli": "MoritzLaurer/deberta-v3-large-zeroshot-v2.0",
}

# German candidate labels + hypothesis template. Both models above
# are multilingual/German-capable, so we phrase the hypothesis in
# German to match the domain (German social media text) as closely
# as possible.
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
# DEVICE SETUP
# ============================================================

DEVICE = 0 if torch.cuda.is_available() else -1

if DEVICE == 0:
    print(f"CUDA available - using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA not available - falling back to CPU (this will be slow).")


def load_classifier(model_path):
    """
    Loads a zero-shot-classification pipeline for the given HF model
    ID, on GPU if available. Call this once per model in
    MODELS_TO_RUN; the caller is responsible for freeing it (see
    unload_classifier) before loading the next one.
    """

    print(f"Loading zero-shot classification model: {model_path} ...")
    clf = pipeline(
        "zero-shot-classification",
        model=model_path,
        device=DEVICE,
    )
    print("Model loaded.\n")
    return clf


def unload_classifier(clf):
    """Frees GPU memory held by a loaded pipeline before loading the next one."""

    del clf
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


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
# The 7 original masking strategies, PLUS the "UNMASKED" strategy
# that sends the raw text through untouched. This acts as the
# baseline every masking strategy gets compared against.
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

def classify_batch_zero_shot(classifier, texts, batch_size=BATCH_SIZE):
    """
    Runs the given local zero-shot-classification pipeline over a
    list of texts. The HF pipeline batches internally (via
    `batch_size`) when given a list, so the whole list is handed
    over in one call and the GPU (if available) does the
    batching/parallel work.

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
# MAIN ZERO-SHOT LOOP (now: one outer pass per model)
# ============================================================

def run_zero_shot():

    # Datasets (+ NER) are computed ONCE and reused for every model.
    datasets = load_and_prepare_datasets()

    entity_pools_by_dataset = {
        dataset_name: build_entity_pools(df)
        for dataset_name, df in datasets.items()
    }

    all_predictions = []
    all_metrics = []

    for model_name, model_path in MODELS_TO_RUN.items():

        print("#" * 70)
        print(f"# MODEL: {model_name}  ({model_path})")
        print("#" * 70)

        classifier = load_classifier(model_path)

        for dataset_name, df in datasets.items():

            entity_pools = entity_pools_by_dataset[dataset_name]

            for strategy in MASKING_STRATEGIES:

                print("=" * 70)
                print(f"[{model_name}] {dataset_name} | Strategy: {strategy} | N={len(df)} | batch_size={BATCH_SIZE}")
                print("=" * 70)

                masked_df = mask_dataframe(
                    df,
                    strategy,
                    seed=SEED,
                    entity_pools=entity_pools,
                )

                texts = masked_df["text"].tolist()
                results = classify_batch_zero_shot(classifier, texts, batch_size=BATCH_SIZE)

                y_true = masked_df["label"].tolist()
                y_pred = [label for label, _, _ in results]

                for i, row in enumerate(masked_df.itertuples(index=False)):
                    label, top_label_text, top_score = results[i]
                    all_predictions.append({
                        "Model": model_name,
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
                    "Model": model_name,
                    "Dataset": dataset_name,
                    "Strategy": strategy,
                    "N": len(masked_df),
                }
                row_result.update(metrics)
                all_metrics.append(row_result)

                print(f"Accuracy: {metrics['Accuracy']:.4f} | Macro F1: {metrics['F1_macro']:.4f}\n")

        # Free GPU memory before loading the next model.
        unload_classifier(classifier)

    return all_predictions, all_metrics


# ============================================================
# SAVE RESULTS
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

    for (model_name, dataset_name), group in metrics_df.groupby(["Model", "Dataset"]):
        safe_model = re.sub(r"[^A-Za-z0-9_.-]+", "_", model_name)
        out_path = os.path.join(OUTPUT_DIR, f"{safe_model}_{dataset_name}_RESULTS.csv")
        group.to_csv(out_path, index=False)
        print(f"Saved {out_path}")

    ranking = metrics_df.sort_values("F1_macro", ascending=False)
    ranking_path = os.path.join(OUTPUT_DIR, "RANKING_BY_F1.csv")
    ranking.to_csv(ranking_path, index=False)
    print(f"Saved {ranking_path}")

    print("\nTop configurations by Macro F1 (across all models):")
    print(ranking[["Model", "Dataset", "Strategy", "F1_macro", "Accuracy"]].head(15).to_string(index=False))

    print("\nBest strategy per model (by mean Macro F1 across datasets):")
    best_per_model = (
        metrics_df.groupby(["Model", "Strategy"])["F1_macro"]
        .mean()
        .reset_index()
        .sort_values(["Model", "F1_macro"], ascending=[True, False])
    )
    print(best_per_model.groupby("Model").head(1).to_string(index=False))

    return metrics_df


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    predictions, metrics = run_zero_shot()
    save_results(predictions, metrics)

    print("\nZero-shot evaluation finished.")