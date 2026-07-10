# ============================================================
# HATE SPEECH CLASSIFICATION
# GERMAN DATASETS
#
# TRUE STRATIFIED K-FOLD CROSS VALIDATION
#
# For every fold:
#   • training data → masked
#   • validation data → original (unmasked)
#
# This fixes the methodological issue present in the original
# implementation where GERMEVAL and HASOC official splits never
# evaluated masking across different train/test partitions.
# ============================================================

# ============================================================
# IMPORTS
# ============================================================

import os
import re
import gc
import json
import random

import emoji
import numpy as np
import pandas as pd
import spacy
import torch

from datasets import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)

# ============================================================
# RANDOM SEEDS
# ============================================================

SEED = 42

random.seed(SEED)

np.random.seed(SEED)
torch.manual_seed(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Running on: {DEVICE}")

# ============================================================
# TRAINING PARAMETERS
# ============================================================

MAX_LENGTH = 128

BATCH_SIZE = 64

LEARNING_RATE = 2e-5

EPOCHS = 3

N_SPLITS = 5

# ============================================================
# MODELS
# ============================================================

MODELS = {
    "GBERT": "deepset/gbert-base",

    # Uncomment as desired

    "GELECTRA_BASE": "deepset/gelectra-base",

    "BERT_GERMAN": "bert-base-german-cased",

    "XLM_ROBERTA": "xlm-roberta-base",

    "XLM_ROBERTA_LARGE": "xlm-roberta-large",
}

# ============================================================
# SPECIAL TOKENS
# ============================================================

SPECIAL_TOKENS = {
    "additional_special_tokens": [
        "[PER]",
        "[ORG]",
        "[LOC]",
        "[ENTITY]",
    ]
}

# ============================================================
# MASKING STRATEGIES
# ============================================================

MASKING_STRATEGIES = [

    "UNMASKED",

    #"PER_ORG_LOC_GENERIC_ENTITY",

    #"PER_ONLY",

    #"ORG_ONLY",

    #"LOC_ONLY",

    #"PER_ORG_LOC_TYPED",

    #"X_LENGTH",

    #"RANDOM_SUBSTITUTION",
]

STRATEGY_TAGS = {

    "UNMASKED":
        "unmasked",

}

# ============================================================
# OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR = "cross_validation_results"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# LOAD SPACY MODEL
# ============================================================

nlp = spacy.load(
    "de_core_news_lg",
    disable=[
        "parser",
        "tagger",
        "lemmatizer",
    ],
)

PERSON_LABELS = {"PER", "PERSON"}

ORG_LABELS = {"ORG"}

LOC_LABELS = {"LOC", "GPE"}

TARGET_LABELS = PERSON_LABELS | ORG_LABELS | LOC_LABELS

# ============================================================
# TOKENIZATION CACHE
# ============================================================

TOKEN_CACHE = {}

print("Configuration loaded successfully.")


# ============================================================
# DATASET URLS
# ============================================================

BASE_URL = (
    "https://raw.githubusercontent.com/"
    "happy522/NE-Masking-for-DeBiasing-Text-Classification/"
    "refs/heads/main/Dataset"
)

DATASET_FILES = {
    "GERMEVAL": f"{BASE_URL}/germeval2018.csv",
    "HASOC":    f"{BASE_URL}/HASOC.csv",
    "GAHD":     f"{BASE_URL}/GAHD.csv",
}

# ============================================================
# TEXT CLEANING
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
# LOAD DATASETS
# ============================================================

DATASETS = {}

for name, path in DATASET_FILES.items():

    print(f"Loading {name}...")

    df = pd.read_csv(path)

    df = clean_dataframe(df)

    DATASETS[name] = df

    print(
        f"{name}: "
        f"{len(df):,} documents | "
        f"{df['label'].nunique()} classes"
    )

print("\nDatasets loaded successfully.")

# ============================================================
# DATASET SUMMARY
# ============================================================

summary = []

for name, df in DATASETS.items():

    counts = df["label"].value_counts().sort_index()

    summary.append({

        "Dataset": name,

        "Documents": len(df),

        "Class0": counts.get(0, 0),

        "Class1": counts.get(1, 0),

    })

summary_df = pd.DataFrame(summary)

print("\nDataset summary\n")

print(summary_df)

# ============================================================
# HUGGINGFACE DATASET CONVERSION
# ============================================================

def dataframe_to_dataset(df):

    return Dataset.from_pandas(

        df[["text", "label"]],

        preserve_index=False,

    )



# ============================================================
# ENTITY EXTRACTION
# ============================================================

def extract_entities(text):
    """
    Run spaCy once and return all named entities with character
    offsets so that any masking strategy can reuse them.
    """

    doc = nlp(text)

    entities = []

    for ent in doc.ents:

        entities.append({

            "start": ent.start_char,

            "end": ent.end_char,

            "text": ent.text,

            "label": ent.label_,

        })

    return entities


# ============================================================
# CACHE ENTITIES FOR ALL DATASETS
# ============================================================

print("\nExtracting named entities...")

for dataset_name, df in DATASETS.items():

    print(dataset_name)

    df["entities"] = df["text"].apply(extract_entities)

print("Entity extraction complete.")
# ============================================================
# BUILD ENTITY POOLS FROM TRAINING DATA ONLY
# ============================================================

def build_entity_pools(train_df):

    pools = {

        "PER": [],

        "ORG": [],

        "LOC": [],

    }

    seen = {

        "PER": set(),

        "ORG": set(),

        "LOC": set(),

    }

    for entity_list in train_df["entities"]:

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



# ============================================================
# RANDOM REPLACEMENT
# ============================================================

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
# APPLY MASKING STRATEGY
# ============================================================

def mask_text(
    text,
    entities,
    strategy,
    rng,
    entity_pools,
):

    if strategy is None:

        return text

    replacements = []

    for ent in entities:

        label = ent["label"]

        replacement = None

        # ----------------------------------------------------
        # Generic masking
        # ----------------------------------------------------

        if strategy == "PER_ORG_LOC_GENERIC_ENTITY":

            if label in TARGET_LABELS:

                replacement = "[ENTITY]"

        # ----------------------------------------------------
        # Individual masking
        # ----------------------------------------------------

        elif strategy == "PER_ONLY":

            if label in PERSON_LABELS:

                replacement = "[PER]"

        elif strategy == "ORG_ONLY":

            if label in ORG_LABELS:

                replacement = "[ORG]"

        elif strategy == "LOC_ONLY":

            if label in LOC_LABELS:

                replacement = "[LOC]"

        # ----------------------------------------------------
        # Typed masking
        # ----------------------------------------------------

        elif strategy == "PER_ORG_LOC_TYPED":

            if label in PERSON_LABELS:

                replacement = "[PER]"

            elif label in ORG_LABELS:

                replacement = "[ORG]"

            elif label in LOC_LABELS:

                replacement = "[LOC]"

        # ----------------------------------------------------
        # X masking
        # ----------------------------------------------------

        elif strategy == "X_LENGTH":

            if label in TARGET_LABELS:

                replacement = "X" * len(ent["text"])

        # ----------------------------------------------------
        # Random substitution
        # ----------------------------------------------------

        elif strategy == "RANDOM_SUBSTITUTION":

            if label in PERSON_LABELS:

                replacement = sample_entity(

                    ent["text"],

                    entity_pools["PER"],

                    rng,

                )

            elif label in ORG_LABELS:

                replacement = sample_entity(

                    ent["text"],

                    entity_pools["ORG"],

                    rng,

                )

            elif label in LOC_LABELS:

                replacement = sample_entity(

                    ent["text"],

                    entity_pools["LOC"],

                    rng,

                )

        if replacement is not None:

            replacements.append(

                (

                    ent["start"],

                    ent["end"],

                    replacement,

                )

            )

    if len(replacements) == 0:

        return text

    replacements.sort(

        key=lambda x: x[0],

        reverse=True,

    )

    result = text

    for start, end, replacement in replacements:

        result = (

            result[:start]

            + replacement

            + result[end:]

        )

    return result

# ============================================================
# MASK AN ENTIRE DATAFRAME
# ============================================================
# ============================================================
# MASK DATAFRAME
# ============================================================

def mask_dataframe(
    df,
    strategy,
    seed,
    entity_pools,
):

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
# STRATIFIED K-FOLD
# ============================================================

def generate_folds(df):
    """
    Generate reproducible stratified folds.

    Returns
    -------
    list[(train_df, test_df, fold_number)]
    """

    splitter = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=SEED,
    )

    folds = []

    X = df["text"]
    y = df["label"]

    for fold_number, (train_idx, test_idx) in enumerate(

        splitter.split(X, y),

        start=1,

    ):

        train_df = (

            df.iloc[train_idx]

            .copy()

            .reset_index(drop=True)

        )

        test_df = (

            df.iloc[test_idx]

            .copy()

            .reset_index(drop=True)

        )

        folds.append(

            (

                train_df,

                test_df,

                fold_number,

            )

        )

    return folds

# ============================================================
# PRECOMPUTE ALL FOLDS
# ============================================================

print("\nGenerating cross-validation folds...")

CV_FOLDS = {}

for dataset_name, df in DATASETS.items():

    CV_FOLDS[dataset_name] = generate_folds(df)

    print(

        f"{dataset_name}: "

        f"{len(CV_FOLDS[dataset_name])} folds"

    )

print("Cross-validation folds ready.")

# ============================================================
# APPLY MASKING TO TRAINING FOLD ONLY
# ============================================================

def prepare_fold(

    dataset_name,

    fold_number,

    masking_strategy,

):

    """
    Returns

        masked_train_df
        original_test_df

    for one fold.

    IMPORTANT

    Only the training data is masked.

    Validation remains untouched.
    """

    train_df, test_df, _ = CV_FOLDS[dataset_name][fold_number - 1]

    entity_pools = build_entity_pools(train_df)

    train_df = mask_dataframe(
        train_df,
        masking_strategy,
        seed=SEED + fold_number,
        entity_pools=entity_pools,
    )

    return train_df, test_df

# ============================================================
# QUICK SANITY CHECK
# ============================================================

for dataset_name in DATASETS:

    print()

    print(dataset_name)

    for train_df, test_df, fold in CV_FOLDS[dataset_name]:

        print(

            f"Fold {fold}: "

            f"train={len(train_df):5d} "

            f"test={len(test_df):5d}"

        )



# ============================================================
# TOKENIZER CACHE
# ============================================================

TOKEN_CACHE = {}

# ============================================================
# LOAD TOKENIZER
# ============================================================

def load_tokenizer(model_name):
    """
    Load tokenizer and add masking tokens.
    """

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    tokenizer.add_special_tokens(
        SPECIAL_TOKENS
    )

    return tokenizer


# ============================================================
# TOKENIZE DATASET
# ============================================================

def tokenize_dataset(

    dataframe,

    tokenizer,

    cache_key,

):
    """
    Tokenize a dataframe only once.

    Parameters
    ----------
    dataframe : pandas.DataFrame

    tokenizer : HuggingFace tokenizer

    cache_key : tuple
        Must uniquely identify the dataset
        (dataset, strategy, fold, split, model).
    """

    if cache_key in TOKEN_CACHE:

        return TOKEN_CACHE[cache_key]

    dataset = Dataset.from_pandas(

        dataframe[["text", "label"]],

        preserve_index=False,

    )

    dataset = dataset.map(

        lambda batch: tokenizer(

            batch["text"],

            truncation=True,

            padding="max_length",

            max_length=MAX_LENGTH,

        ),

        batched=True,

        desc="Tokenizing",

    )

    dataset = dataset.rename_column(

        "label",

        "labels",

    )

    dataset.set_format(

        type="torch",

        columns=[

            "input_ids",

            "attention_mask",

            "labels",

        ],

    )

    TOKEN_CACHE[cache_key] = dataset

    return dataset


# ============================================================
# PREPARE TOKENIZED FOLD
# ============================================================

def prepare_tokenized_fold(

    dataset_name,

    fold,

    strategy,

    tokenizer,

    model_key,

):
    """
    Returns

        train_dataset
        validation_dataset

    already tokenized.
    """

    train_df, valid_df = prepare_fold(

        dataset_name,

        fold,

        strategy,

    )

    train_key = (

        model_key,

        dataset_name,

        strategy,

        fold,

        "train",

    )

    valid_key = (

        model_key,

        dataset_name,

        "UNMASKED",

        fold,

        "validation",

    )

    train_dataset = tokenize_dataset(

        train_df,

        tokenizer,

        train_key,

    )

    validation_dataset = tokenize_dataset(

        valid_df,

        tokenizer,

        valid_key,

    )

    return train_dataset, validation_dataset

# ============================================================
# TRAIN ONE CROSS-VALIDATION FOLD
# ============================================================

def train_one_fold(
    model_name,
    tokenizer,
    train_dataset,
    validation_dataset,
):
    """
    Train one fold and return predictions.
    """

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
    )

    # Resize embeddings for masking tokens
    model.resize_token_embeddings(len(tokenizer))

    model.to(DEVICE)

    training_args = TrainingArguments(

        output_dir="./tmp",

        overwrite_output_dir=True,

        learning_rate=LEARNING_RATE,

        num_train_epochs=EPOCHS,

        per_device_train_batch_size=BATCH_SIZE,

        per_device_eval_batch_size=BATCH_SIZE,

        evaluation_strategy="no",

        save_strategy="no",

        logging_strategy="steps",

        logging_steps=50,

        report_to="none",

        seed=SEED,

        dataloader_num_workers=4,

        fp16=(
            torch.cuda.is_available()
            and not torch.cuda.is_bf16_supported()
        ),

        bf16=(
            torch.cuda.is_available()
            and torch.cuda.is_bf16_supported()
        ),
    )

    trainer = Trainer(

        model=model,

        args=training_args,

        train_dataset=train_dataset,

    )

    trainer.train()

    predictions = trainer.predict(validation_dataset)

    y_true = predictions.label_ids

    y_pred = np.argmax(

        predictions.predictions,

        axis=1,

    )

    del trainer
    del model
    TOKEN_CACHE.clear()

    gc.collect()

    

    if torch.cuda.is_available():

        torch.cuda.empty_cache()

    return y_true, y_pred


# ============================================================
# COMPUTE METRICS
# ============================================================

def compute_metrics(
    y_true,
    y_pred,
):

    report = classification_report(

        y_true,

        y_pred,

        output_dict=True,

        zero_division=0,

    )

    return {

        "Accuracy":
            accuracy_score(
                y_true,
                y_pred,
            ),

        "Precision_macro":
            precision_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            ),

        "Recall_macro":
            recall_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            ),

        "F1_macro":
            f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            ),

        "Weighted_F1":
            report["weighted avg"]["f1-score"],

        "Class0_Precision":
            report["0"]["precision"],

        "Class0_Recall":
            report["0"]["recall"],

        "Class0_F1":
            report["0"]["f1-score"],

        "Class0_Support":
            report["0"]["support"],

        "Class1_Precision":
            report["1"]["precision"],

        "Class1_Recall":
            report["1"]["recall"],

        "Class1_F1":
            report["1"]["f1-score"],

        "Class1_Support":
            report["1"]["support"],

        "Classification_Report":
            json.dumps(report),
    }


# ============================================================
# RUN A SINGLE FOLD
# ============================================================

def run_fold(
    dataset_name,
    strategy,
    model_key,
    model_name,
    fold,
):

    print()

    print("=" * 70)

    print(
        f"{dataset_name}"
        f" | Fold {fold}"
        f" | {strategy}"
        f" | {model_key}"
    )

    print("=" * 70)

    tokenizer = load_tokenizer(model_name)

    train_dataset, validation_dataset = prepare_tokenized_fold(

        dataset_name=dataset_name,

        fold=fold,

        strategy=strategy,

        tokenizer=tokenizer,

        model_key=model_key,

    )

    y_true, y_pred = train_one_fold(

        model_name,

        tokenizer,

        train_dataset,

        validation_dataset,

    )

    metrics = compute_metrics(

        y_true,

        y_pred,

    )

    result = {

        "Dataset": dataset_name,

        "Fold": fold,

        "Strategy": strategy,

        "Model": model_key,

    }

    result.update(metrics)

    print(

        f"Accuracy : {metrics['Accuracy']:.4f}"

    )

    print(

        f"Macro F1 : {metrics['F1_macro']:.4f}"

    )

    return result


# ============================================================
# RUN ALL CROSS-VALIDATION EXPERIMENTS
# ============================================================

all_results = []

for dataset_name in DATASETS.keys():

    print("\n")
    print("#" * 90)
    print(f"DATASET: {dataset_name}")
    print("#" * 90)

    for strategy in MASKING_STRATEGIES:

        print("\n")
        print("-" * 90)
        print(f"MASKING STRATEGY: {strategy}")
        print("-" * 90)

        for model_key, model_name in MODELS.items():

            print("\n")
            print(f"MODEL: {model_key}")

            tokenizer = load_tokenizer(model_name)

            for fold in range(1, N_SPLITS + 1):

                train_dataset, validation_dataset = prepare_tokenized_fold(

                    dataset_name=dataset_name,

                    fold=fold,

                    strategy=strategy,

                    tokenizer=tokenizer,

                    model_key=model_key,

                )

                y_true, y_pred = train_one_fold(

                    model_name=model_name,

                    tokenizer=tokenizer,

                    train_dataset=train_dataset,

                    validation_dataset=validation_dataset,

                )

                metrics = compute_metrics(

                    y_true,

                    y_pred,

                )

                row = {

                    "Dataset": dataset_name,

                    "Model": model_key,

                    "Strategy": strategy,

                    "Fold": fold,

                }

                row.update(metrics)

                all_results.append(row)

                print(

                    f"Fold {fold} | "

                    f"Accuracy={metrics['Accuracy']:.4f} | "

                    f"MacroF1={metrics['F1_macro']:.4f}"

                )

print("\nFinished all experiments.")



# ============================================================
# SAVE RESULTS
# ============================================================

print("\nSaving results...")

results_df = pd.DataFrame(all_results)

results_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "ALL_FOLDS_BACKUP.csv",
    ),
    index=False,
)

# ------------------------------------------------------------
# Save every fold
# ------------------------------------------------------------

all_folds_path = os.path.join(
    OUTPUT_DIR,
    "ALL_FOLDS.csv",
)

results_df.to_csv(
    all_folds_path,
    index=False,
)

print(f"Saved {all_folds_path}")

# ------------------------------------------------------------
# Save per-dataset fold results
# ------------------------------------------------------------

for dataset_name in DATASETS.keys():

    dataset_df = results_df[
        results_df["Dataset"] == dataset_name
    ]

    out_path = os.path.join(
        OUTPUT_DIR,
        f"{dataset_name}_ALL_FOLDS.csv",
    )

    dataset_df.to_csv(
        out_path,
        index=False,
    )

    print(f"Saved {out_path}")

# ============================================================
# SUMMARY STATISTICS
# ============================================================

summary = (

    results_df

    .groupby(

        [

            "Dataset",

            "Model",

            "Strategy",

        ],

        as_index=False,

    )

    .agg(

        Accuracy_Mean=("Accuracy", "mean"),
        Accuracy_SD=("Accuracy", "std"),

        Precision_Mean=("Precision_macro", "mean"),
        Precision_SD=("Precision_macro", "std"),

        Recall_Mean=("Recall_macro", "mean"),
        Recall_SD=("Recall_macro", "std"),

        F1_Mean=("F1_macro", "mean"),
        F1_SD=("F1_macro", "std"),

        WeightedF1_Mean=("Weighted_F1", "mean"),
        WeightedF1_SD=("Weighted_F1", "std"),

        Class0_F1_Mean=("Class0_F1", "mean"),
        Class0_F1_SD=("Class0_F1", "std"),

        Class1_F1_Mean=("Class1_F1", "mean"),
        Class1_F1_SD=("Class1_F1", "std"),

    )

)

summary = summary.sort_values(

    [

        "Dataset",

        "Model",

        "Strategy",

    ]

)

summary_path = os.path.join(

    OUTPUT_DIR,

    "SUMMARY.csv",

)

summary.to_csv(

    summary_path,

    index=False,

)

print(f"Saved {summary_path}")

# ------------------------------------------------------------
# Dataset summaries
# ------------------------------------------------------------

for dataset_name in DATASETS.keys():

    dataset_summary = summary[
        summary["Dataset"] == dataset_name
    ]

    out_path = os.path.join(

        OUTPUT_DIR,

        f"{dataset_name}_SUMMARY.csv",

    )

    dataset_summary.to_csv(

        out_path,

        index=False,

    )

    print(f"Saved {out_path}")

# ============================================================
# BEST MODEL PER DATASET
# ============================================================

best_models = (

    summary

    .sort_values(

        "F1_Mean",

        ascending=False,

    )

    .groupby(

        "Dataset",

        as_index=False,

    )

    .first()

)

best_path = os.path.join(

    OUTPUT_DIR,

    "BEST_MODELS.csv",

)

best_models.to_csv(

    best_path,

    index=False,

)

print(f"Saved {best_path}")

# ============================================================
# OVERALL RANKING
# ============================================================

overall = (

    summary

    .sort_values(

        "F1_Mean",

        ascending=False,

    )

)

overall_path = os.path.join(

    OUTPUT_DIR,

    "OVERALL_RANKING.csv",

)

overall.to_csv(

    overall_path,

    index=False,

)

print(f"Saved {overall_path}")

print()

print("=" * 80)

print("Cross-validation finished successfully.")

print("=" * 80)

print()

print("Top 10 configurations")

print(

    overall.head(10)[

        [

            "Dataset",

            "Model",

            "Strategy",

            "F1_Mean",

            "Accuracy_Mean",

        ]

    ]

)



