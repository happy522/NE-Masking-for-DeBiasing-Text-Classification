# ============================================================
# HATE SPEECH CLASSIFICATION - DYNAMIC MIXED NE MASKING
# ============================================================
# Research condition implemented here:
#   - True stratified 5-fold cross-validation
#   - Training fold only receives masking/augmentation
#   - Validation fold always remains original/unmasked
#   - Every epoch contains ALL configured dynamic strategies
#   - Every training example receives EXACTLY ONE strategy per epoch
#   - Strategy assignments are balanced and reshuffled each epoch
#   - Epoch size stays N (NOT K*N)
#   - One continuous 3-epoch fine-tuning run per fold
#   - Random-substitution pools are built from the current training fold only
# ============================================================

import os
import re
import gc
import json
import random
from collections import Counter

import emoji
import numpy as np
import pandas as pd
import spacy
import torch
from torch.utils.data import Dataset as TorchDataset

from datasets import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    TrainerCallback,
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
# RANDOM SEEDS / REPRODUCIBILITY
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
# DYNAMIC MIXED STRATEGIES
# ============================================================
DYNAMIC_STRATEGIES = [
    "UNMASKED",
    "PER_ORG_LOC_GENERIC_ENTITY",
    "PER_ONLY",
    "ORG_ONLY",
    "LOC_ONLY",
    "PER_ORG_LOC_TYPED",
    "RANDOM_SUBSTITUTION",
    "X_LENGTH"
]

DYNAMIC_CONDITION_NAME = "DYNAMIC_MIXED"

ALL_IMPLEMENTED_STRATEGIES = {
    "UNMASKED",
    "PER_ORG_LOC_GENERIC_ENTITY",
    "PER_ONLY",
    "ORG_ONLY",
    "LOC_ONLY",
    "PER_ORG_LOC_TYPED",
    "X_LENGTH",
    "RANDOM_SUBSTITUTION",
}

if len(DYNAMIC_STRATEGIES) != 8:
    raise ValueError(
        "This experiment was specified for exactly 8 total strategies. "
        f"Currently configured: {len(DYNAMIC_STRATEGIES)}"
    )

unknown = set(DYNAMIC_STRATEGIES) - ALL_IMPLEMENTED_STRATEGIES
if unknown:
    raise ValueError(f"Unknown dynamic strategies: {sorted(unknown)}")

if len(set(DYNAMIC_STRATEGIES)) != len(DYNAMIC_STRATEGIES):
    raise ValueError("DYNAMIC_STRATEGIES contains duplicate entries.")

print("Dynamic strategies:")
for s in DYNAMIC_STRATEGIES:
    print(f"  - {s}")


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR = "cross_validation_results_dynamic_mixed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

ASSIGNMENT_LOG_PATH = os.path.join(
    OUTPUT_DIR,
    "DYNAMIC_ASSIGNMENT_COUNTS.csv",
)

SANITY_LOG_PATH = os.path.join(
    OUTPUT_DIR,
    "DYNAMIC_SANITY_EXAMPLES.csv",
)


# ============================================================
# LOAD SPACY MODEL
# ============================================================

nlp = spacy.load(
    "de_core_news_lg",
    disable=["parser", "tagger", "lemmatizer"],
)

PERSON_LABELS = {"PER", "PERSON"}
ORG_LABELS = {"ORG"}
LOC_LABELS = {"LOC", "GPE"}
TARGET_LABELS = PERSON_LABELS | ORG_LABELS | LOC_LABELS

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
    "HASOC": f"{BASE_URL}/HASOC.csv",
    "GAHD": f"{BASE_URL}/GAHD.csv",
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
        f"{name}: {len(df):,} documents | "
        f"{df['label'].nunique()} classes"
    )

print("\nDatasets loaded successfully.")


# ============================================================
# ENTITY EXTRACTION
# ============================================================

def extract_entities(text):
    """Run spaCy once and cache entities with character offsets."""
    doc = nlp(text)
    entities = []

    for ent in doc.ents:
        entities.append(
            {
                "start": ent.start_char,
                "end": ent.end_char,
                "text": ent.text,
                "label": ent.label_,
            }
        )

    return entities


print("\nExtracting named entities...")
for dataset_name, df in DATASETS.items():
    print(dataset_name)
    df["entities"] = df["text"].apply(extract_entities)
print("Entity extraction complete.")


# ============================================================
# BUILD ENTITY POOLS FROM CURRENT TRAINING FOLD ONLY
# ============================================================

def build_entity_pools(train_df):
    pools = {"PER": [], "ORG": [], "LOC": []}
    seen = {"PER": set(), "ORG": set(), "LOC": set()}

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

    # Finite deterministic alternative to an unbounded while loop.
    candidates = [x for x in pool if x != original]
    if not candidates:
        return original

    return rng.choice(candidates)


# ============================================================
# APPLY ONE MASKING STRATEGY TO ONE ORIGINAL TEXT
# ============================================================

def mask_text(text, entities, strategy, rng, entity_pools):
    if strategy == "UNMASKED":
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
                replacement = sample_entity(
                    ent["text"], entity_pools["PER"], rng
                )
            elif label in ORG_LABELS:
                replacement = sample_entity(
                    ent["text"], entity_pools["ORG"], rng
                )
            elif label in LOC_LABELS:
                replacement = sample_entity(
                    ent["text"], entity_pools["LOC"], rng
                )

        else:
            raise ValueError(f"Unknown masking strategy: {strategy}")

        if replacement is not None:
            replacements.append(
                (ent["start"], ent["end"], replacement)
            )

    if len(replacements) == 0:
        return text

    # Replace from right to left so cached character offsets remain valid.
    replacements.sort(key=lambda x: x[0], reverse=True)

    result = text
    for start, end, replacement in replacements:
        result = result[:start] + replacement + result[end:]

    return result


# ============================================================
# STRATIFIED K-FOLD
# ============================================================

def generate_folds(df):
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
            df.iloc[train_idx].copy().reset_index(drop=True)
        )
        valid_df = (
            df.iloc[test_idx].copy().reset_index(drop=True)
        )

        # Stable IDs are useful for logging/debugging.
        train_df["example_id"] = np.arange(len(train_df), dtype=int)
        valid_df["example_id"] = np.arange(len(valid_df), dtype=int)

        folds.append((train_df, valid_df, fold_number))

    return folds


print("\nGenerating cross-validation folds...")
CV_FOLDS = {}
for dataset_name, df in DATASETS.items():
    CV_FOLDS[dataset_name] = generate_folds(df)
    print(f"{dataset_name}: {len(CV_FOLDS[dataset_name])} folds")
print("Cross-validation folds ready.")


# ============================================================
# TOKENIZER
# ============================================================

def load_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=False,
    )
    tokenizer.add_special_tokens(SPECIAL_TOKENS)
    return tokenizer

# ============================================================
# TOKENIZE VALIDATION DATASET ONCE - ALWAYS UNMASKED
# ============================================================

def tokenize_validation_dataset(valid_df, tokenizer):
    dataset = Dataset.from_pandas(
        valid_df[["text", "label"]],
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
        desc="Tokenizing unmasked validation set",
    )

    dataset = dataset.rename_column("label", "labels")
    dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"],
    )
    return dataset


# ============================================================
# DETERMINISTIC SEED HELPERS
# ============================================================

def stable_dataset_code(dataset_name):
    # Avoid Python's process-randomized hash().
    return sum((i + 1) * ord(ch) for i, ch in enumerate(dataset_name))


def assignment_seed(dataset_name, fold, epoch):
    return (
        SEED
        + stable_dataset_code(dataset_name) * 100_000
        + fold * 1_000
        + epoch
    )


def replacement_seed(dataset_name, fold, epoch, example_index):
    return (
        SEED
        + stable_dataset_code(dataset_name) * 10_000_000
        + fold * 100_000
        + epoch * 10_000
        + example_index
    )


# ============================================================
# BALANCED STRATEGY ASSIGNMENT
# ============================================================

def generate_balanced_strategy_assignments(
    n_examples,
    strategies,
    seed,
):
    """
    Create one strategy per example such that strategy counts differ
    by at most one, then shuffle assignments reproducibly.
    """
    if n_examples <= 0:
        raise ValueError("Training fold is empty.")
    if len(strategies) == 0:
        raise ValueError("No dynamic strategies were provided.")

    rng = random.Random(seed)
    k = len(strategies)
    base_count = n_examples // k
    remainder = n_examples % k

    assignments = []

    # Every strategy gets the same base count.
    for strategy in strategies:
        assignments.extend([strategy] * base_count)

    # Randomize which strategies receive the at-most-one extra item.
    extra_strategies = list(strategies)
    rng.shuffle(extra_strategies)
    assignments.extend(extra_strategies[:remainder])

    # Randomize which examples receive which strategies.
    rng.shuffle(assignments)

    assert len(assignments) == n_examples

    counts = Counter(assignments)
    if max(counts.values()) - min(counts.values()) > 1:
        raise AssertionError("Dynamic strategy allocation is not balanced.")

    return assignments


# ============================================================
# LOG HELPERS
# ============================================================

def append_rows_to_csv(path, rows):
    if not rows:
        return

    df = pd.DataFrame(rows)
    write_header = not os.path.exists(path)
    df.to_csv(path, mode="a", header=write_header, index=False)


# ============================================================
# DYNAMIC MASKING PYTORCH DATASET
# ============================================================

class DynamicMaskingDataset(TorchDataset):
    """
    Stores ORIGINAL cleaned text and cached entities.

    At each epoch:
      1) a balanced strategy assignment is generated;
      2) each stable example index gets exactly one strategy;
      3) __getitem__ starts from the ORIGINAL text;
      4) masking is applied on-the-fly;
      5) transformed text is tokenized on-the-fly.

    IMPORTANT:
    This dataset is used with dataloader_num_workers=0 so callback-driven
    epoch state is guaranteed to be visible to __getitem__.
    """

    def __init__(
        self,
        dataframe,
        tokenizer,
        entity_pools,
        strategies,
        dataset_name,
        fold,
        model_key,
        sanity_example_ids=None,
    ):
        self.df = dataframe.reset_index(drop=True).copy()
        self.tokenizer = tokenizer
        self.entity_pools = entity_pools
        self.strategies = list(strategies)
        self.dataset_name = dataset_name
        self.fold = int(fold)
        self.model_key = model_key
        self.current_epoch = None
        self.current_assignments = None
        self.current_assignment_seed = None

        if sanity_example_ids is None:
            sanity_example_ids = [0, 1, 2]

        self.sanity_example_ids = {
            int(i)
            for i in sanity_example_ids
            if 0 <= int(i) < len(self.df)
        }
        self._logged_sanity_pairs = set()

        # Initialize epoch 1 before Trainer creates/iterates the dataloader.
        self.set_epoch(1, log=False)

    def __len__(self):
        return len(self.df)

    def set_epoch(self, epoch, log=True):
        epoch = int(epoch)

        if epoch < 1 or epoch > EPOCHS:
            raise ValueError(
                f"Epoch must be in [1, {EPOCHS}], received {epoch}."
            )

        seed = assignment_seed(
            self.dataset_name,
            self.fold,
            epoch,
        )

        assignments = generate_balanced_strategy_assignments(
            n_examples=len(self.df),
            strategies=self.strategies,
            seed=seed,
        )

        self.current_epoch = epoch
        self.current_assignment_seed = seed
        self.current_assignments = assignments

        if log:
            counts = Counter(assignments)

            print(
                f"\n[DYNAMIC] Dataset={self.dataset_name} | "
                f"Model={self.model_key} | Fold={self.fold} | "
                f"Epoch={epoch} | Seed={seed}"
            )

            print(
                f"[DYNAMIC] Training examples={len(self.df):,} | "
                f"Strategies={len(self.strategies)}"
            )

            count_rows = []
            for strategy in self.strategies:
                count = counts[strategy]
                print(f"    {strategy:32s} {count:6d}")

                count_rows.append(
                    {
                        "Dataset": self.dataset_name,
                        "Model": self.model_key,
                        "Fold": self.fold,
                        "Epoch": epoch,
                        "Strategy": strategy,
                        "Count": count,
                        "Seed": seed,
                        "Training_Samples": len(self.df),
                    }
                )

            append_rows_to_csv(ASSIGNMENT_LOG_PATH, count_rows)

    def get_strategy(self, index):
        if self.current_assignments is None:
            raise RuntimeError("Dynamic assignments have not been initialized.")
        return self.current_assignments[index]

    def _transform_text(self, index):
        row = self.df.iloc[index]
        original_text = row["text"]
        entities = row["entities"]
        strategy = self.get_strategy(index)

        # Example-specific RNG: independent of dataloader order.
        repl_seed = replacement_seed(
            self.dataset_name,
            self.fold,
            self.current_epoch,
            index,
        )
        rng = random.Random(repl_seed)

        transformed_text = mask_text(
            text=original_text,
            entities=entities,
            strategy=strategy,
            rng=rng,
            entity_pools=self.entity_pools,
        )

        # Small deterministic sanity log.
        key = (self.current_epoch, index)
        if (
            index in self.sanity_example_ids
            and key not in self._logged_sanity_pairs
        ):
            self._logged_sanity_pairs.add(key)
            append_rows_to_csv(
                SANITY_LOG_PATH,
                [
                    {
                        "Dataset": self.dataset_name,
                        "Model": self.model_key,
                        "Fold": self.fold,
                        "Example_ID": int(row["example_id"]),
                        "Epoch": self.current_epoch,
                        "Original_Text": original_text,
                        "Strategy": strategy,
                        "Transformed_Text": transformed_text,
                        "Assignment_Seed": self.current_assignment_seed,
                        "Replacement_Seed": repl_seed,
                    }
                ],
            )

        return transformed_text, int(row["label"])

    def __getitem__(self, index):
        transformed_text, label = self._transform_text(index)

        encoded = self.tokenizer(
            transformed_text,
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )

        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }


# ============================================================
# CALLBACK: RESHUFFLE BALANCED MASKING AT EACH EPOCH
# ============================================================

class DynamicMaskingCallback(TrainerCallback):
    def __init__(self, train_dataset):
        self.train_dataset = train_dataset
        self._last_epoch_set = None

    def on_epoch_begin(self, args, state, control, **kwargs):
        # At epoch start state.epoch is normally 0.0, 1.0, 2.0 ...
        epoch_number = int(state.epoch or 0) + 1

        # Protect against duplicate callback calls.
        if epoch_number != self._last_epoch_set:
            self.train_dataset.set_epoch(epoch_number, log=True)
            self._last_epoch_set = epoch_number

        return control


# ============================================================
# SANITY CHECKS FOR ONE DYNAMIC DATASET
# ============================================================

def run_dynamic_sanity_checks(dynamic_dataset, valid_df):
    n = len(dynamic_dataset)
    k = len(dynamic_dataset.strategies)

    print("\nRunning dynamic-masking sanity checks...")

    # 1. Validation untouched by construction.
    if "text" not in valid_df.columns:
        raise AssertionError("Validation dataframe lost its original text column.")

    # 2. Epoch size remains N.
    assert len(dynamic_dataset) == n

    # 3. Balanced assignments per epoch.
    assignments_by_epoch = []

    for epoch in range(1, EPOCHS + 1):
        dynamic_dataset.set_epoch(epoch, log=False)
        assignments = list(dynamic_dataset.current_assignments)
        assignments_by_epoch.append(assignments)

        counts = Counter(assignments)
        assert len(assignments) == n
        assert set(counts.keys()) == set(dynamic_dataset.strategies)
        assert max(counts.values()) - min(counts.values()) <= 1

    # 4. Assignments should change across epochs for nontrivial datasets.
    if n > k:
        for e1 in range(EPOCHS):
            for e2 in range(e1 + 1, EPOCHS):
                if assignments_by_epoch[e1] == assignments_by_epoch[e2]:
                    raise AssertionError(
                        f"Epoch {e1 + 1} and epoch {e2 + 1} have identical "
                        "strategy assignments."
                    )

    # 5. Reproducibility: same seed -> same assignment.
    test_seed = assignment_seed(
        dynamic_dataset.dataset_name,
        dynamic_dataset.fold,
        1,
    )
    a1 = generate_balanced_strategy_assignments(
        n, dynamic_dataset.strategies, test_seed
    )
    a2 = generate_balanced_strategy_assignments(
        n, dynamic_dataset.strategies, test_seed
    )
    assert a1 == a2

    # Restore epoch 1 before training starts.
    dynamic_dataset.set_epoch(1, log=False)

    print(
        "Sanity checks passed: "
        "N preserved, strategies balanced, assignments vary by epoch, "
        "and assignment generation is reproducible."
    )


# ============================================================
# TRAIN ONE DYNAMIC-MIXED CROSS-VALIDATION FOLD
# ============================================================

def train_one_dynamic_fold(
    dataset_name,
    model_key,
    model_name,
    fold,
):
    print("\n" + "=" * 90)
    print(
        f"{dataset_name} | {model_key} | Fold {fold} | "
        f"{DYNAMIC_CONDITION_NAME}"
    )
    print("=" * 90)

    # Retrieve the exact same precomputed CV partition.
    train_df, valid_df, _ = CV_FOLDS[dataset_name][fold - 1]

    # Leakage protection: pools are created ONLY from current training fold.
    entity_pools = build_entity_pools(train_df)

    tokenizer = load_tokenizer(model_name)

    dynamic_train_dataset = DynamicMaskingDataset(
        dataframe=train_df,
        tokenizer=tokenizer,
        entity_pools=entity_pools,
        strategies=DYNAMIC_STRATEGIES,
        dataset_name=dataset_name,
        fold=fold,
        model_key=model_key,
        sanity_example_ids=[0, 1, 2],
    )

    validation_dataset = tokenize_validation_dataset(
        valid_df,
        tokenizer,
    )

    run_dynamic_sanity_checks(
        dynamic_train_dataset,
        valid_df,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
    )

    # Required because masking special tokens were added to tokenizer.
    model.resize_token_embeddings(len(tokenizer))
    model.to(DEVICE)

    training_args = TrainingArguments(
        output_dir=os.path.join(
            OUTPUT_DIR,
            "tmp",
            dataset_name,
            model_key,
            f"fold_{fold}",
        ),
        overwrite_output_dir=True,
        learning_rate=LEARNING_RATE,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        eval_strategy="no",
        save_strategy="no",
        logging_strategy="steps",
        logging_steps=50,
        report_to="none",
        seed=SEED,

        # CRITICAL for callback-driven mutable epoch state:
        # no worker-local copies of the dataset.
        dataloader_num_workers=0,

        fp16=(
            torch.cuda.is_available()
            and not torch.cuda.is_bf16_supported()
        ),
        bf16=(
            torch.cuda.is_available()
            and torch.cuda.is_bf16_supported()
        ),
    )

    callback = DynamicMaskingCallback(dynamic_train_dataset)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dynamic_train_dataset,
        callbacks=[callback],
    )

    # One continuous training run. Model/optimizer/scheduler are NOT reset.
    trainer.train()

    predictions = trainer.predict(validation_dataset)

    y_true = predictions.label_ids
    y_pred = np.argmax(predictions.predictions, axis=1)

    metrics = compute_metrics(y_true, y_pred)

    result = {
        "Dataset": dataset_name,
        "Model": model_key,
        "Strategy": DYNAMIC_CONDITION_NAME,
        "Fold": fold,
    }
    result.update(metrics)

    print(
        f"Fold {fold} | Accuracy={metrics['Accuracy']:.4f} | "
        f"MacroF1={metrics['F1_macro']:.4f}"
    )

    del trainer
    del callback
    del model
    del dynamic_train_dataset
    del validation_dataset
    del tokenizer

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


# ============================================================
# METRICS
# ============================================================

def compute_metrics(y_true, y_pred):
    report = classification_report(
        y_true,
        y_pred,
        output_dict=True,
        zero_division=0,
    )

    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision_macro": precision_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "Recall_macro": recall_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "F1_macro": f1_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "Weighted_F1": report["weighted avg"]["f1-score"],
        "Class0_Precision": report["0"]["precision"],
        "Class0_Recall": report["0"]["recall"],
        "Class0_F1": report["0"]["f1-score"],
        "Class0_Support": report["0"]["support"],
        "Class1_Precision": report["1"]["precision"],
        "Class1_Recall": report["1"]["recall"],
        "Class1_F1": report["1"]["f1-score"],
        "Class1_Support": report["1"]["support"],
        "Classification_Report": json.dumps(report),
    }


# ============================================================
# CLEAN OLD DYNAMIC LOGS FOR THIS RUN
# ============================================================
# Prevent accidental duplicate log rows when rerunning the script.

for path in [ASSIGNMENT_LOG_PATH, SANITY_LOG_PATH]:
    if os.path.exists(path):
        os.remove(path)


# ============================================================
# RUN DYNAMIC-MIXED CROSS-VALIDATION ONLY
# ============================================================

all_results = []

for dataset_name in DATASETS.keys():
    print("\n" + "#" * 90)
    print(f"DATASET: {dataset_name}")
    print(f"EXPERIMENT: {DYNAMIC_CONDITION_NAME}")
    print("#" * 90)

    for model_key, model_name in MODELS.items():
        print(f"\nMODEL: {model_key}")

        for fold in range(1, N_SPLITS + 1):
            row = train_one_dynamic_fold(
                dataset_name=dataset_name,
                model_key=model_key,
                model_name=model_name,
                fold=fold,
            )
            all_results.append(row)


print("\nFinished all dynamic-mixed experiments.")


# ============================================================
# SAVE FOLD RESULTS
# ============================================================

results_df = pd.DataFrame(all_results)

all_folds_path = os.path.join(
    OUTPUT_DIR,
    "DYNAMIC_MIXED_ALL_FOLDS.csv",
)
results_df.to_csv(all_folds_path, index=False)
print(f"Saved {all_folds_path}")

for dataset_name in DATASETS.keys():
    dataset_df = results_df[
        results_df["Dataset"] == dataset_name
    ]

    out_path = os.path.join(
        OUTPUT_DIR,
        f"{dataset_name}_DYNAMIC_MIXED_ALL_FOLDS.csv",
    )
    dataset_df.to_csv(out_path, index=False)
    print(f"Saved {out_path}")


# ============================================================
# SUMMARY: MEAN +/- SD ACROSS FIVE FOLDS
# ============================================================

summary = (
    results_df
    .groupby(
        ["Dataset", "Model", "Strategy"],
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
    .sort_values(["Dataset", "Model", "Strategy"])
)

summary_path = os.path.join(
    OUTPUT_DIR,
    "DYNAMIC_MIXED_SUMMARY.csv",
)
summary.to_csv(summary_path, index=False)
print(f"Saved {summary_path}")


# ============================================================
# FINAL DISPLAY
# ============================================================

print("\n" + "=" * 90)
print("DYNAMIC MIXED CROSS-VALIDATION FINISHED")
print("=" * 90)

print(
    summary[
        [
            "Dataset",
            "Model",
            "Strategy",
            "F1_Mean",
            "F1_SD",
            "Accuracy_Mean",
        ]
    ]
)

print("\nMethodological checks:")
print("  - Validation is always original/unmasked.")
print("  - Every epoch has one transformed view per training example.")
print("  - All configured strategies are balanced within each epoch.")
print("  - Assignments are reshuffled deterministically each epoch.")
print("  - Training-fold entity pools only are used for random substitution.")
print("  - Model/optimizer/scheduler continue across all 3 epochs.")
