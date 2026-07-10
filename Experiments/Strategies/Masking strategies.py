#torchrun --nproc_per_node=4 test.py
# ============================================================
# HATE SPEECH CLASSIFICATION (GERMAN DATASETS)
# MULTI-STRATEGY ENTITY EXPERIMENTS
# ============================================================
#torchrun --nproc_per_node=4 test5_3.py 2>&1 | tee training_5_3.log

#set cuda device = 0
#CUDA_VISIBLE_DEVICES=0 nohup torchrun test5_8.py 2>&1 | tee training_5_8.log
#nohup python test5_7.py > training_5_7.log 2>&1 &

#watch -n 2 tail -n 20 training_5_7.log
# ============================================================
# 1. IMPORTS
# ============================================================

import os
import re
import json
import random

import numpy as np
import pandas as pd
import torch
import emoji
import spacy

from datasets import Dataset, concatenate_datasets
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

# ============================================================
# 2. CONFIG
# ============================================================

SEED = 42

MODELS = {
    #"BERT_GERMAN": "bert-base-german-cased",
    #"XLM_ROBERTA": "xlm-roberta-base",
    #"XLM_ROBERTA_LARGE": "xlm-roberta-large",
    #"GELECTRA_BASE": "deepset/gelectra-base",
    "GBERT": "deepset/gbert-base",
}

MAX_LENGTH = 128
BATCH_SIZE = 64
EPOCHS = 3
LR = 2e-5

# Use one GPU only if available
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# 3. SPACY NER
# ============================================================

# Install if needed:
#   pip install spacy
#   python -m spacy download de_core_news_lg

nlp = spacy.load("de_core_news_lg", disable=["parser", "tagger", "lemmatizer"])

PERSON_LABELS = {"PER", "PERSON"}
ORG_LABELS = {"ORG"}
LOC_LABELS = {"LOC", "GPE"}

TARGET_LABELS = PERSON_LABELS | ORG_LABELS | LOC_LABELS

# ============================================================
# 4. LOAD DATASETS
# ============================================================

germeval_raw = pd.read_csv(
    "https://raw.githubusercontent.com/happy522/NE-Masking-for-DeBiasing-Text-Classification/refs/heads/main/Dataset/germeval2018.csv"
)
hasoc_raw = pd.read_csv(
    "https://raw.githubusercontent.com/happy522/NE-Masking-for-DeBiasing-Text-Classification/refs/heads/main/Dataset/HASOC.csv"
)
gahd_raw = pd.read_csv(
    "https://raw.githubusercontent.com/happy522/NE-Masking-for-DeBiasing-Text-Classification/refs/heads/main/Dataset/GAHD.csv"
)
ger_train_df = pd.read_csv("https://raw.githubusercontent.com/happy522/NE-Masking-for-DeBiasing-Text-Classification/refs/heads/main/Dataset/germeval2018_train.csv")
ger_test_df = pd.read_csv("https://raw.githubusercontent.com/happy522/NE-Masking-for-DeBiasing-Text-Classification/refs/heads/main/Dataset/germeval2018_test.csv")
has_train_df = pd.read_csv("https://raw.githubusercontent.com/happy522/NE-Masking-for-DeBiasing-Text-Classification/refs/heads/main/Dataset/hasoc_german_train.csv")
has_test_df = pd.read_csv("https://raw.githubusercontent.com/happy522/NE-Masking-for-DeBiasing-Text-Classification/refs/heads/main/Dataset/hasoc_german_test.csv")

# ============================================================
# 5. TEXT CLEANING
# ============================================================

def clean_text(text):
    if pd.isna(text):
        return ""

    text = str(text)

    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = text.replace("@", "").replace("#", "")
    text = emoji.replace_emoji(text, replace="")
    text = re.sub(r"\s+", " ", text).strip()

    return text

def clean_dataframe(df):
    df = df.copy()
    df["text"] = df["text"].apply(clean_text)
    df["label"] = df["label"].astype(int)
    return df

# Clean once; masking happens later per strategy
germeval_clean = clean_dataframe(germeval_raw)
hasoc_clean = clean_dataframe(hasoc_raw)
gahd_clean = clean_dataframe(gahd_raw)
ger_train_df = clean_dataframe(ger_train_df)
ger_test_df = clean_dataframe(ger_test_df)
has_train_df = clean_dataframe(has_train_df)
has_test_df = clean_dataframe(has_test_df)

SOURCE_DFS = {
    "GERMEVAL": germeval_clean,
    "HASOC": hasoc_clean,
    "GAHD": gahd_clean,
    "GERMEVAL_TRAIN": ger_train_df,
    "GERMEVAL_TEST": ger_test_df,
    "HASOC_TRAIN": has_train_df,
    "HASOC_TEST": has_test_df
}

# ============================================================
# 6. SPLIT + DATASET HELPERS
# ============================================================

def split_df(df):
    return train_test_split(
        df,
        test_size=0.3,
        random_state=SEED,
        stratify=df["label"]
    )

def to_dataset(df):
    return Dataset.from_pandas(df[["text", "label"]], preserve_index=False)

def unique_preserve_order(items):
    seen = set()
    out = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out



# ============================================================
# 7. ENTITY POOLS FOR RANDOM SUBSTITUTION
# ============================================================

def build_entity_pools(dfs):
    """
    Build surface-form pools from the cleaned corpora.
    Used only for randomized substitution.
    """
    pools = {
        "PER": [],
        "ORG": [],
        "LOC": []
    }

    seen = {
        "PER": set(),
        "ORG": set(),
        "LOC": set()
    }

    all_texts = []
    for df in dfs:
        all_texts.extend(df["text"].tolist())

    for doc in nlp.pipe(all_texts, batch_size=32):
        for ent in doc.ents:
            ent_text = ent.text.strip()
            if not ent_text:
                continue

            if ent.label_ in PERSON_LABELS:
                key = "PER"
            elif ent.label_ in ORG_LABELS:
                key = "ORG"
            elif ent.label_ in LOC_LABELS:
                key = "LOC"
            else:
                continue

            if ent_text not in seen[key]:
                seen[key].add(ent_text)
                pools[key].append(ent_text)

    return pools

ENTITY_POOLS = build_entity_pools([germeval_clean, hasoc_clean, gahd_clean])

# ============================================================
# 8. MASKING / SUBSTITUTION STRATEGIES
# ============================================================

STRATEGIES = [
    #"None",                   # no masking, use cleaned text as-is
    "PER_ORG_LOC_GENERIC_ENTITY",      # replace PER/PERSON with [ENTITY]
    "PER_ONLY",                # replace PER with [PER]
    "ORG_ONLY",                # replace ORG with [ORG]
    "LOC_ONLY",                # replace LOC/GPE with [LOC]
    "PER_ORG_LOC_TYPED",       # replace PER-> [PER], ORG-> [ORG], LOC-> [LOC]
    "X_LENGTH",                # replace PER/ORG/LOC with X repeated char-length
    "RANDOM_SUBSTITUTION"      # replace with random entity from same type pool
]

STRATEGY_FILE_TAGS = {
    #"None": "none",
    "PER_ORG_LOC_GENERIC_ENTITY": "per_org_loc_generic_entity",
    "PER_ONLY": "per_only",
    "ORG_ONLY": "org_only",
    "LOC_ONLY": "loc_only",
    "PER_ORG_LOC_TYPED": "per_org_loc_typed",
    "X_LENGTH": "x_length",
    "RANDOM_SUBSTITUTION": "random_substitution",
}

def sample_replacement(current_text, pool, rng):
    if not pool:
        return current_text

    if len(pool) == 1:
        return pool[0]

    # Try a few times to avoid self-replacement when possible
    for _ in range(10):
        candidate = rng.choice(pool)
        if candidate != current_text:
            return candidate

    return rng.choice(pool)

def apply_strategy_to_text(text, strategy, rng, entity_pools):
    if not text:
        return text
    if strategy == "None":
        return text

    doc = nlp(text)
    spans = []

    for ent in doc.ents:
        replacement = None

        if strategy == "PER_ORG_LOC_GENERIC_ENTITY":
            if ent.label_ in PERSON_LABELS:
                replacement = "[ENTITY]"
            elif ent.label_ in ORG_LABELS:
                replacement = "[ENTITY]"
            elif ent.label_ in LOC_LABELS:
                replacement = "[ENTITY]"

        elif strategy == "PER_ONLY":
            if ent.label_ in PERSON_LABELS:
                replacement = "[PER]"

        elif strategy == "ORG_ONLY":
            if ent.label_ in ORG_LABELS:
                replacement = "[ORG]"

        elif strategy == "LOC_ONLY":
            if ent.label_ in LOC_LABELS:
                replacement = "[LOC]"

        elif strategy == "PER_ORG_LOC_TYPED":
            if ent.label_ in PERSON_LABELS:
                replacement = "[PER]"
            elif ent.label_ in ORG_LABELS:
                replacement = "[ORG]"
            elif ent.label_ in LOC_LABELS:
                replacement = "[LOC]"

        elif strategy == "X_LENGTH":
            if ent.label_ in TARGET_LABELS:
                replacement = "X" * len(ent.text)

        elif strategy == "RANDOM_SUBSTITUTION":
            if ent.label_ in PERSON_LABELS:
                replacement = sample_replacement(ent.text, entity_pools["PER"], rng)
            elif ent.label_ in ORG_LABELS:
                replacement = sample_replacement(ent.text, entity_pools["ORG"], rng)
            elif ent.label_ in LOC_LABELS:
                replacement = sample_replacement(ent.text, entity_pools["LOC"], rng)

        if replacement is not None:
            spans.append((ent.start_char, ent.end_char, replacement))

    # Reverse order so indices do not shift
    spans = sorted(spans, key=lambda x: x[0], reverse=True)

    result = text
    for start, end, replacement in spans:
        result = result[:start] + replacement + result[end:]

    return result

def transform_dataframe(df, strategy, rng, entity_pools):
    out = df.copy()
    out["text"] = out["text"].apply(
        lambda x: apply_strategy_to_text(x, strategy, rng, entity_pools)
    )
    out["label"] = out["label"].astype(int)
    return out

# ============================================================
# 9. TOKENIZATION
# ============================================================

def tokenize(batch, tokenizer):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH
    )

def prepare(ds, tokenizer):
    ds = ds.map(lambda x: tokenize(x, tokenizer), batched=True)
    ds = ds.rename_column("label", "labels")
    ds.set_format("torch")
    return ds

# ============================================================
# 10. BASE EXPERIMENT SETUP
# ============================================================

EXPERIMENTS = [
    # GERMEVAL AS SOURCE
    ("GERMEVAL", "train", "GERMEVAL", "test"),
    
    ("HASOC", "train", "HASOC", "test"),
    
    ("GAHD", "train", "GAHD", "test"),

    ("GERMEVAL", "full", "HASOC", "full"),
    ("GERMEVAL", "full", "GAHD", "full"),
    ("GERMEVAL", "full", "HASOC+GAHD", "HASOC+GAHD"),

    # HASOC AS SOURCE
    ("HASOC", "full", "GERMEVAL", "full"),
    ("HASOC", "full", "GAHD", "full"),
    ("HASOC", "full", "GERMEVAL+GAHD", "GERMEVAL+GAHD"),

    # GAHD AS SOURCE
    ("GAHD", "full", "GERMEVAL", "full"),
    ("GAHD", "full", "HASOC", "full"),
    ("GAHD", "full", "HASOC+GERMEVAL", "HASOC+GERMEVAL"),
]

def get_dataset(dataset_name, split, dataset_registry, combined_tests):
    if split in ["train", "test", "full"]:
        return dataset_registry[dataset_name][split]
    return combined_tests[split]

# ============================================================
# 11. TRAIN / EVAL FUNCTION
# ============================================================

def train_eval(model_name, train_ds, test_ds, train_name, test_name, strategy_name, results):
    special_tokens = {
        "additional_special_tokens": ["[PER]", "[ORG]", "[LOC]", "[ENTITY]"]
    }

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.add_special_tokens(special_tokens)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2
    )
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)

    train_ds_p = prepare(train_ds, tokenizer)
    test_ds_p = prepare(test_ds, tokenizer)

    args = TrainingArguments(
        output_dir="./results",
        learning_rate=LR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        evaluation_strategy="no",
        save_strategy="no",
        logging_steps=50,
        report_to="none",
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        dataloader_num_workers=4,
        seed=SEED,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds_p
    )

    trainer.train()

    preds = trainer.predict(test_ds_p)
    y_pred = np.argmax(preds.predictions, axis=1)
    y_true = preds.label_ids

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    report_dict = classification_report(
        y_true,
        y_pred,
        output_dict=True,
        zero_division=0
    )

    print("\n", model_name)
    print(classification_report(y_true, y_pred, zero_division=0))

    results.append({
        "Strategy": strategy_name,
        "Model": model_name,
        "Train": train_name,
        "Test": test_name,
        "Accuracy": acc,
        "Precision_macro": prec,
        "Recall_macro": rec,
        "F1_macro": f1,
        "Class0_Precision": report_dict["0"]["precision"],
        "Class0_Recall": report_dict["0"]["recall"],
        "Class0_F1": report_dict["0"]["f1-score"],
        "Class0_Support": report_dict["0"]["support"],
        "Class1_Precision": report_dict["1"]["precision"],
        "Class1_Recall": report_dict["1"]["recall"],
        "Class1_F1": report_dict["1"]["f1-score"],
        "Class1_Support": report_dict["1"]["support"],
        "Macro_F1": report_dict["macro avg"]["f1-score"],
        "Weighted_F1": report_dict["weighted avg"]["f1-score"],
        "Full_Report_JSON": json.dumps(report_dict)
    })

    del model
    del trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ============================================================
# 12. RUN ALL STRATEGIES
# ============================================================

all_strategy_results = []
os.makedirs("final_results", exist_ok=True)
for strategy_idx, strategy_name in enumerate(STRATEGIES):
    
    print("\n" + "#" * 110)
    print(f"MASKING STRATEGY: {strategy_name}")
    print("#" * 110)

    # Use a stable RNG per strategy
    rng = random.Random(SEED + (strategy_idx * 1000))

    transformed_ger_train = transform_dataframe(
        ger_train_df,
        strategy_name,
        rng,
        ENTITY_POOLS
    )

    transformed_ger_test = transform_dataframe(
        ger_test_df,
        strategy_name,
        rng,
        ENTITY_POOLS
    )

    transformed_has_train = transform_dataframe(
        has_train_df,
        strategy_name,
        rng,
        ENTITY_POOLS
    )

    transformed_has_test = transform_dataframe(
        has_test_df,
        strategy_name,
        rng,
        ENTITY_POOLS
    )



    # --------------------------------------------------------
    # APPLY STRATEGY TO FULL DATASETS
    # --------------------------------------------------------

    transformed_full = {
        "GERMEVAL": transform_dataframe(germeval_clean, strategy_name, rng, ENTITY_POOLS),
        "HASOC": transform_dataframe(hasoc_clean, strategy_name, rng, ENTITY_POOLS),
        "GAHD": transform_dataframe(gahd_clean, strategy_name, rng, ENTITY_POOLS),
    }

    # --------------------------------------------------------
    # CREATE STRATEGY-SPECIFIC SPLITS
    # --------------------------------------------------------

    #ger_train_df, ger_test_df = split_df(transformed_full["GERMEVAL"]) we already have official splits for GERMEVAL, so we will use those instead of re-splitting
    #has_train_df, has_test_df = split_df(transformed_full["HASOC"])
    gahd_train_df, gahd_test_df = split_df(transformed_full["GAHD"])

    # --------------------------------------------------------
    # HF DATASETS
    # --------------------------------------------------------

    dataset_registry = {
        "GERMEVAL": {
            "train": to_dataset(transformed_ger_train),
            "test": to_dataset(transformed_ger_test),
            "full": to_dataset(transformed_full["GERMEVAL"])
        },
        "HASOC": {
            "train": to_dataset(transformed_has_train),
            "test": to_dataset(transformed_has_test),
            "full": to_dataset(transformed_full["HASOC"])
        },
        "GAHD": {
            "train": to_dataset(gahd_train_df),
            "test": to_dataset(gahd_test_df),
            "full": to_dataset(transformed_full["GAHD"])
        },
    
        
    }

    combined_tests = {
        "HASOC+GAHD": concatenate_datasets(
            [dataset_registry["HASOC"]["full"], dataset_registry["GAHD"]["full"]]
        ),
        "GERMEVAL+GAHD": concatenate_datasets(
            [dataset_registry["GERMEVAL"]["full"], dataset_registry["GAHD"]["full"]]
        ),
        "HASOC+GERMEVAL": concatenate_datasets(
            [dataset_registry["HASOC"]["full"], dataset_registry["GERMEVAL"]["full"]]
        )
    }

    # --------------------------------------------------------
    # RUN MODELS
    # --------------------------------------------------------

    strategy_results = []

    for model_name in MODELS.values():
        print("\n" + "=" * 80)
        print(f"MODEL: {model_name}")
        print("=" * 80)

        for train_ds_name, train_split, test_ds_name, test_split in EXPERIMENTS:
            train_dataset = get_dataset(
                train_ds_name,
                train_split,
                dataset_registry,
                combined_tests
            )

            test_dataset = get_dataset(
                test_ds_name,
                test_split,
                dataset_registry,
                combined_tests
            )

            train_eval(
                model_name=model_name,
                train_ds=train_dataset,
                test_ds=test_dataset,
                train_name=f"{train_ds_name} ({train_split})",
                test_name=f"{test_ds_name} ({test_split})",
                strategy_name=strategy_name,
                results=strategy_results
            )

    # --------------------------------------------------------
    # SAVE PER-STRATEGY RESULTS
    # --------------------------------------------------------

    strategy_df = pd.DataFrame(strategy_results)
    strategy_df.to_csv(
        f"final_results/german_hate_speech_results_{STRATEGY_FILE_TAGS[strategy_name]}.csv",
        index=False
    )

    print("\nSaved:")
    print(f"final_results/german_hate_speech_results_{STRATEGY_FILE_TAGS[strategy_name]}.csv")

    all_strategy_results.append(strategy_df)

# ============================================================
# 13. SAVE MASTER RESULTS FILE
# ============================================================

master_results = pd.concat(all_strategy_results, ignore_index=True)

print("\nFINAL RESULTS")
print(master_results)

master_results.to_csv(
    "final_results/german_hate_speech_results_ALL_MASKING_STRATEGIES.csv",
    index=False
)

print("\nSaved:")
print("final_results/german_hate_speech_results_ALL_MASKING_STRATEGIES.csv")
