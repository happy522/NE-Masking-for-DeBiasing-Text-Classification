# Without entity masking, just to see the baseline performance of the models on the datasets and their combinations.
import os
import random
import numpy as np
import pandas as pd
import torch
import json
from datasets import Dataset, concatenate_datasets
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# ============================================================
# 2. CONFIG
# ============================================================

import os

# use ONLY one GPU
# change to "1", "2", "3" if needed
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

SEED = 42

MODELS = {
    "BERT_GERMAN": "bert-base-german-cased",
    "XLM_ROBERTA": "xlm-roberta-base",
    "XLM_ROBERTA_LARGE": "xlm-roberta-large",
    "GELECTRA_BASE": "deepset/gelectra-base",

    #note: if possible, run it seprately due to different architecture
    "GBERT": "deepset/gbert-base",
}

MAX_LENGTH = 128
BATCH_SIZE = 16
EPOCHS = 3
LR = 2e-5

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# 3. LOAD DATASETS
# ============================================================

germeval = pd.read_csv("https://raw.githubusercontent.com/happy522/NE-Masking-for-DeBiasing-Text-Classification/refs/heads/main/Dataset/germeval2018.csv")
hasoc = pd.read_csv("https://raw.githubusercontent.com/happy522/NE-Masking-for-DeBiasing-Text-Classification/refs/heads/main/Dataset/HASOC.csv")
gahd = pd.read_csv("https://raw.githubusercontent.com/happy522/NE-Masking-for-DeBiasing-Text-Classification/refs/heads/main/Dataset/GAHD.csv")

# ============================================================
# 4. CLEAN TEXT (NO ENTITY MASKING)
# ============================================================

import re
import emoji

def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text)
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = text.replace("@", "").replace("#", "")
    text = emoji.replace_emoji(text, replace="")
    text = re.sub(r"\s+", " ", text).strip()
    return text

for df in [germeval, hasoc, gahd]:
    df["text"] = df["text"].apply(clean_text)
    df["label"] = df["label"].astype(int)

# ============================================================
# 5. TRAIN-TEST SPLIT
# ============================================================

def split(df):
    return train_test_split(
        df,
        test_size=0.3,
        random_state=SEED,
        stratify=df["label"]
    )

ger_train, ger_test = split(germeval)
has_train, has_test = split(hasoc)
gahd_train, gahd_test = split(gahd)

# ============================================================
# 6. HF DATASET CONVERSION
# ============================================================

def to_dataset(df):
    return Dataset.from_pandas(df[["text", "label"]])

ger_full = to_dataset(germeval)
has_full = to_dataset(hasoc)
gahd_full = to_dataset(gahd)

ger_train = to_dataset(ger_train)
ger_test = to_dataset(ger_test)

has_train = to_dataset(has_train)
has_test = to_dataset(has_test)

gahd_train = to_dataset(gahd_train)
gahd_test = to_dataset(gahd_test)

DATASETS = {
    "GERMEVAL": {
        "train_70": ger_train,
        "test_30": ger_test,
        "full": ger_full
    },
    "HASOC": {
        "train_70": has_train,
        "test_30": has_test,
        "full": has_full
    },
    "GAHD": {
        "train_70": gahd_train,
        "test_30": gahd_test,
        "full": gahd_full
    }
}

hasoc_gahd = concatenate_datasets([has_full, gahd_full])
germeval_gahd = concatenate_datasets([ger_full, gahd_full])
hasoc_germeval = concatenate_datasets([has_full, ger_full])

COMBINED_TESTS = {
    "HASOC+GAHD": hasoc_gahd,
    "GERMEVAL+GAHD": germeval_gahd,
    "HASOC+GERMEVAL": hasoc_germeval
}

EXPERIMENTS = [

    # GERMEVAL AS SOURCE
    ("GERMEVAL", "train_70", "GERMEVAL", "test_30"),
    ("GERMEVAL", "full", "HASOC", "full"),
    ("GERMEVAL", "full", "GAHD", "full"),
    ("GERMEVAL", "full", "HASOC+GAHD", "HASOC+GAHD"),

    # HASOC AS SOURCE
    ("HASOC", "train_70", "HASOC", "test_30"),
    ("HASOC", "full", "GERMEVAL", "full"),
    ("HASOC", "full", "GAHD", "full"),
    ("HASOC", "full", "GERMEVAL+GAHD", "GERMEVAL+GAHD"),

    # GAHD AS SOURCE
    ("GAHD", "train_70", "GAHD", "test_30"),
    ("GAHD", "full", "GERMEVAL", "full"),
    ("GAHD", "full", "HASOC", "full"),
    ("GAHD", "full", "HASOC+GERMEVAL", "HASOC+GERMEVAL"),
]


def get_dataset(dataset_name, split):
    if split in ["train_70", "test_30", "full"]:
        return DATASETS[dataset_name][split]
    return COMBINED_TESTS[split]
# ============================================================
# 7. TOKENIZATION
# ============================================================

def tokenize(batch, tokenizer):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=MAX_LENGTH)

def prepare(ds, tokenizer):
    ds = ds.map(lambda x: tokenize(x, tokenizer), batched=True)
    ds = ds.rename_column("label", "labels")
    ds.set_format("torch")
    return ds

# ============================================================
# 8. TRAIN FUNCTION
# ============================================================

results = []

def train_eval(model_name, train_ds, test_ds, train_name, test_name):

    #tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(
       model_name,
       use_fast=False
    )
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    model.to(device)

    train_ds_p = prepare(train_ds, tokenizer)
    test_ds_p = prepare(test_ds, tokenizer)

    args = TrainingArguments(
        output_dir="./results",
        learning_rate=LR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        #eval_strategy="no",
	evaluation_strategy="no",
        save_strategy="no",
        logging_steps=50,
        report_to="none",
        fp16=torch.cuda.is_available(),
        seed=SEED
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
    report = classification_report(
        y_true,
        y_pred,
        output_dict=True
    )
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro")
    rec = recall_score(y_true, y_pred, average="macro")
    f1 = f1_score(y_true, y_pred, average="macro")
    print("\n", model_name)
    print(classification_report(y_true, y_pred))

    results.append({
    	"Model": model_name,
    	"Train": train_name,
    	"Test": test_name,

    	"Accuracy": acc,
    	"Precision_macro": prec,
    	"Recall_macro": rec,
    	"F1_macro": f1,

    	"Class0_Precision": report["0"]["precision"],
    	"Class0_Recall": report["0"]["recall"],
    	"Class0_F1": report["0"]["f1-score"],
    	"Class0_Support": report["0"]["support"],

    	"Class1_Precision": report["1"]["precision"],
    	"Class1_Recall": report["1"]["recall"],
    	"Class1_F1": report["1"]["f1-score"],
    	"Class1_Support": report["1"]["support"],

    	"Macro_F1": report["macro avg"]["f1-score"],
    	"Weighted_F1": report["weighted avg"]["f1-score"],

    	"Full_Report_JSON": json.dumps(report)
}
)

# ============================================================
# 9. BASELINE EXPERIMENTS
# ============================================================
for model_name in MODELS.values():

    print("\n" + "="*80)
    print(f"MODEL: {model_name}")
    print("="*80)

    for train_ds_name, train_split, test_ds_name, test_split in EXPERIMENTS:

        train_dataset = get_dataset(train_ds_name, train_split)
        test_dataset = get_dataset(test_ds_name, test_split)

        train_eval(
            model_name=model_name,
            train_ds=train_dataset,
            test_ds=test_dataset,
            train_name=f"{train_ds_name} ({train_split})",
            test_name=f"{test_ds_name} ({test_split})"
        )


# ============================================================
# 10. RESULTS TABLE
# ============================================================

results_df = pd.DataFrame(results)
print("\nFINAL RESULTS")
print(results_df)

results_df.to_csv("german_hate_speech_results_no_ne.csv", index=False)
# ============================================================
# 13. SAVE DETAILED RESULTS
# ============================================================

detailed_df = pd.DataFrame(results)

detailed_df.to_csv(
    "german_hate_speech_DETAILED_results.csv",
    index=False
)

print("\nSaved detailed report to german_hate_speech_DETAILED_results.csv")
