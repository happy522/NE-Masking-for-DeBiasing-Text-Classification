# Named Entity Masking for De-Biasing Text Classification: Experiments with German Datasets

## Overview

Hate speech classification systems often develop biases toward named entities that frequently appear in training datasets. As a result, these models may struggle to generalize effectively when evaluated on unseen datasets or real-world data.

This project explores **Named Entity (NE) masking** as a strategy to reduce such bias in hate speech detection systems. We systematically evaluate **eight masking strategies** using two supervised learning models across three German hate speech datasets:

- GermEval 2018
- HASOC
- GAHD

Our experiments show that while NE masking can slightly reduce performance on the original training dataset, it significantly improves **cross-dataset generalization**. In particular, the **GAHD** dataset demonstrates a strong improvement in cross-dataset performance after applying NE masking techniques.

---

# Datasets Used

## 1. GermEval 2018

**Dataset Link:**  
https://heidata.uni-heidelberg.de/dataset.xhtml?persistentId=doi:10.11588/data/0B5VML

The GermEval 2018 dataset was introduced as part of a shared task focused on detecting offensive language in German tweets. The dataset contains **8,541 entries** collected from social media platforms.

### Key Features

- Language: German
- Domain: Social media / Twitter
- Task: Offensive language detection
- Size: 8,541 samples

---

## 2. HASOC

**Dataset Link:**  
https://hasocfire.github.io/hasoc/official/index.html

The HASOC (Hate Speech and Offensive Content Identification) dataset was developed as part of the FIRE 2019 shared task series. Although the competition primarily focused on Indian languages, it also included a German-language dataset.

The German subset contains **4,669 posts** collected from social networks such as Twitter and Facebook.

### Key Features

- Language: German
- Domain: Twitter and Facebook
- Task: Hate speech and offensive content detection
- Size: 4,669 samples

---

## 3. GAHD

**Dataset Link:**  
https://github.com/jagol/gahd

The GAHD (German Adversarial Hate Dataset) was published in 2024 and was specifically designed to evaluate the robustness of hate speech detection systems using adversarial and contrastive examples.

The dataset includes **10,996 German-language texts**, combining authentic social media posts with synthetically generated adversarial content.

### Key Features

- Language: German
- Domain: Social media + synthetic adversarial examples
- Task: Robust hate speech detection
- Size: 10,996 samples

---
