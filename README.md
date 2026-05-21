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
Here is a **professional GitHub README** version of your masking strategy description. It is structured, publication-ready, and written in a formal tone.

---

## Entity Masking Strategies

Seven distinct masking strategies are implemented. Each strategy modifies named entities in the input text in a different way.

---

## 1. Generic Entity Masking (PER_ORG_LOC_GENERIC_ENTITY)

### Description

All named entities belonging to Person, Organization, or Location categories are replaced with a single generic placeholder token.

### Transformation Rule

* PER → `[ENTITY]`
* ORG → `[ENTITY]`
* LOC/GPE → `[ENTITY]`

### Example

**Original:**

> Angela Merkel visited Berlin and met Siemens executives.

**Masked:**

> [ENTITY] visited [ENTITY] and met [ENTITY] executives.

### Objective

This strategy removes all entity-specific identity information while preserving sentence structure. It evaluates whether models rely on named entities as predictive shortcuts.

---

## 2. Person Masking (PER_ONLY)

### Description

Only Person entities are replaced with a placeholder token.

### Transformation Rule

* PER → `[PER]`

### Example

**Original:**

> Angela Merkel visited Berlin.

**Masked:**

> [PER] visited Berlin.

### Objective

This isolates the contribution of person names in classification decisions and evaluates potential person-specific bias.

---

## 3. Organization Masking (ORG_ONLY)

### Description

Only Organization entities are masked.

### Transformation Rule

* ORG → `[ORG]`

### Example

**Original:**

> Angela Merkel met Siemens executives.

**Masked:**

> Angela Merkel met [ORG] executives.

### Objective

This examines whether organization names influence classification performance or introduce dataset-specific bias.

---

## 4. Location Masking (LOC_ONLY)

### Description

Only location entities are replaced.

### Transformation Rule

* LOC/GPE → `[LOC]`

### Example

**Original:**

> Angela Merkel visited Berlin.

**Masked:**

> Angela Merkel visited [LOC].

### Objective

This evaluates the role of geographic information in model predictions and potential regional bias effects.

---

## 5. Typed Entity Masking (PER_ORG_LOC_TYPED)

### Description

Each entity type is replaced with a type-specific token, preserving coarse entity information while removing identity.

### Transformation Rule

* PER → `[PER]`
* ORG → `[ORG]`
* LOC/GPE → `[LOC]`

### Example

**Original:**

> Angela Merkel visited Berlin and met Siemens.

**Masked:**

> [PER] visited [LOC] and met [ORG].

### Objective

This approach preserves entity structure while removing identity. It allows the model to distinguish entity types without memorizing specific entities.

---

## 6. Length-Preserving Masking (X_LENGTH)

### Description

Each named entity is replaced by a string of "X" characters matching the original entity length.

### Transformation Rule

* Entity → `"X" * len(entity)`

### Example

**Original:**

> Berlin

**Masked:**

> XXXXXX

### Objective

This preserves surface-level length characteristics while removing semantic meaning. It helps determine whether models exploit token length patterns.

---

## 7. Random Entity Substitution (RANDOM_SUBSTITUTION)

### Description

Each named entity is replaced with a randomly sampled entity of the same type from the dataset-wide entity pool.

### Transformation Rule

* PER → random PER entity
* ORG → random ORG entity
* LOC → random LOC entity

### Example

**Original:**

> Angela Merkel visited Berlin.

**Masked:**

> Olaf Scholz visited Munich.

---
