# Spam Classification — Domain Adaptation Experiments

## Overview

This project investigates spam detection performance across different text domains: **SMS messages** and **emails**.
The goal is to evaluate how well models trained in one domain generalize to another, and whether combining domains can improve robustness.

Section 7 of the notebook focuses on domain-specific experiments and includes four parts:

1. Single-domain experiments
2. Cross-domain transfer learning
3. Combined dataset analysis
4. README creation and synthesis

---

## Dataset

Two datasets are used:

* **SMS dataset:** A collection of labeled short messages (spam/ham).
* **Email dataset:** A labeled set of email messages for spam classification.

Each experiment modifies the data loading function (`train_eval_sms()`, `train_eval_email()`, `train_sms_eval_email()`, or `train_eval_combined()`) to reflect different training and evaluation settings.

---

## Methodology

### Model

All experiments use a **Logistic Regression classifier** from scikit-learn, chosen for its simplicity, interpretability, and effectiveness in linear text classification problems.

### Workflow

1. **Data preprocessing:** Each dataset is cleaned and prepared (see next section).
2. **Vectorization:** Texts are converted into numerical representations.
3. **Training:** The model is trained on either SMS, email, or combined data, depending on the scenario.
4. **Evaluation:** The model is tested on the chosen domain, reporting **Accuracy**, **Precision**, and **Recall**.
5. **Comparison:** Results are compared across domain settings to analyze generalization performance.

### Evaluation Protocol

* Standard **train/test split** was applied.
* No cross-validation was required, as the focus was on relative domain effects rather than hyperparameter tuning.
* The same preprocessing pipeline and model configuration were used across all experiments to ensure fair comparison.

---

## Preprocessing and Optimization

1. **Text normalization:** Lowercasing, whitespace normalization, and replacing any digit with `<NUM>`.
2. **Tokenization:** Handled by `CountVectorizer` with a custom token pattern `(<NUM>|[a-z]+|[!?]+|[^\w\s])` so words, punctuation, and `<NUM>` are separate tokens.
3. **Stopword removal:** English stopwords are removed (`stop_words='english'`).
4. **Vectorization:** Bag-of-words representation (`CountVectorizer`, unigrams).
5. **Model fitting:** `LogisticRegression(max_iter=1000)` on the vectorized features.

**Evaluation protocol:** 80/20 train–test split with `random_state=3` and stratification.

### Model Configuration

The experiments followed the configuration specified in the assignment instructions.  
No hyperparameter tuning or alternative preprocessing setups were tested.

* **Vectorization:** Bag-of-words using `CountVectorizer` (unigrams) with English stopword removal.
* **Normalization:** Lowercasing, whitespace cleanup, and replacement of digits with `<NUM>`.
* **Model:** Logistic Regression (`max_iter=1000`, default regularization).
* **Evaluation:** Accuracy, precision, and recall on held-out test data (80/20 split).

This setup provided consistently strong performance across both the SMS and email datasets, forming a solid baseline for analyzing domain effects.

---

## 7.1 Single-Domain Experiments

### SMS-only experiment

**Function used:** `train_eval_sms()`
**Results:**

* Accuracy: 0.9826
* Precision: 0.9912
* Recall: 0.8692

**Analysis:**

* The model performs very well within the SMS domain.
* High precision indicates that almost all detected spam messages are indeed spam.
* Recall is somewhat lower, meaning some spam messages are missed.
* Compared to the combined dataset, accuracy is slightly higher, suggesting that focusing on a single, consistent text style helps the model specialize.

**Answers to questions:**

* *How does performance compare to the combined dataset results?*
  The SMS-only model shows slightly higher precision but lower recall than the combined model. It performs marginally better on familiar (in-domain) text.
* *Which metrics show the most significant changes?*
  Recall drops the most compared to the combined case, indicating the model’s limited ability to capture all spam instances in the SMS domain.
* *What might explain any performance differences?*
  SMS data tends to have shorter, more formulaic messages. The model overfits to this style and loses some generalizability, explaining higher precision but lower recall.

---

### Email-only experiment

**Function used:** `train_eval_email()`
**Results:**

* Accuracy: 0.9854
* Precision: 0.9795
* Recall: 0.9709

**Analysis:**

* Both precision and recall are high, indicating strong performance on the email domain.
* Compared to the SMS-only model, recall improves substantially, suggesting that spam patterns in emails are easier for the model to capture.

**Answers to questions:**

* *Do emails and SMS messages show similar classification difficulty?*
  Not entirely. Emails appear slightly easier to classify, as indicated by higher recall and similar accuracy.
* *Which dataset appears more challenging for spam detection?*
  The SMS dataset is more challenging due to shorter message length and less context.
* *How do the optimal features differ between domains?*
  SMS classification relies heavily on short keyword cues, while email detection benefits from richer linguistic and structural features (subject lines, longer content, formatting patterns).

---

## 7.2 Cross-Domain Transfer Learning

### Train on SMS, evaluate on Email

**Function used:** `train_sms_eval_email()`
**Results:**

* Accuracy: 0.3317
* Precision: 0.2932
* Recall: 0.8929

**Analysis:**

* Accuracy and precision drop sharply, while recall remains high.
* The model tends to classify many messages as spam (overgeneralization).
* This illustrates poor generalization when transferring from SMS to email.

**Answers to questions:**

* *How much does performance degrade when transferring across domains?*
  Performance degrades dramatically, with accuracy dropping to around 33%.
* *Which metrics are most affected by domain mismatch?*
  Precision is most affected, showing a steep decline, while recall remains artificially high due to many false positives.
* *What linguistic differences might explain the results?*
  Emails are longer, more formal, and contain different token distributions, punctuation, and structure. SMS training fails to capture these characteristics, leading to misclassification.

---

## 7.3 Combined Dataset Analysis

### Train and evaluate on combined datasets

**Function used:** `train_eval_combined()`
**Results:**

* Accuracy: 0.9727
* Precision: 0.9580
* Recall: 0.9135

**Analysis:**

* Combining datasets slightly reduces accuracy compared to single-domain training but yields balanced metrics overall.
* The model gains robustness across domains, handling both SMS and email inputs moderately well.

**Answers to questions:**

* *Does combined training improve generalization across both domains?*
  Yes, the model generalizes better across SMS and email than either single-domain model when tested cross-domain.
* *How do results compare to single-domain experiments?*
  Accuracy is slightly lower, but the balance between precision and recall is improved.
* *What are the trade-offs of mixed-domain training?*
  Mixed-domain training improves robustness but reduces specialization. The model sacrifices a small amount of domain-specific accuracy in exchange for better adaptability.

---

## Key Takeaways

* **Single-domain training** yields the highest performance when test data matches the training domain.
* **Cross-domain transfer** causes major degradation, confirming that spam classification models are highly domain-sensitive.
* **Combined training** provides a practical compromise: slightly lower accuracy but much better generalization across domains.
* **Precision–recall trade-offs** vary significantly by domain, reflecting differences in message structure and vocabulary.
* **CountVectorizer with unigrams and Logistic Regression** proved to be an effective, interpretable, and efficient baseline for spam detection.

---

## Optimal Preprocessing Configuration

To be determined in the upcoming pipeline phase, where multiple preprocessing strategies (e.g., TF-IDF, n-grams, lemmatization) will be evaluated and compared.  
The current notebook used a fixed configuration (lowercasing, digit replacement with `<NUM>`, stopword removal, and `CountVectorizer` with unigrams) as a consistent baseline.

---

## Practical Deployment Recommendations

Deployment considerations will be developed in the pipeline stage.  
These will include model persistence (saving/loading), integration into an application for real-time spam detection, and strategies to handle new or unseen domains.  
At this stage, the focus is limited to experimental evaluation and performance comparison.
