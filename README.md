# Duplicate Question Detection - NLP Semester Project



## Overview

This project implements three NLP approaches to detect duplicate questions on the Quora dataset:
- **Baseline:** TF-IDF + Logistic Regression
- **Intermediate:** BiLSTM + GloVe Embeddings  
- **Advanced:** Sentence-BERT (Fine-tuned Transformer)

📓 **For detailed implementation, please refer to the Jupyter notebook.**

---

## Dataset

**Source:** Quora Duplicate Questions Dataset

- Question pairs with binary labels (duplicate/not duplicate)
- Train-test split: 80-20
- Preprocessed to remove null values

---

## Models Summary

### 1. TF-IDF + Logistic Regression (Baseline)

**Approach:**
- TF-IDF vectorization (10K features, bigrams)
- Handcrafted features: cosine similarity, word overlap, length difference
- Logistic Regression with threshold tuning (optimal: 0.4)

**Key Parameters:**
```python
TfidfVectorizer(max_features=10000, ngram_range=(1,2))
LogisticRegression(C=5.0, class_weight='balanced')
```

### 2. BiLSTM + GloVe (Intermediate)

**Approach:**
- Pre-trained GloVe embeddings (100-dim)
- Siamese BiLSTM architecture with shared weights
- Feature fusion: concatenation + absolute difference

**Key Parameters:**
```python
MAX_VOCAB=50000, MAX_LEN=30, EMBED_DIM=100
BiLSTM(64 units), Dense(128), Dropout(0.3)
Epochs=5, Batch=256
```

### 3. Sentence-BERT (Advanced)

**Approach:**
- Fine-tuned `all-MiniLM-L6-v2` transformer
- Cosine similarity on sentence embeddings
- Logistic Regression classifier on similarity scores

**Key Parameters:**
```python
Model: all-MiniLM-L6-v2 (384-dim embeddings)
Loss: CosineSimilarityLoss
Epochs=1, Batch=16
```

---

## Results

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **TF-IDF + LR** | 75.37% | 64.55% | 83.11% | 72.66% |
| **BiLSTM + GloVe** | 81.50% | 76.16% | 77.20% | 76.68% |
| **Sentence-BERT** | **85.60%** | **80.16%** | **84.32%** | **82.18%** |

**Key Insights:**
- Progressive improvement across all metrics (72.66% → 82.18% F1)
- SBERT achieves best precision-recall balance
- BiLSTM offers good performance with moderate complexity
- TF-IDF provides fast baseline with high recall

---

## Setup & Usage

**Install Dependencies:**
```bash
pip install pandas numpy scikit-learn tensorflow
pip install sentence-transformers torch gradio
```

**Download Data:**
- Dataset: `quora_duplicate_questions.csv`
- GloVe: `glove.6B.100d.txt` (for BiLSTM)

**Run the Notebook:**
Open `nlp_semester_project_F230022_F230048_.ipynb` and execute cells sequentially.

---

## Interactive Demo

The notebook includes a Gradio interface for testing predictions:

```python
demo = gr.Interface(
    fn=predict_duplicate,
    inputs=["text", "text"],
    outputs="text"
)
demo.launch()
```

---

## Key Takeaways

1. **Transfer learning matters:** Pre-trained models (GloVe, SBERT) significantly outperform from-scratch approaches
2. **Context capture:** BiLSTM > TF-IDF by capturing sequential dependencies
3. **Transformers excel:** Attention mechanisms in SBERT capture deep semantic relationships
4. **Threshold tuning:** Simple optimization yields 3-5% F1 improvement
5. **Trade-offs exist:** Accuracy vs speed vs interpretability

---

## 👥 Authors

| Name | GitHub |
|------|--------|
| Mobeen | [@mobeenrukhsar269-crypto](https://github.com/mobeenrukhsar269-crypto) |
| Muhammad Ahmad | [@](https://github.com/) |


## References

- Reimers & Gurevych (2019). Sentence-BERT
- Pennington et al. (2014). GloVe Embeddings
- Quora Question Pairs Dataset (Kaggle)