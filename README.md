[HuggingFace Space Deployed Link](https://huggingface.co/spaces/AbdulWahab70/emotion-detection-sentiment)

# 🧠 Emotion Sentiment Analysis using LSTM

This project focuses on classifying human emotions in text using an LSTM-based deep learning model. The original dataset had a **7-class emotion distribution**, which was **re-mapped** into broader sentiment categories: `positive`, `negative`, and `neutral`. Special attention was given to class imbalance and model generalization using various preprocessing and resampling techniques.

---

## 📁 Dataset Overview

The original dataset contains emotion labels ranging from 0 to 6:

| Label | Emotion    | Count   |
|-------|------------|---------|
| 0     | No Emotion | 85,572  |
| 1     | Anger      | 1,022   |
| 2     | Disgust    | 353     |
| 3     | Fear       | 174     |
| 4     | Happiness  | 12,885  |
| 5     | Sadness    | 1,150   |
| 6     | Surprise   | 1,823   |

---

## 🔁 Emotion Remapping

To simplify and better balance the classification task, the 7 emotion labels were grouped into 3 broader sentiment classes:

```python
def map_emotion(label):
    if label in [1, 2, 3, 5]:
        return 'negative'
    elif label in [4, 6]:
        return 'positive'
    else:
        return 'neutral'
```

---

## ⚙️ Preprocessing Steps

The text data was cleaned and normalized using the following steps:

- Lowercasing all text
- Removing numbers
- Removing URLs
- Lemmatization (WordNet)
- Removing very short sentences (less than 3 words)

---

## 📉 Dealing with Imbalanced Data

The original mapped sentiment classes were still heavily imbalanced:

| Sentiment | Count   |
|-----------|---------|
| Neutral   | 85,572  |
| Positive  | 14,708  |
| Negative  | 2,699   |

### ✅ Applied Resampling:
- **Upsampled** `positive` and `negative` classes to 10,000
- **Downsampled** `neutral` class to 10,000

---

## 🧠 Model Architecture

The model was built using an LSTM-based neural network with Keras/TensorFlow:

- Embedding Layer
- LSTM Layer
- Dense + Dropout
- Softmax Activation

**Training Results:**
- 📈 **Validation Accuracy:** ~77%
- 🤖 Model trained on the **balanced dataset**
- 📌 Still showed signs of **poor generalization for the `negative` class**

---

## 🔍 Insights & Learnings

- The original dataset was **heavily imbalanced**, especially for the negative class.
- Mapping 7 labels into 3 broader classes helped simplify the task.
- Even after resampling, the **model struggled to generalize well on real negative samples**.
- Preprocessing plays a critical role, but oversampling rare emotion types may not be enough alone.

---


## 🚀 Run Locally

Make sure you install dependencies:

```bash
pip install -r requirements.txt
```

To run the Streamlit app:

```bash
streamlit run app.py
```

## 🏁 Final Notes

This project shows how deep learning models perform on emotion-driven sentiment tasks, but also highlights the importance of balancing, representation, and evaluation strategies—especially for low-resource emotion types.
