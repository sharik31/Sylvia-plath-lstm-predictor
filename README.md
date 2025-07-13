# 🧠 Sylvia Plath Poetry – Next Word Predictor using LSTM

This project is a deep learning-based next-word prediction model trained on the poetic and emotionally rich text of **Sylvia Plath**, along with select lines from Shakespeare. Using an LSTM (Long Short-Term Memory) network, the model attempts to complete poetic lines, preserving the unique style and vocabulary of the author.

---

## ✨ Demo

**Input**: `"The world"`  
**Generated Output**:

---

## 📝 Dataset

The model was trained on a hand-curated corpus that includes:
- Sylvia Plath's poem *Mirror* and other poetic fragments
- Shakespearean monologues from *As You Like It*

This fusion of poetic language enriches the vocabulary and rhythm of the predicted output.

---


---

## 📝 Dataset

The model was trained on a hand-curated corpus that includes:
- Sylvia Plath's poem *Mirror* and other poetic fragments
- Shakespearean monologues from *As You Like It*

This fusion of poetic language enriches the vocabulary and rhythm of the predicted output.

---

## 🧠 Model Architecture

```python
model = Sequential()
model.add(Embedding(434, 100, input_length=14))
model.add(LSTM(150))
model.add(Dense(434, activation='softmax'))

