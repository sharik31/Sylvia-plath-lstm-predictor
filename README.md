# 🧠 Sylvia Plath Poetry – Next Word Predictor using LSTM

This project is a deep learning-based next-word prediction model trained on the poetic and emotionally rich text of **Sylvia Plath**, along with select lines from Shakespeare. Using an LSTM (Long Short-Term Memory) network, the model attempts to complete poetic lines, preserving the unique style and vocabulary of the author.

---

## ✨ Demo

**Input**: `"The world"`  
**Generated Output**: `"The world will go up in a shriek and your head with"`

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
```

## 📊 Training

The model is trained using the following command:

```python
model.fit(X, Y, epochs=100)


- Input Sequences: 850 n-gram samples  
- Labels: One-hot encoded next words  
- Epochs: 100
```

## 🔮 Prediction Logic

Here's how next-word prediction is performed:

```python
text = "The world"
for i in range(10):
    token_text = tokenizer.texts_to_sequences([text])[0]
    padded = pad_sequences([token_text], maxlen=14, padding='pre')
    predicted_index = np.argmax(model.predict(padded))
    for word, index in tokenizer.word_index.items():
        if index == predicted_index:
            text += ' ' + word
            print(text)
```
## 📦 Installation

git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name
pip install -r requirements.txt


