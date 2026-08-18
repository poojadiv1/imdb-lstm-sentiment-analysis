# IMDB LSTM Sentiment Analysis

A deep learning project for **binary sentiment classification of IMDB movie reviews** using an **LSTM (Long Short-Term Memory)** neural network built with TensorFlow and Keras.

The model classifies movie reviews as either **Positive** or **Negative** based on the text.

## Project Overview

This project uses the IMDB dataset provided by Keras, which contains **50,000 movie reviews** divided into training and testing sets.

The workflow includes:

1. Loading the IMDB dataset
2. Limiting the vocabulary to the top 10,000 words
3. Padding/truncating reviews to a fixed length of 200 words
4. Building an LSTM-based neural network
5. Training the model for 5 epochs
6. Evaluating the model on the test dataset
7. Generating a confusion matrix and classification report
8. Saving the trained model as `imdb_lstm.h5`

## Technologies Used

* Python
* TensorFlow
* Keras
* NumPy
* Scikit-learn
* LSTM
* Deep Learning

## Model Architecture

The model consists of the following layers:

```text
Input
  ↓
Embedding Layer
  ↓
LSTM Layer
  ↓
Dense Layer (ReLU)
  ↓
Dropout
  ↓
Dense Layer (Sigmoid)
  ↓
Binary Classification
```

### Configuration

| Parameter               |               Value |
| ----------------------- | ------------------: |
| Vocabulary Size         |              10,000 |
| Maximum Sequence Length |                 200 |
| Embedding Dimension     |                 128 |
| LSTM Units              |                  64 |
| Dense Units             |                  32 |
| LSTM Dropout            |                 0.2 |
| Recurrent Dropout       |                 0.2 |
| Dense Dropout           |                 0.3 |
| Optimizer               |                Adam |
| Loss Function           | Binary Crossentropy |
| Epochs                  |                   5 |
| Batch Size              |                  64 |
| Output Classes          |                   2 |

## Dataset

The project uses the built-in **IMDB Movie Reviews Dataset** from Keras.

The dataset contains:

* **25,000 training reviews**
* **25,000 testing reviews**
* Binary sentiment labels:

  * `0` → Negative
  * `1` → Positive

The vocabulary is limited to the **10,000 most frequently occurring words**.

Each review is padded or truncated to a maximum length of **200 tokens**.

## Preprocessing

The reviews are converted into integer sequences using the IMDB dataset vocabulary.

```python
VOCAB_SIZE = 10000
MAX_LEN = 200
```

Padding is performed after the sequence:

```python
X_train = pad_sequences(
    X_train,
    maxlen=MAX_LEN,
    padding='post',
    truncating='post'
)
```

This ensures that every review has the same input shape:

```text
(25000, 200)
```

## Training

The model is trained using:

```python
history = model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=5,
    batch_size=64
)
```

Adam is used as the optimizer and binary crossentropy is used as the loss function.

## Evaluation

The trained model is evaluated using:

* Test accuracy
* Confusion matrix
* Precision
* Recall
* F1-score

Predictions are converted into binary classes using a threshold of `0.5`:

```python
y_pred = (y_pred_prob > 0.5).astype(int).ravel()
```

The project generates a classification report:

```text
              precision    recall  f1-score

Negative
Positive
```

and a confusion matrix to analyze the classification performance.

## Model Output

After training and evaluation, the model is saved as:

```text
imdb_lstm.h5
```

The model can later be loaded using:

```python
from tensorflow import keras

model = keras.models.load_model("imdb_lstm.h5")
```

> **Note:** `.h5` is the legacy HDF5 format. For newer projects, Keras recommends the native `.keras` format. This project uses `.h5` to maintain the specified project structure.

## Project Structure

```text
imdb-lstm-sentiment-analysis/
│
├── imdb_lstm_sentiment_analysis.ipynb
├── models/
│   └── imdb_lstm.h5
├── README.md
└── requirements.txt
```

## Installation

Clone the repository:

```bash
git clone https://github.com/<your-username>/imdb-lstm-sentiment-analysis.git
cd imdb-lstm-sentiment-analysis
```

Install the required dependencies:

```bash
pip install tensorflow numpy scikit-learn
```

## Running the Project

The project can be run using **Google Colab** or a local Python/Jupyter environment.

Open:

```text
imdb_lstm_sentiment_analysis.ipynb
```

Run the notebook cells sequentially to:

1. Load the dataset
2. Preprocess the reviews
3. Build the LSTM model
4. Train the model
5. Evaluate the model
6. Save the trained model

## Results

The model reports its final test accuracy after evaluation:

```python
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

print(f"Test Accuracy: {accuracy * 100:.2f}%")
```

The notebook also displays the confusion matrix and classification report for more detailed evaluation.

## Future Improvements

Possible improvements include:

* Increasing the number of training epochs
* Hyperparameter tuning
* Using Bidirectional LSTM
* Using GRU layers
* Adding pretrained word embeddings
* Implementing early stopping
* Visualizing training and validation accuracy/loss
* Building a web interface for live sentiment prediction

This project was developed as a deep learning implementation of **sentiment analysis using LSTM networks and the IMDB movie review dataset**.
