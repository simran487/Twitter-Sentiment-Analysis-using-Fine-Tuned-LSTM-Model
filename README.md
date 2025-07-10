# Twitter-Sentiment-Analysis-using-Fine-Tuned-LSTM-Model

This project performs sentiment analysis on Twitter data using a fine-tuned LSTM deep learning model. It classifies tweets as **positive** or **negative** sentiment.

## 📌 Features
- Preprocessing and cleaning of text data
- Tokenization and padding
- Deep learning model with LSTM layer
- Training and evaluation pipeline
- Visualization of accuracy and loss

## 📂 Project Structure
├── Dataset/
│ ├── train.csv
│ └── test.csv
├── Twitter_Sentiment_Analysis_using_LSTM.ipynb
└── README.md

## 🗂️ Dataset
- `train.csv` and `test.csv` contain labeled tweets.
- The dataset is preprocessed to remove noise and tokenize text for model training.

## 🔥 Model
- Built using TensorFlow/Keras.
- Uses Embedding layer and LSTM for sequence modeling.
- Fine-tuned for improved accuracy on sentiment classification.

## 📈 Results
- Training and validation accuracy and loss plotted.
- Example predictions on test data.

## ⚙️ How to Run
1. Clone this repository:
git clone https://github.com/simran487/Twitter-Sentiment-Analysis-using-Fine-Tuned-LSTM-Model.git

2. Open the Jupyter Notebook:

3. Run all cells to train and evaluate the model.

## ✅ Requirements
- Python 3.x
- TensorFlow
- Keras
- pandas
- matplotlib
- scikit-learn

Install dependencies using:
pip install -r requirements.txt
*(create `requirements.txt` as needed)*

## 📜 License
This project is for educational purposes. Feel free to use and modify.

## ✨ Author
- [Simran Kumari](https://github.com/simran487)
