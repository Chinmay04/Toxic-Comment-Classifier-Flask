import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import text, sequence
from tensorflow.keras.layers import Dense, LSTM, Activation, Dropout, Embedding, Bidirectional
from tensorflow.keras import Input, Model
from tensorflow.keras.callbacks import EarlyStopping
import re
import string
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')

class TCC():
  def preprocess(self, text):
    punctuation_edit = string.punctuation +"0123456789"
    text = text.lower()
    #remove numbers
    text = re.sub(r'\d+', '', text)
    #remove extra whitespace
    text = " ".join(text.split())
    #remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    #stop words
    stop_words = stopwords.words('english')
    word_tokens = word_tokenize(text)
    for word in stop_words:
      if word in word_tokens:
        word_tokens.remove(word)
    #Lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmas = [lemmatizer.lemmatize(word, pos ='v') for word in word_tokens]
    text = ' '.join(lemmas)
    text =  [text]
    tok = Tokenizer(num_words=20000, filters=punctuation_edit)
    tok.fit_on_texts(list(text))
    seq = tok.texts_to_sequences(text)
    pad = sequence.pad_sequences(seq, maxlen=100)
    return pad
  def LoadedModel(self):
    inputs = Input(shape=(100, ))
    x = Embedding(20000, 128)(inputs)
    x = Bidirectional(LSTM(50))(x)
    x = Dropout(0.3)(x)
    x = Dense(50, activation="relu")(x)
    x = Dropout(0.1)(x)
    outputs = Dense(6, activation="sigmoid")(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.load_weights('lstm.h5')
    return model
from flask import Flask, render_template
from flask import request


app = Flask(__name__)
@app.route('/', methods=["GET", "POST"])
def input():
  return render_template('input.html')

@app.route('/prediction', methods=["GET", "POST"])
def pred():
  if request.method == "POST":
    text = request.form.get("t")
    tcc = TCC()
    processed = tcc.preprocess(text)
    model = tcc.LoadedModel()
    predictions = model.predict(processed)[0]
  return render_template('preds.html', p = predictions, t = text)

if __name__ == "__main__":
  app.run()
