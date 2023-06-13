from flask import Flask, request, jsonify
from transformers import BertTokenizer, TFBertModel
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import numpy as np
import tensorflow as tf
from tensorflow import keras
import nltk
# import io
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
nltk.download('punkt')
nltk.download('stopwords')

app = Flask(__name__)

# Inisialisasi objek tokenizer dan model BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = TFBertModel.from_pretrained('bert-base-uncased')
# Text preprocessing
stop_words = set(stopwords.words('indonesian'))
stemmer = PorterStemmer()

# Load model Anda di sini
model = tf.keras.models.load_model('model_hoax_new_dataset.h5')

# Mendefinisikan fungsi preprocess_text
def preprocess_text(text):
    tokens = text.lower()
    tokens = nltk.word_tokenize(tokens)
    tokens = [w for w in tokens if not w in stop_words]
    tokens = [stemmer.stem(w) for w in tokens]
    return tokens

# Mendapatkan embedding BERT dari teks
def get_bert_embeddings(text):
    encoded_text = tokenizer.encode_plus(' '.join(
        text), add_special_tokens=True, max_length=512, padding='max_length', truncation=True, return_tensors='tf')
    embeddings = bert_model(encoded_text['input_ids'])[0]
    return embeddings.numpy()

# Melakukan prediksi keaslian teks
def predict_hoax(embeddings):
    predictions = model.predict(np.expand_dims(embeddings, axis=0))
    hoax_label = np.argmax(predictions)
    hoax_percentage = np.max(predictions)
    return hoax_label, hoax_percentage, predictions

@app.route('/hoax-predict', methods=['GET','POST'])
def index():
    # Mengambil data input dari permintaan POST
    data = request.get_json()
    input_text = data['input_text']

    # Preprocessing teks
    preprocessed_text = preprocess_text(input_text)

    # Mendapatkan embedding BERT
    embeddings = get_bert_embeddings(preprocessed_text)

    # Melakukan prediksi
    hoax_label, hoax_percentage, predictions = predict_hoax(embeddings)

    # Menyiapkan respons JSON
    response = {
        'hoax_label': hoax_label.item(),
        'hoax_percentage': hoax_percentage.item(),
        'predictions': predictions.tolist()
    }

    return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True)
