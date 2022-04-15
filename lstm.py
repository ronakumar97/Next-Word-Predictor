import pickle
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import pandas as pd

def lstm(sentence):
    next_words = 2

    model = pickle.load(open('model.pkl', 'rb'))

    dataset = pd.read_csv('dataset.csv')

    dataset['title'] = dataset['title'].apply(lambda x: x.replace(u'\xa0', u' '))
    dataset['title'] = dataset['title'].apply(lambda x: x.replace('\u200a', ' '))

    tokenizer = Tokenizer(oov_token='<oov>')
    tokenizer.fit_on_texts(dataset['title'])

    predictions = []

    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([sentence])[0]
        token_list = pad_sequences([token_list], maxlen=39, padding='pre')
        predicted = np.argmax(model.predict(token_list), axis=-1)
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        predictions.append(output_word)
        sentence += " " + output_word
    return predictions