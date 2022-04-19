from flask import Flask, render_template, request
from ngrams import n_grams
from bert import bert
from lstm import lstm

import pymongo

client = pymongo.MongoClient()
db = client.get_database('next_word_project')

app = Flask(__name__)
app.secret_key = "AI-PROJECT"

@app.route("/", methods=['GET'])
def home():
    return render_template('index.html')

@app.route("/predict", methods=['GET', 'POST'])
def index():
    if(request.method == 'POST'):
        sentence = request.form.get('sentence')
        checked_boxes = request.form.getlist('exampleRadios')
        if(checked_boxes[0] == 'option1'):
            predictions = n_grams(sentence)
            store_db(sentence, predictions)
            return render_template('results.html', predictions=predictions)
        elif(checked_boxes[0] == 'option2'):
            predictions = bert(sentence)
            store_db(sentence, predictions)
            return render_template('results.html', predictions=predictions)
        elif(checked_boxes[0] == 'option3'):
            predictions = lstm(sentence)
            store_db(sentence, predictions)
            return render_template('results.html', predictions=predictions)
        else:
            return render_template('results.html', predictions=[sentence])

@app.route("/accuracy", methods=['GET', 'POST'])
def get_accuracy():
    correct_values = db.correct.count_documents({})
    wrong_values = db.wrong.count_documents({})
    accuracy = str(correct_values / (correct_values + wrong_values) * 100.00) + ' %'
    return render_template('accuracy.html', correct_values=correct_values, wrong_values=wrong_values, accuracy=accuracy)

@app.route('/correct/<prediction>', methods=['GET', 'POST'])
def correct(prediction):
    db.correct.insert_one({
        'prediction': prediction
    })

    return render_template('base.html', message='Saved Successfully')

@app.route('/wrong/<prediction>', methods=['GET', 'POST'])
def wrong(prediction):
    db.wrong.insert_one({
        'prediction': prediction
    })

    return render_template('base.html', message='Saved Successfully')

def store_db(sentence, predictions):
    db.history.insert_one({
        'sentence': sentence,
        'predictions': predictions
    })

if __name__ == '__main__':
    app.run(debug=True)