from flask import Flask, render_template, request
from ngrams import n_grams
from bert import bert

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
            return render_template('results.html', predictions = n_grams(sentence))
        else:
            return render_template('results.html', predictions = bert(sentence))

if __name__ == '__main__':
    app.run(debug=True)