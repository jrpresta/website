from app import application
import sys
from flask import render_template, request
sys.path.insert(0, '../model/')
import model
from .frontend import RGB_calc

# TODO: Handle Key Errors

@application.route('/')
def home():
    return render_template('scratch.html')


@application.route('/', methods=['POST'])
def index():
    # test = 'if this movie were a sandwich it would be gross'
    test = request.form['text']

    p, alpha = model.han_prediction(test,
                                    '../model/HAN_lower.pt',
                                    '../model/reviews.pkl')

    print(alpha)
    colors = [RGB_calc(a) for a in alpha]

    return render_template('index.html',
                           words=zip(test.split(), colors),
                           prob=p)


application.run(host='0.0.0.0', port=8080)
