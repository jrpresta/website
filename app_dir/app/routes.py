from app import application
import sys
from flask import render_template
sys.path.insert(0, '../model/')
import model
from .frontend import RGB_calc

@application.route('/index')
@application.route('/')
def index():
    test = 'favorite movie ever'
    p, alpha = model.han_prediction(test,
                                    '../model/HAN_lower.pt',
                                    '../model/reviews.pkl')
    colors = [RGB_calc(a) for a in alpha]

    # TODO: Write function provides better RBG values for printing

    return render_template('index.html',
                           words=zip(test.split(), colors),
                           prob=p)

    # return render_template('scratch.html',
    #                        words=zip(test.split(), alpha),
    #                        prob=p)


application.run(host='0.0.0.0', port=8080)
