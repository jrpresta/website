from app import application#, classes
# from flask_login import current_user, login_user, login_required, logout_user
from flask import render_template

@application.route('/index')
@application.route('/')
def index():
    return render_template('scratch.html',
                           words=[('testing', 1), ('this', 0.5), ('html', 0.1)],
                           prob=0.69)


application.run(host='0.0.0.0', port=8080)