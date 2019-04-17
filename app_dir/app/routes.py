from app import application#, classes
# from flask_login import current_user, login_user, login_required, logout_user
from flask import render_template

@application.route('/index')
@application.route('/')
def index():
    return '<h1> Welcome to the movie review website </h1>'


application.run(host='0.0.0.0', port=80)
