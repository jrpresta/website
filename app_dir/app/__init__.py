from flask import Flask
from config import Config
# from flask_login import LoginManager

# initialize
# create an application instance which handles all requests
application = Flask(__name__)
application.config.from_object(Config)

from app import routes
