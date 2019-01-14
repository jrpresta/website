# Much of the format of the server adopted from
# https://github.com/ibrahimokdadov/upload_file_python
import os
from flask import Flask, request, redirect, render_template

# TODO: Determine how to actually receive the image and store on the server
# TODO: All the rest of the hard model stuff

UPLOAD_FOLDER = 'img/'
# ALLOWED_EXTENSIONS = {'jpg', 'jpeg'}

app = Flask(__name__)
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def upload_file():
    """Fairly insecure but testing for now"""
    # TODO: Work on making it hard to break
    file = request.files['file']

    if file.filename == '':
        return redirect(request.url)
    else:
        # filename = secure_filename(file.filename)
        filename = UPLOAD_FOLDER + file
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return


@app.route("/")
def homepage():
    """Hosting the boring HTML"""
    return render_template('upload.html')


@app.route("/upload", methods=['POST'])
def upload():
    """
    Writes the uploaded file out to the server's disk
    # TODO: Call the library to predict apple or orange, will write the result to jinja template
    :return:
    """
    target = 'img/'

    # writing out the uploaded file to the server
    file = request.files.getlist("file")[0]
    save_location = f"{target}/{file.filename}"
    file.save(save_location)

    return render_template("result.html")


app.run(host='0.0.0.0', port=5000)