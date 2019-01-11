import os
from flask import Flask, request, redirect, url_for, render_template
# from werzeug.utils import secure_filename

html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Title</title>
</head>
<body>
    <p>
        The purpose of this website is to allow a user to upload an image and have it
        recognized. As of right now, the model that I have distinguishes between
        apples and oranges. I hope to learn how to deploy that model on the server at
        the back-end, and I want the front-end HTML to be decent.
    </p>
    <p>
        Insert image below:
    </p>
    <form action="/">
        <input type="file" name="pic" accept="image/*">
        <input type="submit">
    </form>
</body>
</html>
"""

#TODO: Redirect the submit to a page that predicts with the model (i.e. the /actionpage)
#TODO: Fix the render_template() to work with site.html
#TODO: All the rest of the hard model stuff



UPLOAD_FOLDER = 'img/'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

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
    # return render_template('site.html')
    return html

app.run(host='0.0.0.0', port=5000)