from flask import Flask, render_template, request, redirect, url_for
import os
from werkzeug.utils import secure_filename
from process_image import predict_digits

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files['image']
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            result_path, prediction = predict_digits(filepath)
            return render_template("index.html", result_img=result_path, prediction=prediction)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
