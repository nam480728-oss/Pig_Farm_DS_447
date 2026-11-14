"""Webapp Flask mo phong app dien thoai cua chu trai.

Chay: python webapp/app.py  ->  http://localhost:5000
TODO: them cac API trong muc Huong dan, Buoc 8 cua tai lieu.
"""
from flask import Flask, render_template

app = Flask(__name__)


@app.route("/")
def trang_chu():
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
