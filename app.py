from flask import Flask, render_template, request

from Src_Model.utility import *

import pandas as pd

app = Flask(__name__)
model = Model()


@app.route("/", methods=["POST", "GET"])
def home():
    if request.method == "POST":
        user = request.form['projectFilepath']
        data_2 = model.get_recommendation_for_user(user)
        return render_template('index.html', data=data_2, init_msg = "----")


    if request.method == "GET":
        return render_template('index.html', data="", init_msg = "")


@app.route('/details')
def details():
    print('Details')
    return 'Details'

if __name__ == '__main__':
    print("***************************************")
    model.load_and_clean_reviews()
    app.run()