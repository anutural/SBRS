from flask import Flask, render_template, request

from Src_Model.utility import *

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pickle
import nltk
nltk.download('stopwords')

app = Flask(__name__)

classification_model_file_path = "Src_Model/classification_rf.pkl"
recommendation_model_file_path = "Src_Model/recommendation.pkl"
tfidf_model_file_path = "Src_Model/tfidf_model.pkl"
reviews_file_path = "Data/sample30.csv"

#***********************************************************#
#Loading all the pickle files and inject that in model class#
#***********************************************************#
def save_obj(obj, file_name ):
    with open(file_name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
    
def load_obj(file_name):
    pkl_content = open(file_name, "rb")
    obj = pickle.load(pkl_content)
    return obj

def get_classification_model():
    classification_model = load_obj(classification_model_file_path)
    return classification_model

def get_recommendation_matrix():
    # lets load recommendation model pickel file
    recommendation_matrix = load_obj(recommendation_model_file_path)
    return recommendation_matrix

def get_tfidf_model():
    tfidf_model = load_obj(tfidf_model_file_path)
    return tfidf_model 

def load_reviews():
    review_df = pd.read_csv(reviews_file_path)
    return review_df


def set_up_model():
    model = Model()
    model.clean_reviews(load_reviews())
    model.set_classification_model(get_classification_model())
    model.set_recommendation_model(get_recommendation_matrix())
    model.set_tfidf_model(get_tfidf_model())
    model.set_up_text_processing()
    return model
#***********************************************************#

model = set_up_model()

@app.route("/", methods=["POST", "GET"])
def home():
    if request.method == "POST":
        user = str(request.form['projectFilepath'])
        if model.validate_user(user):
            data_2 = model.get_recommendation_for_user(user)
            return render_template('index.html', data=data_2, init_msg = "Recommendations for " + user)
        else:
            return render_template('index.html', data = [], init_msg = "No such user found")


    if request.method == "GET":
        return render_template('index.html', data="", init_msg = "")

if __name__ == '__main__':
    model = set_up_model()
    app.run()