import os
import pickle

import json
import random

import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import numpy as np

import regex as re

import en_core_web_sm
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier

from nltk.corpus import stopwords

class Model:
    def __init__(self):
        print("*****Model Object has been created*****")

    # *****************************************************************
    # ***************** Setting up the model instance *****************
    # *****************************************************************
    def clean_reviews(self, review_df):
        # Lets drop the records where username isn't given
        #review_df = review_df[review_df['user_sentiment'].notna()]
        review_df = review_df[review_df['reviews_username'].notna()]

        # Lets drop duplicate reviews
        review_df = review_df.drop_duplicates(subset = ['reviews_text','id', 'reviews_username', 'user_sentiment'], keep = 'last').reset_index(drop = True)

        # Lets take this review df's copy in rating df
        rating_df = review_df.copy()

        # Lets also drop the reviews that were collected as part of promotion
        review_df = review_df[~review_df.reviews_text.str.endswith('This review was collected as part of a promotion.')]

        rating_df = rating_df.drop_duplicates(subset = ['reviews_username', 'reviews_rating', 'id'], keep = 'last').reset_index(drop = True)

        self.review_df = review_df
        self.rating_df = rating_df
        return review_df, rating_df

    def set_classification_model(self, classification_model):
        self.classification_model = classification_model

    def set_recommendation_model(self, recommendation_matrix):
        # lets build recommendation model from correlation matrix
        rating_df_pivot = self.rating_df.pivot_table(index='reviews_username',columns='id',values='reviews_rating').T
        item_predicted_ratings = np.dot((rating_df_pivot.fillna(0).T),recommendation_matrix)
        # dummy table to mask already rated products
        dummy_df = self.rating_df.copy()
        dummy_df['reviews_rating'] = dummy_df['reviews_rating'].apply(lambda x: 0 if x>=1 else 1)
        dummy_train_pivot = dummy_df.pivot_table(index='reviews_username', columns='id', values='reviews_rating').fillna(1)
        # final rating dataframe
        self.item_final_rating = np.multiply(item_predicted_ratings,dummy_train_pivot)

    def set_tfidf_model(self, tfidf_model):
        self.tfidf_model = tfidf_model
    # *****************************************************************



    # *****************************************************************
    # ************************ User Interaction ***********************
    # *****************************************************************
    def validate_user(self, username):
        return username in set(self.review_df.reviews_username)

    def get_random_user(self):
        return random.choice(list(self.review_df.reviews_username))

    def get_recommendation_for_user(self, username):
        top_20 = self.item_final_rating.loc[username].sort_values(ascending=False)[0:20]
        top_5 = self.filter_recommended_products(list(top_20.index))
        return  top_5

    def get_purchase_history(self, user):
        user_df = self.rating_df[self.rating_df.reviews_username == user]
        return self.prepare_json_data(user_df, ['name', 'reviews_rating'])
    # *****************************************************************



    # *****************************************************************
    # ***************** Filter by possitive Sentiment *****************
    # *****************************************************************
    def filter_recommended_products(self, recommended_products):
        recommended_df = self.review_df[self.review_df.id.isin(recommended_products)]
        recommended_df['pred_sentiment'] = self.classify_sentiment(recommended_df)
        top_5 = self.get_top_5_positive_items(recommended_df)
        top_5_df = self.review_df[self.review_df.id.isin(top_5)]
        return self.prepare_json_data(top_5_df, ['name', 'brand', 'categories'])

    def prepare_json_data(self, top_5_df, fields):
        top_5_df = top_5_df[fields].drop_duplicates(subset = fields, keep = 'last')
        json_list = []
        for index in  top_5_df.index:
            json_list.append(json.loads(top_5_df.loc[index].to_json()))
        return json_list

    def classify_sentiment(self, recommended_df):
        X = self.text_processing(recommended_df)
        #classification_model = self.get_classification_model()
        y_pred = self.classification_model.predict(X)
        return y_pred
    
    def get_top_5_positive_items(self, recommended_df):
        recommended_df = recommended_df[recommended_df.pred_sentiment == 'Positive']
        top_5 = recommended_df.id.value_counts().index[0:5]
        return top_5
    # *****************************************************************



    # *****************************************************************
    # *****************Text Processing for review text*****************
    # *****************************************************************
    def text_processing(self, recommended_df):
        recommended_df.reviews_text = recommended_df.reviews_text.apply(self.text_lower)
        recommended_df.reviews_title = recommended_df.reviews_title.apply(self.text_lower)
        recommended_df['review'] = recommended_df.reviews_title + ". " + recommended_df.reviews_text

        recommended_df.review = recommended_df.reviews_text.apply(self.lemmatize_and_pos)

        tfidf = self.get_tfidf_vec(recommended_df.review)
        return tfidf

    def set_up_text_processing(self):
        # Load NLP module
        self.nlp = en_core_web_sm.load()
        # Disabling few pipelines since we will not be using that using spacy
        self.nlp.disable_pipes(['parser', 'ner', 'tok2vec'])

        self.stop_words = self.get_stop_words()
        self.compiled_regex = self.get_data_cleaning_regex()

    def text_lower(self, text):
        text = str(text)
        text = text.lower() # Make the text lowercase
        return re.sub(' +', ' ', text) # Replace multiple spaces into one space between words
    
    def get_stop_words(self):
        # Defining stops words
        stop_words_all = stopwords.words('english')
        stop_words_not_to_remove = {'no', 'nor', 'not', "don't", "didn't", "doesn't", "hadn't", "hasn't", "haven't", "isn't", "mustn't", "shan't", "shouldn't", "wasn't", "weren't", "won't", "wouldn't" }
        return list( set(stop_words_all) - stop_words_not_to_remove)
    
    def get_data_cleaning_regex(self):
        # Define a regex to clean the data.
        regex_pattern = '|'.join([
            '[^\w\s]', # Remove punctuations
            '[a-zA-Z]*[0-9]+[a-zA-Z]*' # Remove alphanumeric words and words containing numbers
        ])
        compiled_regex = re.compile(regex_pattern)
        return compiled_regex

    def lemmatize_and_pos(self, text):
        doc = self.nlp(text)
        clean_text = ' '.join([token.lemma_ for token in doc if (token.pos_ in ['NOUN', 'PROPN']) and (token.text not in self.stop_words)])
        clean_text = re.sub(self.compiled_regex, '', clean_text)
        return clean_text

    def get_tfidf_vec(self, text):
        return self.tfidf_model.transform(text)
    # *****************************************************************
    