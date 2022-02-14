import os
import pickle

import json

import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import numpy as np

import regex as re

import en_core_web_sm
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier

from nltk.corpus import stopwords

class Model:

    classification_model_file_path = "Src_Model\\classification_rf.pkl"
    recommendation_model_file_path = "Src_Model\\recommendation.pkl"
    tfidf_model_file_path = "Src_Model\\tfidf_model.pkl"
    reviews_file_path = "Data\\sample30.csv"


    def __init__(self):
        self.review_df, self.rating_df = self.load_and_clean_reviews()
        self.classification_model = self.get_classification_model()
        self.item_final_rating = self.get_recommendation_model()
        self.tfidf_model = self.get_tfidf_model_model()

        self.set_up_text_processing()


    def get_parent_directory(self):
        parent_directory = os.path.abspath('.')
        print(parent_directory)
        return parent_directory

    def save_obj(self, obj, file_name ):
        with open(file_name + '.pkl', 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

    def load_obj(self, file_name):
        pkl_content = open(file_name, "rb")
        obj = pickle.load(pkl_content)
        return obj


    def load_and_clean_reviews(self):
        review_df = pd.read_csv(self.get_parent_directory() + '\\' + self.reviews_file_path)
        print(review_df.shape)
        # Lets drop the records where username isn't given
        #review_df = review_df[review_df['user_sentiment'].notna()]
        review_df = review_df[review_df['reviews_username'].notna()]
        print('Review DF Shape', review_df.shape)

        # Lets drop duplicate reviews
        review_df = review_df.drop_duplicates(subset = ['reviews_text','id', 'reviews_username', 'user_sentiment'], keep = 'last').reset_index(drop = True)
        print('Review DF Shape', review_df.shape)

        # Lets take this review df's copy in rating df
        rating_df = review_df.copy()

        # Lets also drop the reviews that were collected as part of promotion
        review_df = review_df[~review_df.reviews_text.str.endswith('This review was collected as part of a promotion.')]
        print('Review DF Shape', review_df.shape)

        rating_df = rating_df.drop_duplicates(subset = ['reviews_username', 'reviews_rating', 'id'], keep = 'last').reset_index(drop = True)
        print('Rating DF Shape', rating_df.shape)
        return review_df, rating_df


    def save_classification_model(self, obj):
        self.save_obj(obj, self.get_parent_directory() + '\\' + self.classification_model_file_path)

    def get_classification_model(self):
        classification_model = self.load_obj(self.get_parent_directory() + '\\' + self.classification_model_file_path)
        return classification_model

    def save_recommendation_model(self, obj):
        self.save_obj(obj, self.get_parent_directory() + '\\' + self.recommendation_model_file_path)

    def get_recommendation_model(self):
        # lets load recommendation model pickel file
        recommendation_matrix = self.load_obj(self.get_parent_directory() + '\\' + self.recommendation_model_file_path)
    
        # lets build recommendation model from correlation matrix
        rating_df_pivot = self.rating_df.pivot_table(index='reviews_username',columns='id',values='reviews_rating').T
        item_predicted_ratings = np.dot((rating_df_pivot.fillna(0).T),recommendation_matrix)

        # dummy table to mask already rated products
        dummy_df = self.rating_df.copy()
        dummy_df['reviews_rating'] = dummy_df['reviews_rating'].apply(lambda x: 0 if x>=1 else 1)
        dummy_train_pivot = dummy_df.pivot_table(index='reviews_username', columns='id', values='reviews_rating').fillna(1)

        # final rating dataframe
        item_final_rating = np.multiply(item_predicted_ratings,dummy_train_pivot)
        return item_final_rating

    def save_tfidf_model(self, obj):
        self.save_obj(obj, self.get_parent_directory() + '\\' + self.tfidf_model_file_path)

    def get_tfidf_model_model(self):
        tfidf_model = self.load_obj(self.get_parent_directory() + '\\' + self.tfidf_model_file_path)
        print(type(tfidf_model))
        return tfidf_model   


    def get_recommendation_for_user(self, username):
        #item_final_rating = self.get_recommendation_model()
        print("Generated Recommendation Model")
        top_20 = self.item_final_rating.loc[username].sort_values(ascending=False)[0:20]
        print("Got top 20 products")
        top_5 = self.filter_recommended_products(list(top_20.index))
        print(top_5[0])
        return  top_5
    
    # *****************************************************************
    # ***************** Filter by possitive Sentiment *****************
    # *****************************************************************
    def filter_recommended_products(self, recommended_products):
        recommended_df = self.review_df[self.review_df.id.isin(recommended_products)]
        recommended_df['pred_sentiment'] = self.classify_sentiment(recommended_df)
        top_5 = self.get_top_5_positive_items(recommended_df)
        top_5_df = self.review_df[self.review_df.id.isin(top_5)]
        return self.prepare_json_data(top_5_df)

    def prepare_json_data(self, top_5_df):
        top_5_df = top_5_df[['name', 'brand', 'categories']].drop_duplicates(subset = ['name', 'brand', 'categories'], keep = 'last')
        json_list = []
        for index in  top_5_df.index:
            json_list.append(json.loads(top_5_df.loc[index].to_json()))
        return json_list

    def classify_sentiment(self, recommended_df):
        X = self.text_processing(recommended_df)
        #classification_model = self.get_classification_model()
        print('Classification - Got classification model')
        y_pred = self.classification_model.predict(X)
        print('Classification - prediction made')
        return y_pred
    
    def get_top_5_positive_items(self, recommended_df):
        recommended_df = recommended_df[recommended_df.pred_sentiment == 'Positive']
        print(recommended_df.id.value_counts())
        top_5 = recommended_df.id.value_counts().index[0:5]
        print(top_5)
        return top_5
    # *****************************************************************

    # *****************************************************************
    # *****************Text Processing for review text*****************
    # *****************************************************************
    def text_processing(self, recommended_df):
        recommended_df.reviews_text = recommended_df.reviews_text.apply(self.text_lower)
        recommended_df.reviews_title = recommended_df.reviews_title.apply(self.text_lower)
        print('Text Processing - Lower case applied')
        recommended_df['review'] = recommended_df.reviews_title + ". " + recommended_df.reviews_text
        print('Text Processing - got combined Review data:', recommended_df.shape)

        recommended_df.review = recommended_df.reviews_text.apply(self.lemmatize_and_pos)
        print('Text Processing - lemmatization applied')

        tfidf = self.get_tfidf_vec(recommended_df.review)
        print('Text Processing - Got TFIDF')
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
    # *****************************************************************

    def get_data():
        return 'data'