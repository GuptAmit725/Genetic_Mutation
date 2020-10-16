
# importing the necessary dependencies
import re
#import streamlit
from sklearn.feature_extraction.text import TfidfVectorizer
from flask import Flask, render_template, request,jsonify
from flask_cors import CORS,cross_origin
import pickle
import numpy as np
from nltk.corpus import stopwords
import numpy as np
import pickle
import pandas as pd
# from flasgger import Swagger
import streamlit as st

from PIL import Image

# app=Flask(__name__)
# Swagger(app)

#pickle_in = open("classifier.pkl", "rb")
#classifier = pickle.load(pickle_in)

'''
#@app.route('/')
def welcome():
    return "Welcome All"


#@app.route('/predict',methods=["Get"])
def predict_gene_mutation(Gene, Variation, TEXT):
    import nltk
    nltk.download('stopwords')
    # loading stop words from nltk library
    stop_words = set(stopwords.words('english'))

    def nlp_preprocessing(total_text):

        string = ""
        # replace every special char with space
        total_text = re.sub('[^a-zA-Z0-9\n]', ' ', total_text)
        # replace multiple spaces with single space
        total_text = re.sub('\s+', ' ', total_text)
        # converting all the chars into lower-case.
        total_text = total_text.lower()

        for word in total_text.split():
            # if the word is a not a stop word then retain that word from the data
            if not word in stop_words:
                string += word + " "
        total_text = string
        return total_text

    TEXT = nlp_preprocessing(TEXT)
    # Vectorizing
    filename_text = 'text_final_model.sav'
    filename_var = 'var_final_model.sav'
    filename_gene = 'gene_final_model.sav'
    filename_model = 'final_model.sav'
    loaded_model_text = pickle.load(open(filename_text, 'rb'))
    loaded_model_var = pickle.load(open(filename_var, 'rb'))
    loaded_model_gene = pickle.load(open(filename_gene, 'rb'))
    loaded_model = pickle.load(open(filename_model, 'rb'))
    print('vectorizing----')
    text_vec = loaded_model_text.transform([TEXT]).toarray()
    var_vec = loaded_model_var.transform([Variation]).toarray()
    gene_vec = loaded_model_gene.transform([Gene]).toarray()
    final_vec = np.hstack((gene_vec, var_vec, text_vec))[:, 0:2232]

    print('below np')
    loaded_model = pickle.load(open(filename_model, 'rb'))  # loading the model file from the storage

    # predictions using the loaded model file
    print(final_vec.shape)
    prediction = loaded_model.predict(final_vec)
    prediction_prob = loaded_model.predict_proba(final_vec)
    print('prediction is', prediction)
    print('prediction probability is', prediction_prob[0][prediction[0] - 1])
    # showing the prediction results in a UI
    return prediction,prediction_prob

'''

app = Flask(__name__) # initializing a flask app

@app.route('/',methods=['GET'])  # route to display the home page
@cross_origin()
def homePage():
    return render_template("index.html")

@app.route('/predict',methods=['POST','GET']) # route to show the predictions in a web UI
@cross_origin()
def index():
    if request.method == 'POST':
        try:
            #  reading the inputs given by the user
            Gene=request.form['Gene']
            Variation = request.form['Variation']
            TEXT = request.form['TEXT']

            #Doing text-preprocessing of inputs.
            import nltk
            nltk.download('stopwords')
            # loading stop words from nltk library
            stop_words = set(stopwords.words('english'))

            def nlp_preprocessing(total_text):

                    string = ""
                    # replace every special char with space
                    total_text = re.sub('[^a-zA-Z0-9\n]', ' ', total_text)
                    # replace multiple spaces with single space
                    total_text = re.sub('\s+', ' ', total_text)
                    # converting all the chars into lower-case.
                    total_text = total_text.lower()

                    for word in total_text.split():
                        # if the word is a not a stop word then retain that word from the data
                        if not word in stop_words:
                            string += word + " "
                    total_text = string
                    return total_text

            TEXT = nlp_preprocessing(TEXT)
            # Vectorizing
            filename_text = 'text_final_model.sav'
            filename_var = 'var_final_model.sav'
            filename_gene = 'gene_final_model.sav'
            filename_model = 'final_model.sav'
            loaded_model_text = pickle.load(open(filename_text, 'rb'))
            loaded_model_var = pickle.load(open(filename_var, 'rb'))
            loaded_model_gene = pickle.load(open(filename_gene, 'rb'))
            loaded_model = pickle.load(open(filename_model, 'rb'))
            print('vectorizing----')
            text_vec = loaded_model_text.transform([TEXT]).toarray()
            var_vec = loaded_model_var.transform([Variation]).toarray()
            gene_vec = loaded_model_gene.transform([Gene]).toarray()
            final_vec = np.hstack((gene_vec,var_vec,text_vec))[:,0:2232]

            print('below np')
            loaded_model = pickle.load(open(filename_model, 'rb')) # loading the model file from the storage

            # predictions using the loaded model file
            print(final_vec.shape)
            prediction=loaded_model.predict(final_vec)
            prediction_prob = loaded_model.predict_proba(final_vec)
            print('prediction is', prediction)
            print('prediction probability is', prediction_prob[0][prediction[0]-1])
            # showing the prediction results in a UI
            pred_df = pd.DataFrame({'Gene':[1,2,3,4,5,6,7,8,9],'probabilities':prediction_prob[0]}).to_html()
            return render_template('results.html',prediction=prediction[0],prediction_prob=prediction_prob[0][prediction[0]-1],
                                   one=prediction_prob[0][0],two=prediction_prob[0][1],three=prediction_prob[0][2],
                                   four=prediction_prob[0][3],five=prediction_prob[0][4],six=prediction_prob[0][5],
                                   seven=prediction_prob[0][6],eight=prediction_prob[0][7],nine=prediction_prob[0][8])
        except Exception as e:
            print('The Exception message is: ',e)
            return 'something is wrong'
    # return render_template('results.html')
    else:
        return render_template('index.html')



if __name__ == "__main__":
    app.run(host='127.0.0.1', port=8001, debug=True)
    app.run(debug=True) # running the app