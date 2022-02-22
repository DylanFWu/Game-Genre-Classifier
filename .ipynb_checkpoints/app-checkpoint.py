# text preprocessing modules
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from nltk.corpus import stopwords
from sklearn.preprocessing import MultiLabelBinarizer
import pickle
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

clf = pickle.load(open('model.pkl', 'rb'))
tfidf_vectorizer = pickle.load(open('tfidf.pkl','rb'))
binarizer = pickle.load(open('binarizer.pkl','rb'))


# function for text cleaning 
def clean_text(text):
    # remove backslash-apostrophe 
    text = re.sub("\'", "", text) 
    # remove everything except letters
    text = re.sub("[^a-zA-Z]"," ",text) 
    # remove whitespaces 
    text = ' '.join(text.split()) 
    # convert text to lowercase 
    text = text.lower() 
    
    return text

# function to remove stopwords
stop_words = set(stopwords.words('english'))
stop_words.update(['game','content','play','developed'])


def remove_stopwords(text):
    no_stopword_text = [w for w in text.split() if not w in stop_words]
    return ' '.join(no_stopword_text)


def infer_tags(q):
    q = clean_text(q)
    q = remove_stopwords(q)
    q_vec = tfidf_vectorizer.transform([q])
    q_pred = clf.predict(q_vec)
    return binarizer.inverse_transform(q_pred)


@app.route('/')
def home():
    return "Hello!"


@app.route('/predict',methods=['POST'])

def predict():
    """
    A simple function that receives a game description and predicts the genres of the game.
    :param description:
    :return: predictions
    """
    # clean the description
    cleaned_desc = clean_text(desc)
    final_desc = remove_stopwords(cleaned_desc)
    
    # perform prediction
    prediction = infer_tags(final_desc)
    
    
    # show results
    return prediction


if __name__ == "__main__":
    app.run(debug=True)