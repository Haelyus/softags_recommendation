import datetime
import pandas as pd
import re
from bs4 import BeautifulSoup  
from nltk.corpus import stopwords # Import the stop word list
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
import nltk.data # Download the punkt tokenizer for sentence splitting
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.externals import joblib

def add_stopdigits(l_words, l_stopwords):
    for word in l_words:
        if word.isdigit():
            if word not in l_stopwords:
                l_stopwords.append(word)
    return l_stopwords

def extract_tag(str_text, tag):
    soup = BeautifulSoup(str_text, "html5lib")
    for code in soup.find_all(tag):
        _ = code.extract()
    soup.get_text()
    return soup.get_text().replace('\n', ' ').replace('\t', ' ').strip()

def languages_to_words(s_body):
    s_body = s_body.replace("a+", "aplus")
    s_body = s_body.replace("a++", "app")
    s_body = s_body.replace("a#", "asharp")
    s_body = s_body.replace("abal++", "abalpp")
    s_body = s_body.replace(".net", "dotnet")
    s_body = s_body.replace("c--", "cmm")
    s_body = s_body.replace("c++", "cpp")
    s_body = s_body.replace("c#", "csharp")
    s_body = s_body.replace("f#", "fsharp")
    s_body = s_body.replace("@formula", "atformula")
    s_body = s_body.replace("goto++", "gotopp")
    s_body = s_body.replace("j#", "jsharp")
    s_body = s_body.replace("karel++", "karelpp")
    s_body = s_body.replace("l#", "lsharp")
    s_body = s_body.replace("m++", "mpp")
    s_body = s_body.replace("r++", "rpp")
    s_body = s_body.replace("x++", "xpp")
    
    return s_body

def sof_to_words(body, stemm = False, lemmatize = False):
    # Function to convert a post's body to a string of words
    # The input is a single string (body), and 
    # the output is a single string (a "cleaned" body)

    # 0. Convert to lower case, remove special characters
    clean_body = body.lower()
    clean_body = languages_to_words(clean_body)
    clean_body = clean_body.replace('\n', ' ').replace('\t', ' ')
    clean_body = re.sub(r"[!#$%&\\'()*+,-./:;<=>?@^_`\[\]\"{|}~]", "", clean_body)
    
    # 1. Split into individual words
    words = clean_body.split()
    #
    # 2. In Python, searching a set is much faster than searching
    #   a list, so convert the stop words to a set
    spwords = stopwords.words("english")
    sws = add_stopdigits(words, spwords)
    sws = set(sws)                  
    # 
    # 2. Remove stop words
    no_stopwords = [w for w in words if not w in sws]   
    #
    # 4. English Stemming
    if stemm :
        stemmer = SnowballStemmer("english")
        stemm_words = [stemmer.stem(w) for w in no_stopwords]
    else:
        stemm_words = no_stopwords
    #
    # 5. Lemmatisation
    if lemmatize :
        wordnet_lemmatizer = WordNetLemmatizer()
        lemm_words = [wordnet_lemmatizer.lemmatize(w) for w in stemm_words]
    else:
        lemm_words = stemm_words
    #
    # 6. Join the words back into one string separated by space, 
    # and return the result.
    final_words = []
    for lwords in lemm_words:
        if not lwords.isdigit():
            final_words.append(lwords)
    return( " ".join(final_words))

def clean_bodies(str_text, tag, stem = False, lemmatiz = False):
    # 1. tag's extraction
    str_extract = extract_tag(str_text, tag)
    # 2. cleaning + lemmatization
    # and return the result.
    return sof_to_words(str_extract, stemm = stem, lemmatize = lemmatiz)

def cleaned_question(s_title, s_body):
    cleaned_title = sof_to_words(s_title, False, True)
    #cleaned_body = sof_to_words(s_body, stemm = False, lemmatize = True)
    cleaned_body = clean_bodies(s_body, 'code', False, True)
    return cleaned_title + " " + cleaned_body

def tags_recommendation(s_title, s_body):
    # Nettoyage du titre + corps et concaténation
    my_question = cleaned_question(s_title, s_body)
    # Création du dataframe de question
    df_question = pd.DataFrame(columns=['question'])
    df_question.loc[0] = my_question
    # Création du corpus
    corpus_test = df_question['question'].values
    # Chargement du TfIdfVectorizer pikle
    loaded_vectorizer = joblib.load('pkl/vectorizer.pkl')
    # Entrainement du vectorizer
    X_test = loaded_vectorizer.transform(corpus_test)
    # Chargement du classifier pikle
    loaded_classifier = joblib.load('pkl/classifier.pkl')
    # Prédiction des différents tags
    y_prob = loaded_classifier.decision_function(X_test)
    # define an empty list
    file_to_list = []
    # open file and read the content in a list
    with open('listfile.txt', 'r') as filehandle:
        for line in filehandle:
            # remove linebreak which is the last character of the string
            currentTag = line[:-1]
            # add item to the list
            file_to_list.append(currentTag)
    # Création du dataframe de test
    df_test = pd.DataFrame(y_prob, columns=file_to_list)
    # Liste des tags triés par probabilité
    list_recommended_tags = list(df_test.sort_values(by=0, ascending=False, axis=1).columns.values)[:5]
    return list_recommended_tags