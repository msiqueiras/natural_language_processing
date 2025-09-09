import re, string
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import nltk
nltk.download('stopwords') 
nltk.download('rslp')

import spacy
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer
from unidecode import unidecode

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

from itertools import chain
nlp = spacy.load("pt_core_news_sm")

PT_STOPWORDS = set(stopwords.words("portuguese"))
STEMMER = RSLPStemmer()

plt.rcParams["figure.figsize"] = (7,4)