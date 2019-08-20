#!/usr/bin/env python
# coding: utf-8
'''
    Code for Summarize Text
    author : Aditya Perwira Joan Dwitama
'''

from gensim.summarization.summarizer import summarize
from gensim.summarization import keywords
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
import nltk
# nltk.download('all')

text = open("end game.txt", "r").read()


def get_summary(text, pct):
    summary = summarize(text, ratio=pct, split=False)
    return summary


def get_keywords(text):
    res = keywords(text, ratio=0.1, words=None, split=False, scores=False,
                   pos_filter=('NN', 'JJ'), lemmatize=True, deacc=True)
    res = res.split('\n')
    return res

print('The Input Text')
print('--------------------------')
print(text)
print('--------------------------')
print('Printing Summary')
print('--------------------------')
print(get_summary(text, 0.3))
print('-------------------------')
print('Printing Keywords')
print('--------------------------')
print(get_keywords(text))
