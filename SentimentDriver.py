#!/usr/bin/env python3

import csv
import os
import json
import pandas as pd
import numpy as np
from nltk.stem import WordNetLemmatizer
import nltk
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
import re
import spacy
from nltk.corpus import sentiwordnet as swn
from IPython.display import clear_output
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import plotly
import nltk
import ssl
from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

lemmatizer = WordNetLemmatizer()
clean_file = 'CLEAN_TweetData.csv'

def main():
	data = import_data(clean_file)
	data = preprocess(data)
	data = make_sentiment(data)
	write_sent(clean_file, data)
			
def write_sent(filename, data):
	data2 = pd.read_csv(filename)
	senti_score = data['senti_score'].copy()
	data2['senti_score'] = senti_score
	data2.to_csv(filename)
			
def import_data(filename):
	data2 = pd.read_csv(filename)
	data = data2[["full_text"]].copy()
	edited_response = data['full_text'].copy()
	data['Edited_Response'] = edited_response
	return data
	
def preprocess(data):
	preprocess_Reviews_data(data,'Edited_Response')
	rem_stopwords_tokenize(data,'Edited_Response')
	make_sentences(data,'Edited_Response')
	final_Edit = data['Edited_Response'].copy()
	data["After_lemmatization"] = final_Edit
	Lemmatization(data,'After_lemmatization')
	make_sentences(data,'After_lemmatization')
	return data
	
def make_sentiment(data):
	data = make_pos_tags(data)
	data = accum_sent(data)
	return data

def make_pos_tags(data):
	postagging = []
	for review in data['After_lemmatization']:
		list = word_tokenize(review)
		postagging.append(nltk.pos_tag(list))
	data['pos_tags'] = postagging
	return data
	
def accum_sent(data):
	pos = neg = 0
	num_words = 0
	senti_score = []
	for pos_val in data['pos_tags']:
		senti_val = [get_sentiment(x,y) for (x,y) in pos_val]
		for score in senti_val:
			try:
				pos = pos + score[1]
				neg = neg + score[2]
				num_words = num_words + 1
			except:
				continue
		if(num_words != 0):		
			senti_score.append(round((pos - neg)/num_words, 6))
		else:
			senti_score.append(round((pos - neg), 6))
		pos = neg = 0
		num_words = 0
	data['senti_score'] = senti_score
	return data
		
def preprocess_Reviews_data(data,name):
    data[name]=data[name].str.lower()
    data[name]=data[name].apply(lambda x:re.sub(r'\B#\S+','',x))
    data[name]=data[name].apply(lambda x:re.sub(r"http\S+", "", x))
    data[name]=data[name].apply(lambda x:' '.join(re.findall(r'\w+', x)))
    data[name]=data[name].apply(lambda x:re.sub(r'\s+', ' ', x, flags=re.I))
    data[name]=data[name].apply(lambda x:re.sub(r'\s+[a-zA-Z]\s+', '', x))
    data[name]=data[name].apply(lambda x:re.sub('@[^\s]+','',x))
    
def penn_to_wn(tag):
    if tag.startswith('J'):
        return wn.ADJ
    elif tag.startswith('N'):
        return wn.NOUN
    elif tag.startswith('R'):
        return wn.ADV
    elif tag.startswith('V'):
        return wn.VERB
    return None

def get_sentiment(word,tag):
    wn_tag = penn_to_wn(tag)
    if wn_tag not in (wn.NOUN, wn.ADJ, wn.ADV):
        return []
    lemma = lemmatizer.lemmatize(word, pos=wn_tag)
    if not lemma:
        return []   
    synsets = wn.synsets(word, pos=wn_tag)
    if not synsets:
        return []
    synset = synsets[0]
    swn_synset = swn.senti_synset(synset.name())
    return [synset.name(), swn_synset.pos_score(),swn_synset.neg_score(),swn_synset.obj_score()]
    pos=neg=obj=count=0

def rem_stopwords_tokenize(data,name):
      
    def getting(sen):
        example_sent = sen
        filtered_sentence = []
        stop_words = set(stopwords.words('english')) 
        word_tokens = word_tokenize(example_sent) 
        filtered_sentence = [w for w in word_tokens if not w in stop_words] 
        return filtered_sentence

    x=[]
    for i in data[name].values:
        x.append(getting(i))
    data[name]=x

def Lemmatization(data,name):
    def getting2(sen):
        example = sen
        output_sentence =[]
        word_tokens2 = word_tokenize(example)
        lemmatized_output = [lemmatizer.lemmatize(w) for w in word_tokens2]
        without_single_chr = [word for word in lemmatized_output if len(word) > 2]
        cleaned_data_title = [word for word in without_single_chr if not word.isnumeric()]
        return cleaned_data_title

    x=[]
    for i in data[name].values:
        x.append(getting2(i))
    data[name]=x
    
def make_sentences(data,name):
    data[name]=data[name].apply(lambda x:' '.join([i+' ' for i in x]))
    data[name]=data[name].apply(lambda x:re.sub(r'\s+', ' ', x, flags=re.I))
	
if __name__ == '__main__':
    main()
    

#Open each csv sequentially, clean it, carry out sentiment analysis, write to csvs
