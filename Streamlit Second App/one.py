import streamlit as st
#from streamlit_metrics import metric, metric_row
st.set_option('deprecation.showPyplotGlobalUse', False)

import time

import re
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.feature_extraction.text as text
from django.shortcuts import render

import plotly.express as px
#from plotly.subplots import make_subplots
import plotly.graph_objects as go
#import plotly.express as px
#import plotly as py
#import plotly.graph_objs as go

import stylecloud
from nltk.corpus import stopwords
stoplist = stopwords.words('english') + ['though']
from wordcloud import WordCloud, STOPWORDS
from textblob import TextBlob

#https://towardsdatascience.com/text-analysis-basics-in-python-443282942ec5
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import make_pipeline

sys.setrecursionlimit(100000)
#print("Installed Dependencies")

def app():
    st.title("ORCAWISE COLLOCATIONS")
    
    st.header("BI-GRAMS and TRI-GRAMS")
    
    st.subheader("Loading the Text Data....")
    
    file = st.file_uploader("Upload file")
    show_file = st.empty()
    
    if not file:
        show_file.info("Please upload a file of type: " + ", ".join([".txt"]))
        return
    
    label = st.empty()
    bar = st.progress(0)
    
    for i in range(100):
        # Update progress bar with iterations
        label.text(f'Loaded {i+1} %')
        bar.progress(i+1)
        time.sleep(0.01)
      
    ".... and now we're done!!!"
    
    ###########################################################################
    
    content = ""
    for line in file:
        content += str(line) + "\n"
    
    df = pd.DataFrame([content.split('. ')])
    df = df.T
    df.columns = ['Reviews']
    
    if st.checkbox('Show DataFrame'):
        st.dataframe(df.tail())
    
    df['Polarity'] = df['Reviews'].apply(lambda x: TextBlob(x).polarity)
    df['Subjective'] = df['Reviews'].apply(lambda x: TextBlob(x).subjectivity)  
    
    if st.checkbox('Show Sementic DataFrame'):
        st.dataframe(df)
    
    stoplist = stopwords.words('english') + ['though']
    
    ###########################################################################
        
    c_vec = CountVectorizer(stop_words=stoplist, ngram_range=(2,3))
    ngrams = c_vec.fit_transform(df['Reviews']) # matrix of ngrams
    count_values = ngrams.toarray().sum(axis=0) # count frequency of ngrams
    vocab = c_vec.vocabulary_  # list of ngrams
    
    df_ngram = pd.DataFrame(sorted([(count_values[i],k) 
                                    for k,i in vocab.items()], 
                                    reverse=True)).rename(columns={0: 'Frequency', 1:'Bigram/Trigram'})
    
    df_ngram['Polarity'] = df_ngram['Bigram/Trigram'].apply(lambda x: TextBlob(x).polarity)
    df_ngram['Subjective'] = df_ngram['Bigram/Trigram'].apply(lambda x: TextBlob(x).subjectivity)
    
    if st.checkbox("Show Vectorized DataFrame"):
        st.dataframe(df_ngram)
        
    fig = px.bar(df_ngram, x='Bigram/Trigram', y='Frequency',
                 hover_data=['Bigram/Trigram', 'Frequency'], color='Frequency', text='Frequency')
    #fig.update_traces(textfont_size=12, textangle=0, textposition="outside", cliponaxis=False)
    fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')
    
    fig.update_xaxes(title_text = "BIGRAMS/TRIGRAMS", rangeslider_visible=True, showline=True, linewidth=2, linecolor='black', mirror=True)
    fig.update_yaxes(title_text = "FREQUENCY", showline=True, linewidth=2, linecolor='black', mirror=True)
    fig.update_layout(height=700, width=1400, title_text="Contextual Language Processing for BI-GRAMS")
    
    if st.checkbox("Show BI-GRAM Frequency Plot"):
        st.plotly_chart(fig)
    
    """
    # Creating a custom list of stopwords
    customStopwords=list(STOPWORDS) + ['less','Trump','American','politics','country']
     
    wordcloudimage = WordCloud( max_words=50,
                                font_step=2 ,
                                max_font_size=500,
                                stopwords=customStopwords,
                                background_color='black',
                                width=1000,
                                height=720
                              ).generate(df)
     
    plt.figure(figsize=(20,8))
    plt.imshow(wordcloudimage)
    plt.axis("off")"""
    
    #Creating the text variable
    cloud = " ".join(entity for entity in df_ngram['Bigram/Trigram'])
    
    # Creating word_cloud with text as argument in .generate() method
    wordcloud = WordCloud(collocations = False, background_color = 'white').generate(cloud)
    
    # Display the generated Word Cloud
    plt.figure(figsize=(20,5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")    
    #plt.show()
    
    if st.checkbox("Show WordCloud"):
        st.pyplot()

        