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
        
    def get_imp(bow, mf, ngram):
        tfidf = text.CountVectorizer(ngram_range=(ngram, ngram), max_features=mf, stop_words='english')
        matrix = tfidf.fit_transform(bow)
        return pd.Series(np.array(matrix.sum(axis=0))[0], index=tfidf.get_feature_names()).sort_values(
            ascending=False).head(100)

    #path = st.text_input('CSV file path')
    content = ""
    for line in file:
        content += str(line) + "\n"
    #print(content)
    
    """
    if file is not None:
         # To read file as bytes:
         bytes_data = uploaded_file.getvalue()
         st.write(bytes_data)
    
         # To convert to a string based IO:
         stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
         st.write(stringio)
    
         # To read file as string:
         string_data = stringio.read()
         st.write(string_data)
    
         # Can be used wherever a "file-like" object is accepted:
         dataframe = pd.read_csv(uploaded_file)
         st.write(dataframe)"""
    
        
    total_data_unigram = get_imp(bow=[content], mf=5000, ngram=1)
    total_data_unigram = pd.DataFrame(total_data_unigram).reset_index()
    total_data_unigram.columns = ['entity', 'occurance']
    total_data_unigram = total_data_unigram[total_data_unigram['occurance'] > 1]
    
    """if st.checkbox("Show UNI-GRAM DataFrame"):
        st.dataframe(total_data_unigram)"""
    
    
    dictionary_unigram = dict()
    for entities_unigram, occurances in zip(total_data_unigram['entity'], total_data_unigram['occurance']):
        # this condition will check if digits and spaces are only there in string
        if re.match("^[0-9 ]+$", entities_unigram):
            continue
        dictionary_unigram[entities_unigram] = occurances

    df = pd.DataFrame(dictionary_unigram, index = [0])
    df = df.transpose().reset_index()
    df.columns = ['Entity', 'Occurances']
    
    if st.checkbox("Show BI-GRAM Entity Frequency DataFrame"):
        st.dataframe(df)
    
    """trace = go.Scatter(x=df['Entity'], y=df['Occurances'], mode='lines', name='test')
    layout = go.Layout(title='BI-GRAMS OCCURANCES', plot_bgcolor='rgb(230, 230,230)')
    fig = go.Figure(data=[trace], layout=layout)"""
    
    fig = px.bar(df, x='Entity', y='Occurances',
                 hover_data=['Entity', 'Occurances'], color='Occurances', text='Occurances')
    #fig.update_traces(textfont_size=12, textangle=0, textposition="outside", cliponaxis=False)
    fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')
    
    fig.update_xaxes(title_text = "ENTITIES", rangeslider_visible=True, showline=True, linewidth=2, linecolor='black', mirror=True)
    fig.update_yaxes(title_text = "OCCURANCES", showline=True, linewidth=2, linecolor='black', mirror=True)
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
    cloud = " ".join(entity for entity in df.Entity)
    
    # Creating word_cloud with text as argument in .generate() method
    wordcloud = WordCloud(collocations = False, background_color = 'white').generate(cloud)
    
    # Display the generated Word Cloud
    plt.figure(figsize=(20,5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")    
    #plt.show()
    
    if st.checkbox("Show WordCloud"):
        st.pyplot()
    
    df['Polarity'] = df['Entity'].apply(lambda x: TextBlob(x).polarity)
    df['Subjective'] = df['Entity'].apply(lambda x: TextBlob(x).subjectivity)
    
    if st.checkbox("Show UNI-GRAM DataFrame"):
        st.dataframe(df)
        