import streamlit as st
import numpy as np
import pandas as pd
import praw
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
from datetime import datetime
import plotly.express as px
import os

def init_subreddit():
     reddit = praw.Reddit(
     client_id=os.environ['CLIENT_ID'],
     client_secret=os.environ['CLIENT_SECRET'],
     user_agent=os.environ['USER_AGENT']
     )
     for submission in reddit.subreddit("wallstreetbets").hot(limit=1):
          thread = submission
     return thread

def query_comments(thread, duration):
     duration_seconds = duration*60
     thread.comment_sort = "new"
     # thread.comments.replace_more()
     comments = thread.comments.list()
     first_comment_created = comments[0].created
     for idx in range(len(comments)):
          if ((first_comment_created - comments[idx].created) >= duration_seconds):
               last_comment_idx = idx
               break
     comments_body = [comments[i].body for i in range(last_comment_idx)]
     comments_time = [datetime.strftime(datetime.fromtimestamp(comments[i].created), "%m/%d/%Y, %H:%M:%S") for i in range(last_comment_idx)]
     return comments_body, comments_time

def sentiment_analysis(comments, time, duration):
     tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
     model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
     encoded_input = tokenizer(comments, return_tensors='pt', padding=True, truncation=True, max_length=64)
     output = model(**encoded_input)
     scores = output[0].detach().numpy()
     scores = softmax(scores, axis=1)
     df_sentiment = pd.DataFrame({'comment': comments, 'time': time, 'negative': scores[:,0], 'neutral': scores[:,1], 'positive': scores[:,2]})
     df_sentiment["time"] = pd.to_datetime(df_sentiment["time"])
     df_sentiment.set_index('time', inplace=True)
     df_sentiment = df_sentiment.resample(str(duration*60//15)+"S").mean()
     return df_sentiment

st.title("WSB Daily Discussion Sentiment Analysis")

duration = st.slider("Duration in minutes", 0, 60, 10)

col1, col2, col3 = st.columns([1, 1, 1])
with col2:
     calc = st.button('Analyze Sentiment')

if calc:
     thread = init_subreddit()
     st.write(" **Thread Title:** ", thread.title)
     with st.spinner("Querying comments within the last " + str(duration) + " minute(s)..."):
          c_body, c_time = query_comments(thread, duration)

     with st.spinner("Analyzing sentiment of comments..."):
          df = sentiment_analysis(c_body, c_time, duration)

          col1, col2, col3 = st.columns(3)
          col1.metric("Positive", str(round(df['positive'][-1]*100))+"%",  str(round((df['positive'][-1]-df['positive'][0])*100))+"%")
          col2.metric("Neutral",  str(round(df['neutral'][-1]*100))+"%", str(round((df['neutral'][-1]-df['neutral'][0])*100))+"%")
          col3.metric("Negative",  str(round(df['negative'][-1]*100))+"%", str(round((df['negative'][-1]-df['negative'][0])*100))+"%")

          fig = px.line(df, x=df.index, y = ["negative", "neutral", "positive"],
          color_discrete_sequence = ["red", "gray", "green"],
          title="Sentiment Time Series",
          labels = {
               "negative": "Negative",
               "positive": "Positive",
               "neutral": "Neutral",
          }
          )
          fig.update_layout({
          'plot_bgcolor': 'rgba(0, 0, 0, 0)',
          'paper_bgcolor': 'rgba(0, 0, 0, 0)',
          })   
          st.plotly_chart(fig)
