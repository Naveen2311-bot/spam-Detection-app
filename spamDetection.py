import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import streamlit as st
import time as t
import os

st.image("Image.jpg",width=700)

data = pd.read_csv("D:\Project Spam\spam.csv")

data.drop_duplicates(inplace=True)
data['Category'] = data['Category'].replace(['ham','Spam'],['Not Spam','Spam'])

mess = data['Message']
cat = data['Category']

(mess_train,mess_test,cat_train,cat_test) = train_test_split(mess, cat, test_size=0.2)

cv =CountVectorizer(stop_words='english')
features = cv.fit_transform(mess_train)

#creating Model

model = MultinomialNB()
model.fit(features, cat_train)

#Test Our model
features_test = cv.transform(mess_test)
#print(model.score(features_test,cat_test))
#predict Data
def predict(message):
    input_message = cv.transform([message]).toarray()
    result = model.predict(input_message)
    return result

st.header('E-mail Spam Detection')
st.text('Developer : Naveen Kumar Thawait')

input_mess = st.text_input('Enter Message Here')

if st.button('Validate'):
    output = predict(input_mess)
    st.markdown(output)

# Streamlit UI
st.sidebar.header("Welcome to My Project")
st.sidebar.text("Share your experience!")
name = st.sidebar.text_input("Enter your name:")
comment = st.sidebar.text_area("Write your review:")
st.sidebar.button("Submit Review")

    


    