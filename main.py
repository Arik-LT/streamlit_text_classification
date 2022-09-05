import pandas as pd
import pickle

import streamlit as st


from functions import cleaning_text, removing_single_letters

st.title('Email Text Classification')

text = st.text_area('Input your email here: ')

with open('tfidf.pkl', 'rb') as ff:
    tfidf = pickle.load(ff)

with open('finalized_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Changing text to dataframe to clean
df_exp = pd.DataFrame({'text': [text]})
df_exp['text'] = df_exp["text"].apply(cleaning_text)
df_exp['text'] = df_exp["text"].apply(removing_single_letters)

text_for_model = df_exp.iloc[0, 0]

#st.write('Clean Text: ')
# st.write(text_for_model)

input = tfidf.transform([text_for_model]).toarray()

result = model.predict(input).toarray()

if len(text.split()) > 1:  
    st.subheader('This email belongs to the following classes:')
    if result[0][0] == 1:
        st.write('· Politics')
    if result[0][1] == 1:
        st.write('· Entertainment')
    if result[0][2] == 1:
        st.write('· Science')
    if result[0][3] == 1:
        st.write('· Crime')
else:
    pass