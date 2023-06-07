import streamlit as st
import pandas as pd
import re
import numpy as np
import joblib
import altair as alt

# sidebar
with st.sidebar:
    st.header('Pilih Jenis Algoritma')
    option = st.selectbox(
        "",
        ('Multinomial Naive Bayes', 'Decision Tree'))
    if option == 'Multinomial Naive Bayes':
        st.write("Algoritma Multinomial Naive Bayes")
    else:
        st.write("Algoritma Decision Tree")
    TP = st.checkbox('Tuning Hyperparameter')

st.title("Klasifikasi Emosi Twitter")
if option == 'Multinomial Naive Bayes':
#MNB dengan Tuning hyperparameter dan SMOTE   
    if TP :
        pilih = st.radio(
            "",
            ('Bag of Word', 'Bigram'))
        SMOTE = st.checkbox('SMOTE')
        if SMOTE :
            if pilih == 'Bag of Word':
                st.write("Multinomial Naive Bayes dengan feature extraction Bag of Word")
                st.write("Mode SMOTE")
                st.write("Tuning Hyperparameter : alpha = 1.5")
                st.header("Masukkan tweet:")
                name = st.text_input(" ")
                teks = name.title()
                model = joblib.load(open('model7_HP_SM.pkl', 'rb'))
                fe = joblib.load(open('bow_SM.pkl', 'rb'))
            else :
                st.write("Multinomial Naive Bayes dengan feature extraction Bigram ")
                st.write("Mode SMOTE")
                st.write("Tuning Hyperparameter : alpha = 5.0")
                st.header("Masukkan tweet:")
                name = st.text_input(" ")
                teks = name.title()
                model = joblib.load(open('model8_HP_SM.pkl', 'rb'))
                fe = joblib.load(open('bigram_SM.pkl', 'rb'))
#MNB dengan Tuning hyperparameter dan tanpa SMOTE  
        else:  
            if pilih == 'Bag of Word':
                st.write("Multinomial Naive Bayes dengan feature extraction Bag of Word")
                st.write("Parameter Default : alpha = 0.9")
                st.header("Masukkan tweet:")
                name = st.text_input(" ")
                teks = name.title()
                model = joblib.load(open('model3_HP.pkl', 'rb'))
                fe = joblib.load(open('bow.pkl', 'rb'))
            else :
                st.write("Multinomial Naive Bayes dengan feature extraction Bigram ")
                st.write("Parameter Default : alpha = 5.0")
                st.header("Masukkan tweet:")
                name = st.text_input(" ")
                teks = name.title()
                model = joblib.load(open('model4_HP.pkl', 'rb'))
                fe = joblib.load(open('bigram.pkl', 'rb'))
#MNB SMOTE dan tanpa Tuning hyperparameter 
    else: 
        pilih = st.radio(
            "",
            ('Bag of Word', 'Bigram'))
        SMOTE = st.checkbox('SMOTE')
        if SMOTE :
            if pilih == 'Bag of Word':
                st.write("Multinomial Naive Bayes dengan feature extraction Bag of Word")
                st.write("Mode SMOTE")
                st.write("Parameter Default : alpha = 1")
                st.header("Masukkan tweet:")
                name = st.text_input(" ")
                teks = name.title()
                model = joblib.load(open('model5_SM.pkl', 'rb'))
                fe = joblib.load(open('bow_SM.pkl', 'rb'))
            else :
                st.write("Multinomial Naive Bayes dengan feature extraction Bigram ")
                st.write("Mode SMOTE")
                st.write("Parameter Default : alpha = 1")
                st.header("Masukkan tweet:")
                name = st.text_input(" ")
                teks = name.title()
                model = joblib.load(open('model6_SM.pkl', 'rb'))
                fe = joblib.load(open('bigram_SM.pkl', 'rb'))
#MNB Default
        else: 
            if pilih == 'Bag of Word':
                st.write("Multinomial Naive Bayes dengan feature extraction Bag of Word")
                st.write("Parameter Default : alpha = 1")
                st.header("Masukkan tweet:")
                name = st.text_input(" ")
                teks = name.title()
                model = joblib.load(open('model1.pkl', 'rb'))
                fe = joblib.load(open('bow.pkl', 'rb'))
            else:
                st.write("Multinomial Naive Bayes dengan feature extraction Bigram ")
                st.write("Parameter Default : alpha = 1")
                st.header("Masukkan tweet:")
                name = st.text_input(" ")
                teks = name.title()
                model = joblib.load(open('model2.pkl', 'rb'))
                fe = joblib.load(open('bigram.pkl', 'rb'))
#Decision Tree
else:
    if TP : 
        pilih = st.radio(
            "",
            ('Bag of Word', 'Bigram'))
        SMOTE = st.checkbox('SMOTE')
#DT dengan Tuning hyperparameter dan SMOTE 
        if SMOTE :
            if pilih == 'Bag of Word':
                st.write("Decision Tree dengan feature extraction Bag of Word")
                st.write("Mode SMOTE")
                st.write("Tuning Hyperparameter : max_depth=100 , min_samples_leaf=9 ")
                st.header("Masukkan tweet:")
                name = st.text_input(" ")
                teks = name.title()
                model = joblib.load(open('model15_HP_SM.pkl', 'rb'))
                fe = joblib.load(open('bow_SM.pkl', 'rb'))
            else :
                st.write("Decision Tree dengan feature extraction Bigram ")
                st.write("Mode SMOTE")
                st.write("Tuning Hyperparameter : max_depth=160, min_samples_leaf=11")
                st.header("Masukkan tweet:")
                name = st.text_input(" ")
                teks = name.title()
                model = joblib.load(open('model16_HP_SM.pkl', 'rb'))
                fe = joblib.load(open('bigram_SM.pkl', 'rb'))
#DT dengan Tuning hyperparameter dan tanpa SMOTE  
        else:  
            if pilih == 'Bag of Word':
                st.write("Decision Tree dengan feature extraction Bag of Word")
                st.write("Parameter Default : max_depth=100, min_samples_leaf= 15")
                st.header("Masukkan tweet:")
                name = st.text_input(" ")
                teks = name.title()
                model = joblib.load(open('model11_HP.pkl', 'rb'))
                fe = joblib.load(open('bow.pkl', 'rb'))
            else:
                st.write("Decision Tree dengan feature extraction Bigram ")
                st.write("Parameter Default : max_depth= 160, min_samples_leaf= 11")
                st.header("Masukkan tweet:")
                name = st.text_input(" ")
                teks = name.title()
                model = joblib.load(open('model12_HP.pkl', 'rb'))
                fe = joblib.load(open('bigram.pkl', 'rb'))
#DT SMOTE dan tanpa Tuning hyperparameter 
    else: 
        pilih = st.radio(
            "",
            ('Bag of Word', 'Bigram'))
        SMOTE = st.checkbox('SMOTE')
        if SMOTE :
            if pilih == 'Bag of Word':
                st.write("Decision Tree dengan feature extraction Bag of Word")
                st.write("Mode SMOTE")
                st.write("Parameter Default : max_depth= None , min_samples_leaf= 1")
                st.header("Masukkan tweet:")
                name = st.text_input(" ")
                teks = name.title()
                model = joblib.load(open('model13_SM.pkl', 'rb'))
                fe = joblib.load(open('bow_SM.pkl', 'rb'))
            else:
                st.write("Decision Tree dengan feature extraction Bigram ")
                st.write("Mode SMOTE")
                st.write("Parameter Default : max_depth= None , min_samples_leaf= 1")
                st.header("Masukkan tweet:")
                name = st.text_input(" ")
                teks = name.title()
                model = joblib.load(open('model14_SM.pkl', 'rb'))
                fe = joblib.load(open('bigram_SM.pkl', 'rb'))
#DT Default
        else: 
            if pilih == 'Bag of Word':
                st.write("Decision Tree dengan feature extraction Bag of Word")
                st.write("Parameter Default : max_depth= None , min_samples_leaf= 1")
                st.header("Masukkan tweet:")
                name = st.text_input(" ")
                teks = name.title()
                model = joblib.load(open('model9.pkl', 'rb'))
                fe = joblib.load(open('bow.pkl', 'rb'))
            else:
                st.write("Decision Tree dengan feature extraction Bigram ")
                st.write("Parameter Default : max_depth= None , min_samples_leaf= 1")
                st.header("Masukkan tweet:")
                name = st.text_input(" ")
                teks = name.title()
                model = joblib.load(open('model10.pkl', 'rb'))
                fe = joblib.load(open('bigram.pkl', 'rb'))
        
def preprocessing(tweet):
    tweet = data_cleaning(tweet)
    tweet = case_folding(tweet)
    tweet = normalisasi_kata(tweet)
    return tweet

def data_cleaning(tweet):
    # remove special characters 
    tweet = re.sub('[^ a-zA-Z0-9]', ' ', tweet)
    # remove number
    tweet = re.sub(r'[0-9]+', '', tweet)
    #remove url 
    tweet = re.sub("url", "", tweet)
    #remove username 
    tweet = re.sub('username', '', tweet)
    #remove sensitiveno
    tweet = re.sub('sensitiveno', '', tweet)
    #remove URL 
    tweet = re.sub("https", "", tweet)
    # remove askfm
    tweet = re.sub('askfm', '', tweet)
    #remove double spasi
    tweet= " ".join(tweet.split()) 
    return tweet

def case_folding(tweet): 
    case_fold=tweet.lower()
    return case_fold

def normalisasi_kata(tweet):
    key_norm = pd.read_csv('kamus_normalisasi.csv')
    tweet = ' '.join([key_norm[key_norm['singkat'] == word]['hasil'].values[0] 
                      if (key_norm["singkat"] == word).any() else word for word in tweet.split()])
    return tweet

tweets = preprocessing(teks)
data = fe.transform([tweets])
hasil = model.predict(data)
hasil1 = " ".join(hasil)
hasil_proba = model.predict_proba(data)
if st.button('Mulai'):
     st.write('**Raw Tweet :**')
     st.write(name)
     st.write('**Preprocessing Tweet :**')
     st.write(tweets)
     st.write('**Hasil kelas emosi adalah :**',hasil1)
     st.write('**Nilai probabilitas :**')
     df0 = pd.DataFrame(hasil_proba, columns=model.classes_)
     st.dataframe(df0)