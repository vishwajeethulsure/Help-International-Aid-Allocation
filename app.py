import numpy as np
import pandas as pd
import streamlit as st
import pickle

# Load all the nstances that are required
with open('model.pkl','rb') as file:
    model = pickle.load(file)

with open('scaler.pkl','rb') as file:
    scaler = pickle.load(file)


with open('pca.pkl','rb') as file:
    pca = pickle.load(file)



def prediction(input_data):

    scaled_data = scaler.transform(input_data)
    pca_data = pca.transform(scaled_data)
    pred = model.predict(pca_data)[0]

    if pred == 0:
        return 'Developed'
    elif pred == 1:
        return 'Developing'
    else:
        return 'Under Developed'
    



def main():

    st.title('HELP International Foundation')
    st.subheader('This application helps to classify the country on the basis of its socio-economic and health factors')
    chld_mor = st.text_input('Enter child mortaity rate')
    lf_exp = st.text_input('Enter average life expectancy')
    tol_fer = st.text_input('Enter total fertility rate')
    health = st.text_input('Enter the % of GDP spent on health')
    export = st.text_input('Enter the % of GDP spent on exports')
    imports = st.text_input('Enter the % of GDP spent on imports')
    gdp = st.text_input('Enter GDP per population')
    income = st.text_input('Enter the income per person')
    infl = st.text_input('Enter the inflation rate')

    inp_list = [[chld_mor, export, health,imports,income,infl,lf_exp,tol_fer,gdp]]

    if st.button('Predict'):
        response = prediction(inp_list)
        st.success(response)

if __name__ == '__main__':
    main()
