import streamlit as st
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import plotly.express as px 
import seaborn as sns 
from sklearn.preprocessing import StandardScaler
import datetime as dt

st.set_page_config(page_title='Groundwater level prediction',
                   layout="wide")
#Titre et plan de l'application
st.title('Groundwater level prediction')
st.write('This web application has been designed to draft a first work on predicting groundwater levels')
st.write('We have selected 5 different locations in France to work on, you will be able to choose from those during your test of our app')
st.write("""The content is designed in 6 steps :  
                1. Data Geography  
                2. Data Exploration  
                3. Supervised Machine Learning  
                4. Deep Learning  
                5. General conclusions  
                6. References & Credits""")


#Data Gegraphy
st.header('Data Geography')
st.write('In this section we will choose on the map the city we want to explore')
#DATA_PATH = "/Users/dorian.erkens/Desktop/Jedha_FS_Bootcamp/Final Project/Streamlit/Final Project/venday_dataset.csv"
DATA_PATH = " "
cities = ["Athée-sur-Cher, 37","Cleppé, 42", "Vendays-Montalivet, 33","Saint-Jean-de-Védas, 34","Beauvois-en-Vermandois, 02"]
st.sidebar.header('Data Exploration')
localisation = st.sidebar.selectbox("What localisation do you want to see ?",["Athée-sur-Cher, 37","Cleppé, 42", "Vendays-Montalivet, 33","Saint-Jean-de-Védas, 34","Beauvois-en-Vermandois, 02"] )

if localisation == cities[0]: 
    #get the dataset on S3 for the selected city
    DATA_PATH = 'https://erdo-streamlit-911.s3.eu-central-1.amazonaws.com/Dataset_final_project/Full_DataSet.csv'
elif localisation == cities[1]: 
    DATA_PATH = "https://erdo-streamlit-911.s3.eu-central-1.amazonaws.com/Dataset_final_project/cleppe_dataset.csv"
elif localisation == cities[2]: 
    DATA_PATH = "https://erdo-streamlit-911.s3.eu-central-1.amazonaws.com/Dataset_final_project/venday_dataset.csv"
elif localisation == cities[3]: 
    DATA_PATH = "https://erdo-streamlit-911.s3.eu-central-1.amazonaws.com/Dataset_final_project/saint_jean_de_vedas_dataset.csv"
elif localisation == cities[4]: 
    DATA_PATH = "https://erdo-streamlit-911.s3.eu-central-1.amazonaws.com/Dataset_final_project/beauvois-en-vermandois_dataset.csv"

@st.cache
def load_process_data():
    data = pd.read_csv(DATA_PATH)
    data = data.rename(columns={
        'PRECTOT':"Precipitation",
        "TS":'Earth Skin Temperature',
        "RH2M":"Relative Humidity at 2 meters",
        "T2M":"Temperature at 2 meters",
        "WS10M":"Wind speed at 10 meters",
        'ALLSKY_SFC_SW_DWN':"Sky Insolation Incident",
        'PS':'Surface Pressure',
        #'QV2M':'Specific Humidity at 2 meters',
        'T2MWET':'Wet Bulb temperature at 2 meters',
        'T2MDEW':'Dew/Frost point at 2 meters'})
    data['Sky Insolation Incident'] = np.where(data['Sky Insolation Incident']<-10, data['Sky Insolation Incident'].median(),data['Sky Insolation Incident'])
    data['Cote d-1'] = data.Cote.shift(1)
    data = data.drop(data.index[0])
    if localisation == cities[0]:
        data = data.drop(data.index[0:726])
    data =  data[['Precipitation',
    'Earth Skin Temperature',
    'Relative Humidity at 2 meters',
    'Temperature at 2 meters',
    'Wind speed at 10 meters',
    "Sky Insolation Incident",
    "Surface Pressure",
    #'Specific Humidity at 2 meters',
    'Wet Bulb temperature at 2 meters',
    'Dew/Frost point at 2 meters',
    'DateTime',
    'Cote',
    'Cote d-1']]
    data = data.reset_index(drop=True)
    return data
data = load_process_data()


#Data Exploration
st.header('Data Exploration')
st.write('In this section we will explore quick descriptive analytics on the dataset')
st.subheader('A glimpse at the dataset')
st.dataframe(data=data.head())

st.subheader('Quick descriptive analytics for {}'.format(localisation))
st.write('Lorem ipsum')
left_column, right_column = st.beta_columns([2,2])
with left_column :
    cote_time = px.line(data_frame=data, x='DateTime',y='Cote',title='Cote across time')
    st.plotly_chart(cote_time, use_container_width=True)
#Here we will propose on-demand graphs
st.write("You can also explore the dataset on your own, and get a glimpse of inter-variables relationship")
st.sidebar.write('Explore on your own')
def choose_ur_graph():
    x_value = st.sidebar.selectbox("What do you want to see on the x scale ?", data.columns)
    y_value = st.sidebar.selectbox("What do you want to see on the y scale ?", data.columns)
    fig = px.scatter(data_frame=data, x=x_value,y=y_value,title=f'{y_value} in fonction of {x_value}')
    st.plotly_chart(fig,use_container_width=True)
choose_ur_graph()

st.write('And if you want to see the evolution of one of the variables through time, you can do it hereafter')
st.sidebar.write('Trend over time')
variable = st.sidebar.selectbox("What trend across time do you want to explore ?", data.columns.drop("DateTime"))
variable_accross_time = px.line(data_frame=data, x=data.DateTime,y=variable,title=f'{variable} evolution across time')
st.plotly_chart(variable_accross_time,use_container_width=True)

st.subheader('Evaluation of correlations between variables')
st.write('In order to properly analyze your dataset, you should always look at the correlation each variable has between one another')
corr = data.corr()
fig, ax = plt.subplots(figsize=(20, 15))
cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(corr, cmap=cmap,annot=True, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
st.write(fig)

#Supervised Machine Learning
st.header('ARIMA Method')



#Deep Learning
st.header('Rolling in the deep ...')
st.subheader('Deep Neural Network')

st.subheader('Recurrent Neural Network')

st.subheader('LSTM')
