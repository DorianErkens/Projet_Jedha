from numpy.core.fromnumeric import size
import streamlit as st
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import plotly.express as px 
import seaborn as sns 
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
"""import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense,Dropout,Embedding
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import Accuracy
import tensorflow_addons as tfa
from tensorflow_addons.metrics import RSquare
"""
import os
import IPython
import IPython.display
from datetime import date,datetime
import datetime as dt

st.set_page_config(page_title='Groundwater level prediction - Data Project',
                   layout="wide" )
#Titre et plan de l'application
st.title('Groundwater level prediction')
st.write('This web application has been designed to draft a first work on predicting groundwater levels')
st.write('We have selected 5 different locations in France to work on, you will be able to choose from those during your test of our app')
st.write("""The content is designed in 5 steps :  
                1. Data Geography  
                2. Data Exploration  
                3. Supervised Machine Learning  
                4. Deep Learning    
                5. References & Credits""")


#Data Gegraphy
st.header('Data Geography')
st.write('In this section we will choose on the map the city we want to explore')
#DATA_PATH = "/Users/dorian.erkens/Desktop/Jedha_FS_Bootcamp/Final Project/Streamlit/Final Project/venday_dataset.csv"
DATA_PATH = " "
geo_data = pd.DataFrame({
    'City' : ["Athée-sur-Cher, 37","Cleppé, 42", "Vendays-Montalivet, 33","Saint-Jean-de-Védas, 34","Beauvois-en-Vermandois, 02"],
    'Lat' : [47.32111,45.77081,45.35601,43.572309999999995,49.842009999999995],
    'Lon' : [0.91541,4.18431,-1.05959,3.85371,3.10031],
    'Size' : [1]*5})
#st.dataframe(geo_data)
def two_side_graph():
    left_column, right_column = st.beta_columns([4,2])
    with left_column :
        st.subheader("All dataset locations on the map of France")
        fig = px.scatter_mapbox(data_frame=geo_data, lat="Lat", lon="Lon",color="City",mapbox_style="open-street-map",
        color_continuous_scale=px.colors.cyclical.IceFire, size_max=10,zoom=4,size="Size")
        st.plotly_chart(fig, use_container_width=True)
    with right_column:
        st.subheader('Clustering groundwater in France')
        st.write(" \n Source : BRGM ")
        img = "https://cdn-s-www.dna.fr/images/44BFDCCC-D30B-4464-9840-360F0567C6C1/NW_raw/infographie-brgm-1557918355.jpg"
        st.image(img)
two_side_graph()

st.sidebar.header('Data Geography')
localisation = st.sidebar.selectbox("What localisation do you want to see ?",["Athée-sur-Cher, 37","Cleppé, 42", "Vendays-Montalivet, 33","Saint-Jean-de-Védas, 34","Beauvois-en-Vermandois, 02"] )

if localisation == geo_data.City[0]: 
    #get the dataset on S3 for the selected city
    DATA_PATH = 'https://erdo-streamlit-911.s3.eu-central-1.amazonaws.com/Dataset_final_project/Full_DataSet.csv'
elif localisation == geo_data.City[1]: 
    DATA_PATH = "https://erdo-streamlit-911.s3.eu-central-1.amazonaws.com/Dataset_final_project/cleppe_dataset.csv"
elif localisation == geo_data.City[2]: 
    DATA_PATH = "https://erdo-streamlit-911.s3.eu-central-1.amazonaws.com/Dataset_final_project/venday_dataset.csv"
elif localisation == geo_data.City[3]: 
    DATA_PATH = "https://erdo-streamlit-911.s3.eu-central-1.amazonaws.com/Dataset_final_project/saint_jean_de_vedas_dataset.csv"
elif localisation == geo_data.City[4]: 
    DATA_PATH = "https://erdo-streamlit-911.s3.eu-central-1.amazonaws.com/Dataset_final_project/beauvois-en-vermandois_dataset.csv"

@st.cache()
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
    if localisation == geo_data.City[0]:
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
    #process specifically the dataset for ml purposes
    data_ml = data.copy()
    data_ml.DateTime = pd.to_datetime(data.DateTime)
    data_ml.DateTime = pd.to_numeric(data_ml.DateTime)
    data_ml = data_ml.reset_index(drop=True)
    
    return data,data_ml
data,data_ml = load_process_data()

st.sidebar.header('Data Exploration')
#Data Exploration
st.header('Data Exploration')
st.write('In this section we will explore quick descriptive analytics on the dataset')
st.subheader('A glimpse at the dataset')
st.dataframe(data=data.head())

st.subheader('Descriptive analytics for {}'.format(localisation))
st.write('Hereafter, you can have some informations about the dataset and the relationship and behaviors of some of its features ')
def cote_time():
    st.subheader('State of groundwater across time')
    cote_time = px.line(data_frame=data, x='DateTime',y='Cote')
    st.plotly_chart(cote_time, use_container_width=True)
cote_time()
#Here we will propose on-demand graphs
st.write("You can also explore the dataset on your own, and get a glimpse of inter-variables relationship")
st.sidebar.write('Explore on your own')
def choose_ur_graph():
    x_value = st.sidebar.selectbox("What do you want to see on the x scale ?", data.columns)
    y_value = st.sidebar.selectbox("What do you want to see on the y scale ?", data.columns)
    #x,y = data[x_value], data[y_value]
    df = data[[x_value,y_value]]
    corr = df.corr()
    corr_nb = round(corr.iloc[0,1],4)
    st.subheader(f'{y_value} in fonction of {x_value} - Correlation of {corr_nb}')
    fig = px.scatter(data_frame=data, x=x_value,y=y_value)
    st.plotly_chart(fig,use_container_width=True)
choose_ur_graph()

def time_graph():
    st.write('And if you want to see the evolution of one of the variables through time, you can do it hereafter')
    st.sidebar.write('Trend over time')
    variable = st.sidebar.selectbox("What trend across time do you want to explore ?", data.columns.drop("DateTime"))
    variable_accross_time = px.line(data_frame=data, x=data.DateTime,y=variable)
    st.subheader(f'{variable} evolution across time')
    st.plotly_chart(variable_accross_time,use_container_width=True)
time_graph()

def heatmap():
    st.subheader('Evaluation of correlations between variables')
    st.write('In order to properly analyze your dataset, you should always look at the correlation each variable has between one another')
    corr = data.corr()
    fig, ax = plt.subplots(figsize=(20, 15))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(corr, cmap=cmap,annot=True, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    st.write(fig)
heatmap()

#Supervised Machine Learning
st.header("Supervised Machine Learning")
#Function that will preprocess the dataset to be properly managed with ML/DL algo, namely Date





st.subheader('ARIMA Method')



#Deep Learning
st.header('Groundwater level prediction')
st.subheader('Predictions are based on a single-shot multi-steps LSTM model ')
st.write('In order to properly take previous observations into account, we have decided to based our model on RNN deep learning algorithms')
#Processing for deep learning

#Split the dataset into three : train, validation and test
column_indices = {name: i for i,name in enumerate(data_ml.columns)}
n=len(data_ml)
train_df = data_ml[0:int(n*0.7)]
val_df = data_ml[int(n*0.7):int(n*.9)]
test_df = data_ml[int(n*.9):]
num_features = data_ml.shape[1]

#Normalize the dataset, using only training datas as you can only use this ds since others cannot be seen by the model to properly work
#La moyenne et l'écart type doivent être calculés uniquement à l'aide des données d'apprentissage afin que les modèles n'aient pas accès aux valeurs des ensembles de validation et de test.
train_mean = train_df.mean()
train_std = train_df.std()

train_df = (train_df - train_mean)/train_std
valid_df = (val_df - train_mean)/train_std
test_df = (test_df - train_mean)/train_std

df_std = (data_ml - train_mean) / train_std
df_std = df_std.melt(var_name='Column', value_name='Normalized')

class WindowGenerator():
    def __init__(self, input_width, label_width, shift,
               train_df=train_df, val_df=val_df, test_df=test_df,
               label_columns=None):
        # Store the raw data.
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        # Work out the label column indices.
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in
                                        enumerate(label_columns)}
        self.column_indices = {name: i for i, name in
                            enumerate(train_df.columns)}

        # Work out the window parameters.
        #width = le pas de temps avec lequel on veut travailler, cad le nombre de données historiques qu'on va mettre dans la fenêtre pour prédire la suite
        self.input_width = input_width
        self.label_width = label_width
        #shift = le pas de temps que l'on veut prédire à la suite - offset
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def __repr__(self):
        return '\n'.join([
        f'Total window size: {self.total_window_size}',
        f'Input indices: {self.input_indices}',
        f'Label indices: {self.label_indices}',
        f'Label column name(s): {self.label_columns}'])
# Étant donné une liste d'entrées consécutives, la méthode split_window les convertira en une fenêtre d'entrées et une fenêtre d'étiquettes.
def split_window(self, features):
  inputs = features[:, self.input_slice, :]
  labels = features[:, self.labels_slice, :]
  if self.label_columns is not None:
    labels = tf.stack(
        [labels[:, :, self.column_indices[name]] for name in self.label_columns],
        axis=-1)

  # Slicing doesn't preserve static shape information, so set the shapes
  # manually. This way the `tf.data.Datasets` are easier to inspect.
  inputs.set_shape([None, self.input_width, None])
  labels.set_shape([None, self.label_width, None])

  return inputs, labels

WindowGenerator.split_window = split_window

def plot(self, model=None, plot_col='Cote', max_subplots=3):
    inputs, labels = self.example
    plt.figure(figsize=(12, 8))
    plot_col_index = self.column_indices[plot_col]
    max_n = min(max_subplots, len(inputs))
    for n in range(max_n):
        plt.subplot(max_n, 1, n+1)
        plt.ylabel(f'{plot_col} [normed]')
        plt.plot(self.input_indices, inputs[n, :, plot_col_index],
                label='Inputs', marker='.', zorder=-10)

        if self.label_columns:
            label_col_index = self.label_columns_indices.get(plot_col, None)
        else:
            label_col_index = plot_col_index

        if label_col_index is None:
            continue

    plt.scatter(self.label_indices, labels[n, :, label_col_index],
                edgecolors='k', label='Labels', c='#2ca02c', s=64)
    if model is not None:
      predictions = model(inputs)
      plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                  marker='X', edgecolors='k', label='Predictions',
                  c='#ff7f0e', s=64)

    if n == 0:
      plt.legend()

    plt.xlabel('Cote')

WindowGenerator.plot = plot

"""def make_dataset(self,data):
    data = np.array(data, dtype=np.float32)
    ds = tf.keras.preprocessing.timeseries_dataset_from_array(
      data=data,
      targets=None,
      sequence_length=self.total_window_size,
      sequence_stride=1,
      shuffle=False,
      batch_size=32)
    ds = ds.map(self.split_window)
    return ds 

WindowGenerator.make_dataset = make_dataset

@property
def train(self):
  return self.make_dataset(self.train_df)

@property
def val(self):
  return self.make_dataset(self.val_df)

@property
def test(self):
  return self.make_dataset(self.test_df)

@property
def example(self):
  #Get and cache an example batch of `inputs, labels` for plotting.
  result = getattr(self, '_example', None)
  if result is None:
    # No example batch was found, so get one from the `.train` dataset
    result = next(iter(self.train))
    # And cache it for next time
    self._example = result
  return result

WindowGenerator.train = train
WindowGenerator.val = val
WindowGenerator.test = test
WindowGenerator.example = example"""


"""@st.cache()
def load_ml_model():
    LSTM_model = keras.models.load_model('/Users/dorian.erkens/Desktop/Jedha_FS_Bootcamp/Test_new_env/saved_model.pb')
    return LSTM_model
LSTM_model = load_ml_model()

predict = LSTM_model.predict()"""

























predict_button, model_architecture = st.beta_columns([2,2])
with predict_button:
    with st.form('Predict Groundwater level'):
        st.write(' ')
        st.write(' ')
        st.write(' ')
        st.write(' ')
        st.header('Predict Groundwater level')
        input_var = st.slider(label='Predicted time frame (days)', min_value=1,max_value=50,key=4)
        multi_window = WindowGenerator(input_width=365,label_width=input_var,shift=input_var)
        submitted = st.form_submit_button('Predict')
        st.write(' ')
        st.write(' ')
        st.write(' ')
        st.write(' ')
        if submitted : 
            prediction = [1]*input_var
            st.write(f'The prediction will be here: {prediction}')
        
with model_architecture:
    img = 'https://erdo-streamlit-911.s3.eu-central-1.amazonaws.com/Dataset_final_project/LSTM_model_architecture_1.png'
    st.subheader('Example : Single-shot multi-step LSTM model')
    st.image(img)

#multi_window.plot(LSTM_model)

st.header("Reference")
st.subheader('Bibliography')
st.write('Géron, A. (2019, September). Hands-on Machine Learning with Scikit-Learn,Keras and Tensorflow')
st.write("Zhang, J., Zhu, Y., Zhang, X., Ye, M., Yang, J., Developing a Long Short-Term Memory(LSTM) based Model for Predicting Water Table Depth in Agricultural Areas, Journal of Hydrology (2018),doi: https://doi.org/10.1016/j.jhydrol.2018.04.065")
st.write("Jalalkamali, A., Sedghi, H.,Manshouri, M., Monthly groundwater level prediction using ANN and neuro-fuzzy models: a case study on Kerman plain, Iran in Agricultural Areas, Journal of Hydroinformatics (2011)")