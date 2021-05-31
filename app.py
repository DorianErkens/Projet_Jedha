from numpy.core.fromnumeric import size
from numpy.core.records import array
import streamlit as st
import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px  
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from streamlit.caching import cache
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model, save_model
from tensorflow.keras.layers import Dense, Dropout,Embedding, LSTM
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import Accuracy
import tensorflow_addons as tfa
from tensorflow_addons.metrics import RSquare
import os
import IPython
import IPython.display
from datetime import date,datetime
import datetime as dt
import h5py

st.set_page_config(page_title='Groundwater level prediction - Data Project',
                   layout="wide" )
#Titre et plan de l'application
st.title('Groundwater level prediction')
st.write('This web application has been designed to draft a first work on predicting groundwater levels')
st.write('We have selected 5 different locations in France to work on, you will be able to choose from those during your test of our app')
st.write("""The content is designed in 5 steps :  
                1. Data Geography  
                2. Data Exploration  
                3. Exploring Time Series Data Analysis   
                4. Ground Water Level Prediction    
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
    'Cote']]
    data = data.reset_index(drop=True)
    #process specifically the dataset for ml purposes
    data_ml = data.copy()
    data_ml.DateTime = pd.to_datetime(data.DateTime)
    data_ml.DateTime = pd.to_numeric(data_ml.DateTime)
    data_ml = data_ml.reset_index(drop=True)
    # datetime as index to properly manage the time serie
    data = data.set_index("DateTime")
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
    cote_time = px.line(data_frame=data,y='Cote')
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
    variable = st.sidebar.selectbox("What trend across time do you want to explore ?", data.columns)
    variable_accross_time = px.line(data_frame=data,y=variable)
    st.subheader(f'{variable} evolution across time')
    st.plotly_chart(variable_accross_time,use_container_width=True)
time_graph()

#Supervised Machine Learning
st.header("Exploring Time Series Data Analysis ")
#Function that will preprocess the dataset to be properly managed with ML/DL algo, namely Dat
#add graph for autocorrelation and lags
#explication pour chacun des graphes
st.subheader('SARIMAX Method')
st.write("""SARIMAX is an extension of the ARIMA model in Python, which stands for seasonal autoregressive integrated moving average with exogenous factors.   SARIMAX models support seasonality and exogenous factors besides the Autoregression, integration and Moving averages from ARIMA.   
          Data scientists usually apply SARIMAX when they have to deal with time series data sets that have seasonal cycles.""")
imaget1, imaget30 = st.beta_columns([2,2])
with imaget1 : 
    st.image('https://erdo-streamlit-911.s3.eu-central-1.amazonaws.com/athee_arima_t%2B1.png')
    st.write('One-day prediction with R squared score of 0.97')
with imaget30 : 
    st.image('https://erdo-streamlit-911.s3.eu-central-1.amazonaws.com/athee_arima_t%2B30_1.png')
    st.write('Thirty-day prediction with R squared score of 0.70 ')

#Deep Learning
st.header('Groundwater level prediction')
st.subheader('Predictions are based on a single-shot multi-steps LSTM model ')
st.write('In order to properly take previous observations into account, we have decided to based our model on RNN deep learning algorithms')

#Hereafter, all the codes to process the data in order to infer data_windows that will help us in the modelisation of time series for deep learning

#Split the dataset into three : train, validation and test
column_indices = {name: i for i,name in enumerate(data.columns)}
n=len(data)
train_df = data[0:int(n*0.7)]
val_df = data[int(n*0.7):int(n*.9)]
test_df = data[int(n*.9):]
num_features = data.shape[1]

#Normalize the dataset, using only training datas as you can only use this ds since others cannot be seen by the model to properly work
#La moyenne et l'écart type doivent être calculés uniquement à l'aide des données d'apprentissage afin que les modèles n'aient pas accès aux valeurs des ensembles de validation et de test.
train_mean = train_df.mean()
train_std = train_df.std()
train_df = (train_df - train_mean)/train_std
valid_df = (val_df - train_mean)/train_std
test_df = (test_df - train_mean)/train_std
#la classe WindowGenerator permet de construire un vecteur de données temporelles, ce qui permettra de proposer des données liées entre elles et de faire comprendre
# au modèle que nous sommes sur des données dont la valeur peut êre expliquée par celle qui la précède
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

# Étant donné une liste d'entrées consécutives, la fonction split_window les convertira en une fenêtre d'entrées et une fenêtre d'étiquettes.
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
#Fonction de viz pour les graphs afin de montrer la fenêtre temporelle (input,label, prediction)
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
                edgecolors='k', label='Labels', c='green', s=64)
        if model is not None:
            predictions = model(inputs)
            plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                  marker='X', edgecolors='k', label='Predictions',
                  c='#ff7f0e', s=64)

        if n == 0:
            plt.legend()

    plt.xlabel('Cote')
WindowGenerator.plot = plot
#Fonction pour créer des datasets TF à partir du data set de time series
def make_dataset(self,data):
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
#Faciliter l'accès au dataset au travers de noms de variables plus logiques
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
  """Get and cache an example batch of `inputs, labels` for plotting."""
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
WindowGenerator.example = example

#Fonction de compilation & fit pour le modèle
MAX_EPOCHS = 50
def compile_and_fit(model,window,patience=2):
    early_stopping=tf.keras.callbacks.EarlyStopping(monitor='mean_absolute_error',patience=patience,mode= 'min')
    model.compile(loss=tf.losses.MeanSquaredError(),
                optimizer=tf.optimizers.Adam(),
                metrics=[tf.metrics.MeanAbsoluteError()])
    history = model.fit(window.train,epochs=MAX_EPOCHS,
                      validation_data=window.val,
                      callbacks=[early_stopping])
    return history


#chargement du modèle LSTM entraîné sur Tensorflow
model = tf.keras.models.load_model('./multi_lstm_model.h5')
#si cela ne marche pas on passera par un dossier github

#End of the code for it, when time we will need to improve it 

#Choose the model you want to predict
#st.subheader('Encode here the geophyscial features to help the model predict groundwater levels')
#precipitation, earth_skin_temperature = st.beta_columns([3,3])
#with precipitation : 
#    precipitation = st.number_input('Precipitation')
#with earth_skin_temperature : 
#    earth_skin_temperature = st.number_input('Earth Skin Temperature',)

#humidity, surface_pressure = st.beta_columns([3,3])
#with humidity : 
#    humidity = st.number_input('Relative Humidity at 2 meters')
#with surface_pressure : 
#    surface_pressure = st.number_input('Surface Pressure')"""

#Variables for the input of the model.predict
#precipitation = precipitation
#earth_skin_temperature = earth_skin_temperature
#humidity = humidity
#temperature_2m = data['Temperature at 2 meters'].median
#wind_speed_10m = data['Wind speed at 10 meters'].median
#sky_incident = data['Sky Insolation Incident'].median
#surface_pressure = surface_pressure
#wet_bulb_temp_2m = data['Wet Bulb temperature at 2 meters'].median
#frost_point_2m = data['Dew/Frost point at 2 meters'].median
#pour la facilité du modèle nous prendrons la dernière Cote enregistrée le 29/03/21
#cote = data['Cote'].iloc[-1]"""

#problème de nan dans un des datasets
if localisation == geo_data.City[0]:
    i=30
    precipitation = data['Precipitation'].iloc[-i]
    earth_skin_temperature = data['Earth Skin Temperature'].iloc[-i]
    humidity = data['Relative Humidity at 2 meters'].iloc[-i]
    temperature_2m = data['Temperature at 2 meters'].iloc[-i]
    wind_speed_10m = data['Wind speed at 10 meters'].iloc[-i]
    sky_incident = data['Sky Insolation Incident'].iloc[-i]
    surface_pressure = data['Surface Pressure'].iloc[-i]
    wet_bulb_temp_2m = data['Wet Bulb temperature at 2 meters'].iloc[-i]
    frost_point_2m = data['Dew/Frost point at 2 meters'].iloc[-i]
    #pour la facilité du modèle nous prendrons la dernière Cote enregistrée le 29/03/21
    cote = data['Cote'].iloc[-i]
else :
    i=1
    precipitation = data['Precipitation'].iloc[-i]
    earth_skin_temperature = data['Earth Skin Temperature'].iloc[-i]
    humidity = data['Relative Humidity at 2 meters'].iloc[-i]
    temperature_2m = data['Temperature at 2 meters'].iloc[-i]
    wind_speed_10m = data['Wind speed at 10 meters'].iloc[-i]
    sky_incident = data['Sky Insolation Incident'].iloc[-i]
    surface_pressure = data['Surface Pressure'].iloc[-i]
    wet_bulb_temp_2m = data['Wet Bulb temperature at 2 meters'].iloc[-i]
    frost_point_2m = data['Dew/Frost point at 2 meters'].iloc[-i]
    #pour la facilité du modèle nous prendrons la dernière Cote enregistrée le 29/03/21
    cote = data['Cote'].iloc[-i]

X = tf.constant([[[precipitation,
                earth_skin_temperature,
                humidity,
                temperature_2m,
                wind_speed_10m,
                sky_incident,
                surface_pressure,
                wet_bulb_temp_2m,
                frost_point_2m,
                cote ]]])
#st.write(X)


#X=tf.constant([[[1,2,3,4,5,6,7,8,9,323.53]]])

predict_button, model_architecture = st.beta_columns([2,2])
with predict_button:
    with st.form('Predict Groundwater level'):
        st.write(' ')
        st.write(' ')
        st.write(' ')
        st.subheader('Predict Groundwater level')
        time_frame = st.slider(label='Predicted time frame (days)', min_value=1,max_value=50,key=4)
        #multi_window = WindowGenerator(input_width=365,label_width=input_var,shift=input_var)
        submitted = st.form_submit_button('Predict')
        st.write(' ')
        st.write(' ')
        st.write(' ')
        st.write(' ')
        if submitted : 
            prediction = model.predict(X)
            
            focused_prediction = prediction[0]
            focused_prediction = focused_prediction[-time_frame:]

            final_prediction = []
            for array in focused_prediction[-time_frame:] :
                final_prediction.append(array[-1])
            

            #final_prediction = [final_prediction.append(array[-1]) for array in focused_prediction]
            #create a dataframe to manipulate the normalized prediction and make them normal again
            prediction_cote = pd.DataFrame(final_prediction, columns=['Pred cote'])
            prediction_cote['Pred de_normalize'] = prediction_cote['Pred cote'] * train_std['Cote'] + train_mean['Cote']
            exposed_prediction = prediction_cote['Pred de_normalize'].tolist()
            st.write(f'The levels for the next {time_frame} days will be : {exposed_prediction}')
        
with model_architecture:
    img = 'https://static.prod-cms.saurclient.fr/sites/default/files/styles/w1440/public/images/Cycle_eau1.png?itok=X1x6dT9S'
    st.subheader('The Water Cycle ')
    st.image(img)


st.subheader('Bibliography')
st.write('Géron, A. (2019, September). Hands-on Machine Learning with Scikit-Learn,Keras and Tensorflow')
st.write("Zhang, J., Zhu, Y., Zhang, X., Ye, M., Yang, J., Developing a Long Short-Term Memory(LSTM) based Model for Predicting Water Table Depth in Agricultural Areas, Journal of Hydrology (2018),doi: https://doi.org/10.1016/j.jhydrol.2018.04.065")
st.write("Jalalkamali, A., Sedghi, H.,Manshouri, M., Monthly groundwater level prediction using ANN and neuro-fuzzy models: a case study on Kerman plain, Iran in Agricultural Areas, Journal of Hydroinformatics (2011)")