import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate
import pickle
import plotly.graph_objs as go
import pandas as pd
#import lightgbm as lgb
import numpy as np
import plotly.express as px 
import warnings

import os
import json

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

#os.getcwd() # Verificar el directorio de trabajo
#os.chdir("C:\\Misdocs\\cursoSEE\\ProyectoFinal\\Python\\script\\")

#os.listdir("./")

def listToString(s): 
    
    # initialize an empty string
    str1 = "" 
    
    # traverse in the string  
    for ele in s: 
        str1 += ele  
    
    # return string  
    return str1 

#
path='DATA_DRIVE.xlsx'
data = pd.read_excel(path,sheet_name='EDAD Y GENERO')
data.sample(10)

filter1 = data.Year == 2021  
filter2 = data.Age != 'unknown'
filter3 = data.Gender != 'unknown'
filter4 = data["Campaign objective"] == "LEAD_GENERATION"

##filter3 = df_inicial.Month.isin([4,5,6,7])  
data = data[  filter1 & filter2 & filter3 & filter4 ]

data=data.dropna(subset = ["Cost per on-Facebook lead"])
data=data.dropna(subset = ["Promoted post description"])

data =data.dropna(subset=["Cost per on-Facebook lead"])
q1= np.percentile(data["Cost per on-Facebook lead"], 25)  
q2= np.percentile(data["Cost per on-Facebook lead"], 50)
q3= np.percentile(data["Cost per on-Facebook lead"], 75) 

data['Clasificacion'] = "Ninguno"
data['ClasificacionDes'] = 'Ninguno'
data['Clasificacion'].loc[(data['Cost per on-Facebook lead']<= q1)] = 2  #"Mejores"
data['Clasificacion'].loc[(data['Cost per on-Facebook lead']> q1 ) & (data['Cost per on-Facebook lead'] <= q3)] = 1 # "Regulares"
data['Clasificacion'].loc[(data['Cost per on-Facebook lead']> q3 )] = 0 #"Peores"


data['ClasificacionDes'].loc[(data['Cost per on-Facebook lead']<= q1)] = "Mejores"
data['ClasificacionDes'].loc[(data['Cost per on-Facebook lead']> q1 ) & (data['Cost per on-Facebook lead'] <= q3)] = "Regulares"
data['ClasificacionDes'].loc[(data['Cost per on-Facebook lead']> q3 )] = "Peores"

data['Clasificacion']= data['Clasificacion'].astype('int')

data['Descripcion'] = data['Promoted post description'] 
data['Message'] = data['Promoted post message'] 
data['Caption'] = data['Promoted post caption'] 
data['NumDescripcion'] = data.Descripcion.str.split().str.len()
data['NumMessage'] = data.Message.str.split().str.len()
data['NumCaption'] = data.Caption.str.split().str.len()

data['Rate'] = data['Unique leads']/data['Reach'] 
data['CTR'] = data['Link clicks']/data['Impressions'] 



warnings.filterwarnings('ignore')

data['intereses'] = '0' 
#data['education_statuses'] = '0'

data['ComprasOnline'] = '0'
data['Seguridad'] = '0'
data['SeguridadAlarmas'] = '0'
data['DispositivoInteligente'] = '0'
data['DispositivosGPS'] = '0'
data['SmartTechnologies'] = '0'
data['TarjetasCredito'] = '0'
data['Transporte'] = '0'
data['Vehiculos'] = '0'
#data['ServicioTecnico'] ='0'
ComprasOnline = ['Compras online']
Seguridad = ['Seguridad', 'Security', 'Sistema de alarma']
SeguridadAlarmas = ['Anti-theft system', 'Security alarm', 'Sistema de alarma']
DispositivoInteligente = ['Dispositivo inteligente']
DispositivosGPS = ['Dispositivos GPS','GPS Asistido','Conducci??n','Google Maps','Vehicle tracking system']
SmartTechnologies = ['Smart Technologies']
TarjetasCredito = ['Tarjetas de cr??dito']
Vehiculos = ['Sector automotor','Veh??culos','Autom??viles','Carros','Chevrolet','Dispositivos GPS','Honda','Mantenimiento preventivo','Motocicletas','Seguridad vial','Tarjetas de cr??dito','Toyota','Volkswagen']
Transporte = ['Transporte']
#ServicioTecnico = ['Servicios t??cnicos y de TI']
#education_statuses = ['10']

for i in data.index:

  #if (i == 3):
    campo_json = json.loads(data.loc[i]['Ad set targeting'])
    #print(campo_json)
    #hay registros que no tienen en el json el flexible_spec
    try:
        lista_intereses = pd.get_dummies(pd.json_normalize( campo_json ,record_path=['flexible_spec','interests'],errors='ignore').rename(columns={'name': 'interes'})['interes']).columns.values.tolist()
        #lista_educacion = pd.json_normalize(cuack_json,record_path=['flexible_spec'],errors='ignore')['education_statuses'][0]
        cadena_interes = '| ' + ''.join([  str( item + ' | ') for item in lista_intereses ])
        #print(cadena_interes)
        #cadena_educacion = '| ' + ''.join([  str( str(item) + ' | ') for item in lista_educacion ])
        data['intereses'][data.index == i] = cadena_interes

        if any(x in cadena_interes  for x in ComprasOnline) :
             data['ComprasOnline'][data.index == i] = '1'
        else :
             data['ComprasOnline'][data.index == i] = '0'

        if any(x in cadena_interes for x in Seguridad ) :
             data['Seguridad'][data.index == i] = '1'
        else :
             data['Seguridad'][data.index == i] = '0'

        if any(x in cadena_interes  for x in SeguridadAlarmas) :
             data['SeguridadAlarmas'][data.index == i] = '1'
        else :
             data['SeguridadAlarmas'][data.index == i] = '0'

        if any(x in cadena_interes  for x in DispositivoInteligente ) :
             data['DispositivoInteligente'][data.index == i] = '1'
        else :
             data['DispositivoInteligente'][data.index == i] = '0'

        if any(x in cadena_interes  for x in DispositivosGPS ) :
             data['DispositivosGPS'][data.index == i] = '1'
        else :
             data['DispositivosGPS'][data.index == i] = '0'

        if any(x in cadena_interes  for x in SmartTechnologies ) :
             data['SmartTechnologies'][data.index == i] = '1'
        else :
             data['SmartTechnologies'][data.index == i] = '0'

        if any(x in cadena_interes  for x in TarjetasCredito ) :
             data['TarjetasCredito'][data.index == i] = '1'
        else :
             data['TarjetasCredito'][data.index == i] = '0'

        if any(x in cadena_interes  for x in Transporte ) :
             data['Transporte'][data.index == i] = '1'
        else :
             data['Transporte'][data.index == i] = '0'

        if any(x in cadena_interes  for x in Vehiculos ) :
             data['Vehiculos'][data.index == i] = '1'
        else :
             data['Vehiculos'][data.index == i] = '0'


    except:
        #print('no entra')
        data.loc[i]['intereses'] = '0' 
        #data.loc[i]['education_statuses'] = '0' 
        data.loc[i]['ComprasOnline'] = '0'
        data.loc[i]['Seguridad'] = '0'
        data.loc[i]['SeguridadAlarmas'] = '0'
        data.loc[i]['DispositivoInteligente'] = '0'
        data.loc[i]['DispositivosGPS'] = '0'
        data.loc[i]['SmartTechnologies'] = '0'
        data.loc[i]['TarjetasCredito'] = '0'
        data.loc[i]['Vehiculos'] = '0'
        data.loc[i]['Transporte'] = '0'


#Transformar a enteros
#data['education_statuses'] = data['education_statuses'].astype('int')
data['ComprasOnline'] = data['ComprasOnline'].astype('int')
data['Seguridad'] = data['Seguridad'].astype('int')
data['SeguridadAlarmas'] = data['SeguridadAlarmas'].astype('int')
data['DispositivoInteligente'] = data['DispositivoInteligente'].astype('int')
data['DispositivosGPS'] = data['DispositivosGPS'].astype('int')
data['SmartTechnologies'] = data['SmartTechnologies'].astype('int')
data['TarjetasCredito'] = data['TarjetasCredito'].astype('int')
data['Vehiculos'] = data['Vehiculos'].astype('int')
data['Transporte'] = data['Transporte'].astype('int')

# leer el  modelo

filename = 'lgbm.pkl'

# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))

df= data.drop(['Year', 'Year & month', 'Campaign ID', 'Campaign name',
       'Campaign objective', 'Ad set ID', 'Ad set name', 'Ad ID', 'Ad name',
       'Promoted post type','Promoted post description', 'Promoted post message',
       'Promoted post caption', 'Ad set start time', 'Ad set end time',
       'Ad set targeting', 'Frequency', 'ClasificacionDes',
       'Descripcion', 'Message', 'Caption','Promoted post created date','intereses'], axis=1)

a = pd.get_dummies(data['Gender'], prefix = "type")
b = pd.get_dummies(data['Age'], prefix = "type")

frames = [df,a,b]
df = pd.concat(frames, axis = 1)
df = df.drop(columns = ['Gender','Age'])
x_data = df.drop(['Clasificacion'], axis = 1)
x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data)).values


## Importar la data


#df2_pred = pd.read_csv('x_total.csv', delimiter = ',' , header=0)

x_temp = x.drop(['Unique leads','Impressions', 'Reach', 'Link clicks','Cost','CTR','Cost per on-Facebook lead'],axis=1 )

y_predicted = loaded_model.predict(x_temp)

y_pred_lgbm = [np.argmax(line) for line in y_predicted]

y_pred_lgbm = np.array(y_pred_lgbm)
y_pred_lgbm



df2_origen = data.copy()

df2_origen['y_predicted'] =  y_pred_lgbm


df2_origen['conteo'] = 1 



lista_numericas = df2_origen.select_dtypes(include=np.number).columns.tolist()
categorical_columns = ['Age', 'Gender']
metricas = ['Cost per on-Facebook lead','Unique leads','Link clicks','Impressions']


import calendar
df2_origen['MonthName'] = df2_origen['Month'].apply(lambda x: calendar.month_abbr[x])

meses = list(df2_origen['Month'].unique())



df_columnas = pd.DataFrame()

df_columnas['label'] = list(df2_origen['MonthName'].unique())
df_columnas['value'] = list(df2_origen['Month'].unique())

df_columnas = df_columnas.append({'label':'Todos','value':13}, ignore_index = True)
df_columnas = df_columnas[df_columnas.label != 'Unnamed: 0']

columnas_dic = df_columnas.to_dict('records')



app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([

    
    html.P("x-axis:"),
    dcc.Checklist(
        id='x-axis', 
        options=[{'value': x, 'label': x} 
                 for x in categorical_columns],
        value=['Gender'], 
        labelStyle={'display': 'inline-block'}
    ),
    html.P("y-axis:"),
    dcc.RadioItems(
        id='y-axis', 
        options=[{'value': x, 'label': x} 
                 for x in metricas],
        value='Cost per on-Facebook lead', 
        labelStyle={'display': 'inline-block'}
    ),
    
    dcc.Dropdown(
        id='columns-dropdown',
        options=columnas_dic,
        value=13
    ),    
    dcc.Graph(id="box-plot"),
    dcc.Graph(id="example-graph2"),
               
])
    
    
@app.callback(
  Output(component_id='box-plot',component_property='figure'),
  [Input("x-axis", "value"),
   Input("y-axis", "value"),
   Input("columns-dropdown", "value")]

)

def box_plot_update(x, y, input_data3):
    if x is None:
        raise PreventUpdate 
    
    if ( input_data3 != 13):
        df2_origen_tmp = df2_origen[df2_origen['Month'] == input_data3  ] 
    else:
        df2_origen_tmp = df2_origen.copy()
        
    df_seleccion = df2_origen_tmp[['Age', 'Gender','y_predicted','Cost per on-Facebook lead','Unique leads','Link clicks','Impressions','conteo']].groupby(categorical_columns).agg('sum')
    
    df_seleccion = df_seleccion.reset_index()        
    fig = px.box(df_seleccion, x=x, y=y)
    return fig

@app.callback(  
  Output(component_id='example-graph2',component_property='figure'),
  [Input("x-axis", "value"),
   Input("y-axis", "value"),
   Input("columns-dropdown", "value")]

)

def figure2_update(x, y, input_data3):
    if x is None:
        raise PreventUpdate 
    
    if ( input_data3 != 13):
        df2_origen_tmp = df2_origen[df2_origen['Month'] == input_data3  ] 
    else:
        df2_origen_tmp = df2_origen.copy()
    
    df2_origen_tmp['Clase'] =df2_origen_tmp['y_predicted'].replace({0:'Peores',1:'Regulares',2:'Mejores'})
    data = df2_origen_tmp[['Clase','conteo']].groupby(['Clase']).agg('sum')
    

    
    data = data.reset_index()

    fig1 = px.bar(data, x='Clase', y='conteo', height=300 , title='Conteo de Publicidades')



    return fig1

if __name__ == '__main__':
  app.run_server(debug=True)