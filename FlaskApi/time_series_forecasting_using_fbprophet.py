# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 10:20:13 2021

@author: kalas
"""
from flask import Flask,render_template,redirect,request
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import itertools
import matplotlib.pyplot as plt
import numpy as np
from fbprophet import Prophet
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')
from random import randint
import plotly.graph_objs as go
import plotly.offline as py

import os
app = Flask("__name__")
app.config["IMAGE_UPLOADS"] = "static/img/"
@app.route('/')
def hello():
    return render_template("step1.html")




@app.route("/home")
def home():
    return redirect('/')


@app.route('/',methods=['POST'])
def submit_data():
    
        
    f=request.files['userfile']
    f.save(f.filename)    
    s1=request.form['query1']
    s2=request.form['query2']  
    t=int(request.form['query3'])
    s4=request.form['query4']
    d1=f.filename
    df=pd.read_csv(d1)
    
    
    #Prophet
    df = df.rename(columns={s2: 'y', s1:'ds'})
    df['y_orig'] = df['y'] # to save a copy of the original data..you'll see why shortly. 
    df['y'] = np.log(df['y'])
    model = Prophet() #instantiate Prophet
    model.fit(df)

    
    
    ''' 'year': 'A',
        'quarter': 'Q',
                'month': 'M',
                'day': 'D',
                'hour': 'H',
                'minute': 'T',
                'second': 'S',
                'millisecond': 'L',
                'microsecond': 'U',
                'nanosecond': 'N'}
        '''
        
        
    future_data = model.make_future_dataframe(periods=t, freq = s4)

    future_data
    forecast_data = model.predict(future_data)

    
    forecast_data[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(10)
    
    model.plot(forecast_data) 
    model.plot_components(forecast_data)
    forecast_data_orig = forecast_data # make sure we save the original forecast data
    forecast_data_orig['yhat'] = np.exp(forecast_data_orig['yhat'])
    forecast_data_orig['yhat_lower'] = np.exp(forecast_data_orig['yhat_lower'])
    forecast_data_orig['yhat_upper'] = np.exp(forecast_data_orig['yhat_upper'])
    model.plot(forecast_data_orig)
    df['y_log']=df['y'] #copy the log-transformed data to another column
    df['y']=df['y_orig']
    final_df = pd.DataFrame(forecast_data_orig)
    
    final_df_1=final_df[['ds','yhat']].tail(t)
    final_df_1 = final_df_1.rename(columns={'yhat': 'Sales', 'ds':'Month'})
    

    #rmse = mean_squared_error(df["y_orig"].iloc[24:], final_df['yhat'].iloc[24:36])**0.5
    #print('Test MSE: %.3f' % rmse)
                
      
    fig,ax=plt.subplots(nrows=1, ncols=1)
    ax.plot(df["y_orig"],label="Actual")
    ax.plot(final_df["yhat"],label="Predicted")
    ax.legend()

    #plt.xticks(rotation=90)
    #plt.show()
    n=randint(0,1000000000000)
    n=str(n)
    fig.savefig(os.path.join(app.config["IMAGE_UPLOADS"],n+'time_series.png'))  
    full_filename= os.path.join(app.config["IMAGE_UPLOADS"],n+'time_series.png')
    
    
    return render_template('step1.html',user_image = full_filename,tables=[final_df_1.to_html(classes='forecast')],titles=['na','forecast'],query1 = request.form['query1'],query2 = request.form['query2'],query3 = request.form['query3'], query4 = request.form['query4'])
     
'''
    import plotly.graph_objs as go
    import plotly.offline as py
    #Plot predicted and actual line graph with X=dates, Y=Outbound
    actual_chart = go.Scatter(y=df["y_orig"], name= 'Actual')
    predict_chart = go.Scatter(y=final_df["yhat"], name= 'Predicted')
    predict_chart_upper = go.Scatter(y=final_df["yhat_upper"], name= 'Predicted Upper')
    predict_chart_lower = go.Scatter(y=final_df["yhat_lower"], name= 'Predicted Lower')
    #py.plot([actual_chart, predict_chart, predict_chart_upper, predict_chart_lower])
    py.plot([actual_chart, predict_chart, predict_chart_upper, predict_chart_lower], filename = 'templates/' +'filename.html', auto_open=False, image_width=200, image_height=200)
    
'''
    #return render_template('step1.html',user_image = full_filename,tables=[final_df_1.to_html(classes='forecast')],titles=['na','forecast'],query1 = request.form['query1'],query2 = request.form['query2'],query3 = request.form['query3'], query4 = request.form['query4'])
    
   
if __name__ =="__main__":
    app.run()
    
