from matplotlib.pyplot import plot, title
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
#import pandas as pd
from plotly.offline import iplot
#import plotly.express as px 
from plotly.graph_objs import Scatter
import plotly.graph_objs as go

from flask import Markup
#import json
import plotly
#print(plotly.__version__)

#Creat flask app
app = Flask(__name__)

#load the pickle model
model = pickle.load(open("Final_Cobb_Progress.pkl", "rb"))

@app.route("/")
def Home():
    return render_template("index.html")

@app.route("/predict", methods = ["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = model.predict(features)
    print(prediction)
    prediction = np.round(prediction[0],1)

    # Single plot age at predicition vs final cobb
    age_last = float_features[4]
    final_predict = prediction

    #prediction with 5% error at different age
    
    age_first = float_features[7]
    age_arr = np.arange(np.floor(age_first+1), 19)
    #age_arr = np.insert(age_arr,0,age_first)

    angle = float_features[0]
    angle_arr = [angle -5, angle, angle+5]
    new_float_features=float_features.copy()
    pred_angle = []
    for i in angle_arr:
        new_float_features[0]= i
        features_new = [np.array(new_float_features)]

        for a in age_arr:
            new_float_features[4] = a
            features_new = [np.array(new_float_features)]
            pred_angle.append(model.predict(features_new)[0])
    #print(type(pred_angle))
    print(type(age_arr))

    pred_angle1 = pred_angle[0:len(age_arr)] #Cobb -5
    pred_angle2 = pred_angle[len(age_arr): 2*len(age_arr)] #Cobb
    pred_angle3 = pred_angle[2*len(age_arr):] #Cobb +5
    pred_angle1 = np.insert(pred_angle1,0,angle_arr[0])
    pred_angle2 = np.insert(pred_angle2,0,angle_arr[1])
    pred_angle3 = np.insert(pred_angle3,0,angle_arr[2])
    age_arr = np.insert(age_arr,0,age_first)
    
    plus_one_STD=np.full(np.shape(pred_angle2),4.3)
    minus_one_STD=np.full(np.shape(pred_angle2),-4.3)
    plus_two_STD=np.full(np.shape(pred_angle2),8.6)
    minus_two_STD=np.full(np.shape(pred_angle2),-8.6)
    
    pred_angle2_plus_one_STD=np.add(pred_angle2,plus_one_STD)
    pred_angle2_minus_one_STD=np.add(pred_angle2,minus_one_STD)
    pred_angle2_plus_two_STD=np.add(pred_angle2,plus_two_STD)
    pred_angle2_minus_two_STD=np.add(pred_angle2,minus_two_STD)
    
    #fig 1 
    layout = go.Layout(title = "Cobb Angle prediction from current age to 18 years", xaxis = {'title':'Age (years)'}, yaxis = {'title':'Cobb Angle (degrees)'})  
    fig = go.Figure(data =[go.Scatter(x = age_arr,y = np.round((pred_angle2),1),
                        mode ='lines+markers', marker={'color' : 'blue', 'size' : 10,'symbol' : 'square'},
                        name ='Curve Progression Prediction')], layout = layout)
   # fig=go.Figure()
    #fig.add_trace(go.Scatter(x=age_arr, y=np.round((pred_angle1),1)),line = dict(color='royalblue', width=3) , name='Cobb Angle -5')
    """
    fig.add_trace(go.Scatter(x=age_arr, y=np.round((pred_angle1),1), 
    mode='lines+markers',marker={'color' : 'violet', 'size' : 10,'symbol' : 'circle'}, 
    line={'dash' : 'dash'},name='Cobb Angle -5'))
    fig.add_trace(go.Scatter(x=age_arr, y=np.round((pred_angle3),1),
    line={'dash' : 'dot'}, 
    mode='lines+markers',marker={'color' : 'lightgreen','size' : 10,'symbol' : 'triangle-up'}, name='Cobb Angle +5'))
    """
    fig.add_trace(go.Scatter(
        x = np.concatenate((age_arr, age_arr[::-1]),axis=None),
        #y = np.concatenate((pred_angle1,  pred_angle3[::-1]),axis=None),
        y = np.concatenate((pred_angle2_minus_one_STD,  pred_angle2_plus_one_STD[::-1]),axis=None),
        fill='toself',
        fillcolor='rgba(0,0,255,0.2)',
        line_color='rgba(255,255,255,0)',
        showlegend=True,
        name='68% Confidence Interval',
    ))
    fig.add_trace(go.Scatter(
        x = np.concatenate((age_arr, age_arr[::-1]),axis=None),
        #y = np.concatenate((pred_angle1,  pred_angle3[::-1]),axis=None),
        y = np.concatenate((pred_angle2_minus_two_STD,  pred_angle2_plus_two_STD[::-1]),axis=None),
        fill='toself',
        fillcolor='rgba(255,0,0,0.2)',
        line_color='rgba(255,255,255,0)',
        showlegend=True,
        name='95% Confidence Interval',
    ))
    fig.add_trace(go.Scatter(x=np.array(age_last), y=np.array(final_predict), 
    mode='markers',marker={'color' : 'red','size' : 15,'symbol' : 'star'},  text= 'Current predict', textposition = 'bottom right' ,name='Current Prediction'))

    fig.add_trace(go.Scatter(x=np.array(age_first), y=np.array(angle), 
    mode='markers',marker={'color' : 'green','size' : 15,'symbol' : 'star'},  name='Current Cobb Angle'))
    fig.update_layout(font = {'size' : 18})
    my_plot=plotly.offline.plot(fig, include_plotlyjs=False, output_type='div') 

# 'frames': [{'data': [{'x': [age_last], 'y': [final_predict]}]}]
   # my_plot_div = plot([Scatter(x=age_arr, y=(pred_angle1))], output_type='div', image_filename="Cobb angle predicition")
    #my_plot_div.add_trace(go.scatterr(x=age_last, y=final_predict, mode='markers'))
    #my_plot_div = plot([Scatter(x=[age_arr,age_last], y=[(pred_angle1),final_predict])], output_type='div')
    #my_plot_div.plot([Scatter(x=age_last, y=final_predict)])

    #my_plot_div.add_scatter(x=age_arr, y=(pred_angle2))
   #my_plot_div1 = plot([Scatter(x=age_arr, y=(pred_angle2))], output_type='div')    

#fig2 start ------------------------------------------------------------
    layout2 = go.Layout(title = "Cobb Angle prediction from current age to 18 years with initial Cobb angle measurment error of +/-5 degree", xaxis = {'title':'Age (years)'}, yaxis = {'title':'Cobb Angle (degrees)'})  
    fig2 = go.Figure(data =[go.Scatter(x = age_arr,y = np.round((pred_angle2),1),
                        mode ='lines+markers', marker={'color' : 'blue', 'size' : 10,'symbol' : 'square'},
                        name ='Curve Progression Prediction')], layout = layout2)
    fig2.add_trace(go.Scatter(x=age_arr, y=np.round((pred_angle1),1), 
    mode='lines',marker={'color' : 'violet', 'size' : 10}, 
    line={'dash' : 'dash'},name='Cobb Angle -5'))
    fig2.add_trace(go.Scatter(x=age_arr, y=np.round((pred_angle3),1), 
    mode='lines',marker={'color' : 'violet', 'size' : 10}, 
    line={'dash' : 'dash'},name='Cobb Angle +5'))
    fig2.add_trace(go.Scatter(x=np.array(age_last), y=np.array(final_predict), 
    mode='markers',marker={'color' : 'red','size' : 15,'symbol' : 'star'},  text= 'Current predict', textposition = 'bottom right' ,name='Current Prediction'))
    fig2.add_trace(go.Scatter(x=np.array(age_first), y=np.array(angle), 
    mode='markers',marker={'color' : 'green','size' : 15,'symbol' : 'star'},  name='Current Cobb Angle'))
    fig2.update_layout(font = {'size' : 18})
    my_plot2=plotly.offline.plot(fig2, include_plotlyjs=False, output_type='div')  

#fig2 end ----------------------------------------------------------------------
    return render_template("index.html", 
        InitialCobb=float_features[0],
        Flexibility=float_features[1],
        Lordosis=float_features[2],
        Kyphosis=float_features[3],
        AgePredict=float_features[4],
        Levels=float_features[5],
        Risser=float_features[6],
        AgeFirst=float_features[7],
        Gender=float_features[8],
        prediction_text = " The Final predicted Cobb angle is {} degrees".format(prediction),  
        pred_angle1=pred_angle1,
        pred_angle2=pred_angle2,
        pred_angle3=pred_angle3,
        age_arr=age_arr,
        div_placeholder=Markup(my_plot),
        div_placeholder2=Markup(my_plot2)
        )

        

"""""
age_last = float_features[0,3]
print(age_last)
age_first = float_features[0,2]

age_arr = np.arange(np.floor(age_first+1), 19)
print(age_arr)
age_arr = np.insert(age_arr,0,age_first)
print(age_arr)

angle = test_f[0,5]
angle_arr = [angle -5, angle, angle+5]
print(angle_arr)

pred_angle = []
for i in angle_arr:
    test_f[0,5] = i
    for a in age_arr:
        test_f[0,3] = a
        pred_angle.append(rf.predict(test_f))
print(pred_angle)

pred_angle1 = pred_angle[0:len(age_arr)] #Cobb -5
pred_angle2 = pred_angle[len(age_arr): 2*len(age_arr)] #Cobb
pred_angle3 = pred_angle[2*len(age_arr):] #Cobb +5
pred_angle1 = np.insert(pred_angle1,0,angle_arr[0])
pred_angle2 = np.insert(pred_angle2,0,angle_arr[1])
pred_angle3 = np.insert(pred_angle3,0,angle_arr[2])

age_arr = np.insert(age_arr,0,age_first)

from matplotlib import pyplot as plt
plt.plot(age_arr, pred_angle1,label='Initial Cobb-5', marker='o')
plt.plot(age_arr, pred_angle2,label='Initial Cobb', marker='o')
plt.plot(age_arr, pred_angle3,label='Initial Cobb+5', marker='o')
plt.grid(True)
#matplotlib.pyplot.grid(visible=None, which='major', axis='both', grid(color='r', linestyle='-', linewidth=2))
plt.legend(loc='best')
plt.xlabel('Age')
plt.ylabel('Cobb Angle')

    original_input={
        "Initial Cobb angle (°)":float_features[0],
        "Flexibility (%)":float_features[1],
        "Lumbar lordosis angle (°)":float_features[2],
        "Thoracic kyphosis angle (°)":float_features[3],
        "Age at prediction (years)":float_features[4],
        "Number of levels involved":float_features[5],
        "Risser"+" stage at initial diagnosis":float_features[6],
        "Age at first visit":float_features[7],
        "Gender":float_features[8],
"""""
if __name__ == "__main__":
    app.run(debug=True)


