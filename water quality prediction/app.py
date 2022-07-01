# -*- coding: utf-8 -*-
import numpy as np
import pickle
from flask import Flask, request, render_template, json,redirect
import pandas as pd 
import os
import database
# import outlier_detector
import descriptive_graph

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

# Load ML model
model = pickle.load(open('model2.pkl', 'rb')) 

# Create application
app = Flask(__name__)

#init db 
db =  database.Database()

#login & register

#login status
loginStatus = False
Login_user = ""
is_admin =False

#desriptive chart option
features =['Ambient-Temperature(amb-temp)', \
            'Dissolve Oxygen(DO)', \
            'Electric Conductivity(EC)',\
            'Oxidation-Reduction Potential(ORP)',\
            'Water pH(ph)',\
            'Month']

#desriptive chart option
features2 =['Ambient-Temperature(amb-temp)', \
            'Dissolve Oxygen(DO)', \
            'Electric Conductivity(EC)',\
            'Oxidation-Reduction Potential(ORP)',\
            'Water pH(ph)']

chart_type = ['Heatmap', 'Bar chart', 'Line graph']

feature_mapping ={
    'Ambient-Temperature(amb-temp)':'amb-temp',
    'Dissolve Oxygen(DO)':'do',
    'Electric Conductivity(EC)':'ec',
    'Oxidation-Reduction Potential(ORP)':'orp',
    'Water pH(ph)':'ph',
    'Month':'month'
}

chart_type_selection = 'Heatmap'
feature1 = 'amb-temp'
feature2 = 'amb-temp'
graph_title = 'Heat map for Ambient-Temperature(amb-temp) vs Ambient-Temperature(amb-temp)'
ui_sync_f1 = 'Ambient-Temperature(amb-temp)'
ui_sync_f2 = 'Ambient-Temperature(amb-temp)'

@app.route("/")
def Main():
    return render_template("login.html")

@app.route('/descriptive')  
def main ():  
    global loginStatus
    if not loginStatus:
        return redirect('/')
    return render_template("descriptive.html", features = features,features2=features2, \
                            chart_type=chart_type, \
                            chart_type_selection=chart_type_selection,\
                            ui_sync_f1 = ui_sync_f1,\
                            ui_sync_f2 = ui_sync_f2 )

# =============================================================================
# @app.route('/login_nav')  
def login_nav ():  
    return render_template("login.html", )
# =============================================================================
     
@app.route('/login', methods=['GET', 'POST'])
def login():
    global loginStatus
    global Login_user
    global is_admin
    username = request.form['username']
    password = request.form['password']
    account = database.Account.USER
    if request.form['is_admin'] == "true":
        account = database.Account.ADMIN

    response = db.login( username, password,account) #check JSON DB 
    is_admin = db.query_user_type(username,password)
  
    data = {}

    if response == True :
        print("true::")
        data = {"login":"true"} # Your data in JSON-serializable type
        Login_user = username
        loginStatus = True
      
    else :
        data = {"login":"false"} # Your data in JSON-serializable type
        loginStatus = False

    response = app.response_class(response=json.dumps(data),status=200,mimetype='application/json')
    return response

#feedback
@app.route('/feedbackpage')
def feedbackpage():
     return render_template("feedback.html")

@app.route('/feedback', methods=['GET', 'POST'])
def feedback():
    name = request.form['name']
    email = request.form['e-mail']
    message = request.form['message']

    db.submit_feedback( name,email, message) #register in JSON DB
    data = {"submit":"true"} # Your data in JSON-serializable type

    response = app.response_class(response=json.dumps(data),status=200,mimetype='application/json')
    return response

@app.after_request
def add_header(response):
    response.headers['X-UA-Compatible'] = 'IE=Edge,chrome=1'
    response.headers['Cache-Control'] = 'public, max-age=0'
    return response

#logout
@app.route('/logout', methods=['POST'])
def logout():
    global loginStatus
    global Login_user
    global is_admin
    loginStatus = False
    Login_user = ""
    is_admin = False
    data = {"logout":"true"} # Your data in JSON-serializable type

    response = app.response_class(response=json.dumps(data),status=200,mimetype='application/json')
    return response

@app.route('/logout_nav')
def logout_nav():
    global loginStatus
    global Login_user
    global is_admin
    loginStatus = False
    Login_user = ""
    is_admin = False
    data = {"logout":"true"} # Your data in JSON-serializable type

    return render_template("login.html")

# @app.route("/show_water_quality_do", methods=['GET', 'POST'])
# def show_water_quality_do():
#     return outlier_detector.generate_do_scatter_plot()

# @app.route("/show_water_quality_do_outlier", methods=['GET', 'POST'])
# def show_water_quality_do_outlier():
#     return outlier_detector.generate_do_outlier_scatter_plot()


@app.route('/register', methods=['GET', 'POST'])
def register():
    global loginStatus
    global Login_user
    global is_admin
    username = request.form['username']
    password = request.form['password']
    email = request.form['email']
    account = database.Account.USER
    is_admin = False
    if request.form['is_admin'] == "true":
        account = database.Account.ADMIN
        is_admin = True

    response = db.register_user( username, password, email, account) #register in JSON DB
    if response == True :
        print("true::")
        data = {"register":"true"}# Your data in JSON-serializable type
        loginStatus = True
        Login_user = username
      
    else :
        data = {"register":"false"} # Your data in JSON-serializable type
        loginStatus = False

    response = app.response_class(response=json.dumps(data),status=200,mimetype='application/json')
    return response


# reading the data in the csv file
df = pd.read_csv('20220207_lake_with category.csv')
df.to_csv('20220207_lake_with category.csv', index=None)

# route to html page - "table"
#function name and route name must be the same 
@app.route('/descriptive')
def descriptive():
    if is_admin:
        return render_template("descriptive_admin.html", features = features, features2=features2,\
                            chart_type=chart_type, \
                            chart_type_selection=chart_type_selection,\
                            ui_sync_f1 = ui_sync_f1,\
                            ui_sync_f2 = ui_sync_f2 )
    return render_template("descriptive.html", features = features,features2=features2, \
                            chart_type=chart_type, \
                            chart_type_selection=chart_type_selection,\
                            ui_sync_f1 = ui_sync_f1,\
                            ui_sync_f2 = ui_sync_f2 )

#descriptive graph generation
@app.route('/generate_descriptive_image', methods=['GET', 'POST'])
def generate_descriptive_image():
    chart = request.form['chart_type']
    f1 = request.form['f1']
    f2 = request.form['f2']

    global feature_mapping
    global chart_type
    global chart_type_selection
    global feature1
    global feature2
    global graph_title 
    global ui_sync_f1
    global ui_sync_f2

    chart_type_selection = chart
    feature1 =  feature_mapping[f1]
    feature2 = feature_mapping[f2]
    ui_sync_f1 = f1
    ui_sync_f2 = f2
    if chart == 'Heatmap':
        graph_title = 'Heatmap for '+f2 + ' versus '+ f1
    else:
        if feature1 == 'month':
            graph_title = 'Average of ' +f2 + ' by Month'
        else:
            graph_title = 'Visualization of '+f1 + ' and its frequency'

    print("generate_descriptive_image",chart_type, feature1, feature2)
    data = {"ok":"true"}# Your data in JSON-serializable type
    response = app.response_class(response=json.dumps(data),status=200,mimetype='application/json')
    return response

@app.route('/descriptive_graph_src', methods=['GET', 'POST'])
def descriptive_graph_src():
    global chart_type_selection
    global feature1
    global feature2
    global graph_title
    print("descriptive_graph_src",chart_type_selection, feature1, feature2)
    return descriptive_graph.geenerate_chart(chart_type_selection, feature1, feature2, graph_title)


@app.route('/table')
def table():
    global is_admin
    data = pd.read_csv('20220207_lake_with category.csv')
    return render_template('table.html', tables=[data.to_html()], titles=[''])

# Bind home function to URL
@app.route('/landing')
def landing():
    return render_template('landing.html')

# Bind home function to URL
@app.route('/anomaly')
def anomaly():
    return render_template('anomaly.html')              

@app.route('/predictpage')
def predictpage():
    return render_template('predict.html')    

# Bind predict function to URL
@app.route('/predict', methods =['POST'])
def predict():
    
    # Put all form entries values in a list 
    features = [float(i) for i in request.form.values()]
    # Convert features to array
    array_features = [np.array(features)]
    # Predict features
    prediction = model.predict(array_features)
    output = prediction
    
    # Check the output values and retrive the result with html tag based on the value
    
    #pH:
    if (df['ph'] == 7):
        return 'Neutral'
    elif (7 > df['ph'] >= 0):
        return 'Acidic'
    elif (14 >= df['ph'] > 7):
        return 'Basic'
    else: 
        return 'Abnormal - as the values are out of range'

#orp:
    if (500 >= df['orp'] >= 200):
        return 'Normal'
    elif (200 > df['orp'] >= -1000):
        return 'Low'
    elif (1000 >= df['orp'] > 500):
        return 'High'
    else: 
        return 'abnormal - as the values are out of range'
    
    #do:
    if (8.5 >= df['do'] >= 6):
        return 'Normal'
    elif (6 > df['do'] >= 0):
        return 'Low'
    elif (10 >= df['do'] > 8.5):
        return 'High'
    else: 
        return 'abnormal - as the values are out of range'
    
        return render_template('predict.html')

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=False)
#Run the application
    
    
    