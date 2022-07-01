import io
import time
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from flask import send_file
from datetime import datetime

df = pd.read_csv("static/20220207_lake_with category.csv")

def month_extractor(row):
    date_object = datetime.strptime(row['date'], "%d/%m/%Y")
    return date_object.strftime("%b") + '-' + date_object.strftime("%y")
     

df['month'] = df.apply(month_extractor, axis=1)
# df['do'].max()
# bins = np.linspace(df['do'].min(), df['do'].max(), 10)
# bins = np.linspace(df['do'].min(), df['do'].max(), 10)
# digitized = np.digitize(df['do'], bins)
# test = np.array([np.count_nonzero(digitized == d) for d in range(1,10)])

# test_filter = df[ (-6<df['do']) & (df['do']<3) ]
# digitized2 = np.digitize(test_filter['do'], bins)
# test2 = np.array([np.count_nonzero(digitized2 == d) for d in range(1,10)])
# print(bins)
# print(digitized)
sns.set(rc={'figure.figsize':(8,9)})

plotting_do_chart = False

def generate_heatmap(feature1, feature2, title):
    global plotting_do_chart 
    while plotting_do_chart:
        time.sleep(0.5)
    plotting_do_chart = True
    Xbins = np.linspace(df[feature1].min(), df[feature1].max(), 10)
    Ybins = np.linspace(df[feature2].min(), df[feature2].max(), 10)

    heatmap_data =[]
    Y_labels = []
    X_labels = []
    for index in range(0, len(Ybins)-1):
        Ybin_filter_result = df[ (Ybins[index] <= df[feature2]) & (df[feature2]< Ybins[index+1]) ]
        digitized_data =np.digitize(Ybin_filter_result[feature1], Xbins)
        count =  np.array([np.count_nonzero(digitized_data == d) for d in range(1,10)])
        heatmap_data = np.append(heatmap_data, np.flip(count))
        Y_labels = np.append(Y_labels, str(round(Ybins[index], 2))+'-'+str(round(Ybins[index+1], 2)))
        X_labels = np.append(X_labels, str(round(Xbins[index], 2))+'-'+str(round(Xbins[index+1], 2)))

    Y_labels = np.flip(Y_labels)
    heatmap_data = np.reshape(heatmap_data,(9,9))
    heatmap_data = heatmap_data.astype(int)
    print("next")
    if df[feature1].min() == df[feature1].max()== 0 :
        num_rows, num_cols = df.shape
        heatmap_data[-1,0] = num_rows
    
    if df[feature2].min() == df[feature2].max()== 0 :
        num_rows, num_cols = df.shape
        heatmap_data[-1,0] = num_rows
    
    fig = plt.figure(0)
    plt.clf()
    df_cm = pd.DataFrame(heatmap_data, index = Y_labels, columns = X_labels)
    plt.title(title)
    sns.heatmap(df_cm, annot=True, annot_kws={"fontsize":8})
    plt.xlabel(feature1)
    plt.ylabel(feature2)
    plt.yticks(rotation = 45)
    plt.xticks(rotation = 45)
    img = io.BytesIO()
    fig.savefig(img)
    img.seek(0)
    plotting_do_chart = False
    return send_file(img,mimetype='img/png')


def generate_bar(feature1, title):
    global plotting_do_chart 
    while plotting_do_chart:
        time.sleep(0.5)
    plotting_do_chart = True
    Xbins = np.linspace(df[feature1].min(), df[feature1].max(), 10)

    chart_data =[]
    X_labels = []
    digitized_data =np.digitize(df[feature1], Xbins)
    chart_data =  np.array([np.count_nonzero(digitized_data == d) for d in range(1,10)])
    for index in range(0, len(Xbins)-1):
        X_labels = np.append(X_labels, str(round(Xbins[index], 2))+'-'+str(round(Xbins[index+1], 2)))

    if df[feature1].min() == df[feature1].max():
        chart_data[0] = len(df[feature1])
    
    chart_data = chart_data.astype(int)
    print("next")
    

    fig = plt.figure(0)
    plt.clf()
    df_bar = pd.DataFrame({'Xaxis': X_labels, 'val': chart_data })
    fig = sns.barplot(x = 'Xaxis', y = 'val', data = df_bar)
    plt.title(title)
    plt.xlabel(feature1)
    plt.ylabel("Frequency")
    plt.xticks(rotation = 45)
    img = io.BytesIO()
    fig.figure.savefig(img)
    img.seek(0)
    plotting_do_chart = False
    return send_file(img,mimetype='img/png')

# generate_bar('do','test')

def generate_line(feature1, title):
    global plotting_do_chart 
    while plotting_do_chart:
        time.sleep(0.5)
    plotting_do_chart = True
    Xbins = np.linspace(df[feature1].min(), df[feature1].max(), 10)

    chart_data =[]
    X_labels = []
    digitized_data =np.digitize(df[feature1], Xbins)
    chart_data =  np.array([np.count_nonzero(digitized_data == d) for d in range(1,10)])
    for index in range(0, len(Xbins)-1):
        X_labels = np.append(X_labels, str(round(Xbins[index], 2))+'-'+str(round(Xbins[index+1], 2)))

    if df[feature1].min() == df[feature1].max():
        chart_data[0] = len(df[feature1])

    chart_data = chart_data.astype(int)
    print("next")
    
    fig = plt.figure(0)
    plt.clf()
    df_line = pd.DataFrame({'Xaxis': X_labels, 'val': chart_data })
    fig = sns.lineplot(data=df_line, x="Xaxis", y="val")
    plt.title(title)
    plt.xlabel(feature1)
    plt.ylabel("Frequency")
    plt.xticks(rotation = 45)
    img = io.BytesIO()
    fig.figure.savefig(img)
    img.seek(0)
    plotting_do_chart = False
    return send_file(img,mimetype='img/png')

##Geenrate chart for date 
#feature1 is date
def generate_heatmap_date(feature1, feature2, title):
    global plotting_do_chart 
    while plotting_do_chart:
        time.sleep(0.5)
    plotting_do_chart = True
    #Xbins = np.linspace(df[feature1].min(), df[feature1].max(), 10)
    Ybins = np.linspace(df[feature2].min(), df[feature2].max(), 9)
    dateXLabel = np.unique(df['month'])
    Y_labels = []
    if len(Ybins) == 2 and Ybins[0] == Ybins[1]:
        Ybins[1] = Ybins[1]+0.01

    heatmap_data = np.empty((0, len(dateXLabel)), int)
    for index in range(0, len(Ybins)-1):
        Ybin_filter_result = df[ (Ybins[index] <= df[feature2]) & (df[feature2]< Ybins[index+1]) ]
        count =  np.array([np.count_nonzero(Ybin_filter_result['month'] == d) for d in dateXLabel])
        print(count)
        #heatmap_data = np.append(heatmap_data, count)
        heatmap_data = np.append(heatmap_data, np.array([np.flip(count)]), axis=0)
        Y_labels = np.append(Y_labels, str(round(Ybins[index], 2))+'-'+str(round(Ybins[index+1], 2)))

    #heatmap_data = np.reshape(heatmap_data,(-1, len(dateXLabel)))
    #heatmap_data = heatmap_data.astype(int)
    print("next")
    Y_labels = np.flip(Y_labels)
    
    fig = plt.figure(0)
    plt.clf()
    df_cm = pd.DataFrame(heatmap_data, index = Y_labels , columns = dateXLabel)
    plt.title(title)
    sns.heatmap(df_cm, annot=True, annot_kws={"fontsize":8})
    plt.xlabel(feature1)
    plt.ylabel(feature2)
    plt.yticks(rotation = 45)
    plt.xticks(rotation = 45)
    img = io.BytesIO()
    fig.savefig(img)
    img.seek(0)
    plotting_do_chart = False
    return send_file(img,mimetype='img/png')

def generate_bar_date(feature2, title):
    global plotting_do_chart 
    while plotting_do_chart:
        time.sleep(0.5)
    plotting_do_chart = True

    Xbins = np.unique(df['month'])
    chart_data =[]
    X_labels = Xbins

    for index in range(0, len(Xbins)):
        month_filter_result = df[ (df['month'] == Xbins[index])]
        chart_data = np.append(chart_data, np.mean(month_filter_result[feature2]))
    
    print("next")
    

    fig = plt.figure(0)
    plt.clf()
    df_bar = pd.DataFrame({'Xaxis': X_labels, 'val': chart_data })
    fig = sns.barplot(x = 'Xaxis', y = 'val', data = df_bar)
    plt.title(title)
    plt.xlabel('Month')
    plt.ylabel("Frequency")
    plt.xticks(rotation = 45)
    img = io.BytesIO()
    fig.figure.savefig(img)
    img.seek(0)
    plotting_do_chart = False
    return send_file(img,mimetype='img/png')


def generate_line_by_date(feature2, title):
    global plotting_do_chart 
    while plotting_do_chart:
        time.sleep(0.5)
    plotting_do_chart = True

    Xbins = np.unique(df['month'])
    chart_data =[]
    X_labels = Xbins

    for index in range(0, len(Xbins)):
        month_filter_result = df[ (df['month'] == Xbins[index])]
        chart_data = np.append(chart_data, np.mean(month_filter_result[feature2]))
    
    print("next")
    
    fig = plt.figure(0)
    plt.clf()
    df_line = pd.DataFrame({'Xaxis': X_labels, 'val': chart_data })
    fig = sns.lineplot(data=df_line, x="Xaxis", y="val")
    plt.title(title)
    plt.xlabel('Month')
    plt.ylabel("Average "+feature2)
    plt.xticks(rotation = 45)
    img = io.BytesIO()
    fig.figure.savefig(img)
    img.seek(0)
    plotting_do_chart = False
    return send_file(img,mimetype='img/png')

def geenerate_chart(chart_type, feature1, feature2, title):
    chart_type_opn = ['Heatmap', 'Bar chart', 'Line graph']
    print(chart_type + "generate " + feature1)
    if feature1 == 'month':
        if chart_type == chart_type_opn[0]:
            return generate_heatmap_date(feature1,feature2, title)
        elif chart_type == chart_type_opn[1]:
            return generate_bar_date(feature2, title)
        elif chart_type == chart_type_opn[2]:
            return generate_line_by_date(feature2, title)
        else:
            return generate_heatmap_date(feature1,feature2, title)

    if chart_type == chart_type_opn[0]:
        return generate_heatmap(feature1, feature2, title)
    elif chart_type == chart_type_opn[1]:
        return generate_bar(feature1, title)
    elif chart_type == chart_type_opn[2]:
        return generate_line(feature1, title)
    else:
        return generate_heatmap(feature1, feature2, title)