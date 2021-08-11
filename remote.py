# Importing the libraries
import pickle
import pandas as pd
import webbrowser
# !pip install dash
import dash
import dash_html_components as html
import dash_core_components as dcc
#!pip install dash_bootstrap_components
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np
import sqlite3 as sql

from dash.dependencies import Input, Output , State
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

# favicon  == 16x16 icon ----> favicon.ico  ----> assests

# Declaring Global variabless
project_name = None
app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

get_data = None

# Defining My Functions
def load_model():
    global scrappedReviews
    #scrappedReviews = pd.read_csv('scrappedReviews.csv')
    
    global pickle_model
    file = open("pickle_model.pkl", 'rb') 
    pickle_model = pickle.load(file)

    global vocab
    file = open("features.pkl", 'rb') 
    vocab = pickle.load(file)
    
 
def load_data():
    global get_data
    conn = sql.connect('reviews_data.db')
    c=conn.cursor()
    conn.commit()
    get_data = pd.DataFrame(c.execute('select * from reviews'))
    return get_data
    
def check_review(reviewText):

    #reviewText has to be vectorised, that vectorizer is not saved yet
    #load the vectorize and call transform and then pass that to model preidctor
    #load it later

    transformer = TfidfTransformer()
    loaded_vec = CountVectorizer(decode_error="replace",vocabulary=vocab)
    vectorised_review = transformer.fit_transform(loaded_vec.fit_transform([reviewText]))


    # Add code to test the sentiment of using both the model
    # 0 == negative   1 == positive
    
    return pickle_model.predict(vectorised_review)


def open_browser():
    webbrowser.open_new('http://127.0.0.1:8050/')
    
def create_app_ui():
    global labels
    global values
    labels = ['Positive reviews','Negative reviews']
    values = [519360,519360]
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, textinfo='label+percent',
                             insidetextorientation='radial',pull=[0.1]
                            )])
    main_layout = html.Div(
    [
    html.Div([
            dcc.Graph(figure=fig)
        ]),
    html.H1(id='Main_title', children = "Sentiment Analysis with Insights"),
    dcc.Textarea(
        id = 'textarea_review',
        placeholder = 'Enter the review here.....',
        value='',
        style = {'width':'100%', 'height':100}
        
        ),
    dcc.Dropdown(
            id='demo-dropdown',
            options=[
                {'label': i, 'value': get_data[0][get_data[1]==i].to_string(index=False)} 
                for i in get_data[1]
                ]
            ),
    dbc.Button(
        children = 'FInd Review',
        id = 'button_review',
        color = 'dark',
        style= {'width':'100%'}
        ),
    html.Label(
            id='label1',
            children='Positive Reviews',
            style=None
            ),
        html.Br(),
        html.Label(
            id='label2',
            children='Negative Reviews',
            style = None
            ),
    html.Br(),
    html.Br(),
    html.Br(),
    html.Br()    
    ]    
    ) 
    return main_layout

@app.callback(
    dash.dependencies.Output('textarea_review', 'value'),
    [dash.dependencies.Input('demo-dropdown', 'value')])
def update_output(value):
    print('the value is ========'+value)
    check_value = int(value)
    global get_data
    filter_review = get_data[1][check_value-1]
    print('the filter values is =========================='+filter_review)
    if value == None:
        return 
    else:
        return filter_review

@app.callback(
    Output('label1', 'children'),
    Output('label2', 'children'),
    [Input('button_review', 'n_clicks')],
    [State('textarea_review', 'value')])

def update_app_ui_2(n_clicks, textarea_value):
    print("Data Type = ", str(type(n_clicks)))
    print("Value = ", str(n_clicks))


    print("Data Type = ", str(type(textarea_value)))
    print("Value = ", str(textarea_value))
  
    if (n_clicks>0):
        response = check_review(textarea_value)
        if(textarea_value == ""):
            return True,True
        elif(textarea_value.isalnum() or textarea_value.startswith('-') or textarea_value[1:].isdigit()):
            return 'Unknown','Unknown'
        elif (response[0] == 0):
            result = 'Negative'
            return None,result
        elif (response[0] == 1 ):
            result = 'Positive'
            return result,None
    else:
        return ""
    
# Main Function to control the Flow of your Project
def main():
    print("Start of your project")
    load_model()
    print(load_data())
    open_browser()
    #update_app_ui()
    
    
    global scrappedReviews
    global project_name
    global app
    
    project_name = "Sentiment Analysis with Insights"
    #print("My project name = ", project_name)
    #print('my scrapped data = ', scrappedReviews.sample(5) )
    
    # favicon  == 16x16 icon ----> favicon.ico  ----> assests
    app.title = project_name
    app.layout = create_app_ui()
    app.run_server()    
    
    print("End of my project")
    project_name = None
    scrappedReviews = None
    app = None
    
        
# Calling the main function 
if __name__ == '__main__':
    main()


"""
import pickle
def check_review(textData):
    file = open("pickle_model.pkl", 'rb')
    #load model
    recreated_model = pickle.load(file)
    
    #load the vocab
    vocab = pickle.load((open('features.pkl', 'rb')))
    
    from sklearn.feature_extraction.text import CountVectorizer
    recreated_vec = CountVectorizer(decode_error ='replace' ,vocabulary = vocab)

    from sklearn.feature_extraction.text  import TfidfTransformer
    
    transformer = TfidfTransformer()
    
    return recreated_model.predict(transformer.fit_transform(recreated_vec.fit_transform([textData])))
    
list4=[
       "I picked these up for my son in middle school. He loves shoes (basketball shoe crazy) and pays close attention to all the details. He was pleased with these for a dress shoe to add to his collection. He is hard on his shoes and these hold up better than most.",
       "I picked these up for my son in middle school. He loves shoes (basketball shoe crazy) and pays close attention to all the details. He was pleased with these for a dress shoe to add to his collection. He is hard on his shoes and these hold up better than most.",
       "Looks just like I imagined and performs well with all the twists and turns in a busy print shop.",
       'Great would buy again',
       'Very please with the looks, one size fit all, very good quality and i will recommend this product to my friends.'        
       ] 
    
check_review("very good product")
"""