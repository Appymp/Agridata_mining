#!/usr/bin/env python
# coding: utf-8

# # Instagram vector space semantics research

## ver 3.2 

#Monday Sept 6th-Morn
## Identified the language detection model as the slowest part of the program.
## Used print statements to follow the progress of the program on the console.
## Following the “complete guide” YT video on plotly dash.


# In[1]:
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import unicodedata
from langdetect import detect

from wordcloud import WordCloud, STOPWORDS
from nltk.corpus import stopwords

import warnings
warnings.filterwarnings('ignore')

import os


# In[2]:
##Load dataset
pd.set_option('display.max_columns', None) #display all columns.pass none to the max_col parameter
pd.set_option('display.max_colwidth', None) #for indivdual cell full display

organic=pd.read_csv('Organic_6_aug.csv')
organic.head()
print("Dataset loaded")


# In[3]:
##Drop empty rows and create an SI number column
organic=pd.read_csv('Organic_6_aug.csv')
organic.drop(list(range(0,2)),inplace=True) #Drop first 2 entries since Nan failed attempts
organic.reset_index(inplace= True,drop=True)
organic.reset_index(inplace= True)
organic.rename(columns={"index": "ix"},inplace=True) #replace index so that can keep proper ref after dropping rows
organic['ix']=organic['ix'].astype(str) #convert to string type
organic.head(2)

print("Redundant rows removed")


# In[4]:
##Functions for splitting into new columns
def desc_cleaning(org_str): #some hashtags are not separated by a space. This affects dash table display.
  orig_str=str(org_str)
  new_str = orig_str.replace('#'," #") 
  new_str2 = new_str.replace('  '," ")
  return new_str2

def desc_splitting(df): 
  df_upd=df
  df_upd['clean_captions']=" "
  df_upd['hashtags']=" "
  df_upd['cap_mentions']=" "
  df_upd['web_links']=" "
  
  for ind,row in df_upd.iterrows():
      X=str(row['new_desc']).split() #Make sure the description column is cleaned before this.
      X_cc=[]
      X_hstgs=[]
      X_cms=[]
      X_http=[]
      
      for x in X:  
          if not (x.startswith(('#','@','http')) or ('www.' in x) or ('.com' in x)) :
              X_cc.append(x)
              
          if x.startswith('#'):
              X_hstgs.append(x)    
              
          if x.startswith('@'):
              X_cms.append(x)
              
          if (x.startswith('http') or ('www.' in x) or ('.com' in x)):
              X_http.append(x)
              


      df_upd.at[ind,'clean_captions'] = ' '.join(X_cc)        
      df_upd.at[ind,'hashtags'] = ' '.join(X_hstgs)
      df_upd.at[ind,'cap_mentions'] = ' '.join(X_cms)
      df_upd.at[ind,'web_links'] = ' '.join(X_http)
  
  return df_upd


def new_cols(df): ##wrapped for all above functions
  df['new_desc'] = df['description'].apply(lambda x : desc_cleaning(x))
  df_updated=desc_splitting(df)
  return df_updated


# In[5]:
##New columns dataset "df_updated"
df_updated=new_cols(organic)
print("New columns created from splitting func")
#df_updated.tail(1) #to check last line of the dataset.

# In[6]:
##Column display template
cols = df_updated.columns.tolist()
print(cols)

# df_updated=df_updated[['query',
#  'timestamp',
#  'error',
#  'postUrl',
#  'profileUrl',
#  'username',
#  'fullName',
#  'commentCount',
#  'likeCount',
#  'pubDate',
#  'description',
#  'clean_captions',
#  'hashtags',
#  'cap_mentions',
#  'web_links',
#  'imgUrl',
#  'postId',
#  'ownerId',
#  'type',
#  'videoUrl',
#  'viewCount',
#  ]]
#pd.DataFrame(df_updated.iloc[6])
#df_updated.iloc[4:5]

#Cut the dataframe into desired views 
view1=['description','new_desc','clean_captions' ]

# pd.set_option('display.max_columns', 500)
# pd.set_option('expand_frame_repr', False)

# print(pd.DataFrame(df_updated[view1].loc[[11]])) #indexing template example
# print(type(pd.DataFrame(df_updated[view1].loc[[11]])))

# In[7]:
##Text preprocessing
#Tried to make granular transformations for specificity of processing.
#Inorprate num2words if necessary. 

def font_uniformity(x):
    return unicodedata.normalize('NFKC', x)

def convert_lower_case(s):
    return np.char.lower(s)

def remove_punctuation(data):
    symbols = "!\"#$%&()*+-.•/:;<=>?@[\]^_`{|}~\n"
    for i in range(len(symbols)):
        data = np.char.replace(data, symbols[i], ' ')
    
    data = np.char.replace(data, "  ", " ")
    data = np.char.replace(data, ',', '')
    return data

def diff_encodings(s):
    s = np.char.replace(s, "…", " ")
    s = np.char.replace(s, "’", "")
    return s


#Change to reference 'clean_captions' bypass step 1
df_updated['caption_processed']=df_updated['clean_captions'].apply(lambda x: font_uniformity(x))
df_updated['caption_processed_2']=df_updated['caption_processed'].apply(lambda x: convert_lower_case(x))
df_updated['caption_processed_3']=df_updated['caption_processed_2'].apply(lambda x: remove_punctuation(x))
df_updated['caption_processed_4']=df_updated['caption_processed_3'].apply(lambda x: diff_encodings(x))

print("Text preprocessing done")
# In[8]:
##Define view frame to view previous processing steps with exemplars
view2=[]
view2=['clean_captions','caption_processed','caption_processed_2', 'caption_processed_3', 'caption_processed_4']
#view2=view1+view2

#define the rows to display-exemplars of diff test cases
ex_row_list=[11,13,23,106] #subtract 2 because index is reset.

#df_updated[view2].loc[ex_row_list]

#df_updated[view2].head(30)

# In[9]:
#Visualise how many languages are there:    
def lang_det(st):
    try:
        lang=detect(st)
        return lang
    
    except:
        lang="error"
        return lang

view3=['det_lang']
view3= view2+view3

print("Running language detection..")
df_updated['det_lang']= df_updated['caption_processed_2'].apply(lambda x: lang_det(x))
print("Language detection complete")

##visualise the new transformations
plt.figure(figsize=(16,6))
ax= sns.countplot(x= 'det_lang', data=df_updated, order = df_updated['det_lang'].value_counts(ascending=False).index)
ax.set_title('Language distribution')
ax.set_xlabel('Languages')
#ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
#plt.xticks(rotation=90) #outputs array before the graph

for tick in ax.get_xticklabels():
    tick.set_rotation(90)

print("Language countplot graph loaded")

# In[10]:
##View the table with languages included
#Ipython.OutputArea.auto_scroll_threshold = 10 #Tried to set the scroll display threshold. unsuccesful.
view_extracts=['new_desc','det_lang','clean_captions','caption_processed_4','hashtags','cap_mentions','web_links' ]
#df_updated[view_extracts].loc[ex_row_list]

df_updated[view_extracts].head(100)

#add new elemnts to exemplar list
ex_row_list.append(3) #add exemplar accents in differnt language


#df_updated[view3][df_updated['det_lang']=='error'].head(10) 
#Language
#Errors for descriptions which do not have readable text. Either blank or emojis. 
#Fonts normalised but not much improvement.

# In[15]:
""" Co-occurence matrix
# 
# Consider using a co-occurence matrix from the 3_Mar NLP notebook. In this matrix, it summarises the total extent of coccurence of any 2 unique words.
# So each tweet (or caption text) is a matri with either '1' or '0'.
# 
# EAch iteration of the for loop (at lowest nested level), 1 unique word is tested for co-occurence with other words.
# 
# - for loop of tweets
#     - for loop of 1st unique word in range of all unique
#         - for loop of 1st vs {2nd, 3rd..nth word}
#             - lowest level iteration adds a '1' or '0' to a list L1
#         - At end of 1 full cycle of lowest level loop we have list of co-occ of first word with all other words L1:[1,0,0,1,..n]
#     - At end of 2nd level loop, we have a co-occ matrix for a single tweet made of only 1's or 0's
#     - [1st:[0,0,0,1,..n]  
#          2nd:[1,0,1,1,..n]  
#          nth:[0,1,1,0,..n]]  
#          
#          n*n matrix for each tweet.
#          
# - At the end of highest level loop (3rd level) sum of co-occ arrays of all tweets. So a single n*n matrix
#             
# ## Use case of Co-occ matrixes:
# ### To make categories of types of businesses based on co-occuring hashtags
#     - Get co-occurences by different categories. pharma, skincare, etc. Or causally interpret the categories based on the co-occurences. 
#     - In this case, source the full list of co-occurences and just manually classify these into categories.
#     
# ### Define a scale of business-likeness
#     - A higher score may indicate more influence on things like(look for correlation):
#         - choice of hashtags
#         - frequency of posting 
#         - co-occs with "trending" hashtags or terminology
#         - count of hashtags
#         - website :T or F
#         - 
#         
# ### Account for unrelated "#organic" posts
#     - That is think about the usage where they just throw in the term to do some "virtue signalling".
#     - If so what is the context of their signalling. exemplar index '80'. 
#         - It is a vegan food item. But here maybe they are trying to signal "healthy" to eat? 
#         
#     
#     
#     
#Wordcloud for English stopwords
"""

# In[13]:
##Wordcloud with English and French stopwords
print("Loading test Word Cloud")

type(STOPWORDS) #set
len(STOPWORDS) #192

#stop_words = set(stopwords.words("english"))
#len(stop_words)#179

stop_words_fr= set(stopwords.words("french"))
len(stop_words_fr) #157


word_string = ""
for ind,row in df_updated.iterrows():
    word_string += (row['web_links']+" ")
    
#size of plot
fig_dims = (8, 8)
fig, ax = plt.subplots(figsize=fig_dims)

#Define stopwords
en_stopwords = stopwords.words('english')
fr_stopwords = stopwords.words('french')
web_links_sw = ['www','http','https','com']

combined_stopwords = en_stopwords + fr_stopwords 

wordcloud = WordCloud(max_words=100,    
                      stopwords= combined_stopwords,
                      collocations=False,
                      color_func=lambda *args, **kwargs: "orange",
                      background_color='white',
                      width=1200,     
                      height=1000).generate(word_string)
plt.title("#Test word cloud")
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()
#wordcloud.to_file("static/images/en_fr_sw_wc.png")
print("Test Word Cloud displayed")

#With only EN stopwords:
    #  "u", "de", "la", "e", 
    
    
# With EN and FR :
# 'u', 'e', 'o'

""" Notes
#     Punctutations removed and hence ('s ) used for plurals appears to dominate the wordcloud

#     Construct the dahsboard as you move along. It is useful to use a dashboard view to compare charts and build a story as you move through it.

# ## Build a dashboard for the live view
# 
#     Have the graph objects in program buffer.
#     Have a template which accepts the name of the graph objects. So just pass the graph objects into the template and render it.
#     
#     Use plotly dash. All these apps work well with a standalone .py file as obtained from a spyder env.

# #### Notes about plotly wiht dash from the example code: https://www.youtube.com/watch?v=lVYRhHREkGo
#     - First the layout is defined. A parent Div feeds its value to the child Divs. So the dropdown value updates both graphs.
#     - Then for each graph, it has its own "callback" function. This has the input and output components defined, and an associated function which accepts the input and returns the output. The component property which is mentioned in the callback specifies the action type which trigger the value to be passed to the relevant graph.
"""
# In[14]:
##Dashboard application Test shift to dbc.Container
print("Starting dashboard app")

#Import libraries and dataset
import pandas as pd     #(version 1.0.0)
import plotly           #(version 4.5.4) pip install plotly==4.5.4
import plotly.express as px

import dash             #(version 1.9.1) pip install dash==1.9.1
import dash_table
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

import dash_bootstrap_components as dbc

# dff=df_updated[view_extracts]

disp1=['ix','caption_processed_4','hashtags','cap_mentions','web_links'] #select columns to display in the dashtable
df_disp_1 = df_updated[disp1]
df_disp_1.rename(columns={"caption_processed_4": "description"},inplace=True)

app = dash.Dash(__name__, external_stylesheets=[
    "https://stackpath.bootstrapcdn.com/bootstrap/4.1.3/css/bootstrap-grid.min.css"
])
#the rows and columns from dbc worl only with an external stylesheet

#---------------------------------------------------------------

col_sels = ['description','hashtags','cap_mentions','web_links']   #values for dropdown


#---------------------------------------------------------------
# app.layout = html.Div([ 

app.layout = dbc.Container([
    
################################### ROW1-Headers ########################### 
    
    # html.Div([
    #     dbc.Col(html.H3("Instagram data"), className = 'eight columns'),
    #     dbc.Col(html.H3("Wordcloud"), style={'text-align' : "center"})], #one way to align        
    #     className = 'row'),
    
    # html.Div([
    dbc.Row([
        dbc.Col(html.H3("Instagram data"), width={'size':4}),
        dbc.Col(
            html.Button(id='sel-button', n_clicks=0, children="Sel_all"),
            width={'size': 1}, style={'textAlign':"center"}), #another way to align
        dbc.Col(
            html.Button(id='desel-button', n_clicks=0, children="Des_all"),
            width={'size': 1}, style={'textAlign':"center"}),
        
        
        dbc.Col(html.H3("Wordcloud"), style={'size':6,'textAlign' : "center"}) #one way to align        
        ]),
    
    
################################### ROW2-Dropdown and Render Button ########################### 

    
    dbc.Row([
        dbc.Col([
            dcc.Dropdown(id='my-dropdown', multi=True,
                      options=[{'label': x, 'value': x} for x in col_sels],
                      value=["description"] #initial values to pass
                        )],
            width={'size': 6, 'order': 1}
            ),
    
        dbc.Col([
            html.Button(id='my-button', n_clicks=0, children="Render wordcloud")],
            width={'size': 2, "offset": 2, 'order': 2}) #another way to align
            ]),
    
    
################################### ROW3 ###########################    


    html.Div([
        html.Div([
            dash_table.DataTable(
                id='datatable_id',
                data=df_disp_1.to_dict('records'),
                columns=[
                    {"name": i, "id": i, "deletable": False, "selectable": True} for i in df_disp_1.columns
                ],
                editable=False,
                filter_action="native",
                sort_action="native",
                sort_mode="multi",
                row_selectable="multi",
                row_deletable=False,
                selected_rows=[], #this parameter gets updated by interactive selection
                # column_selectable= "multi",
                # selected_columns =[], # parameter updates by column selection
                page_action="native",
                page_current= 0,
                page_size= 10,

                fixed_rows={ 'headers': True, 'data': 0 }, #if the header style is not defined and this is True, then the widths are not properly aligned
    #             virtualization=False,
    #             page_action = 'none',

                style_cell={ #creates a wrapping of the data to constrain column widths. applies to header and data cells
                    'whiteSpace': 'normal',
                    'height': 'auto',
                    'minWidth': '50px', 'width' : '50px','maxwidth': '50px',  # minwidth of col.             
                    #'overflow': 'hidden'
                    },

                style_table={ #For parameters of the table container
                    'height': '400px',
                    'width': '700px',
                    'overflowY': 'auto'
                },

                style_cell_conditional=[
                    {'if': {'column_id': 'ix'},
                     'width': '50px', 'textAlign': 'left'}, #Setting the px values 1400px seems to cover the width of the screen. 

                    {'if': {'column_id': 'description'},
                     'width': '200px', 'textAlign': 'left'},

                    {'if': {'column_id': 'hashtags'},
                     'width': '200px', 'textAlign': 'left'},
                    
                    {'if': {'column_id': 'cap_mentions'},
                     'width': '125px', 'textAlign': 'left'},
                    
                    {'if': {'column_id': 'web_links'},
                     'width': '125px', 'textAlign': 'left'},
                ],
            ),
        ],className='six columns'),
        
        html.Div([
            dcc.Graph(id='wordcloud', figure={}, config={'displayModeBar': False}),
            ], className = 'six columns')
    ]),
])


################################ Render button callback ################################
@app.callback(
    Output('wordcloud','figure'),
    [Input(component_id='datatable_id',component_property='selected_rows'),
    # Input(component_id='my-dropdown', component_property='value')],
    Input(component_id='my-button', component_property='n_clicks')],
    [State(component_id='my-dropdown', component_property='value')],
    prevent_initial_call=False
)


def ren_wordcloud(chosen_rows, n, chosen_cols):
    if len(chosen_cols) > 0: #atleast 1 col to be selected
        if len(chosen_rows)==0:                    
            df_filtered = df_disp_1[chosen_cols]
            # df_filtered = df_disp_1.iloc[:, [chosen_cols]]
            print("Atleast 1 col chosen but no rows. Dataype of df_filtered is", type(df_filtered))

        elif len(chosen_rows) > 0 :
            # df_filtered = df_disp_1.iloc[chosen_rows,[chosen_cols]] #filter by selected rows
            df_filtered = df_disp_1[chosen_cols]
            df_filtered=df_filtered[df_filtered.index.isin(chosen_rows)]
            print("Atleast 1 col chosen and multiple rows selected type. Dataype of df_filtered is", type(df_filtered))
            
    elif len(chosen_cols) == 0:
        raise dash.exceptions.PreventUpdate

    #df_filtered only has the selected rows and columns. So us only the split columns and avoid redunancy
    
    #Combine multiple columns into a single series for wordcloud generation
    df_filtered['comb_cols'] = df_filtered[df_filtered.columns[0:]].apply(
        lambda x: ' '.join(x.dropna().astype(str)),        
        axis=1)

    en_stopwords = stopwords.words('english')
    fr_stopwords = stopwords.words('french')
    web_links_sw = ['www','http','https','com']
    
    combined_stopwords = en_stopwords + fr_stopwords + web_links_sw   

    wordcloud = WordCloud(max_words=100,    
                          stopwords= combined_stopwords,
                          collocations=False,
                          color_func=lambda *args, **kwargs: "orange",
                          background_color='white',
                          width=1600, #1200     
                          height=1000).generate(' '.join(df_filtered['comb_cols'])) #df_filtered has to be a series



    fig_wordcloud = px.imshow(wordcloud, template='ggplot2',
                              ) #title="test wordcloud of eng and fr stopwords"

    fig_wordcloud.update_layout(margin=dict(l=20, r=20, t=30, b=20))
    fig_wordcloud.update_xaxes(visible=False)
    fig_wordcloud.update_yaxes(visible=False)

    return fig_wordcloud


################################ Column highlighting ################################

@app.callback(
    Output('datatable_id', 'style_data_conditional'),
    Input('datatable_id', 'selected_columns')
)

def update_styles(selected_columns):
    return [{
        'if': { 'column_id': i },
        'background_color': '#D2F3FF'
    } for i in selected_columns]

if __name__ == '__main__':
    app.run_server(debug=False)
    
print("Dashboard app running in background")

    # In[16 ]:




# In[16 ]:




# In[15]:

    