#!/usr/bin/env python
# coding: utf-8

# # Instagram vector space semantics research

## ver 3.4

##Sunday Sept 26th-Morning. 
    # Saving the co_occ matrix directly from df to csv takes too long. Pickle also not good.
    # So convert co_occ df to array and then save the file as .npy binary. Big file but quick save.
    
#Next :
    # Lemmatization, Stemming.
    # Attempt creating a co-occurence matrix on hopefully much smaller bag of words
    # Done. If still not possible, use moving window concept to create a cooccurence matrix
    

# In[1]:
print("Starting program..")    
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
import os.path 

from datetime import datetime
app_launch_start=datetime.now() #Set start time for program start
# In[2]:
##Stack multiple datasets saved in "datasets" folder
#Initiate combining the datasets 
pd.set_option('display.max_columns', None) #display all columns.pass none to the max_col parameter
pd.set_option('display.max_colwidth', None) #for indivdual cell full display

print("\nReading datasets in the folder :")
start=datetime.now()
working_dir = "datasets" #folder where the datasets exist
all_files=[]
for root, dirs, files in os.walk(working_dir):
    file_list = []
    for filename in files:
        if filename.endswith('.csv'):
            print(filename)
            file_list.append(os.path.join(root, filename)) 
    # print(file_list)
    # all_files.append(file_list)
    all_files+=file_list
print("Number of datasets considered:", len(all_files))


read_df_sh=[]
df_list=[] 
for file in all_files:
    read_df= pd.read_csv(file)
    # print(read_df.shape)
    read_df_sh.append(read_df.shape)
    df_list.append(read_df)
    # print(read_df.shape)
print("\nList of df shapes are: ",read_df_sh)

df_shape_sum=sum(i for i, j in read_df_sh) 
print("\nSum of rows of multiple datasets", df_shape_sum)

#Check if the exisitng dataframe already includes all sub datasets
existing_comb_df=pd.read_csv('combined_df.csv')
existing_comb_df_rows=existing_comb_df.shape[0]
print("Sum of rows of existing combined dataset", existing_comb_df_rows)

if existing_comb_df_rows<df_shape_sum:
    print("\nNew dataframe exists in the dataset folder..")
    print("Create new combined dataframe")
    

    if df_list:
            final_df = pd.concat(df_list,ignore_index=True) 
            # final_df.to_csv(os.path.join(root, "combined_df.csv"))
            final_df.to_csv("combined_df.csv")
   
else:
    print("\nAll datasets already included in combined dataset")


t=datetime.now() - start #datetime object
s=str(t) #string object
print("Execution time ", s[:-5])

# In[3]:
##Drop empty rows and create an SI number column
print("\nDrop empty rows and add SI number column..")
start=datetime.now()
comb_df=pd.read_csv('combined_df.csv')
print(comb_df.shape)
# comb_df.info() #Check where the empty rows are
comb_df = comb_df.dropna(axis=0, subset=['description']) #Drop empty rows in description
print("Redundant rows removed")
print(comb_df.shape)

#Hardcode index values
comb_df.reset_index(inplace= True,drop=True)
comb_df.reset_index(inplace= True)
comb_df.rename(columns={"index": "ix"},inplace=True) #replace index so that can keep proper ref after dropping rows
comb_df['ix']=comb_df['ix'].astype(str) #convert to string type

t=datetime.now() - start 
s=str(t) 
print("Execution time ", s[:-5])


# In[4]:
##Functions for splitting into new columns
print("\nLoading column splitting functions..")

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

print("Loaded")


# In[5]:
##New columns dataset "df_updated"

print("Applying splitting functions...")

start=datetime.now()

df_updated=new_cols(comb_df)
print("New columns created from splitting func")

t=datetime.now() - start 
s=str(t) 
print("Execution time ", s[:-5])


#df_updated.tail(1) #to check last line of the dataset.

# In[6]:
##Column display template
cols = df_updated.columns.tolist()
#print(cols)

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

print("\nLoading text preprocessing functions..")

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

print("Loaded")
print("Applying text preprocessing...")
start=datetime.now()

df_updated['caption_processed']=df_updated['clean_captions'].apply(lambda x: font_uniformity(x))
df_updated['caption_processed_2']=df_updated['caption_processed'].apply(lambda x: convert_lower_case(x))
df_updated['caption_processed_3']=df_updated['caption_processed_2'].apply(lambda x: remove_punctuation(x))
df_updated['caption_processed_4']=df_updated['caption_processed_3'].apply(lambda x: diff_encodings(x))

print("Text preprocessing done")
t=datetime.now() - start 
s=str(t) 
print("Execution time ", s[:-5])
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
print("\nLoading language detection function..") #Takes 11:30 mins to execute
# start=datetime.now()
    
# def lang_det(st):
#     try:
#         lang=detect(st)
#         return lang
    
#     except:
#         lang="error"
#         return lang

# view3=['det_lang']
# view3= view2+view3

# print("Running language detection...")
# df_updated['det_lang']= df_updated['caption_processed_2'].apply(lambda x: lang_det(x))
# print("Language detection complete")
# t=datetime.now() - start #datetime object
# s=str(t) #string object
# print("Execution time ", s[:-5])


##Export to app referenced df after this last transformation:
##df_updated.to_csv("App_dataframe.csv")


# In[9]:
# ##visualise the new transformations
# print("Making countplot..")
# plt.figure(figsize=(16,6))
# ax= sns.countplot(x= 'det_lang', data=df_updated, order = df_updated['det_lang'].value_counts(ascending=False).index)
# ax.set_title('Language distribution')
# ax.set_xlabel('Languages')
# #ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
# #plt.xticks(rotation=90) #outputs array before the graph

# for tick in ax.get_xticklabels():
#     tick.set_rotation(90)

# print("Language countplot graph loaded")

# In[10]:
##View the table with languages included
#Ipython.OutputArea.auto_scroll_threshold = 10 #Tried to set the scroll display threshold. unsuccesful.
view_extracts=['new_desc','det_lang','clean_captions','caption_processed_4','hashtags','cap_mentions','web_links' ]
#df_updated[view_extracts].loc[ex_row_list]

#df_updated[view_extracts].head(100)

#add new elemnts to exemplar list
ex_row_list.append(3) #add exemplar accents in differnt language


#df_updated[view3][df_updated['det_lang']=='error'].head(10) 
#Language
#Errors for descriptions which do not have readable text. Either blank or emojis. 
#Fonts normalised but not much improvement.


# In[11]:
##Wordcloud with English and French stopwords
print("\nLoading test Word Cloud..")
start=datetime.now()

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
print("Test Word Cloud loaded")
t=datetime.now() - start 
s=str(t) 
print("Execution time ", s[:-5])
#With only EN stopwords:
    #  "u", "de", "la", "e", 
    
    
# With EN and FR :
# 'u', 'e', 'o'


# In[14]

# In[14]
# pd.read_csv("combined_df.csv").shape
# pd.read_csv("App_dataframe.csv").shape

# In[15]
#Filter the dataframe to English, #organic, and remove duplicate posts
##Export to a new dataframe
print("Reading in dataframe for app with filters and duplicate removal..")
ad_1=pd.read_csv("App_dataframe.csv")
ad_1.shape
ad_2=ad_1[ad_1['det_lang']=='en'] #filter only english, removes appr 33%
ad_3=ad_2[ad_2['query']=='#organic'] #filter only #organic scrape query, removes 33%
ad_4=ad_3.drop_duplicates(subset=['postId']) #remove dupl posts, removes 12.5% 

ad_4.drop(['Unnamed: 0','ix','Unnamed: 0.1'],axis =1,inplace=True)

#Create new index for proper selection of the filtered table in app
ad_4.reset_index(inplace= True,drop=True)
ad_4.reset_index(inplace= True)
ad_4.rename(columns={"index": "ix"},inplace=True) #replace index so that can keep proper ref after dropping rows
ad_4['ix']=comb_df['ix'].astype(str) #convert to string type

#ad_4.to_csv("App_dataframe_2.csv", index_label = False)


# In[15] Co-occurrence matrix
#Implement Vecotrization.
#Tokenize the processed description

import ast
import preprocessor as p

ad_4= pd.read_csv("App_dataframe_2.csv")

def remove_stop_words(data): #takes 5:25 mins for the full dataset
    """
    Split/tokenize the words first. Create a list of tokens for each.
    Iterate each token removing stop words from text. 
    """
    row_tokens = data.split()
    return [word for word in row_tokens if not word in stopwords.words("english")] # 25 s for 100return the word only if not in the stop words list
    # return [word for word in row_tokens if not word in set(stopwords.words("english"))] 

#define function to remove newly identified punctuations/encodings
def addi_treatment(s): #Move this to the main preprocessing block if required
    s = np.char.replace(s, "”", "")
    s = np.char.replace(s, "“", "")
    s = np.char.replace(s, "'", "")
    s = np.char.replace(s,"-","") #from viewing bag of wrods in first 100 of cp4
    return s


def preprocess_tweet(row):
    text = p.clean(row) # removes hashtags,emojis,urls, mentions, smileys
    return text



start= datetime.now() 

#Create new columns
ad_4['caption_processed_4'] = ad_4['caption_processed_4'].apply(lambda x : addi_treatment(x))
ad_4['cp4_no_emojis']=ad_4['caption_processed_4'].apply(lambda x: preprocess_tweet(x))
ad_4['cp4_rm_sw'] = ad_4['cp4_no_emojis'].apply(remove_stop_words) 
ad_4['cp4_rm_sw'] = ad_4['cp4_rm_sw'].astype(str) #convert to string for literal eval
ad_4['cp4_rm_sw'] = ad_4['cp4_rm_sw'].apply(lambda x : ast.literal_eval(x)) #preserve lists.
al=datetime.now() - start #start time logged at start
sal=str(al) #string object
print("Execution time for all rows : ", sal[:-5])

print("Type of rows in cp4_rm_sw are ", type(ad_4['cp4_rm_sw'][32000]))
ch_list=['caption_processed_4', 'cp4_rm_sw']
view=ad_4[['caption_processed_4', 'cp4_rm_sw']] #index 35129
# print(view)

ad_4['cp4_rm_sw'][32000]

#ad_4.to_csv("App_dataframe_3.csv", index_label=False) #After filtering for stopwords.
#ad_4.to_pickle("App_dataframe_3.pkl", protocol=0) #preserves the list datatypes. protocol 0 so that it reads back properly in google colab
#App_dataframe_3 now has 
##words minus stopwords 
##treated for "addi_treatment
## tweet preprocessor removed emojis



# In[15]
# Create class of cooccurence embeddings
from sklearn.decomposition import TruncatedSVD

class CooccEmbedding:
    def __init__(self, corpus):
        """
        Takes in the corpus in the form of a list of lists (list of tweets, 
        each tweet being a list of tokens)
        """
        self.corpus = corpus
       
    
    def vocabulary(self):
        """
        Returns the vocabulary associated with the corpus (a list of unique tokens from the corpus)
        """
        self.vocab = []
        [self.vocab.append(word) for tweet in self.corpus for word in tweet if word not in self.vocab] #sent= tweet (made more sense as single tweet)
        return self.vocab #outputs single list of unique words
    
    def coocc_matrix(self): #Note that this is not looking at window size for cooccurence. Considers all words in the tweet.
        """
        Returns the co-occurrence matrix associated with the corpus.
        The co-occurrence matrix is first calculated as a list using comprehensive lists to speed up 
        the iteration process over the vocabulary and the corpus.
        """
        self.len_vocab = len(self.vocab)
        coocc_list = [sum([1 if (self.vocab[i] in tweet and self.vocab[j] in tweet and i != j) else 0 for tweet in self.corpus]) #checks condition each tweet, and sums all tweets.
                      for i in range(self.len_vocab) for j in range(self.len_vocab)] #1st word at 'i' compared with other words at 'j' in unqiue words list.
        coocc_list = np.array(coocc_list)
        # transform the coocc_list into a matrix
        self.coocc_mat = coocc_list.reshape([self.len_vocab, self.len_vocab])
        return self.coocc_mat
        
    def svd_reduction(self, n_components = 2, n_iter = 10):
        """
        Performs a singular value decomposition on the co-occurrence matrix M and returns the truncated
        matrix U, where M = UDV^T.
        """
        svd = TruncatedSVD(n_components = n_components, n_iter = n_iter)
        self.reduced_matrix = svd.fit_transform(self.coocc_mat)
        return self.reduced_matrix
    
    def vocab_ind_to_plot(self, vocab_to_plot):
        """
        Takes in the a list of vocabulary to plot and returns a dictionary in the form of
        {word:index} which is a subeset of the vocabulary.
        """
        vocab_dict = dict(zip(self.vocab, range(self.len_vocab)))
        self.dic_vocab_to_plot = {key:value for key, value in vocab_dict.items() if key in vocab_to_plot} #Returns index of original vocab for each word in the vocab to plot.
        return self.dic_vocab_to_plot
        



# In[12]: #Test bag of words sizes
#Trim the vocab so that cooccurnce matrix can be computed.
ad_5=pd.read_pickle('App_dataframe_3.pkl')

def list_of_words(data): #takes 5:25 mins for the full dataset
    row_tokens = data.split()
    return [word for word in row_tokens] # 25 s for 100return the word only if not in the stop words list

#Prepare the list of lists for the bag of words input in Coocc class:
ad_5['caption_processed_4'] = ad_5['caption_processed_4'].apply(list_of_words)
words_cp4=[row_list for row_list in ad_5['caption_processed_4']] 
words_cp4[0:3]

ad_5['cp4_no_emojis'] = ad_5['cp4_no_emojis'].apply(list_of_words)
words_cp4_no_emojis=[row_list for row_list in ad_5['cp4_no_emojis']] 

words_cp4_rm_sw=[row_list for row_list in ad_5['cp4_rm_sw']] # List of lists

##Tests on the first 100 rows
inst = CooccEmbedding(words_cp4[:100]) 
print("Number of unique words: ",len(inst.vocabulary())) #1587 words in cp4

inst = CooccEmbedding(words_cp4_no_emojis[:100]) 
print("Number of unique words: ",len(inst.vocabulary())) #1418 words in words_cp4_no_emojis

inst = CooccEmbedding(words_cp4_rm_sw[:100]) 
print("Number of unique words: ",len(inst.vocabulary())) #1309 words in words_cp4_rm_sw


#For full dataframe:
inst_1 = CooccEmbedding(words_cp4) 
inst_2 = CooccEmbedding(words_cp4_no_emojis)
inst_3 = CooccEmbedding(words_cp4_rm_sw) 

#Takes time so time it
time_vocab=datetime.now()

print("cp4 unique words: ",len(inst_1.vocabulary())) #71616 words,#46066 words,#45925 words
t=datetime.now() - time_vocab 
s=str(t) 
print("cp4 counting time: ", s[:-5])

print("cp4_no_emojis unique words: ",len(inst_2.vocabulary())) #46066 words
t=datetime.now() - time_vocab 
s=str(t) 
print("cp4_no_emojis counting time: ", s[:-5])

print("cp4_rm_sw unique words: ",len(inst_3.vocabulary())) #45925 words
t=datetime.now() - time_vocab 
s=str(t) 
print("cp4_rm_sw counting time: ", s[:-5])



# In[12]: #Lemmatization and stemming

ad_5=pd.read_pickle('App_dataframe_3.pkl')

#Porter Stemming
from nltk.stem import PorterStemmer
porter = PorterStemmer()

def porter_stemming(row_list):
    new_word_list=[porter.stem(word) for word in row_list]    
    return new_word_list

start = datetime.now()
ad_5['rm_sw_stem']=ad_5['cp4_rm_sw'].apply(porter_stemming)
t=str(datetime.now()-start)
print("Time for stemming full dataframe: ", t[:-5]) #27.2 seconds for full dataframe

words_rm_sw_stem=[row_list for row_list in ad_5['rm_sw_stem']]

print(words_cp4_rm_sw[:2])
print("\n",words_rm_sw_stem[:2])

inst_4=CooccEmbedding(words_rm_sw_stem) 

start=datetime.now()
print("rm_sw_stem unique words: ",len(inst_4.vocabulary())) #34051 unique words
t=str(datetime.now() - start)
print("rm_sw_stem counting time: ", t[:-5]) #1:26 mins


#Lemmatization 
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()

def wordnet_lemmatizer_func(row_list):
    new_word_list=[wordnet_lemmatizer.lemmatize(word) for word in row_list]    
    return new_word_list

ad_5['cp4_rm_sw'][3:5]

#For testing the functions
test_series= ad_5['cp4_rm_sw'][3:5].apply(wordnet_lemmatizer_func)
print(ad_5['cp4_rm_sw'][3:5])
print("\n Stemmed",ad_5['rm_sw_stem'][3:5])
print("\nLemmatized",test_series)

##Repeat as in previous step for stemming

start = datetime.now()
ad_5['rm_sw_lemt']=ad_5['cp4_rm_sw'].apply(wordnet_lemmatizer_func)
t=str(datetime.now()-start)
print("Time for lemmatizing full dataframe: ", t[:-5]) #5.5 seconds for full dataframe

words_rm_sw_lemt=[row_list for row_list in ad_5['rm_sw_lemt']]

inst_5=CooccEmbedding(words_rm_sw_lemt) 

start=datetime.now()
print("rm_sw_lemt unique words: ",len(inst_5.vocabulary())) #41526 unique words
t=str(datetime.now() - start)
print("rm_sw_lemt counting time: ", t[:-5]) #1:48 mins





#ad_5.to_pickle("App_dataframe_4.pkl", protocol=0) #with 2 new columns stemming and lemmatization
#make sure to export the new dataframe with the stemming and lemmatization



# In[12]: #Moving window size to create the cooccurence matrix
#Takes a long time. Run this only for new datasets.

# from collections import defaultdict

# ad_6=pd.read_pickle('App_dataframe_4.pkl')

# words_rm_sw_lemt=[row_list for row_list in ad_6['rm_sw_lemt']]

# text=words_rm_sw_lemt #full dataset around 11 mins?. window 2 [shape 45925]
  
# def co_occ_windows(sentences, window_size):
#     d = defaultdict(int)
#     vocab = set()
#     for text in sentences: #text is a list of lists
#         # preprocessing (use tokenizer instead)
#         # text = text.lower().split()
#         # iterate over sentences
#         for i in range(len(text)):
#             token = text[i]
#             # print("\ntoken is: ",token)
#             vocab.add(token)  # add to vocab
#             # print("vocab set now contains: ",vocab)
#             next_token = text[i+1 : i+1+window_size] #Only forward co-occurence?
#             # print("next token is: ",next_token) #test
#             for t in next_token:
#                 key = tuple( sorted([t, token]) )
#                 # print("key is: ",key)
#                 d[key] += 1
#                 print("default dict value at key is: ",d[key]) #Each key is a tuple and is unique. added with 1. And these will sum over themselves for other posts. 
    
    
#     # formulate the dictionary into dataframe
#     vocab = sorted(vocab) # sort vocab
#     df = pd.DataFrame(data=np.zeros((len(vocab), len(vocab)), dtype=np.int16),
#                       index=vocab,
#                       columns=vocab) #Initialise a dataframe of zeroes
#     for key, value in d.items():
#         df.at[key[0], key[1]] = value
#         df.at[key[1], key[0]] = value
#     return df
    
# #Test out the function
# start=datetime.now()
# co_occ_df = co_occ_windows(text, 2)
# print("shape of co_occ matrix is: ",co_occ_df.shape)
# t=str(datetime.now() - start)
# print("Time for execution: ", t[:-5])
   
# #Time for execution 10:43 mins  for rm_sw_lemt
# co_occ_arr =co_occ_df.to_numpy() #convert to an array
# co_occ_arr 

###SVD seems to be useful but takes too long to execute.
## start = datetime.now()
## svd = TruncatedSVD(n_components = 2, n_iter = 10)
## Coocc_svd_matrix = svd.fit_transform(co_occ_arr) #Takes too long. Create a funcion to remove occurences which are sparse.
## t = str(datetime.now()-start)
## print("Time taken for svd: ",t[:-5])



# In[12]: #Import relevant files without to avoid all the preproessing of previous steps
# save to npy file
#from numpy import save #Have to import explicitly to save array as binary
#save('co_occ_arr.npy', co_occ_arr)

# load npy from local
from numpy import load
co_occ_arr = load('co_occ_arr.npy')

ad_6=pd.read_pickle('App_dataframe_4.pkl')







# co_occ_df.to_csv("Coocc_rm_sw_lemt.csv", index_label=False)#Takes too long. consider SVD to eliminate redundant 0 co-occurence terms.


# In[12]:

    



# In[12]:
##Dashboard application Test shift to dbc.Container
print("\nStarting dashboard app...")

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

print("Loading the dataframe..")
start = datetime.now()

df_updated=pd.read_csv("App_dataframe_2.csv")

print("Dataframe loaded")
t=datetime.now() - start 
s=str(t) 
print("Execution time: ", s[:-5], "\n\n")

df_disp_1 = df_updated[disp1]
df_disp_1.rename(columns={"caption_processed_4": "description"},inplace=True)

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
#the rows and columns from dbc worl only with an external stylesheet

al=datetime.now() - app_launch_start #start time logged at start of program code
sal=str(al) #string object
print("Total time taken to launch app", sal[:-5])

#---------------------------------------------------------------

col_sels = ['description','hashtags','cap_mentions','web_links']   #values for dropdown


#---------------------------------------------------------------
# app.layout = html.Div([ 

app.layout = dbc.Container([    
################################### ROW1-Headers ########################### 
    
    dbc.Row([
        dbc.Col(html.H3("Instagram data"), width={'size':3}),
        dbc.Col(
            # html.Button(id='sel-button', n_clicks=0, children="Sel_all"),
            dbc.Button(id='sel-button', n_clicks=0, children="Sel_all", className="mt-5 mr-2"),            
            width={'size': 0.5}, style={'textAlign': "left"}), #another way to align
        dbc.Col(
            dbc.Button(id='desel-button', n_clicks=0, children="Des_all", className="mt-5"),
            width={'size': 0.5}, style={'textAlign':"left"}),
        
        dbc.Col(html.H3("Wordcloud"), width={'size':5, 'offset':2}, style={'textAlign' : "center"}),         
        # dbc.Col([
        #     html.Button(id='my-button', n_clicks=0, children="Render wordcloud")],
        #     width={'size': 3}, style = {'textAlign' : "left"})
        ], no_gutters=False),
    
    
################################### ROW2-Dropdown ########################### 

    
    dbc.Row([
        dbc.Col([
            dcc.Dropdown(id='my-dropdown', multi=True,
                      options=[{'label': x, 'value': x} for x in col_sels],
                      value=["description"], #initial values to pass
                      style={'width':'690px'}
                        )])
            ]),
    # width={'size': 6}), #this was in dbc.Col.This is fluid width while resizing
    # But since DashTable is fixed width, width is specified(acts fixed) in dcc.Dropdown
    
################################### ROW3 ###########################    

    dbc.Row([        
        dbc.Col(
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
                    'height': '300px',
                    'width': '690px',
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
            width={'size': 6}),
        
        dbc.Col(
            dcc.Graph(id='wordcloud', figure={}, config={'displayModeBar': True},
                       style={'width':'600px' ,'height':'350px'}),
                )
        #,width={'size': 6},  width = {'size': 5, 'offset':1}
            ],no_gutters=False)
        ],fluid=True) #use fluid to strecth to the sides of the webpage
#'layout': {'height': '100px'}


################################ Select all button ################################

@app.callback(
    [Output('datatable_id', 'selected_rows')], #references ordered 0 to n index of larger table irrespective of actual index value.
    [Input('sel-button', 'n_clicks'),
    Input('desel-button', 'n_clicks')],
    [State('datatable_id', 'derived_virtual_selected_rows'), #virtual selected row is what is selected.
     State('datatable_id', 'derived_virtual_data')], #virtual_data is displayed table. Even after filtering. 
    prevent_initial_call=True
)

def select_deselect(selbtn, deselbtn, selected_rows,filtered_table):
    ctx = dash.callback_context
    if ctx.triggered:
        print(ctx.triggered)
        trigger = (ctx.triggered[0]['prop_id'].split('.')[0])
        if trigger == 'sel-button':
            print("\n\nSelect button clicked")
            print("Length of Selected_rows is: ", len(selected_rows))
            wc_list= []            
            wc_list=[[row['ix'] for row in filtered_table]] 
            wc_list = [[int(i) for i in wc_list[0]]] #convert to int for index reference
           
            print("Displayed table has:", len(filtered_table), 'rows')                
            print("Wordcloud list contains:", len(wc_list[0]), "elements")

            return wc_list
        else:
            print("\n\nDeselect button clicked")
            print("Length of Selected_rows is: ", len(selected_rows))
            return [[]]


################################ Wordcloud callback ################################

@app.callback(
    Output('wordcloud','figure'),
    [Input(component_id='datatable_id',component_property='selected_rows'),
    Input(component_id='my-dropdown', component_property='value')],
    # Input('sel-button', 'n_clicks'),
    # Input('desel-button', 'n_clicks')],
    # Input(component_id='my-button', component_property='n_clicks')],
    # [State(component_id='my-dropdown', component_property='value')],
    prevent_initial_call=False
)


def ren_wordcloud(chosen_rows, chosen_cols):
    if len(chosen_cols) > 0: #atleast 1 col to be selected
        if len(chosen_rows)==0:                    
            df_filtered = df_disp_1[chosen_cols] #if no rows selected consider all rows
            # df_filtered = df_disp_1.iloc[:, [chosen_cols]]
            #print("Atleast 1 col chosen but no rows. Dataype of df_filtered is", type(df_filtered))

        elif len(chosen_rows) > 0 :
            # df_filtered = df_disp_1.iloc[chosen_rows,[chosen_cols]] #filter by selected rows
            df_filtered = df_disp_1[chosen_cols]
            df_filtered=df_filtered[df_filtered.index.isin(chosen_rows)]
            #print("Atleast 1 col chosen and multiple rows selected type. Dataype of df_filtered is", type(df_filtered))
            
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
    
    print("\nNo of rows in WC are: ",len(df_filtered))
    wordcloud = WordCloud(max_words=100,    
                          stopwords= combined_stopwords,
                          collocations=False,
                          color_func=lambda *args, **kwargs: "orange",
                          background_color='white',
                          width=1700, #1200,1700    
                          height=1000, #1000
                          random_state=1).generate(' '.join(df_filtered['comb_cols'])) #df_filtered has to be a series

    #print(' '.join(df_filtered['comb_cols']))

    fig_wordcloud = px.imshow(wordcloud, template='ggplot2',
                              ) #title="test wordcloud of eng and fr stopwords"

    fig_wordcloud.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    fig_wordcloud.update_xaxes(visible=False)
    fig_wordcloud.update_yaxes(visible=False)

    return fig_wordcloud


################################ Column highlighting ################################

@app.callback(
    Output('datatable_id', 'style_data_conditional'),
    # Input('datatable_id', 'selected_columns')
    Input(component_id='my-dropdown', component_property='value')
)

def update_styles(selected_columns):
    return [{
        'if': { 'column_id': i },
        'background_color': '#D2F3FF'
    } for i in selected_columns]

if __name__ == '__main__':
    app.run_server(debug=False)

# app.run_server(debug=True)    
# print("Dashboard app running in background")

     # In[13]:




# In[14]:




# In[15]:

    