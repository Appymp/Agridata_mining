#!/usr/bin/env python
# coding: utf-8

# # Instagram vector space semantics research

## New version 4!-Final stretch additions

## Saturday Jan 15th-evening. TSB
    # Deleted old matplotlib graph
    
    

    

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
# In[2]: Combine datasets; final_df.to_csv("combined_df.csv")
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
comb_df.rename(columns={"index": "ix"},inplace=True) #create col out of index so that can keep proper ref after dropping rows
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

# In[6]: Table column names
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


# In[7]: Text preprocessing; fonts, lower, punct, char_encodings
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

# In[8]: "views" and "exemplars"
##Define view frame to view previous processing steps with exemplars
view2=[]
view2=['clean_captions','caption_processed','caption_processed_2', 'caption_processed_3', 'caption_processed_4']
#view2=view1+view2

#define the rows to display-exemplars of diff test cases
ex_row_list=[11,13,23,106] #subtract 2 because index is reset.

#df_updated[view2].loc[ex_row_list]

#df_updated[view2].head(30)

# In[9]: Language detection to_csv("App_dataframe.csv")
# Visualise how many languages are there:    
print("\nLoading language detection function..") #Takes 11:30 mins to execute
start=datetime.now()
    
def lang_det(st):
    try:
        lang=detect(st)
        return lang
    
    except:
        lang="error"
        return lang

view3=['det_lang']
view3= view2+view3

print("Running language detection...")
df_updated['det_lang']= df_updated['caption_processed_2'].apply(lambda x: lang_det(x))
print("Language detection complete")
t=datetime.now() - start #datetime object
s=str(t) #string object
print("Execution time ", s[:-5])


#Export to app referenced df after this last transformation:
#df_updated.to_csv("App_dataframe.csv")


# In[9]: Language countplot
##visualise the new transformations
df_updated = pd.read_csv('App_dataframe.csv')

print("Making countplot..")
plt.figure(figsize=(16,6))
ax= sns.countplot(x= 'det_lang', data=df_updated, order = df_updated['det_lang'].value_counts(ascending=False).index)
ax.set_title('Language distribution')
ax.set_xlabel('Languages')
#ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
#plt.xticks(rotation=90) #outputs array before the graph

for tick in ax.get_xticklabels():
    tick.set_rotation(90)

print("Language countplot graph loaded")

# In[10]: #Selective viewing of dataframe with "views" and "exmplars"
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

# df_updated['clean_captions'][10] #test individual rows

word_string = " ".join(df_updated['clean_captions'].dropna().astype(str)) #which column to generate WC for
# print(word_string[10:1000])


#Define stopwords
# type(STOPWORDS) #set
# len(STOPWORDS) #192
en_stopwords = stopwords.words('english') #179ii
fr_stopwords = stopwords.words('french') #157
web_links_sw = ['www','http','https','com']

combined_stopwords = en_stopwords + fr_stopwords 

wordcloud = WordCloud(max_words=100,    
                      stopwords= combined_stopwords,
                      collocations=False,
                      color_func=lambda *args, **kwargs: "orange",
                      background_color='white',
                      width=1200,     
                      height=1000).generate(word_string)

fig_dims = (8, 8) #size of plot
fig, ax = plt.subplots(figsize=fig_dims)
plt.title("#Test word cloud")
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()
#wordcloud.to_file("static/images/en_fr_sw_wc.png")
print("Test Word Cloud loaded")
t=datetime.now() - start 
s=str(t) 
print("Execution time ", s[:-5])

#With only EN stopwords:  "u", "de", "la", "e", still present 
#With EN and FR : 'u', 'e', 'o' still present


# In[14]
pd.read_csv("combined_df.csv").shape
pd.read_csv("App_dataframe.csv").shape

# In[14] Filter; read_csv("App_dataframe.csv") -> to_csv("App_dataframe_2.csv")
#Filter the dataframe to English, #organic, and remove duplicate posts
##Export to a new dataframe
print("Reading in dataframe for app with filters and duplicate removal..")
ad_1=pd.read_csv("App_dataframe.csv")
ad_1.shape
ad_2=ad_1[ad_1['det_lang']=='en'] #filter only english, removes appr 33%
print("Only English posts",len(ad_2))

ad_3=ad_2[ad_2['query']=='#organic'] #filter only #organic scrape query, removes 33%
print("Only organic posts: ", len(ad_3))

ad_4=ad_3.drop_duplicates(subset=['postId']) #remove dupl posts, removes 12.5% 
print("After duplicate removal: ", len(ad_4))

ad_4.drop(['Unnamed: 0','ix','Unnamed: 0.1'],axis =1,inplace=True)

#Create new index for proper selection of the filtered table in appl
ad_4.reset_index(inplace= True,drop=True)
ad_4.reset_index(inplace= True)
ad_4.rename(columns={"index": "ix"},inplace=True) #replace index so that can keep proper ref after dropping rows
ad_4['ix']=ad_4['ix'].astype(str) #convert to string type

#ad_4.to_csv("App_dataframe_2.csv", index_label = False)


# In[15] Co-occurrence matrix; read_csv("App_dataframe_2.csv") --> to_pickle("App_dataframe_3.pkl")
# Implement Vecotrization.
# Tokenize the processed description

import ast
import preprocessor as p

ad_4= pd.read_csv("App_dataframe_2.csv")


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

def remove_stop_words(data): #takes 5:25 mins for the full dataset
    """
    Split/tokenize the words first. Create a list of tokens for each.
    Iterate each token removing stop words from text. 
    """
    row_tokens = data.split()
    return [word for word in row_tokens if not word in stopwords.words("english")] # 25 s for 100return the word only if not in the stop words list
    # return [word for word in row_tokens if not word in set(stopwords.words("english"))] 




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

##App_dataframe_3 now has 
#words minus stopwords 
#treated for "addi_treatment
#tweet preprocessor removed emojis


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
        self.len_vocab = len(self.vocab) #Initiliase this return because coocc module not used.
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
        



# In[12]: #Test bag of words sizes and instantiate Coocc_Embedding class
# Trim the vocab so that cooccurnce matrix can be computed.

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


# In[12]: #Lemmatization and Stemming ->create new pkl App_dataframe_4 

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
# In[12.b]: #More filtering of the dataframe App_dataframe_4->App_dataframe_5

 ##Filter out duplicate posts and create another pickle App_dataframe_5

ad_6=pd.read_pickle('App_dataframe_4.pkl')
len(ad_6) # 35131

ad_7 = ad_6.drop_duplicates(subset=['caption_processed_4'])
len(ad_7) # from 35131 of ad_6, ad_7 is reduced to 30240; Another 14.2% drop in the rows.

ad7.drop(['ix'], axis=1, inplace = True) #drop any older 'ix' columns
ad7.reset_index(inplace= True,drop=True) #reset and drop old index
ad7.reset_index(inplace= True) #create new ix column
ad7.rename(columns={"index": "ix"},inplace=True) 

# ad7.to_pickle("App_dataframe_5.pkl", protocol=0)

len(ad_7)

# In[12]: #Cooccurence array; save('co_occ_arr.npy'); choose 'rm_sw_lemt'
# Takes a long time. Run this only for new datasets.

from collections import defaultdict

# ad_6=pd.read_pickle('App_dataframe_4.pkl')
ad_6=pd.read_pickle('App_dataframe_5.pkl') #duplicate posts removed. 

words_rm_sw_lemt=[row_list for row_list in ad_6['rm_sw_lemt']] #row_list represents each words in each row
print("Bag of words created..")
len(words_rm_sw_lemt)
# words_rm_sw_lemt

text=words_rm_sw_lemt #full dataset 
# text=words_rm_sw_lemt[:1000] #4862secs 6584 shape; 3.7 secs for 1000
# text=words_rm_sw_lemt[:200] \
# text=words_rm_sw_lemt[:1000]

# text = words_rm_sw_lemt[5:6]
text
def co_occ_windows(sentences, window_size):
    d = defaultdict(int) #gives default vaue for non-existing keys
    vocab = set()
    for text in sentences: #text is each post within full bag
        print(text)
        for i in range(len(text)):
            token = text[i]
            # print("\ntoken is: ",token)
            vocab.add(token)  # add to vocab
            # print("vocab set now contains: ",vocab)
            
            # #test----
            # text=['1','2','3','4','5','6','7']
            # i= 1
            # window_size =2
            # #test----
            
            
            # coocc_window = text[i+1 : i+1+window_size] #Rolling ahead window scope; sparse
            coocc_window = text[i-window_size : i] + text[i+1 : i+1+window_size]  #Rolling bidirectional window. Blind spots at start and end.
            # print("coocc_window is: ",coocc_window) #test
            
                        
            for t in coocc_window:
                key = tuple( sorted([t, token]) )
                # print("key is: ",key)
                d[key] += 1 #at the tuple key, increase the value by 1
                # print("default dict value at key is: ",d[key]) #Each key is a tuple and is unique. added with 1. And these will sum over themselves for other posts. 
    
    
    # formulate the dictionary into dataframe
    vocab = sorted(vocab) # sort vocab
    global custom_vocab
    custom_vocab = vocab #test object
    
    # print("Sorted vocab at index is", vocab[0])
    df = pd.DataFrame(data=np.zeros((len(vocab), len(vocab)), dtype=np.int16),
                      index=vocab,
                      columns=vocab) #Initialise a dataframe of zeroes
    for key, value in d.items():
        df.at[key[0], key[1]] = value
        df.at[key[1], key[0]] = value
    return df
    
#Test out the function
start=datetime.now()
# co_occ_df = co_occ_windows(text,5) #full array w4 takes 24:25 mins rolling ahead 
co_occ_df = co_occ_windows(text,2) #bidrectional. Effective window size is 5 takes  mins
print("shape of co_occ matrix is: ",co_occ_df.shape)
t=str(datetime.now() - start)
print("Time for execution: ", t[:-5])

print("Custom vocab shape: ", custom_vocab)
print("First 10 vocab shape: ", custom_vocab[:10])
    

co_occ_arr =co_occ_df.to_numpy() #convert to an array
co_occ_arr 
from numpy import save #Have to import explicitly to save array as binary
# save('ad5_co_occ_arr_w4.npy', co_occ_arr)
# save('ad5_co_occ_arr_w5.npy', co_occ_arr)
# save('ad5_co_occ_arr_w5_bi.npy', co_occ_arr)

# In[12]Perform Singular Value Decomposition on the array. SVD_matrix.npy

from numpy import load
# co_occ_arr = load('co_occ_arr_w2.npy') #41526
# co_occ_arr = load('ad5_co_occ_arr_w4.npy') #shape 41526
# co_occ_arr = load('ad5_co_occ_arr_w5.npy') #shape 41526
co_occ_arr = load('ad5_co_occ_arr_w5_bi.npy') #shape 41526

print("shape of co-occ matrix is: ", co_occ_arr.shape)

# ad_6=pd.read_pickle('App_dataframe_4.pkl')
ad_6=pd.read_pickle('App_dataframe_5.pkl') #duplicate posts removed. 

from sklearn.decomposition import TruncatedSVD    

# co_occ_arr_sl=co_occ_arr[0:10000,0:10000] #slice array
co_occ_arr_sl=co_occ_arr #Full cooccurence matrix

start = datetime.now()

from scipy.sparse import coo_matrix
co_occ_arr_sl_coo=coo_matrix(co_occ_arr_sl) 
co_occ_arr_sl_coo = co_occ_arr_sl_coo.asfptype() #convert to COO format before arpack

svd = TruncatedSVD(n_components = 2, n_iter = 10, algorithm = "arpack")  #28.4 secs
# svd = TruncatedSVD(n_components = 2, n_iter = 10, algorithm = "randomized") #18.3 secs
Coocc_svd_matrix = svd.fit_transform(co_occ_arr_sl_coo)
Coocc_svd_matrix.shape

t = str(datetime.now()-start)
print("Time taken for svd: ",t[:-5]) 

print(type(Coocc_svd_matrix))

from numpy import save 
# save('svd_rand_w2.npy', Coocc_svd_matrix) #Randomised svd done here
# save('svd_arpack_w4.npy', Coocc_svd_matrix) #arpack algo used
# save('ad5_svd_arpack_w4.npy', Coocc_svd_matrix) #arpack algo used
# save('ad5_svd_arpack_w5.npy', Coocc_svd_matrix) #arpack algo used
# save('ad5_svd_arpack_w5_bi.npy', Coocc_svd_matrix) #arpack algo used

# In[12]: svd array->plotting dataframe
########## For specific word list.


# coocc_svd_matrix = load('svd_arpack_w2.npy') #Load arpack mode
# coocc_svd_matrix = load('svd_arpack_w3.npy') #Load arpack mode
# coocc_svd_matrix = load('ad5_svd_arpack_w4.npy') #Load arpack mode
# coocc_svd_matrix = load('ad5_svd_arpack_w5.npy') #Load arpack mode
coocc_svd_matrix = load('ad5_svd_arpack_w5_bi.npy') #Load arpack mode

ad_6=pd.read_pickle('App_dataframe_5.pkl') #df_8hrs is converted to string for col combining in app

print("Custom vocab head 10: ",custom_vocab[:10])
# print("Coocc vocab head 10: ",coocc_vocab[:10])
custom_vocab[-10:-1]
cust_vocab = custom_vocab #entire word vocab. Common for all window sizes 
# len(vocab_full) #41526
# vocab_to_plot = ['packaging', 'vanilla', 'coffee','cacao', 'sustainable','skincare','aroma']
certs = ['gots','oeko','cosmos','ecocert','fsc'] #certifications
fruits = ['apple', 'orange']
rel_concs = ['fruits', 'certification','textile']

# vocab_to_plot = certs + fruits + rel_concs
vocab_to_plot = cust_vocab #full vocab

print("Making dict_to_plot..")
start=datetime.now()

#custom_vocab and co_occ vocab are indexed differently. coocc mat is based on custom(sorted) vocab
# dict_to_plot = inst.vocab_ind_to_plot(vocab_to_plot) #with above finding instantiating coocc class is redundant.

#dict_to_plot for the custom_vocab 
def vocab_ind_to_plot(cust_vocab, vocab_to_plot):
        """
        Takes in the a list of full vocabulary and list to plot and returns a dictionary in the form of
        {word:index} which is a subeset of the vocabulary.
        """
        vocab_dict = dict(zip(cust_vocab, range(len(cust_vocab)))) #custom vocab is sorted
        dic_to_plot = {key:value for key, value in vocab_dict.items() if key in vocab_to_plot} #Returns index of original vocab for each word in the vocab to plot.
        return dic_to_plot

dict_to_plot = vocab_ind_to_plot(cust_vocab, cust_vocab)

print("Dict_to_plot is ",dict_to_plot)

wl1=str(datetime.now()-start)
print("Time to make dict_to_plot: ", wl1[:-5])

data_list=[]
for word, ind in dict_to_plot.items():
    # print(word, coocc_svd_matrix[ind, 0],coocc_svd_matrix[ind, 1])
    #coocc mat does not have words. And it is indexed in vocab sorted.
    row_list=[word, coocc_svd_matrix[ind, 0], coocc_svd_matrix[ind, 1]] 
    data_list.append(row_list)
    
# print(data_list)
type(coocc_svd_matrix)
print(coocc_svd_matrix.shape)

vocab_words_df= pd.DataFrame.from_records(data_list, columns=['word','x','y'])  
wl2=str(datetime.now()-start)
print("Time till vocab_words_df ready: ", wl2[:-5]) #55secs for full vocab 41526 words
vocab_words_df
vocab_words_df.dtypes

# vocab_words_df.to_pickle("vocab_words_svd_w2.pkl", protocol=0)
# vocab_words_df.to_pickle("vocab_words_svd_w3.pkl", protocol=0)
# vocab_words_df.to_pickle("vocab_words_svd_w4.pkl", protocol=0)
# vocab_words_df.to_pickle("ad5_vocab_words_svd_w5.pkl", protocol=0)
# vocab_words_df.to_pickle("ad5_vocab_words_svd_w5_bi.pkl", protocol=0)
# print(vocab_words_df)

# In[12]: Visualise plotly scatter  - Coocc SVD vectors
import plotly.io as pio #To plot in browser
pio.renderers.default='browser' 
# pio.renderers.default='svg' #inside spyder. not working
# pio.renderers
# vocab_words_df_full=pd.read_pickle('ad5_vocab_words_svd_w5.pkl') #corrected word index
vocab_words_df_full=pd.read_pickle('ad5_vocab_words_svd_w5_bi.pkl') #corrected word index

certs = ['gots','oeko','cosmos','ecocert','fsc'] #certifications
fruits = ['apple', 'orange', 'vanilla','cacao']
consc = ['sustainably', 'ethically']
rel_concs = ['fruits', 'certification','textile', 'cosmetics']
vocab_to_plot = certs + fruits + rel_concs

#filtered plot df
vocab_words_df = vocab_words_df_full[vocab_words_df_full['word'].isin(vocab_to_plot)]

fig = px.scatter(vocab_words_df_full, x="x", y="y", text="word", log_x=False, size_max=60,
                  # trendline = "rolling",
                  hover_name = 'word'
                  )
fig.update_traces(textposition='top center')
fig.show()    

    
# In[12]: ## Word2vec model and saving .txt files
import os
import re
import time
from gensim.models import Word2Vec
from gensim.models import KeyedVectors #For loading the model
from sklearn.manifold import TSNE #for reducing the dimensions 
from tqdm import tqdm
tqdm.pandas()


ad_7 = pd.read_pickle('App_dataframe_5.pkl')


words_rm_sw_lemt = [row_list for row_list in ad_7['rm_sw_lemt']]
# len(words_rm_sw_lemt)

print("number of sentences is: ",len(words_rm_sw_lemt))
train_sentences = words_rm_sw_lemt #list of lists sentences and words

start=datetime.now()
model = Word2Vec(sentences=train_sentences, sg=1,vector_size=300, window=5, min_count=1, workers=4)
# model.save("thesis_word2vec.model") #save a part of the training
# t=str(datetime.now()-start)
# print("Time taken for Word2vec on full dataset is: ", t[:-5]) #28 secs for full dataset


# model = Word2Vec.load("thesis_word2vec.model") #load and continue training the sample
# model.train([["hello", "world"]], total_examples=1, epochs=1) #train new samples
# model.wv.vector_size #100 if vector_size set to this
# model.wv.get_vector('vanilla')


# model.wv.most_similar('sustainable')
# model.wv.most_similar('organic', topn = 20)
# model.wv.most_similar(['agriculture','agritech','coffee'])
# model.wv.most_similar(['agriculture','sustainable','coffee','labour'])

# model.wv.save_word2vec_format('organic_glove_300d_w5.txt') 
#After training full dataset save in the format
#Try avoiding creating this text file because they are large.


print("Time taken for creating model: ", str(datetime.now()-start)[:-5])
#44 secs for 300d_w4



# In[12]: Export Tsne dataframe by transforming w2c text files 
# Note how before the saving the model, called as model.wv 
# Takes 45-60 mins for full dataset.

# w2v = KeyedVectors.load_word2vec_format('organic_glove_300d.txt') # To load text file
w2v = model.wv

#To plot
def tsne_df(w2v):
    "Creates and TSNE model and plots it"
    labels = []
    tokens = []

    words = list(w2v.index_to_key) #instead of model.wv.vocab
    len(words)
    words_cut=words[:1000] #To test time on a smaller df
    
    for word in words: #Test with 'words_cut' first
        tokens.append(w2v[word]) #Fetches the vector for the word. same as model.wv
        # print(model.wv[word])        
        labels.append(word) #Word
        # print(word)

    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(tokens) #fit_transform on vectors

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])
        
    tsne_df = pd.DataFrame(
    {'word': labels,
     'x': x,
     'y': y
    })
        
    # tsne_df.to_pickle("tsne_300d_w2_df.pkl", protocol=0)
    print("Time taken for TSNE: ",str(datetime.now()-start)[:-5])
    
print("Creating new TSNE dataframe..") 
start=datetime.now()
tsne_df(w2v) 

# test_tsne_df=pd.read_pickle('tsne_100d_w5_df.pkl')
# test_tsne_df
# In[12]: Addition logic word2vec
# w2v = KeyedVectors.load_word2vec_format('organic_glove_300d.txt') # To load text file
# w2v.most_similar(positive=['king','woman'], negative= ['man'])
print(w2v.most_similar(positive=['oil','essential','carrier']))

# w2v2 = KeyedVectors.load_word2vec_format('organic_glove_300d_w5.txt') # To load text file
# w2v2.most_similar(positive=['king','food'], negative= ['man'])
print("\n", w2v2.most_similar(positive=['oil','essential','carrier']))
 


# In[12]: w2v Scatter plot logic including hue.
# import plotly as
# Toggle text dislay on and off. 
# Bold word name in hover text

import plotly.io as pio #To plot in browser
pio.renderers.default='browser'


vocab_to_plot = ['vanilla','organic','sustainable','cacao']
tsne_df_full=pd.read_pickle('tsne_100d_w5_df.pkl')
w2v = KeyedVectors.load_word2vec_format('organic_glove_300d.txt') 
nearest_size = 5  #make input for nearestneighbor size

#index of input words

tsne_df_full['type'] = 'Sel' 
ip_vocab_index=list(tsne_df_full[tsne_df_full['word'].isin(vocab_to_plot)].index.values)  # 

#index of closest 10 words
clos_ten_out = []
for word in vocab_to_plot: 
    a=w2v.most_similar(word, topn = nearest_size)
    clos_ten_in = [i[0] for i in a]
    clos_ten_out.append(clos_ten_in)

clos_ten_flat = [item for sublist in clos_ten_out for item in sublist]
clos_ten_index = list(tsne_df_full[tsne_df_full['word'].isin(clos_ten_flat)].index.values)  # 

plot_index = ip_vocab_index + clos_ten_index
indices = [0,1,3,6,10,15]
tsne_df_full.loc[ip_vocab_index,'type'] = 'ip'
tsne_df_full.loc[clos_ten_index,'type'] = 'near'

tsne_plot_df = tsne_df_full.loc[plot_index]
tsne_plot_df
# tsne_plot_df = tsne_df_full[tsne_df_full['word'].isin(vocab_to_plot)] #filter to display

fig3 = px.scatter(tsne_plot_df, x="x", y="y", text="word",color='type', log_x=False, size_max=60,
                  # trendline = "rolling",
                  hover_name = 'word'
                  )
fig3.update_traces(textposition='top center')
fig3.show()    

#Make the text justify options to make viewing better in app
# In[13]: co-occurrence visualisations
#instantiate the sorted vocab in the co-occ code block
#
from numpy import load 
co_occ_arr = load('ad5_co_occ_arr_w5_bi.npy') #shape 41526
# clf_table=pd.read_pickle("df_8hrs.pkl") #str col for rm_sw_lemt. but not unique words
clf_table = pd.read_pickle("App_dataframe_5.pkl") #list col rm_sw_lemt

# flat_list = [item for row_str in clf_table['rm_sw_lemt'] for item in row_str.split()] #str col
flat_list = [item for row_list in clf_table['rm_sw_lemt'] for item in row_list] #list col
len(flat_list) #total words in the corpus 799068 (df_8hr), 813922(App_dataframe_5)
len(set(flat_list)) #41526 (AD_5)

vocab = sorted(set(flat_list))

co_occ_df = pd.DataFrame(data = co_occ_arr, 
                        index = vocab, 
                        columns = vocab)

co_occ_df
# co_occ_df.to_pickle("co_occ_df.pkl", protocol=0) #hangs the kernel

#datastructure for bar chart
type(co_occ_df['organic'])
word = 'textile'
co_occ_plot = co_occ_df[word].sort_values(ascending=False).head(30)
co_occ_plot = co_occ_plot.reset_index()
mapping = {co_occ_plot.columns[0]:'word', co_occ_plot.columns[1]: 'counts'}
co_occ_plot = co_occ_plot.rename(columns=mapping)

co_occ_plot


# In[12]:
##Dashboard application Test shift to dbc.Container
print("\nStarting dashboard app...")

##############Import libraries and dataset
import pandas as pd     #(version 1.0.0)
import plotly           #(version 4.5.4) pip install plotly==4.5.4
import plotly.express as px

import dash             #(version 1.9.1) pip install dash==1.9.1
import dash_table
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

import dash_bootstrap_components as dbc

# import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# import unicodedata
# from langdetect import detect
from wordcloud import WordCloud, STOPWORDS
from nltk.corpus import stopwords
# import warnings
# warnings.filterwarnings('ignore')
# import os
# import os.path 
from datetime import datetime
from numpy import load

import os
import re
import time
from gensim.models import Word2Vec
from gensim.models import KeyedVectors #For loading the model

from scipy import stats # for zscore
import numpy as np #with zscore

from collections import Counter

import ast

############
app_launch_start=datetime.now() #Set start time for program start

# dff=df_updated[view_extracts]



print("Loading the dataframe..")
start = datetime.now()


#Other supporting dataframes
# coocc_svd_matrix_2 = load('svd_arpack_w2.npy') 
# coocc_svd_matrix_3 = load('svd_arpack_w3.npy') 
# coocc_svd_matrix_4 = load('svd_arpack_w4.npy') 

# vocab_words_df_2 = pd.read_pickle('vocab_words_svd_w2.pkl')    
# vocab_words_df_3 = pd.read_pickle('vocab_words_svd_w3.pkl')      
# vocab_words_df_4 = pd.read_pickle('vocab_words_svd_w4.pkl')  

# tsne_100d_w5_df = pd.read_pickle('tsne_100d_w5_df.pkl')
# tsne_200d_w5_df = pd.read_pickle('tsne_200d_w5_df.pkl')
tsne_300d_w5_df = pd.read_pickle('tsne_300d_w5_df.pkl')

# tsne_300d_w4_df = pd.read_pickle('tsne_300d_w4_df.pkl')
# tsne_300d_w3_df = pd.read_pickle('tsne_300d_w3_df.pkl')
# tsne_300d_w2_df = pd.read_pickle('tsne_300d_w2_df.pkl')


#try and except for quick reloading of the w2v object.
# global w2v
try:
    w2v
except:
    w2v = KeyedVectors.load_word2vec_format('organic_glove_300d_w5.txt') #initialise only once per app run

#Dashtables
# df_updated=pd.read_pickle('App_dataframe_5.pkl') #duplicates removed
df_updated = pd.read_pickle("df_8hrs.pkl") 
disp1=['ix','caption_processed_4','hashtags','cap_mentions','web_links'] #select columns to display in the dashtable
df_disp_1 = df_updated[disp1]
df_disp_1.rename(columns={"caption_processed_4": "description"},inplace=True)

clf_table=pd.read_pickle("df_8hrs.pkl") #outlier table
clf_table['Inf_Non']='' #new entry columns
clf_table['Domain']='' #new entry columns

clf_view_6=['ix','time_delta_hours','likeCount','commentCount','likes_pr_min','caption_processed_4','Inf_Non', 'Domain','hashtags','username','fullName']
clf_dashtable=clf_table[clf_view_6]
clf_dashtable.rename(columns={"caption_processed_4": "description"},inplace=True)


#------get vocab & initialise co-occurence dataframe
co_occ_table = pd.read_pickle("App_dataframe_5.pkl") #list col rm_sw_lemt
flat_list_co_occ = [item for row_list in co_occ_table['rm_sw_lemt'] for item in row_list] #list col
vocab = sorted(set(flat_list_co_occ))

co_occ_arr = load('ad5_co_occ_arr_w5_bi.npy') #shape 41526
co_occ_df = pd.DataFrame(data = co_occ_arr, 
                        index = vocab, 
                        columns = vocab)
#------


print("Dataframes loaded")
t=datetime.now() - start 
s=str(t) 
print("Execution time: ", s[:-5], "\n\n")

############




# app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP]) #needs internet
# app = dash.Dash(__name__, assets_url_path ='/assets/bootstrap.css') #loads any CSS directly added in the "assets" folder
app = dash.Dash(__name__) #loads any CSS directly added in the "assets" folder

#the rows and columns from dbc worl only with an external stylesheet

al=str(datetime.now() - app_launch_start) #start time logged at start of program code
print("Total time taken to launch app", al[:-5])

#---------------------------------------------------------------
#Initialising variables outside the callbacks

#for dropdowns
col_sels = ['description','hashtags','cap_mentions','web_links']   #values for dropdown_1
col_sels_2 = ['description','hashtags','web_links','cap_mentions','username','fullName']   #values for dropdown

input_boxs = ['text','text']
input_boxes1 = ('text','text','text')

#Default vocab to plot
certs = ['gots','oeko','cosmos','ecocert','fsc'] #certifications
fruits = ['apple', 'orange', 'vanilla','cacao']
consc = ['sustainably', 'ethically']
rel_concs = ['fruits', 'certification','textile', 'cosmetics']
vocab_plot_list = certs + fruits + rel_concs

#other arbitrary initialisations
z_init = 5 
z_init2 = 5 
wc_init = 0 

#---------------------------------------------------------------

app.layout = dbc.Container([    
################################### ROW1-Instagram Data, Sel-Desel button_1,Wordcloud ########################### 
    
    dbc.Row([
        dbc.Col(html.H2("Instagram DataTable"), width={'size':3}),
        dbc.Col(
            # html.Button(id='sel-button', n_clicks=0, children="Sel_all"),
            dbc.Button(id='sel-button', n_clicks=0, children="Sel_all", className="mt-5 mr-2"),            
            width={'size': 0.5}, style={'textAlign': "left"}), #another way to align
        dbc.Col(
            dbc.Button(id='desel-button', n_clicks=0, children="Des_all", className="mt-5"),
            width={'size': 0.5}, style={'textAlign':"left"}),
        
        dbc.Col(html.H2("Wordcloud - Word frequency"), width={'size':5, 'offset':2}, style={'textAlign' : "center"}),         
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
    
################################### ROW3-Dashtable & Wordcloud ###########################    

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
            ],no_gutters=False),
    

################################### ROW4-Headers 2  ###########################
    dbc.Row([
            dbc.Col(html.H2("Vector exploration"), width={'size':2}),
            
            
            dbc.Col(
                dbc.Button(id='add1_button', n_clicks=0, children="Add", className="mt-5 mr-2"),            
                width={'size': 0.5}, style={'textAlign': "left"}),
        
            dbc.Col(
                dbc.Button(id='rem1_button', n_clicks=0, children="Rem", className="mt-5 mr-2"),            
                width={'size': 0.5}, style={'textAlign': "left"}),
        
            dbc.Col(
                dbc.Button(id='clr1_button', n_clicks=0, children="Clr", className="mt-5 mr-2"),            
                width={'size': 0.5}, style={'textAlign': "left"}),
        
            dbc.Col(
                dbc.Button(id='plt1_button', n_clicks=0, children="Plot", className="mt-5 mr-2"),            
                width={'size': 0.5}, style={'textAlign': "left"}),
            
            dbc.Col(
                dbc.Button(id='words_tog', n_clicks=0, children="words", className="mt-5 mr-2"),            
                width={'size': 1, 'offset':1}, style={'textAlign': "left"}), #size 0.5
            
            
            dbc.Col(html.H2("Vector math"), width={'size':5}, style={'textAlign': "center"}), #If the width of col breaks, then width parameter is ignored 
           
            
            ], no_gutters=False),

 
################################### ROW5-Input_boxs ###########################
    dbc.Row([
        dbc.Col(
            dcc.Input(
            id='input_1',
            type='text',
            placeholder="",          #pass list vocab to display here   
            debounce=False,           # changes to input are sent to Dash server only on enter or losing focus
            pattern=r"^[A-Za-z].*",  # Regex: string must start with letters only
            spellCheck=True,
            inputMode='latin',       # provides a hint to browser on type of data that might be entered by the user.
            name='text',             # the name of the control, which is submitted with the form data
            list='browser',          # identifies a list of pre-defined options to suggest to the user
            n_submit=0,              # number of times the Enter key was pressed while the input had focus
            n_submit_timestamp=-1,   # last time that Enter was pressed
            autoFocus=False,          # the element should be automatically focused after the page loaded
            n_blur=0,                # number of times the input lost focus
            n_blur_timestamp=-1,     # last time the input lost focus.
            size="30"                # Width of box which can display placeholders
                ), width={'size': 4}, style={'textAlign': "left"}   #size was 6
            ),
            
        

        #---------------Word2vec_2 math buttons-----------
        
        dbc.Col(
            dcc.Input(
                id='input_2',
                type='text',
                placeholder='',  # A hint to the user of what can be entered in the control
                debounce=True,                      # Changes to input are sent to Dash server only on enter or losing focus
                min=2015, max=2019, step=1,         # Ranges of numeric value. Step refers to increments
                minLength=0, maxLength=50,          # Ranges for character length inside input box
                autoComplete='on',
                disabled=False,                     # Disable input box
                readOnly=False,                     # Make input box read only
                required=False,                     # Require user to insert something into input box
                size="7",                          # Sets width of box. If exceeds col width then overrides it
                ), width={'size': 1, 'offset' : 2}, style={'textAlign': "center"}  #text size 5 corresponds to width size 0.75
            ),
        
        dbc.Col(html.H3("-"), width={'size':0.5}, style={'textAlign' : "center"}), #total width to be 0.6. Balance with offset. textAlign:center not working
        
        dbc.Col(
            dcc.Input(
                id='input_3',
                type='text',
                placeholder='',  # A hint to the user of what can be entered in the control
                debounce=True,                      # Changes to input are sent to Dash server only on enter or losing focus
                min=2015, max=2019, step=1,         # Ranges of numeric value. Step refers to increments
                minLength=0, maxLength=50,          # Ranges for character length inside input box
                autoComplete='on',
                disabled=False,                     # Disable input box
                readOnly=False,                     # Make input box read only
                required=False,                     # Require user to insert something into input box
                size="7",                          # Number of characters that will be visible inside box
                ), width={'size': 1}, style={'textAlign': "center"} 
            ),
        
        dbc.Col(html.H3("+"),width={'size': 0.5}, style={'textAlign' : "center"}),

        dbc.Col(
            dcc.Input(
                id='input_4',
                type='text',
                placeholder='',  # A hint to the user of what can be entered in the control
                debounce=True,                      # Changes to input are sent to Dash server only on enter or losing focus
                min=2015, max=2019, step=1,         # Ranges of numeric value. Step refers to increments
                minLength=0, maxLength=50,          # Ranges for character length inside input box
                autoComplete='on',
                disabled=False,                     # Disable input box
                readOnly=False,                     # Make input box read only
                required=False,                     # Require user to insert something into input box
                size="7",                          # Number of characters that will be visible inside box
                ), width={'size': 1}, style={'textAlign': "center"} 
            ),
        
        dbc.Col(html.H3("="),width={'size':0.5}, style={'textAlign' : "center"}),
        
        dbc.Col(
            dcc.Input(
                id='output_1',
                type='text',
                placeholder='',  # A hint to the user of what can be entered in the control
                debounce=True,                      # Changes to input are sent to Dash server only on enter or losing focus
                min=2015, max=2019, step=1,         # Ranges of numeric value. Step refers to increments
                minLength=0, maxLength=50,          # Ranges for character length inside input box
                autoComplete='on',
                disabled=True,                     # Disable input box
                readOnly=False,                     # Make input box read only
                required=False,                     # Require user to insert something into input box
                size="7",                          # Number of characters that will be visible inside box
                ), width={'size': 1}, style={'textAlign': "center"} 
            ),
        ],no_gutters=False),


################################### ROW6-word2vec_1  ###########################
    dbc.Row([
        dbc.Col(
            # dcc.Graph(id='svd_1', figure={}, config={'displayModeBar': True},
            #            style={'width':'600px' ,'height':'350px'}),
            dcc.Graph(id='word2vec_1'), 
            width={'size': 6}
                ),

################################### ROW6-word2vec_2  ###########################
        dbc.Col(
            dcc.Graph(id='word2vec_2'), 
            width={'size': 6}
                )
            ],no_gutters=False),
################################### ROW7-range slider  ###########################    
    dbc.Row([
        dbc.Col(
            dcc.Slider( 
                id='my_slider',
                min=0,
                max=20,
                step=1,
                value=0,
                marks={
                    0: {'label': '0', 'style': {'color': '#77b0b1'}},
                    5: {'label': '5'},
                    10: {'label': '10'},
                    15: {'label': '15', 'style': {'color': '#f50'}},
                    20: {'label': '20'}
                    }),                 
            width={'size': 6}),
        
        dbc.Col(
            dcc.Slider( 
                id='my_slider_2',
                min=0,
                max=20,
                step=1,
                value=0, #default value
                marks={
                    0: {'label': '0', 'style': {'color': '#77b0b1'}},
                    5: {'label': '5'},
                    10: {'label': '10'},
                    15: {'label': '15', 'style': {'color': '#f50'}},
                    20: {'label': '20'}
                    }),                 
            width={'size': 6})
        
        

        # dbc.Col(
        #     dcc.Graph(id='word2vec_2'), 
        #     width={'size': 6}
        #         )
            ],no_gutters=False),
    
############################ ROW8-"Classification Dashtable", Sel-Desel #######################

    dbc.Row([
        dbc.Col(html.H2("Z-Score adjustable DataTable"), width={'size':2}),
        
        dbc.Col(
            dbc.Button(id='sel-button_2', n_clicks=0, children="Sel_all", className="mt-5 mr-2"),            
                    width={'size': 0.5}, style={'textAlign': "left"}), 
        
        dbc.Col(
            dbc.Button(id='desel-button_2', n_clicks=0, children="Des_all", className="mt-5"),
            width={'size': 0.5}, style={'textAlign':"left"}),
        
        dbc.Col(html.H2("Wordcloud - Post frequency"), width={'size':5, 'offset':3}, style={'textAlign' : "center"}),       
        
    
    
            ],no_gutters=False),

############################ ROW9-Dropdown_2 ##############################


    dbc.Row([
        dbc.Col(
            dcc.Dropdown(id='my-dropdown_2', multi=True,
                      options=[{'label': x, 'value': x} for x in col_sels_2],
                      value=["description"], #initial values to pass
                      style={'width':'490px'}
                        ), width={ 'offset':0}),
        
        dbc.Col(
            dcc.Slider( 
                id='my_slider_zscore',
                min=0,
                max=4,
                step=0.25,
                value=0,
                marks={
                    0: {'label': '0', 'style': {'color': '#77b0b1'}},
                    1: {'label': '1'},
                    2: {'label': '2'},
                    3: {'label': '3', 'style': {'color': '#f50'}},
                    4: {'label': '4'}
                    }),                 
            width={'size': 2}),
        
        
            ], no_gutters = False),    
    
    
    

############################ ROW10-Clasification dashtable, WordCloud_2 ##############################
    dbc.Row([    
            dbc.Col(
                dash_table.DataTable(
                    id='datatable_2',
                    data=clf_dashtable.to_dict('records'),
                    columns=[
                        {"name": i, "id": i, "deletable": False, "selectable": True} for i in clf_dashtable.columns
                    ],
                    editable=True,
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
                        'height': '600px', #was 300px
                        'width': '690px',
                        'overflowY': 'auto'
                    },
                    
                    style_cell_conditional=[
    
                        {'if': {'column_id': 'description'},
                         'width': '200px', 'textAlign': 'left'},
                    ],
                    export_format='xlsx',
                    
                ),            
                width={'size': 6}),   
            
             dbc.Col(
                dcc.Graph(id='wordcloud_2', figure={}, config={'displayModeBar': True},
                           style={'width':'600px' ,'height':'600px'}), #ht was 350px
                    )
            #,width={'size': 6},  width = {'size': 5, 'offset':1}
                ],no_gutters=False),


############################ ROW11-Header "Posts count of unique words"  ##############################    

    dbc.Row([
        dbc.Col(html.H2("Posts count of distinct words"), width={'size':2}),
         
        dbc.Col(html.H2("Co-occurrence counts"), width={'size':2, 'offset':4}),
            ],no_gutters=False),

############################ ROW12-Slider for most common ##############################    
    dbc.Row([
        dbc.Col(
            dcc.Slider( 
                id='my_slider_mc',
                min=10,
                max=30,
                step=1,
                value=10,
                marks={
                    10: {'label': '10', 'style': {'color': '#77b0b1'}},
                    20: {'label': '20'},
                    30: {'label': '30'},
                    40: {'label': '40', 'style': {'color': '#f50'}},
                    }),                 
            width={'size': 2}),
       
        dbc.Col(
            dcc.Input(
                id='co_occ_ip',
                type='text',
                placeholder='',  # A hint to the user of what can be entered in the control
                debounce=True,                      # Changes to input are sent to Dash server only on enter or losing focus
                minLength=0, maxLength=50,          # Ranges for character length inside input box
                autoComplete='on',
                disabled=False,                     # Disable input box
                readOnly=False,                     # Make input box read only
                required=False,                     # Require user to insert something into input box
                size="7",                          # Sets width of box. If exceeds col width then overrides it
                ), width={'size': 2, 'offset' : 4}, style={'textAlign': "center"}),  #text size 5 corresponds to width size 0.75
        dbc.Col(
            dcc.Slider( 
                id='slider_co_occ',
                min=10,
                max=40,
                step=1,
                value=10,
                marks={
                    10: {'label': '10', 'style': {'color': '#77b0b1'}},
                    20: {'label': '20'},
                    30: {'label': '30'},
                    40: {'label': '40', 'style': {'color': '#f50'}},
                    }),                 
            width={'size': 2})
        
        
        
            ], no_gutters=False),  

############################ ROW13-Bar_chart for distinct word count ##############################    
    dbc.Row([
        dbc.Col(
            dcc.Graph(id='bar_chart_1'), 
            width={'size': 6}
                ),
        dbc.Col(
            dcc.Graph(id='bar_chart_2'), 
            width={'size': 6}
                ),
        
            ],no_gutters=False),
        ],fluid=True) #closes initial dbc container



################################ App Callbacks #########################################################
         ############           #CALLBACKS         ############
################################ App Callbacks #########################################################

################################ Barcharts co-occ counts################################

@app.callback(
    Output(component_id='bar_chart_2', component_property='figure'),
    [Input(component_id='co_occ_ip', component_property='value'),
    Input(component_id='slider_co_occ', component_property='value')],
    prevent_initial_call=True,
)


def barchart_2(co_occ_ip,slider):
    
    word = co_occ_ip
    head_values = slider
    
    try:
        co_occ_plot = co_occ_df[word].sort_values(ascending=False).head(head_values)
        co_occ_plot = co_occ_plot.reset_index()

    except KeyError as e:
        print("\n word not in vocab: ", e)

    try:
        mapping = {co_occ_plot.columns[0]:'word', co_occ_plot.columns[1]: 'counts'}
        co_occ_plot = co_occ_plot.rename(columns=mapping)
        fig_br2 = px.bar(co_occ_plot, x="counts", y="word", title="Count of co-occurences", orientation = 'h')
        fig_br2.update_layout(yaxis=dict(autorange="reversed"))
        return fig_br2

    except UnboundLocalError as e:
        print(e)
        raise dash.exceptions.PreventUpdate

    

################################ Barcharts distinct words per post ###############


@app.callback(
    Output(component_id='bar_chart_1', component_property='figure'),
    [Input(component_id='datatable_2', component_property='selected_rows'),
    Input(component_id='my-dropdown_2', component_property='value'),
    Input(component_id='my_slider_zscore', component_property='value'),
    Input(component_id='my_slider_mc', component_property='value')],
    prevent_initial_call=False,
)

def barchart_1(chosen_rows, chosen_cols, zscore, mostcomm):
    global z_init2, flat_list, fig_br1 
    ctx = dash.callback_context
    trigger = (ctx.triggered[0]['prop_id'].split('.')[0])
    
    z = np.abs(stats.zscore(clf_dashtable[['time_delta_hours','likes_pr_min']])) #assign z-score to bivariate   
    view_rm_lm= ['rm_sw_lemt']
    view_rm_lm = clf_view_6 + view_rm_lm
    clf_dashtable_z=clf_table[view_rm_lm].iloc[np.unique(np.where(z > zscore)[0])] #table should have rm_sw_lm  
   
    #clean up index for "select_all" functionality
    clf_dashtable_z.drop(['ix'], axis=1, inplace = True) #drop any older 'ix' columns
    clf_dashtable_z.reset_index(inplace= True,drop=True) #reset and drop old index
    clf_dashtable_z.reset_index(inplace= True) #create new ix column
    clf_dashtable_z.rename(columns={"index": "ix"},inplace=True) 

    if len(chosen_cols) > 0: #consider rm_sw_lemt instead
        if 'description' in chosen_cols:
            chosen_cols.remove('description')
            chosen_cols.append('rm_sw_lemt')
            print("Chosen cols now contain",chosen_cols)#atleast 1 col to be selected
            
        if len(chosen_rows)==0: #if no rows selected consider all rows of z score filtered table-defined again                   
            
            #-----fast render optimization
            if ((z_init2 == zscore) and (trigger not in ['my_slider_mc','my-dropdown_2'])): #if no change in the score return same object. random init at app start.
                print("\nz_init2 is equal to zscore, z_init2 value is: ", z_init2)
                return (fig_br1)
            
            if ((z_init2 == zscore) and (trigger in ['my_slider_mc'])): #uses flat_list global for most common change. 
                counts = dict(Counter(flat_list).most_common(mostcomm))
                labels, values = zip(*counts.items())
                indSort = np.argsort(values)[::-1] # sort your values in descending order
                labels = np.array(labels)[indSort]
                values = np.array(values)[indSort]
                indexes = np.arange(len(labels))
                
                word_count_df = pd.DataFrame({'word': labels, 'posts_count': values}, columns=['word', 'posts_count'])  
                
                fig_br_12 = px.bar(word_count_df, x="posts_count", y="word", title="Lemmatised words by post count", orientation = 'h')
                fig_br_12.update_layout(yaxis=dict(autorange="reversed"))
                
                return (fig_br_12)
                
            else: #initialise object and set z_init value
                print("\nElse block reached, z_init2 value is; ", z_init2)
                z_init2 = zscore
                print("z_init2 value updated to: ", z_init2) 
            #-----
                
                mc= mostcomm #set state to track change
                df_filtered = clf_dashtable_z[chosen_cols]
    
                #Combine multiple columns into a single series for wordcloud generation
                df_filtered['comb_cols'] = df_filtered[df_filtered.columns[0:]].apply( 
                    lambda x: ' '.join(x.dropna().astype(str)),        
                    axis=1)
                
                #Use only unique words per post in wordcloud
                df_filtered['comb_cols'] = df_filtered['comb_cols'].apply(
                    lambda x: ' '.join(set(x.split())))
                
                #Create word counts dataframe
                flat_list = [item for row_list in df_filtered['comb_cols'] for item in row_list.split()]
                counts = dict(Counter(flat_list).most_common(mostcomm))
                labels, values = zip(*counts.items())
                indSort = np.argsort(values)[::-1] # sort your values in descending order
                labels = np.array(labels)[indSort]
                values = np.array(values)[indSort]
                indexes = np.arange(len(labels))
                
                word_count_df = pd.DataFrame({'word': labels, 'posts_count': values}, columns=['word', 'posts_count'])  
                
                fig_br1 = px.bar(word_count_df, x="posts_count", y="word", title="Lemmatised words by post count", orientation = 'h')
                fig_br1.update_layout(yaxis=dict(autorange="reversed"))
                
                return (fig_br1)

            

        elif len(chosen_rows) > 0 :
            df_filtered = clf_dashtable_z[chosen_cols]
            df_filtered = df_filtered[df_filtered.index.isin(chosen_rows)]

            
    elif len(chosen_cols) == 0:
        raise dash.exceptions.PreventUpdate

    ##figure logic repeated for row selections
    #Combine multiple columns into a single series for wordcloud generation
    df_filtered['comb_cols'] = df_filtered[df_filtered.columns[0:]].apply( 
        lambda x: ' '.join(x.dropna().astype(str)),        
        axis=1)
    
    #Use only unique words per post in wordcloud
    df_filtered['comb_cols'] = df_filtered['comb_cols'].apply(
        lambda x: ' '.join(set(x.split())))
    
    #Create word counts dataframe
    flat_list = [item for row_list in df_filtered['comb_cols'] for item in row_list.split()]
    counts = dict(Counter(flat_list).most_common(mostcomm))
    labels, values = zip(*counts.items())
    indSort = np.argsort(values)[::-1] # sort your values in descending order
    labels = np.array(labels)[indSort]
    values = np.array(values)[indSort]
    indexes = np.arange(len(labels))
    
    word_count_df = pd.DataFrame({'word': labels, 'posts_count': values}, columns=['word', 'posts_count'])  
    
    fig_br_1 = px.bar(word_count_df, x="posts_count", y="word", title="Lemmatised words by post count", orientation = 'h')
    fig_br_1.update_layout(yaxis=dict(autorange="reversed"))
    return (fig_br_1) #note change in return object name





################################ Column highlighting Dashtable2 ################################

@app.callback(
    Output('datatable_2', 'style_data_conditional'),
    Input(component_id='my-dropdown_2', component_property='value')
)

def update_styles(selected_columns):
    return [{
        'if': { 'column_id': i },
        'background_color': '#D2F3FF'
    } for i in selected_columns]


################################ Dashtable_2 Zscore virtual table################################
@app.callback(
    Output('datatable_2', 'data'), #virtual_data is displayed table. Even after filtering. #references ordered 0 to n index of larger table irrespective of actual index value.
    Input('my_slider_zscore','value'),
    prevent_initial_call=False
)

def zscore_outliers(zscore):
    z = np.abs(stats.zscore(clf_dashtable[['time_delta_hours','likes_pr_min']])) #assign z-score to bivariate 
    clf_dashtable_z=clf_dashtable.iloc[np.unique(np.where(z > zscore)[0])] #create outlier table
    
    #clean up index for "select_all" functionality
    clf_dashtable_z.drop(['ix'], axis=1, inplace = True) #drop any older 'ix' columns
    clf_dashtable_z.reset_index(inplace= True,drop=True) #reset and drop old index
    clf_dashtable_z.reset_index(inplace= True) #create new ix column
    clf_dashtable_z.rename(columns={"index": "ix"},inplace=True) 
    print("\nZscore slider value is: ", zscore)
    print("No of rows are: ", len(clf_dashtable_z))    
    return clf_dashtable_z.to_dict('records')




################################ Dashtable_2 Sel_Desel button_2 ################################

@app.callback(
    [Output('datatable_2', 'selected_rows')], #references ordered 0 to n index of larger table irrespective of actual index value.
    [Input('sel-button_2', 'n_clicks'),
    Input('desel-button_2', 'n_clicks'),
    Input('my_slider_zscore','value')],
    [State('datatable_2', 'derived_virtual_selected_rows'), #virtual selected row is what is selected.
     State('datatable_2', 'derived_virtual_data')], #virtual_data is displayed table. Even after filtering. 
    prevent_initial_call=True
)

def select_deselect(selbtn, deselbtn, z_score_trigger,selected_rows,filtered_table):
    ctx = dash.callback_context
    if ctx.triggered:
        print(ctx.triggered)
        trigger = (ctx.triggered[0]['prop_id'].split('.')[0])
        if trigger == 'sel-button_2':
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

################################ Wordcloud_2  ################################

@app.callback(
    Output('wordcloud_2','figure'),
    [Input(component_id='datatable_2',component_property='selected_rows'),
    Input(component_id='my-dropdown_2', component_property='value'),
    Input('my_slider_zscore','value')],
    prevent_initial_call=False
)

# chosen_cols = 'description'
def ren_wordcloud(chosen_rows, chosen_cols, zscore):
    global z_init, fig_wordcloud2 #for try and except. otherwise destroyed at end of callback.
    ctx = dash.callback_context
    trigger = (ctx.triggered[0]['prop_id'].split('.')[0])
    
    z = np.abs(stats.zscore(clf_dashtable[['time_delta_hours','likes_pr_min']])) #assign z-score to bivariate 
    clf_dashtable_z=clf_dashtable.iloc[np.unique(np.where(z > zscore)[0])] #create outlier table    
    #clean up index for "select_all" functionality
    clf_dashtable_z.drop(['ix'], axis=1, inplace = True) #drop any older 'ix' columns
    clf_dashtable_z.reset_index(inplace= True,drop=True) #reset and drop old index
    clf_dashtable_z.reset_index(inplace= True) #create new ix column
    clf_dashtable_z.rename(columns={"index": "ix"},inplace=True) 
    
    # try:
    #     z_init
    # except:
    #     z_init = 5 #arbitrary initiatialisation
    
    if len(chosen_cols) > 0: #atleast 1 col to be selected
        if len(chosen_rows)==0:       
            #if zscore changed then initilaise the wordcloud to updated df
            
            #-----fast render optimization
            if ((z_init == zscore) and (trigger != 'my-dropdown_2')): #if no change in the score return same object. random init at app start.
                print("\nz_init is equal to zscore, z_init value is: ", z_init)
                return fig_wordcloud2
                
            else: #initialise object and set z_init value
                print("\nElse block reached, z_init value is; ", z_init)
                z_init=zscore
                print("z_init value updated to: ", z_init) 
            #-----
            
                #if no rows selected consider all rows of z score filtered table-defined again            
                df_filtered = clf_dashtable_z[chosen_cols]
    
                print("Full Word cloud not initialised")
                print("Initialising full word cloud..")
               
                df_filtered['comb_cols'] = df_filtered[df_filtered.columns[0:]].apply( #Combine multiple columns into a single series for wordcloud generation
                    lambda x: ' '.join(x.dropna().astype(str)),        
                    axis=1)
                
                #Use only unique words per post in wordcloud
                df_filtered['comb_cols'] = df_filtered['comb_cols'].apply(
                    lambda x: ' '.join(set(x.split())))
                
                en_stopwords = stopwords.words('english')
                fr_stopwords = stopwords.words('french')
                web_links_sw = ['www','http','https','com']
                
                combined_stopwords = en_stopwords + fr_stopwords + web_links_sw
                
                print("\nNo of rows in WC are: ",len(df_filtered))
                wordcloud = WordCloud(max_words=100,    
                                      # stopwords= combined_stopwords,
                                       # collocations=False,
                                      # color_func=lambda *args, **kwargs: "orange",
                                      colormap='tab20b',
                                      background_color='white',
                                      width=1200, #1200,1700    
                                      height=1000, #1000
                                      random_state=1).generate_from_text(' '.join(df_filtered['comb_cols'])) #df_filtered has to be a series
            
                # print(' '.join(df_filtered['comb_cols']))
            
                fig_wordcloud2 = px.imshow(wordcloud, template='ggplot2',
                                          ) #title="Post frequency"
            
                fig_wordcloud2.update_layout(margin=dict(l=0, r=0,b=0,t=0))
                fig_wordcloud2.update_xaxes(visible=False)
                fig_wordcloud2.update_yaxes(visible=False)
                
                
                return fig_wordcloud2

            

        elif len(chosen_rows) > 0 :
            # df_filtered = df_disp_1.iloc[chosen_rows,[chosen_cols]] #filter by selected rows
            df_filtered = clf_dashtable_z[chosen_cols]
            df_filtered = df_filtered[df_filtered.index.isin(chosen_rows)]
            #print("Atleast 1 col chosen and multiple rows selected type. Dataype of df_filtered is", type(df_filtered))
            
    elif len(chosen_cols) == 0:
        raise dash.exceptions.PreventUpdate

## load wordcloud only once
    

    df_filtered['comb_cols'] = df_filtered[df_filtered.columns[0:]].apply( #Combine multiple columns into a single series for wordcloud generation
        lambda x: ' '.join(x.dropna().astype(str)),        
        axis=1)
    
    #Use only unique words per post in wordcloud
    df_filtered['comb_cols'] = df_filtered['comb_cols'].apply(
        lambda x: ' '.join(set(x.split()))) #because changes up the order, bigrams become irrelevant

    en_stopwords = stopwords.words('english')
    fr_stopwords = stopwords.words('french')
    web_links_sw = ['www','http','https','com']
    
    combined_stopwords = en_stopwords + fr_stopwords + web_links_sw
    
    print("\nNo of rows in WC are: ",len(df_filtered))
    wordcloud = WordCloud(max_words=100,    
                          # stopwords= combined_stopwords,
                          # collocations=False,
                          # color_func=lambda *args, **kwargs: "orange",
                          colormap='tab20b',
                          background_color='white',
                          width=1200, #1200,1700    
                          height=1000, #1000
                          random_state=1).generate_from_text(' '.join(df_filtered['comb_cols'])) #df_filtered has to be a series

    # print(' '.join(df_filtered['comb_cols']))

    fig_wordcloud_2 = px.imshow(wordcloud, template='ggplot2',
                              ) #title="Post frequency" 

    fig_wordcloud_2.update_layout(margin=dict(l=0, r=0,b=0,t=0))
    fig_wordcloud_2.update_xaxes(visible=False)
    fig_wordcloud_2.update_yaxes(visible=False)

    return fig_wordcloud_2





################################ word2vec_2 math graph ################################
@app.callback(
    [Output(component_id='word2vec_2', component_property='figure'),
    Output(component_id='output_1', component_property='placeholder')],
    # [Input('vector_dd_2','value'),
     # Input('wind_dd_1','value'),
     [Input('input_2','value'),
      Input('input_3','value'),
      Input('input_4','value'),
      Input('my_slider_2','value'),
      Input('words_tog','n_clicks')
      ],
    # State('datatable_id', 'derived_virtual_data'),
    # Input(component_id='my-dropdown', component_property='value')],
    prevent_initial_call=False
    )
    
def word2vec_math(ip1,ip2,ip3,slider2_val,word_tog): 
    tsne_df_full = tsne_300d_w5_df 
    # tsne_df_full.index
    vocab_plot_list_2 = [ip1,ip2,ip3]
    print("\n\nvocab_plot_list is: ", vocab_plot_list_2)
    # vocab_to_plot = vocab_plot_list_2
    vocab_to_plot = list(filter(None, vocab_plot_list_2))
    
    print("Filtered vocab_to_plot is: ", vocab_to_plot) #Only non-empty inputs
   
    print("Slider2 val: ",slider2_val)

    nearest_size = slider2_val #nearest size defined
    
    tsne_df_full['type'] = 'Sel' 
    ip_vocab_index=list(tsne_df_full[tsne_df_full['word'].isin(vocab_to_plot)].index.values)  
    
    # print("vocab to plot of slider graph is: ",vocab_to_plot )
 
    clos_ten_out = [] #index of closest 10 words
    try:           
        for i,word in enumerate(vocab_to_plot):
            ix_word = tsne_df_full[tsne_df_full['word'].isin([word])].index.values
            tsne_df_full.loc[ix_word,'type'] = "ip_{}".format(i)
            if nearest_size != 0:
                a=w2v.most_similar(word, topn = nearest_size)
                clos_ten_in = [i[0] for i in a] #grab vectors and not similarity probability
                clos_ten_out.append(clos_ten_in) #gather words in a single list                
            else:
                 clos_ten_out = []

    except ValueError as e:#Empty word None input
            print(e)
            pass
    except KeyError as e: #Word not in vocab
            print(e)    
            pass 

    ip_nearest=[]
    clos_ten_dict={}
    try:
        for i,sub_list in enumerate(clos_ten_out):
            clos_ten_dict[i] = list(tsne_df_full[tsne_df_full['word'].isin(sub_list)].index.values)  
            tsne_df_full.loc[clos_ten_dict[i],'type'] = "near_ip_{}".format(i)
            ip_nearest.append(clos_ten_dict[i]) #index values
        
        
    except Exception as e:
            print("clos_ten_out is empty: ", e)
        #     plot_index = ip_vocab_index + clos_ten_index
            pass    
    
    ip_nearest_ix = [item for sublist in ip_nearest for item in sublist]
    # ip_nearest_ix
    # ip1='king'
    # ip2='man'
    # ip3='woman'
    
    #Try catches no input case
    try:
        r=w2v.most_similar(positive=[ip1,ip3], negative= [ip2], topn = nearest_size+1)
        res_ten_in = [i[0] for i in r]
        print("Res words are: ", res_ten_in)
        res_ten_index = list(tsne_df_full[tsne_df_full['word'].isin(res_ten_in)].index.values)
  
    except Exception as e:      #TypeError   
        print("Trying addition ")
        try:
            r=w2v.most_similar(positive=[ip1,ip3], topn = nearest_size+1)
            res_ten_in = [i[0] for i in r]
            print("Res words of addition: ", res_ten_in)
            res_ten_index = list(tsne_df_full[tsne_df_full['word'].isin(res_ten_in)].index.values)
            
        except Exception as e:        
            print("Trying subtraction: ")
            try:
                r=w2v.most_similar(positive=[ip1], negative= [ip2], topn = nearest_size+1)
                res_ten_in = [i[0] for i in r]
                print("Res words of subtraction: ", res_ten_in)
                res_ten_index = list(tsne_df_full[tsne_df_full['word'].isin(res_ten_in)].index.values)
            
                
            except Exception as e:
                print("Insufficient inputs",e)
                pass
            
    try:
        tsne_df_full.loc[res_ten_index[0],'type'] = 'res'
        tsne_df_full.loc[res_ten_index[1:],'type'] = 'near_res'
        
    except Exception as e:
        print("res nearest not defined: ", e)
        pass
    
    try:
        plot_index = ip_vocab_index + ip_nearest_ix + res_ten_index
        
    except Exception as e:
        print("plot index not fully defined: ",e)
        # plot_index = list(filter(None, plot_index))
        # print("After filtering, plot_index now is: ", plot_index)
        pass
    
    # print("ip_vocab_index, word, type is: ", ip_vocab_index, tsne_df_full.loc[ip_vocab_index,'word'],tsne_df_full.loc[ip_vocab_index,'type'])
    # print("ip_nearest_ix, word, type is: ", ip_nearest_ix,tsne_df_full.loc[ip_nearest_ix,'word'], tsne_df_full.loc[ip_nearest_ix,'type'])
    
    
    try:
        print("res_ten_index, word, type is: ", res_ten_index,tsne_df_full.loc[res_ten_index,'word'], tsne_df_full.loc[res_ten_index,'type'])
    except UnboundLocalError as e:
        print(e)
        pass
    
    
    try:
        tsne_plot_df = tsne_df_full.loc[plot_index]
    except UnboundLocalError as e:
        print(e)
        pass
            
    ctx = dash.callback_context
    if ctx.triggered:
        # print(ctx.triggered)
        trigger = (ctx.triggered[0]['prop_id'].split('.')[0])
        print("trigger is: ", trigger)
        print("word_tog nclick value is: ", word_tog)
        

        if trigger == 'words_tog':
            if (word_tog % 2) == 0:                
                print("Words display toggled")
                text_disp = 'word'
                
            else:
                text_disp = None        
    
    if nearest_size != 0 :
        order  = ["ip_0", "ip_1", "ip_2", "res","near_ip_0", "near_ip_1", "near_ip_2", "near_res"]
 
    else:
        order  = ["ip_0", "ip_1", "ip_2", "res"]
        
    #accomodate edge cases where input ex "near_ip_0" does not exist etc.        
    pop_list_flag = 1
    while pop_list_flag == 1: #To iteratively remove undefined inputs
        try:
            df = tsne_plot_df.set_index('type') #set as index so that order can be specified 
            df_ordered = df.T[order].T.reset_index()
    
        except KeyError as e:
            print("Exception in df_ordered dataframe: ",e)
            print("Entering while loop to remove items from order list")        
            #Edge cases where same word is near multiple classifications like "near_res" and "near_ip_0".
            #near_res must assume precedence. So drop the redundant types till no error. 
            print("Initial order is: ", order)        

            s=str(e)
            
            rem_val=s[s.find("["):s.find("]")+1]
            
            rem_val = ast.literal_eval(rem_val)
           
            order.remove(rem_val[0])
            print("order is now: ", order)
        
        except UnboundLocalError as e:
            print(e)
            pop_list_flag =0 #Else it continues in an endless while loop
            break
        
        else: #if try is succesful, end the while loop.s
            pop_list_flag =0
        
    try:    #if fig_1 is in the main program, "referenced before assignment" error.
        fig_2 = px.scatter(df_ordered, x="x", y="y", text= text_disp, color='type', 
                            color_discrete_map={
                                 "ip_0": "#7bc043", #green- most natural word by essence "concept1" 
                                 "ip_1": "#fdf498", #yellow- noticeable contrast from "concept1"; "concept2a"                                
                                 "ip_2": "#f37736", #orange- "concept2b"lighter contrast from "concept2a"; clear contrast from concept1 and noticeable difference from 2a
                                 "res": "#ee4035", #red- attracts first attention; strange that it should make sense. result should be analagous to ip1. The largest perceived context for the user. 

                                 "near_ip_0": "#339900", 
                                 "near_ip_1": "#ffcc00",
                                 "near_ip_2": "#ff9966",                                 
                                 "near_res": "#cc3300"},                           
                           log_x=False, hover_name = 'word') 
    
    except UnboundLocalError as e:
        print (e)
        try:
            fig_2 = px.scatter(df_ordered, x="x", y="y", text= 'word', color='type', 
                               color_discrete_map={
                                    "ip_0": "#7bc043", #green- most natural word by essence "concept1" 
                                     "ip_1": "#fdf498", #yellow- noticeable contrast from "concept1"; "concept2a"                                
                                     "ip_2": "#f37736", #orange- "concept2b"lighter contrast from "concept2a"; clear contrast from concept1 and noticeable difference from 2a
                                     "res": "#ee4035", #red- attracts first attention; strange that it should make sense. result should be analagous to ip1. The largest perceived context for the user. 
                                     "near_ip_0": "#339900", 
                                     "near_ip_1": "#ffcc00",
                                     "near_ip_2": "#ff9966",                                 
                                     "near_res": "#cc3300"},
                               hover_name = 'word', log_x=False)
        
        except UnboundLocalError as e:
            print(e)
            pass
    
    
    try:
        fig_2.update_traces(textposition='top center')      
        fig_2.update_layout(legend_traceorder="normal")
        fig_2.update_layout(margin=dict(l=0, r=0), uirevision = True)
        return (fig_2), res_ten_in[0]
    
    except Exception as e:
        print("No return values. Error: ",e)
        raise dash.exceptions.PreventUpdate
        pass
  

################################ Arbitrary_graphing_inputbox ################################

@app.callback(
    [Output(component_id='input_1', component_property='placeholder'),
     Output(component_id='input_1', component_property='value')],    
    [Input('add1_button','n_clicks'),
     Input('rem1_button','n_clicks'),
     Input('clr1_button','n_clicks')],
    [State(component_id='input_1', component_property='value')],
    prevent_initial_call=False
    )

def vocab_list(add,rem,clr,inp_1):
    
    ctx = dash.callback_context
    if ctx.triggered:
        print(ctx.triggered)
        trigger = (ctx.triggered[0]['prop_id'].split('.')[0])
        
        if trigger == 'add1_button':
            if ',' in inp_1: #if typing a list directly split into its words
                inp_1=inp_1.replace(" ", "") #remove whitespace
                split_words=inp_1.split(',')
                print("Input words added are: ",split_words)
                vocab_plot_list.extend(split_words)
                value=''
                return vocab_plot_list, value
            
            else:
                print("Input word added: ",inp_1)
                vocab_plot_list.append(inp_1)
                print("vocab_list is: ",vocab_plot_list )
                value=''
                return vocab_plot_list, value
        
        
        if trigger == 'rem1_button':   
            print("Last word removed")
            vocab_plot_list.pop()
            print("vocab_list is now: ",vocab_plot_list)
            value=''
            return vocab_plot_list, value
 

        if trigger == 'clr1_button':   
            print("Clr button clicked")
            vocab_plot_list.clear()
            print("vocab_list is now: ",vocab_plot_list )
            value=''
            return vocab_plot_list, value
        
    else:
        vocab_plot_list.append(inp_1)
        value=''
        return vocab_plot_list, value
        
        
       
################################ Arb_Word2vec_1 graph ################################

@app.callback(
    Output(component_id='word2vec_1', component_property='figure'),  
    [Input('add1_button','n_clicks'),
     Input('rem1_button','n_clicks'),
     Input('clr1_button','n_clicks'),
     Input('plt1_button','n_clicks'),
     # Input('wind_dd_1','value'),
     # Input('vector_dd_1','value'),
     Input('my_slider','value'),
     Input('words_tog','n_clicks'),
     Input('input_1', 'value')], #debounce of input box triggers callback of figure
    prevent_initial_call=False
    )

# def svd_user_inputs(ad,rem,clr,plot_butt,window,vector,slider_val,word_tog):
def svd_user_inputs(ad,rem,clr,plot_butt,slider_val,word_tog,ip_tog):    
    
    # if vector == 100:
    #     tsne_df = tsne_100d_w5_df

    # elif vector == 200:
    #     tsne_df = tsne_200d_w5_df
    
    # elif vector ==300:
    #     if window == 2:
    #         tsne_df = tsne_300d_w2_df
        
    #     if window == 3:
    #         tsne_df = tsne_300d_w3_df
        
    #     if window == 4:
    #         tsne_df = tsne_300d_w4_df
        
    #     if window == 5:
    #         tsne_df = tsne_300d_w5_df
      
    
    ####-------------
    # Add lighter shade of the parent category for the discerning of which point is for which word.
    # Then execute the Word math input fields.
    
    # vocab_to_plot = ['vanilla','organic','sustainable','cacao']
    vocab_to_plot = vocab_plot_list #initialised at the start of program
    # vocab_to_plot = vocab_full#initialised in Co-occ code chunk ()
    
    # tsne_df_full = tsne_df
    tsne_df_full = tsne_300d_w5_df
    
    # w2v = KeyedVectors.load_word2vec_format('organic_glove_300d.txt') 
    print("Slider val: ",slider_val)
    # nearest_size = 20  #make input for nearestneighbor size
    nearest_size = slider_val   
    #index of input words
    
    tsne_df_full['type'] = 'Sel' 
    ip_vocab_index=list(tsne_df_full[tsne_df_full['word'].isin(vocab_to_plot)].index.values)  
    
    print("vocab to plot of slider graph is: ",vocab_to_plot )
    
    clos_ten_out = [] #index of closest 10 words
    for word in vocab_to_plot: 
        try:
            a=w2v.most_similar(word, topn = nearest_size)
            clos_ten_in = [i[0] for i in a]
            clos_ten_out.append(clos_ten_in)
            # clos_ten_words = [i[1] for i in a]
            
        except ValueError as e:#Empty word None input
            print(e)
            pass
        except KeyError as e: #Word not in vocab
            print(e)    
            pass            
    print("close_ten_words are: ",clos_ten_out)
    clos_ten_flat = [item for sublist in clos_ten_out for item in sublist]
    clos_ten_index = list(tsne_df_full[tsne_df_full['word'].isin(clos_ten_flat)].index.values)  
    
    plot_index = ip_vocab_index + clos_ten_index
    tsne_df_full.loc[ip_vocab_index,'type'] = 'ip' # indices = [0,1,3,6,10,15]
    tsne_df_full.loc[clos_ten_index,'type'] = 'near'
    
    tsne_plot_df = tsne_df_full.loc[plot_index]
    # tsne_plot_df
    # tsne_plot_df = tsne_df_full[tsne_df_full['word'].isin(vocab_to_plot)] #filter to display
            
    ctx = dash.callback_context
    if ctx.triggered:
        # print(ctx.triggered)
        trigger = (ctx.triggered[0]['prop_id'].split('.')[0])
        print("trigger is: ", trigger)
        print("word_tog nclick value is: ", word_tog)
        

        if trigger == 'words_tog':
            if (word_tog % 2) == 0:                
                print("Words display toggled")
                # text_disp = '' #pass the words col #Getting ref before ass error
                # fig_1 = px.scatter(tsne_plot_df, x="x", y="y", color='type', log_x=False)
                text_disp = 'word'
                
            else:
                # fig_1 = px.scatter(tsne_plot_df, x="x", y="y", text= 'word',color='type', log_x=False)
                text_disp = None
           
                    
           
    try:    #if fig_1 is in the main program, "referenced before assignment" error.
        fig_1 = px.scatter(tsne_plot_df, x="x", y="y", text= text_disp, color='type', 
                           log_x=False, hover_name = 'word')    
    
    
    except UnboundLocalError as e:
        print (e)
        fig_1 = px.scatter(tsne_plot_df, x="x", y="y", text= 'word', color='type', 
                           hover_name = 'word', log_x=False)
        pass
                
    
    # fig_1 = px.scatter(tsne_plot_df, x="x", y="y", text=text_disp, color='type', log_x=False,
    #                    # color_discrete_map = {'ip': 'rgb(255,0,0)', 
    #                    #                       'setosa': 'rgb(0,255,0)', 
    #                    #                       'versicolor': 'rgb(0,0,255)'},
    #                    size_max=60)
    fig_1.update_traces(textposition='top center')
    
    
    ####-------------    
    # vocab_words_df_list = vocab_words_df[vocab_words_df['word'].isin(to_plot)]
    # fig_1 = px.scatter(vocab_words_df_list, x="x", y="y", text="word", log_x=False, size_max=60)
    # fig_1.update_traces(textposition='top center')    
    
    fig_1.update_layout(margin=dict(l=0, r=0), uirevision = True)
        
    return (fig_1)


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

# chosen_cols = 'description'
def ren_wordcloud(chosen_rows, chosen_cols):
    global wc_init, fig_wordcloud1 #global for pre-app saving of the object
    if len(chosen_cols) > 0: #atleast 1 col to be selected
        if len(chosen_rows)==0:                    
            df_filtered = df_disp_1[chosen_cols] #if no rows selected consider all rows
            if wc_init == 1: #load once.
                return fig_wordcloud1
                
                
            else:
                wc_init = 1 #object initialised flag set
                
                print("Full Word cloud not initialised")
                print("Initialising full word cloud..")
                df_filtered['comb_cols'] = df_filtered[df_filtered.columns[0:]].apply( #Combine multiple columns into a single series for wordcloud generation
                    lambda x: ' '.join(x.dropna().astype(str)),        
                    axis=1)
            
                en_stopwords = stopwords.words('english')
                fr_stopwords = stopwords.words('french')
                web_links_sw = ['www','http','https','com']
                
                combined_stopwords = en_stopwords + fr_stopwords + web_links_sw
                
                print("\nNo of rows in WC are: ",len(df_filtered))
                wordcloud = WordCloud(max_words=100,    
                                      # stopwords= combined_stopwords,
                                      # collocations=False,
                                      # color_func=lambda *args, **kwargs: "orange",
                                      colormap='tab20c',
                                      background_color='white',
                                      width=1700, #1200,1700    
                                      height=1000, #1000
                                      random_state=1).generate(' '.join(df_filtered['comb_cols'])) #df_filtered has to be a series
            
                # print(' '.join(df_filtered['comb_cols']))
            
                fig_wordcloud1 = px.imshow(wordcloud, template='ggplot2',
                                          )#title="Word frequency" 
            
                fig_wordcloud1.update_layout(margin=dict(l=0, r=0,b=0,t=0))
                fig_wordcloud1.update_xaxes(visible=False)
                fig_wordcloud1.update_yaxes(visible=False)
                return fig_wordcloud1
            

        elif len(chosen_rows) > 0 :
            # df_filtered = df_disp_1.iloc[chosen_rows,[chosen_cols]] #filter by selected rows
            df_filtered = df_disp_1[chosen_cols]
            df_filtered = df_filtered[df_filtered.index.isin(chosen_rows)]
            #print("Atleast 1 col chosen and multiple rows selected type. Dataype of df_filtered is", type(df_filtered))
            
    elif len(chosen_cols) == 0:
        raise dash.exceptions.PreventUpdate

## load wordcloud only once
    

    df_filtered['comb_cols'] = df_filtered[df_filtered.columns[0:]].apply( #Combine multiple columns into a single series for wordcloud generation
        lambda x: ' '.join(x.dropna().astype(str)),        
        axis=1)

    en_stopwords = stopwords.words('english')
    fr_stopwords = stopwords.words('french')
    web_links_sw = ['www','http','https','com']
    
    combined_stopwords = en_stopwords + fr_stopwords + web_links_sw
    
    print("\nNo of rows in WC are: ",len(df_filtered))
    wordcloud = WordCloud(max_words=100,    
                          # stopwords= combined_stopwords,
                          # collocations=False,
                          # color_func=lambda *args, **kwargs: "orange",
                          colormap='tab20c',
                          background_color='white',
                          width=1700, #1200,1700    
                          height=1000, #1000
                          random_state=1).generate(' '.join(df_filtered['comb_cols'])) #df_filtered has to be a series

    # print(' '.join(df_filtered['comb_cols']))

    fig_wordcloud = px.imshow(wordcloud, template='ggplot2',
                              )#title="Word frequency" 

    fig_wordcloud.update_layout(margin=dict(l=0, r=0,b=0,t=0))
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


# In[14]:

# In[15]:

    