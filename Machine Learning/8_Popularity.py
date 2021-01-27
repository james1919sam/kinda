import pandas
from sklearn.model_selection import train_test_split
import numpy as np
import time
import Recommendor as Rec

#Read user_id, song_id, listen_count 
triplets = 'https://static.turi.com/datasets/millionsong/10000.txt'
songs_metadata = 'https://static.turi.com/datasets/millionsong/song_data.csv'

song_df_a = pandas.read_table(triplets,header=None)
song_df_a.columns = ['user_id', 'song_id', 'listen_count']

#Read song  metadata [song_id,title, release,artist_name, year]

song_df_b =  pandas.read_csv(songs_metadata)

#Merge the two dataframes above to create input dataframe for recommender systems
song_df1 = pandas.merge(song_df_a, song_df_b.drop_duplicates(['song_id']), on="song_id", how="left")
song_df1.head()
print("Total no of songs:",len(song_df1))

song_df1 = song_df1.head(10000)

#Merge song title and artist_name columns to make a new column
song_df1['song'] = song_df1['title'].map(str) + " - " + song_df1['artist_name']

song_gr = song_df1.groupby(['song']).agg({'listen_count': 'count'}).reset_index()
grouped_sum = song_gr['listen_count'].sum()
song_gr['percentage']  = song_gr['listen_count'].div(grouped_sum)*100
song_gr.sort_values(['listen_count', 'song'], ascending = [0,1])
print(song_gr)

u = song_df1['user_id'].unique()
print("The no. of unique users:", len(u))

#['user_id', 'song_id', 'listen_count',title,release,artist_name,year,song]

train, test_data = train_test_split(song_df1, test_size = 0.20, random_state=0)
print("*****Training data*****")
print(train.head(5))

pm = Rec.popularity_recommender()                               #create an instance of the class
pm.create_p(train, 'user_id', 'song') 
print("******starting the recommendation****")
user_id1 = u[8]                                                          #Recommended songs list for a user
print(pm.recommend_p(user_id1))

#print("**** starting the recommendation2****")
#user_id2 = u[8]
#print(pm.recommend_p(user_id2))


