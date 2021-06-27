import numpy as np
import pandas as pd

from surprise import SVD, NMF, SVDpp

columns = ['user_id','item_id','ratings','timestamp']
df_read = pd.read_csv("ml-100k/u.data", sep = '\t', names=columns)


columns = ['item_id', 'movie title', 'release date', 'video release date', 'IMDb URL', 'unknown', 'Action', 'Adventure',
          'Animation', 'Childrens', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
          'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

movies = pd.read_csv('ml-100k/u.item', sep='|', names=columns, encoding='latin-1')
movie_names = movies[['item_id', 'movie title']]

cm = pd.merge(df_read, movie_names, on='item_id')

cm = cm[['user_id','movie title', 'ratings']]
cm.head()

my_ratings = pd.read_csv('myratings.csv')

