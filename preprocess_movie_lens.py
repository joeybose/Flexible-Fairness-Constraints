import numpy as np
import pandas as pd
import ipdb

def make_dataset(load_sidechannel=False):
    r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
    train_ratings = pd.read_csv('./ml-100k/u1.base', sep='\t', names=r_cols,
			  encoding='latin-1')
    test_ratings = pd.read_csv('./ml-100k/u1.test', sep='\t', names=r_cols,
			  encoding='latin-1')
    if load_sidechannel:
        u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
        users = pd.read_csv('./ml-100k/u.user', sep='|', names=u_cols,
                            encoding='latin-1', parse_dates=True)
        # ipdb.set_trace()
        m_cols = ['movie_id', 'title', 'release_date', 'video_release_date', 'imdb_url']
        movies = pd.read_csv('./ml-100k/u.item', sep='|', names=m_cols, usecols=range(5),
                             encoding='latin-1')
        movie_ratings = pd.merge(movies, train_ratings)
        df = pd.merge(movie_ratings, users)
        df.drop(df.columns[[3,4,7]], axis=1, inplace=True)
        movies.drop(movies.columns[[3,4]], inplace = True, axis = 1 )

    train_ratings.drop( "unix_timestamp", inplace = True, axis = 1 )
    train_ratings_matrix = train_ratings.pivot_table(index=['movie_id'],\
            columns=['user_id'],values='rating').reset_index(drop=True)
    train_ratings_matrix.fillna( 0, inplace = True )
    test_ratings.drop( "unix_timestamp", inplace = True, axis = 1 )
    test_ratings_matrix = train_ratings.pivot_table(index=['movie_id'],\
            columns=['user_id'],values='rating').reset_index(drop=True)
    test_ratings_matrix.fillna( 0, inplace = True )
    train_ratings_matrix.head()
    columnsTitles=["user_id","rating","movie_id"]
    train_ratings=train_ratings.reindex(columns=columnsTitles)-1
    test_ratings=test_ratings.reindex(columns=columnsTitles)-1
    users['user_id'] = users['user_id'] - 1
    movies['movie_id'] = movies['movie_id'] - 1
    if load_sidechannel:
        return train_ratings,test_ratings,users,movies
    else:
        return train_ratings,test_ratings,

# if __name__ == '__main__':
    # make_dataset(True)
