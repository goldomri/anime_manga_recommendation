# %%
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import ast

NOT_FOUND_TITLE_MSG = "Didn't find anime title, please try again"

# %% Load and preprocess data
anime_data = pd.read_csv("anime.csv")

# Change list features to type list
anime_list_features = ['genres', 'demographics', 'themes', 'studios', 'producers']
for col in anime_list_features:
    anime_data[col] = anime_data[col].apply(ast.literal_eval)

# Process score feature
anime_data['score'].fillna(np.round((anime_data['score'].mean())), inplace=True)
anime_data['score'] = anime_data['score'].astype('Float64')

# Remove unnecessary columns
anime_drop = ['start_date', 'end_date', 'created_at', 'updated_at', 'episode_duration', 'total_duration',
              'background', 'main_picture', 'url', 'title_english', 'title_japanese', 'title_synonyms',
              'anime_id', 'scored_by', 'members', 'favorites', 'episodes', 'rating', 'start_year', 'start_season',
              'real_start_date', 'real_end_date', 'broadcast_day', 'broadcast_time', 'licensors', 'trailer_url']
anime_data = anime_data.drop(anime_drop, axis=1)

# Choose only approved anime
anime_data = anime_data.loc[anime_data['approved'] == True]
anime_data = anime_data.loc[anime_data['sfw'] == True]
anime_data = anime_data.drop(['approved', 'sfw'], axis=1)

# Create Categorical features
anime_categorical_features = ['type', 'status', 'source']
for col in anime_categorical_features:
    anime_data = pd.get_dummies(anime_data, columns=[col]).groupby(level=0).sum()

for col in anime_list_features:
    categorical_features = pd.get_dummies(anime_data[col].explode(), prefix=col)
    anime_data = pd.concat([anime_data, categorical_features.groupby(level=0).sum()], axis=1).drop(col, axis=1)

del categorical_features

# %% Learning Process
anime_features = anime_data.drop(['title', 'synopsis'], axis=1)

num_of_neighbors = 6
model = NearestNeighbors(n_neighbors=num_of_neighbors).fit(anime_features)
distances, indices = model.kneighbors(anime_features)


# %% Main output functions
def get_index_from_title(title):
    """
    Gets index of anime from title.
    :param title: The title of the wanted anime
    :return: The index of the anime in the dataset
    """
    try:
        return anime_data[anime_data['title'].str.casefold().str.replace(' ', "") ==
                          title.casefold().replace(' ', "")].index.tolist()[0]
    except IndexError:
        print(NOT_FOUND_TITLE_MSG)


def print_similar_anime(title, synopsis=False):
    """
    Prints 5 similar anime to a given title.
    :param title: The specific anime title
    :param synopsis: If True, the program will print the synopsis of similar anime
    :return: None
    """
    found_id = get_index_from_title(title)
    for id in indices[found_id][1:]:
        if synopsis:
            print("*" * 300)
            print(anime_data.iloc[id]['title'])
            print(anime_data.iloc[id]['synopsis'])
            print("*" * 300)
            print("\n")
        else:
            print(anime_data.iloc[id]['title'])
