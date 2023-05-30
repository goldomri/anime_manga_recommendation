# %%
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import ast

# %% Load and preprocess data
manga_data = pd.read_csv("manga.csv")

manga_list_features = ['genres', 'demographics', 'themes', 'serializations']
for col in manga_list_features:
    manga_data[col] = manga_data[col].apply(ast.literal_eval)

# Remove unnecessary columns
manga_drop = ['start_date', 'end_date', 'created_at_before', 'updated_at', 'real_start_date', 'real_end_date',
              'background', 'main_picture', 'url', 'title_english', 'title_japanese', 'title_synonyms', 'authors',
              'manga_id', 'scored_by', 'score', 'volumes', 'chapters', 'members', 'favorites']
manga_data = manga_data.drop(manga_drop, axis=1)

# Choose only approved manga
manga_data = manga_data.loc[manga_data['approved'] == True]
manga_data = manga_data.loc[manga_data['jikan'] == True]
manga_data = manga_data.loc[manga_data['sfw'] == True]
manga_data = manga_data.drop(['approved', 'jikan', 'sfw'], axis=1)

# Create Categorical features
manga_categorical_features = ['type', 'status']
for col in manga_categorical_features:
    manga_data = pd.get_dummies(manga_data, columns=[col]).groupby(level=0).sum()

for col in manga_list_features:
    categorical_features = pd.get_dummies(manga_data[col].explode())
    manga_data = pd.concat([manga_data, categorical_features.groupby(level=0).sum()], axis=1).drop(col, axis=1)

del categorical_features

# %% Learning Process
manga_features = manga_data.drop(['title', 'synopsis'], axis=1)

num_of_neighbors = 6
nbrs = NearestNeighbors(n_neighbors=num_of_neighbors, algorithm='ball_tree').fit(manga_features)
distances, indices = nbrs.kneighbors(manga_features)


# %% Main output functions
def get_index_from_title(title):
    """
    gets index of manga from title.
    :param title: The title of the wanted manga
    :return: The index of the manga in the dataset
    """
    return manga_data[manga_data['title'].str.lower() == title.lower()].index.tolist()[0]


def print_similar_manga(title, synopsis=False):
    """
    Prints 5 similar manga to a given title.
    :param title: The specific manga title
    :return: None
    """
    found_id = get_index_from_title(title)
    for id in indices[found_id][1:]:
        if synopsis:
            print("*" * 300)
            print(manga_data.iloc[id]['title'])
            print(manga_data.iloc[id]['synopsis'])
            print("*" * 300)
            print("\n")
        else:
            print(manga_data.iloc[id]['title'])
