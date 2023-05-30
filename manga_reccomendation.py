# %%
import numpy as np
import pandas as pd
from sklearn.feature_extraction import text
from sklearn.metrics.pairwise import linear_kernel

# %% Load and preprocess data
manga_data = pd.read_csv("manga.csv")
manga_drop = ['start_date', 'end_date', 'created_at_before', 'updated_at', 'real_start_date', 'real_end_date',
              'synopsis', 'background', 'main_picture', 'url', 'title_english', 'title_japanese', 'title_synonyms',
              'serializations', 'themes']
manga_data = manga_data.drop(manga_drop, axis=1)
manga_data = manga_data.loc[manga_data['approved'] == True]
manga_data = manga_data.loc[manga_data['jikan'] == True]
manga_data = manga_data.loc[manga_data['sfw'] == True]
manga_data = manga_data.drop(['approved', 'jikan', 'sfw'], axis=1)

manga_data = pd.get_dummies(manga_data, columns=['type'])

manga_list_features = ['genres', 'demographics']
for col in manga_list_features:
    manga_data[col] = manga_data[col].str.strip("[]").str.split('\s*,\s*')

    manga_data1 = (
        manga_data[col].explode()
        .str.get_dummies().add_prefix(col)
    )

    manga_data1 = manga_data.drop(col, axis=1).join(manga_data1)
    manga_data = manga_data1
del manga_data1
