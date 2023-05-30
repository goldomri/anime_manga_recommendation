# %%
import numpy as np
import pandas as pd
from sklearn.feature_extraction import text
from sklearn.metrics.pairwise import linear_kernel

# %%
manga_data = pd.read_csv("manga.csv")
manga_drop = ['start_date', 'end_date', 'created_at_before', 'updated_at', 'real_start_date', 'real_end_date',
              'synopsis', 'background', 'main_picture', 'url', 'title_english', 'title_japanese', 'title_synonyms']
manga_data.drop(manga_drop, axis=1)

