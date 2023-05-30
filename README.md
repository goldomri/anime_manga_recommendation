# Manga Recommendation
This is a simple ML project for manga recommendations based on KNN algorithm.

<img src="https://www.worldatlas.com/r/w960-q80/upload/89/75/f3/shutterstock-413397052.jpg" width=60% height=60%>

## Dataset
I used the MyAnimeList Anime and Manga Datasets: https://www.kaggle.com/datasets/andreuvallhernndez/myanimelist.

This dateset includes two files, anime.csv and manga.csv. each file contains data extracted from the site: http://myanimelist.net.

## Preproccessing
I chose to filter out unapproved and not sfw manga, and to use these specific features:
1. score
2. genres
3. themes
4. demographic
5. serializations  

Execpt for the score feature, I changed all of the other used features to categorical features.

## Learning Algorithm
I used the KNN algorithm implemented in sklearn library: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html.
