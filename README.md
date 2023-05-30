# Anime & Manga Recommendation Project
This is a simple ML project consists of two programs for anime and manga recommendations. Each program is independant.

<img src="https://www.worldatlas.com/r/w960-q80/upload/89/75/f3/shutterstock-413397052.jpg" width=60% height=60%>

## Datasets
I used the MyAnimeList Anime and Manga Datasets: https://www.kaggle.com/datasets/andreuvallhernndez/myanimelist.

This dateset includes two files, anime.csv and manga.csv. each file contains data extracted from the site: http://myanimelist.net.

## Preproccessing
### Manga
For the manga recommendations I chose to filter out unapproved and not sfw manga, and to use these specific features:
1. score
2. type
3. status
4. genres
5. themes
6. demographic
7. serializations  

Execpt for the score feature, I converted all of the other used features into categorical features.
### Anime
For the anime recommendations I chose to filter out unapproved and not sfw anime, and to use these specific features:
1. score
2. type
3. source
4. genres
5. themes
6. demographic
7. studios
8. producers 

Execpt for the score feature, I converted all of the other used features into categorical features.


## Learning Algorithm
I used the KNN algorithm implemented in sklearn library: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html.
