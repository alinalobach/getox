import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Загрузка данных о фильмах
movies_data = pd.read_csv('movies.csv')

# Предобработка данных
movies_data['genres'] = movies_data['genres'].str.replace('|', ' ')

# Создание матрицы TF-IDF на основе описания фильмов
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies_data['genres'])

# Расчет косинусного сходства между фильмами
cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)

# Функция для получения рекомендаций фильмов на основе сходства
def get_recommendations(movie_title, cosine_similarities, movies_data):
    movie_index = movies_data[movies_data['title'] == movie_title].index[0]
    similarity_scores = list(enumerate(cosine_similarities[movie_index]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    top_similar_movies = similarity_scores[1:11]
    movie_indices = [index for index, _ in top_similar_movies]
    recommended_movies = movies_data['title'].iloc[movie_indices]
    return recommended_movies

# Пример использования
movie_title = '
