import nltk
from fastapi import FastAPI
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from rake_nltk import Rake
import numpy as np
import fastparquet
import nltk
nltk.download('stopwords')
nltk.download('punkt_tab')


app = FastAPI()

@app.get("/cantidad_filmaciones_mes/{mes}")
def cantidad_filmaciones_mes(mes:str):
    df_endpoint1 = pd.read_parquet('end1.parquet')
    filmaciones = df_endpoint1['title'][df_endpoint1['release_month'] == mes]
    cantidad = filmaciones.count()

    return f'{cantidad} películas fueron estrenadas en el mes de {mes}'


@app.get("/cantidad_filmaciones_dia/{dia}")
def cantidad_filmaciones_dia(dia:str):
    df_endpoint2 = pd.read_parquet('end2.parquet')
    filmaciones = df_endpoint2['title'][df_endpoint2['release_day'] == dia]
    cantidad = filmaciones.count()
    return f'{cantidad} películas fueron estrenadas en el dia {dia}'


@app.get("/score_titulo/{titulo}")
def score_titulo(titulo:str):
    df_endpoint3 = pd.read_parquet('end3.parquet')
    film = df_endpoint3[df_endpoint3['title'] == titulo]
    title = film['title'].values[0]
    year = film['release_year'].values[0]
    score = film['popularity'].values[0]
    return f'La película {title} fue estrenada en el año {int(year)} con un score/popularidad de {score}'


@app.get("/votos_titulo/{titulo}")
def votos_titulo(titulo:str):
    df_endpoint4 = pd.read_parquet('end4.parquet')
    film = df_endpoint4[df_endpoint4['title'] == titulo]
    valoraciones = film['vote_count'].values[0]
    promedio = film['vote_average'].values[0]
    año = film['release_year'].values[0]
    titulo = film['title'].values[0]
    if valoraciones < 2000:
        return 'no cumple con la cantidad de valoraciones minimas'
    else:
        return f'La película {titulo} fue estrenada en el año {int(año)}. La misma cuenta con un total de {valoraciones} valoraciones, con un promedio de {promedio}'


@app.get("/get_actor/{nombre}")
def get_actor(nombre:str):
    df_endpoint5 = pd.read_parquet('end5.parquet')
    peliculas = df_endpoint5[df_endpoint5['actors'].apply(lambda actors: nombre in actors)]
    pelis = peliculas['title'].tolist()
    retorno = peliculas['return'].tolist()
    cantidad = len(pelis)
    ganancia = sum(retorno)
    promedio = ganancia/cantidad

    return f'El actor {nombre} ha participado en {cantidad} filmaciones, el mismo ha conseguido un retorno de {round(ganancia,3)} millones con un promedio de {round(promedio,3)} millones por filmación'


@app.get("/get_director/{nombre}")
def get_director(nombre:str):
    df_endpoint6 = pd.read_parquet('end6.parquet')
    peliculas_director = df_endpoint6[df_endpoint6['director'] == nombre]
    
    if peliculas_director.empty:
        return f"No se encontraron películas para el director: {nombre}"
    
    retorno_total = peliculas_director['return'].sum()
    
    # Crear el DataFrame con la información requerida
    informacion_peliculas = peliculas_director[['title', 'release_date', 'return', 'budget', 'revenue']]
    
    return {
        'nombre_director': nombre,
        'retorno_total': retorno_total,
        'informacion_peliculas': informacion_peliculas
    }


@app.get("/recomendacion/{titulo}")
def recomendacion(titulo: str):
    df_reco = pd.read_parquet('reco.parquet')
    
    # Inicializar el extractor RAKE
    rake = Rake()

    # Función para extraer palabras clave
    def extract_keywords(text):
        if isinstance(text, str):
            rake.extract_keywords_from_text(text)
            return ' '.join(rake.get_ranked_phrases())
        else:
            return ""

    # Aplicar la extracción de palabras clave a la columna 'overview'
    df_reco['keywords'] = df_reco['overview'].apply(extract_keywords)

    # Crear una nueva columna combinando 'title', 'overview' y 'keywords'
    df_reco['combined_text'] = df_reco['title'] + ' ' + df_reco['keywords']

    # Aplicar TfidfVectorizer sobre el texto combinado, limitando el número de características
    vect = TfidfVectorizer(stop_words='english', max_features=5000)  # Ajustar 'max_features' según tu memoria
    vect_matrix = vect.fit_transform(df_reco['combined_text'])

    # Usar NearestNeighbors para encontrar las películas más similares
    nn_model = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=6)
    nn_model.fit(vect_matrix)

    # Verificar si el título está en el DataFrame
    if titulo not in df_reco['title'].values:
        return {"error": f"No se encontró la película: {titulo}"}

    # Obtener el índice de la película en cuestión
    idx = df_reco[df_reco['title'] == titulo].index[0]

    # Encontrar los índices de las películas más cercanas
    distances, indices = nn_model.kneighbors(vect_matrix[idx], n_neighbors=6)

    # Obtener los títulos de las películas más similares (excluir la misma película)
    similar_titles = df_reco['title'].iloc[indices[0][1:]].tolist()
    
    return {"recomendaciones": similar_titles}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)