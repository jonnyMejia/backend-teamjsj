from fastapi import FastAPI
from app.db.database import selectQuery
from sklearn.preprocessing import StandardScaler, Normalizer, LabelEncoder
from sklearn.cluster import KMeans
import pandas as pd
from typing import List
from app.models import ClassifierKmeansInDb, ClassifierKmeans
from .generate import generate_statistics, generate_clusters_kmeans, generate_WCSS


app = FastAPI()

@app.get("/")
async def root():
    return {"TEAM": "JSJ"}

@app.post("/kmeans/")
def classifier_kmeans(classifier_kmeans_db: ClassifierKmeansInDb):
    data, columnsNames = selectQuery(classifier_kmeans_db.query)
    classifier_kmeans = ClassifierKmeans(n_clusters=classifier_kmeans_db.n_clusters,
    init= classifier_kmeans_db.init,
    max_iter = classifier_kmeans_db.max_iter,
    n_init = classifier_kmeans_db.n_init,
    random_state = classifier_kmeans_db.random_state)
    num_variables = len(columnsNames)
    dataset = pd.DataFrame(data, columns=columnsNames).reset_index(drop=True)
    x_kmeans = dataset.apply(LabelEncoder().fit_transform).values
    x_kmeans = StandardScaler().fit_transform(x_kmeans)
    kmeans = KMeans(**classifier_kmeans.dict())
    y_kmeans = kmeans.fit_predict(x_kmeans)  
    dataset.insert(0, "cluster", y_kmeans)
    dataset.insert(0, "num", [x+1 for x in range(len(dataset))])
    statistics, baricentro = generate_statistics(dataset, kmeans, num_variables)

    image = None
    if len(columnsNames) == 2:
        image = generate_clusters_kmeans(x_kmeans, y_kmeans, kmeans, **classifier_kmeans.dict())
    codo, wcss = generate_WCSS(x_kmeans, **classifier_kmeans.dict())
    return {"classifier_kmeans": classifier_kmeans, 'names':columnsNames, 'results': dataset.values.tolist(), 
    "statistics":statistics.values.tolist(), 'codo':codo, 'image':image, 'baricentro':baricentro, 'wcss':wcss }
