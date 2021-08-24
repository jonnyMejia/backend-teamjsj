from enum import Enum
from pydantic import BaseModel


class InitKmeans(str, Enum):
    kmeans = "k-means++"
    random = "random"

class ClassifierKmeansBase(BaseModel):
    n_clusters : int
    init : InitKmeans = InitKmeans.kmeans
    max_iter : int = 300
    n_init : int = 10
    random_state : int = 0

class ClassifierKmeans(ClassifierKmeansBase): 
    pass

class ClassifierKmeansInDb(ClassifierKmeansBase):
    query : str