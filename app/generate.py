import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
import itertools
import io, base64

color_cycle= itertools.cycle(["orange","pink","blue","brown","red","grey","yellow","green"])

def generate_clusters_kmeans(X, y_kmeans, kmeans, n_clusters, init = "k-means++", max_iter = 300, n_init = 10, random_state = 0):
   
    for i in range(0, n_clusters):
        plt.scatter(X[y_kmeans == i, 0], X[y_kmeans == i, 1], s = 100, c = next(color_cycle), label = "Cluster " + str(i))
    plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s = 50, c = next(color_cycle), label = "Baricentros")
    
    for x, y in kmeans.cluster_centers_:
        plt.annotate(str((round(x,1), round(y,1))), (x, y))

    plt.title("Clusters ....")
    plt.xlabel("X label")
    plt.ylabel("Y label")
    plt.legend()

    flike = io.BytesIO()
    plt.savefig(flike)
    b64 = base64.b64encode(flike.getvalue()).decode()
    plt.close()
    return (b64)


def generate_WCSS(x_kmeans, n_clusters=6, init = "k-means++", max_iter = 300, n_init = 10, random_state = 0):
    wcss = []
    wcss_res = []
    for i in range(2, n_clusters*3):
        kmeans = KMeans(n_clusters = i, init = init, max_iter = max_iter, n_init = n_init, random_state = random_state)
        kmeans.fit(x_kmeans)
        wcss_res.append([i, kmeans.inertia_])
        wcss.append(kmeans.inertia_)
    
    plt.plot(range(2,n_clusters*3), wcss, markersize=30, lw=2)
    plt.grid(True)
    plt.title("Método del codo")
    plt.xlabel("Número de Clusters")
    plt.ylabel("WCSS(k)")
    for i, label in enumerate(wcss):
        plt.annotate(str(round(label)), (i+2, label + label/(n_clusters*3)))

    plt.scatter(range(2,n_clusters*3), wcss, s = 50, c = 'red', label = "Suma de la distancia cuadrada(Error)")
    plt.legend()

    flike = io.BytesIO()
    plt.savefig(flike)
    b64 = base64.b64encode(flike.getvalue()).decode()
    plt.close()
    return b64, wcss_res


def generate_statistics(dataset, kmeans, num_variables):
    dataset = dataset.iloc[:,[1, 2]]
    statistics = dataset.groupby(['cluster'], as_index=False).count()
    num_cluster = len(statistics['cluster'])
    statistics['porcentaje']=(statistics.iloc[:,-1] / statistics.iloc[:,-1].sum() *100).round(2)
    statistics = statistics.sort_values('porcentaje', ascending=False)
    statistics['numero'] = [x+1 for x in range(len(statistics))]
    baricentro=[]
    baricentro.append(["Cluster {}".format(x+1) for x in range(num_cluster)])
    for x in range(0, num_variables):
        baricentro.append(kmeans.cluster_centers_[:,x].tolist())

    # Cluster, Count, Mode, Percentage, Numero_Id
    return statistics, baricentro