#preparation_donnees.py
"""
-Preparation du fichier de features

-Traitement des données en vue de la recherche de clusters:

*standardisation des données (log+normalisation)

*analyse en composantes principales pour réduire le nombre de features

"""
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer
from sklearn.decomposition import PCA

def preparation(df):

    #on retire tout ce qui n'est pas features
    features=df.drop(columns=["siren","poids_tirage","numero_echantillon"])

    #prétraitement des données
    #standardisation des données (log transformée et normalisation)
    #réduction du nombre features par PCA

    pipeline = Pipeline([
            ('pt', PowerTransformer(method='yeo-johnson',standardize=True)), #log transformée+standardisation
            ("pca",PCA(n_components=0.90,svd_solver='covariance_eigh')) #PCA, on retient les dimensions de la PCA qui intègrent 90% de la variance
        ]) 

    X=pipeline.fit_transform(features)

    return X