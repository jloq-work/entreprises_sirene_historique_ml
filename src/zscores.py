#zscores.py
"""
-Etablissement des profils des clusters: mesurer l'écart entre le cluster et l'échantillon qui le contient

-Calcul des zscores associés aux features au sein de chaque cluster

-Plus le zscore s'éloigne de 0, plus la feature définit le cluster

-Le calcul du zscore se fait sur les features avant standardisation et pca (!) mais en prenant en compte les pondérations de tirage

"""
import pandas as pd
import numpy as np

def zscores(df,weights)-> pd.DataFrame:

    feature_names= df.drop(columns=["Cluster"]).columns.tolist()

    w_global = np.array(weights).flatten()
    sum_w_global = np.sum(w_global)

    #Calcul de la moyenne pondérée des features dans l'échantillon'
    global_means_np = (
        np.sum(df[feature_names].values * w_global[:, np.newaxis], axis=0)
        / sum_w_global
    )
    global_vars_np = (
        np.sum(w_global[:, np.newaxis] * (df[feature_names].values - global_means_np) ** 2, axis=0)
        / sum_w_global
    )
    global_stds_np = np.sqrt(global_vars_np)

    unique_clusters_clean = sorted([c for c in df["Cluster"].unique() if c != -1])

    z_scores_list = []
    y_axis_labels = []

    for cluster in unique_clusters_clean:
        cluster_mask = df["Cluster"] == cluster
        cluster_data = df[cluster_mask]
        w_cluster = w_global[cluster_mask]

        count = len(cluster_data)
        sum_w_cluster = np.sum(w_cluster)

        # CORRECTION : Le pourcentage est calculé sur la somme des poids de tirage
        percentage_weighted = (sum_w_cluster / sum_w_global) * 100
        y_axis_labels.append(
            f"Cluster {cluster}\n(N={count:,} | {percentage_weighted:.1f}%)"
        )

        #Calcul de la moyenne pondérée des features dans le cluster
        cluster_means_np = (
            np.sum(cluster_data[feature_names].values * w_cluster[:, np.newaxis], axis=0)
            / sum_w_cluster
        )

        #Calcul du Z-score pondéré 
        cluster_z_scores = (cluster_means_np - global_means_np) / global_stds_np
        z_scores_list.append(cluster_z_scores)

    # Matrice des zscores
    z_scores_matrix = pd.DataFrame(
        z_scores_list, index=unique_clusters_clean, columns=feature_names
    )

    return z_scores_matrix