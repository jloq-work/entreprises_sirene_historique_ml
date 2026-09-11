#robustesse_significativite_clusters.py
"""
-Evaluation de la significativité dezs clusters: estimations des coeffcients de variation moyens des features au sein des clusters

-Evaluation de la stabilité-reproductibilité des clusters par rééchantillonnage avec remplacement (bootstrap)


"""

import numpy as np
from sklearn.metrics import adjusted_rand_score
import hdbscan


#fonctions pour optimiser les hyperparamètres de la méthode de clusterisation HDBSCAN afin:
#assurer un bon score de DBCV
#optimiser la significativité des clusters (coefficient de variation)
#optimiser la robustesse des clusters (bootstrap)

# robustesse des clusters (bootstrap)
def evaluate_stability_fast(
    X, weights, min_cluster_size, min_samples, labels_orig, n_bootstraps=3
) -> float:
    # On vérifie s'il y a assez de clusters valides à la base (au moins 2)
    if len(set(labels_orig)) - (1 if -1 in labels_orig else 0) < 2:
        return 0.0

    ari_scores = []
    n_samples = X.shape[0]
    boot_size = int(n_samples * 0.8)  # Sous-échantillonnage à 80%

    # Normalisation des poids pour créer des probabilités de tirage
    prob_weights = weights.squeeze() / np.sum(weights)

    for _ in range(n_bootstraps):
        # Tirage pondéré par les poids de tirage
        boot_indices = np.random.choice(n_samples, size=boot_size, replace=True, p=prob_weights)
        X_boot = X[boot_indices]
        
        # Labels originaux correspondants aux indices tirés
        labels_orig_boot = labels_orig[boot_indices]

        # Entraînement du modèle de bootstrap
        clusterer_boot = hdbscan.HDBSCAN(
            min_cluster_size=max(2, int(min_cluster_size * 0.8)),
            min_samples=max(1, int(min_samples * 0.8)),
            core_dist_n_jobs=-1,
        )
        labels_boot = clusterer_boot.fit_predict(X_boot)

        # Pour mesurer la similarité des clusters entre le jeu de données original et les échantillons bootstrap, utlisation de l'Indice de Rand Ajusté (ARI). 
        
        ari = adjusted_rand_score(labels_boot, labels_orig_boot)
        ari_scores.append(ari)

    return float(np.mean(ari_scores))


# significativité des clusters (coefficient de variation)
def evaluate_significance(X, weights, labels) -> float:
    unique_labels = set(labels) - {-1}
    if not unique_labels:
        return -10.0

    cv_par_cluster = []
    for label in unique_labels:
        cluster_points = X[labels == label]
        cluster_weights = weights[labels == label]
        
        if len(cluster_points) == 0:
            continue
            
        # 1. On s'assure que cluster_weights est strictement à 1 dimension (forme: (N,))
        weights_1d = cluster_weights.values.flatten()
        sum_weights = np.sum(weights_1d)

        # 2. On ajoute np.newaxis pour aligner les dimensions (forme finale: (N, 1))
        weights_2d = weights_1d[:, np.newaxis]

        # 3. Calcul de la moyenne pondérée (axis=0 par feature)
        means_weighted = np.sum(cluster_points * weights_2d, axis=0) / sum_weights

        # 4. Calcul de la variance et de l'écart-type pondérés
        variance_weighted = np.sum(weights_2d * (cluster_points - means_weighted) ** 2, axis=0) / sum_weights
        stds_weighted = np.sqrt(variance_weighted)


        means_weighted = np.where(means_weighted == 0, 1e-6, means_weighted)
    
        cv_features = np.abs(stds_weighted / means_weighted)
        cv_par_cluster.append(np.mean(cv_features))

    return float(-np.mean(cv_par_cluster))




