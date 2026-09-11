
## Analyse et caractérisation exploratoires des trajectoires des entreprises immatriculées au Répertoire Sirene de l'Insee

### 1. Idée

Le Répertoire Sirene est diffusé en open data depuis la Loi pour une République Numérique notamment à travers un jeu de données open data sur la plateforme des données ouvertes data.gouv.fr

Dans ce jeu de données, plusieurs fichiers sont disponibles sur les entreprises et leurs établissements immatriculés au Répertoire Sirene de l'Insee.

Deux fichiers sont ici utilisés: les données historiques des entreprises et de leurs établissements enregistrés au Répertoire Sirene.

Dans ces fichiers, tout une partie des vies économiques et administratives des entreprises et de leurs établissements sont décrites à travers des périodes caractérisées par plusieurs features: dates de début et de fin de la période, secteur d'activité/catégorie juridique/état administratif etc 

**Ainsi à partir de ces fichiers, il est possible de reconstituer des signatures temporelles des entreprises:**

- combien de périodes différentes elles ont connu;
- combien d'établissements ont-elles crées/fermés;
- ont-elles souvent changé de secteur d'activié ou de catégorie juridique;



**Puis ensuite d'essayer d'analyser si des groupes d'entreprises aux signatures temporelles similaires existent et si oui comment caractériser ces groupes au regard des features économiques et administratifs présents dans le Répertoire Sirene** 

### 2. Méthodologie

- Récupération des fichiers au format Parquet des données historiques du Répertoire Sirene sur la plateforme data.gouv.fr
(jeu de données: *base-sirene-des-entreprises-et-de-leurs-etablissements-siren-siret*)

- Apurement des données

- Tirage d'un échantillon de 500 000 entreprises stratifiées par le nombre de périodes*secteur d'activité NAF

- Calcul des features (nombre de changements de secteur d'activité, durée de vie de l'entreprises, nombre d'établissements actifs, etc)

- Répartition équitable des 500 000 observations en 10 sous échantillons indépendants (utlité plus bas: vérifier la reproductibilité des profils des clusters sur plusieurs échantillons)

- Préparation des données (transformée logartithmique et normalisation des features, analyse en composantes principales pour essayer de ne garder que les features saillants)

- Recherche de clusters d'entreprises dans chaque échantillon via machine learning non supervisé (algorithme HDBSCAN)

- Tentatives d'optimisation des hyperparamètres de l'algorithme de recherche de clusters via 3 objectifs: 
    + score de densité inter et intra cluster au delà d'un seuil minimal
    + score de stabilité des clusters (bootstrap)
    + score de significativité des clusters (coefficients de variation des features au sein des clusters)

- Calcul des Zscores pour chaque feature de chaque cluster de chaque échantillon pour caractériser les clusters (faits saillants administratifs et économiques)

- Calcul de similitude (similarité cosinus) entre les Zscores pour former des groupes de clusters similaires entre échantillons et mettre potentiellement en exergue des faits saillants par rapport à l'ensemble de la population d'entreprises immatriculées au Répertoire Sirene de l'Insee 



#### 2.1 Récupération des données open data, apurement et tirage d'échantillons

DuckDB est utilisé pour récupérer les données au format Parquet, tirer les échantillons et calculer une partie des features (les moins complexes), Python est ensuite utilisé pour finir de calculer les features les plus complexes.

Les permaliens des datasets suivants sont utilisés:
- Sirene: Fichier stock établissement historique (format parquet)
- Sirene: Fichier stock unité légale historique (format parquet)

Les choix arbitraires:

- Les codes NAF vides sont supprimés et non imputés
- Le tirage des 500 000 entreprises (nécessaire ensuite pour optimiser les temps de recherche des clusters) est stratifié sur le nombre de périodes des entreprises et leur code secteur d'activité en période courante (c'est à dire leur période actuelle). La sous stratification en nombre de périodes est forcée pour être représentative de vies temporelles d'entreprises diversifiées 
- Les features relatifs aux établissements actifs des entreprises sont imputés par 0 lorsque les entreprises n'ont que des établissements fermés

#### 2.2 Préparation des données avant la recherche des clusters et choix de l'algorithme de recherche des clusters

La méthode Yeo-Johnson (avec standardisation) est utilisée pour transformer les features avant d'appliquer la réduction de leur nombre via l'Analyse en composantes principales.

Plusieurs algortihmes de recherche de clusters ont été testés: K-means, DBSCAN, HDBSCAN.
La représentation graphique des clusters disparate n'allait pas dans le sens d'utiliser K-means et le choix a priori du nombre de clusters n'était pas convaincant, ni le score de validité (score Silhouette).
La littérature semble promouvoir l'efficacité supérieure de l'algorithme HDBSCAN sur DBSCAN et il y a moins d'hyper paramètres à optimiser.

Du fait du nombre d'observations conséquent et de la finalité de l'analyse exploratoire (définir des clusters et estimer les caractéristiques de ces clusters dans la population d'entreprises), les algorithmes de dendogramme et mix de gaussiennes ont été écartés.


#### 2.3 Tentatives d'optimisation des hyperparamètres de HDBSCAN

Les choix arbitraires:
- plusieurs objectifs à essayer d'optimiser simultanément donc la bibliothèque Optuna est choisie
- le choix des hyperparamètres de l'algorithme de recherche de clusters HDBSCAN est restreint au 2-uplet {min_cluster_size; min_samples} sur des intervalles de valeurs aux bornes arbitraires
- la sélection des objectifs de validation du 2-uplet optimisé est complexe:
    + le 1er objectif est le score par défaut  DBCV pour attester des qualités de cohésion entre les entreprises au sein d'un même cluster et de séparation nette des entreprises entre clusters
    + le 2e objectif est là pour s'assurer de la reproductibilité des clusters trouvés avec HDBSCAN. L'approche choisie est le bootstrap (des choix arbitraires sont aussi réalisés sur le nombre et la taille du bootstrap compte tenu des temps de calcul) 
    + le 3e et dernier objectif choisi concerne la significativité des clusters trouvés par HDBSCAN. L'approche choisie est le coefficient de variation moyen des features du cluster et sa minimisation (avec Optuna il faut maximiser l'opposé du coefficient de variation)
- le choix de la règle d'optimisation des objectifs de validation: 
    + score DBCV > 0.1
    + ET au moins 2 clusters
    + ET minimisation de la distance euclidienne entre les cibles parfaites de reproductibilité et significativité et leurs métriques associées calculées sur le cluster
- les tentatives dans la fonction d'optimisation d'Optuna sont limitées à 25 du fait du temps de calcul


#### 2.4 Calcul du zscore, utilisation des poids de tirage et calcul de la similitude entre zscores

Le Zscore permet rapidement de voir quels sont les features sous et sur réprésentés au sein de chaque cluster par rapport à l'échantillon dont ils font partie.

Choix arbitraire: les poids de tirage ne sont utilisés que pour estimer les zscores des features des clusters dans la population générale des entreprises immatriculées au Répertoire Sirene.

Pour le calcul de la similitude entre les zscores des clusters des différents échantillons, c'est la similitude cosinus qui est choisie.

### 3. Outputs

Matrice de similitudes cosinus entre zscores des clusters, appartenance des clusters à tel ou tel échantillon, zscores des clusters

### 4. Interprétations administratives et économiques exploratoires

L'idée est de regarder les groupes de clusters similaires entre échantillons à partir de la similitude cosinus, puis les valeurs des zscores associées aux features (les plus positives et les plus négatives pour mettre en exergue des features saillants), la taille des groupes de clusters et enfin la part pondérée dans la population générale des entreprises immatriculées au Répertoire Sirene.

*Résultats exploratoires réalisés à partir des fichiers open data sur data gouv datant de mars 2026.*

#### 1er groupe de clusters similaires entre les 10 échantillons — entreprises à trajectoire relativement stable et peu dense

zscores

-> très positif :

durée minimale des périodes
durée moyenne des périodes

-> très négatif :

nombre de périodes
changements NAF
densité
nombre maximal d'établissements actifs simultanément

=> persona d'entreprises qui :

changent peu d'activité ;
changent peu de statut ;
ont peu d'établissements ;
connaissent peu d'événements administratifs ;
ont une trajectoire relativement linéaire.

#### 2e groupe de clusters similaires entre les 10 échantillons — entreprises avec de nombreux changements de catégorie juridique

zscores

-> très positif : nombre de changements de catégorie juridique 

=> persona d'entreprises qui ont beaucoup de changements de catégorie juridique.

#### 3e groupe de clusters similaires entre les 10 échantillons — le pendant du groupe précédent entreprises avec très peu de changements de catégorie juridique

zscores

-> très négatif : nombre de changements de catégorie juridique 

=> persona d'entreprises qui ont très peu de changements de catégorie juridique.

#### 4e groupe de clusters similaires entre les 10 échantillons — entreprises peu changeantes avec une durée de vie relativement courte (attention dans la trajectoire observée dans les données du Répertoire Sirene)

zscores

-> très positif: durée minimale des périodes.

-> très négatif :

nombre de périodes
changements NAF 
changements CJ 
durée de vie

#### 5e groupe de clusters similaires entre les 10 échantillons — entreprises avec des trajectoires fortement dynamique

zscores

-> très positif:
nombre de périodes
changements d'état administratif 
changements CJ 
changements NAF

=> persona d'entreprises qui :

beaucoup de changements 
évolution de l'activité 
transformations juridiques 
modifications du statut administratif

#### 6e et dernier groupe de clusters similaires entre les 10 échantillons — pendant du groupe précédent entreprises avec des trajectoires faiblement dynamique

zscores

-> très négatif:
nombre de périodes
changements d'état administratif 
changements CJ 
changements NAF

=> persona d'entreprises qui :

peu de changements 
pas/peu d'évolution de l'activité 
pas/peu de transformations juridiques 
pas/peu modifications du statut administratif

**Attention:
Certaines variables sont mécaniquement ou conceptuellement liées dans la manière dont est constuit l'historique des données du Répertoire Sirene.
Par exemple, une entreprise qui change fréquemment de NAF va probablement générer davantage de périodes.**

Essai de typologie exploratoire correspondante:
| Typologie potentielle                          | Dynamique administrative (changement d'état administratif) | Dynamique sectorielle (changement de secteur d'activité) | Complexité organisationnelle (établissements actifs et fermés) | Temporalité (durée de vie et nombre de périodes)        |
| ---------------------------------------------- | ------------------------ | --------------------- | ---------------------------- | ------------------- |
| **Trajectoire stable**                         | faible                   | faible                | faible                       | périodes longues    |
| **Trajectoire juridiquement évolutive**        | nombreux changements de catégorie juridique                 | variable              | variable                     | variable            |
| **Trajectoire courte**                         | faible                   | faible                | faible                       | durée de vie faible |
| **Trajectoire administrative très dynamique**  | forte                    | forte                 | variable                     | périodes courtes    |
| **Trajectoire organisationnellement complexe** | variable                 | variable              | forte                        | variable            |

**=> Les résultats suggèrent l'existence de configurations récurrentes de trajectoires d'entreprises, caractérisées notamment par leurs dynamiques administrative et sectorielle, leur temporalité et leur complexité organisationnelle. La répétition de profils similaires sur plusieurs échantillons indépendants suggère que ces configurations ne sont pas propres à un échantillon particulier.**


### 5. Repository Structure

```
├── src/
│   ├── chargement_tirage_donnees.py
│   ├── preparation_donnees.py
│   ├── robustesse_significativite_clusters.py
│   └── zscores.py
├── notebook_demo.ipynb
├── requirements.txt
└── README.md
```

### 6. Stack

Python
pandas
numpy
duckdb
scikit-learn
hdbscan
optuna


### 7. Pistes de réflexion

- benchmarker par rapport à un pipeline qui utiliserait les poids de tirage dès le traitement des données (Yeo-Johnson et PCA) pour avoir un résultat vraiment en population générale?
- benchmarker par rapport à un pipeline: données échantillonnées -> traitement des données (Yeo-Johson et PCA) -> calcul de zscores au niveau observation puis de similitudes entre zscores -> clustering sur la métrique similitude cosinus