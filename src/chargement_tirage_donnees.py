#chargement_tirage_donnees.py
"""
-Chargement des donnees depuis data.gouv.fr: fichiers Sirene donnees historiques niveau unites legales et etablissements (utilisation des permaliens)

-Appurement des donnees pour lesquelles le code NAF est inconnu

-Tirage de 500 000 entreprises et ses établissements, stratification code NAF*nombre de periodes historique de lentreprise, la stratification est faite sur la periode courante des entreprises

-Calcul des poids de tirage (serviront à calculer plus tard les zscores)

-Calcul des features aux niveau unite legale et etablissement:

*unite legale:
-nombre de periodes économiques connues par lentreprise
-nombre de changements detats administratifs (actif, cesse)
-nombre de changements de secteur dactivite
-nombre de changements de catégorie juridique
-duree moyenne des periodes
-duree minimale des periodes
-ecart type des durees des periodes
-duree de vie entreprise
-densite entreprise (nombre periodes/duree de vie)

*etablissement:
-nombre detablissements de lentreprise (actif et ferme)
-nombre detablissements actifs max en simultane
-nombre detablissements actifs moyen
-nombre detablissements fermes
-evolution du nombre detablissements actifs

-Repartition equilibree des 500 000 observations en 10 echantillons (s'assurer de la robustesse de la méthode de clusterisation et temps de calcul acceptable)

"""

import pandas as pd
from sklearn.impute import SimpleImputer


def chargement_tirage(con):
    #permaliens des jeux de donnees Sirene historique sur data.gouv.fr
    #unite legale: https://www.data.gouv.fr/api/1/datasets/r/1b9290ed-d0bc-461f-ba31-0250a99cc140
    #etablissement: https://www.data.gouv.fr/api/1/datasets/r/2b3a0c79-f97b-46b8-ac02-8be6c1f01a8c

    # requete pour constituer les donnees unite legale et les features correspondantes

    query="""
    WITH 
        table_ul_stratifiee_naf AS (
        SELECT siren,
        CASE 
                WHEN SUBSTR(activitePrincipaleUniteLegale,1,2) IN ('01','02','03') THEN 'A'
                WHEN SUBSTR(activitePrincipaleUniteLegale,1,2) IN ('05','06','07','08','09') THEN 'B'
                WHEN SUBSTR(activitePrincipaleUniteLegale,1,2) IN ('10','11','12','13','14','15','16','17','18',
                '19','20','21','22','23','24','25','26','27','28','29','30','31','32','33') THEN 'C'
                WHEN SUBSTR(activitePrincipaleUniteLegale,1,2) IN ('35') THEN 'D'
                WHEN SUBSTR(activitePrincipaleUniteLegale,1,2) IN ('36','37','38','39') THEN 'E'
                WHEN SUBSTR(activitePrincipaleUniteLegale,1,2) IN ('41','42','43') THEN 'F'
                WHEN SUBSTR(activitePrincipaleUniteLegale,1,2) IN ('45','46','47') THEN 'G'
                WHEN SUBSTR(activitePrincipaleUniteLegale,1,2) IN ('49','50','51','52','53') THEN 'H'
                WHEN SUBSTR(activitePrincipaleUniteLegale,1,2) IN ('55','56') THEN 'I'
                WHEN SUBSTR(activitePrincipaleUniteLegale,1,2) IN ('58','59','60','61','62','63') THEN 'J'
                WHEN SUBSTR(activitePrincipaleUniteLegale,1,2) IN ('64','65','66') THEN 'K'
                WHEN SUBSTR(activitePrincipaleUniteLegale,1,2) IN ('68') THEN 'L'
                WHEN SUBSTR(activitePrincipaleUniteLegale,1,2) IN ('69','70','71','72','73','74','75') THEN 'M'
                WHEN SUBSTR(activitePrincipaleUniteLegale,1,2) IN ('77','78','79','80','81','82') THEN 'N'
                WHEN SUBSTR(activitePrincipaleUniteLegale,1,2) IN ('84') THEN 'O'
                WHEN SUBSTR(activitePrincipaleUniteLegale,1,2) IN ('85') THEN 'P'
                WHEN SUBSTR(activitePrincipaleUniteLegale,1,2) IN ('86','87','88') THEN 'Q'
                WHEN SUBSTR(activitePrincipaleUniteLegale,1,2) IN ('90','91','92','93') THEN 'R'
                WHEN SUBSTR(activitePrincipaleUniteLegale,1,2) IN ('94','95','96') THEN 'S'
                WHEN SUBSTR(activitePrincipaleUniteLegale,1,2) IN ('97','98') THEN 'T'
                WHEN SUBSTR(activitePrincipaleUniteLegale,1,2) IN ('99') THEN 'U'
                ELSE NULL
        END AS naf_section,
        FROM READ_PARQUET('https://www.data.gouv.fr/api/1/datasets/r/1b9290ed-d0bc-461f-ba31-0250a99cc140')
        WHERE dateFin IS NULL AND naf_section IS NOT NULL
        GROUP BY siren,naf_section
        ),

     table_ul_stratifiee_nbperiode AS (
       SELECT siren,
              CASE 
                     WHEN count(*)=1 THEN '1'
                     WHEN count(*)<=3 THEN '2-3'
                     WHEN count(*)<=6 THEN '4-6'
                     WHEN count(*)<=10 THEN '7-10' 
                     ELSE '10+'
              END AS nb_periodes, 
       FROM READ_PARQUET('https://www.data.gouv.fr/api/1/datasets/r/1b9290ed-d0bc-461f-ba31-0250a99cc140')
       WHERE NOT (etatadministratifunitelegale='C' AND datedebut IS NULL AND datefin IS NULL)
       GROUP BY siren
       ),

     table_ul_stratifiee AS (
        SELECT table_ul_stratifiee_nbperiode.siren,
               table_ul_stratifiee_nbperiode.nb_periodes,
               table_ul_stratifiee_naf.naf_section,
        FROM table_ul_stratifiee_nbperiode
        INNER JOIN table_ul_stratifiee_naf ON table_ul_stratifiee_nbperiode.siren=table_ul_stratifiee_naf.siren
        ),

      table_ul_stratifiee_ponderee AS (
        SELECT *,
              ROW_NUMBER() OVER (
                     PARTITION BY nb_periodes,naf_section
                     ORDER BY RANDOM()
              ) AS rn,
              (
              COUNT(*) OVER (
                     PARTITION BY nb_periodes,naf_section
              ) 
              /
              COUNT(*) OVER (
                     PARTITION BY nb_periodes
              ) 
              ) AS naf_poids,
              (
              COUNT(*) OVER () 
              /
              COUNT(*) OVER (
                     PARTITION BY nb_periodes,naf_section
              ) 
              ) AS poids_tirage              
        FROM table_ul_stratifiee
        ),

       table_ul_sample_step0 AS (
        SELECT *,
        CAST(naf_poids * 
             CASE 
                WHEN nb_periodes = '1' THEN 100000
                WHEN nb_periodes = '2-3' THEN 125000
                WHEN nb_periodes = '4-6' THEN 125000
                WHEN nb_periodes = '7-10' THEN 100000
                ELSE 50000
             END AS INTEGER
            ) AS quota,
       ((rn - 1) % 10) + 1 AS numero_echantillon
       FROM table_ul_stratifiee_ponderee
       WHERE rn <= quota
       ),
       
       table_ul_sample_step1 AS (
        SELECT table_ul_sample_step0.siren, 
              changementEtatAdministratifUniteLegale, 
              changementActivitePrincipaleUniteLegale, 
              changementCategorieJuridiqueUniteLegale , 
              IF(dateFin IS NULL,DATE_DIFF('day',dateDebut,TODAY()),DATE_DIFF('day',dateDebut,dateFin)) AS duree_periode,
        FROM READ_PARQUET('https://www.data.gouv.fr/api/1/datasets/r/1b9290ed-d0bc-461f-ba31-0250a99cc140') AS table_ul_complete 
        RIGHT JOIN table_ul_sample_step0 ON table_ul_complete.siren=table_ul_sample_step0.siren
       ),
       
      table_ul_sample_step2 AS (  
        SELECT siren,
        COUNT(*) AS nb_periodes,
        SUM(changementEtatAdministratifUniteLegale) AS nb_chgt_etatadm,
        SUM(changementActivitePrincipaleUniteLegale) AS nb_chgt_naf,
        SUM(changementCategorieJuridiqueUniteLegale) AS nb_chgt_cj,
        MEAN(duree_periode) AS mean_duree_periodes,
        SEM(duree_periode) AS std_duree_periodes,
        MIN(duree_periode) AS min_duree_periodes,
        MAX(duree_periode) AS max_duree_periodes,
        SUM(duree_periode) AS duree_vie,
        nb_periodes/duree_vie AS densite,
        FROM table_ul_sample_step1
        GROUP BY siren
       ),

      table_ul_sample_step3 AS (
        SELECT table_ul_sample_step2.*, 
        siret, 
        etatadministratifetablissement,
        IF(dateFin IS NULL,TODAY(),dateFin) as datefin
        FROM READ_PARQUET('https://www.data.gouv.fr/api/1/datasets/r/2b3a0c79-f97b-46b8-ac02-8be6c1f01a8c') AS table_etab_complete 
        RIGHT JOIN table_ul_sample_step2 ON table_etab_complete.siren=table_ul_sample_step2.siren
      )


       SELECT table_ul_sample_step3.*,
       table_ul_sample_step0.poids_tirage,
       table_ul_sample_step0.numero_echantillon
       FROM table_ul_sample_step3 
       INNER JOIN table_ul_sample_step0 ON table_ul_sample_step3.siren=table_ul_sample_step0.siren
      
    """
    df = con.execute(query).fetchdf()

    #calcul des features niveau etablissement
    nb_etab=(df.groupby("siren"))["siret"].nunique()
    nb_etab_actifs_periode=(df[df["etatAdministratifEtablissement"].astype("string")=="A"].groupby(["siren","datefin"]))["siret"].nunique()
    nb_etab_max_simul=nb_etab_actifs_periode.groupby("siren").max()
    nb_etab_moy_simul=nb_etab_actifs_periode.groupby("siren").mean().round()
    nb_etab_actifs_periode_first=nb_etab_actifs_periode.groupby("siren").first()
    nb_etab_actifs_periode_last=nb_etab_actifs_periode.groupby("siren").last()
    growth_rate_etab_actifs= nb_etab_actifs_periode_last / nb_etab_actifs_periode_first
    nb_etab_ferme=(df[df["etatAdministratifEtablissement"].astype("string")=="F"].groupby("siren"))["siret"].nunique()

    features_siret = pd.DataFrame({
            "nb_etab": nb_etab,
            "nb_etab_actif_max_simul": nb_etab_max_simul,
            "nb_etab_actif_moy_simul": nb_etab_moy_simul,
            "nb_etab_ferme": nb_etab_ferme,
            "evol_nb_etab_actifs": growth_rate_etab_actifs
        })
    features_siret.reset_index(inplace=True)

    #imputation par la valeur 0 du nombre d'établissements actifs max, moyen et son évolution si l'entreprise n'a qu'un seul établissement fermé
    features_siret_impute=pd.DataFrame(SimpleImputer(strategy="constant",fill_value=0).fit_transform(features_siret), 
                                    columns=features_siret.columns.tolist())

    features_siren=df.drop(columns=["siret", "etatAdministratifEtablissement","datefin"])
    features_siren=features_siren.drop_duplicates(subset=["siren"])

    #appariemment features siren et features siret
    features = pd.merge(features_siren,  
                    features_siret_impute, 
                    on="siren", 
                    how="inner")
    features['nb_etab'] = features['nb_etab'].astype(int)
    features['nb_etab_actif_max_simul'] = features['nb_etab_actif_max_simul'].astype(float)
    features['nb_etab_actif_moy_simul'] = features['nb_etab_actif_moy_simul'].astype(float)
    features['nb_etab_ferme'] = features['nb_etab_ferme'].astype(float)
    features['evol_nb_etab_actifs'] = features['evol_nb_etab_actifs'].astype(float)
    #repartition des features en 10 echantillons
    return [features[features["numero_echantillon"] == i] for i in range(1, 11)]

