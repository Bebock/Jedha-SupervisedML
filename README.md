# Walmart : Prédiction des ventes hebdomadaires

----

## 1. Overview du projet 

Walmart Inc. est une multinationale de vente au détail qui comprend une chaîne d'hypermarchés, de grands magasins discount et d'épiceries aux États-Unis. La société a été fondée par Sam Walton en 1962.

Le service marketing de Walmart souhaite construire un **modèle de machine learning** capable d'estimer les ventes hebdomadaires dans leurs magasins, avec la meilleure précision possible sur les prédictions faites. Un tel modèle les aiderait à mieux comprendre comment les ventes sont influencées par les indicateurs économiques et pourrait être utilisé pour planifier de futures campagnes de marketing.

Les variables disponibles sont : 
  * L'identifiant du magasin 
  * La date
  * Les ventes hebdomadaires réalisées
  * Si la date est un jour férié ou non
  * La température
  * Le prix du carburant
  * Le Consumer Price Index (CPI) qui est un indicateur qui mesure le changement de prix dans les biens et services essentiels tels que les loyers, la nourriture et l'energie. 
  * Le taux de chômage

----

## 2. Objectifs 

Ce projet est découpé en 3 étapes : 

  * Partie 1 : Analyse Descriptive Exploratoire et pré-traitement des données 
  * Partie 2 : Construire un modèle de régression linéaire (baseline) 
  * Partie 3 : Optimisation du modèle à l'aide de techniques de régularisation (Lasso, Ridge et Elasticnet)

----

## 3. Comment procéder ?

### Pré-requis

Les librairies suivantes sont nécessaires : 

  * General : 
    * warnings
    * math
    * pandas 
    * numpy 
    * scipy 
    * pickle 
    * bioinfokit
  * Graphiques : 
    * matplotlib
    * seaborn 
    * plotly
  * Preprocessing 
    * missingno 
    * fancyimpute 
  * Sélection & évaluation des modèles
    * mlxtend
    * sklearn
    * yellowbrick

### Les fichiers

Le notebook 

----

## 4. Overview des principaux résultats

### Analyse descriptive exploratoire

![image](https://user-images.githubusercontent.com/38078432/186984905-73191b43-bdaf-4ab9-97f6-eb36dc7f941f.png)


### Modélisation Weekly_Sales ~ f(X)



----

## 5. Informations

### Outils

Les notebooks ont été développés avec [Visual Studio Code](https://code.visualstudio.com/). 

### Auteurs & contributeurs

Auteur : 
  * Helene alias [@Bebock](https://github.com/Bebock/)

La dream team :
  * Henri alias [@HenriPuntous](https://github.com/HenriPuntous/)
  * Jean alias [@Chedeta](https://github.com/Chedeta/)
  * Nicolas alias [@NBridelance](https://github.com/NBridelance/)
  
### Sites sources des données

Kaggle competittion : 


----



