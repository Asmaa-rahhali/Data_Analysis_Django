# Analyse de Données et Visualisation

## Présentation

Ce projet propose une interface interactive pour l'analyse de données et la visualisation. Il permet aux utilisateurs d'importer des fichiers Excel, d'explorer les données, de calculer des statistiques et de générer divers types de visualisations et de distributions de probabilités.

---

## Fonctionnalités

- **Importation et aperçu des données :** Affichage des premières lignes pour un aperçu rapide.  
- **Statistiques descriptives :** Calcul de la moyenne, médiane, mode, variance, écart type et étendue.
- **Visualisations interactives :** Histogramme, Boxplot, KDE Plot, Scatterplot, Heatmap,Countplot, Pie Chart, Barplot et Violin Plot.  
- **Probabilités :** Analyse des distributions discrètes (Bernoulli, Binomiale, Poisson, Uniforme) et continues (Uniforme, Exponentielle, Normale). 
- **Choix personnalisés :** L'utilisateur peut sélectionner les colonnes et les conditions pour ses visualisations et calculs.  

---

## Aperçu des fonctionnalités

### Aperçu des données
![Aperçu](./captures/aperçu.png)

### Calcul des Statistiques
![Statistiques](./captures/stats.png)

### Visualisations
Exemples :  
- **Heatmap :** ![Heatmap](./captures/heatmap.png)  
- **Histogramme :** ![Histogramme](./captures/histplot.png)  
- **Violin Plot :** ![Violin](./captures/violin.png)  
- **Pie Chart :** ![Pie Chart](./captures/pie.png)  

### Probabilités 
![Probas-d](./captures/probas.png) 
![Probas-c](./captures/probas-c.png)
Exemples :  
- **Bernoulli :** ![Bernoulli](./captures/bernoulli.png)  
- **Uniforme Discontinue :** ![Uniforme Discontinue](./captures/unif-disc.png)  
- **Normale Gaussienne :** ![Normale](./captures/gauss.png)  

---

## Prérequis

- Python 3.12.5 ou plus récent  
- Django 5.1.4 ou plus récent  
- Bibliothèques nécessaires (voir `requirements.txt`)  

---

## Installation

1. Clonez le dépôt :
   ```bash
   git clone https://github.com/Asmaa-rahhali/Data_Analysis_Django

2. Créez un environnement virtuel :
   ```bash
   python -m venv env
   ```
3. Activez l'environnement virtuel :
   ```bash
   .\env\Scripts\activate
   ```
4. Installez les dépendances :
   ```bash
   pip install -r requirements.txt
   ```
---

### Contributors

- **EL MALKY Douaa**
- **RAHHALI Asmaa**