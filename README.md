# BrainTumorClassification

<!-- TABLE des matières -->
<!-- <details> -->
## Table des matières
  <ol>
    <li>
      <a href="#a-propos-du-projet">À propos du projet</a>
    </li>
    <li><a href="#prealables">Préalables</a></li>
    <li><a href="#Membres">Membres</a></li>
    <li><a href="#Ressources">Ressources</a></li>
  </ol>
<!-- </details> -->

<!-- À propos -->
## A propos du projet
L'objectif est de développer un algorithme capable d'effectuer de la classification sur des images de cerveau ayant un type de tumeur (meningioma, glioma, pituary).

L'atteinte de l'objectif a nécessité une analyse approfondie des données disponibles. 
Suite à l'analyse, nous avons effectuer plusieurs transformations de données, en plus d'augmenter le nombre de données avec des transformations.

Finalement, nous avons pu obtenir des résultats qui ont dépassé nos objectifs de départ.

### Travail réalisé
Le travail réalisé se situe dans les différents notebooks du projet. Les notebooks contiennent plusieurs images pour en faciliter la compréension.
### Forage de données
- ```notebook_partie_1.ipynb``` : Travail préliminaire effectué lors de la première partie du projet.
- ```data_analysis.ipynb``` : Tout le travail relatif à l'analyse des données disponibles.
- ```roi_augmentation.ipynb``` : Travail concernan l'augmentation de la région d'intérêt. Une ébauche d'une étude sur le partitionnement est également comprise, mais elle ne devrait pas être considérée dans le cadre du cours.
- ```DataTransformation.py``` : Contient les classes développées spécifiquement pour effectuer les transformations de données.
- ```data_augmentation.ipynb``` : Expérimentations des transformations développées pour l'augmentation des données, soit des transformations (rotation, réflexion et bruit gaussien).
- ```data_separation.ipynb``` : Séparation des données en Test, Validation et Entraînement. La séparation comprend la partie sur l'égalisation des données et sur l'augmentation des données de l'enemble d'entraînement.
- ```resultats.ipynb``` : Présente les résultats obtenus avec l'algorithme de classification. Ce notebook traite seulement des matrices de confusion. Les autres résultats se trouvent dans ```model_results```.

### Algorithme de classification
- ```models/unet.py``` : Implémentation du UNet.
- ```models/vgg.py``` :  Implémentation du VGGNet.
- ```./model_results``` : Résultats obtenus à la suite des expérimentations.

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- Préalables -->
## Prealables
* python 3.8.10

* pip 22.0.3

* packages
  * ```numpy```
  * ```pandas```
  * ```matplotlib```
  * ```scikit-learn```
  * ```scikit-image```
  * ```pymatreader```
  * ```jupyter```
  * ```jupyterlab```
  * ```tensorflow```
  * ```opencv-python```
  * ```seaborn```

### Pour débuter rapidement :
1. ```python -m venv venv```
2. Windows:  ```.\env\Scripts\activate``` 
3. ```pip install -r requirements.txt```
4. ```jupyter lab```

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- Membres -->
## Membres
|NOM              | CIP       |
|-----------------|-----------|
|Honorine Chantre | CHAH2807  |
|Manon Cottart    | COTM3313  |
|Lucas Gonthier   | GONL3002  |
|Étienne Penelle  | PENE2002  |

<p align="right">(<a href="#top">back to top</a>)</p>

## Ressources

<p align="right">(<a href="#top">back to top</a>)</p>