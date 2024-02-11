# Random_procedural_environment
Ce projet python a pour but de générer une carte semi-aléatoire représentant un environnement complexe et réaliste vu du dessus en 2D. L'environnement contient des :
- biomes différents
- dénivelés adaptés
- rivières
- fleuves
- végétations

# Installation
- Installer les librairies Python
```sh
pip install -r requirements.txt
```

# Configuration de l'algorithme
Les valeurs internes utilisées pour l'algorithme peuvent être modifiés dans le main du fichier `labyrinthe_resolution.py`. Ce main se situe tout en bas du code.

Ce projet contients énormément de paramètres. Je ne liste ici que les plus importants qui peuvent être modifier :

```python
monde = world(130) # taille du monde

# Températures et précipitation
T_min = 0
T_max = 30  # T max est 30°
P_min = 0
P_max = 100  # P max est 100mm

# Relief et frontières
nbr_cellules = 10 # Plus il y a de cellules, plus il peut y avoir de biomes différents. Mais, attention, ca dépend aussi de la taille des cartes de Perlin de températures et précipitations 
seuil_visualisation_relief = 0.007  # plus c'est élevé moins on vois le relief

# Paramètres mer
# -1=normal / -0.3=Minecraft_zoomé (avec -1 on a des valeurs de Perlin entre [-1 , 1]) / ne pas faire plus que 1 (car bezier non défini au dessus)
min_hauteur_terre = -0.4

archipel = False # En True, la carte passe sous forme d'archipel

# paramètres rivières
nb_cours_eau = 20

# paramètres plages
seuil_plage = (hauteur_mer) + 0.003

# génération
monde.generate(...)
```

# Exemples
Voici quelques captures d'écran de mondes générés avec ce code : 
![Screenshot of aa continent map](/pictures/continent_01.png)
![Screenshot of a archipelago map](/pictures/archipel_01.png)
