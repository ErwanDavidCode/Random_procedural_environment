# Random_procedural_environment
This python project aims to generate a semi-random map representing a complex and realistic environment seen from above in 2D. The environment contains:
- different biomes
- adapted elevations
- rivers
- streams
- vegetation

# Installation
- Install Python libraries
```sh
pip install -r requirements.txt
```

# Algorithm configuration
The internal values ​​used for the algorithm can be modified in the main of the `labyrinthe_resolution.py` file. This main is located at the bottom of the code.

This project contains a lot of parameters. I only list here the most important ones that can be modified:

```python
monde = world(130) # size of the world

# Temperatures and precipitation
T_min = 0
T_max = 30  # T max is 30°
P_min = 0
P_max = 100  # P max is 100mm

# Relief and frontier
nbr_cellules = 10 # The more cells there are, the more different biomes there can be. But be careful, it also depends on the size of the Perlin maps of temperatures and precipitations
seuil_visualisation_relief = 0.007  # the higher it is, the less you can see the relief

# Sea settings
# -1=normal / -0.3=Minecraft Zoom (with -1 we have Perlin values ​​between [-1, 1]) / do not do more than 1 (because bezier not defined above)
min_hauteur_terre = -0.4

archipel = False # In True, the map changes to an archipelago.

# river settings
nb_cours_eau = 20

# settings ranges
seuil_plage = (hauteur_mer) + 0.003

# generation
monde.generate(...)
```

# Examples
Here are some screenshots of worlds generated with this code:
![Screenshot of aa continent map](/pictures/continent_01.png)
![Screenshot of a archipelago map](/pictures/archipel_01.png)
