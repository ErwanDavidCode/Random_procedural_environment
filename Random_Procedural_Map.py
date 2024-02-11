# -*- coding: utf-8 -*-

import numpy as np
import random
import matplotlib.pyplot as plt


# CLASSE WOLRD

class world():
    def __init__(self, taille):
        self.N = taille
        self.matrice = [[[] for i in range(self.N)] for j in range(self.N)]
        self.matrice_ajout = [[[]
                               for i in range(self.N)] for j in range(self.N)]

    def generate(self, precision, nombre_de_centroides, mesh_H, nb_octave_H, mesh_H_bruit, nb_octave_H_bruit, variation_brutale_bruit, min_bruit, max_bruit, precision_bruit,  mesh_H_errosion, nb_octave_H_errosion, variation_brutale_errosion, min_errosion, max_errosion, precision_errosion, min_val, mesh_T_P, nb_octave_T_P, hauteur_max, variation_brutale, mesh_blur, boundary_displacement, precipitation_min, precipitation_max, temperature_min, temperature_max, seuil, taille_noyeau_cellule, taille_noyeau_biome, hauteur_max_riviere, hauteur_mer, facteur, nb_cours_eau, hauteur_cours_eau, taille_noyau_plage, seuil_plage, terre, bool_archipel):
        self.hauteur_mer = hauteur_mer
        self.nombre_de_centroides = nombre_de_centroides
        self.nombre_de_centroides = nombre_de_centroides
        self.seuil = seuil

        # =============================================================================
        #  Génération des cellules
        # =============================================================================
        print("Génération des cellules")
        print("...")
        new_liste_centroides = generation_centroides_bis(
            self.N, self.nombre_de_centroides)  # le nouveau truc avec la physique
        self.liste_coord_cellules_origine = voronoi(self.N, new_liste_centroides)[
            0]  # on créer les cellule avec les bons blocs
        self.liste_coord_cellules = reconstruction_voronoi(nombre_de_centroides, self.liste_coord_cellules_origine, blur_3(
            self.N, mesh_H, voronoi(self.N, new_liste_centroides)[1], mesh_blur, boundary_displacement))

        # =============================================================================
        # Génération des biomes pour chaque cellule
        # =============================================================================
        print("Génération des biomes pour chaque cellule ")
        print("...")
        T_P = generate_t_p(self.N, mesh_T_P, nb_octave_T_P, precipitation_min,
                           precipitation_max, temperature_min, temperature_max, precision)
        # on récupère une température et précipitation moyenne pour chaque cellule ET on ajoute le nom du biome :[[[centroid],T,P,"nom],...]
        self.liste_T_P_cellules = moyenne_cellule(
            T_P, self.liste_coord_cellules, self.nombre_de_centroides)
        self.carte_precipitation = T_P[1]
        self.carte_temperature = T_P[0]
        # plt.imshow(self.carte_precipitation)
        # plt.show()
        # plt.imshow(self.carte_temperature)
        # plt.show()

        # =============================================================================
        # Génération des hauteurs ajustées
        # =============================================================================
        print("Génération des hauteurs")
        print("...")

        self.H = generate_h(self.N, mesh_H, nb_octave_H,
                            variation_brutale, min_val, precision)[0]
        self.H_bruit = generate_h_bruit(self.N, mesh_H_bruit, nb_octave_H_bruit,
                                        variation_brutale_bruit, min_bruit, max_bruit, precision_bruit)[0]
        self.H_errosion = generate_h_bruit(self.N, mesh_H_errosion, nb_octave_H_errosion,
                                           variation_brutale_errosion, min_errosion, max_errosion, precision_errosion)[0]

        # plt.imshow(self.H)
        # plt.show()
        # plt.imshow(self.H_bruit)
        # plt.show()
        # plt.imshow(self.H_errosion)
        # plt.show()

        print("Ajustement des hauteurs")
        print("...")
        self.carte_hauteur = appliquer_filtre_2(
            self.N, self.H, self.H_errosion, self.liste_T_P_cellules, self.liste_coord_cellules, self.hauteur_mer, terre)

        self.matrice = associate(self.N, self.liste_T_P_cellules, self.liste_coord_cellules,
                                 self.carte_temperature, self.carte_precipitation)
        # si on met archipel a vrai on créer un archipel
        nvle_matrice_hauteur_avec_eau = eau(
            self.N, self.carte_hauteur, self.hauteur_mer, self.matrice, facteur, archipel=archipel)
        self.matrice, self.carte_hauteur = nvle_matrice_hauteur_avec_eau[
            0], nvle_matrice_hauteur_avec_eau[1]

        self.carte_hauteur += self.H_bruit

        # =============================================================================
        # Génération des rivières
        # =============================================================================
        print("Génération des rivières")
        print("...")
        rivière = riviere(self.N, taille_noyeau_cellule, taille_noyeau_biome,
                          self.matrice, self.carte_hauteur, hauteur_max_riviere, hauteur_mer)
        self.matrice, self.carte_hauteur = rivière[0], rivière[1]
        self.matrice = cours_eau(
            self.N, self.matrice, self.carte_hauteur, nb_cours_eau, hauteur_cours_eau)

        # =============================================================================
        # Génération des plages
        # =============================================================================
        print("Génération des plages")
        print("...")
        self.matrice = plage(self.N, self.matrice, self.carte_hauteur,
                             hauteur_mer, taille_noyau_plage, seuil_plage)

        # =============================================================================
        # Génération des arbres
        # =============================================================================
        print("Génération des arbres")
        print("...")
        self.matrice_ajout = arbres(self.N, self.matrice_ajout, self.matrice)

    def visualise(self):
        print("Visualise")
        print("...")

        random_couleur = generation_liste_couleurs(self.nombre_de_centroides)
        # visualise_voronoi(self.N, self.nombre_de_centroides, self.liste_coord_cellules_origine, self.liste_coord_cellules, random_couleur)
        # visualise_biomes(self.N, self.liste_T_P_cellules, self.carte_hauteur, self.hauteur_mer, self.liste_coord_cellules, self.nombre_de_centroides, self.matrice, random_couleur)
        visualise_map(self.N, self.matrice, self.matrice_ajout,
                      self.carte_hauteur, self.H, self.hauteur_mer, self.seuil)


# =============================================================================
# Fonctions
# =============================================================================


def copy_list(list):
    list_bis = []
    for i in range(len(list)):
        list_bis.append(list[i])
    return list_bis


def find_nearest_point(point_list, point, distance, give_indice=False, give_distance=False):
    '''ENTREE : liste de points,le point,fonction de distance, renvoyer avec la distance 
    au point le plus proche ou non
    SORTIE: coordonnees du point le plus proche, (indice dans la liste de ce point) (distance)'''
    nearest_point = [point_list[0], distance(point_list[0], point)]
    indice = 0
    for i in range(1, len(point_list)):
        checked_distance = distance(point_list[i], point)
        if nearest_point[1] > checked_distance:
            nearest_point = [point_list[i], checked_distance]
            indice = i
    if give_distance and not (give_indice):
        return nearest_point[0], nearest_point[1]
    elif give_distance and give_indice:
        return nearest_point[0], indice, nearest_point[1]
    elif give_indice and not (give_distance):
        return nearest_point[0], indice
    else:
        return nearest_point[0]


def distance(liste_coord1, liste_coord2):
    '''ENTREE : [x_point_1,y_point_1],[x_point_2,y_point_2]
    SORTIE : distance'''
    x_1 = liste_coord1[0]
    y_1 = liste_coord1[1]
    x_2 = liste_coord2[0]
    y_2 = liste_coord2[1]
    return np.sqrt((x_1-x_2)**2+(y_1-y_2)**2)

def generation_centroides_bis(N, nombre_centroides, facteur_pas=0.1, facteur_step=5):
    '''ENTREE : dimension du carre, nombre de centroides
    SORTIE : liste de centroides appartenant à la map de taille n [coord_c_1,coord_c_2,...] avec coord_c_1 = [x,y]'''

    def cost_function(solution, N, nombre_centroides):
        '''ENTREE : solution generee
        SORTIE : coût de cette solution à minimiser : energie potentielle
        NOTE : expression du potentiel de Lennard-Jones'''
        cost = 0
        nbr_pts = len(solution)
        for i in range(nbr_pts-1):
            for j in range(i+1, nbr_pts):
                d_i_j = distance(solution[i], solution[j])
                # critere de distance entre atomes
                if d_i_j < 2*N/np.sqrt(nombre_centroides):
                    epsilon = 10  # choix
                    R_0 = N/np.sqrt(nombre_centroides)
                    sigma = epsilon*R_0/(2**(1/6))
                    cost += 4*epsilon*((sigma/d_i_j)**12-(sigma/d_i_j)**6)
        return cost

    def mutate(solution, N):
        '''ENTREE : solution generee,dimension de la map
        SORTIE : solution mutee (points deplaces)'''
        nbr_pts = len(solution)
        pas_max = N/np.sqrt(nbr_pts)*facteur_pas  # arbitraire
        new_solution = []
        for i in range(nbr_pts):
            x, y = solution[i]
            delta_x, delta_y = np.random.uniform(-pas_max, pas_max, 2)
            x_m, y_m = (x+delta_x), (y+delta_y)
            # si depasse bord de map :
            if x_m >= N:
                x_m = N-1
            elif x_m < 0:
                x_m = 0
            if y_m >= N:
                y_m = N-1
            elif y_m < 0:
                y_m = 0
            new_solution.append([x_m, y_m])
        return new_solution

    # nombre de steps:
    nbre_steps = int(facteur_step*nombre_centroides)

    # nombre de solution mutées generée à chaque step
    nbre_mutations = 5

    # solution initiale : (random dans cases...)
    solution = []
    nbr_centroides_non_random = int(np.sqrt(nombre_centroides))**2
    pas_cadrillage = int(N/np.sqrt(nbr_centroides_non_random))
    for i in range(int(np.sqrt(nbr_centroides_non_random))):
        for j in range(int(np.sqrt(nbr_centroides_non_random))):
            x = np.random.uniform(i*pas_cadrillage, (i+1)*pas_cadrillage)
            y = np.random.uniform(j*pas_cadrillage, (j+1)*pas_cadrillage)
            solution.append([x, y])
    for i in range(nombre_centroides-len(solution)):
        x, y = np.random.uniform(0, N, 2)
        solution.append([x, y])

    # Pour un nombre de steps:
    for i in range(nbre_steps):
        # cout de la solution initiale :
        cost_best_sol = cost_function(solution, N, nombre_centroides)

        # generation des solution mutees :
        new_best_solution = solution
        new_best_cost = cost_best_sol
        for j in range(nbre_mutations):
            new_solution = mutate(solution, N)
            new_cost = cost_function(new_solution, N, nombre_centroides)
            if new_cost < new_best_cost:
                new_best_cost = new_cost
                new_best_solution = new_solution

        # update de la meilleure solution
        solution = new_best_solution

    # discretisation de la meilleure solution:
    for i in range(len(solution)):
        for j in range(2):
            solution[i][j] = int(solution[i][j])

    # renvoit de la meilleure solution
    return solution


def reconstruction_voronoi(nb_centroid, liste_coord_cellules_origine, matrice):
    """permet de retrouver SORTIE : [[coord_centroid,coord_bloc1,coord_bloc2],[#une autre suface]] avec coord_centroid = [x,y]
    avec en ENTREE : matrice (N,N) ou chaque élement est le uméro de la cellule où il appartient"""
    liste_coord_cellules_total = []

    for i in range(nb_centroid):
        liste_coord_cellules = []
        # on ajoute les coord du centroids
        liste_coord_cellules.append(liste_coord_cellules_origine[i][0])

        # on parcourt tous les indices dans l'ordre
        liste = np.where(matrice == i)
        # ce sont les listes des indices (x,y)
        ind_x, ind_y = liste[0], liste[1]
        for ii in range(len(ind_x)):
            liste_coord_cellules.append([ind_x[ii], ind_y[ii]])

        liste_coord_cellules_total.append(liste_coord_cellules)

    return liste_coord_cellules_total


def voronoi(N, liste_centroides):
    '''ENTREE : dimension du carre,liste des centroides
    SORTIE : [[coord_centroid,coord_bloc1,coord_bloc2],[#une autre suface]] avec coord_centroid = [x,y]
    Associe aussi un id unique à chaque cellule, Cette fonction renvoie une matrice (N,N) ou chaque bloc port l'id de sa cellule'''
    # creation de la liste de sortie avec centroides en premières coordonnées
    liste_sortie = []
    matrice_id_cellule = np.zeros((N, N))

    for i in range(len(liste_centroides)):
        liste_sortie.append([liste_centroides[i]])

    # attribution des points de la map aux surfaces reperees par leur centroide
    for i in range(N):
        for j in range(N):
            centroid, indice_c = find_nearest_point(
                liste_centroides, [i, j], distance, give_indice=True)

            if [i, j] != centroid:

                # attribuer le point au bon centroide :
                liste_sortie[indice_c].append([i, j])

            # construction maatrice ou chaque bloc porte l'indice de sa cellule
            matrice_id_cellule[i][j] = indice_c

    return liste_sortie, matrice_id_cellule


def average_points_int(liste_points):
    '''ENTREE : [[x_1,y_1],[x_2,y_2],...] liste des points
    SORTIE : [x_average,y_average] point moyen entier'''
    n = len(liste_points)
    x_m, y_m = 0, 0
    for i in range(n):
        x_i, y_i = liste_points[i][0], liste_points[i][1]
        x_m += x_i
        y_m += y_i
    x_m = int(x_m/n)
    y_m = int(y_m/n)
    return [x_m, y_m]


# ==============================================================================================
# Biomes :
# ==============================================================================================

def appartient_polygon(liste_sommets, point):
    '''ENTREE : liste_sommets : [[x_sommet_1,y_sommet_1],[,],...] , [x_point,y_point]
    Attention : sommets dans l'ordre de parcours des segements du polygone
    SORTIE : booleen appartenance au polygone

    NE MARCHE PAS'''
    n = len(liste_sommets)
    X = []
    Y = []
    x, y = point[0], point[1]
    nbre_segments_coupes = 0
    for i in range(n):
        X.append(liste_sommets[i][0])
        Y.append(liste_sommets[i][1])
    # demi droite coupe un segment ? :
    for i in range(n-1):
        if (X[i]-x) > 0 or (X[i+1]-x) > 0:  # segment a droite ou partiellement a droite du point
            if (Y[i]-y)*(Y[i+1]-y) < 0:  # demi droite horizontale passant par le point coupe le segment
                nbre_segments_coupes += 1
    if nbre_segments_coupes % 2 == 0:
        return False
    return True


def is_point_in_polygon(x, y, polygon):
    """sortie = booléen si est dans le polygone"""
    n = len(polygon)
    inside = False
    p1x, p1y = polygon[0]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        x_inters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= x_inters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside


def visualise_graph_biomes(liste_biomes):
    '''ENTREE : [[liste_sommets_polygon_1,name_1],[[liste_sommmets_polygon_2],name_2],...] 
    liste_sommets : [[x_sommet_1,y_sommet_1],[,],...]
    Attention : sommets dans l'ordre de parcours des segements du polygone
    SORTIE : affichage matplotlib des polygons'''
    for i in range(len(liste_biomes)):
        text = liste_biomes[i][1]
        X_plot = []
        Y_plot = []
        for j in range(len(liste_biomes[i][0])):
            X_plot.append(liste_biomes[i][0][j][0])
            Y_plot.append(liste_biomes[i][0][j][1])
        X_plot.append(X_plot[0])
        Y_plot.append(Y_plot[0])
        plt.plot(X_plot, Y_plot, label=text)

        x_text = average_points_int(liste_biomes[i][0])[0]
        y_text = average_points_int(liste_biomes[i][0])[1]
        plt.text(x_text, y_text, str(
            liste_biomes[i][1]), horizontalalignment='center', verticalalignment='center')

    # visualiser les points des polygones
    for i in range(len(liste_points)):
        x = liste_points[i][0]
        y = liste_points[i][1]
        plt.plot(x, y, "o")
        plt.text(x, y, str(i+1))
    # plt.legend()
    plt.show()


def biome(T, P, Liste_biomes):
    '''ENTREE : Temperature [°C], Precipitation [cm]
    SORTIE : Nom du biome correspondant (chaine de caracteres)'''
    # test d'appartenance:
    for i in range(len(Liste_biomes)):
        if is_point_in_polygon(T, P, Liste_biomes[i][0]):
            return Liste_biomes[i][1]
    return "undefined biome"


def get_bloc(T, P, Liste_bloc):
    '''ENTREE : Temperature [°C], Precipitation [cm]
    SORTIE : Nom du bloc et couleur du bloc correspondant (chaine de caracteres)'''
    # test d'appartenance:
    for i in range(len(Liste_bloc)):
        if is_point_in_polygon(T, P, Liste_bloc[i][0]):
            return [Liste_bloc[i][1], Liste_bloc[i][2]]
    return "undefined bloc"


# ==============================================================================================
# Définition des biomes :
# ==============================================================================================

# liste des points définissant les sommets des polygones de biomes
# jusqu'à limite + ou - 1 pour que les points limites soient inclus
p1 = [-1, 101]
p2 = [4, 101]
p3 = [11, 101]
p20 = [20, 101]
p4 = [31, 101]
p5 = [-1, 72]
p6 = [8, 56]
p7 = [17, 78]
p8 = [31, 80]
p9 = [4.5, 45]
p10 = [14, 22]
p11 = [31, 40]
p12 = [5, 9]
p13 = [12, 10]
p14 = [23, 38]
p15 = [31, 50]
p16 = [-1, -1]
p17 = [4, -1]
p18 = [13, -1]
p19 = [31, -1]

# liste des points pour pouvoir les afficher sur le plot
liste_points = [p1, p2, p3, p4, p5, p6, p7, p8, p9, p10,
                p11, p12, p13, p14, p15, p16, p17, p18, p19, p20]

# liste des polygones qui représentent les biomes
# ORDRE : POINTS, Type, Couleur1, Couleur2, Couleur3, Type arbre, densité arbre, couleur point arbre, Type buisson, densité buisson
polygon_0 = [[p1, p2, p6, p9, p5], "Taiga", np.array([(11, 102, 89)]), np.array([(7, 86, 73)]), np.array(
    [(3, 70, 57)]), "Sapin", "Elevée", "blue", "Herbe", "Moyenne"]  # forets de pins beaucoup de dénivelé
polygon_1 = [[p2, p3, p6], "Jungle", np.array([92, 176, 62]), np.array([76, 160, 46]), np.array(
    [60, 144, 30]), "Dinizia", "Moyenne", "red", "Herbe", "Elevée"]  # tout plat avec des petites bosses localement (dans l'eau)
polygon_2 = [[p20, p4, p8, p7], "Marais", np.array([111, 129, 105]), np.array(
    [101, 115, 92]), np.array([84, 100, 74]), "Aulne", "Faible",  "black", "Herbe", "Faible"]  # dénivelé
polygon_3 = [[p6, p3, p20, p7, p10], "Foret",  np.array([100, 176, 36]), np.array([90, 152, 30]), np.array(
    [70, 124, 25]), "Chêne", "Elevée", "white", "Herbe", "Moyenne"]  # moyen dénivelé
polygon_4 = [[p7, p8, p11, p15, p14, p10], "Plaine",  np.array([124, 176, 56]), np.array([108, 152, 48]), np.array(
    [88, 124, 41]), "Chêne", "Faible", "white", "Herbe", "Moyenne"]  # pas beaucoup de dénivelé
polygon_5 = [[p5, p9, p12, p17, p16], "Pic", np.array([253, 253, 253]), np.array([217, 217, 217]), np.array(
    [178, 178, 178]), "Sapin_gelé", "Très Faible", "blue", "Herbe_gelée", "Moyenne"]  # énorme dénivelé
polygon_6 = [[p9, p6, p10, p13, p18, p17, p12], "Montagnes", np.array([111, 111, 111]), np.array([95, 95, 95]), np.array(
    [78, 78, 78]), "Sapin", "Faible", "blue", "Herbe_gelée", "Moyenne"]  # tout plat avec des gros plateaux
polygon_7 = [[p10, p14, p18, p13], "Badland", np.array([215, 126, 42]), np.array([183, 108, 43]), np.array(
    [151, 88, 37]), "Cactus", "Très faible", "green", "Herbe_sèche", "Moyenne"]  # tout plat avec des motagnes fréquentes
polygon_8 = [[p14, p15, p19, p18], "Desert", np.array([243, 229, 161]), np.array([210, 199, 139]), np.array(
    [172, 162, 115]), "Cactus", "Faible", "green", "Herbe_sèche", "Moyenne"]  # assez plat
polygon_9 = [[], "Plage", np.array([243, 229, 161]), np.array([210, 199, 139]), np.array(
    [172, 162, 115]), "Cactus", "Faible", "green", "Herbe_sèche", "Très Faible"]  # en bord d'eau

# creation de la liste pour manipulation plus simple
Liste_biomes = [polygon_0, polygon_1, polygon_2, polygon_3,
                polygon_4, polygon_5, polygon_6, polygon_7, polygon_8, polygon_9]

# liste de la végétation
Chêne = ["Chêne", "Moyen", np.array([(0, 122, 0)]), np.array(
    [(0, 104, 0)]), np.array([(0, 85, 0)])]
Sapin = ["Sapin", "Moyen", np.array([(0, 104, 0)]), np.array(
    [(0, 85, 0)]), np.array([(0, 65, 0)])]
Aulne = ["Aulne", "Grand", np.array([(92, 110, 40)]), np.array(
    [(80, 95, 35)]), np.array([(70, 78, 28)])]
Sapin_gelé = ["Sapin_gelé", "Moyen", np.array([(157, 157, 250)]), np.array(
    [(135, 135, 216)]), np.array([(110, 110, 176)])]
Dinizia = ["Dinizia", "Grand", np.array([(39, 150, 9)]), np.array(
    [(27, 130, 7)]), np.array([(15, 110, 6)])]
Cactus = ["Cactus", "Petit", np.array([(0, 122, 0)]), np.array(
    [(0, 122, 0)]), np.array([(0, 122, 0)])]
Herbe = ["Herbe", "Petit", np.array([(0, 122, 0)]), np.array(
    [(0, 122, 0)]), np.array([(0, 122, 0)])]
Herbe_sèche = ["Herbe_sèche", "Petit", np.array([(140, 117, 71)]), np.array(
    [(140, 117, 71)]), np.array([(140, 117, 71)])]
Herbe_gelée = ["Herbe_gelée", "Petit", np.array([(157, 157, 250)]), np.array(
    [(157, 157, 250)]), np.array([(157, 157, 250)])]

Liste_arbre = [Chêne, Sapin, Aulne, Sapin_gelé, Dinizia, Cactus]
Liste_herbe = [Herbe, Herbe_sèche, Herbe_gelée]

# liste des points définissant les sommets des polygones de blloc
# jusqu'à limite + ou - 1 pour que les points limites soient inclus
p1 = [-1, 101]
p2 = [4, 101]
p3 = [11, 101]
p20 = [20, 101]
p4 = [31, 101]
p5 = [-1, 72]
p6 = [8, 56]
p7 = [17, 78]
p8 = [31, 80]
p9 = [4.5, 45]
p10 = [14, 22]
p11 = [31, 40]
p12 = [5, 9]
p13 = [12, 10]
p14 = [23, 38]
p15 = [31, 50]
p16 = [-1, -1]
p17 = [4, -1]
p18 = [13, -1]
p19 = [31, -1]

# liste des points pour pouvoir les afficher sur le plot
liste_points = [p1, p2, p3, p4, p5, p6, p7, p8, p9, p10,
                p11, p12, p13, p14, p15, p16, p17, p18, p19, p20]


# liste des polygones qui représentent les biomes
polygon_0 = [[p1, p2, p6, p9, p5], "Herbe gelée", np.array(
    [(31, 160, 85)])]  # forets de pins beaucoup de dénivelé
# tout plat avec des petites bosses localement (dans l'eau)
polygon_1 = [[p2, p3, p6], "Herbe grasse", np.array([31, 100, 56])]
polygon_2 = [[p20, p4, p8, p7], "Boue", np.array([72, 206, 95])]  # dénivelé
polygon_3 = [[p6, p3, p20, p7, p10], "Herbe",
             np.array([20, 240, 53])]  # moyen dénivelé
polygon_4 = [[p7, p8, p11, p15, p14, p10], "Herbe sèche",
             np.array([151, 215, 67])]  # pas beaucoup de dénivelé
polygon_5 = [[p5, p9, p12, p17, p16], "Neige",
             np.array([255, 255, 255])]  # énorme dénivelé
polygon_6 = [[p9, p6, p10, p13, p18, p17, p12], "Roche", np.array(
    [127, 143, 166])]  # tout plat avec des gros plateaux
# tout plat avec des motagnes fréquentes
polygon_7 = [[p10, p14, p18, p13], "Grès", np.array([205, 160, 102])]
polygon_8 = [[p14, p15, p19, p18], "Sable",
             np.array([255, 247, 0])]  # assez plat

# creation de la liste pour manipulation plus simple
Liste_bloc = [polygon_0, polygon_1, polygon_2, polygon_3,
              polygon_4, polygon_5, polygon_6, polygon_7, polygon_8]

# =============================================================================================
# Filtre hauteur par biomes :
# ==============================================================================================

# courbes de Bezier :

def decast(t, P):
    liste = [P]
    for l in range(len(P)-1):
        liste.append([])
        for k in range(len(liste[-2])-1):
            P_new = (1-t)*liste[-2][k] + t*liste[-2][k+1]
            liste[-1].append(P_new)
    return liste[-1][0]


def return_bezier(x, P, ecart_x):
    '''ENTREE : valeur en x plutôt qu'en t pour l'utilisation de Bezier, 
    P avec points classes par ordre de x croissants.
    SORTIE : valeur approchee en x de la courbe de Bezier associee a la liste P'''
    x_min = P[0][0]
    x_max = P[-1][0]
    if x < x_min or x > x_max:  # point x en dehors de la courbe
        return 0
    else:
        # methode de dichotomie pour trouver le bon t qui correspondra
        ecart = ecart_x+1  # au moins une passe dans l'algo de dichotomie
        a = 0
        b = 1
        point_result = [0, 0]
        while ecart > ecart_x:
            t = (a + b)/2
            point_result = decast(t, P)
            if point_result[0] > x:
                b = t
            elif point_result[0] <= x:
                a = t
            ecart = np.abs(point_result[0]-x)
        return point_result[1]


# application aux filtres :

def filtre(hauteur, nom_du_biome):
    '''ENTREE : hauteur du bloc
    SORTIE : coeff multiplicatif'''
    if nom_du_biome == "Taiga":
        ecart_x = 0.0001
        P = [np.array([0, 0]), np.array([0.55, 0.4]), np.array(
            [0.5, 0.4]), np.array([1, 0.7]), np.array([2, 1.4])]
        return return_bezier(hauteur, P, ecart_x)
    elif nom_du_biome == "Marais":
        ecart_x = 0.0001
        P = [np.array([0, 0]), np.array([0.3, 0.08]),
             np.array([1, 0.07]), np.array([2, 0.07])]
        return return_bezier(hauteur, P, ecart_x)
    elif nom_du_biome == "Jungle":
        ecart_x = 0.0001
        P = [np.array([0, 0]), np.array([0.35, 0.3]),
             np.array([1, 0.16]), np.array([2, 0.2])]
        return return_bezier(hauteur, P, ecart_x)
    elif nom_du_biome == "Foret":
        ecart_x = 0.0001
        P = [np.array([0, 0]), np.array([0.5, 0.2]), np.array(
            [0.5, 0.3]), np.array([1, 0.2]), np.array([2, 0.2])]
        return return_bezier(hauteur, P, ecart_x)
    elif nom_du_biome == "Plaine":
        ecart_x = 0.0001
        P = [np.array([0, 0]), np.array([0.3, 0.1]),
             np.array([1, 0.1]), np.array([2, 0.1])]
        return return_bezier(hauteur, P, ecart_x)
    elif nom_du_biome == "Pic":
        ecart_x = 0.0001
        P = [np.array([0, 0]), np.array([0.55, 0.7]), np.array([0.5, 0.8]), np.array(
            [1, 0.8]), np.array([1.1, 0.8]), np.array([2, 1.6])]
        return return_bezier(hauteur, P, ecart_x)
    elif nom_du_biome == "Montagnes":
        ecart_x = 0.0001
        P = [np.array([0, 0]), np.array([0.55, 0.5]), np.array(
            [0.5, 0.5]), np.array([1, 0.7]), np.array([2, 1.4])]
        return return_bezier(hauteur, P, ecart_x)
    elif nom_du_biome == "Badland":
        ecart_x = 0.0001
        P = [np.array([0, 0]), np.array([0.1, 0]), np.array([0.1, 0]), np.array([0.2, 0.1]), np.array([0.3, 0.7]), np.array([0.4, 0.7]), np.array([0.4, 0]), np.array([0.4, 0]), np.array([0.4, 0]), np.array(
            [0.4, 0]), np.array([0.4, 0]), np.array([0.6, 0]), np.array([0.6, 0]), np.array([0.8, 1.3]), np.array([0.8, 1.2]), np.array([1, 0]), np.array([1.05, 0]), np.array([1.1, 0]), np.array([2, 0.3])]
        return return_bezier(hauteur, P, ecart_x)
    elif nom_du_biome == "Desert":
        ecart_x = 0.0001
        P = [np.array([0, 0]), np.array([0.5, 0.2]), np.array(
            [0.6, 0.2]), np.array([1, 0.2]), np.array([2, 0.2])]
        return return_bezier(hauteur, P, ecart_x)
    elif nom_du_biome == "Droite y=ax":
        ecart_x = 0.0001
        P = [np.array([0, 0]), np.array([2, 2])]
        return return_bezier(hauteur, P, ecart_x)


def visu_filtre(nom_du_biome):
    '''ENTREE : nom du biome (chaine de caracteres)
    SORTIE : graphe des coeff multiplicatifs'''
    hauteur_max = 500
    hauteur_min = 0
    H = np.linspace(0, 500, 1000)
    Y = [filtre(h, nom_du_biome) for h in H]
    plt.plot(H, Y, color="orange")
    plt.title("Filtre en hauteur du biome "+nom_du_biome)
    plt.show()


def generation_liste_couleurs(nbr):
    color = []
    for i in range(nbr):
        color.append(np.array(np.random.choice(range(256), size=3))/255)
    return color


def visualise_voronoi(N, nombre_centroides, liste_cellules_origine, liste_cellule, random_couleur):
    '''ENTREE : dimension du carre N
    SORTIE imshow'''
    carte_cellule_origine = np.zeros((N, N, 3))
    carte_cellule = np.zeros((N, N, 3))

    compteur = 0  # pour passer aux couleurs suivantes
    # parcourt de tous les cellules de la liste des cellules issue de voronoi
    for cellule in range(len(liste_cellule)):
        couleur = random_couleur[compteur]
        compteur += 1

        # on se balade dans liste_cellule_origine
        for case in range(len(liste_cellules_origine[cellule])):
            coord_x_origine, coord_y_origine = liste_cellules_origine[
                cellule][case][0], liste_cellules_origine[cellule][case][1]
            carte_cellule_origine[coord_x_origine][coord_y_origine] = couleur

        # on se balade dans liste_cellule
        for case in range(len(liste_cellule[cellule])):
            coord_x, coord_y = liste_cellule[cellule][case][0], liste_cellule[cellule][case][1]
            carte_cellule[coord_x][coord_y] = couleur

    # on affiche les points centroides
    for i in range(len(liste_cellule)):
        # ATTENTION : plot affiche les élements en partant de l'origine du repère (ici en haut à gauche) (comme text)
        x = liste_cellule[i][0][1]
        y = liste_cellule[i][0][0]
        plt.plot(x, y, "o")

    plt.imshow(carte_cellule_origine)
    plt.show()
    plt.imshow(carte_cellule)
    plt.show()


# =============================================================================
# Perlin Noise
# =============================================================================
def smoothstep(edge0, edge1, x):
    '''fade interpolation, edge0 = 0 et edge1 = 1'''
    if x < edge0:
        return 0

    elif x >= edge1:
        return 1

    # rescale x
    x = (x - edge0) / (edge1 - edge0)
    return x * x * (3 - 2 * x)


def interpol_fade(t, a1, a2):
    '''Fade interpolation. t appartient à [0,1], f est soit smoothstep, soit y=x'''

    f = smoothstep(0, 1, t)
    return a1 + f*(a2-a1)


def interpol_lin(t, a1, a2):
    '''linear interpolation. T appartient à [0,1], f est soit smoothstep, soit y=x'''

    return a1 + t*(a2-a1)


def generation_vecteurs(N):
    '''ENTREE : N la taille de la cartede Perlin
    SORTIE : liste des vecteurs gradients de norme 1 aux noeuds'''

    liste_gradients_totale = np.zeros((N+1, N+1, 2))

    # RANDOM PARMIS 4 VECTEURS
    liste_4_vect = [np.array([1, 1]), np.array(
        [1, -1]), np.array([-1, 1]), np.array([-1, -1])]

    # création matrice ou chaque elem est la liste des 4 vecteurs de la case (Rappel : 1 par coin)
    for ligne in range(N+1):
        for colonne in range(N+1):

            # FULL RANDOM
            a = random.randint(1, 255)
            b = random.randint(1, 255)
            norme = np.sqrt(a**2 + b**2)
            gradient = np.array([a, b])/norme

            liste_gradients_totale[ligne][colonne] = gradient

    return liste_gradients_totale


def Perlin(N, mesh):
    '''ENTREE : N la taille de la carte, mesh taille de la carte de Perlin, la précision (les valeur dans perlin sont arrondi au combientieme)
    SORTIE : carte N*mesh avec nuances de gris'''

    # taille_case = 1 par défaut
    # distance : haut gauche - haut droite - bas gauche - bas droite

    taille_fenetre = N//mesh  # // pour empecher mesh de forcement être un diviseur de N
    carte = np.zeros((N, N))
    liste_gradients_totale = generation_vecteurs(mesh)

    for ligne in range(mesh):
        for colonne in range(mesh):

            # on se balade dans la fenêtre de taille taille_fenetre*taille_fenetre
            for i in range(taille_fenetre):
                for j in range(taille_fenetre):

                    # taille_case = 1 par défaut
                    # distance : haut gauche - haut droite - bas gauche - bas droite (i = ligne, j = colonne)
                    distance = [[np.array([-(i + 0.5), (j + 0.5)]), np.array([-(i + 0.5), -(taille_fenetre - j - 0.5)])], [np.array(
                        [(taille_fenetre - i - 0.5), (j + 0.5)]), np.array([(taille_fenetre - i - 0.5), -(taille_fenetre - j - 0.5)])]]

                    # générer la liste des 4 produits scalaires
                    prod = []
                    for ii in range(2):
                        for jj in range(2):
                            prod_dot = np.dot(
                                liste_gradients_totale[ligne+ii][colonne+jj], distance[ii][jj])
                            prod.append(prod_dot)

                    # Facteur d'interpolation
                    fact_1 = (i + 0.5)/taille_fenetre
                    fact_2 = (j + 0.5)/taille_fenetre

                    # interpolation (haut gauche - bas gauche puis droite puis les 2)
                    interpol1 = interpol_lin(fact_1, prod[0], prod[2])
                    interpol2 = interpol_lin(fact_1, prod[1], prod[3])
                    interpol3 = interpol_lin(fact_2, interpol1, interpol2)

                    carte[i+ligne*taille_fenetre][j +
                                                  colonne*taille_fenetre] = interpol3

    return carte


def normalize_carte(image, min_val, max_val, precision=2, centree=False):
    """ENTREE image, max_val, et precision. Centree pour savoir si on centre autour de 0 ou non
    SORTIE : renvoie perlin normalisé entre 0 et max_val"""
    # plot imshow
    N = len(image)
    carte = np.zeros((N, N))

    mini = np.min(image)
    maxi = np.max(image)

    # imshow
    for ligne in range(N):
        for colonne in range(N):
            val = (image[ligne][colonne] - mini) / (maxi - mini)
            carte[ligne][colonne] = round(val * (max_val-min_val), precision)

    # centree autour de 0 si c'est précisé
    if centree:
        carte = carte - max_val/2
    else:
        carte += min_val

    return carte


def get_histogram(image, precision):
    """Renvoie l'histogramme des fréquence de chaque valeur"""

    # nombre de valeur différente : fonction de la précision
    nbr_val = int(round((np.max(image) - np.min(image)) * 10**precision, 0))
    # put pixels in a 1D array by flattening out img array
    flat = image.flatten()

    # afficher la liste des occurences
    plt.hist(flat, nbr_val)


def liste_remplace(image, L1, L2):
    """ENTREE : L1 et L2 de meme taille, image contient les valeurs de L1
    BUT : remplacer les valeurs de L1 par L2 dans image"""
    N = len(image)
    for ligne in range(N):
        for colonne in range(N):

            indice = 0
            while image[ligne][colonne] != L1[indice]:
                indice += 1
            image[ligne][colonne] = L2[indice]

    return image


def histo_equal(image, min_val, max_val, precision=2):
    """ENTREE: un carte(perlin), max_val la val max sur laquelle entre 0 et elle  on veut faire l'égalisation(la même que dans normalize), une precision(la même que dans normalize)
    SORTIE : carte de perlin egalisée"""

    N = len(image)
    liste_val = []

    mini = np.min(image)
    maxi = np.max(image)
    nbr_val = int((maxi - mini) * 10**precision)

    # rescencer toutes les valeurs dans l'image
    liste_val = [round(mini + i*10**(-precision), precision)
                 for i in range(nbr_val+1)]  # +1 pour aller de [0 à 1] inclus

    # Compter les occurences de chaque valeur
    liste_occurence = np.zeros(nbr_val+1)
    for i in range(nbr_val+1):
        liste_occurence[i] = len(np.where(np.array(image) == liste_val[i])[0])

    # calculer la cumulative distrib function (cdf)
    liste_occurence_cumul = np.cumsum(liste_occurence)

    # calculer les nouvelles valeurs
    cdf_min = np.min(liste_occurence_cumul)
    liste_val_new = np.zeros(nbr_val+1)
    for i in range(nbr_val+1):
        liste_val_new[i] = round(
            (max_val-min_val) * (liste_occurence_cumul[i] - cdf_min)/(N*N - cdf_min), precision) + min_val
    liste_remplace(image, liste_val, liste_val_new)
    return image


def visu_perlin(N, image, max_val=1):
    # plot imshow
    carte = np.zeros((N, N, 3))
    # imshow
    for ligne in range(N):
        for colonne in range(N):

            carte[ligne][colonne] = np.array(
                [image[ligne][colonne], image[ligne][colonne], image[ligne][colonne]])/max_val

    plt.axis("off")
    plt.imshow(carte)
    plt.show()

def fractal_brownian_motion(N, mesh, nbr_octave, variation_brutale):
    """ENTREE : le nbr d'octaves
    SORTIE : renvoie la somme des perlin (avec la mesh qui bouge = on augmente la fréquence)"""
    carte = np.zeros((N, N))
    liste_cartes_perlin = [
        Perlin(N, mesh*(i+1))/(variation_brutale**i) for i in range(0, nbr_octave)]

    for ligne in range(N):
        for colonne in range(N):

            somme = 0
            for i in range(nbr_octave):
                somme += liste_cartes_perlin[i][ligne][colonne]
            carte[ligne][colonne] = somme

    return carte


def generate_t_p(N, mesh, nbr_octave, min_val_precip, max_val_precip, min_val_temp, max_val_temp, precision=2):
    """ENTREE : N, mesh car on appelle Perlin 2 fois (temperature et precipitation)
    SORTIE : carte des températures et des precipitations normalisée et des hauteurs
    cette fonction permet de 'sauvegarder' les températures et précipitations  sur chaque bloc"""

    # #carte perlin precip normalisée
    perlin_precip = Perlin(N, mesh)
    perlin_precip_norm = histo_equal(normalize_carte(
        perlin_precip, min_val_precip, max_val_precip, precision), min_val_precip, max_val_precip, precision)

    # carte perlin temp normalisée
    perlin_temp = Perlin(N, mesh)
    perlin_temp_norm = histo_equal(normalize_carte(
        perlin_temp, min_val_temp, max_val_temp, precision), min_val_temp, max_val_temp, precision)
    return [perlin_temp_norm, perlin_precip_norm]


def generate_h(N, mesh, nbr_octave, variation_brutale, min_val, precision=2):
    """ENTREE : N, mesh car on appelle Perlin 2 fois (temperature et precipitation)
    MAX_VAL = Min val + 2. On force une amplitude de 2 pour perlin
    Attention à ne pas mettre min_val supérieur à 0. Les filtres  de bezier ne sont pas définis après
    SORTIE : carte des températures et des precipitations normalisée et des hauteurs
    cette fonction permet de 'sauvegarder' les températures et précipitations  sur chaque bloc"""

    # #carte perlin hauteur normalisée
    perlin_haut = fractal_brownian_motion(
        N, mesh, nbr_octave, variation_brutale)
    perlin_haut_norm = normalize_carte(
        perlin_haut, min_val, min_val+2, precision)

    return perlin_haut_norm, perlin_haut


def generate_h_bruit(N, mesh_bruit, nbr_octave_bruit, variation_brutale_bruit, min_bruit, max_bruit, precision=2):
    """ENTREE : N, mesh car on appelle Perlin 2 fois (temperature et precipitation)
    MAX_VAL = Min val + 2. On force une amplitude de 2 pour perlin
    Attention à ne pas mettre min_val supérieur à 0. Les filtres  de bezier ne sont pas définis après
    SORTIE : carte des températures et des precipitations normalisée et des hauteurs
    cette fonction permet de 'sauvegarder' les températures et précipitations  sur chaque bloc"""

    # #carte perlin hauteur normalisée
    perlin_haut = fractal_brownian_motion(
        N, mesh_bruit, nbr_octave_bruit, variation_brutale_bruit)
    perlin_haut_norm = normalize_carte(
        perlin_haut, min_bruit, max_bruit, precision)

    return perlin_haut_norm, perlin_haut


def moyenne_cellule(T_P, liste_cellule, nombre_centroides):
    """ENTREE : rien, tout est déjà dans le main
    SORTIE : [[coord_centroid, temperature moy, precipitation moy], [#une autre cellule], ... ]
    retour Lloyd : [[coord_centroid,coord_bloc1,coord_bloc2, ...],[#une autre suface], ...]"""
    liste_temp_precip = []

    # génération des Perlin temp et précip normalisées en dehors de la fonction
    perlin_temp, perlin_precip = T_P[0], T_P[1]

    for cellule in range(nombre_centroides):
        liste_temp_cellule, liste_precip_cellule = [], []

        liste_coord = liste_cellule[cellule]
        # recuperation des temp et precip de chaque bloque dans la cellule active
        for coord in range(len(liste_coord)):
            coord_x = liste_coord[coord][0]
            coord_y = liste_coord[coord][1]

            # print(perlin_temp[coord_x][coord_y], coord)
            liste_temp_cellule.append(perlin_temp[coord_x][coord_y])
            liste_precip_cellule.append(perlin_precip[coord_x][coord_y])

        # calcul valeur moyenne de chaque cellule
        moy_temp_cellule = sum(liste_temp_cellule)/len(liste_temp_cellule)
        moy_precip_cellule = sum(liste_precip_cellule) / \
            len(liste_precip_cellule)
        # on remplit la liste finale à retourner
        liste_temp_precip.append(
            [[liste_cellule[cellule][0][0], liste_cellule[cellule][0][1]], moy_temp_cellule, moy_precip_cellule])

        # parcourt de tous les cellules de la liste des cellules issue de voronoi (une cellule par centroide) pour ajouter le nom du biome
        Temp = liste_temp_precip[cellule][1]
        Precip = liste_temp_precip[cellule][2]
        nom_biome = biome(Temp, Precip, Liste_biomes)

        # ajout du nom du biome dans liste_temp_precip
        liste_temp_precip[cellule].append(nom_biome)

    return liste_temp_precip


def visualise_biomes(N, liste_temp_precip, new_H, hauteur_mer, liste_cellule, nombre_centroides, matrice, random_couleur):
    """ENTREE : rien
    SORTIE : ajoute le nom du biome à list_temp_precip. 
            Schéma avec les cellules et les biomes correspondants"""

    carte_cellule = np.zeros((N, N, 3))
    hauteur = new_H*2

    # pour passer aux couleurs suivantes
    compteur = 0

    # parcourt de tous les cellules de la liste des cellules issue de voronoi (une cellule par centroide)
    for cellule in range(nombre_centroides):
        nom_biome = liste_temp_precip[cellule][3]

        # ATTENTION : la méthode .text() place les points en partant d'en bas à droite
        # En matrice on fait ~ligne(ordonnée) et colonne(abscisse) et en plot abscisse et ordonnée
        x_text = liste_temp_precip[cellule][0][1]
        y_text = liste_temp_precip[cellule][0][0]
        plt.text(x_text, y_text, str(nom_biome),
                 horizontalalignment='center', verticalalignment='center')

        couleur = random_couleur[compteur]
        compteur += 1

        for case in range(len(liste_cellule[cellule])):
            coord_x = liste_cellule[cellule][case][0]
            coord_y = liste_cellule[cellule][case][1]

            carte_cellule[coord_x][coord_y] = couleur
    plt.imshow(carte_cellule)
    plt.show()


# GESTION DES MASK POUR LES LIAISONS ENTRE BIOMES + FILTRES
def gaussienne_2D(x, y, x_0, y_0, sigma, A):
    """x0 et y0 sont le centre de la gaussienne"""
    return A*np.exp(-((x-x_0)**2/(2*sigma**2)+(y-y_0)**2/(2*sigma**2)))


def generate_kernel(taille_noyau):
    """genere le noyau de gauss. Taille_noyau à prendre impair de préférence. Sigma = 1 souvent"""
    centre = taille_noyau//2
    noyau = np.zeros((taille_noyau, taille_noyau))
    sigma = taille_noyau/(4*np.sqrt(2*np.log(2)))

    for i in range(taille_noyau):
        for j in range(taille_noyau):
            noyau[i][j] = gaussienne_2D(i, j, centre, centre, sigma, 1)

    noyau = noyau/np.sum(noyau)

    return noyau


def gaussain_blur(image, taille_noyau):
    """fait la convulition par le noyau de gauss. Image est une array. Ne traite pas les bordures
    Matrice est utile pour nous car on veut "concaténer" toutes les gaussian blur dans une seule est même matrice"""
    len_x = len(image)
    len_y = len(image[0])
    bordure = taille_noyau//2
    bordure = 0

    nouvelle_image = np.zeros((len_x, len_y))
    noyau = generate_kernel(taille_noyau)

    # on parcourt la carte
    for ligne in range(bordure, bordure+len_x):
        for colonne in range(bordure, bordure+len_y):

            # on n'applique le blur que dans le carré pas au dehors (nécessaire pour la carte car on veut une valeur proche de 0 sur le carré justement et proche des valeurs 0)
            # NECESSAIRE pour les mask (car biomes à un et reste à 0); Mais pas pour la carte des hauteurs
            if image[ligne][colonne] != 0:
                # définition origine en haut à gauche du filtre
                origine_ligne = ligne - taille_noyau//2
                origine_colonne = colonne - taille_noyau//2
                liste = []

                # on calcul la somme dans le noyau
                for i in range(taille_noyau):
                    for j in range(taille_noyau):

                        # condition si le point considéré est dans la carte
                        if (origine_ligne + i) >= 0 and (origine_ligne + i) < len_x and (origine_colonne + j) >= 0 and (origine_colonne + j) < len_y:
                            liste.append(
                                image[origine_ligne + i][origine_colonne + j] * noyau[i][j])
                nouvelle_image[ligne][colonne] = np.sum(liste)

    # utile si on fait du vrai "gaussian blur" sans modifier une matrice déja existante pour plusieurs biomes
    return nouvelle_image


def gaussain_blur_out(image, taille_noyau):
    """fait la convulition par le noyau de gauss. Image est une array. Ne traite pas les bordures
    Matrice est utile pour nous car on veut "concaténer" toutes les gaussian blur dans une seule est même matrice"""
    len_x = len(image)
    len_y = len(image[0])
    bordure = taille_noyau//2
    bordure = 0

    nouvelle_image = np.zeros((len_x, len_y))
    noyau = generate_kernel(taille_noyau)

    # on parcourt la carte
    for ligne in range(bordure, bordure+len_x):
        for colonne in range(bordure, bordure+len_y):

            # on n'applique le blur que dans le carré pas au dehors (nécessaire pour la carte car on veut une valeur proche de 0 sur le carré justement et proche des valeurs 0)
            # définition origine en haut à gauche du filtre
            origine_ligne = ligne - taille_noyau//2
            origine_colonne = colonne - taille_noyau//2
            liste = []

            # on calcul la somme dans le noyau
            for i in range(taille_noyau):
                for j in range(taille_noyau):

                    # condition si le point considéré est dans la carte
                    if (origine_ligne + i) >= 0 and (origine_ligne + i) < len_x and (origine_colonne + j) >= 0 and (origine_colonne + j) < len_y:
                        liste.append(image[origine_ligne + i]
                                     [origine_colonne + j] * noyau[i][j])
            nouvelle_image[ligne][colonne] = np.sum(liste)

    return nouvelle_image


def mesh_mer(N, matrice_des_hauteurs, hauteur_mer):
    """on créer un mask avec des 0 pour l'eau et des 1 pour la terre"""
    mask_mer = np.zeros((N, N))
    for ligne in range(N):
        for colonne in range(N):
            if matrice_des_hauteurs[ligne][colonne] > hauteur_mer:
                mask_mer[ligne][colonne] = 1

    return mask_mer


def mesh_eau(N, matrice, matrice_des_hauteurs, hauteur_mer):
    """on créer un mask avec des 0 pour l'eau et des 1 pour la terre"""
    mask_eau = np.zeros((N, N))
    for ligne in range(N):
        for colonne in range(N):
            if matrice[ligne][colonne][0] != "eau" and matrice[ligne][colonne][0] != "eau douce":
                mask_eau[ligne][colonne] = 1

    return mask_eau


def plage(N, matrice, matrice_des_hauteurs, hauteur_mer, taille_noyau_plage, seuil_plage):
    """créer des plages en frontière avec l'eau si le delta de hauteur n'est pas trop grand
    Retourne la matrice de la carte avec attributs plage pour le sable
    A appliquer apres les rivières pour avoir mesh_eau fonctionnel"""
    mask_eau = mesh_eau(N, matrice, matrice_des_hauteurs, hauteur_mer)

    Liste_terre = np.where(mask_eau == 1)
    taille_noyau_plage = 5
    for bloc in range(len(Liste_terre[0])):

        # définition origine en haut à gauche du filtre
        origine_ligne = Liste_terre[0][bloc] - taille_noyau_plage//2
        origine_colonne = Liste_terre[1][bloc] - taille_noyau_plage//2
        liste_libélé = []
        liste = []

        # on calcul la somme dans le noyau
        for i in range(taille_noyau_plage):
            for j in range(taille_noyau_plage):

                # condition si le point considéré est dans la carte
                if (origine_ligne + i) >= 0 and (origine_ligne + i) < N and (origine_colonne + j) >= 0 and (origine_colonne + j) < N:
                    liste_libélé.append(
                        matrice[origine_ligne + i][origine_colonne+j][0])
                    liste.append([[origine_ligne + i, origine_colonne + j],
                                 matrice_des_hauteurs[origine_ligne + i][origine_colonne + j]])

        # si on est en bordure d'eau
        if "eau" in liste_libélé:
            for elem in range(len(liste)):
                ligne = liste[elem][0][0]
                colonne = liste[elem][0][1]
                # si on est pas dans l'eau ET qu'on est à la bonne hauteur
                if (matrice[ligne][colonne][0] != "eau" and matrice[ligne][colonne][0] != "eau douce") and liste[elem][1] <= seuil_plage:
                    matrice[ligne][colonne][0] = "Plage"
                    matrice[ligne][colonne][1] = "Sable"
                    matrice[ligne][colonne][3] = Liste_biomes[9][2]
                    matrice[ligne][colonne][4] = Liste_biomes[9][3]
                    matrice[ligne][colonne][5] = Liste_biomes[9][4]

    return matrice


def appliquer_filtre_2(N, matrice_des_hauteurs, matrice_des_hauteurs_errosion, matrice_des_biomes, matrice_des_blocs, hauteur_mer, terre):
    '''ENTREE : matrice des hauteurs, 
    matrices des biomes ([[coord_centroid, temperature moy, precipitation moy, nom ubiome], [#une autre cellule], ... ]
    matrice_des_blocs ([[coord_centroid,coord_bloc1,coord_bloc2, ...],[#une autre suface], ...])
    un delta max de hauteur entre deux blocs (à voir pour le delta hauteur)
    SORTIE : matrice des hauteurs'''
    # IDEE : On se balade par biome.
    # on applique un masque ou on met tout à 0 et le biome à 1. (sert à gérer les liaisons entre biomes)
    # on blur le mask
    # on applique à la matrice_des_hauteurs les coeff multiplicateur : bezier et mask blur
    # on profite du résultat ??

    nouvelle_matrice_des_hauteurs = np.zeros((N, N))
    matrice_filtre = np.zeros((N, N))
    bezier_non_blur = np.zeros((N, N))
    bezier_blur = np.zeros((N, N))

    # on se balade dans les biomes
    for biome in range(len(Liste_biomes)):
        nom_biome = Liste_biomes[biome][1]
        mask = np.zeros((N, N))
        liste_bloc_temporaire = []

        localisation = np.where(
            np.array([ligne[3] for ligne in matrice_des_biomes]) == nom_biome)[0]

        # on parcourt toutes ces localisations
        for cellule in localisation:
            # on récupère le centroid qui porte le nom de ce biome
            centroid = matrice_des_biomes[cellule][0]

            # on va récupérer tous les bloque qui porte le nom de ce biome
            for i in range(len(matrice_des_blocs)):
                if matrice_des_blocs[i][0] == centroid:
                    # on a maintenant la liste des blocs qui porte le nom du biome
                    for bloc in matrice_des_blocs[i]:
                        mask[bloc[0]][bloc[1]] = 1
                        liste_bloc_temporaire.append(bloc)

        # on floute le mask fini pour ce biome. ET QUE pour ce biome
        compteur = True
        for i in matrice_des_biomes:
            if compteur and nom_biome == i[3]:

                if nom_biome == "Montagnes":
                    matrice_filtre += gaussain_blur(
                        mask, 80)  # on modifie """matrice_filtre""" créer la matrice des flitre des biomes concaténés
                elif nom_biome == "Taiga":
                    matrice_filtre += gaussain_blur(
                        mask, 40)  # on modifie """matrice_filtre""" créer la matrice des flitre des biomes concaténés
                elif nom_biome == "Jungle":
                    matrice_filtre += gaussain_blur(
                        mask, 40)  # on modifie """matrice_filtre""" créer la matrice des flitre des biomes concaténés
                else:
                    matrice_filtre += gaussain_blur(
                        mask, 30)  # on modifie """matrice_filtre""" créer la matrice des flitre des biomes concaténés
                compteur = False

        # on applique les filtres des hauteurs
        for bloc in liste_bloc_temporaire:
            if terre:  # full terre (donc hauteur que positives)
                if matrice_des_hauteurs[bloc[0]][bloc[1]] < hauteur_mer:
                    hauteur = -matrice_des_hauteurs[bloc[0]][bloc[1]]
                else:
                    hauteur = matrice_des_hauteurs[bloc[0]][bloc[1]]

            else:
                hauteur = matrice_des_hauteurs[bloc[0]][bloc[1]]

            # vaut 0 quand hauteur négative
            facteur_bezier = filtre(hauteur, nom_biome)
            bezier_non_blur[bloc[0]][bloc[1]] = facteur_bezier

    # on floute la carte
    bezier_blur = bezier_non_blur
    for i in range(1):
        # 20 c'est bien pour une map bien lisse si on a pas les masks
        bezier_blur = gaussain_blur(bezier_blur, 20)

    # Carte d'errosion : objectif est d'érroder localement au frontières des biomes
    # on a des valeurs élevées aux frontières
    inv_matrice_filtre = 1 - matrice_filtre
    # notre carte errosion ne prend effet qu'aux frontirèes
    matrice_des_hauteurs_errosion *= inv_matrice_filtre

    # on pondère les hauteurs. Hauteurs proches de bezier blur aux frontières et proches de bezier non blur au centre des biomes. BUT : ne pas blur le centre des biomes
    for ligne in range(N):
        for colonne in range(N):
            hauteur_blur = bezier_blur[ligne][colonne]
            hauteur_non_blur = bezier_non_blur[ligne][colonne]
            facteur_mask = matrice_filtre[ligne][colonne]
            facteur_errosion = matrice_des_hauteurs_errosion[ligne][colonne]

            # on traite l'eau commeun biome à part entière, on le le smooth pas
            if matrice_des_hauteurs[ligne][colonne] > hauteur_mer:
                nouvelle_matrice_des_hauteurs[ligne][colonne] = (
                    (1-facteur_mask)*hauteur_blur + facteur_mask * hauteur_non_blur) * (1-facteur_errosion)**5
            else:
                if terre:  # si on veut full terre
                    nouvelle_matrice_des_hauteurs[ligne][colonne] = (
                        (1-facteur_mask)*hauteur_blur + facteur_mask * hauteur_non_blur) * (1-facteur_errosion)**5

                else:
                    nouvelle_matrice_des_hauteurs[ligne][colonne] = matrice_des_hauteurs[ligne][colonne]

    return nouvelle_matrice_des_hauteurs


def associate(N, matrice_des_biomes, matrice_des_blocs, matrice_des_temperatures, matrice_des_precipitations):
    '''ENTREE : map 
    matrices des biomes ([[coord_centroid, temperature moy, precipitation moy, nom biome], [#une autre cellule], ... ]
    matrice_des_blocs ([[coord_centroid,coord_bloc1,coord_bloc2, ...],[#une autre suface], ...])
    un delta max de hauteur entre deux blocs (à voir pour le delta hauteur)
    SORTIE : map avec le nom des biomes et leur couleur et (x,y) à chaque case'''
    matrice = [[[] for i in range(N)] for j in range(N)]
    id_cellule = 0
    # on se balade dans les biomes
    for biome in range(len(Liste_biomes)):
        nom_biome = Liste_biomes[biome][1]

        # donne les celulles qui portent ce nom de biome
        localisation = np.where(
            np.array([ligne[3] for ligne in matrice_des_biomes]) == nom_biome)[0]

        # on parcourt toutes ces localisations
        for cellule in localisation:
            # on récupère le centroid qui porte le nom de ce biome
            centroid = matrice_des_biomes[cellule][0]
            # on va récupérer tous les bloque qui porte le nom de ce biome
            for i in range(len(matrice_des_blocs)):
                if matrice_des_blocs[i][0] == centroid:
                    # on a maintenant la liste des blocs qui porte le nom du biome
                    for bloc in matrice_des_blocs[i]:
                        temp, precip = matrice_des_temperatures[bloc[0]][bloc[1]
                                                                         ], matrice_des_precipitations[bloc[0]][bloc[1]]

                        # ATTENTION : On a pas le type de blopc correspondant au biome tout le temsp car pour les biomes c'est une température moyenne sur tout le biome qui est pris. Pour les bloc c'est localement la temp et la precip
                        matrice[bloc[0]][bloc[1]].append(
                            nom_biome)  # le libélé du biome
                        matrice[bloc[0]][bloc[1]].append(Liste_bloc[biome][1])
                        matrice[bloc[0]][bloc[1]].append(
                            get_bloc(temp, precip, Liste_bloc)[0])  # le libélé du bloc

                        matrice[bloc[0]][bloc[1]].append(
                            Liste_biomes[biome][2])
                        matrice[bloc[0]][bloc[1]].append(
                            Liste_biomes[biome][3])
                        matrice[bloc[0]][bloc[1]].append(
                            Liste_biomes[biome][4])

                        matrice[bloc[0]][bloc[1]].append(
                            id_cellule)  # on ajoute un id par cellule

            id_cellule += 1

    return matrice


def eau(N, new_H, hauteur_mer, matrice, facteur, archipel=False):
    """on remplace par de l'eau les hauteurs trop faibles
    Plus le facteur est grand, plus la gaussienne est plate. Un facteur de 1 implique une hauteur divisée par 2 à mi hauteur"""
    centre = N//2
    #plus le facteur est grand, plus la gaussienne est plate. Un facteur de 1 implique une hauteur divisée par 2 à mi hauteur
    sigma = N/(4*np.sqrt(2*np.log(2)))*facteur
    couleur_mer = np.array([64, 63, 252])
    facteur = np.max(new_H)*0.5

    for ligne in range(N):
        for colonne in range(N):
            if archipel:
                new_H[ligne][colonne] *= gaussienne_2D(
                    ligne, colonne, centre, centre, sigma, 1)
                # BIEN Pour une grande ile : * (facteur*np.sqrt((centre-ligne)**2+(centre-colonne)**2)/(N/2))
                new_H[ligne][colonne] -= (1-gaussienne_2D(ligne,
                                          colonne, centre, centre, sigma, 1))*(facteur*2)

            if new_H[ligne][colonne] <= hauteur_mer:
                matrice[ligne][colonne][0] = "eau"
                matrice[ligne][colonne][3] = couleur_mer
                matrice[ligne][colonne][4] = couleur_mer
                matrice[ligne][colonne][5] = couleur_mer

    return matrice, new_H

def blur_3(N, mesh, vor_map, mesh_blur, boundary_displacement):
    """fonction blur prise d'internet"""
    bruit_x, bruit_y = Perlin(N, mesh_blur), Perlin(N, mesh_blur)
    boundary_noise = np.dstack([bruit_x, bruit_y])
    boundary_noise = np.indices(
        (N, N)).T + boundary_displacement*boundary_noise
    boundary_noise = boundary_noise.clip(0, N-1).astype(np.uint32)

    blurred_vor_map = np.zeros_like(vor_map)

    for x in range(N):
        for y in range(N):
            j, i = boundary_noise[x, y]
            blurred_vor_map[x, y] = vor_map[i, j]

    return blurred_vor_map


def riviere(N, taille_noyau_cellule, taille_noyau_biome, matrice, matrice_des_hauteurs, hauteur_max_riviere, hauteur_mer):
    """créer des grosses rivieres entre les biomes.
    créer des petites rivieres entre les cellules"""
    mask_mer = mesh_mer(N, matrice_des_hauteurs, hauteur_mer)

    bordure_biome = []
    bordure_cellule = []
    couleur_mer = np.array([64, 63, 252])

    for ligne in range(N):
        for colonne in range(N):

            # on limite les rivières à des altitudes basses
            if matrice_des_hauteurs[ligne][colonne] < hauteur_max_riviere:
                origine_ligne = ligne-1
                origine_colonne = colonne-1
                biome_actuel = matrice[ligne][colonne][0]
                cellule_actuel = matrice[ligne][colonne][6]

                # INTER CELLULE
                for i in range(taille_noyau_cellule):
                    for j in range(taille_noyau_cellule):
                        # condition si le point considéré est dans la carte
                        if (origine_ligne + i) >= 0 and (origine_ligne + i) < N and (origine_colonne + j) >= 0 and (origine_colonne + j) < N:

                            # on vérifie si c'est en bordure de cellule
                            if cellule_actuel != matrice[origine_ligne + i][origine_colonne + j][6]:
                                bordure_cellule.append([ligne, colonne])
                                break
                    else:
                        continue
                    break  # on sort des 2 boucles for si le dernier if est vérifier. Sinon on continue à la prochaine itération

                # INTER BIOME
                for i in range(taille_noyau_biome):
                    for j in range(taille_noyau_biome):
                        # condition si le point considéré est dans la carte
                        if (origine_ligne + i) >= 0 and (origine_ligne + i) < N and (origine_colonne + j) >= 0 and (origine_colonne + j) < N:

                            # on vérifie si c'est en bordure de biome
                            # on pourrait ajouter pour empecher d'élargir les rivage et les côtes "and  matrice[origine_ligne +i][origine_colonne +j][0] != "eau""
                            if matrice[origine_ligne + i][origine_colonne + j][0] != "eau" and biome_actuel != matrice[origine_ligne + i][origine_colonne + j][0]:
                                bordure_biome.append([ligne, colonne])
                                break
                    else:
                        continue
                    break  # on sort des 2 boucles for si le dernier if est vérifier. Sinon on continue à la prochaine itération

    # on blur le mask rivière pour créer un lit de rivière et non une rivière brutale à hauteur 0 sans transition avec la côte à côté
    mask_riviere = np.zeros((N, N))
    for case in range(len(bordure_biome)):
        mask_riviere[bordure_biome[case][0]][bordure_biome[case][1]] = 1
    for case in range(len(bordure_cellule)):
        mask_riviere[bordure_cellule[case][0]][bordure_cellule[case][1]] = 1

    mask_riviere *= mask_mer
    mask_riviere_blur = gaussain_blur_out(mask_riviere, 5)

    # Ici les rivières/fleuves ne sont pas forcément à hauteur de la mer. Ils peuvent être plus haut.
    # on applique le filtre de tel sort à ce que les hauteurs tendent vers 0 au centre des rivières
    for ligne in range(N):
        for colonne in range(N):
            matrice_des_hauteurs[ligne][colonne] *= (
                1 - mask_riviere_blur[ligne][colonne])

    # on modifie matrice en ajoutant de l'eau en bordure de biome
    for case in range(len(bordure_biome)):
        matrice[bordure_biome[case][0]][bordure_biome[case][1]][0] = "eau"
        matrice[bordure_biome[case][0]][bordure_biome[case][1]][3] = couleur_mer
        matrice[bordure_biome[case][0]][bordure_biome[case][1]][4] = couleur_mer
        matrice[bordure_biome[case][0]][bordure_biome[case][1]][5] = couleur_mer

    # on modifie matrice en ajoutant de l'eau en bordure de cellule
    for case in range(len(bordure_cellule)):
        matrice[bordure_cellule[case][0]][bordure_cellule[case][1]][0] = "eau"
        matrice[bordure_cellule[case][0]
                ][bordure_cellule[case][1]][3] = couleur_mer
        matrice[bordure_cellule[case][0]
                ][bordure_cellule[case][1]][4] = couleur_mer
        matrice[bordure_cellule[case][0]
                ][bordure_cellule[case][1]][5] = couleur_mer

    return matrice, matrice_des_hauteurs


def cours_eau(N, matrice, carte_hauteur, nb_rivières, hauteur_cours_eau):
    # counter = 0
    couleur = np.array([70, 130, 255])
    # couleur_mer = np.array([64, 63, 252])

    # on sélectionne les origine de cours d'eau
    liste_bloc = []
    for ligne in range(N):
        for colonne in range(N):
            if carte_hauteur[ligne][colonne] >= hauteur_cours_eau:
                liste_bloc.append([ligne, colonne])

    # on créer les rivières à partir des origines de cours d'eau
    nb_rivières_adapté = min(nb_rivières, len(liste_bloc))
    # on prend autant de rivière que possible pour avoir le nb souhaité en consigne. Parfois on a pas assez de source haute c'est pas possible
    for n in range(nb_rivières_adapté):
        # choix de l'origine de la rivière
        origine = random.randint(0, len(liste_bloc)-1)

        # alt_init = carte_hauteur[origine[0]][origine[1]]

        # on continue tant que c'est pas de l'eau
        ligne_courant, colonne_courant = liste_bloc[origine][0], liste_bloc[origine][1]
        elem = matrice[ligne_courant][colonne_courant][0]

        while elem != "eau" and elem != "eau douce":
            liste_hauteurs = []
            liste_coord = []

            # parcours du filtre 3x3
            origine_ligne = ligne_courant - 1
            origine_colonne = colonne_courant - 1
            for i in range(3):
                for j in range(3):
                    if (origine_ligne + i) >= 0 and (origine_ligne + i) < N and (origine_colonne + j) >= 0 and (origine_colonne + j) < N:
                        liste_hauteurs.append(
                            [carte_hauteur[origine_ligne + i][origine_colonne + j]])
                        liste_coord.append(
                            [origine_ligne + i, origine_colonne + j])

            # remplissage d'eau à l'indice mini
            indice_mini = np.argmin(liste_hauteurs)
            ligne = liste_coord[indice_mini][0]
            colonne = liste_coord[indice_mini][1]

            # mise à jour elem et origine
            elem = matrice[ligne][colonne][0]
            ligne_courant, colonne_courant = ligne, colonne
            # print(elem)

            matrice[ligne][colonne][0] = "eau douce"

            matrice[ligne_courant][colonne_courant][3] = couleur
            matrice[ligne_courant][colonne_courant][4] = couleur
            matrice[ligne_courant][colonne_courant][5] = couleur

        # on enlève la source des sources possible pour plus tard
        liste_bloc.pop(origine)

    return matrice


def stat_par_biome(N, matrice):
    """Renvoie le nombre de case par biome 
    et les coord de chaque case pour chaque biome"""

    compteur_case_par_biome = np.zeros(len(Liste_biomes)+2)
    liste_case_par_biome = [[] for i in range(len(Liste_biomes)+2)]

    for ligne in range(N):
        for colonne in range(N):
            for biome in range(len(Liste_biomes)):
                if matrice[ligne][colonne][0] == Liste_biomes[biome][1]:
                    compteur_case_par_biome[biome] += 1
                    liste_case_par_biome[biome].append([ligne, colonne])
                elif matrice[ligne][colonne][0] == "eau":
                    compteur_case_par_biome[-2] += 1
                    liste_case_par_biome[-2].append([ligne, colonne])
                    break
                elif matrice[ligne][colonne][0] == "eau douce":
                    compteur_case_par_biome[-1] += 1
                    liste_case_par_biome[-1].append([ligne, colonne])
                    break

    return compteur_case_par_biome, liste_case_par_biome


def arbres(N, matrice_ajout, matrice):
    """
    Par biome
    On fait generation_centroid_bis sur toutes la carte
    On garde que les points dans les cellules du biome*land_mask*river_mask (si c'est pas de l'eau en fait)
    On place des points pour tester déja
    Si ca marche on fait un feuillage qui dépend de l'arbre"""

    # compter nombre de case par biome
    stat = stat_par_biome(N, matrice)
    # compteur, liste
    compteur_case_par_biome, liste_case_par_biome = stat[0], stat[1]

    # ARBRE
    for biome in range(len(Liste_biomes)):
        # on récupère les cases du biome
        liste_case_biome_arbre = liste_case_par_biome[biome]

        # ARBRES
        nombre_de_troncs = 0
        if Liste_biomes[biome][6] == "Elevée":
            nombre_de_troncs = int(compteur_case_par_biome[biome]/32)
        elif Liste_biomes[biome][6] == "Moyenne":
            nombre_de_troncs = int(compteur_case_par_biome[biome]/70)
        elif Liste_biomes[biome][6] == "Faible":
            nombre_de_troncs = int(compteur_case_par_biome[biome]/130)
        elif Liste_biomes[biome][6] == "Très faible":
            nombre_de_troncs = int(compteur_case_par_biome[biome]/160)
        if nombre_de_troncs == 0:  # ca veut dire quon a pas ce biome sur la carte
            continue

        # on attribue les arbres aléatoirement sur les cases du biomes en question
        for arbre in range(nombre_de_troncs):
            # on récupère l'indice de l'arbre
            indice_arbre = random.randint(0, len(liste_case_biome_arbre)-1)
            x_arbre, y_arbre = liste_case_biome_arbre[indice_arbre][0], liste_case_biome_arbre[indice_arbre][1]
            matrice_ajout[x_arbre][y_arbre].append("Arbre")
            matrice_ajout[x_arbre][y_arbre].append(Liste_biomes[biome][5])
            # on supprime l'arbre qui vient d'être placé
            liste_case_biome_arbre.pop(indice_arbre)

    # HERBE
    for biome in range(len(Liste_biomes)):
        # on récupère les cases du biome
        liste_case_biome_herbe = liste_case_par_biome[biome]

        # HERBES
        nombre_herbes = 0
        if Liste_biomes[biome][9] == "Elevée":
            nombre_herbes = int(compteur_case_par_biome[biome]/10)
        elif Liste_biomes[biome][9] == "Moyenne":
            nombre_herbes = int(compteur_case_par_biome[biome]/30)
        elif Liste_biomes[biome][9] == "Faible":
            nombre_herbes = int(compteur_case_par_biome[biome]/60)
        elif Liste_biomes[biome][9] == "Très faible":
            nombre_herbes = int(compteur_case_par_biome[biome]/90)
        if nombre_herbes == 0:  # ca veut dire quon a pas ce biome sur la carte
            continue

        # on attribue les herbes aléatoirement sur les cases du biomes en question
        for herbe in range(nombre_herbes):
            # on récupère l'indice de l'herbe
            indice_herbe = random.randint(0, len(liste_case_biome_herbe)-1)
            x_herbe, y_herbe = liste_case_biome_herbe[indice_herbe][0], liste_case_biome_herbe[indice_herbe][1]
            matrice_ajout[x_herbe][y_herbe].append("Herbe")
            matrice_ajout[x_herbe][y_herbe].append(Liste_biomes[biome][8])
            # on supprime l'arbre qui vient d'être placé
            liste_case_biome_herbe.pop(indice_herbe)

    return matrice_ajout


def visualise_map(N, matrice, matrice_ajout, new_H, H, hauteur_mer, seuil):
    """Permet de visualiser la carte. On visualise les bloc. C'est la fonctions définitive de visualisation de la carte 2D"""

    def design_arbre_1(libélé, coord):
        coord_x, coord_y = coord[0], coord[1]
        Liste_feuilles = []
        taille_moyen = 5
        taille_grand = 7
        seuil_disparition = 1

        proba_init = [0.7, 0.2, 0.1]
        proba_max = [0.1, 0.2, 0.7]

        # on récupère le libélé de l'abre dont il est question dans la base de données
        for vegetation in range(len(Liste_arbre)):
            if Liste_arbre[vegetation][0] == str(libélé):
                caracteristique = Liste_arbre[vegetation][1]
                valeurs = [Liste_arbre[vegetation][2],
                           Liste_arbre[vegetation][3], Liste_arbre[vegetation][4]]

                if caracteristique == "Petit":
                    Liste_feuilles.append([coord, Liste_arbre[vegetation][2]])
                    break
                elif caracteristique == "Moyen":
                    taille = taille_moyen
                elif caracteristique == "Grand":
                    taille = taille_grand
            else:
                continue

        # on se balade dans le feuillage
        origine_ligne = coord_x - 1
        origine_colonne = coord_y - 1
        for i in range(taille):
            for j in range(taille):
                if (origine_ligne + i) >= 0 and (origine_ligne + i) < N and (origine_colonne + j) >= 0 and (origine_colonne + j) < N:

                    # disparition des feuilles en bordure
                    nombre = random.random()
                    if (i == 0 or i == taille-1 or j == 0 or j == taille-1) and (nombre > seuil_disparition):
                        print(i, j)
                        continue
                    # si la feuille est toujours là
                    else:

                        # Calcul de la probabilité en fonction de la position dans la liste
                        proba = [
                            proba_init[k] + (proba_max[k] - proba_init[k]) * i / taille for k in range(3)]
                        # Sélection de la valeur en fonction de la probabilité croissante
                        rand = random.random()  # Génère un nombre aléatoire entre 0 et 1
                        if rand <= proba[0]:
                            valeur = valeurs[0]
                        elif rand <= proba[1]:
                            valeur = valeurs[1]
                        else:
                            valeur = valeurs[2]

                        Liste_feuilles.append(
                            [[origine_ligne + i, origine_colonne + j], valeur])

        return Liste_feuilles

    def design_arbre_2(libélé, coord):
        coord_x, coord_y = coord[0], coord[1]
        Liste_feuilles = []
        taille_moyen = 5
        taille_grand = 7
        seuil_disparition = 0.75  # plus c'est petit, plus on a de disparition des feuilles

        # on récupère le libélé de l'abre dont il est question dans la base de données
        for vegetation in range(len(Liste_arbre)):
            if Liste_arbre[vegetation][0] == str(libélé):
                caracteristique = Liste_arbre[vegetation][1]
                valeurs = [Liste_arbre[vegetation][2],
                           Liste_arbre[vegetation][3], Liste_arbre[vegetation][4]]
                if caracteristique == "Petit":
                    Liste_feuilles.append([coord, Liste_arbre[vegetation][2]])
                    return Liste_feuilles
                elif caracteristique == "Moyen":
                    taille = taille_moyen
                elif caracteristique == "Grand":
                    taille = taille_grand
            else:
                continue

        # on se balade dans le feuillage
        origine_ligne = coord_x - taille//2
        origine_colonne = coord_y - taille//2
        for i in range(taille):
            for j in range(taille):
                if (origine_ligne + i) >= 0 and (origine_ligne + i) < N and (origine_colonne + j) >= 0 and (origine_colonne + j) < N:
                    # disparition des feuilles en bordure
                    nombre = random.random()
                    if (i == 0 or i == taille-1 or j == 0 or j == taille-1) and (nombre > seuil_disparition):
                        continue
                    # si la feuille est toujours là
                    else:
                        couleur = random.choice(valeurs)

                    Liste_feuilles.append(
                        [[origine_ligne + i, origine_colonne + j], couleur])
        return Liste_feuilles

    facteur_hauteur_mer = new_H*100
    facteur_hauteur_terre = new_H*60
    carte = np.zeros((N, N, 3))
    Liste_troncs = []
    Liste_Herbe = []

    for ligne in range(N):
        for colonne in range(N):
            # afficher le nom du biome/bloc sur la case
            # plt.text(colonne, ligne, str(matrice[ligne][colonne][2]), horizontalalignment='center', verticalalignment='center')
            if matrice[ligne][colonne][0] == "eau":  # on dit que l'eau profonde est foncée

                couleur = matrice[ligne][colonne][3]
                carte[ligne][colonne] = couleur/255 + np.array(
                    [facteur_hauteur_mer[ligne][colonne], facteur_hauteur_mer[ligne][colonne], facteur_hauteur_mer[ligne][colonne]])/255
            else:
                # ON GERE LES RELIEFS ICI
                # on créer les moyennes des 3 cases au nord et au sud
                if ligne > 0 and ligne < N-1 and colonne > 0 and colonne < N-1:
                    moyenne_nord = (new_H[ligne-1][colonne-1]+new_H[ligne-1]
                                    [colonne]+new_H[ligne-1][colonne+1])/3  # le plus au nord
                    moyenne_sud = (new_H[ligne+1][colonne-1]+new_H[ligne+1]
                                   [colonne]+new_H[ligne+1][colonne+1])/3  # le plus au sud
                    moyenne_ouest = (new_H[ligne-1][colonne-1]+new_H[ligne]
                                     [colonne-1]+new_H[ligne+1][colonne-1])/3  # le plus au nord
                    moyenne_est = (new_H[ligne-1][colonne+1]+new_H[ligne]
                                   [colonne+1]+new_H[ligne+1][colonne+1])/3  # le plus au sud

                    # pente au nord
                    if moyenne_nord+seuil < new_H[ligne][colonne] < moyenne_sud-seuil:
                        carte[ligne][colonne] = (matrice[ligne][colonne][3])/255 - np.array([facteur_hauteur_terre[ligne]
                                                                                             [colonne], facteur_hauteur_terre[ligne][colonne], facteur_hauteur_terre[ligne][colonne]])/255
                    # pente au sud
                    elif moyenne_sud+seuil < new_H[ligne][colonne] < moyenne_nord-seuil:
                        carte[ligne][colonne] = (matrice[ligne][colonne][5])/255 - np.array([facteur_hauteur_terre[ligne]
                                                                                             [colonne], facteur_hauteur_terre[ligne][colonne], facteur_hauteur_terre[ligne][colonne]])/255
                    # pente à l'ouest
                    elif moyenne_ouest+seuil < new_H[ligne][colonne] < moyenne_est-seuil:
                        carte[ligne][colonne] = (matrice[ligne][colonne][5])/255 - np.array([facteur_hauteur_terre[ligne]
                                                                                             [colonne], facteur_hauteur_terre[ligne][colonne], facteur_hauteur_terre[ligne][colonne]])/255
                    # pente à l'est
                    elif moyenne_est+seuil < new_H[ligne][colonne] < moyenne_ouest-seuil:
                        carte[ligne][colonne] = (matrice[ligne][colonne][3])/255 - np.array([facteur_hauteur_terre[ligne]
                                                                                             [colonne], facteur_hauteur_terre[ligne][colonne], facteur_hauteur_terre[ligne][colonne]])/255
                    # plateau
                    else:
                        carte[ligne][colonne] = (matrice[ligne][colonne][4])/255 - np.array([facteur_hauteur_terre[ligne]
                                                                                             [colonne], facteur_hauteur_terre[ligne][colonne], facteur_hauteur_terre[ligne][colonne]])/255
                else:
                    carte[ligne][colonne] = (matrice[ligne][colonne][4])/255 - np.array([facteur_hauteur_terre[ligne]
                                                                                         [colonne], facteur_hauteur_terre[ligne][colonne], facteur_hauteur_terre[ligne][colonne]])/255
            # arbre
            if len(matrice_ajout[ligne][colonne]) != 0 and matrice_ajout[ligne][colonne][0] == "Arbre":
                Liste_troncs.append([ligne, colonne])
            # herbe
            elif len(matrice_ajout[ligne][colonne]) != 0 and matrice_ajout[ligne][colonne][0] == "Herbe":
                Liste_Herbe.append([ligne, colonne])
                # on récupère le libélé de l'herbe dont il est question dans la base de données
                libélé_herbe = matrice_ajout[ligne][colonne][1]
                for herbe in range(len(Liste_herbe)):
                    if Liste_herbe[herbe][0] == str(libélé_herbe):
                        carte[ligne][colonne] = Liste_herbe[herbe][2]/255

    # feuillage
    for tronc in range(len(Liste_troncs)):
        libélé = matrice_ajout[Liste_troncs[tronc]
                               [0]][Liste_troncs[tronc][1]][1]
        Liste_feuilles = design_arbre_2(
            libélé, [Liste_troncs[tronc][0], Liste_troncs[tronc][1]])
        for feuille in range(len(Liste_feuilles)):
            carte[Liste_feuilles[feuille][0][0]][Liste_feuilles[feuille]
                                                 [0][1]] = Liste_feuilles[feuille][1][0]/255

    plt.figure(figsize=(120, 120))
    plt.imshow(carte)
    # plt.savefig('C:/Users/ederw/OneDrive/Documents/Programmation/Python/Map aléatoire/mon_image.png',
    #             format='png', transparent=True)
    plt.show()

    # # # faire un plot 3D
    # # #plot pas imshow
    X, Y = [], []
    Z, Zp = [], []
    for ligne in range(N):
        for colonne in range(N):
            X.append(ligne)
            Y.append(colonne)
            if new_H[ligne][colonne] > hauteur_mer:
                Z.append(new_H[ligne][colonne])
            else:
                Z.append(-0.05)
            if H[ligne][colonne] > hauteur_mer:
                Zp.append(H[ligne][colonne])
            else:
                Zp.append(-0.05)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_zlim(-1, 1)
    ax.plot_trisurf(X, Y, Z)
    plt.show()


# =============================================================================
# Main
# =============================================================================
# np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning) #ignorer les warning
seed = 114
random.seed(seed)  # fonctionne dans la globalité pour la topologie
monde = world(80) # taille du monde

# Paramètres de perlin
mesh_T_P = 3  # on veut qu'une tache recouvre plusieurs cellules
nb_octave_T_P = 2
mesh_H = 4
nb_octave_H = 10
variation_brutale = 1.5  # plus c'est élevé plus les amplitudes de perlin du fractal motion sont divisées successivement et moins les bruits locals ont d'impact
mesh_blur = 15  # plus c'est grand plus les frontières sont blur localement
precision = 1

# paramètres de perlin bruit
mesh_H_bruit = mesh_H*2  # les bruits sont plus locaux
nb_octave_H_bruit = nb_octave_H
variation_brutale_bruit = variation_brutale
min_bruit = -0.2
max_bruit = 0.2
precision_bruit = precision + 1

# paramètres de perlin érrosion frontières
mesh_H_errosion = mesh_H*3  # les bruits sont plus locaux
nb_octave_H_errosion = nb_octave_H
variation_brutale_errosion = 1
min_errosion = 0
max_errosion = 1
precision_errosion = precision + 1

# Températures et précipitation
T_min = 0
T_max = 30  # T max est 30°
P_min = 0
P_max = 100  # P max est 100mm
H_max = 13  # ne sert à rien pour l'instant. Peut servire à afficher la matrice avec des hateurs sympa à la fin

# relief et frontières
nbr_cellules = 10 #plus il y a de cellules, plus il peut y avoir de biomes différents. Mais ca dépend aussi de la taille des cartes de Perlin de températures et précipitations 
deplacement_frontière_blur = 3
seuil_visualisation_relief = 0.007  # plus c'est élevé moins on vois le relief

# paramètres mer
# ne pas trop y toucher (si on le met trop bas on aura des plats et des creux d'eau d'un coup. TOUCHER min_hauteur_terre, ca revient au même
hauteur_mer = 0
# -1=normal / -0.3=Minecraft_zoomé (avec -1 on a des valeurs de Perlin entre [-1 , 1]) / ne pas faire plus que 1 (car bezier non défini au dessus)
min_hauteur_terre = -0.4
# on l'active on a que de la terre (les hauteurs de perlin neg sont multipliées par -1)
full_terre = False
archipel = False
largeur_eau = 3.3  # plus c'est élevé moins on est archipel. Ce facteur doit être augmenté si on basse min_terre

# paramètres rivières
taille_noyeau_cellule = 2
taille_noyeau_biome = 5
hauteur_max_riviere = 0.08  # 0.08 bien. Plus c'est haut, plus il y a de rivières
nb_cours_eau = 20
hauteur_cours_eau = 0.4

# paramètres plages
taille_noyau_plage = 7
# le sable apparait tans que la hauteur du bloc est inférieur à cette valeur
seuil_plage = (hauteur_mer) + 0.003

# génération
monde.generate(precision, nbr_cellules, mesh_H, nb_octave_H, mesh_H_bruit, nb_octave_H_bruit, variation_brutale_bruit, min_bruit, max_bruit, precision_bruit,  mesh_H_errosion, nb_octave_H_errosion, variation_brutale_errosion, min_errosion, max_errosion, precision_errosion, min_hauteur_terre, mesh_T_P, nb_octave_T_P, H_max, variation_brutale, mesh_blur, deplacement_frontière_blur,
               P_min, P_max, T_min, T_max, seuil_visualisation_relief, taille_noyeau_cellule, taille_noyeau_biome, hauteur_max_riviere, hauteur_mer, largeur_eau, nb_cours_eau, hauteur_cours_eau, taille_noyau_plage, seuil_plage, terre=full_terre, bool_archipel=archipel)  # nombre_de_centroides, mesh, precision, nbr_octave, hauteur_max, precipitation_max, temperature_max, hauteur_mer
# Il faut essayer de garder une proportion telle que N = 150 => Nbr centroid = 22
monde.visualise()

# np.savetxt('Carte_500x500.txt', monde.matrice)