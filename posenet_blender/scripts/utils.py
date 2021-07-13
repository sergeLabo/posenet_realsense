
import json
# #import numpy as np

from bge import logic as gl


def read_json(fichier):
    try:
        with open(fichier) as f:
            data = json.load(f)
        f.close()
    except:
        data = None
        print("Fichier inexistant ou impossible à lire:")
    return data


def get_all_scenes():
    """Récupération des scènes"""
    # Liste des objets scènes
    activeScenes = gl.getSceneList()

    # Liste des noms de scènes
    scene_name = []
    for scn in activeScenes:
        scene_name.append(scn.name)

    return activeScenes, scene_name


def get_scene_with_name(scn):
    """Récupération de la scène avec le nom"""

    activeScenes, scene_name = get_all_scenes()
    if scn in scene_name:
        return activeScenes[scene_name.index(scn)]
    else:
        print(scn, "pas dans la liste")
        return None


def get_all_objects():
    """
    Trouve tous les objets des scènes actives
    Retourne un dict {nom de l'objet: blender object}
    """
    activeScenes, scene_name = get_all_scenes()

    all_obj = {}
    for scn_name in scene_name:
        scn = get_scene_with_name(scn_name)
        for blender_obj in scn.objects:
            blender_objet_name = blender_obj.name
            all_obj[blender_objet_name] = blender_obj

    return all_obj


def add_object(obj, position, life):
    """
    Ajoute obj à la place de Empty
    position liste de 3

    addObject(object, reference, time=0)
    Adds an object to the scene like the Add Object Actuator would.
    Parameters:
        object (KX_GameObject or string) – The (name of the) object to add.
        reference (KX_GameObject or string) – The (name of the) object which
        position, orientation, and scale to copy (optional), if the object
        to add is a light and there is not reference the light’s layer will be
        the same that the active layer in the blender scene.
        time (integer) – The lifetime of the added object, in frames. A time
        of 0 means the object will last forever (optional).

    Returns: The newly added object.
    Return type: KX_GameObject
    """

    gl.empty.worldPosition = position
    game_scn = get_scene_with_name("Scene")
    return game_scn.addObject(obj, gl.empty, life)


JOINTS = ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10",
            "11", "12", "13", "14", "15", "16", "17"]


# Non utilisé
"""
    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16



    EDGES = (   (0, 1),
            (0, 2),
            (0, 3),
            (0, 4),
            (3, 1),
            (4, 2),
            (1, 2),
            (5, 6),
            (5, 7),
            (5, 11),
            (6, 8),
            (6, 12),
            (7, 9),
            (8, 10),
            (11, 12),
            (11, 13),
            (12, 14),
            (13, 15),
            (14, 16))

"""
# Définition des points origine, direction des cubes de matérialisation des os
PAIRS = {  "upper_arm.L": [0, 1],
            "forearm.L": [0, 2],
            "upper_arm.R": [0, 3],
            "forearm.R": [0, 4],
            "thigh.L": [3, 1],
            "shin.L": [4, 2],
            "thigh.R": [1, 2],
            "shin.R": [5, 6],
            "shoulder.L": [5, 7],
            "shoulder.R": [5, 11],
            "tronc.L": [6, 8],
            "tronc.R": [6, 12],
            "bassin": [7, 9],
            "cou": [8, 10],
            "yeux": [11, 12],
            "oreille.R": [11, 13],
            "oreille.L": [12, 14],
            "head": [13, 15],
            "jambe.R": [14, 16]}

def get_points_blender(data):
    """frame_data = list(coordonnées des points empilés d'une frame
            soit 3*17 items avec:
            mutipliées par 1000
            les None sont remplacés par (-1000000, -1000000, -1000000)
            le numéro du body (dernier de la liste) doit être enlevé
        Conversion:
            Les coords sont multipliées par 1000 avant envoi en OSC
            Permutation de y et z, z est la profondeur pour RS et OpenCV
            et inversion de l'axe des y en z
            C'est la conversion de openpose en blender
    """

    if len(data) == 51:
        nombre = int(len(data)/3)
        points = []
        for i in range(nombre):
            # Reconstruction par 3
            val = [ data[(3*i)],
                    data[(3*i)+1],
                    data[(3*i)+2]]
            if val == [-1000000, -1000000, -1000000]:
                points.append(None)
            else:
                # Conversion cubemos vers blender
                points.append([val[0]/1000, val[2]/1000, -val[1]/1000])
    else:
        points = None

    return points
