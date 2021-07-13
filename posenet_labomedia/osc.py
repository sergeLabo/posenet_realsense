

from oscpy.client import OSCClient

class OscClient:
    """Client OSC pour envoyer les points de la capture avec Realsense,
    possibilté de sauvegarder une capture dans un json pour développer
    """

    def __init__(self, **kwargs):

        self.ip = kwargs.get('ip', None)
        self.port = kwargs.get('port', None)
        # Pour l'enregistrement d'un json à la fin de la capture
        self.all_data = []
        # Pour envoi multiple, si point différent de plus de 1%
        self.previous = None

        self.client = OSCClient(self.ip, self.port)

    def send_global_message(self, points3D, bodyId=110):
        """Envoi du point en OSC en 3D
            Liste de n°body puis toutes les coordonnées sans liste de 3
            oscpy n'envoie pas de liste de listes
        """

        msg = []
        for point in points3D:
            if point:
                for i in range(3):
                    # Envoi en int
                    msg.append(int(point[i]*1000))
            # Si pas de point ajout arbitraire de 3 fois -1000000
            # pour avoir toujours 3*18 valeurs dans la liste
            else:
                msg.extend((-1000000, -1000000, -1000000))  # tuple ou list

        # N° body à la fin
        msg.append(bodyId)
        self.all_data.append(msg)
        self.client.send_message(b'/points', msg)

    def save(self):
        dt_now = datetime.now()
        dt = dt_now.strftime("%Y_%m_%d_%H_%M")
        fichier = f"./json/cap_{dt}.json"
        with open(fichier, "w") as fd:
            fd.write(dumps(self.all_data))
            print(f"{fichier} enregistré.")
        fd.close()
