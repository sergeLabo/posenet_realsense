# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Modification by La Labomedia July 2021


import os
import time
import enum

import numpy as np
import cv2
import pyrealsense2 as rs

from pose_engine import PoseEngine
from pose_engine import KeypointType

from myconfig import MyConfig
from osc import OscClient


EDGES = (
    (KeypointType.NOSE, KeypointType.LEFT_EYE),
    (KeypointType.NOSE, KeypointType.RIGHT_EYE),
    (KeypointType.NOSE, KeypointType.LEFT_EAR),
    (KeypointType.NOSE, KeypointType.RIGHT_EAR),
    (KeypointType.LEFT_EAR, KeypointType.LEFT_EYE),
    (KeypointType.RIGHT_EAR, KeypointType.RIGHT_EYE),
    (KeypointType.LEFT_EYE, KeypointType.RIGHT_EYE),
    (KeypointType.LEFT_SHOULDER, KeypointType.RIGHT_SHOULDER),
    (KeypointType.LEFT_SHOULDER, KeypointType.LEFT_ELBOW),
    (KeypointType.LEFT_SHOULDER, KeypointType.LEFT_HIP),
    (KeypointType.RIGHT_SHOULDER, KeypointType.RIGHT_ELBOW),
    (KeypointType.RIGHT_SHOULDER, KeypointType.RIGHT_HIP),
    (KeypointType.LEFT_ELBOW, KeypointType.LEFT_WRIST),
    (KeypointType.RIGHT_ELBOW, KeypointType.RIGHT_WRIST),
    (KeypointType.LEFT_HIP, KeypointType.RIGHT_HIP),
    (KeypointType.LEFT_HIP, KeypointType.LEFT_KNEE),
    (KeypointType.RIGHT_HIP, KeypointType.RIGHT_KNEE),
    (KeypointType.LEFT_KNEE, KeypointType.LEFT_ANKLE),
    (KeypointType.RIGHT_KNEE, KeypointType.RIGHT_ANKLE),
)


class KeypointType(enum.IntEnum):
    """Pose keypoints avec leur indice pour la liste OSC."""
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


class PoseRealsense:
    """Capture avec  Camera RealSense D455,
    détection de la pose avec Coral USB Stick,
    calcul des coordonnées 3D,
    et envoi en OSC des listes de ces coordonnées
    """

    def __init__(self, **kwargs):
        """Les paramètres sont à définir dans le fichier posenet.ini"""

        self.threshold = kwargs.get('threshold', 0.2)
        self.around = kwargs.get('around', 5)
        self.width = kwargs.get('width_input', 1280)
        self.height = kwargs.get('height_input', 720)

        self.set_pipeline()
        self.get_engine()
        self.osc = OscClient(**kwargs)
        self.get_colors()

    def get_colors(self):
        """Crée une liste de 5 couleur"""
        self.color = [[250, 0, 0], [0, 250, 0], [0, 0, 250],
                        [250, 250, 0],[120, 122, 120]]

    def get_engine(self):
        res = str(self.width) + 'x' + str(self.height)
        print("width:", self.width, ", height:", self.height)
        print("Résolution =", res)

        if res == "1280x720":
            self.src_size = (1280, 720)
            self.appsink_size = (1280, 720)
            model_size = (721, 1281)

        elif res == "640x480":
            self.src_size = (640, 480)
            self.appsink_size = (640, 480)
            model_size = (481, 641)

        else:
            print(f"La résolution {res} n'est pas possible.")
            os._exit(0)

        model = (f'models/mobilenet/posenet_mobilenet_v1_075_'
                 f'{model_size[0]}_{model_size[1]}'
                 f'_quant_decoder_edgetpu.tflite'   )
        print('Loading model: ', model)
        self.engine = PoseEngine(model, mirror=False)

    def set_pipeline(self):

        self.pipeline = rs.pipeline()
        config = rs.config()
        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        config.enable_stream(rs.stream.color, self.width, self.height,
                                                            rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, self.width, self.height,
                                                            rs.format.z16, 30)
        self.pipeline.start(config)
        self.align = rs.align(rs.stream.color)
        unaligned_frames = self.pipeline.wait_for_frames()
        frames = self.align.process(unaligned_frames)
        depth = frames.get_depth_frame()
        self.depth_intrinsic = depth.profile.as_video_stream_profile().intrinsics

        # Vérification de la taille des images
        color_frame = frames.get_color_frame()
        img = np.asanyarray(color_frame.get_data())
        print(f"Vérification de la taille des images:"
              f"     {img.shape[1]}x{img.shape[0]}")

    def run(self, **kwargs):

        t0 = time.time()
        nbr = 0
        while True:
            nbr += 1
            points_3D = None

            frames = self.pipeline.wait_for_frames()
            # Align the depth frame to color frame
            aligned_frames = self.align.process(frames)

            color = aligned_frames.get_color_frame()
            depth = aligned_frames.get_depth_frame()
            if not depth:
                continue

            color_data = color.as_frame().get_data()
            color_arr = np.asanyarray(color_data)

            outputs, inference_time = self.engine.DetectPosesInImage(color_arr)

            # Pour tous les personnages
            for pose in outputs:
                # Index dans la liste des poses
                ind = outputs.index(pose)
                # choix de la couleur
                col = outputs.index(pose) % 5

                # Récup des xys = {0: [790, 331], 2: [780, 313],  ... }
                xys = get_points_2D(outputs, threshold=self.threshold)
                # Seul le premier est converti en 3D, et envoyé en OSC
                if ind == 0:
                    points_2D = xys

                # Dessin en couleur
                draw_pose(color_arr,  xys, color=self.color[col])

            # Pour le premier
            if len(outputs) > 0:
                points_3D = get_points_3D(points_2D, depth,
                                            self.depth_intrinsic, self.around)
                if points_3D:
                    self.osc.send_global_message(points_3D, bodyId=0)

            cv2.imshow('color', color_arr)

            if time.time() - t0 > 1:
                print("FPS =", nbr)
                t0, nbr = time.time(), 0

            if cv2.waitKey(1) & 0xFF == 27:
                break

        cv2.destroyAllWindows()
        return


def draw_pose(img, xys, color):
    """Affiche les points 2D dans l'image
    xys = {0: [790, 331], 2: [780, 313],  ... }
    """
    points = []
    for xy in xys.values():
        points.append(xy)

    # Dessin des points
    for point in points:
        x = point[0]
        y = point[1]
        cv2.circle(img, (x, y), 5, color=(209, 156, 0), thickness=-1)
        cv2.circle(img, (x, y), 6, color=color, thickness=1)

    # Dessin des os
    for a, b in EDGES:
        if a not in xys or b not in xys: continue
        ax, ay = xys[a]
        bx, by = xys[b]
        cv2.line(img, (ax, ay), (bx, by), color, 2)


def get_points_2D(outputs, threshold=0.2):
    """Pour 1 personnage capté:
     [Pose(keypoints={
    <KeypointType.NOSE: 0>: Keypoint(point=Point(x=717.3, y=340.8), score=0.98),
    <KeypointType.LEFT_EYE: 1>: Keypoint(point=Point(x=716.2, y=312.5), score=0.6),
    <KeypointType.RIGHT_EYE: 2>: Keypoint(point=Point(x=699.6, y=312.8), score=0.98),
    <KeypointType.LEFT_EAR: 3>: Keypoint(point=Point(x=720.13306, y=314.34964)
                    },
    score=0.34098125)]
    xys = {0: (698, 320), 1: (698, 297), 2: (675, 295), .... }
    """

    pose = outputs[0]
    xys = {}
    for label, keypoint in pose.keypoints.items():
        if keypoint.score > threshold:
            xys[label.value] = [int(keypoint.point[0]), int(keypoint.point[1])]
    return xys


def get_points_3D(xys, depth, depth_intrinsic, around):
    """Calcul des coordonnées 3D dans un repère centré sur la caméra,
    avec le z = profondeur
    La profondeur est une moyenne de la profondeur des points autour,
    sauf les trop loins et les extrêmes.
    """

    points_3D = [None]*17
    for key, val in xys.items():
        if val:
            #
            distance_in_square = []
            x, y = val[0], val[1]
            # nombre de pixel autour du points = 5
            x_min = max(x - around, 0)
            x_max = min(x + around, depth.width)
            y_min = max(y - around, 0)
            y_max = min(y + around, depth.height)

            for u in range(x_min, x_max):
                for v in range(y_min, y_max):
                    distance_in_square.append(depth.get_distance(u, v))

            if distance_in_square:
                dists = np.asarray(distance_in_square)
            else:
                dists = None

            if dists.any():
                # Suppression du plus petit et du plus grand
                dists = np.sort(dists)
                dists = dists[1:-1]
                # Moyenne
                average = np.average(dists)
                # Exclusion des trop éloignés
                reste = dists[ (dists >= average*0.8) & (dists <= average*1.2) ]
                # Eloignement estimé du points
                profondeur = np.average(reste)

                # Calcul les coordonnées 3D avec x et y coordonnées dans
                # l'image et la profondeur du point
                point_with_deph = rs.rs2_deproject_pixel_to_point(depth_intrinsic,
                                                                  [x, y],
                                                                  profondeur)
                if not np.isnan(point_with_deph[0]):
                    points_3D[key] = point_with_deph

    return points_3D


if __name__ == '__main__':

    ini_file = 'posenet.ini'
    my_config = MyConfig(ini_file)
    kwargs = my_config.conf['pose_camera_realsense_cv']
    print(f"Configuration:\n{kwargs}\n\n")


    pose_realsense = PoseRealsense(**kwargs)
    pose_realsense.run()
