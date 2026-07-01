#logique de MediaPipe
import math
import os

import cv2

from mediapipe import Image, ImageFormat
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision as mp_vision

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "pose_landmarker_lite.task")

# Connexions du squelette (33 points BlazePose), fournies par la nouvelle API Tasks
POSE_CONNECTIONS = [(c.start, c.end) for c in mp_vision.PoseLandmarksConnections.POSE_LANDMARKS]


class _Landmark:
    """Reproduit l'interface (.x, .y, .z, .visibility) de l'ancienne API solutions."""
    __slots__ = ("x", "y", "z", "visibility")

    def __init__(self, lm):
        self.x = lm.x
        self.y = lm.y
        self.z = lm.z
        self.visibility = lm.visibility


class _PoseLandmarks:
    """Reproduit results.pose_landmarks.landmark de l'ancienne API solutions."""

    def __init__(self, landmarks):
        self.landmark = [_Landmark(lm) for lm in landmarks]


class _PoseResult:
    """Reproduit l'objet results renvoyé par pose.process() dans l'ancienne API."""

    def __init__(self, task_result):
        if task_result.pose_landmarks:
            self.pose_landmarks = _PoseLandmarks(task_result.pose_landmarks[0])
        else:
            self.pose_landmarks = None


def _draw_landmarks(img, pose_landmarks, couleur_point=(0, 0, 255), couleur_ligne=(0, 255, 0)):
    h, w = img.shape[:2]
    points = [(int(lm.x * w), int(lm.y * h)) for lm in pose_landmarks.landmark]

    for start, end in POSE_CONNECTIONS:
        if start < len(points) and end < len(points):
            cv2.line(img, points[start], points[end], couleur_ligne, 2)
    for x, y in points:
        cv2.circle(img, (x, y), 3, couleur_point, cv2.FILLED)


class poseDetector():
    def __init__(self, mode=False, upBody=False, smooth=True, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        running_mode = mp_vision.RunningMode.IMAGE if mode else mp_vision.RunningMode.VIDEO

        options = mp_vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=running_mode,
            num_poses=1,
            min_pose_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon,
            output_segmentation_masks=self.upBody,
        )
        self.landmarker = mp_vision.PoseLandmarker.create_from_options(options)
        self._running_mode = running_mode
        self._timestamp_ms = 0

        self.results = None
        self.lmList = []
        self.lmList3D = []

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = Image(image_format=ImageFormat.SRGB, data=imgRGB)

        if self._running_mode == mp_vision.RunningMode.VIDEO:
            self._timestamp_ms += 33
            task_result = self.landmarker.detect_for_video(mp_image, self._timestamp_ms)
        else:
            task_result = self.landmarker.detect(mp_image)

        self.results = _PoseResult(task_result)

        if self.results.pose_landmarks:
            if draw:
                _draw_landmarks(img, self.results.pose_landmarks)
        return img

    def findPosition(self, img, draw=True, couleur=(0, 0, 0)):
        self.lmList = []
        self.lmList3D = []
        if self.results and self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                # print(id, lm)
                cx, cy = int(lm.x * w), int(lm.y * h)
                # z est une profondeur relative aux hanches, sur ~la même échelle que x
                cz = round(lm.z * w, 1)
                self.lmList.append([id, cx, cy])
                self.lmList3D.append([id, cx, cy, cz])
                if draw:
                    cv2.circle(img, (cx, cy), 5, couleur, cv2.FILLED)
        return self.lmList

    def findAngle(self, img, p1, p2, p3, draw=True):

        # Get the landmarks
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        x3, y3 = self.lmList[p3][1:]

        # Calculate the Angle
        angle = math.degrees(math.atan2(y3 - y2, x3 - x2) -
                             math.atan2(y1 - y2, x1 - x2))
        if angle < 0:
            angle += 360

        # Draw
        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
            cv2.line(img, (x3, y3), (x2, y2), (255, 255, 255), 3)
            cv2.circle(img, (x1, y1), 3, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x1, y1), 5, (0, 0, 255), 2)
            cv2.circle(img, (x2, y2), 3, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 5, (0, 0, 255), 2)
            cv2.circle(img, (x3, y3), 3, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x3, y3), 5, (0, 0, 255), 2)
            cv2.putText(img, str(int(angle)), (x2 - 50, y2 + 50),
                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        return angle

    def findAngle3D(self, p1, p2, p3):
        """Angle (en degres) au point p2, calcule en 3D avec x, y et z.
        Plus fiable que findAngle quand le corps n'est pas face a la camera,
        car il n'est pas fausse par la projection sur le plan de l'image."""
        x1, y1, z1 = self.lmList3D[p1][1:]
        x2, y2, z2 = self.lmList3D[p2][1:]
        x3, y3, z3 = self.lmList3D[p3][1:]

        v1 = (x1 - x2, y1 - y2, z1 - z2)
        v2 = (x3 - x2, y3 - y2, z3 - z2)

        norm1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2 + v1[2] ** 2)
        norm2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2 + v2[2] ** 2)
        if norm1 == 0 or norm2 == 0:
            return 0.0

        dot = v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]
        cos_angle = max(-1.0, min(1.0, dot / (norm1 * norm2)))
        return math.degrees(math.acos(cos_angle))

    def findCenter(self, img, p1, p2, draw=True, couleur=(0, 0, 255)):

        # Get the landmarks
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]

        x3, y3 = (x1+x2)//2, (y1+y2)//2

        # Draw
        if draw:
            cv2.line(img, (x1, y1), (x2, y2), couleur, 2)
            cv2.circle(img, (x1, y1), 3, couleur, cv2.FILLED)
            cv2.circle(img, (x2, y2), 3, couleur, cv2.FILLED)
            cv2.circle(img, (x2, y2), 3, couleur, cv2.FILLED)
            cv2.circle(img, (x2, y2), 5, couleur, 2)
            cv2.putText(img, f'({x3}, {y3})', (x3, y3), cv2.FONT_HERSHEY_PLAIN, 2, couleur,2)

        return x3, y3

    def findDistance(self, img, p1, p2, draw=True, couleur=(0, 0, 255)):

        # Get the landmarks
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]

        # Calculate the euclidian distance
        distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

        # Draw
        if draw:
            cv2.line(img, (x1, y1), (x2, y2), couleur, 2)
            cv2.circle(img, (x1, y1), 3, couleur, cv2.FILLED)
            cv2.circle(img, (x1, y1), 5, couleur, 2)
            cv2.circle(img, (x2, y2), 3, couleur, cv2.FILLED)
            cv2.circle(img, (x2, y2), 5, couleur, 2)
            cv2.putText(img, str(int(distance)), ((x1+x2)//2, (y1+y2)//2), cv2.FONT_HERSHEY_PLAIN, 2, couleur, 2)
        return distance


def detection_pose(complexity=1):
    #Initialisation du model Pose (complexity ignoré, conservé pour compatibilité)
    options = mp_vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.5,
    )
    return mp_vision.PoseLandmarker.create_from_options(options)

#traite et retourne une image annotée ainsi que les résultats
def processus_par_image(image, pose_model):
    #MediaPipe nécessite du RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = Image(image_format=ImageFormat.SRGB, data=image_rgb)

    #execution de l'analyse
    task_result = pose_model.detect(mp_image)
    results = _PoseResult(task_result)

    #on travaille sur une copie pour ne pas modifier l'originale
    image_annoter = image.copy()

    #On dessine les lignes
    if results.pose_landmarks:
        _draw_landmarks(image_annoter, results.pose_landmarks)

    return image_annoter, results
