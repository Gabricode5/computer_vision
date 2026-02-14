#logique de MediaPipe
import cv2
import math

# Importations forcées pour éviter l'AttributeError sur Python 3.12
from mediapipe.python.solutions import pose as mp_pose
from mediapipe.python.solutions import drawing_utils as mp_drawing

class poseDetector():
    def __init__(self, mode=False, upBody=False, smooth=True, detectionCon=0.5, trackCon=0.5):
        # On utilise DIRECTEMENT les imports forcés mp_pose et mp_drawing
        self.mpPose = mp_pose 
        self.mpDraw = mp_drawing
        
        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        # --- LES LIGNES CI-DESSOUS ONT ÉTÉ SUPPRIMÉES POUR ÉVITER L'ERREUR ---
        
        self.pose = self.mpPose.Pose(
            static_image_mode=self.mode,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=self.upBody,
            smooth_segmentation=self.smooth,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon)
        self.results = None
        self.lmList = []

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)

        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return img

    def findPosition(self, img, draw=True, couleur=(0, 0, 0)):
        self.lmList = []
        if self.results and self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                # print(id, lm)
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
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
    #Initialisation du model Pose
    return mp_pose.Pose(
        static_image_mode=True,
        model_complexity=complexity,
        min_detection_confidence=0.5
    )

#traite et retourne une image annotée ainsi que les résultats
def processus_par_image(image, pose_model):
    #MediaPipe nécessite du RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    #execution de l'analyse
    results = pose_model.process(image_rgb)

    #on travaille sur une copie pour ne pas modifier l'originale
    image_annoter = image.copy()

    #On dessine les lignes
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image_annoter, 
            results.pose_landmarks, 
            mp_pose.POSE_CONNECTIONS
        )

    return image_annoter, results
