from statistics import mode

import cv2
from keras.models import load_model
import numpy as np
import dlib
import time
import math

BLINK_RATIO_THRESHOLD = 5.7


# -----Step 5: Getting to know blink ratio

def midpoint(point1, point2):
    return (point1.x + point2.x) / 2, (point1.y + point2.y) / 2


def euclidean_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def get_blink_ratio(eye_points, facial_landmarks):
    # loading all the required points
    corner_left = (facial_landmarks.part(eye_points[0]).x,
                   facial_landmarks.part(eye_points[0]).y)
    corner_right = (facial_landmarks.part(eye_points[3]).x,
                    facial_landmarks.part(eye_points[3]).y)

    center_top = midpoint(facial_landmarks.part(eye_points[1]),
                          facial_landmarks.part(eye_points[2]))
    center_bottom = midpoint(facial_landmarks.part(eye_points[5]),
                             facial_landmarks.part(eye_points[4]))

    # calculating distance
    horizontal_length = euclidean_distance(corner_left, corner_right)
    vertical_length = euclidean_distance(center_top, center_bottom)

    ratio = horizontal_length / vertical_length

    return ratio

detector = dlib.get_frontal_face_detector()

# -----Step 4: Detecting Eyes using landmarks in dlib-----
predictor = dlib.shape_predictor("D:/face_classification-master/src/models/shape_predictor_68_face_landmarks.dat")
# these landmarks are based on the image above
left_eye_landmarks = [36, 37, 38, 39, 40, 41]
right_eye_landmarks = [42, 43, 44, 45, 46, 47]

waterstart = time.time()
eyestrainstart = time.time()

from utils.datasets import get_labels
from utils.inference import detect_faces
from utils.inference import draw_text
from utils.inference import draw_bounding_box
from utils.inference import apply_offsets
from utils.inference import load_detection_model
from utils.preprocessor import preprocess_input

# parameters for loading data and images
detection_model_path = 'D:/face_classification-master/trained_models/detection_models/haarcascade_frontalface_default.xml'
emotion_model_path = 'D:/face_classification-master/trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5'
emotion_labels = get_labels('fer2013')

# hyper-parameters for bounding boxes shape
frame_window = 10
emotion_offsets = (20, 40)

# loading models
face_detection = load_detection_model(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)

# getting input model shapes for inference
emotion_target_size = emotion_classifier.input_shape[1:3]

# starting lists for calculating modes
emotion_window = []

def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((68, 2), dtype=dtype)
    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    # return the list of (x, y)-coordinates
    return coords


def eye_on_mask(mask, side):
    points = [shape[i] for i in side]
    points = np.array(points, dtype=np.int32)
    mask = cv2.fillConvexPoly(mask, points, 255)
    return mask


def contouring(thresh, mid, img, right=False):
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    try:
        cnt = max(cnts, key=cv2.contourArea)
        M = cv2.moments(cnt)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        if right:
            cx += mid
        cv2.circle(img, (cx, cy), 4, (0, 0, 255), 2)
        coords = [cx, cy]
        return coords

    except:
        pass


file = 'D:/face_classification-master/src/models/shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(file)

left = [36, 37, 38, 39, 40, 41]
right = [42, 43, 44, 45, 46, 47]


# starting video streaming

blinkcount = 0
video_capture = cv2.VideoCapture(0)
ret, img = video_capture.read()
thresh = img.copy()
cv2.namedWindow('window_frame')

minutetimer = time.time()

kernel = np.ones((9, 9), np.uint8)
def nothing(x):
    pass
cv2.createTrackbar('threshold', 'image', 0, 255, nothing)
x = 0
fineye = [300, 300, 200, 200]
while True:
    ret, img = video_capture.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)


    frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # -----Step 3: Face detection with dlib-----
    # detecting faces in the frame
    faces, _, _ = detector.run(image=frame, upsample_num_times=0,
                               adjust_threshold=0.0)

    # -----Step 4: Detecting Eyes using landmarks in dlib-----
    for face in faces:

        landmarks = predictor(frame, face)

        # -----Step 5: Calculating blink ratio for one eye-----
        left_eye_ratio = get_blink_ratio(left_eye_landmarks, landmarks)
        right_eye_ratio = get_blink_ratio(right_eye_landmarks, landmarks)
        blink_ratio = (left_eye_ratio + right_eye_ratio) / 2

        if blink_ratio > BLINK_RATIO_THRESHOLD:
            # Blink detected! Do Something!
            blinkcount += 1
            print(blinkcount)
            cv2.putText(frame, "BLINKING", (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        2, (255, 255, 255), 2, cv2.LINE_AA)

    for rect in rects:
        shape = predictor(gray, rect)
        shape = shape_to_np(shape)
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        mask = eye_on_mask(mask, left)
        mask = eye_on_mask(mask, right)
        mask = cv2.dilate(mask, kernel, 5)
        eyes = cv2.bitwise_and(img, img, mask=mask)
        mask = (eyes == [0, 0, 0]).all(axis=2)
        eyes[mask] = [255, 255, 255]
        mid = (shape[42][0] + shape[39][0]) // 2
        eyes_gray = cv2.cvtColor(eyes, cv2.COLOR_BGR2GRAY)
        threshold = 115
        _, thresh = cv2.threshold(eyes_gray, threshold, 255, cv2.THRESH_BINARY)
        thresh = cv2.erode(thresh, None, iterations=2)  # 1
        thresh = cv2.dilate(thresh, None, iterations=4)  # 2
        thresh = cv2.medianBlur(thresh, 3)  # 3
        thresh = cv2.bitwise_not(thresh)
        lefteye = contouring(thresh[:, 0:mid], mid, img)
        righteye = contouring(thresh[:, mid:], mid, img, True)
        if x == 0:
            fineye = [righteye[0], righteye[0], righteye[1], righteye[1]]
            x += 1

        if righteye is None:
            righteye = []
            righteye.append(fineye[0])
            righteye.append(fineye[2])

        if righteye[0] < fineye[0]:
            fineye[0] = righteye[0]
        elif righteye[0] > fineye[1]:
            fineye[1] = righteye[0]

        if righteye[1] < fineye[2]:
            fineye[2] = righteye[1]
        elif righteye[1] > fineye[3]:
            fineye[3] = righteye[1]

        if fineye[1] - fineye[0] >= 30:
            state = "NO"
            fineye = [righteye[0], righteye[0], righteye[1], righteye[1]]
            print(state)
        elif fineye[3] - fineye[2] >= 30:
            state = "YES"
            fineye = [righteye[0], righteye[0], righteye[1], righteye[1]]
            print(state)
        else:
            state = "INCONCLUSIVE"
        # for (x, y) in shape[36:48]:
        #     cv2.circle(img, (x, y), 2, (255, 0, 0), -1)
    # show the image with the face detections + facial landmarks

    bgr_image = video_capture.read()[1]
    gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    faces = detect_faces(face_detection, gray_image)

    for face_coordinates in faces:

        x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
        gray_face = gray_image[y1:y2, x1:x2]
        try:
            gray_face = cv2.resize(gray_face, (emotion_target_size))
        except:
            continue

        gray_face = preprocess_input(gray_face, True)
        gray_face = np.expand_dims(gray_face, 0)
        gray_face = np.expand_dims(gray_face, -1)
        emotion_prediction = emotion_classifier.predict(gray_face)
        emotion_probability = np.max(emotion_prediction)
        emotion_label_arg = np.argmax(emotion_prediction)
        emotion_text = emotion_labels[emotion_label_arg]
        emotion_window.append(emotion_text)

        if len(emotion_window) > frame_window:
            emotion_window.pop(0)
        try:
            emotion_mode = mode(emotion_window)
        except:
            continue

        if emotion_text == 'angry':
            color = emotion_probability * np.asarray((255, 0, 0))
        elif emotion_text == 'sad':
            color = emotion_probability * np.asarray((0, 0, 255))
        elif emotion_text == 'happy':
            color = emotion_probability * np.asarray((255, 255, 0))
        elif emotion_text == 'surprise':
            color = emotion_probability * np.asarray((0, 255, 255))
        else:
            color = emotion_probability * np.asarray((0, 255, 0))

        color = color.astype(int)
        color = color.tolist()

        draw_bounding_box(face_coordinates, rgb_image, color)
        draw_text(face_coordinates, rgb_image, emotion_mode,
                  color, 0, -45, 1, 1)

    waterend = time.time()
    eyestrainend = time.time()

    if (waterend - waterstart > 3600):
        print("DRINK WATER")
        waterstart = time.time()

    if (eyestrainend - eyestrainstart > 7200):
        print("TAKE A BREAK")
        eyestrainstart = time.time()

    checkminute = time.time()
    if checkminute - minutetimer > 60:
        minutetimer = time.time()
        if blinkcount < 15:
            print("BLINK MORE")
        blinkcount = 0

    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    added_image = cv2.addWeighted(img, 0.4, bgr_image, 0.5, 0)
    cv2.imshow('window_frame', added_image)
    # cv2.imshow('eyes', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()