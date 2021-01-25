import cv2
import numpy as np
import dlib
import pandas as pd
from imutils import face_utils
import os
from object_detection.utils import label_map_util
from object_detection.builders import model_builder
from object_detection.utils import config_util
import tensorflow as tf
from tensorflow.keras.models import load_model

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('track dependencies\shape_predictor_68_face_landmarks.dat')
CONFIG_PATH = 'pupil track dependencies/pipeline.config'
CHECKPOINT_PATH = 'pupil track dependencies/'
WEIGHT_PATH = "4layer/Weights-9788--0.92333--0.96371.hdf5"

configs = config_util.get_configs_from_pipeline_file(CONFIG_PATH)
detection_model = model_builder.build(model_config=configs['model'], is_training=False)
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(CHECKPOINT_PATH, 'ckpt-7')).expect_partial()
category_index = label_map_util.create_category_index_from_labelmap('pupil track dependencies/label_map.pbtxt')
model = load_model('4layer/Model.h5')
model.load_weights(WEIGHT_PATH)

@tf.function
def detect_pupil(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections

def main_pupil(frame):
    image_np = np.array(frame)
    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections = detect_pupil(input_tensor)
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    box = np.squeeze(detections['detection_boxes'])
    j = []
    for i in range(3):
        ymin = (int(box[i, 0] * 480))
        xmin = (int(box[i, 1] * 640))
        ymax = (int(box[i, 2] * 480))
        xmax = (int(box[i, 3] * 640))
        z = (xmin, ymin, xmax, ymax)
        j.append(z)
    return j[0][0], j[0][1], j[0][2], j[0][3], j[1][0], j[1][1], j[1][2], j[1][3], j[2][0], j[2][1], j[2][2], j[2][3]

def get_face_points(face):
    (x, y, w, h) = face_utils.rect_to_bb(face)
    return x,y,w,h

def get_head_pose(img, facial_landmarks):
    hp = [30,8,36,45,59,55]
    size = img.shape
    image_points = np.array([(facial_landmarks.part(point).x, facial_landmarks.part(point).y) for point in hp], dtype="double")
    model_points = np.array([(0.0, 0.0, 0.0),(0.0, -330.0, -65.0),(-225.0, 170.0, -135.0),(225.0, 170.0, -135.0),(-150.0, -150.0, -125.0),(150.0, -150.0, -125.0)])
    focal_length = size[1]
    center = (size[1] / 2, size[0] / 2)
    camera_matrix = np.array([[focal_length, 0, center[0]],[0, focal_length, center[1]],[0, 0, 1]], dtype="double")
    dist_coeffs = np.zeros((4, 1))
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)
    (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector,
                                                     translation_vector, camera_matrix, dist_coeffs)
    p1 = (int(image_points[0][0]), int(image_points[0][1]))
    p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
    return p1[0],p1[1],p2[0],p2[1]

def get_eye_points(img, facial_landmarks):
    lp = [36, 37, 38, 39, 40, 41]
    rp = [42, 43, 44, 45, 46, 47]
    mask = np.zeros((480, 640), np.uint8)

    lr = np.array([(facial_landmarks.part(point).x, facial_landmarks.part(point).y) for point in lp])
    rr = np.array([(facial_landmarks.part(point).x, facial_landmarks.part(point).y) for point in rp])

    mask1 = cv2.fillPoly(mask, [lr], 255)
    mask1 = cv2.fillPoly(mask1, [rr], 255)
    eye = cv2.bitwise_and(img, img, mask=mask1)

    leftEyeCenter = lr.mean(axis=0).astype("int")
    rightEyeCenter = rr.mean(axis=0).astype("int")
    dY = rightEyeCenter[1] - leftEyeCenter[1]
    dX = rightEyeCenter[0] - leftEyeCenter[0]
    angle = np.degrees(np.arctan2(dY, dX))
    angle = round(angle, 2)

    return eye, angle

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    key = cv2.waitKey(1)
    # cv2.imshow('ouput', frame)

    if (len(faces) >= 1):
        for face in faces:
            landmarks = predictor(gray, face)
            CC, angle=get_eye_points(frame, landmarks)
            x,y,w,he = get_face_points(face)
            x0,y0,x1,y1 = get_head_pose(gray, landmarks)
            a,b,c,d,e,f,g,h,i,j,k,l = main_pupil(CC)
            m = np.array([a,b,c,d,e,f,g,h,i,j,k,l,angle, x,y,w,he,x0,y0,x1,y1])
            m = tf.convert_to_tensor(np.expand_dims(m, 0), dtype=tf.float32)
            n = model.predict(m)
            if (n > 0.5):
                print('Yes')
            else:
                print('No')

    else:
        continue

    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        break