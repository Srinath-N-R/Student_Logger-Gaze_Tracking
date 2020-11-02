import cv2
import numpy as np
import dlib
import face_recognition as fc
from imutils import face_utils

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('track dependencies\shape_predictor_68_face_landmarks.dat')

def rescale_frame(img, percent=100):
    w = int(img.shape[1] * percent/100)
    h = int(img.shape[0] * percent/100)
    dim = (w, h)
    return cv2.resize(img, dim, interpolation =cv2.INTER_AREA)
def resize_eye(eye):
    mask1 = np.zeros((25, 180, 3), np.uint8)
    a, b, _ = eye.shape
    c, d, _ = mask1.shape
    ha = np.rint(a / 2)
    hb = np.rint(b / 2)
    hc = np.rint(c / 2)
    hd = np.rint(d / 2)
    e = hc - ha
    f = e + a
    g = hd - hb
    h = g + b
    lm = f-a
    if lm == eye.shape[0]:
        eye = mask1[int(e):int(f), int(g):int(h), :] = mask1[int(e):int(f), int(g):int(h), :] + eye[:, :, :]
    return eye

def get_eye_mask(img, facial_landmarks, points, overlay):
    eye_region1 = np.array([(facial_landmarks.part(points[0]).x, facial_landmarks.part(points[0]).y),
                                (facial_landmarks.part(points[1]).x, facial_landmarks.part(points[1]).y), \
                                (facial_landmarks.part(points[2]).x, facial_landmarks.part(points[2]).y), \
                                (facial_landmarks.part(points[3]).x, facial_landmarks.part(points[3]).y), \
                                (facial_landmarks.part(points[4]).x, facial_landmarks.part(points[4]).y), \
                                (facial_landmarks.part(points[5]).x, facial_landmarks.part(points[5]).y)])
    eye_region2 = np.array([(facial_landmarks.part(points[6]).x, facial_landmarks.part(points[6]).y),
                                (facial_landmarks.part(points[7]).x, facial_landmarks.part(points[7]).y), \
                                (facial_landmarks.part(points[8]).x, facial_landmarks.part(points[8]).y), \
                                (facial_landmarks.part(points[9]).x, facial_landmarks.part(points[9]).y), \
                                (facial_landmarks.part(points[10]).x, facial_landmarks.part(points[10]).y), \
                                (facial_landmarks.part(points[11]).x, facial_landmarks.part(points[11]).y)])
    eye_region1 = eye_region1 * 1
    eye_region2 = eye_region2 * 1
    eye_region1[0, 0] -= 5
    eye_region1[1, 1] -= 5
    eye_region1[2, 1] -= 5
    eye_region1[3, 0] += 5
    eye_region1[4, 1] += 7.5
    eye_region1[5, 1] += 7.5
    eye_region2[0, 0] -= 5
    eye_region2[1, 1] -= 5
    eye_region2[2, 1] -= 5
    eye_region2[3, 0] += 5
    eye_region2[4, 1] += 7.5
    eye_region2[5, 1] += 7.5
    cv2.polylines(overlay, [eye_region1], True, 255, 2)
    cv2.fillPoly(overlay, [eye_region1], 255)
    cv2.polylines(overlay, [eye_region2], True, 255, 2)
    cv2.fillPoly(overlay, [eye_region2], 255)
    eye_n = cv2.bitwise_and(img, img, mask = overlay)
    mn_x = np.min(eye_region1[:, 0] - 1)
    mx_x = np.max(eye_region2[:, 0] + 1)
    mn_y = np.min(eye_region1[:, 1] - 1)
    mx_y = np.max(eye_region2[:, 1] + 1)
    eye_n = eye_n[mn_y:mx_y, mn_x:mx_x]
    return eye_n
def get_face_mask_1(img, c, overlay):
    face_region = np.array([(c[0,3] , c[0,0] ), (c[0,1] , c[0,0] ), (c[0,1] , c[0,2] ), (c[0,3], c[0,2] )])
    # face_region = np.array([(c[0,3] - 100, c[0,0] - 100), (c[0,1] + 100, c[0,0] - 100), (c[0,1] + 100, c[0,2] + 100), (c[0,3] - 100, c[0,2] + 100)])
    cv2.polylines(overlay, [face_region], True, 255, 2)
    cv2.fillPoly(overlay, [face_region], 255)
    face_f = cv2.bitwise_and(img, img, mask=overlay)
    return face_f

def get_face_mask_2(img, x, y, w, h, overlay):
    x = x * 1
    y = y * 1
    w = w * 1
    h = h * 1
    face_region = np.array([(x, y), (x + w, y), (x +w, y + h), (x, y + h)])
    cv2.polylines(overlay, [face_region], True, 255, 2)
    cv2.fillPoly(overlay, [face_region], 255)
    face_f = cv2.bitwise_and(img, img, mask=overlay)
    return face_f

empty_1 = np.zeros((25, 180), np.uint8)
empty_2 = np.zeros((480, 640), np.uint8)
eye_region_n = [36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]
img_counter = 1

cap = cv2.VideoCapture(0)

while True:
    img_name_1 = "{} E.jpg".format(img_counter)
    img_name_2 = "{} F.jpg".format(img_counter)
    img_name_3 = "{} EB.jpg".format(img_counter)
    img_name_4 = "{} FB.jpg".format(img_counter)
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces_E = detector(gray)
    faces_F = fc.face_locations(gray)
    fl = np.array(faces_F)
    fl = fl * 1
    height, width = gray.shape
    key = cv2.waitKey(1)

    if (len(fl) >= 1) and (len(faces_E) == 1):
        for face in faces_E:
            mask = np.zeros((height, width), np.uint8)
            landmarks = predictor(gray, face)
            full_eye = get_eye_mask(frame, landmarks, eye_region_n, mask)
            full_eye = resize_eye(full_eye)
            mask = np.zeros((height, width), np.uint8)
            full_face = get_face_mask_1(frame, fl, mask)
            a, b, _ = full_eye.shape
            a1, b1, _ = full_face.shape
            if (a and b and a1 and b1) > 0:
                cv2.imwrite(img_name_1, full_eye)
                cv2.imwrite(img_name_2, full_face)
                img_counter += 1
                print("eye {} written!".format(img_counter))
                print("face {} written!".format(img_counter))
                break
            else:
                cv2.imwrite(img_name_3, empty_1)
                cv2.imwrite(img_name_4, empty_2)
                img_counter += 1
                print("eye {} written!".format(img_counter))
                print("face {} written!".format(img_counter))

    elif (len(fl) == 0) and (len(faces_E) == 1):
        for face in faces_E:
            mask = np.zeros((height, width), np.uint8)
            landmarks = predictor(gray, face)
            full_eye = get_eye_mask(frame, landmarks, eye_region_n, mask)
            full_eye = resize_eye(full_eye)
            mask = np.zeros((height, width), np.uint8)
            c = face_utils.rect_to_bb(face)
            c = np.array(c)
            full_face = get_face_mask_2(frame, c[0], c[1], c[2], c[3], mask)
            a, b, _ = full_eye.shape
            a1, b1, _ = full_face.shape
            if (a and b and a1 and b1) > 0:
                cv2.imwrite(img_name_1, full_eye)
                cv2.imwrite(img_name_2, full_face)
                img_counter += 1
                print("eye {} written!".format(img_counter))
                print("face {} written!".format(img_counter))
                break
            else:
                cv2.imwrite(img_name_3, empty_1)
                cv2.imwrite(img_name_4, empty_2)
                img_counter += 1
                print("eye {} written!".format(img_counter))
                print("face {} written!".format(img_counter))
    elif (len(fl) >= 1) and (len(faces_E) != 1):
        mask = np.zeros((height, width), np.uint8)
        full_face = get_face_mask_1(frame, fl, mask)
        a, b, _ = full_face.shape
        if (a and b) > 0:
            cv2.imwrite(img_name_3, empty_1)
            cv2.imwrite(img_name_2, full_face)
            print("eye {} written!".format(img_counter))
            print("face {} written!".format(img_counter))
            break
        else:
            cv2.imwrite(img_name_3, empty_1)
            cv2.imwrite(img_name_4, empty_2)
            img_counter += 1
            print("eye {} written!".format(img_counter))
            print("face {} written!".format(img_counter))

    else:
        cv2.imwrite(img_name_3, empty_1)
        cv2.imwrite(img_name_4, empty_2)
        img_counter += 1
        print("eye {} written!".format(img_counter))
        print("face {} written!".format(img_counter))

    if key == 13:
        break