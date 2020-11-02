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
    eye_region1[0, 0] -= 15
    eye_region1[1, 1] -= 15
    eye_region1[2, 1] -= 15
    eye_region1[3, 0] += 15
    eye_region1[4, 1] += 25
    eye_region1[5, 1] += 25
    eye_region2[0, 0] -= 15
    eye_region2[1, 1] -= 15
    eye_region2[2, 1] -= 15
    eye_region2[3, 0] += 15
    eye_region2[4, 1] += 25
    eye_region2[5, 1] += 25
    cv2.polylines(overlay, [eye_region1], True, 255, 2)
    cv2.fillPoly(overlay, [eye_region1], 255)
    cv2.polylines(overlay, [eye_region2], True, 255, 2)
    cv2.fillPoly(overlay, [eye_region2], 255)
    eye_n = cv2.bitwise_and(img, img, mask = overlay)
    mn_x = np.min(eye_region1[:, 0] - 5)
    mx_x = np.max(eye_region2[:, 0] + 5)
    mn_y = np.min(eye_region1[:, 1] - 5)
    mx_y = np.max(eye_region2[:, 1] + 5)
    eye_n = eye_n[mn_y:mx_y, mn_x:mx_x]
    return eye_n
def get_face_mask_1(img, c, overlay):
    # face_region = np.array([(c[0,3] , c[0,0] ), (c[0,1] , c[0,0] ), (c[0,1] , c[0,2] ), (c[0,3], c[0,2] )])
    face_region = np.array([(c[0,3] - 100, c[0,0] - 100), (c[0,1] + 100, c[0,0] - 100), (c[0,1] + 100, c[0,2] + 100), (c[0,3] - 100, c[0,2] + 100)])
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

# for i in range(1,992):
#     frame = cv2.imread('fuji_no\{}.jpg'.format(i))
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     height, width = gray.shape
#     gray = rescale_frame(gray, 100)
#     faces_E = detector(gray)
#     faces_F = fc.face_locations(gray)
#     fl = np.array(faces_F)
#     fl = fl * 1
#     img_name_1 = 'eye_converted_{}.jpg'.format(i)
#     img_name_2 = 'face_converted_{}.jpg'.format(i)
#     img_name_3 = 'eye_converted_{}_B.jpg'.format(i)
#     img_name_4 = 'face_converted_{}_B.jpg'.format(i)
#     empty_1 = np.zeros((1500, 300), np.uint8)
#     empty_2 = np.zeros((1500, 1500), np.uint8)
#     eye_region_n = [36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]
#
#     if (len(fl) >= 1) and (len(faces_E) == 1):
#         for face in faces_E:
#             mask = np.zeros((height, width), np.uint8)
#             landmarks = predictor(gray, face)
#             full_eye = get_eye_mask(frame, landmarks, eye_region_n, mask)
#             a, b, _ = full_eye.shape
#             if (a and b) > 0:
#                 cv2.imwrite(img_name_1, full_eye)
#             else:
#                 cv2.imwrite(img_name_1, empty_1)
#             print("eye {} written!".format(i))
#
#         mask = np.zeros((height, width), np.uint8)
#         full_face = get_face_mask_1(frame, fl, mask)
#         a, b, _ = full_face.shape
#         if (a and b) > 0:
#             cv2.imwrite(img_name_2, full_face)
#             print("face {} written!".format(i))
#         else:
#             cv2.imwrite(img_name_2, empty_2)
#             print("face {} written!".format(i))
#
#     elif (len(fl) == 0) and (len(faces_E) == 1):
#         for face in faces_E:
#             mask = np.zeros((height, width), np.uint8)
#             landmarks = predictor(gray, face)
#             full_eye = get_eye_mask(frame, landmarks, eye_region_n, mask)
#             a, b, _ = full_eye.shape
#             if (a and b) > 0:
#                 cv2.imwrite(img_name_1, full_eye)
#             else:
#                 cv2.imwrite(img_name_1, empty_1)
#             print("eye {} written!".format(i))
#
#             mask = np.zeros((height, width), np.uint8)
#             c = face_utils.rect_to_bb(face)
#             c = np.array(c)
#             full_face = get_face_mask_2(frame, c[0], c[1], c[2], c[3], mask)
#             a, b, _ = full_face.shape
#             if (a and b) > 0:
#                 cv2.imwrite(img_name_2, full_face)
#             else:
#                 cv2.imwrite(img_name_2, empty_2)
#             print("face {} written!".format(i))
#
#     elif (len(fl) >= 1) and (len(faces_E) != 1):
#         cv2.imwrite(img_name_1, empty_1)
#         print("eye {} written!".format(i))
#         mask = np.zeros((height, width), np.uint8)
#         full_face = get_face_mask_1(frame, fl, mask)
#         a, b, _ = full_face.shape
#         if (a and b) > 0:
#             cv2.imwrite(img_name_2, full_face)
#             print("face {} written!".format(i))
#         else:
#             cv2.imwrite(img_name_2, empty_2)
#             print("face {} written!".format(i))
#
#     else:
#         print(len(fl))
#         print(len(faces_E))
#         cv2.imwrite(img_name_3, empty_1)
#         print("eye {} written!".format(i))
#         cv2.imwrite(img_name_4, empty_2)
#         print("face {} written!".format(i))

img_counter = 0
video_capture = cv2.VideoCapture(0)
cv2.namedWindow("Window")
frame = cv2.imread('Yes 36.jpg')
while True:
    ret, frame1 = video_capture.read()
    key = cv2.waitKey(1)
    cv2.imshow("Window", frame1)
    if key % 256 == 27:
        frame = frame1
    cv2.imshow("Frame", frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    gray = rescale_frame(gray, 100)
    faces_E = detector(gray)
    faces_F = fc.face_locations(gray)
    fl = np.array(faces_F)
    fl = fl * 1
    # img_name_1 = 'eye_converted_{}.jpg'.format(i)
    # img_name_2 = 'face_converted_{}.jpg'.format(i)
    # img_name_3 = 'eye_converted_{}_B.jpg'.format(i)
    # img_name_4 = 'face_converted_{}_B.jpg'.format(i)
    empty_1 = np.zeros((1500, 300), np.uint8)
    empty_2 = np.zeros((1500, 1500), np.uint8)
    eye_region_n = [36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]
    if key == 13:
        break
    else:
        continue

    if (len(fl) >= 1) and (len(faces_E) == 1):
        for face in faces_E:
            mask = np.zeros((height, width), np.uint8)
            landmarks = predictor(gray, face)
            full_eye = get_eye_mask(frame, landmarks, eye_region_n, mask)
            a, b, _ = full_eye.shape
            if (a and b) > 0:
                half = full_eye
                # cv2.imwrite(img_name_1, full_eye)
            else:
                half = empty_1
            #     cv2.imwrite(img_name_1, empty_1)
            # print("eye {} written!".format(i))

        mask = np.zeros((height, width), np.uint8)
        full_face = get_face_mask_1(frame, fl, mask)
        a, b, _ = full_face.shape
        if (a and b) > 0:
            full = full_face
            # cv2.imwrite(img_name_2, full_face)
            # print("face {} written!".format(i))
        else:
            full = empty_2
            # cv2.imwrite(img_name_2, empty_2)
            # print("face {} written!".format(i))

    elif (len(fl) == 0) and (len(faces_E) == 1):
        for face in faces_E:
            mask = np.zeros((height, width), np.uint8)
            landmarks = predictor(gray, face)
            full_eye = get_eye_mask(frame, landmarks, eye_region_n, mask)
            a, b, _ = full_eye.shape
            if (a and b) > 0:
                half = full_eye
                # cv2.imwrite(img_name_1, full_eye)
            else:
                half = empty_1
            #     cv2.imwrite(img_name_1, empty_1)
            # print("eye {} written!".format(i))

            mask = np.zeros((height, width), np.uint8)
            c = face_utils.rect_to_bb(face)
            c = np.array(c)
            full_face = get_face_mask_2(frame, c[0], c[1], c[2], c[3], mask)
            a, b, _ = full_face.shape
            if (a and b) > 0:
                full = full_face
                # cv2.imwrite(img_name_2, full_face)
            else:
                full = empty_2
            #     cv2.imwrite(img_name_2, empty_2)
            # print("face {} written!".format(i))

    elif (len(fl) >= 1) and (len(faces_E) != 1):
        half = empty_1
        # cv2.imwrite(img_name_1, empty_1)
        # print("eye {} written!".format(i))
        mask = np.zeros((height, width), np.uint8)
        full_face = get_face_mask_1(frame, fl, mask)
        a, b, _ = full_face.shape
        if (a and b) > 0:
            full = full_face
            # cv2.imwrite(img_name_2, full_face)
            # print("face {} written!".format(i))
        else:
            full = empty_2
            cv2.imwrite(img_name_2, empty_2)
            print("face {} written!".format(i))

    else:
        half = empty_1
        full = empty_2
        # print(len(fl))
        # print(len(faces_E))
        # cv2.imwrite(img_name_3, empty_1)
        # print("eye {} written!".format(i))
        # cv2.imwrite(img_name_4, empty_2)
        # print("face {} written!".format(i))
    if key % 256 == 32:
        # SPACE pressed
        img_name_1 = "{} E.jpg".format(img_counter)
        img_name_2 = "{} F.jpg".format(img_counter)
        cv2.imwrite(img_name_1, half)
        cv2.imwrite(img_name_2, full)
        print("{} written!".format(img_counter))
        img_counter += 1

cap.release()
cv2.destroyAllWindows()