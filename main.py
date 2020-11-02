import cv2
import numpy as np
import dlib

cap = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("track dependencies\shape_predictor_68_face_landmarks.dat")

def get_eye_mask(img,facial_landmarks, points, overlay):
    eye_region1 = np.array([(facial_landmarks.part(points[0]).x, facial_landmarks.part(points[0]).y), \
                                (facial_landmarks.part(points[1]).x, facial_landmarks.part(points[1]).y), \
                                (facial_landmarks.part(points[2]).x, facial_landmarks.part(points[2]).y), \
                                (facial_landmarks.part(points[3]).x, facial_landmarks.part(points[3]).y), \
                                (facial_landmarks.part(points[4]).x, facial_landmarks.part(points[4]).y), \
                                (facial_landmarks.part(points[5]).x, facial_landmarks.part(points[5]).y)])
    eye_region2 = np.array([(facial_landmarks.part(points[6]).x, facial_landmarks.part(points[6]).y), \
                                (facial_landmarks.part(points[7]).x, facial_landmarks.part(points[7]).y), \
                                (facial_landmarks.part(points[8]).x, facial_landmarks.part(points[8]).y), \
                                (facial_landmarks.part(points[9]).x, facial_landmarks.part(points[9]).y), \
                                (facial_landmarks.part(points[10]).x, facial_landmarks.part(points[10]).y), \
                                (facial_landmarks.part(points[11]).x, facial_landmarks.part(points[11]).y)])
    cv2.polylines(overlay, [eye_region1], True, 255, 2)
    cv2.fillPoly(overlay, [eye_region1], 255)
    cv2.polylines(overlay, [eye_region2], True, 255, 2)
    cv2.fillPoly(overlay, [eye_region2], 255)
    eye_n = cv2.bitwise_and(img, img, mask=overlay)
    mn_x = np.min(eye_region1[:, 0] - 5)
    mx_x = np.max(eye_region2[:, 0] + 5)
    mn_y = np.min(eye_region1[:, 1] - 5)
    mx_y = np.max(eye_region2[:, 1] + 5)
    eye_n = eye_n[mn_y:mx_y, mn_x:mx_x]
    return eye_n
img_counter = 0
# calib = cv2.imread('caliberate.png' )
# cv2.imshow("Frame", calib)

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    key = cv2.waitKey(1)
    if key == 13:
        break
    for face in faces:
        landmarks = predictor(gray, face)
        x1,y1 = landmarks.part(36).x - 5, landmarks.part(38).y - 5
        x2,y2 = landmarks.part(45).x + 5, landmarks.part(46).y + 5
        #cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        eye_region_n = [36,37,38,39,40,41,42,43,44,45,46,47]
        height, width, a = frame.shape
        mask = np.zeros((height, width), np.uint8)
        full_eye = get_eye_mask(frame, landmarks, eye_region_n, mask)
        a,b,_ = full_eye.shape
        if (a and b) > 0:
            cv2.imshow("Full_eye", full_eye)
        else:
            continue
        if key % 256 == 32:\

            # SPACE pressed
            img_name = "opencv_frame_{}.png".format(img_counter)
            cv2.imwrite(img_name, full_eye)
            print("{} written!".format(img_name))
            img_counter += 1

    #cv2.imshow("Frame", frame)
# for i in range(9,10):
#     frame = cv2.imread('from camera\Capture0000{}.jpg'.format(i) )
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = detector(gray)
#     key = cv2.waitKey(1)
#     if key == 13:
#         break
#     for face in faces:
#         landmarks = predictor(gray, face)
#         x1, y1 = landmarks.part(36).x - 5, landmarks.part(38).y - 5
#         x2, y2 = landmarks.part(45).x + 5, landmarks.part(46).y + 5
#         # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
#         eye_region_n = [36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]
#         height, width, a = frame.shape
#         mask = np.zeros((height, width), np.uint8)
#         full_eye = get_eye_mask(frame, landmarks, eye_region_n, mask)
#         a, b, _ = full_eye.shape
#         if (a and b) > 0:
#             cv2.imshow("eye{}".format(i), full_eye)
#         else:
#             continue
#         img_name = "opencv_frame_{}.png".format(i)
#         cv2.imwrite(img_name, full_eye)
#         print("{} written!".format(i))

    # cv2.imshow("Frame", frame)



cap.release()
cv2.destroyAllWindows()