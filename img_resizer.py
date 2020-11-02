import cv2
import numpy as np
def rescale_frame(img, percent=100):
    w = int(img.shape[1] * percent/100)
    h = int(img.shape[0] * percent/100)
    dim = (w, h)
    return cv2.resize(img, dim, interpolation =cv2.INTER_AREA)
for i in range(1,2501):
    if cv2.imread('Yes\{} E.jpg'.format(i)) is not None:
        frame = cv2.imread('Yes\{} E.jpg'.format(i))
    elif cv2.imread('Yes\{}  E.jpg'.format(i)) is not None:
        frame = cv2.imread('Yes\{}  E.jpg'.format(i))
    elif cv2.imread('Yes\{}B  E.jpg'.format(i)) is not None:
        frame = cv2.imread('Yes\{}B  E.jpg'.format(i))
    else:
        frame = cv2.imread('Yes\{} E.jpg'.format(i))

    mask = np.zeros((2200, 2200, 3), np.uint8)
    if frame.shape[0] == 1500 and frame.shape[1] == 300:
        mask = np.zeros((2200, 2200, 3), np.uint8)
    # elif frame.shape[0] > 4160 or frame.shape[1] > 6240:
    #     frame = rescale_frame(frame, 70)
    #     a, b, _ = frame.shape
    #     c, d, _ = mask.shape
    #     ha = np.rint(a / 2)
    #     hb = np.rint(b / 2)
    #     hc = np.rint(c / 2)
    #     hd = np.rint(d / 2)
    #     e = hc - ha
    #     f = e + a
    #     g = hd - hb
    #     h = g + b
    #     mask[int(e):int(f), int(g):int(h), :] = mask[int(e):int(f), int(g):int(h), :] + frame[0:int(a), 0:int(b), :]
    #     mask = rescale_frame(mask, 10)
    else:
        a, b, _ = frame.shape
        c, d, _ = mask.shape
        ha = np.rint(a / 2)
        hb = np.rint(b / 2)
        hc = np.rint(c / 2)
        hd = np.rint(d / 2)
        e = hc - ha
        f = e + a
        g = hd - hb
        h = g + b
        mask[int(e):int(f), int(g):int(h), :] = mask[int(e):int(f), int(g):int(h), :] + frame[0:int(a), 0:int(b), :]
    mask = rescale_frame(mask, 10)

    filename = '{} E.jpg'.format(i)
    cv2.imwrite(filename, mask)
    print('{} written'.format(i))