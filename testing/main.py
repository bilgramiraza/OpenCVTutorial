import cv2
import random
import numpy as np
# img = cv2.imread(
#     '/home/razaworksystem/Desktop/projects/OpenCVTutorial/assets/bird.jpg', cv2.IMREAD_COLOR)
# imgSmall = cv2.resize(img, (300, 300))

# imgHalf = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
# img90 = cv2.rotate(imgHalf, cv2.ROTATE_90_CLOCKWISE)
# cv2.imwrite('bird90.jpg', img90)
# imgRandom = cv2.resize(img, (300, 300))
# for i in range(imgRandom.shape[0]):
#     for j in range(imgRandom.shape[1]):
#         imgRandom[i][j] = [255-imgRandom[i][j][0], 255 -
#                            imgRandom[i][j][1], 255-imgRandom[i][j][2]]
# imgFlip = np.zeros((imgSmall.shape[0], imgSmall.shape[1], 3), np.uint8)

# for i in range(imgSmall.shape[0]):
#     for j in range(imgSmall.shape[1]):
#         flipRow = imgSmall.shape[0]-i-1
#         flipCol = imgSmall.shape[1]-j-1
#         imgFlip[i][j] = [imgSmall[flipRow][flipCol][0],
#                          imgSmall[flipRow][flipCol][1],
#                          imgSmall[flipRow][flipCol][2]]
# cv2.imshow('img', imgSmall)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Getting Video Capture and flipping it all around
# cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read()
#     width = int(cap.get(3))
#     height = int(cap.get(4))
#     canvas = np.zeros(frame.shape, np.uint8)

#     shrunk_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

#     top_right = shrunk_frame
#     top_left = cv2.flip(shrunk_frame, 1)
#     bottom_right = cv2.flip(shrunk_frame, 0)
#     bottom_left = cv2.flip(shrunk_frame, -1)
#     canvas[:height//2, :width//2] = top_left
#     canvas[:height//2, width//2:] = top_right
#     canvas[height//2:, :width//2] = bottom_left
#     canvas[height//2:, width//2:] = bottom_right

#     cv2.imshow('Frame', canvas)

#     if cv2.waitKey(1) == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()

# # adding lines and shapes to video capture
# cap = cv2.VideoCapture(0)
# while True:
#     ret, frame = cap.read()
#     width = int(cap.get(3))
#     height = int(cap.get(4))

#     width_quarter = int(width//4)
#     height_quarter = int(height//4)
#     width_half = int(width//2)
#     height_half = int(height//2)
#     width_three_quarter = int(width_quarter*3)
#     height_three_quarter = int(height_quarter*3)
#     radius = int(((width_half**2+height_half**2)**(1/2))//2)
#     font = cv2.FONT_HERSHEY_SIMPLEX

#     image = cv2.line(frame, (0, 0), (width, height), (127, 127, 127), 6)
#     image = cv2.line(image, (0, height), (width, 0), (127, 127, 127), 6)

#     image = cv2.rectangle(image, (width_quarter, height_quarter), (width_three_quarter,
#                           height_three_quarter), (127, 127, 127), 6)

#     image = cv2.circle(image, (width_half, height_half),
#                        radius, (127, 127, 127), 6)

#     image = cv2.putText(image, 'Video Drawing Tutorial',
#                         (width_quarter, 30), font, 1, (127, 127, 127), 3, cv2.LINE_AA)

#     canvas = image
#     cv2.imshow('Video', canvas)

#     if cv2.waitKey(1) == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()

# # Color Detection
# cap = cv2.VideoCapture(0)

# color = np.uint8([[[255, 0, 0]]])  # Selected BGR Color
# color_hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)[0][0]
# color_hue = color_hsv[0]

# while True:
#     ret, frame = cap.read()
#     width = int(cap.get(3))
#     height = int(cap.get(4))

#     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

#     lower_color = np.array([color_hue-30, 50, 50])
#     upper_color = np.array([color_hue+10, 255, 255])

#     mask_layer = cv2.inRange(hsv, lower_color, upper_color)

#     canvas = cv2.bitwise_and(frame, frame, mask=mask_layer)

#     cv2.imshow('Video', frame)
#     cv2.imshow('mask', mask_layer)
#     cv2.imshow('filter', canvas)
#     if cv2.waitKey(1) == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()

# # Corner Detection
# image = cv2.imread(
#     '/home/razaworksystem/Desktop/projects/OpenCVTutorial/assets/chessboard.png')
# img = cv2.resize(image, (0, 0), fx=0.6, fy=0.6)

# gray_scale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# total_corners = 100
# confidence = 0.01
# min_distance = 10

# corners = cv2.goodFeaturesToTrack(
#     gray_scale_img, total_corners, confidence, min_distance)

# corners = np.intp(corners)

# circle_radius = 5
# circle_color = (255, 0, 0)
# circle_fill = -1

# for corner in corners:
#     x, y = corner.ravel()
#     cv2.circle(img, (x, y), circle_radius, circle_color, circle_fill)

# for i in range(len(corners)):
#     for j in range(i+1, len(corners)):
#         corner1 = tuple(corners[i][0])
#         corner2 = tuple(corners[j][0])
#         color = [int(x) for x in np.random.randint(0, 255, size=3)]
#         cv2.line(img, corner1, corner2, color, 1)


# cv2.imshow('baseImage', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # Template Object Detection

# img = cv2.resize(cv2.imread(
#     '/home/razaworksystem/Desktop/projects/OpenCVTutorial/assets/soccer_practice.jpg', 0), (0, 0), fx=0.8, fy=0.8)
# template = cv2.resize(cv2.imread(
#     '/home/razaworksystem/Desktop/projects/OpenCVTutorial/assets/shoe.png', 0), (0, 0), fx=0.8, fy=0.8)

# h, w = template.shape

# methods = [cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR,
#            cv2.TM_CCORR_NORMED, cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]

# for method in methods:
#     img2 = img.copy()

#     result = cv2.matchTemplate(img2, template, method)
#     min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
#     if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
#         location_top_left = min_loc
#     else:
#         location_top_left = max_loc
#     location_bottom_right = (location_top_left[0]+w, location_top_left[1]+h)
#     cv2.rectangle(img2, location_top_left, location_bottom_right, 255, 5)
#     cv2.imshow('Match', img2)
#     cv2.waitKey(0)

# Face and Eye Detection
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades+'haarcascade_eye.xml')

while True:
    ret, frame = cap.read()

    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_img, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 5)

        roi_gray = gray_img[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(roi_gray, 1.05, 6)

        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 5)

    cv2.imshow('canvas', frame)

    if (cv2.waitKey(1) == ord('q')):
        break

cap.release()
cv2.destroyAllWindows()
