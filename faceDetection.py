import cv2

trained_face_data = cv2.CascadeClassifier(
    'haarcascade_frontalface_default.xml')

webCam = cv2.VideoCapture(0)

while True:
    successfulFrameRead, img = webCam.read()

    grayScaledImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faceCoordinates = trained_face_data.detectMultiScale(grayScaledImage)

    for (x, y, w, h) in faceCoordinates:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)

    cv2.imshow('Muneeba', img)
    key = cv2.waitKey(1)

    if key == 81  or key == 113:
        break


print("Code Completed!")