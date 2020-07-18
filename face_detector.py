import cv2
from random import randrange

trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


# choose image to detect faces
# img = cv2.imread('RDJ.jpg')
img = cv2.imread('group.jpg')

# convert image to grayscale-color
grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect Faces
face_coordinates = trained_face_data.detectMultiScale(grayscaled_img) # puts all the faces coordinates in array 

#Draw rectangles around the faces
# cv2.rectangle(img,(35,78),( 258+35, 258+78),(0,255,0),3) # 3 is the thickness of rectangle,
# x,y,w,h = face_coordinates[0]
# cv2.rectangle(img,(x,y),(x+w, y+h),(0,255,0),5)

for x,y,w,h in face_coordinates:
	cv2.rectangle(img,(x,y),(x+w, y+h),(randrange(128, 256),randrange(128, 256),randrange(128, 256)),5)


# print(face_coordinates)

#Display the images with the faces
cv2.imshow('Robert Downey Junior', img)



cv2.waitKey()
print("Code Completed!")