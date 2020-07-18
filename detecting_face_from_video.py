import cv2
from random import randrange

# load trained face data
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


# choose image to detect faces
# img = cv2.imread('RDJ.jpg')
# img = cv2.imread('group.jpg')

# reading video
webcam = cv2.VideoCapture(0)

while True:
	# Read the current frame
	successful_frame_read, frame = webcam.read()

	# convert image to grayscale-color
	grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	
	# Detect Face
	face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)
	
	# assigning face coordinates
	x,y,w,h = face_coordinates[0]

	# drawing rectangle around face
	cv2.rectangle(frame,(x,y),(x+w, y+h),(randrange(128,256), randrange(128,256), randrange(128,256)),5)

	# showing the image
	cv2.imshow("Prem Kumar Tamang", frame)
	
	key = cv2.waitKey(1)

	# stop if 0 is pressed
	if key == 81 and key == 113:
		break

webcam.release()