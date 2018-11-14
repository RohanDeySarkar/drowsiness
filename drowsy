from scipy.spatial import distance
from imutils import face_utils
import imutils
import dlib
import cv2

def eye_aspect_ratio(eye):

    A = distance.euclidean(eye[1], eye[5]) # Acc to formula 
    B = distance.euclidean(eye[2], eye[4]) # start from pt 0 to 5
    C = distance.euclidean(eye[0], eye[3]) # EAR = eye aspect ratio

    ear = (A + B) / (2.0 * C)

    return ear

thresh = 0.25
frame_check = 20
detect = dlib.get_frontal_face_detector()  # Returns the default face detector
predict = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]  # dimensions of start and end of left eye
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"] # dimensions of start and end of right eye

cap = cv2.VideoCapture(0)
flag=0

while True:
	ret, frame=cap.read()
	frame = imutils.resize(frame, width=450) # resizing the frame
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	subjects = detect(gray, 0)
	
	for subject in subjects:
            
		shape = predict(gray, subject)
		shape = face_utils.shape_to_np(shape) # converting to numpy array
		
		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]       # getting the left_eye from the default shape
		
		leftEAR = eye_aspect_ratio(leftEye) # will have 6 pts, which will later calculate EAR
		rightEAR = eye_aspect_ratio(rightEye)
		
		ear = (leftEAR + rightEAR) / 2.0  # avg EAR
		
		leftEyeHull = cv2.convexHull(leftEye)  # convex bulged out structure around left eye
		rightEyeHull = cv2.convexHull(rightEye)
		
		cv2.drawContours(frame, [leftEyeHull], -1, (255, 0, 0), 1)  # colors the convex_hull
		cv2.drawContours(frame, [rightEyeHull], -1, (255, 0, 0), 1)
		
		if ear < thresh:
			flag += 1
			#print (flag)
			if flag >= frame_check:
				cv2.putText(frame, "****************ALERT!****************", (10, 30),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
				cv2.putText(frame, "****************ALERT!****************", (10,325),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		else:
			flag = 0
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		cv2.destroyAllWindows()
		cap.release()
		break




