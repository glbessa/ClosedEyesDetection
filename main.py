from scipy.spatial import distance as dist
import cv2
import dlib
from imutils import face_utils
import argparse

SHAPE_PREDICTOR_PATH = './resources/shape_predictor_81_face_landmarks.dat'
LEFT_EYE_LANDMARK_START = 36
LEFT_EYE_LANDMARK_END = 42
RIGHT_EYE_LANDMARK_START = 42
RIGHT_EYE_LANDMARK_END = 48

EYE_AR_CONSEC_FRAMES = 3

def calculate_eye_aspect_ratio(eye_points):
    a = dist.euclidean(eye_points[1], eye_points[5])
    b = dist.euclidean(eye_points[2], eye_points[4])
    c = dist.euclidean(eye_points[0], eye_points[3])

    return (a + b) / (2.0 * c)

def main(eye_ar_thresh, eye_ar_consec_frames):
    counter = 0

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)

    video_stream = cv2.VideoCapture(0)

    while True:
        ret, frame = video_stream.read()

        frame = cv2.resize(frame, (450, 450))
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = detector(gray_frame, 0)
        for face in faces:
            cv2.rectangle(frame, (face.left(), face.top()), (face.left() + face.width(), face.top() + face.height()), (0, 255, 0), 2)

            facial_landmarks = predictor(gray_frame, face)
            facial_landmarks = face_utils.shape_to_np(facial_landmarks)

            if len(facial_landmarks) == 0:
                break

            left_eye = facial_landmarks[LEFT_EYE_LANDMARK_START:LEFT_EYE_LANDMARK_END]
            right_eye = facial_landmarks[RIGHT_EYE_LANDMARK_START:RIGHT_EYE_LANDMARK_END]

            left_ear = calculate_eye_aspect_ratio(left_eye)
            right_ear = calculate_eye_aspect_ratio(right_eye)

            ear = (left_ear + right_ear) / 2.0

            left_eye_hull = cv2.convexHull(left_eye)
            right_eye_hull = cv2.convexHull(right_eye)

            cv2.putText(frame, f"EAR: {round(ear, 2)}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color=(0, 255, 0))
            cv2.drawContours(frame, [left_eye_hull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [right_eye_hull], -1, (0, 255, 0), 1)

            if ear < eye_ar_thresh:
                counter += 1
                
                if counter >= eye_ar_consec_frames:
                    cv2.putText(frame, f"OLHOS FECHADOS!", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color=(0, 255, 0))
            else:
                counter = 0


        cv2.imshow("Closed eyes detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_stream.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--eye-ar-thresh", type=float, default=0.31, help="Adjust this thresh if the closed eye detection is not right.")
    args = vars(ap.parse_args())
    
    main(args['eye_ar_thresh'], EYE_AR_CONSEC_FRAMES)