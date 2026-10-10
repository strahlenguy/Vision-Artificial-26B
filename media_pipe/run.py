import cv2
import mediapipe as mp

P = mp.solutions.pose.PoseLandmark
pose = mp.solutions.pose.Pose()
dibujo = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

brazo_der = [P.RIGHT_SHOULDER, P.RIGHT_ELBOW, P.RIGHT_WRIST]  # 12, 14, 16
brazo_izq = [P.LEFT_SHOULDER, P.LEFT_ELBOW, P.LEFT_WRIST]     # 11, 13, 15


def punto(landmarks, idx, w, h):
    lm = landmarks.landmark[idx]
    return int(lm.x * w), int(lm.y * h)


while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        break
    res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if res.pose_landmarks:
        dibujo.draw_landmarks(frame, res.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)
        h, w = frame.shape[:2]
        for brazo in (brazo_der, brazo_izq):
            pts = [punto(res.pose_landmarks, i, w, h) for i in brazo]
            cv2.line(frame, pts[0], pts[1], (0, 255, 0), 3)
            cv2.line(frame, pts[1], pts[2], (0, 255, 0), 3)
    cv2.imshow("MediaPipe", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
