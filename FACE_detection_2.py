import cv2
import face_recognition

known_image = face_recognition.load_image_file("Person2.jpg")
known_encodings = face_recognition.face_encodings(known_image)[0]

video = cv2.VideoCapture(0)

while True:
    ret, frame = video.read()
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (left, top, right, bottom), face_encodings in zip(face_locations, face_encodings):
        match = face_recognition.compare_faces([known_encodings], face_encodings)

        if match[0]:
            label ="Authorized"
            color = (0,255,0)
        else:
            label = "Unauthorized"
            color = (0,0,255)

        cv2.rectangle(frame, (left, top), (right, bottom), color , 2)
        cv2.putText(frame,  label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(1)== ord("q"):
        break

video.release()
cv2.destroyAllWindows()