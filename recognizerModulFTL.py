import cv2
import dlib
import os
import speech_recognition as sr
import pygame
from pygame.locals import *

# Set up face recognition model (dlib's pre-trained model)
detector = dlib.get_frontal_face_detector()
face_recognizer = dlib.face_recognition_model_v1("path/to/dlib_face_recognition_resnet_model_v1.dat")

# Create a database to store face encodings
face_database = {}

# Initialize the speech recognition
recognizer = sr.Recognizer()

def encode_face(image):
    # Detect face
    faces = detector(image, 1)

    # Assume there is only one face in the image for simplicity
    if len(faces) == 1:
        # Encode face
        face_encoding = face_recognizer.compute_face_descriptor(image, faces[0])
        return face_encoding
    else:
        return None

def recognize_face(encoding):
    # Compare the encoding with the ones in the database
    for name, face_db_encoding in face_database.items():
        if dlib.face_distance([face_db_encoding], encoding) < 0.6:
            return name
    return None

def add_to_database(name, encoding):
    face_database[name] = encoding

# Initialize Pygame for touch events
pygame.init()

# Open camera
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Find face encoding
    encoding = encode_face(frame)

    if encoding is not None:
        # Recognize face
        name = recognize_face(encoding)

        if name is not None:
            print(f"Welcome back, {name}!")

            # Audio recognition
            with sr.Microphone() as source:
                print("Say something:")
                audio_text = recognizer.listen(source)

                try:
                    print("Text: " + recognizer.recognize_google(audio_text))
                except sr.UnknownValueError:
                    print("Google Speech Recognition could not understand audio")
                except sr.RequestError as e:
                    print("Could not request results from Google Speech Recognition service; {0}".format(e))

            # Touch input
            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    cap.release()
                    cv2.destroyAllWindows()
                    exit()

                # Handle touch events here

        else:
            print("New face detected. Adding to the database.")
            # Add the face to the database
            person_name = input("Enter the person's name: ")
            add_to_database(person_name, encoding)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()

