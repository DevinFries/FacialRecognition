# FacialRecognition
Python program that performs facial recognition using the face_recognition library along with OpenCV. This script loads known faces from a folder, encodes them, and then uses your webcam to identify faces in real time.


How It Works

Loading Known Faces:

The load_known_faces function reads all image files from the specified folder.

For each image, it calculates a face encoding using the face_recognition library.

The filename (minus the extension) is used as the individual's name.

Real-Time Recognition:

The recognize_faces function captures video from the webcam.

It processes every other frame for efficiency, detects faces, and computes face encodings.

It compares the detected face encodings to the known ones.

If a match is found (using a set tolerance), the corresponding name is used; otherwise, the face is labeled as "Unknown."

Rectangles and labels are drawn on the video feed to display the results.

Exiting the Program:

Press 'q' on the video window to exit the application.
