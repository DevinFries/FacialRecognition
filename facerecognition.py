import os
import cv2
import numpy as np
import face_recognition

def load_known_faces(known_faces_folder):
    """
    Loads and encodes images of known faces from the specified folder.
    The image filenames (without extension) are used as the person's name.
    
    Args:
        known_faces_folder (str): Path to the folder containing known face images.
    
    Returns:
        tuple: (list of face encodings, list of names)
    """
    known_encodings = []
    known_names = []
    
    for file in os.listdir(known_faces_folder):
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            path = os.path.join(known_faces_folder, file)
            image = face_recognition.load_image_file(path)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                known_encodings.append(encodings[0])
                name, _ = os.path.splitext(file)
                known_names.append(name)
            else:
                print(f"[Warning] No face found in {file}")
    
    return known_encodings, known_names

def recognize_faces(known_encodings, known_names):
    """
    Uses the webcam to capture video, detects faces, and performs recognition.
    Identified faces are drawn with rectangles and names on the video feed.
    
    Args:
        known_encodings (list): List of face encodings for known individuals.
        known_names (list): List of names corresponding to the known encodings.
    """
    video_capture = cv2.VideoCapture(0)
    process_frame = True

    print("Starting video stream. Press 'q' to exit.")
    
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        # Resize frame to speed up processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        # Convert from BGR (OpenCV format) to RGB (face_recognition format)
        rgb_small_frame = small_frame[:, :, ::-1]

        if process_frame:
            # Detect faces and encode them from the current frame
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            face_names = []

            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.5)
                name = "Unknown"
                
                # Compute distances to get the best match
                distances = face_recognition.face_distance(known_encodings, face_encoding)
                if len(distances) > 0:
                    best_match_index = np.argmin(distances)
                    if matches[best_match_index]:
                        name = known_names[best_match_index]
                face_names.append(name)

        process_frame = not process_frame

        # Display the results by drawing rectangles and names on the original frame
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale face locations back up since the frame was resized
            top, right, bottom, left = top * 4, right * 4, bottom * 4, left * 4
            
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.rectangle(frame, (left, bottom - 30), (right, bottom), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, name, (left + 5, bottom - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)

        cv2.imshow('Facial Recognition', frame)
        
        # Press 'q' to exit the video stream
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Folder where images of known faces are stored.
    known_faces_folder = "known_faces"
    known_encodings, known_names = load_known_faces(known_faces_folder)
    
    if known_encodings:
        recognize_faces(known_encodings, known_names)
    else:
        print("No known faces found. Please add images to the 'known_faces' folder.")
