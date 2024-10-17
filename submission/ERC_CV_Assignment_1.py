# %% [markdown]
# # Problem Statement
# 
# Write a Python program using MediaPipe to detect and draw hand contours on a live webcam feed. The program should:
# 
# Use a live feed from the webcam to continuously detect hands. Draw the detected hand landmarks and connections (contours) on the video frame in real-time. Flip the frame horizontally for a mirror effect. Stop the video feed when the user presses the 'q' key.

# %% [markdown]
# # Working behind Hand Detection:
# 
# 
# ## MediaPipe
# 
# 
# MediaPipe is an open-source Python library developed by Google that provides a comprehensive suite of tools for building applications that process and analyze multimedia data, such as images and videos. It offers a wide range of pre-built machine learning models and pipelines for tasks like facial recognition, hand tracking, pose estimation, object detection, and more. MediaPipe simplifies the development of computer vision, making it accessible to developers with various levels of expertise. It is often used in applications related to augmented reality, gesture recognition, and real-time tracking, among others.
# 
# ## Hand Landmarks in MediaPipe
# 
# In MediaPipe, hand landmarks refer to the precise points or landmarks detected on a human hand in an image or video. The library's Hand module provides a machine learning model that can estimate the 21 key landmarks on a hand, including the tips of each finger, the base of the palm, and various points on the fingers and hand. These landmarks can be used for various applications, such as hand tracking, gesture recognition, sign language interpretation, and virtual reality interactions. MediaPipe's hand landmarks model makes it easier for developers to create applications that can understand and respond to hand movements and gestures in real-time.
# 
# ## Reference material
# https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker https://pypi.org/project/mediapipe/

# %% [markdown]
# # Solution

# %%
import cv2
import mediapipe as mp

# Initialize MediaPipe Hands and Drawing utils so that we can use them later
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# %%
# Start capturing video from the webcam of the laptop as this will be fed directly
cap = cv2.VideoCapture(0)

# %%
# Initialize the MediaPipe Hands model with default settings
with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame and nothing to do.")
            continue

        # Flip the frame horizontally for a mirror effect as adviced
        frame = cv2.flip(frame, 1)

        # Convert the frame from BGR to RGB due to reading from open CV
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the RGB frame to detect hands as that is our objective in the assignment
        results = hands.process(frame_rgb)

        # Draw hand landmarks and connections if any hand is detected as requested in the problem assignment
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Display the processed video frame
        cv2.imshow('Hand Contours', frame)

        # Press 'q' to quit from the frame reading
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

# %%
# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()


