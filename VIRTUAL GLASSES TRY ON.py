#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2
import os
import numpy as np

# Load the webcam
cap = cv2.VideoCapture(0)

# Load the face cascade classifier
cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize the overlay folder and counter
overlay_folder = r"C:\Users\hp\Downloads\GLASSES"
num = 1

# Function to update the overlay path
def update_overlay_path():
    global overlay_path
    overlay_path = os.path.join(overlay_folder, f'glasses_{num:02d}.png')

# Initialize the overlay path
update_overlay_path()

while True:
    k = cv2.waitKey(100)
    if k == ord('s'):
        num = (num % 29) + 1  # Cycle through glasses from 1 to 29
        update_overlay_path()

    # Read the overlay image with alpha channel
    overlay = cv2.imread(overlay_path, cv2.IMREAD_UNCHANGED)

    # Read a frame from the webcam
    _, frame = cap.read()

    # Convert the frame to grayscale for face detection
    gray_scale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = cascade.detectMultiScale(gray_scale)

    # Overlay the glasses on each detected face
    for (x, y, w, h) in faces:
        # Resize the overlay to fit the detected face
        overlay_resized = cv2.resize(overlay, (w, int(h * 0.8)))

        # Extract the alpha channel from the overlay image
        alpha = overlay_resized[:, :, 3] / 255.0

        # Create a mask for the glasses
        mask = np.zeros_like(frame)
        mask[y:y + overlay_resized.shape[0], x:x + overlay_resized.shape[1]] = 255

        # Apply the mask to the overlay and frame
        overlay_rgb = overlay_resized[:, :, :3]
        masked_overlay = cv2.bitwise_and(overlay_rgb, mask)
        masked_frame = cv2.bitwise_and(frame, cv2.bitwise_not(mask))

        # Combine the masked overlay and frame
        frame[y:y + overlay_resized.shape[0], x:x + overlay_resized.shape[1]] = masked_overlay + masked_frame

    # Display the frame with glasses overlay
    cv2.imshow('SnapLens', frame)

    # Exit the loop if 'q' key is pressed
    if cv2.waitKey(10) == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()


# In[ ]:




