import cv2
import torch
import kornia as K

# Set up webcam capture
cap = cv2.VideoCapture(0)  # 0 for the default webcam

# Set the resolution to Full HD (1920x1080)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Read the first frame from the webcam
ret, frame1 = cap.read()
frame1_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

# Convert the first frame to a torch tensor and move it to the GPU
frame1_tensor = torch.from_numpy(frame1_gray).float().unsqueeze(0).unsqueeze(0) / 255.0
frame1_tensor = frame1_tensor.cuda()

# Define Gaussian blur kernel using Kornia
blur_kernel = K.filters.get_gaussian_kernel2d((5, 5), (1.5, 1.5)).cuda()

while True:
    # Capture the next frame from the webcam
    ret, frame2 = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale
    frame2_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Convert the second frame to a torch tensor and move it to the GPU
    frame2_tensor = torch.from_numpy(frame2_gray).float().unsqueeze(0).unsqueeze(0) / 255.0
    frame2_tensor = frame2_tensor.cuda()

    # Apply Gaussian blur to both frames using Kornia
    blurred_frame1 = K.filters.filter2d(frame1_tensor, blur_kernel)
    blurred_frame2 = K.filters.filter2d(frame2_tensor, blur_kernel)

    # Compute the absolute difference between the blurred frames
    diff_tensor = torch.abs(blurred_frame1 - blurred_frame2)

    # Apply a threshold to highlight motion areas
    thresholded = (diff_tensor > 0.1).float() * 1.0  # Adjust threshold as needed

    # Convert the tensor back to a NumPy array for OpenCV display
    motion_map = (thresholded.squeeze().cpu().numpy() * 255).astype('uint8')

    # Convert motion map to a 3-channel (RGB) image for overlay
    motion_map_colored = cv2.cvtColor(motion_map, cv2.COLOR_GRAY2BGR)

    # Blend the original frame and motion map with transparency (alpha)
    alpha = 0.6  # Transparency factor for overlay
    overlay = cv2.addWeighted(frame2, alpha, motion_map_colored, 1 - alpha, 0)

    # Display the overlaid motion visualization
    cv2.imshow('Motion Visualization', overlay)

    # Update the first frame for the next iteration
    frame1_tensor = frame2_tensor

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
