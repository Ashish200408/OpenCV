"""
show_image.py — A beginner-friendly OpenCV example
=====================================================
Demonstrates how to:
  1. Read an image from disk  (cv2.imread)
  2. Display it in a window   (cv2.imshow)
  3. Wait for a key press     (cv2.waitKey)
  4. Close the window cleanly (cv2.destroyAllWindows)
"""

# Step 1: Import the OpenCV library
import cv2

# Step 2: Read the image from the current directory
# cv2.imread() returns a NumPy array, or None if the file is not found
image = cv2.imread("sample.jpg")

# Step 3: Check whether the image was loaded successfully
if image is None:
    # The file was not found or could not be decoded — print an error and stop
    print("Error: Could not load 'sample.jpg'.")
    print("Make sure the file exists in the same folder as this script.")
else:
    # Step 4: Display the image in a window titled "My Image"
    cv2.imshow("My Image", image)

    # Step 5: Wait indefinitely until the user presses any key
    # (passing 0 means "wait forever")
    cv2.waitKey(0)

    # Step 6: Close all OpenCV windows once a key has been pressed
    cv2.destroyAllWindows()
