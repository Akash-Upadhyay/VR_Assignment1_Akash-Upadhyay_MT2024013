from panorama import Panorama
import imutils
import cv2
import glob

# Prompt the user for image file pattern and load all matching images
image_pattern = input("Enter the pattern for image files (e.g., 'images/*.jpg'): ")
images = [cv2.imread(img) for img in sorted(glob.glob(image_pattern))]

# Check if at least two images are provided
if len(images) < 2:
    raise ValueError("At least 2 images are required for stitching")

# Resize all images to a fixed width while maintaining aspect ratio
images = [imutils.resize(img, width=600) for img in images]

# Initialize the Panorama class
panorama = Panorama()

# Stitch the last two images first to start the process
result, matched_points = panorama.image_stitch([images[-2], images[-1]], match_status=True)

# Iterate over the remaining images from right to left and stitch them sequentially
for i in range(len(images) - 2):
    result, matched_points = panorama.image_stitch([images[len(images) - i - 3], result], match_status=True)

# Display the keypoint matches between images
cv2.imshow("Keypoint Matches", matched_points)
cv2.imshow("Panorama", result)

# Save the stitched panorama and keypoint matches output
cv2.imwrite("output/matched_points.jpg", matched_points)
cv2.imwrite("output/panorama_image.jpg", result)

# Wait for user input before closing windows
cv2.waitKey(0)
cv2.destroyAllWindows()
