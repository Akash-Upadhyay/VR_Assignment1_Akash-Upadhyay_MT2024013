import cv2
import glob
import imutils
from panorama import Panorama

class ImageStitcher:
    def __init__(self, image_pattern):
        self.image_pattern = image_pattern
        self.images = self._load_images()
        self.panorama = Panorama()

    def _load_images(self):
        image_files = sorted(glob.glob(self.image_pattern))
        images = [cv2.imread(img) for img in image_files]
        if len(images) < 2:
            raise ValueError("At least 2 images are required for stitching")
        return [imutils.resize(img, width=600, height=600) for img in images]

    def stitch_images(self):
        # Start with the last two images
        result, matched_points = self.panorama.image_stitch([self.images[-2], self.images[-1]], match_status=True)
        
        if result is None:
            raise RuntimeError("Initial stitching failed. Check the input images.")

        # Iterate over the remaining images
        for i in range(len(self.images) - 2):
            result, matched_points = self.panorama.image_stitch([self.images[len(self.images) - i - 3], result], match_status=True)
            
            if result is None:
                raise RuntimeError(f"Stitching failed at image index {len(self.images) - i - 3}. Check the input images.")
        
        return result, matched_points

    def display_and_save_results(self, result, matched_points):
        cv2.imshow("Keypoint Matches", matched_points)
        cv2.imshow("Panorama", result)
        cv2.imwrite("output/matched_points.jpg", matched_points)
        cv2.imwrite("output/panorama_image.jpg", result)
        
        # Wait for a key press and close windows
        print("Press any key in the image window to close and exit.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def main():
    image_pattern = input("Enter the pattern for image files (e.g., 'images/*.jpg'): ")
    stitcher = ImageStitcher(image_pattern)
    try:
        result, matched_points = stitcher.stitch_images()
        stitcher.display_and_save_results(result, matched_points)
    except RuntimeError as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()

