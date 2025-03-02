import numpy as np
import cv2

class Panorama:
    def image_stitch(self, images, lowe_ratio=0.75, max_threshold=4.0, match_status=False):
        """
        Stitches two images together using feature matching and homography.
        
        Parameters:
        - images: List containing two images to be stitched.
        - lowe_ratio: Lowe's ratio test threshold for filtering matches.
        - max_threshold: RANSAC threshold for computing homography.
        - match_status: If True, returns visualization of matches.
        
        Returns:
        - Stitched panorama image.
        - If match_status is True, also returns keypoint match visualization.
        """
        imgB, imgA = images
        kpA, fA = self.detect_feature_and_keypoints(imgA)
        kpB, fB = self.detect_feature_and_keypoints(imgB)
        
        # Match keypoints and compute homography
        vals = self.match_keypoints(kpA, kpB, fA, fB, lowe_ratio, max_threshold)
        if vals is None:
            return None
        matches, H, status = vals
        
        # Warp perspective to align images
        result = self.get_warp_perspective(imgA, imgB, H)
        result[0:imgB.shape[0], 0:imgB.shape[1]] = imgB
        
        if match_status:
            vis = self.draw_matches(imgA, imgB, kpA, kpB, matches, status)
            return result, vis
        
        return result

    def detect_feature_and_keypoints(self, img):
        """
        Detects keypoints and computes descriptors using SIFT.
        """
        sift = cv2.SIFT_create()
        keypoints, features = sift.detectAndCompute(img, None)
        return np.float32([kp.pt for kp in keypoints]), features

    def match_keypoints(self, kpA, kpB, fA, fB, lowe_ratio, max_threshold):
        """
        Matches keypoints between two images and computes homography.
        """
        all_matches = self.get_all_possible_matches(fA, fB)
        valid_matches = self.get_all_valid_matches(all_matches, lowe_ratio)
        
        if len(valid_matches) <= 4:
            return None  # Not enough matches to compute homography
        
        ptsA, ptsB = self.extract_matched_points(kpA, kpB, valid_matches)
        H, status = self.compute_homography(ptsA, ptsB, max_threshold)
        
        return valid_matches, H, status

    def get_all_possible_matches(self, fA, fB):
        """
        Finds all possible matches between two sets of feature descriptors.
        """
        return cv2.DescriptorMatcher_create("BruteForce").knnMatch(fA, fB, 2)

    def get_all_valid_matches(self, all_matches, lowe_ratio):
        """
        Filters matches using Lowe's ratio test.
        """
        return [(m[0].trainIdx, m[0].queryIdx) for m in all_matches if len(m) == 2 and m[0].distance < m[1].distance * lowe_ratio]
    
    def extract_matched_points(self, kpA, kpB, matches):
        """
        Extracts matched keypoints from both images.
        """
        ptsA = np.float32([kpA[i] for _, i in matches])
        ptsB = np.float32([kpB[i] for i, _ in matches])
        return ptsA, ptsB
    
    def compute_homography(self, ptsA, ptsB, max_threshold):
        """
        Computes the homography matrix using RANSAC.
        """
        return cv2.findHomography(ptsA, ptsB, cv2.RANSAC, max_threshold)
    
    def get_warp_perspective(self, imgA, imgB, H):
        """
        Warps imgA to align with imgB using homography transformation.
        """
        return cv2.warpPerspective(imgA, H, (imgA.shape[1] + imgB.shape[1], imgA.shape[0]))
    
    def draw_matches(self, imgA, imgB, kpA, kpB, matches, status):
        """
        Draws matches between keypoints in two images.
        """
        hA, wA = imgA.shape[:2]
        vis = self.get_combined_canvas(imgA, imgB)
        
        for (trainIdx, queryIdx), s in zip(matches, status):
            if s == 1:
                ptA = (int(kpA[queryIdx][0]), int(kpA[queryIdx][1]))
                ptB = (int(kpB[trainIdx][0]) + wA, int(kpB[trainIdx][1]))
                cv2.line(vis, ptA, ptB, (0, 255, 0), 1)
        
        return vis
    
    def get_combined_canvas(self, imgA, imgB):
        """
        Creates a blank canvas large enough to display both images side by side.
        """
        hA, wA = imgA.shape[:2]
        hB, wB = imgB.shape[:2]
        vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
        vis[0:hA, 0:wA] = imgA
        vis[0:hB, wA:] = imgB
        return vis
