import numpy as np
import cv2

class Panorama:
    def image_stitch(self, images, lowe_ratio=0.75, max_threshold=4.0, match_status=False):
        imgB, imgA = images
        kpA, fA = self._detect_features_and_keypoints(imgA)
        kpB, fB = self._detect_features_and_keypoints(imgB)
        
        matches, H, status = self._match_keypoints(kpA, kpB, fA, fB, lowe_ratio, max_threshold)
        if matches is None:
            return None
        
        result = self._warp_images(imgA, imgB, H)
        
        if match_status:
            vis = self._draw_matches(imgA, imgB, kpA, kpB, matches, status)
            return result, vis
        
        return result

    def _detect_features_and_keypoints(self, img):
        sift = cv2.SIFT_create()
        keypoints, features = sift.detectAndCompute(img, None)
        return np.float32([kp.pt for kp in keypoints]), features

    def _match_keypoints(self, kpA, kpB, fA, fB, lowe_ratio, max_threshold):
        all_matches = self._get_all_possible_matches(fA, fB)
        valid_matches = self._get_all_valid_matches(all_matches, lowe_ratio)
        
        if len(valid_matches) <= 4:
            return None
        
        ptsA, ptsB = self._extract_matched_points(kpA, kpB, valid_matches)
        H, status = self._compute_homography(ptsA, ptsB, max_threshold)
        
        return valid_matches, H, status

    def _get_all_possible_matches(self, fA, fB):
        return cv2.DescriptorMatcher_create("BruteForce").knnMatch(fA, fB, 2)

    def _get_all_valid_matches(self, all_matches, lowe_ratio):
        return [(m[0].trainIdx, m[0].queryIdx) for m in all_matches if len(m) == 2 and m[0].distance < m[1].distance * lowe_ratio]

    def _extract_matched_points(self, kpA, kpB, matches):
        ptsA = np.float32([kpA[i] for _, i in matches])
        ptsB = np.float32([kpB[i] for i, _ in matches])
        return ptsA, ptsB

    def _compute_homography(self, ptsA, ptsB, max_threshold):
        return cv2.findHomography(ptsA, ptsB, cv2.RANSAC, max_threshold)

    def _warp_images(self, imgA, imgB, H):
        result = cv2.warpPerspective(imgA, H, (imgA.shape[1] + imgB.shape[1], imgA.shape[0]))
        result[0:imgB.shape[0], 0:imgB.shape[1]] = imgB
        return result

    def _draw_matches(self, imgA, imgB, kpA, kpB, matches, status):
        hA, wA = imgA.shape[:2]
        vis = np.zeros((max(hA, imgB.shape[0]), wA + imgB.shape[1], 3), dtype="uint8")
        vis[0:hA, 0:wA] = imgA
        vis[0:imgB.shape[0], wA:] = imgB
        
        for (trainIdx, queryIdx), s in zip(matches, status):
            if s == 1:
                ptA = (int(kpA[queryIdx][0]), int(kpA[queryIdx][1]))
                ptB = (int(kpB[trainIdx][0]) + wA, int(kpB[trainIdx][1]))
                cv2.line(vis, ptA, ptB, (0, 255, 0), 1)
        
        return vis

    def get_image_dimension(self, img):
        return img.shape[:2]

    def get_points(self, imgA, imgB):
        hA, wA = self.get_image_dimension(imgA)
        hB, wB = self.get_image_dimension(imgB)
        vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
        vis[0:hA, 0:wA] = imgA
        vis[0:hB, wA:] = imgB
        return vis

    def draw_matches(self, imgA, imgB, kpA, kpB, matches, status):
        hA, wA = self.get_image_dimension(imgA)
        vis = self.get_points(imgA, imgB)
        for (trainIdx, queryIdx), s in zip(matches, status):
            if s == 1:
                ptA = (int(kpA[queryIdx][0]), int(kpA[queryIdx][1]))
                ptB = (int(kpB[trainIdx][0]) + wA, int(kpB[trainIdx][1]))
                cv2.line(vis, ptA, ptB, (0, 255, 0), 1)
        return vis
