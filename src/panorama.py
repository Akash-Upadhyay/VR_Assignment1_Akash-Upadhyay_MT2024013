import numpy as np
import cv2

class Panorama:
    def image_stitch(self, images, lowe_ratio=0.75, max_Threshold=4.0, match_status=False):
        imgB, imgA = images
        kpA, fA = self.detect_feature_and_keypoints(imgA)
        kpB, fB = self.detect_feature_and_keypoints(imgB)
        vals = self.match_keypoints(kpA, kpB, fA, fB, lowe_ratio, max_Threshold)
        if vals is None: return None
        matches, H, status = vals
        result = self.get_warp_perspective(imgA, imgB, H)
        result[0:imgB.shape[0], 0:imgB.shape[1]] = imgB
        if match_status:
            vis = self.draw_matches(imgA, imgB, kpA, kpB, matches, status)
            return result, vis
        return result

    def get_warp_perspective(self, imgA, imgB, H):
        return cv2.warpPerspective(imgA, H, (imgA.shape[1] + imgB.shape[1], imgA.shape[0]))

    def detect_feature_and_keypoints(self, img):
        kp, f = cv2.SIFT_create().detectAndCompute(img, None)
        return np.float32([i.pt for i in kp]), f

    def get_all_possible_matches(self, fA, fB):
        return cv2.DescriptorMatcher_create("BruteForce").knnMatch(fA, fB, 2)

    def get_all_valid_matches(self, all_matches, lowe_ratio):
        return [(v[0].trainIdx, v[0].queryIdx) for v in all_matches if len(v) == 2 and v[0].distance < v[1].distance * lowe_ratio]

    def compute_homography(self, ptsA, ptsB, max_Threshold):
        return cv2.findHomography(ptsA, ptsB, cv2.RANSAC, max_Threshold)

    def match_keypoints(self, kpA, kpB, fA, fB, lowe_ratio, max_Threshold):
        matches = self.get_all_valid_matches(self.get_all_possible_matches(fA, fB), lowe_ratio)
        if len(matches) <= 4: return None
        ptsA = np.float32([kpA[i] for _, i in matches])
        ptsB = np.float32([kpB[i] for i, _ in matches])
        return matches, self.compute_homography(ptsA, ptsB, max_Threshold)[0], self.compute_homography(ptsA, ptsB, max_Threshold)[1]

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