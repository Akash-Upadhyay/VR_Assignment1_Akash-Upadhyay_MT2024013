import cv2
import numpy as np
import matplotlib.pyplot as plt

class CoinDetector:
    def __init__(self, image_path):
        self.image_path = image_path
        self.original_img = cv2.imread(image_path)
        self.contours = []
        self.coin_count = 0

    def detect_coins(self):
        gray = cv2.cvtColor(self.original_img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (11, 11), 0)
        edges = cv2.Canny(blurred, 30, 150)
        dilated = cv2.dilate(edges, None, iterations=2)
        contours, _ = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        min_area = 1000
        self.contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
        self.coin_count = len(self.contours)

    def visualize_detection(self):
        outlined_img = self.original_img.copy()
        cv2.drawContours(outlined_img, self.contours, -1, (0, 255, 0), 3)
        return outlined_img

    def segment_coins(self):
        mask = np.zeros(self.original_img.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, self.contours, -1, 255, -1)
        mask_img = cv2.bitwise_and(self.original_img, self.original_img, mask=mask)
        
        segmented_coins = []
        for contour in self.contours:
            x, y, w, h = cv2.boundingRect(contour)
            coin_segment = mask_img[y:y+h, x:x+w]
            segmented_coins.append(coin_segment)
        
        return segmented_coins, mask_img

    def display_results(self):
        outlined_img = self.visualize_detection()
        segmented_coins, mask_img = self.segment_coins()
        
        plt.figure(figsize=(15, 10))
        plt.subplot(2, 2, 1)
        plt.imshow(cv2.cvtColor(self.original_img, cv2.COLOR_BGR2RGB))
        plt.title('Original Image')
        plt.axis('off')
        
        plt.subplot(2, 2, 2)
        plt.imshow(cv2.cvtColor(outlined_img, cv2.COLOR_BGR2RGB))
        plt.title(f'Detected Coins: {self.coin_count}')
        plt.axis('off')
        
        plt.subplot(2, 2, 3)
        plt.imshow(cv2.cvtColor(mask_img, cv2.COLOR_BGR2RGB))
        plt.title('Segmented Coins (Mask)')
        plt.axis('off')
        
        if segmented_coins:
            rows = int(np.ceil(len(segmented_coins) / 4))
            plt.figure(figsize=(15, rows * 3))
            for i, coin in enumerate(segmented_coins):
                plt.subplot(rows, 4, i + 1)
                plt.imshow(cv2.cvtColor(coin, cv2.COLOR_BGR2RGB))
                plt.title(f'Coin {i+1}')
                plt.axis('off')
            plt.tight_layout()
        
        plt.tight_layout()
        plt.show()
        
        print(f"Total coins detected: {self.coin_count}")

def main(image_path):
    detector = CoinDetector(image_path)
    detector.detect_coins()
    detector.display_results()
    return detector.coin_count

if __name__ == "__main__":
    image_path = "./coins_images/coins.jpg"
    main(image_path)
    