import cv2
import numpy as np
from matplotlib import pyplot as plt

class Histogram():
    def __init__ (self):
        self.img = cv2.imread("cicek.png")
        self.color = ("b", "g", "r")

    def renkli_hist(self):    
        for i,col in enumerate(self.color):
            hist = cv2.calcHist(self.img, [i], None, [255], [0,255])
            plt.subplot(3,3,1)
            plt.plot(hist, color = col)
            plt.title("Orjinal Renkli Seviye Histogram ")
        self.renkli_equ()

    def renkli_equ(self):
        b, g, r = cv2.split(self.img)
        equal_b = cv2.equalizeHist(b)
        equal_g = cv2.equalizeHist(g)
        equal_r = cv2.equalizeHist(r)
        self.merged = cv2.merge([equal_b, equal_g, equal_r])
        chanels = [equal_b, equal_g, equal_r]
        colors = ["blue", "green", "red"]
        for i in range(3):
            hist_equal_bgr = cv2.calcHist(chanels[i], [0], None, [255], [0,255])
            plt.subplot(3,3,i+2)
            plt.plot(hist_equal_bgr, color = colors[i])
            plt.title(f"Renkli Seviye Histogram Eşitleme \n{colors[i]}")
        self.gri_hist()

    def gri_hist(self):
        self.img_gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        hist_gray = cv2.calcHist(self.img_gray, [0], None, [255], [0,255])
        plt.subplot(3,3,5)
        plt.plot(hist_gray)
        plt.title("Orjinal Gri Seviye Histogram")
        self.gri_equ()
    
    def gri_equ(self):
        self.equal = cv2.equalizeHist(self.img_gray)
        hist_equal_gray = cv2.calcHist(self.equal, [0], None, [255], [0,255])
        plt.subplot(3,3,6)
        plt.plot(hist_equal_gray)
        plt.title("Gri Seviye Histogram Eşitleme")
        self.adaptive_equ()
    
    def adaptive_equ(self):
        clahe = cv2.createCLAHE(clipLimit=5, tileGridSize=(8,8))
        self.clahe_img = clahe.apply(self.img_gray)
        hist_adaptive_gray = cv2.calcHist(self.clahe_img, [0], None, [255], [0,255])
        plt.subplot(3,3,7)
        plt.plot(hist_adaptive_gray)
        plt.title("Gri Seviye Adaptive Eşitleme")
        self.yazdır()
    
    def yazdır(self):
        cv2.imshow("orjinal gri seviye", self.img_gray)
        cv2.imshow("orjinal renkli seviye", self.img)
        cv2.imshow("Gri Seviye equalize", self.equal)
        cv2.imshow("Gri seviye adaptive", self.clahe_img)
        cv2.imshow("renkli seviye equalize", self.merged)
        plt.tight_layout(pad=2.0) 
        plt.show()

        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    baslat = Histogram()
    baslat.renkli_hist()
        
