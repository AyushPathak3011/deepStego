from math import log10, sqrt 
import cv2 
import numpy as np
from skimage.metrics import structural_similarity as ssim
  
def PSNR(original, compressed): 
    original = np.array(original)
    compressed = np.array(compressed)
    
    # Check if the shapes match
   
    
    mse = np.mean((original - compressed) ** 2) 
    if(mse == 0):  
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
   

    return psnr, mse

def calculate_ssim(img1, img2):
    img1 = np.array(img1)
    img2 = np.array(img2)

    # Ensure the images are the same size
    if img1.shape != img2.shape:
        raise ValueError("Input Images must have the same dimensions")

    return ssim(img1, img2, multichannel=True)

def difference(c1, c2, s1, s2):
    c1 = np.array(c1)
    c2 = np.array(c2)
    s1 = np.array(s1)
    s2 = np.array(s2)
    diff_S, diff_C = np.abs(s2 - s1), np.abs(c2 - c1)
    return diff_S, diff_C

def inputPSNR(cover, encoded, og_secret, secret): 
    #  original = cv2.imread("/Users/ayushpathak/deep-steganography/app/cover.png") 
    #  compressed = cv2.imread("/Users/ayushpathak/deep-steganography/app/secret.png", 1) 
    cover = cover
    encoded = encoded
    psnr, mse = PSNR(cover, encoded) 
    print(f"PSNR value is {psnr} dB and MSE value is {mse}")
    diff_S, diff_C = difference(cover, encoded, og_secret, secret)
    return psnr, mse, diff_S, diff_C

