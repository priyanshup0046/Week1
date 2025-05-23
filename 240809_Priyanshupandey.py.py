import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
# converting the image in a grey scale due to the fact that the sobel operator is used for edge detection and it is better to use it on a grey scale image (vs code ne automatic type kia lol)
img = Image.open("mario.jpeg").convert("L")
img_np = np.array(img)
# I am using sobe cornel in this case and defining it here as asked 
sobel_x = np.array([[ -1, 0, 1],[ -2, 0, 2],[ -1, 0, 1]])
sobel_y = np.array([[ -1, -2, -1],[  0,  0,  0],[  1,  2,  1]])

# Padding the image
def convolve(img, kernel):
    h, w = img.shape
    kh, kw = kernel.shape
    pad_h = kh // 2
    pad_w = kw // 2
    padded = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
    output = np.zeros_like(img, dtype=np.float32)

    for i in range(h):
        for j in range(w):
            region = padded[i:i+kh, j:j+kw]
            output[i, j] = np.sum(region * kernel)
    
    return output

grad_x = convolve(img_np, sobel_x)
grad_y = convolve(img_np, sobel_y)
magnitude = np.sqrt(grad_x**2 + grad_y**2)

# Load original in color for display
original_img = Image.open("mario.jpeg")
# Normalize to 0-255
magnitude = (magnitude / magnitude.max()) * 255
magnitude = magnitude.astype(np.uint8)

# Plotting
plt.subplot(1, 2, 2)
plt.title("Edge Detection (Sobel)")
plt.imshow(magnitude, cmap='gray')
plt.axis('off')
plt.subplot(1, 2, 1)
plt.title("Original")
plt.imshow(original_img)
plt.axis('off')
plt.tight_layout()
plt.show()
