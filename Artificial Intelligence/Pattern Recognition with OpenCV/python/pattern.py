
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the image
img = cv2.imread('images/pattern.jpeg')
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to remove salt and pepper noises
filtered_image = cv2.GaussianBlur(gray_img, (5, 5), 0)

# Display noisy and filtered images
plt.subplot(1, 2, 1), plt.imshow(gray_img, cmap='gray')
plt.title('Noisy Image'), plt.xticks([]), plt.yticks([])

plt.subplot(1, 2, 2), plt.imshow(filtered_image, cmap='gray')
plt.title('Filtered Image'), plt.xticks([]), plt.yticks([])

# Save filtered image 
# cv2.imwrite('filtered_image.jpg', filtered_image)

plt.show()

# Thresholding

# Global thresholding 
ret1, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# Otsu's thresholding 
ret2, th2 = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Otsu's thresholding after Gaussian filtering 

ret3, th3 = cv2.threshold(filtered_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# plot all the images and their histograms
images = [img, 0, th1, gray_img, 0, th2, filtered_image, 0, th3]
titles = ['Original Noisy Image', 'Histogram', 'Global Thresholding (v=127)',
          'Original Noisy Image', 'Histogram', "Otsu's Thresholding",
          'Gaussian filtered Image', 'Histogram', "Otsu's Thresholding"]


for i in range(3):
    plt.subplot(3, 3, i*3+1), plt.imshow(images[i*3], 'gray')
    plt.title(titles[i*3]), plt.xticks([]), plt.yticks([])
    plt.subplot(3, 3, i*3+2), plt.hist(images[i*3].ravel(), 256)
    plt.title(titles[i*3+1]), plt.xticks([]), plt.yticks([])
    plt.subplot(3, 3, i*3+3), plt.imshow(images[i*3+2], 'gray')
    plt.title(titles[i*3+2]), plt.xticks([]), plt.yticks([])

plt.show()

# Connectivity Analysis
def connected_component_label(path):     
    # Getting the input image
    img = cv2.imread('pattern.jpg', 0)     
    # Converting those pixels with values 1-127 to 0 and others to 1     
    img = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)[1]    
    # Applying cv2.connectedComponents()      
    num_labels, labels = cv2.connectedComponents(img)    
    # Map component labels to hue val, 0-179 is the hue range in OpenCV
    label_hue = np.uint8(179*labels/np.max(labels))     
    blank_ch = 255*np.ones_like(label_hue)     
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])     
    # Converting cvt to BGR     
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)     
    # set bg label to black     
    labeled_img[label_hue==0] = 0     
    #Showing Image after Component Labeling     
    plt.imshow(cv2.cvtColor(labeled_img, cv2.COLOR_BGR2RGB))     
    plt.axis('off')    
    plt.title("Image after Component Labeling")     
    plt.show() 

connected_component_label(filtered_image)

# Pattern Recognition (Recognizing the type of Objects in the image)
font = cv2.FONT_HERSHEY_COMPLEX
_, threshold = cv2.threshold(filtered_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

for cnt in contours:
    approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
    cv2.drawContours(img, [approx], 0, (0), 5)
    x = approx.ravel()[0]
    y = approx.ravel()[1]
    if len(approx) == 4:
        cv2.putText(img, "Square", (x, y), font, 1, (255))
    else:
        cv2.putText(img, "Circle", (x, y), font, 1, (255))

cv2.drawContours(img, contours, -1, (255, 0, 0), 2)
cv2.imshow("Pattern recognition", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
