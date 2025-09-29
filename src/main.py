import cv2
import cv2.data
import matplotlib.pyplot as plt

img = cv2.imread("21699.jpg")

if img is None:
    print("Image not loaded. Check the path or file.")
else:
    print("Image loaded successfully.")


# Convert from BGR (OpenCV default) to RGB (for matplotlib display)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Load the cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Convert to grayscale (detection works better)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

# Draw rectangles
for (x, y, w, h) in faces:
    cv2.rectangle(img_rgb, (x, y), (x+w, y+h), (255, 0, 0), 2)

# Convert to HSV
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Define skin color range
lower = (0, 48, 80)
upper = (20, 255, 255)

mask = cv2.inRange(hsv, lower, upper)

# Bitwise AND to extract skin
skin = cv2.bitwise_and(img, img, mask=mask)

plt.imshow(cv2.cvtColor(skin, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()
