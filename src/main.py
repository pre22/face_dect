import cv2
import cv2.data
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread("img/person.jpg")

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

# ---- Calculate Average Skin Color ----
skin_pixels = img[mask > 0]  # take only pixels where mask is white
if len(skin_pixels) > 0:
    avg_color_bgr = np.mean(skin_pixels, axis=0).astype(int)  # BGR average
    avg_color_rgb = avg_color_bgr[::-1]  # convert BGR → RGB
    avg_color_hex = "#{:02x}{:02x}{:02x}".format(*avg_color_rgb)

    print("Average Skin Color (BGR):", avg_color_bgr)
    print("Average Skin Color (RGB):", avg_color_rgb)
    print("Average Skin Color (HEX):", avg_color_hex)

    # ---- Display color as text on the image ----
    cv2.putText(img_rgb, f"Skin Color: {avg_color_rgb}", (30, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

# plt.imshow(cv2.cvtColor(skin, cv2.COLOR_BGR2RGB))
# plt.axis("off")
# plt.show()

# Show face + text overlay
plt.imshow(img_rgb)
plt.axis("off")
plt.show()
