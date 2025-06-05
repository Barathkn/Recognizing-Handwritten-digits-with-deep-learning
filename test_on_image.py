

import numpy as np
import cv2
import imutils
import joblib

# Load image
img = cv2.imread('test image2.jpg')
img = imutils.resize(img, width=300)
cv2.imshow("Original Image", img)
cv2.waitKey(0)

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("Grayscale", gray)
cv2.waitKey(0)

# Preprocessing
kernel = np.ones((40, 40), np.uint8)
blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
ret, thresh = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

# Find contours
cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Sort contours from left to right
cnts = sorted(cnts, key=lambda c: cv2.boundingRect(c)[0])

# Load trained model
model = joblib.load('model.pkl')

predicted_digits = ""

for c in cnts:
    (x, y, w, h) = cv2.boundingRect(c)

    # Skip small noise
    if w < 10 or h < 10:
        continue

    try:
        # Create mask and extract digit
        mask = np.zeros(gray.shape, dtype="uint8")
        hull = cv2.convexHull(c)
        cv2.drawContours(mask, [hull], -1, 255, -1)
        mask = cv2.bitwise_and(thresh, thresh, mask=mask)

        roi = mask[y-7:y+h+7, x-7:x+w+7]
        roi = cv2.resize(roi, (28, 28))
        roi = roi.reshape(1, 784).astype("float32")

        prediction = model.predict(roi)[0]
        predicted_digits += str(int(prediction))

        # Draw results on image
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
        cv2.putText(img, str(int(prediction)), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    except Exception as e:
        print("Error processing contour:", e)

# Resize for display
img = imutils.resize(img, width=500)
cv2.imshow("Result", img)
cv2.imwrite("result2.jpg", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Print final result
print("Predicted Digits:", predicted_digits)
