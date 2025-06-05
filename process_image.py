import cv2
import numpy as np
import imutils
import joblib
import os

model = joblib.load("model.pkl")

def predict_digits(image_path):
    img = cv2.imread(image_path)
    img = imutils.resize(img, width=300)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    kernel = np.ones((40, 40), np.uint8)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, thresh = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=lambda c: cv2.boundingRect(c)[0])

    prediction = ""

    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        if w < 10 or h < 10:
            continue
        try:
            mask = np.zeros(gray.shape, dtype="uint8")
            hull = cv2.convexHull(c)
            cv2.drawContours(mask, [hull], -1, 255, -1)
            mask = cv2.bitwise_and(thresh, thresh, mask=mask)

            roi = mask[y-7:y+h+7, x-7:x+w+7]
            roi = cv2.resize(roi, (28, 28))
            roi = roi.reshape(1, 784).astype("float32")

            digit = model.predict(roi)[0]
            prediction += str(int(digit))

            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 1)
            cv2.putText(img, str(int(digit)), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        except Exception as e:
            print("Error:", e)

    result_path = os.path.join("static", "uploads", "result.jpg")
    img = imutils.resize(img, width=500)
    cv2.imwrite(result_path, img)
    return result_path, prediction
