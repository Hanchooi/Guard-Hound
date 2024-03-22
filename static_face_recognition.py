import cv2
import matplotlib.pyplot as plt

imagePath = 'input_image.jpg'
img = cv2.imread(imagePath)

face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Check if the image was successfully loaded
if img is not None:
    # Perform further operations here, such as displaying or processing the image
    cv2.imshow("Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(img.shape)
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if gray_image is not None:
        cv2.imshow("Image", gray_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print(gray_image.shape)

        face = face_classifier.detectMultiScale(
            gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
        )
        for (x, y, w, h) in face:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 4)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imshow("Image", img_rgb)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        plt.figure(figsize=(20, 10))
        plt.imshow(img_rgb)
        plt.axis('off')
        plt.show()


    else:
        print("Failed to load gray_image.")
else:
    print("Failed to load image.")
