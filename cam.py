import cv2 as cv
from tensorflow.keras.models import load_model
import numpy as np

class_names = ["bench press", "squat", "standing"]

model_name = "sportsPosesClassifier.keras"

model = load_model(model_name)
model.summary()

vid = cv.VideoCapture(0)

while (True):

    # Capture the video frame
    # by frame
    ret, frame = vid.read()

    formattedFrame = np.array([cv.resize(frame, (125,125))])/255

    result = model.predict(formattedFrame)
    index = np.argmax(result)
    print("prediction : ", result, index, class_names[index])
    # Display the resulting frame
    cv.imshow('frame', frame)

    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv.destroyAllWindows()
