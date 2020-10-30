import cv2 as cv
import numpy as np
from PIL import Image
from face_mask_detection import FaceMaskModel
from detect import object_detection
from cv2 import CascadeClassifier

from variables import*

if __name__ == "__main__":
        model = FaceMaskModel()
        model.run()
        video = cv.VideoCapture(0)

        while True:
                _, frame = video.read()
                frame = cv.flip(frame, 1)
                im = Image.fromarray(frame, 'RGB')
                img_array = np.array(im)
                img_array = cv.resize(img_array, target_size)
                img = object_detection(CascadeClassifier, img_array)

                img_array = img_array.astype("float32")  * rescale
                assert (img_array.shape == input_shape), "corrupted image"

                img_array = np.array([img_array])
                pred = model.predition(img_array)

                img = cv.resize(img, (800, 500))
                text = "Status : {}".format(pred)
                cv.putText(img, text, (35, 50), cv.FONT_HERSHEY_SIMPLEX,
                                1.25, (0, 255, 0), 5)

                cv.imshow("Capturing", img)
                key=cv.waitKey(1)
                if key == ord('q'):
                        break
        video.release()
        cv.destroyAllWindows()