import cv2 as cv
import numpy as np
from PIL import Image
from face_mask_detection import FaceMaskModel

from variables import*

if __name__ == "__main__":
        model = FaceMaskModel()
        model.run()
        video = cv.VideoCapture(0)

        while True:
                _, frame = video.read()

                im = Image.fromarray(frame, 'RGB')
                img_array = np.array(im)
                img_array = cv.resize(img_array, target_size).astype("float32") 
                img_array = img_array * rescale
                assert (img_array.shape == input_shape), "corrupted image"

                img_array = np.array([img_array])
                pred = model.predition(img_array)

                text = "Status : {}".format(pred)
                cv.putText(frame, text, (35, 50), cv.FONT_HERSHEY_SIMPLEX,
                                1.25, (0, 255, 0), 5)

                cv.imshow("Capturing", frame)
                key=cv.waitKey(1)
                if key == ord('q'):
                        break
        video.release()
        cv.destroyAllWindows()