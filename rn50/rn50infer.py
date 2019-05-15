import numpy as np
import onnxruntime as rt
import random
from PIL import Image

import sys, os
sys.path.insert(1, os.path.join(sys.path[0], "trt"))
import common

INPUT_SHAPE = (1, 3, 224, 224)
#INPUT_SHAPE = (64, 3, 7, 7)

def load_normalized_test_case(test_image): #, pagelocked_buffer):
    # Converts the input image to a CHW Numpy array
    def normalize_image(image):
        # Resize, antialias and transpose the image to CHW.
        n, c, h, w = INPUT_SHAPE
        image_arr = np.asarray(image.resize((w, h), Image.ANTIALIAS)).reshape(INPUT_SHAPE).\
            astype(np.float32)#.ravel()
        #        transpose([2, 0, 1]).astype(np.float32)#.ravel()
        # This particular ResNet50 model requires some preprocessing, specifically, mean normalization.
        return (image_arr / 255.0 - 0.45) / 0.225

    # Normalize the image and copy to pagelocked memory.
    #np.copyto(pagelocked_buffer, normalize_image(Image.open(test_image)))
    return normalize_image(Image.open(test_image)) #test_image


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType


# iris = load_iris()
# X, y = iris.data, iris.target
# X_train, X_test, y_train, y_test = train_test_split(X, y)
#
# clr = LogisticRegression()
# clr.fit(X_train, y_train)
# print(clr)
#
# initial_type = [('float_input', FloatTensorType([1, 4]))]
# onx = convert_sklearn(clr, initial_types=initial_type)
# with open("logreg_iris.onnx", "wb") as f:
#     f.write(onx.SerializeToString())
#sess = rt.InferenceSession("logreg_iris.onnx")


#dummy_input = np.random.rand(10, 3, 224, 224).astype(np.float32)

data_path, data_files = common.find_sample_data(
    description="Runs a ResNet50 network with a TensorRT inference engine.",
    subfolder="trt", find_files=["binoculars.jpeg", "reflex_camera.jpeg", "tabby_tiger_cat.jpg",
    "/home/snikolaev/pytorch2/joc/rn50/resnet50.onnx", "class_labels.txt"])
# Get test images, models and labels.
test_images = data_files[0:3]
onnx_model_file, labels_file = data_files[3:]
labels = open(labels_file, 'r').read().split('\n')

test_image = random.choice(test_images)
test_case = load_normalized_test_case(test_image)



sess_options = rt.SessionOptions()

#sess_options.enable_profiling = True
#3sess_options.profile_file_prefix = os.path.basename(args.model_path)

#3sess = onnxrt.InferenceSession(args.model_path, sess_options)

sess = rt.InferenceSession(onnx_model_file, sess_options)

meta = sess.get_modelmeta()


input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name
pred_onx = sess.run([label_name], {input_name: test_case})[0]
#print(pred_onx)
# We use the highest probability as our prediction. Its index corresponds to the predicted label.
pred = labels[np.argmax(pred_onx)]
if "_".join(pred.split()) in os.path.splitext(os.path.basename(test_image))[0]:
    print("Correctly recognized " + test_image + " as " + pred)
else:
    print("Incorrectly recognized " + test_image + " as " + pred)
