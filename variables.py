import os 
train_dir = os.path.join(os.getcwd(), 'data/Train/')
test_dir = os.path.join(os.getcwd(), 'data/Test/')
val_dir = os.path.join(os.getcwd(), 'data/Val/')
model_weights = "data/weights/model_weights.h5"
model_architecture = "data/weights/model_architecture.json"

test_split = 0.2
val_split = 0.15
seed = 42

crop_size_middle = 150
target_size=(224, 224)
batch_size = 8
valid_size = 4
color_mode = 'rgb'
width = 224
height = 224
target_size = (width, height)
input_shape = (width, height, 3)
shear_range = 0.2
zoom_range = 0.15
rotation_range = 20
shift_range = 0.2
rescale = 1./255
dense_1 = 512
dense_2 = 256
dense_3 = 64
epochs = 10
verbose = 1