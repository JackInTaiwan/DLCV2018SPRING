import numpy as np
np.random.seed(10)
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import load_model
from keras import backend as K

from load_data import load_data, load_data_split, load_image,load_labels,load_labels_split

import sys
import os


test_dir = sys.argv[1]
csv_dir = sys.argv[2]

csv_name = os.path.join(csv_dir,'predict.csv')

h=28
w=28
num_classes = 10

X_test  =load_image(test_dir)
X_test = X_test.reshape(10000,h,w,1)
X_test = X_test.astype('float32')
X_test /= 127.5
X_test-=1



Y_train = load_labels()
Y_train = keras.utils.to_categorical(Y_train, num_classes)




model = load_model('2conv.h5')



# model.save(save_model_path)

predict_test = model.predict_classes(X_test)

# print(predict_test)

f = open( csv_name,'w' )
f.write( 'image_id,predicted_label\n' )
for i in range( len( predict_test ) ):
    f.write( str( i ) + ',' + str( predict_test[ i ] ) + '\n' )
f.close()
