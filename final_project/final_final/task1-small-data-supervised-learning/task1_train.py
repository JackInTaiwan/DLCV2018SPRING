import numpy as np
np.random.seed(10)
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

from load_data import load_data, load_data_split, load_image,load_labels,load_labels_split

import sys

train_dir = sys.argv[1]
save_model_path = sys.argv[2]


batch_size = 16
num_classes = 10
epochs = 100

h=28
w=28    



# train_dir = 'Fashion_MNIST_student/train'
# test_dir = 'Fashion_MNIST_student/test'



# X_train,X_val = load_data_split(train_dir)
X_train = load_data(train_dir)

# X_test  =load_image(test_dir)

# X_test = X_test.reshape(10000,h,w,1)

# X_test = X_test.astype('float32')
# X_test /= 127.5
# X_test-=1



Y_train = load_labels()
Y_train = keras.utils.to_categorical(Y_train, num_classes)

# Y_train, Y_val = load_labels_split()
# Y_train = keras.utils.to_categorical(Y_train, num_classes)
# Y_val = keras.utils.to_categorical(Y_val, num_classes)



input_shape = (28,28,1)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

print(model.summary())

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

model.fit(X_train, Y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          # validation_data = (X_val,Y_val)
          )

model.save(save_model_path)

# predict_test = model.predict_classes(X_test)

# print(predict_test)

# f = open( csv_name,'w' )
# f.write( 'image_id,predicted_label\n' )
# for i in range( len( predict_test ) ):
#     f.write( str( i ) + ',' + str( predict_test[ i ] ) + '\n' )
# f.close()



