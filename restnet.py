from tensorflow import keras
from tensorflow.keras.layers import Conv2D,Flatten,Dense,Input,BatchNormalization,Add,AveragePooling2D,ZeroPadding2D

input_shape = [224,224,3]
classes=1000

def Conv_block(X,X_shortcut,feature_map,stride,name):
    for i in range(2):
        X = keras.layers.ZeroPadding2D((1, 1),name='padding'+name+str(i+1))(X)
        X = keras.layers.Conv2D(feature_map, (3, 3), strides=stride, name='Conv'+name+str(i+1))(X) 
        X = keras.layers.BatchNormalization(axis=3, name='bn_conv'+name+str(i+1))(X)
        if i == 0:
            X = keras.layers.Activation('relu',name='RELU_'+name+str(i+1))(X)
        else:
            X=keras.layers.Add(name='Skip_Connect_'+name)([X, X_shortcut])
            X = keras.layers.Activation('relu',name='RELU_'+name)(X)
            X_shortcut=X
    return X,X_shortcut

def Skip_connect_block(X,X_shortcut,feature_map,name):
    X_shortcut=keras.layers.ZeroPadding2D((1, 1),name='skip_connect_padding'+name)(X_shortcut)
    X_shortcut = keras.layers.Conv2D(feature_map, (3, 3), strides=(2,2), name='skip_connect_Conv'+name)(X_shortcut)
    X_shortcut = keras.layers.BatchNormalization(axis=3, name='skip_connect_bn'+name)(X_shortcut)
    
    for i in range(2):
        X = keras.layers.ZeroPadding2D((1, 1),name='padding'+name+str(i+1))(X)
        if i==0:
            X = keras.layers.Conv2D(feature_map, (3, 3), strides=(2,2), name='Conv'+name+str(i+1))(X) 
        else:
            X = keras.layers.Conv2D(feature_map, (3, 3), strides=(1,1), name='Conv'+name+str(i+1))(X) 
            
        X = keras.layers.BatchNormalization(axis=3, name='bn_conv'+name+str(i+1))(X)
        if i == 0:
            X = keras.layers.Activation('relu',name='RELU_'+name+str(i+1))(X)
        else:
            X=keras.layers.Add(name='Skip_Connect_'+name)([X, X_shortcut])
            X = keras.layers.Activation('relu',name='RELU_'+name)(X)
            X_shortcut=X
    return X,X_shortcut
#Model
X_input = keras.layers.Input(input_shape,name='Input')
X = keras.layers.ZeroPadding2D((3, 3),name='padding_0')(X_input)

#Conv1
X = keras.layers.Conv2D(64, (7, 7), strides=(2, 2), name='Conv1')(X)
X = keras.layers.BatchNormalization(axis=3, name='bn_conv1')(X)
X = keras.layers.ZeroPadding2D((1, 1),name='padding_1')(X)
X = keras.layers.Activation('relu',name='RELU_1')(X)
X = keras.layers.MaxPooling2D((3, 3), strides=(2, 2), name='max2d_1')(X)

X_shortcut=X

#Conv2
X,X_shortcut=Conv_block(X,X_shortcut,64,(1, 1),'2.1.')
X,X_shortcut=Conv_block(X,X_shortcut,64,(1, 1),'2.2.')
X,X_shortcut=Conv_block(X,X_shortcut,64,(1, 1),'2.3.')

#Conv3
X,X_shortcut=Skip_connect_block(X,X_shortcut,128,'3.1.')
X,X_shortcut=Conv_block(X,X_shortcut,128,(1, 1),'3.2.')
X,X_shortcut=Conv_block(X,X_shortcut,128,(1, 1),'3.3.')
X,X_shortcut=Conv_block(X,X_shortcut,128,(1, 1),'3.4.')

#Conv4
X,X_shortcut=Skip_connect_block(X,X_shortcut,256,'4.1.')
X,X_shortcut=Conv_block(X,X_shortcut,256,(1, 1),'4.2.')
X,X_shortcut=Conv_block(X,X_shortcut,256,(1, 1),'4.3.')
X,X_shortcut=Conv_block(X,X_shortcut,256,(1, 1),'4.4.')
X,X_shortcut=Conv_block(X,X_shortcut,256,(1, 1),'4.5.')
X,X_shortcut=Conv_block(X,X_shortcut,256,(1, 1),'4.6.')

#Conv5
X,X_shortcut=Skip_connect_block(X,X_shortcut,512,'5.1.')
X,X_shortcut=Conv_block(X,X_shortcut,512,(1, 1),'5.2.')
X,X_shortcut=Conv_block(X,X_shortcut,512,(1, 1),'5.3.')

X = keras.layers.AveragePooling2D((2,2), name="avg_pool")(X)
#Fully connect
X = keras.layers.Flatten(name='Flatten')(X)
X=keras.layers.Dense(classes, activation='softmax', name='Fully_connected')(X)
model = keras.models.Model(inputs = X_input, outputs = X, name='ResNet')

print(model.summary())


