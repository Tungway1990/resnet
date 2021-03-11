import tensorflow as tf
import keras

input_shape = (224,224,3)
classes=1000

def Conv_block(X,X_shortcut,feature_map,name):
    
    if name[0]=='2' or (name[0]!='2' and name[-2]!='1'):
        X = keras.layers.Conv2D(feature_map, kernel_size = (1, 1), strides=(1,1), name='Conv'+name+str(1))(X) 
    else:
        X = keras.layers.Conv2D(feature_map, kernel_size = (1, 1), strides=(2,2), name='Conv'+name+str(1))(X) 
    
    X = keras.layers.BatchNormalization(axis=3, name='bn_conv'+name+str(1))(X)
    X = keras.layers.Activation('relu',name='RELU_'+name+str(1))(X)
    
    X = keras.layers.ZeroPadding2D((1, 1),name='padding'+name+str(2))(X)
    X = keras.layers.Conv2D(feature_map, kernel_size = (3, 3), strides=(1,1), name='Conv'+name+str(2))(X) 
    X = keras.layers.BatchNormalization(axis=3, name='bn_conv'+name+str(2))(X)
    X = keras.layers.Activation('relu',name='RELU_'+name+str(2))(X)

    X = keras.layers.Conv2D(feature_map*4, kernel_size = (1, 1), strides=(1,1), name='Conv'+name+str(3))(X) 
    X = keras.layers.BatchNormalization(axis=3, name='bn_conv'+name+str(3))(X)

    X=keras.layers.Add(name='Skip_Connect_'+name)([X, X_shortcut])

    X = keras.layers.Activation('relu',name='RELU_'+name+str(3))(X)

    X_shortcut=X
    
    return X,X_shortcut

#Model
def Resnet50(input_shape,classes):
    X_input = keras.layers.Input(input_shape,name='Input')
    '''
    Stardardization of image
    '''
    X = keras.layers.experimental.preprocessing.Rescaling(1./255)(X_input)
    X = keras.layers.ZeroPadding2D((3, 3),name='padding_0')(X)
    '''
    Conv1
    Input shape: 224
    Output shape: 56
    '''
    X = keras.layers.Conv2D(64, (7, 7), strides=(2, 2), name='Conv1')(X)
    X = keras.layers.BatchNormalization(axis=3, name='bn_conv1')(X)
    X = keras.layers.ZeroPadding2D((1, 1),name='padding_1')(X)
    X = keras.layers.Activation('relu',name='RELU_1')(X)
    X = keras.layers.MaxPooling2D((3, 3), strides=(2, 2), name='max2d_1')(X)
    
    X_shortcut=keras.layers.Conv2D(64*4, (1, 1), strides=(1,1))(X)
    
    '''
    Conv2
    Input shape: 56
    Output shape: 56
    '''
    X,X_shortcut=Conv_block(X,X_shortcut,64,'2.1.')
    X,X_shortcut=Conv_block(X,X_shortcut,64,'2.2.')
    X,X_shortcut=Conv_block(X,X_shortcut,64,'2.3.')

    X_shortcut=keras.layers.Conv2D(128*4, (1, 1), strides=(2,2))(X)

    '''
    Conv3
    Input shape: 56
    Output shape: 28
    '''
    X,X_shortcut=Conv_block(X,X_shortcut,128,'3.1.')
    X,X_shortcut=Conv_block(X,X_shortcut,128,'3.2.')
    X,X_shortcut=Conv_block(X,X_shortcut,128,'3.3.')
    
    X_shortcut=keras.layers.Conv2D(256*4, (1, 1), strides=(2,2))(X)
    '''
    Conv4
    Input shape: 28
    Output shape: 14
    '''
    X,X_shortcut=Conv_block(X,X_shortcut,256,'4.1.')
    X,X_shortcut=Conv_block(X,X_shortcut,256,'4.2.')
    X,X_shortcut=Conv_block(X,X_shortcut,256,'4.3.')
    X,X_shortcut=Conv_block(X,X_shortcut,256,'4.4.')
    X,X_shortcut=Conv_block(X,X_shortcut,256,'4.5.')
    X,X_shortcut=Conv_block(X,X_shortcut,256,'4.6.')
    
    X_shortcut=keras.layers.Conv2D(512*4, (1, 1), strides=(2,2))(X)
    '''
    Conv5
    Input shape: 14
    Output shape: 7
    '''
    X,X_shortcut=Conv_block(X,X_shortcut,512,'5.1.')
    X,X_shortcut=Conv_block(X,X_shortcut,512,'5.2.')
    X,X_shortcut=Conv_block(X,X_shortcut,512,'5.3.')

    '''Average pooling, and fully connect layer'''
    X = keras.layers.AveragePooling2D((2,2), name="avg_pool")(X)
    X = keras.layers.Flatten(name='Flatten')(X)
    X=keras.layers.Dense(classes, activation='softmax', name='Fully_connected')(X)

    model = keras.models.Model(inputs = X_input, outputs = X, name='ResNet')
    return model

model=Resnet50(input_shape,classes)
print(model.summary())