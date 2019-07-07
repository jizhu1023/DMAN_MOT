from keras.layers import Input,TimeDistributed,LSTM,Bidirectional
from keras.layers.core import Lambda,Flatten,Dense,Reshape,Activation,Lambda,Permute
from keras.layers.convolutional import Conv2D,UpSampling2D,Conv1D
from keras.layers.pooling import MaxPooling2D, AveragePooling1D,AveragePooling2D
from keras.layers.merge import Add,Concatenate,Dot,Multiply
from keras.regularizers import l2
from keras.optimizers import SGD,Adam
from keras.models import Model
from keras import backend as K
from keras.applications.resnet50 import ResNet50

train_classes = 314

def focal_loss(target, output, gamma=2):
    output /= K.sum(output, axis=-1, keepdims=True)
    eps = K.epsilon()
    output = K.clip(output, eps, 1. - eps)
    return -K.sum(K.pow(1. - output, gamma) * target * K.log(output),
                  axis=-1)

def generate_model(weight_decay=0.0005):

    # spatial attention network
    merged_input = Input(shape=(224, 224, 6))
    split1 = Lambda(lambda x: x[:, :, :, 0:3], name='split1')
    split2 = Lambda(lambda x: x[:, :, :, 3:], name='split2')

    data1 = split1(merged_input)
    data2 = split2(merged_input)

    base_model = ResNet50(weights=None, include_top=False) # weights=None for test, weights='iamgenet' for train
    share_conv_1 = Model(input=base_model.input, output=base_model.get_layer('activation_49').output)

    x1 = share_conv_1(data1)
    x2 = share_conv_1(data2)
    
    reshape1 = Reshape((49, 2048))
    x1 = reshape1(x1)
    x2 = reshape1(x2)

    l2_norm_channel = Lambda(lambda  x: K.l2_normalize(x,axis=-1))
    x1_l2 = l2_norm_channel(x1)
    x2_l2 = l2_norm_channel(x2)
    x2_l2 = Permute((2, 1))(x2_l2)

    matrix_dot = Lambda(lambda x: K.batch_dot(x[0], x[1]))
    x_com = matrix_dot([x1_l2, x2_l2])
    x_com_T = Permute((2, 1))(x_com)

    share_conv_2 = Conv1D(1, 1, padding="same", kernel_regularizer=l2(weight_decay))
    x1_att = share_conv_2(x_com)
    x2_att = share_conv_2(x_com_T)

    reshape2 = Reshape((49,))
    x1_att = reshape2(x1_att)
    x2_att = reshape2(x2_att)

    softmax = Activation('softmax')
    x1_att = softmax(x1_att)
    x2_att = softmax(x2_att)

    reshape3 = Reshape((49, 1))
    x1_att = reshape3(x1_att)
    x2_att = reshape3(x2_att)

    h1 = Multiply()([x1, x1_att])
    h2 = Multiply()([x2, x2_att])

    summary = Lambda(lambda x: K.sum(x, axis=1))
    h1 = summary(h1)
    h2 = summary(h2)

    id_layer = Dense(train_classes, kernel_regularizer=l2(weight_decay), activation='softmax') 
    y1 = id_layer(h1)
    y2 = id_layer(h2)
    x_concat = Concatenate()([h1, h2])
    x_concat = Dense(512, kernel_regularizer=l2(weight_decay), activation='relu')(x_concat)

    spatial_model = Model(inputs=merged_input, outputs=[y1, y2, x_concat]) # spatial attention model
    spatial_model.summary()
    #spatial_model.load_weights('/media/tensend/dish_disk/MOT_keras/weights_spatial_dot_softmax/my_weights_on_mot16_0_0_15.h5', by_name=True) # fix the weights of the spatial attention network to train the temporal attention network
    for layer in spatial_model.layers[:]:
        layer.trainable = False
    spatial_model.layers[-2].trainable = True
    print spatial_model.layers[-2].name

    # temporal attention network
    time_steps = 8
    seq_merged_input = Input(shape=(time_steps, 224, 224, 6))
    ST_outputs = []
    for i in range(2):
        ST_outputs.append(TimeDistributed(Model(spatial_model.input, spatial_model.output[i]))(seq_merged_input))
    lstm_input = TimeDistributed(Model(spatial_model.input, spatial_model.output[2]))(seq_merged_input)
    temporal_model = Bidirectional(LSTM(512, kernel_regularizer=l2(weight_decay), recurrent_regularizer=l2(weight_decay), return_sequences=True))(lstm_input)
    beta = TimeDistributed(Dense(1, kernel_regularizer=l2(weight_decay)))(temporal_model)
    beta = Reshape((time_steps,))(beta)
    beta = Activation('softmax')(beta)
    beta = Reshape((time_steps, 1))(beta)
    weighted_output = Multiply()([temporal_model, beta])
    summary = Lambda(lambda x: K.sum(x, 1))
    h = summary(weighted_output)

    yf = Dense(2, kernel_regularizer=l2(weight_decay), activation='softmax')(h)
    ST_outputs.append(yf)
    ST_model = Model(inputs=seq_merged_input, outputs=ST_outputs)
    ST_model.summary()

    return ST_model

def compile_model(model, *args, **kw):
    
    class SGD_new(SGD):
        '''
        redefinition of the original SGD
        '''
        def __init__(self, lr=0.01, momentum=0., decay=0.,
                     nesterov=False, **kwargs):
            super(SGD, self).__init__(**kwargs)
            self.__dict__.update(locals())
            self.iterations = K.variable(0.)
            self.lr = K.variable(lr)
            self.momentum = K.variable(momentum)
            self.decay = K.variable(decay)
            self.inital_decay = decay
    
        def get_updates(self, params, constraints, loss):
            grads = self.get_gradients(loss, params)
            self.updates = []
    
            lr = self.lr
            if self.inital_decay > 0:
                lr *= (1. / (1. + self.decay * self.iterations)) ** 0.75
                self.updates .append(K.update_add(self.iterations, 1))
    
            # momentum
            shapes = [K.get_variable_shape(p) for p in params]
            moments = [K.zeros(shape) for shape in shapes]
            self.weights = [self.iterations] + moments
            for p, g, m in zip(params, grads, moments):
                v = self.momentum * m - lr * g  # velocity
                self.updates.append(K.update(m, v))
    
                if self.nesterov:
                    new_p = p + self.momentum * v - lr * g
                else:
                    new_p = p + v
    
                # apply constraints
                if p in constraints:
                    c = constraints[p]
                    new_p = c(new_p)
    
                self.updates.append(K.update(p, new_p))
            return self.updates 
    all_classes = {
        'sgd_new': 'SGD_new(lr=0.01, momentum=0.9)',        
        'sgd': 'SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)',
        'rmsprop': 'RMSprop(lr=0.0001, rho=0.9, epsilon=1e-06)',
        'adagrad': 'Adagrad(lr=0.01, epsilon=1e-06)',
        'adadelta': 'Adadelta(lr=1.0, rho=0.95, epsilon=1e-06)',
        'adam': 'Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)',
        'adamax': 'Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08)',
        'nadam': 'Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)',
    }
    param = {'optimizer': 'adam', 'loss': 'categorical_crossentropy', 'metrics': 'accuracy'}
    config = ''
    if len(kw):    
        for (key, value) in kw.items():
            if key in param:            
                param[key] = kw[key]
            elif key in all_classes:
                config = kw[key]
            else:
                print 'error'
    if not len(config):
        config = all_classes[param['optimizer']]
    optimiz = eval(config)

    model.compile(optimizer=optimiz,
              loss=['categorical_crossentropy', 'categorical_crossentropy', focal_loss],
              loss_weights=[0.5, 0.5, 1.0],
              metrics=['accuracy'])
    
    print("Model Compile Successful.")
    return model

if __name__ == "__main__":
    """
    Just for model testing.
    """
    model = generate_model()
    model = compile_model(model)