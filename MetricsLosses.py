from keras import backend as K
from keras.layers import Input, Dense
from keras.models import Model
from keras.models import load_model
from keras.models import Sequential
from keras.layers.core import Dense,Activation, Dropout

def kullback_leibler_divergence(y_true, y_pred):
    y_true = K.clip(y_true, K.epsilon(), 1)
    y_pred = K.clip(y_pred, K.epsilon(), 1)
    return K.sum(y_true * K.log(y_true / y_pred), axis=-1)

# def contractive_loss(file_name):
    # custom_obj['contractive_loss']=contractive_loss
    # model = load_model(file_name, custom_objects=custom_obj)
    # def loss(y_pred, y_true):
        # lam = 1e-4
        # mse = K.mean(K.square(y_true - y_pred), axis=1)

        # W = K.variable(value=model.get_layer('encoded').get_weights()[0])  # N x N_hidden
        # W = K.transpose(W)  # N_hidden x N
        # h = model.get_layer('encoded').output
        # dh = h * (1 - h)  # N_batch x N_hidden

        # # N_batch x N_hidden * N_hidden x 1 = N_batch x 1
        # contractive = lam * K.sum(dh**2 * K.sum(W**2, axis=1), axis=1)

        # return mse + contractive

def contractive_loss(hn,input_dim,hidden_activation,output_activation):
    #custom_obj={}
    #custom_obj['contractive_loss']=contractive_loss(file_name)
    #model = load_model(file_name, custom_objects=custom_obj)
    model = Sequential()
    model.add(Dense(hn, input_dim=input_dim, name='encoded'))
    model.add(Activation(hidden_activation))
    model.add(Dense(input_dim))
    model.add(Activation(output_activation))
    def loss(y_pred, y_true):
        lam = 1e-4
        mse = K.mean(K.square(y_true - y_pred), axis=1)

        W = K.variable(value=model.get_layer('encoded').get_weights()[0])  # N x N_hidden
        W = K.transpose(W)  # N_hidden x N
        h = model.get_layer('encoded').output
        dh = h * (1 - h)  # N_batch x N_hidden

        # N_batch x N_hidden * N_hidden x 1 = N_batch x 1
        contractive = lam * K.sum(dh**2 * K.sum(W**2, axis=1), axis=1)

        return mse + contractive
    return loss
