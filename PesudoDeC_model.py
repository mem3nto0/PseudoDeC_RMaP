import tensorflow as tf
from keras.models import Model
from keras.layers import Conv1D, MaxPooling1D , Add, Dropout ,  Conv1DTranspose, Dense
from keras.layers import Input, Activation , Concatenate, LSTM ,  BatchNormalization
from keras.layers import Flatten , TimeDistributed , Resizing , Masking


def swish_activation(x):

    return x*tf.math.sigmoid(x)


def Conv1D_swish_bn(x, N_filters, kernel, strides):

    x = Conv1D(N_filters, kernel, strides=strides, padding="same")(x)
    x = BatchNormalization()(x)
    x = swish_activation(x)

    return x

def Inception_res_block(x, N_filters):

    short_filters = int(0.2*N_filters) + int(0.3*N_filters) +  int(0.5*N_filters) +1

    short = Conv1D(short_filters, 1, strides = 1, padding="same")(x)
    short = BatchNormalization()(short)

    x_Inc_1 = Conv1D_swish_bn(x, int(0.2*N_filters), kernel= 3, strides= 1)
    x_Inc_2 = Conv1D_swish_bn(x_Inc_1, int(0.3*N_filters), kernel= 3, strides= 1)
    x_Inc_3 = Conv1D_swish_bn(x_Inc_2, int(0.5*N_filters) + 1, kernel= 3, strides= 1)

    x_conc = Concatenate(axis=-1)([x_Inc_1, x_Inc_2, x_Inc_3])
    x_conc = BatchNormalization()(x_conc)

    out = Add()([short,x_conc])
    out = swish_activation(out)
    out = BatchNormalization()(out)

    return out


def Inception_resnet2inp(Inp_1, Inp_2, labels):

    input_layer1 = Input((Inp_1,1), name='Input_1')

    x1 = Inception_res_block(input_layer1,64)
    x1 = MaxPooling1D(2)(x1)

    x1 = Inception_res_block(x1,128)
    x1 = MaxPooling1D(2)(x1)

    x1 = Inception_res_block(x1,256)
    x1 = MaxPooling1D(2)(x1)

    x1 = Inception_res_block(x1,384)

    x1 = tf.transpose(x1, perm=[1, 2, 0])
    x1 = Resizing(Inp_2, 384)(x1)
    x1 = tf.transpose(x1, perm=[2, 0, 1])

    input_layer2 = Input((Inp_2,4), name='Input_2')
    masked_input = Masking(mask_value=0.0)(input_layer2)

    x2 = Inception_res_block(masked_input, 64)
    x2 = Inception_res_block(x2, 128)
    x2 = Inception_res_block(x2, 256)

    x_con = Concatenate(axis=-1)([x1,x2])
    x_con = Inception_res_block(x_con, 512)

    x_con = Dropout(0.2)(x_con)

    x_LSTM = tf.keras.layers.Bidirectional(LSTM(64, return_sequences=True))(x_con)
    x_LSTM = tf.keras.layers.Bidirectional(LSTM(64, return_sequences=True))(x_LSTM)
    x_LSTM = tf.keras.layers.Bidirectional(LSTM(64, return_sequences=True))(x_LSTM)

    out_2 = Dense(labels, activation="sigmoid")(x_LSTM)
    model = Model(inputs = [input_layer1, input_layer2] , outputs = [out_2])

    return model
 