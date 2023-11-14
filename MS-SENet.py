import sys
import tensorflow.keras.backend as K
import tensorflow as tf
import os
import numpy as np
from tensorflow.keras import callbacks
from tensorflow.keras.optimizers import SGD,Adam
from tensorflow.keras.layers import Activation, Lambda, Dense
from tensorflow.keras.layers import Conv1D, Conv2D, SpatialDropout1D,add,GlobalAveragePooling1D,AveragePooling2D,ReLU
from tensorflow.keras.layers import BatchNormalization,Concatenate,SpatialDropout2D
from tensorflow.keras.activations import sigmoid

from tensorflow.keras.layers import Layer,Dense,Input
from tensorflow.keras.models import Model
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import datetime
import pandas as pd

import argparse
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class Common_Model(object):

    def __init__(self, save_path: str = '', name: str = 'Not Specified'):
        self.model = None
        self.trained = False

    def train(self, x_train, y_train, x_val, y_val):
        raise NotImplementedError()

    def predict(self, samples):
        raise NotImplementedError()


    def predict_proba(self, samples):
        if not self.trained:
            sys.stderr.write("No Model.")
            sys.exit(-1)
        return self.model.predict_proba(samples)

    def save_model(self, model_name: str):
        raise NotImplementedError()


def Temporal_Aware_Block(x, s, i, activation, nb_filters, kernel_size, dropout_rate=0, name=''):

    original_x = x
    #1.1
    conv_1_1 = Conv1D(filters=nb_filters, kernel_size=kernel_size,
                  dilation_rate=i, padding='causal')(x)
    conv_1_1 = BatchNormalization(trainable=True,axis=-1)(conv_1_1)
    conv_1_1 =  Activation(activation)(conv_1_1)
    output_1_1 =  SpatialDropout1D(dropout_rate)(conv_1_1)
    # 2.1
    conv_2_1 = Conv1D(filters=nb_filters, kernel_size=kernel_size,
                  dilation_rate=i, padding='causal')(output_1_1)
    conv_2_1 = BatchNormalization(trainable=True,axis=-1)(conv_2_1)
    conv_2_1 = Activation(activation)(conv_2_1)
    output_2_1 =  SpatialDropout1D(dropout_rate)(conv_2_1)

    if original_x.shape[-1] != output_2_1.shape[-1]:
        original_x = Conv1D(filters=nb_filters, kernel_size=1, padding='same')(original_x)

    output_2_1 = Lambda(sigmoid)(output_2_1)
    F_x = Lambda(lambda x: tf.multiply(x[0], x[1]))([original_x, output_2_1])
    return F_x

def se_module(inputs):
    # Squeeze 阶段：全局平均池化
    squeeze = tf.reduce_mean(inputs, axis=[1, 2])

    # Excitation 阶段：全连接层
    excitation = Dense(units=39, activation='relu')(squeeze)
    excitation = Dense(units=39, activation='sigmoid')(excitation)

    # Scale 阶段：将学习到的权重应用于输入特征
    excitation = tf.expand_dims(excitation, axis=1)
    excitation = tf.expand_dims(excitation, axis=1)
    scaled_inputs = inputs * excitation

    return scaled_inputs

class TIMNET:
    def __init__(self,
                 nb_filters=64,
                 kernel_size=2,
                 nb_stacks=1,
                 dilations=None,
                 activation = "relu",
                 dropout_rate=0.1,
                 return_sequences=True,
                 name='TIMNET'):
        self.name = name
        self.return_sequences = return_sequences
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.dilations = dilations
        self.nb_stacks = nb_stacks
        self.kernel_size = kernel_size
        self.nb_filters = nb_filters

        self.supports_masking = True
        self.mask_value=0.

        if not isinstance(nb_filters, int):
            raise Exception()

    def __call__(self, inputs, mask=None):
        if self.dilations is None:
            self.dilations = 8
        forward = inputs
        #backward = inputs
        #forward = K.reverse(inputs,axes=1)
        backward = K.reverse(inputs,axes=1)
        forward = tf.expand_dims(forward, axis=-1)
        backward = tf.expand_dims(backward, axis=-1)
        print("Input Shape=",inputs.shape)
        path1 = Conv2D(39, (11,1), padding="same", strides=(1,1))(forward)
        path2 = Conv2D(39, (1, 9), padding="same", strides=(1,1))(forward)
        path3 = Conv2D(39, (3, 3), padding="same", strides=(1,1))(forward)
        path1 = BatchNormalization()(path1)
        path2 = BatchNormalization()(path2)
        path3 = BatchNormalization()(path3)
        path1 = ReLU()(path1)
        path2 = ReLU()(path2)
        path3 = ReLU()(path3)
        path1 = SpatialDropout2D(0.2)(path1)
        path2 = SpatialDropout2D(0.2)(path2)
        path3 = SpatialDropout2D(0.2)(path3)


        '''
        path1 = AveragePooling2D(pool_size=2, padding="same")(path1)
        path2 = AveragePooling2D(pool_size=2, padding="same")(path2)
        path3 = AveragePooling2D(pool_size=2, padding="same")(path3)
        '''

        forward = tf.keras.layers.Concatenate(axis=2)([path1, path2, path3])
        forward = se_module((forward))
        forward = tf.keras.layers.Conv2D(filters=1, kernel_size=(1, 1), activation='relu')(forward)
        forward = forward[:, :, :, 0]
        print(forward.shape)
        forward = tf.keras.layers.Concatenate(axis=2)([forward, inputs])


        path1 = Conv2D(39, (11,1), padding="same", strides=(1,1))(backward)
        path2 = Conv2D(39, (1, 9), padding="same", strides=(1,1))(backward)
        path3 = Conv2D(39, (3, 3), padding="same", strides=(1,1))(backward)
        path1 = BatchNormalization()(path1)
        path2 = BatchNormalization()(path2)
        path3 = BatchNormalization()(path3)
        path1 = ReLU()(path1)
        path2 = ReLU()(path2)
        path3 = ReLU()(path3)
        path1 = SpatialDropout2D(0.2)(path1)
        path2 = SpatialDropout2D(0.2)(path2)
        path3 = SpatialDropout2D(0.2)(path3)
        '''
        path1 = AveragePooling2D(pool_size=2, padding="same")(path1)
        path2 = AveragePooling2D(pool_size=2, padding="same")(path2)
        path3 = AveragePooling2D(pool_size=2, padding="same")(path3)
        '''

        backward = tf.keras.layers.Concatenate(axis=2)([path1, path2, path3])
        backward = se_module(backward)
        backward = tf.keras.layers.Conv2D(filters=1, kernel_size=(1, 1), activation='relu')(backward)
        backward = backward[:, :, :, 0]
        backward = tf.keras.layers.Concatenate(axis=2)([backward, K.reverse(inputs, axes=1)])


        forward_convd = Conv1D(filters=self.nb_filters,kernel_size=1, dilation_rate=1, padding='causal')(forward)
        backward_convd = Conv1D(filters=self.nb_filters,kernel_size=1, dilation_rate=1, padding='causal')(backward)



        final_skip_connection = []

        skip_out_forward = forward_convd
        skip_out_backward = backward_convd

        for s in range(self.nb_stacks):
            for i in [2 ** i for i in range(self.dilations)]:
                skip_out_forward = Temporal_Aware_Block(skip_out_forward, s, i, self.activation,
                                                        self.nb_filters,
                                                        self.kernel_size,
                                                        self.dropout_rate,
                                                        name=self.name)
                skip_out_backward = Temporal_Aware_Block(skip_out_backward, s, i, self.activation,
                                                        self.nb_filters,
                                                        self.kernel_size,
                                                        self.dropout_rate,
                                                        name=self.name)

                temp_skip = add([skip_out_forward, skip_out_backward],name = "biadd_"+str(i))
                temp_skip=GlobalAveragePooling1D()(temp_skip)
                temp_skip=tf.expand_dims(temp_skip, axis=1)
                final_skip_connection.append(temp_skip)

        output_2 = final_skip_connection[0]
        for i,item in enumerate(final_skip_connection):
            if i==0:
                continue
            output_2 = K.concatenate([output_2,item],axis=-2)
        x = output_2

        return x



def smooth_labels(labels, factor=0.1):
    # smooth the labels
    labels *= (1 - factor)
    labels += (factor / labels.shape[1])
    return labels

class WeightLayer(Layer):
    def __init__(self, **kwargs):
        super(WeightLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[1],1),
                                      initializer='uniform',
                                      trainable=True)
        super(WeightLayer, self).build(input_shape)

    def call(self, x):
        tempx = tf.transpose(x,[0,2,1])
        x = K.dot(tempx,self.kernel)
        x = tf.squeeze(x,axis=-1)
        return  x

    def compute_output_shape(self, input_shape):
        return (input_shape[0],input_shape[2])

def softmax(x, axis=-1):
    ex = K.exp(x - K.max(x, axis=axis, keepdims=True))
    return ex/K.sum(ex, axis=axis, keepdims=True)

class TIMNET_Model(Common_Model):
    def __init__(self, args, input_shape, class_label, **params):
        super(TIMNET_Model,self).__init__(**params)
        self.args = args
        self.data_shape = input_shape
        self.num_classes = len(class_label)
        self.class_label = class_label
        self.matrix = []
        self.eva_matrix = []
        self.acc = 0
        print("TIMNET MODEL SHAPE:",input_shape)

    def create_model(self):
        self.inputs=Input(shape = (self.data_shape[0],self.data_shape[1]))
        self.multi_decision = TIMNET(nb_filters=self.args.filter_size,
                                kernel_size=self.args.kernel_size,
                                nb_stacks=self.args.stack_size,
                                dilations=self.args.dilation_size,
                                dropout_rate=self.args.dropout,
                                activation = self.args.activation,
                                return_sequences=True,
                                name='TIMNET')(self.inputs)

        self.decision = WeightLayer()(self.multi_decision)
        self.predictions = Dense(self.num_classes, activation='softmax')(self.decision)
        self.model = Model(inputs = self.inputs, outputs = self.predictions)

        self.model.compile(loss = "categorical_crossentropy",
                           optimizer =Adam(learning_rate=self.args.lr, beta_1=self.args.beta1, beta_2=self.args.beta2, epsilon=1e-8),
                           metrics = ['accuracy'])
        print("Temporal create succes!")

    def train(self, x, y):

        filepath = self.args.model_path
        resultpath = self.args.result_path

        if not os.path.exists(filepath):
            os.mkdir(filepath)
        if not os.path.exists(resultpath):
            os.mkdir(resultpath)

        i=1
        now = datetime.datetime.now()
        now_time = datetime.datetime.strftime(now,'%Y-%m-%d_%H-%M-%S')
        kfold = KFold(n_splits=self.args.split_fold, shuffle=True, random_state=self.args.random_seed)
        avg_accuracy = 0
        avg_loss = 0
        for train, test in kfold.split(x, y):
            self.create_model()
            y[train] = smooth_labels(y[train], 0.1)
            folder_address = filepath+self.args.data+"_"+str(self.args.random_seed)+"_"+now_time
            if not os.path.exists(folder_address):
                os.mkdir(folder_address)
            weight_path=folder_address+'/'+str(self.args.split_fold)+"-fold_weights_best_"+str(i)+".hdf5"
            checkpoint = callbacks.ModelCheckpoint(weight_path, monitor='val_accuracy', verbose=1,save_weights_only=False,save_best_only=False,mode='max')
            max_acc = 0
            best_eva_list = []
            h = self.model.fit(x[train], y[train],validation_data=(x[test],  y[test]),batch_size = self.args.batch_size, epochs = self.args.epoch, verbose=1,callbacks=[checkpoint])
            self.model.load_weights(weight_path)
            best_eva_list = self.model.evaluate(x[test],  y[test])
            avg_loss += best_eva_list[0]
            avg_accuracy += best_eva_list[1]
            print(str(i)+'_Model evaluation: ', best_eva_list,"   Now ACC:",str(round(avg_accuracy*10000)/100/i))
            i+=1
            y_pred_best = self.model.predict(x[test])
            self.matrix.append(confusion_matrix(np.argmax(y[test],axis=1),np.argmax(y_pred_best,axis=1)))
            em = classification_report(np.argmax(y[test],axis=1),np.argmax(y_pred_best,axis=1), target_names=self.class_label,output_dict=True)
            self.eva_matrix.append(em)
            print(classification_report(np.argmax(y[test],axis=1),np.argmax(y_pred_best,axis=1), target_names=self.class_label,digits=6))

        print("Average ACC:",avg_accuracy/self.args.split_fold)
        self.acc = avg_accuracy/self.args.split_fold
        writer = pd.ExcelWriter(resultpath+self.args.data+'_'+str(self.args.split_fold)+'fold_'+str(round(self.acc*10000)/100)+"_"+str(self.args.random_seed)+"_"+now_time+'.xlsx')
        for i,item in enumerate(self.matrix):
            temp = {}
            temp[" "] = self.class_label
            for j,l in enumerate(item):
                temp[self.class_label[j]]=item[j]
            data1 = pd.DataFrame(temp)
            data1.to_excel(writer,sheet_name=str(i), encoding='utf8')

            df = pd.DataFrame(self.eva_matrix[i]).transpose()
            df.to_excel(writer,sheet_name=str(i)+"_evaluate", encoding='utf8')
        writer.save()
        writer.close()

        K.clear_session()
        self.matrix = []
        self.eva_matrix = []
        self.acc = 0
        self.trained = True

    def test(self, x, y, path):
        i=1
        kfold = KFold(n_splits=self.args.split_fold, shuffle=True, random_state=self.args.random_seed)
        avg_accuracy = 0
        avg_loss = 0
        x_feats = []
        y_labels = []
        for train, test in kfold.split(x, y):
            self.create_model()
            weight_path=path+'/'+str(self.args.split_fold)+"-fold_weights_best_"+str(i)+".hdf5"
            self.model.fit(x[train], y[train],validation_data=(x[test],  y[test]),batch_size = 64,epochs = 0,verbose=0)
            self.model.load_weights(weight_path)#+source_name+'_single_best.hdf5')
            best_eva_list = self.model.evaluate(x[test],  y[test])
            avg_loss += best_eva_list[0]
            avg_accuracy += best_eva_list[1]
            print(str(i)+'_Model evaluation: ', best_eva_list,"   Now ACC:",str(round(avg_accuracy*10000)/100/i))
            i+=1
            y_pred_best = self.model.predict(x[test])
            self.matrix.append(confusion_matrix(np.argmax(y[test],axis=1),np.argmax(y_pred_best,axis=1)))
            em = classification_report(np.argmax(y[test],axis=1),np.argmax(y_pred_best,axis=1), target_names=self.class_label,output_dict=True)
            self.eva_matrix.append(em)
            print(classification_report(np.argmax(y[test],axis=1),np.argmax(y_pred_best,axis=1), target_names=self.class_label,digits=6))
            caps_layer_model = Model(inputs=self.model.input,
            outputs=self.model.get_layer(index=-2).output)
            feature_source = caps_layer_model.predict(x[test])
            x_feats.append(feature_source)
            y_labels.append(y[test])
        print("Average ACC:",avg_accuracy/self.args.split_fold)
        self.acc = avg_accuracy/self.args.split_fold
        return x_feats, y_labels



parser = argparse.ArgumentParser()


parser.add_argument('--mode', type=str, default="train")
parser.add_argument('--model_path', type=str, default='./Models/')
parser.add_argument('--result_path', type=str, default='./Results/')
parser.add_argument('--test_path', type=str, default='/home/j-limengbo-jk/TIM-Net_SER/Code/Models/EMODB_46_2023-08-09_15-17-56')
parser.add_argument('--data', type=str, default='SAVEE')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--beta1', type=float, default=0.93)
parser.add_argument('--beta2', type=float, default=0.98)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--epoch', type=int, default=200)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--random_seed', type=int, default=44)
#parser.add_argument('--activation', type=str, default='LeakyReLU')
parser.add_argument('--activation', type=str, default='relu')
parser.add_argument('--filter_size', type=int, default=39)
parser.add_argument('--dilation_size', type=int, default = 10)# If you want to train model on IEMOCAP, you should modify this parameter to 10 due to the long duration of speech signals.
parser.add_argument('--kernel_size', type=int, default=2)
parser.add_argument('--stack_size', type=int, default=1)
parser.add_argument('--split_fold', type=int, default=10)
parser.add_argument('--gpu', type=str, default='3')

#args = parser.parse_args()
args = parser.parse_known_args()[0]
if args.data=="IEMOCAP" and args.dilation_size!=10:
    args.dilation_size = 10

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
session = tf.compat.v1.Session(config=config)
print(f"###gpus:{gpus}")

EMOPIA_LABELS = ('angry', 'happy', 'sad', 'neutral')
TELE_LABELS = ("normal", "abnormal")
CLASS_LABELS_finetune = ("angry", "fear", "happy", "neutral","sad")
CASIA_CLASS_LABELS = ("angry", "fear", "happy", "neutral", "sad", "surprise")#CASIA
EMODB_CLASS_LABELS = ("angry", "boredom", "disgust", "fear", "happy", "neutral", "sad")#EMODB
SAVEE_CLASS_LABELS = ("angry","disgust", "fear", "happy", "neutral", "sad", "surprise")#SAVEE
RAVDE_CLASS_LABELS = ("angry", "calm", "disgust", "fear", "happy", "neutral","sad","surprise")#rav
IEMOCAP_CLASS_LABELS = ("angry", "happy", "neutral", "sad")#iemocap
EMOVO_CLASS_LABELS = ("angry", "disgust", "fear", "happy","neutral","sad","surprise")#emovo
CLASS_LABELS_dict = {"CASIA": CASIA_CLASS_LABELS,
               "EMODB": EMODB_CLASS_LABELS,
               "EMOVO": EMOVO_CLASS_LABELS,
               "IEMOCAP": IEMOCAP_CLASS_LABELS,
               "RAVDE": RAVDE_CLASS_LABELS,
               "SAVEE": SAVEE_CLASS_LABELS,
               "TELE":TELE_LABELS,
               "EMOPIA":EMOPIA_LABELS}


data = np.load("/home/j-limengbo-jk/"+args.data+".npy",allow_pickle=True).item()
x_source = data["x"]
y_source = data["y"]
CLASS_LABELS = CLASS_LABELS_dict[args.data]
'''
x_source = np.load("/content/drive/MyDrive/emodata_30s.npy",allow_pickle=True)
y_source = np.load("/content/drive/MyDrive/emodata_y7.npy",allow_pickle=True)
CLASS_LABELS = CLASS_LABELS_dict[args.data]
print(CLASS_LABELS)
print(x_source.shape)
print(y_source.shape)'''

model = TIMNET_Model(args=args, input_shape=x_source.shape[1:], class_label= CLASS_LABELS)
if args.mode=="train":
    model.train(x_source, y_source)
elif args.mode=="test":
    x_feats, y_labels = model.test(x_source, y_source, path=args.test_path)# x_feats and y_labels are test datas for t-sne
