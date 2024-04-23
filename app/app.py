import streamlit as st
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np
import tensorflow as tf
tf.config.list_physical_devices('GPU')
from PIL import Image
import metrics
import matplotlib.pyplot as plt
from tqdm import *
from skimage.color import rgb2gray


st.markdown("<h1 style='text-align: center; color: white;'>Deep Steganography </h1>", unsafe_allow_html=True)

def file_selector(folder_path='./images'):
    filenames = os.listdir(folder_path)
    st.markdown("<h2 style='text-align: center; color: white;'>Select the secret image</h2>", unsafe_allow_html=True)
    selected_filename = st.selectbox("", filenames, key = "secret")
    return os.path.join(folder_path, selected_filename)

def file_selector2(folder_path='./images'):
    filenames = os.listdir(folder_path)
    st.markdown("<h2 style='text-align: center; color: white;'>Select the cover image</h2>", unsafe_allow_html=True)

    selected_filename = st.selectbox('Select a cover image', filenames, key = "cover")
    return os.path.join(folder_path, selected_filename)



class ConvLayer(tf.keras.layers.Layer):
    def __init__(self, n_layers, filters=50, kernel_size=(3, 3), activation=tf.nn.relu, **kwargs):
        super().__init__(**kwargs)
        self.convs = []
        for conv in range(n_layers):
            self.convs.append(
                tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, activation=activation, padding='same')
            )

    def call(self, input_tensor, training=False):
        x = self.convs[0](input_tensor, training=training)
        for i in range(1, len(self.convs)):
            x = self.convs[i](x, training=training)

        return x
    
class PrepLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.conv_layer_4_3x3 = ConvLayer(4, filters=50, kernel_size=(3, 3), activation=tf.nn.relu)
        self.conv_layer_4_4x4 = ConvLayer(4, filters=50, kernel_size=(4, 4), activation=tf.nn.relu)
        self.conv_layer_4_5x5 = ConvLayer(4, filters=50, kernel_size=(5, 5), activation=tf.nn.relu)

        self.concat_1 = tf.keras.layers.Concatenate(axis=3)

        self.conv_1_3x3 = ConvLayer(1, filters=50, kernel_size=(3, 3), activation=tf.nn.relu)
        self.conv_1_4x4 = ConvLayer(1, filters=50, kernel_size=(4, 4), activation=tf.nn.relu)
        self.conv_1_5x5 = ConvLayer(1, filters=50, kernel_size=(5, 5), activation=tf.nn.relu)

        self.concat_2 = tf.keras.layers.Concatenate(axis=3)

    def call(self, input_tensor, training=False):
        prep_input = tf.keras.layers.Rescaling(1./255, input_shape=input_tensor.shape)(input_tensor)
        conv_4_3x3 = self.conv_layer_4_3x3(prep_input, training=training)
        conv_4_4x4 = self.conv_layer_4_4x4(prep_input, training=training)
        conv_4_5x5 = self.conv_layer_4_5x5(prep_input, training=training)

        concat_1 = self.concat_1([conv_4_3x3, conv_4_4x4, conv_4_5x5])

        conv_1_3x3 =  self.conv_1_3x3(concat_1)
        conv_1_4x4 =  self.conv_1_4x4(concat_1)
        conv_1_5x5 =  self.conv_1_5x5(concat_1)

        return self.concat_2([conv_1_3x3, conv_1_4x4, conv_1_5x5])

class HideLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.prep_layer = PrepLayer()
        self.concat_1 = tf.keras.layers.Concatenate(axis=3)

        self.conv_layer_4_3x3 = ConvLayer(4, filters=50, kernel_size=(3, 3), activation=tf.nn.relu)
        self.conv_layer_4_4x4 = ConvLayer(4, filters=50, kernel_size=(4, 4), activation=tf.nn.relu)
        self.conv_layer_4_5x5 = ConvLayer(4, filters=50, kernel_size=(5, 5), activation=tf.nn.relu)

        self.concat_2 = tf.keras.layers.Concatenate(axis=3)

        self.conv_1_3x3 = ConvLayer(1, filters=50, kernel_size=(3, 3), activation=tf.nn.relu)
        self.conv_1_4x4 = ConvLayer(1, filters=50, kernel_size=(4, 4), activation=tf.nn.relu)
        self.conv_1_5x5 = ConvLayer(1, filters=50, kernel_size=(5, 5), activation=tf.nn.relu)

        self.concat_3 = tf.keras.layers.Concatenate(axis=3)

        self.conv_1_1x1 = ConvLayer(1, filters=3, kernel_size=(1, 1), activation=tf.nn.relu)

    def call(self, input_tensor, training=False):
        prep_input = input_tensor[0]
        hide_input = tf.keras.layers.Rescaling(1./255, input_shape=input_tensor[1].shape)(input_tensor[1])
        concat_1 = self.concat_1([prep_input, hide_input])

        conv_4_3x3 = self.conv_layer_4_3x3(concat_1, training=training)
        conv_4_4x4 = self.conv_layer_4_4x4(concat_1, training=training)
        conv_4_5x5 = self.conv_layer_4_5x5(concat_1, training=training)

        concat_2 = self.concat_2([conv_4_3x3, conv_4_4x4, conv_4_5x5])

        conv_1_3x3 =  self.conv_1_3x3(concat_2)
        conv_1_4x4 =  self.conv_1_4x4(concat_2)
        conv_1_5x5 =  self.conv_1_5x5(concat_2)

        concat_3 = self.concat_3([conv_1_3x3, conv_1_4x4, conv_1_5x5])

        return self.conv_1_1x1(concat_3)
    

class RevealLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.conv_layer_4_3x3 = ConvLayer(4, filters=50, kernel_size=(3, 3), activation=tf.nn.relu)
        self.conv_layer_4_4x4 = ConvLayer(4, filters=50, kernel_size=(4, 4), activation=tf.nn.relu)
        self.conv_layer_4_5x5 = ConvLayer(4, filters=50, kernel_size=(5, 5), activation=tf.nn.relu)

        self.concat_1 = tf.keras.layers.Concatenate(axis=3)

        self.conv_1_3x3 = ConvLayer(1, filters=50, kernel_size=(3, 3), activation=tf.nn.relu)
        self.conv_1_4x4 = ConvLayer(1, filters=50, kernel_size=(4, 4), activation=tf.nn.relu)
        self.conv_1_5x5 = ConvLayer(1, filters=50, kernel_size=(5, 5), activation=tf.nn.relu)

        self.concat_2 = tf.keras.layers.Concatenate(axis=3)

        self.conv_1_1x1 = ConvLayer(1, filters=3, kernel_size=(1, 1), activation=tf.nn.relu)

    def call(self, input_tensor, training=False):

        conv_4_3x3 = self.conv_layer_4_3x3(input_tensor, training=training)
        conv_4_4x4 = self.conv_layer_4_4x4(input_tensor, training=training)
        conv_4_5x5 = self.conv_layer_4_5x5(input_tensor, training=training)

        concat_1 = self.concat_1([conv_4_3x3, conv_4_4x4, conv_4_5x5])

        conv_1_3x3 =  self.conv_1_3x3(concat_1)
        conv_1_4x4 =  self.conv_1_4x4(concat_1)
        conv_1_5x5 =  self.conv_1_5x5(concat_1)

        concat_2 = self.concat_2([conv_1_3x3, conv_1_4x4, conv_1_5x5])

        return self.conv_1_1x1(concat_2)
    
class StenographyLoss(tf.keras.losses.Loss):
    def __init__(self, beta=1.0, **kwargs):
        super().__init__(**kwargs)
        self.beta = beta

    def call(self, y_true, y_pred):
        beta = tf.constant(self.beta, name='beta')

        secret_true = y_true[0]
        secret_pred = y_pred[0]

        cover_true = y_true[1]
        cover_pred = y_pred[1]

        secret_mse = tf.losses.MSE(secret_true, secret_pred)
        cover_mse = tf.losses.MSE(cover_true, cover_pred)

        return tf.reduce_mean(cover_mse + beta * secret_mse)
    
def main():
    complete_model = tf.keras.models.load_model(
        '../models/complete_model.h5', 
        custom_objects={
            'PrepLayer': PrepLayer, 
            'HideLayer': HideLayer, 
            'RevealLayer': RevealLayer, 
            'StenographyLoss': StenographyLoss
        }
    )

    complete_model.summary()

    filename1 = file_selector()    
    print(filename1)
    secret = Image.open(filename1).convert('RGB')

    col1, col2, col3 = st.columns(3)

    with col1:
        st.write(' ')

    with col2:
        st.image(secret)

    with col3:
        st.write(' ')
    filename2 = file_selector2()    
    print(filename2)
    cover = Image.open(filename2).convert('RGB')
    #cover
    col1, col2, col3 = st.columns(3)

    with col1:
        st.write(' ')

    with col2:
        st.image(cover)

    with col3:
        st.write(' ')

    # cover = Image.open('/Users/ayushpathak/Downloads/1.jpg').convert('RGB')
    # cover

    revealed, cover_with_secret = complete_model.predict(
        [np.expand_dims(np.array(secret), axis=0), np.expand_dims(np.array(cover), axis=0)]
    )

    print("AFTER")
    #Image.fromarray(np.array(cover_with_secret[0], dtype=np.uint8), 'RGB')
    sec1 = Image.fromarray(np.array(revealed[0], dtype=np.uint8), 'RGB')
    encoded1 = Image.fromarray(np.array(cover_with_secret[0], dtype=np.uint8), 'RGB')
    st.markdown("<h2 style='text-align: center; color: white;'>The encrypted image </h2>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.write(' ')

    with col2:
        st.image(Image.fromarray(np.array(cover_with_secret[0], dtype=np.uint8), 'RGB'))

    with col3:
        st.write(' ')
   

    #st.image(Image.fromarray(np.array(cover_with_secret[0], dtype=np.uint8), 'RGB'))

    st.markdown("<h2 style='text-align: center; color: white;'>The secret image </h2>", unsafe_allow_html=True)

    #Image.fromarray(np.array(revealed[0], dtype=np.uint8), 'RGB')

    col1, col2, col3 = st.columns(3)

    with col1:
        st.write(' ')

    with col2:
        st.image(Image.fromarray(np.array(revealed[0], dtype=np.uint8), 'RGB'))

    with col3:
        st.write(' ')

    #st.image(Image.fromarray(np.array(revealed[0], dtype=np.uint8), 'RGB'))

    psnr, mse, diff_S, diff_C = metrics.inputPSNR(cover, encoded1, secret, sec1)

    st.markdown(f"<h2 style='text-align: center; color: white;'>PSNR value is {psnr} dB</h2>", unsafe_allow_html=True)
    st.markdown(f"<h2 style='text-align: center; color: white;'>MSE value is {mse}</h2>", unsafe_allow_html=True)

    # plt.imshow(rgb2gray(diff_C), cmap = plt.get_cmap('gray'))
    # plt.imshow(rgb2gray(diff_S), cmap = plt.get_cmap('gray'))


    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(rgb2gray(diff_C), cmap = plt.get_cmap('gray'))
    plt.title('diff_C')

    plt.subplot(1, 2, 2)
    plt.imshow(rgb2gray(diff_S), cmap = plt.get_cmap('gray'))
    plt.title('diff_S')

    # Replace plt.show() with st.pyplot()
    st.pyplot(plt.gcf())

    # st.image(Image.fromarray(np.array(diff_S, dtype=np.uint8), 'RGB'))
    # st.image(Image.fromarray(np.array(diff_C, dtype=np.uint8), 'RGB'))




if __name__ == "__main__":
    main()