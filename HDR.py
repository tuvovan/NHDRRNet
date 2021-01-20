import tensorflow as tf

from tensorflow.keras import Input, Model
from tensorflow.keras.activations import sigmoid, tanh
from tensorflow_addons.layers import AdaptiveAveragePooling2D
from tensorflow.keras.layers import Conv2D, LeakyReLU, ReLU, BatchNormalization, Conv2DTranspose, Reshape, Add, UpSampling2D 


class NHDRRNet(Model):
    def __init__(self, config):
        super(NHDRRNet, self).__init__()

        self.filter = config.filter
        self.encoder_kernel = config.encoder_kernel
        self.decoder_kernel = config.decoder_kernel
        self.triple_pass_filter = config.triple_pass_filter

    def adaptive_interpolation(self, required_size, img):
        pool_size = (int(required_size[0]/img.shape[1]), int(required_size[1]/img.shape[2]))
        return UpSampling2D(size=pool_size)(img) 

    def encoder_1(self, X, i):
        X = Conv2D(int(self.filter*i), self.encoder_kernel, strides=(2,2), padding='same')(X)
        X = BatchNormalization()(X)
        X = ReLU()(X)
        return X

    def encoder_2(self, X, i):
        X = Conv2D(int(self.filter*i), self.encoder_kernel, strides=(2,2), padding='same')(X)
        X = BatchNormalization()(X)
        X = ReLU()(X)
        return X

    def encoder_3(self, X, i):
        X = Conv2D(int(self.filter*i), self.encoder_kernel, strides=(2,2), padding='same')(X)
        X = BatchNormalization()(X)
        X = ReLU()(X)
        return X

    def decoder_last(self, X):
        X = Conv2DTranspose(3, self.decoder_kernel, strides=(2,2), padding='same')(X)
        X = BatchNormalization()(X)
        X = ReLU()(X)
        return X

    def decoder(self, X, i):
        X = Conv2DTranspose(int(self.filter*i), self.decoder_kernel, strides=(2,2), padding='same')(X)
        X = BatchNormalization()(X)
        X = LeakyReLU()(X)
        return X

    def triplepass(self, T0):
        T1 = Conv2D(self.triple_pass_filter, kernel_size=(1,1), strides=(1,1), padding='same')(T0)
        T1 = ReLU()(T1)

        T2 = Conv2D(self.triple_pass_filter, kernel_size=(3,3), strides=(1,1), padding='same')(T0)
        T2 = ReLU()(T2)

        T3 = Conv2D(self.triple_pass_filter, kernel_size=(5,5), strides=(1,1), padding='same')(T0)
        T3 = ReLU()(T3)

        T3 = Add()([T1, T2, T3])

        T4 = Conv2D(self.triple_pass_filter, kernel_size=(3,3), strides=(1,1), padding='same')(T3)
        T5 = Add()([T4, T0])

        return T5

    def global_non_local(self, X):
        h, w , c = list(X.shape)[1], list(X.shape)[2], list(X.shape)[3]
        theta = Conv2D(128, kernel_size=(1,1), padding='same')(X)
        theta_rsh = Reshape((h*w, 128))(theta)

        phi = Conv2D(128, kernel_size=(1,1), padding='same')(X)
        phi_rsh = Reshape((128, h*w))(phi)

        g = Conv2D(128, kernel_size=(1,1), padding='same')(X)
        g_rsh = Reshape((h*w, 128))(g)

        theta_phi = tf.matmul(theta_rsh, phi_rsh)
        theta_phi = tf.keras.layers.Softmax()(theta_phi)

        theta_phi_g = tf.matmul(theta_phi, g_rsh)
        theta_phi_g = Reshape((h, w, 128))(theta_phi_g)

        theta_phi_g = Conv2D(256, kernel_size=(1,1), padding='same')(theta_phi_g)

        out = Add()([theta_phi_g, X])

        return out

    def main_model(self, X):
        ## attention network
        X_i = X[:,0,:,:,:]
        X_r = X[:,1,:,:,:]
        X_j = X[:,2,:,:,:]

        X_i_32 = self.encoder_1(X_i, 1)
        X_i_64 = self.encoder_1(X_i_32, 2)
        X_i_128 = self.encoder_1(X_i_64, 4)
        X_i_256 = self.encoder_1(X_i_128, 8)

        X_r_32 = self.encoder_2(X_r, 1)
        X_r_64 = self.encoder_2(X_r_32, 2)
        X_r_128 = self.encoder_2(X_r_64, 4)
        X_r_256 = self.encoder_2(X_r_128, 8)

        X_j_32 = self.encoder_3(X_j, 1)
        X_j_64 = self.encoder_3(X_j_32, 2)
        X_j_128 = self.encoder_3(X_j_64, 4)
        X_j_256 = self.encoder_3(X_j_128, 8)

        encoder_cat = tf.concat([X_j_256, X_r_256, X_i_256], axis=-1)
        encoder_last = Conv2D(256, kernel_size=(1,1), padding='same')(encoder_cat)
        encoder_last = BatchNormalization()(encoder_last)
        encoder_last = ReLU()(encoder_last)

        ## upper path ##
        tpl_out = self.triplepass(encoder_last)
        tpl_out = self.triplepass(tpl_out)
        tpl_out = self.triplepass(tpl_out)
        tpl_out = self.triplepass(tpl_out)
        tpl_out = self.triplepass(tpl_out)
        tpl_out = self.triplepass(tpl_out)
        tpl_out = self.triplepass(tpl_out)
        tpl_out = self.triplepass(tpl_out)
        tpl_out = self.triplepass(tpl_out)
        tpl_out = self.triplepass(tpl_out)

        ## lower path ##
        glb_out = AdaptiveAveragePooling2D(output_size=(16,16))(encoder_last)
        glb_out = self.global_non_local(glb_out)
        required_size = [encoder_last.shape[1], encoder_last.shape[2]]
        glb_out = self.adaptive_interpolation(required_size, glb_out)

        ## cat ##
        merger = tf.concat([tpl_out, glb_out], axis=-1)
        O_128 = self.decoder(merger, 4)
        O_128 = Add()([X_i_128, X_r_128, X_j_128, O_128])

        O_64 = self.decoder(O_128, 2)
        O_64 = Add()([X_i_64, X_r_64, X_j_64, O_64])
        O_32 = self.decoder(O_64, 1)
        O_32 = Add()([X_i_32, X_r_32, X_j_32, O_32])

        out = self.decoder_last(O_32)

        return out



