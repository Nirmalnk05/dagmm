import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler
#from sklearn.externals import joblib
import joblib
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from dagmm.compression_net import CompressionNet
from dagmm.estimation_net import EstimationNet
from dagmm.gmm import GMM

from os import makedirs
from os.path import exists, join


    
class DAGMM(tf.Module):
    """ Deep Autoencoding Gaussian Mixture Model.

    This implementation is based on the paper:
    Bo Zong+ (2018) Deep Autoencoding Gaussian Mixture Model
    for Unsupervised Anomaly Detection, ICLR 2018
    (this is UNOFFICIAL implementation)
    """

    MODEL_FILENAME = "DAGMM_model"
    SCALER_FILENAME = "DAGMM_scaler"

    def __init__(self, comp_hiddens,comp_activation,est_hiddens,est_activation,
                 est_dropout_ratio=0.1,
            minibatch_size=1024, epoch_size=30,
            learning_rate=0.0001, lambda1=0.1, lambda2=0.0001,
            normalize=False, random_seed=123):
        """
        Parameters
        ----------
        comp_hiddens : list of int
            sizes of hidden layers of compression network
            For example, if the sizes are [n1, n2],
            structure of compression network is:
            input_size -> n1 -> n2 -> n1 -> input_sizes
        comp_activation : function
            activation function of compression network
        est_hiddens : list of int
            sizes of hidden layers of estimation network.
            The last element of this list is assigned as n_comp.
            For example, if the sizes are [n1, n2],
            structure of estimation network is:
            input_size -> n1 -> n2 (= n_comp)
        est_activation : function
            activation function of estimation network
        est_dropout_ratio : float (optional)
            dropout ratio of estimation network applied during training
            if 0 or None, dropout is not applied.
        minibatch_size: int (optional)
            mini batch size during training
        epoch_size : int (optional)
            epoch size during training
        learning_rate : float (optional)
            learning rate during training
        lambda1 : float (optional)
            a parameter of loss function (for energy term)
        lambda2 : float (optional)
            a parameter of loss function
            (for sum of diagonal elements of covariance)
        normalize : bool (optional)
            specify whether input data need to be normalized.
            by default, input data is normalized.
        random_seed : int (optional)
            random seed used when fit() is called.
        """
        super(DAGMM, self).__init__()
        self.comp_net = CompressionNet(comp_hiddens, comp_activation)
        self.est_net = EstimationNet(est_hiddens, est_activation)
        n_comp = est_hiddens[-1]
        self.gmm = GMM(n_comp)
        

        self.est_dropout_ratio = est_dropout_ratio
        self.minibatch_size = minibatch_size
        self.epoch_size = epoch_size
        self.learning_rate = learning_rate
        self.lambda1 = lambda1
        self.lambda2 = lambda2

        self.normalize = normalize
        self.scaler = None
        self.seed = random_seed
    
    
    def fit(self, x):
        """ Fit the DAGMM model according to the given data.

        Parameters
        ----------
        x : array-like, shape (n_samples, n_features)
            Training data.
        """
        n_samples, n_features = x.shape

        if self.normalize:
            self.scaler = StandardScaler()
            x = self.scaler.fit_transform(x)
        tf.random.set_seed(self.seed)
        np.random.seed(seed=self.seed)

        self.model_input = tf.Variable(x, dtype=tf.float32)

        # self.model_input = input = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, n_features])
        # self.drop = drop = tf.compat.v1.placeholder(dtype=tf.float32, shape=[])
        # self.drop = tf.Variable(0.0, dtype=tf.float32)

        # Number of batch
        n_batch = (n_samples - 1) // self.minibatch_size + 1

        # Create tensorflow session and initilize
        #init = tf.global_variables_initializer()

        # Training
        idx = np.arange(x.shape[0])
        np.random.shuffle(idx)

        for epoch in range(self.epoch_size):
            for batch in range(n_batch):
                i_start = batch * self.minibatch_size
                i_end = (batch + 1) * self.minibatch_size
                x_batch = x[idx[i_start:i_end]]

                with tf.GradientTape() as tape:
                    #z, x_dash = self.comp_net.inference(x_batch)
                    z, x_dash = self.comp_net.call(x_batch)
                    #print('ch',np.isnan(x_batch[0]).sum())

                    #gamma = self.est_net.inference(z,self.est_dropout_ratio)
                    gamma = self.est_net.call(z,self.est_dropout_ratio)

                    self.gmm.fit(z, gamma)
                    energy = self.gmm.energy(z)
                
                    # Loss function
                    loss = (self.comp_net.reconstruction_error(x_batch, x_dash) +
                        self.lambda1 * tf.reduce_mean(energy) +
                        self.lambda2 * self.gmm.cov_diag_loss())
                
                gradients = tape.gradient(loss,  self.comp_net.trainable_variables +
                                          self.est_net.trainable_variables +
                                          self.gmm.trainable_variables)
                optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
                optimizer.apply_gradients(zip(gradients, self.comp_net.trainable_variables +
                                              self.est_net.trainable_variables +
                                              self.gmm.trainable_variables))
                #model.train_on_batch(x_batch, None)  # Pass x_batch as input to train_on_batch

            #if (epoch + 1) % 100 == 0:
            #loss_val = self.sess.run(loss, feed_dict={input:x, drop:0})
            loss_val = loss.numpy() #model.evaluate(x, None)  # Evaluate the loss on the entire dataset
            print(" epoch {}/{} : loss = {:.3f}".format(epoch + 1, self.epoch_size, loss_val))

        # Fix GMM parameter
        fix_op = self.gmm.fix_op()
        #print(len(gmm_params),gmm_params[0:])
        fix_op()

        #self.sess.run(fix, feed_dict={input:x, drop:0})
        self.energy = self.gmm.energy(z)
        # Create a path for the new folder in the current directory
        folder_path = os.path.join(os.getcwd(), 'models')

        # Create a new folder named 'results' in the current directory
        os.makedirs(folder_path, exist_ok=True)
        tf.saved_model.save(self, folder_path)  # Save the model using SavedModel format

        # tf.add_to_collection("save", self.model_input)
        # tf.add_to_collection("save", self.energy)

        # self.saver = tf.train.Saver()

    def predict(self, x):
        """ Calculate anormaly scores (sample energy) on samples in X.

        Parameters
        ----------
        x : array-like, shape (n_samples, n_features)
            Data for which anomaly scores are calculated.
            n_features must be equal to n_features of the fitted data.

        Returns
        -------
        energies : array-like, shape (n_samples)
            Calculated sample energies.
        """
        
        if self.normalize:
            x = self.scaler.transform(x)

        #self.comp_net.call(tf.convert_to_tensor(x, dtype=tf.float32))[0]
        print(self.comp_net.call(tf.convert_to_tensor(x, dtype=tf.float32))[0])
        #print('s',tf.reduce_any(tf.math.is_nan(self.gmm.energy(self.comp_net.call(tf.convert_to_tensor(x, dtype=tf.float32))[0]))))

        # Check for NaN values in input data
        # nan_indices = np.isnan(x)
        # if np.any(nan_indices):
        #     # Handle NaN values: for example, replace with zeros
        #     x[nan_indices] = 0
        #print(tf.Variable(x, dtype=tf.float32)) 
        tf.debugging.check_numerics(tf.convert_to_tensor(x, dtype=tf.float32), "Energy has NaN or Inf values!")
        #print(type(x),type(tf.convert_to_tensor(x, dtype=tf.float32)),type(tf.Variable(x, dtype=tf.float32)))
        #print(np.isnan(x[0]).sum(),tf.convert_to_tensor(x, dtype=tf.float32)[0], tf.reduce_any(tf.math.is_nan(tf.convert_to_tensor(x, dtype=tf.float32)[0])))
        energies = self.gmm.energy(self.comp_net.call(tf.convert_to_tensor(x, dtype=tf.float32))[0])
        #energies = self.gmm.energy(tf.convert_to_tensor(x[0], dtype=tf.float32))

        #energies = self.sess.run(self.energy, feed_dict={self.model_input:x})
        return energies

    def save(self, fdir):
        """ Save trained model to designated directory.
        This method have to be called after training.
        (If not, throw an exception)

        Parameters
        ----------
        fdir : str
            Path of directory trained model is saved.
            If not exists, it is created automatically.
        """

        if not exists(fdir):
            makedirs(fdir)

        #model_path = join(fdir, self.MODEL_FILENAME)

        if self.normalize:
            scaler_path = join(fdir, self.SCALER_FILENAME)
            joblib.dump(self.scaler, scaler_path)

    def restore(self, fdir):
        """ Restore trained model from designated directory.

        Parameters
        ----------
        fdir : str
            Path of directory trained model is saved.
        """
        if not exists(fdir):
            raise Exception("Model directory does not exist.")

        model_path = join(fdir, self.MODEL_FILENAME)
        #meta_path = model_path + ".meta"

        loaded_model = tf.keras.models.load_model(model_path)
        #self.model_input, self.energy = tf.get_collection("save")

        if self.normalize:
            scaler_path = join(fdir, self.SCALER_FILENAME)
            self.scaler = joblib.load(scaler_path)
