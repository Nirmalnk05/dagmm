# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

class GMM(tf.keras.layers.Layer):
    """ Gaussian Mixture Model (GMM) """
    def __init__(self, n_comp):
        super(GMM, self).__init__()

        self.n_comp = n_comp
        self.phi = self.mu = self.sigma = None
        self.training = False

    def create_variables(self, n_features):

        with tf.name_scope("GMM"):
            phi = tf.Variable(tf.zeros(shape=[self.n_comp]),
                dtype=tf.float32, name="phi")
            mu = tf.Variable(tf.zeros(shape=[self.n_comp, n_features]),
                dtype=tf.float32, name="mu")
            sigma = tf.Variable(tf.zeros(
                shape=[self.n_comp, n_features, n_features]),
                dtype=tf.float32, name="sigma")
            L = tf.Variable(tf.zeros(
                shape=[self.n_comp, n_features, n_features]),
                dtype=tf.float32, name="L")

        return phi, mu, sigma, L

    def fit(self, z, gamma):
        """ fit data to GMM model

        Parameters
        ----------
        z : tf.Tensor, shape (n_samples, n_features)
            data fitted to GMM.
        gamma : tf.Tensor, shape (n_samples, n_comp)
            probability. each row is correspond to row of z.
        """

        with tf.name_scope("GMM"):
            # Calculate mu, sigma
            # i   : index of samples
            # k   : index of components
            # l,m : index of features
            gamma_sum = tf.reduce_sum(gamma, axis=0)
            self.phi = phi = tf.reduce_mean(gamma, axis=0)
            self.mu = mu = tf.einsum('ik,il->kl', gamma, z) / gamma_sum[:,None]
            z_centered = tf.sqrt(gamma[:,:,None]) * (z[:,None,:] - mu[None,:,:])
            self.sigma = sigma = tf.einsum(
                'ikl,ikm->klm', z_centered, z_centered) / gamma_sum[:,None,None]

            # Calculate a cholesky decomposition of covariance in advance
            n_features = z.shape[1]
            min_vals = tf.linalg.diag(tf.ones(n_features, dtype=tf.float32)) * 1e-6
            #self.L = tf.linalg.cholesky(sigma + tf.expand_dims(min_vals, axis=0)) #min_vals[None,:,:])
            self.L = tf.linalg.cholesky(sigma + min_vals[None, :, :] + 1e-6 * tf.eye(n_features))

        self.training = False

    def fix_op(self):
        """ return operator to fix paramters of GMM
        Using this operator outside of this class,
        you can fix current parameter to static tensor variable.

        After you call this method, you have to run result
        operator immediatelly, and call energy() to use static
        variables of model parameter.

        Returns
        -------
        op : operator of tensorflow
            operator to assign current parameter to variables
        """
        phi, mu, sigma, L = self.create_variables(self.mu.shape[1])

        self.phi, self.phi_org = phi, self.phi
        self.mu, self.mu_org = mu, self.mu
        self.sigma, self.sigma_org = sigma, self.sigma
        self.L, self.L_org = L, self.L
        self.training = False
        # return self.phi, self.mu, self.sigma, self.L
        #print('fix',self.phi_org,self.mu_org,self.sigma_org,self.L_org)
        # Create a callable graph
        @tf.function
        def assign_op():
            phi.assign(self.phi_org)
            mu.assign(self.mu_org)
            sigma.assign(self.sigma_org)
            L.assign(self.L_org)

        return assign_op

    def energy(self, z):
        """ calculate an energy of each row of z

        Parameters
        ----------
        z : tf.Tensor, shape (n_samples, n_features)
            data each row of which is calculated its energy.

        Returns
        -------
        energy : tf.Tensor, shape (n_samples)
            calculated energies
        """
        
        if self.training and self.phi is None:
            self.phi, self.mu, self.sigma, self.L = self.create_variables(z.shape[1])
        #print('tst',self.phi, self.mu, self.sigma, self.L)

        with tf.name_scope("GMM_energy"):
            # Instead of inverse covariance matrix, exploit cholesky decomposition
            # for stability of calculation.
            z_centered = z[:,None,:] - self.mu[None,:,:]  #ikl
            #print('z_c',z_centered)
            #print('ts',tf.transpose(z_centered, [1, 2, 0]))
            #print('l',self.L)
            v = tf.linalg.triangular_solve(self.L, tf.transpose(z_centered, [1, 2, 0]))  # kli
            #print('v',v)
            # log(det(Sigma)) = 2 * sum[log(diag(L))]
            log_det_sigma = 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(self.L)), axis=1)

            # To calculate energies, use "log-sum-exp" (different from orginal paper)
            d = z.get_shape().as_list()[1]
            #print('tst',tf.math.log(self.phi[:,None]) , (tf.reduce_sum(tf.square(v), axis=1)))
            #print('t2',tf.math.log(2.0 * np.pi) + log_det_sigma[:,None])
            logits = tf.math.log(self.phi[:,None]) - 0.5 * (tf.reduce_sum(tf.square(v), axis=1)
                + d * tf.math.log(2.0 * np.pi) + log_det_sigma[:,None])
           # print('log',logits)
            energies = - tf.reduce_logsumexp(logits, axis=0)

        return energies

    def cov_diag_loss(self):
        with tf.name_scope("GMM_diag_loss"):
            diag_loss = tf.reduce_sum(tf.divide(1, tf.linalg.diag_part(self.sigma)))

        return diag_loss
