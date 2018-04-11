# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 15:38:16 2017

@author: dedden
"""

import numpy as np
import scipy

from filterpy.kalman import KalmanFilter


class KalmanGradient:
    """ 
    Implements a Kalman filter that can estimate a gradient in a recursive
    manner. 


    Attributes
    ----------
    x : numpy.array(dim_x, 1)
        State estimate vector

    P : numpy.array(dim_x, dim_x)
        Covariance matrix

    R : numpy.array(dim_z, dim_z)
        Measurement noise matrix

    Q : numpy.array(dim_x, dim_x)
        Process noise matrix

    F : numpy.array()
        State Transition matrix

    H : numpy.array(dim_x, dim_x)
        Measurement function

    """
    def __init__(self, n_gates: int, n_params: int, *,
                 state_transition_function=None,
                 initial_gradient=None,
                 initial_covariance_matrix=None,
                 measurement_covariance_matrix=None, process_noise=None, alpha=1):
        """
        Create the Kalman filter. You have to specify the number of gates and
        parameters. Additionally, there are some initial conditions and 
        important functions such as the state transition function that you can
        specify but are optional.

        Parameters
        ----------
        n_gates : int
            Number of Gates in your system.
        
        n_params : int
            Number of Parameters in your system.
        
        state_transition_function : 2D np.array with dimensions = (n_params*n_gates, n_params*n_gates), optional
            Optional state transition function. If you know your system has a
            certain determinable  dynamic, you can specify it here to greatly
            increase filter performance. When no function is given, a diagonal
            unit matrix is used (that means the filter expects no dynamics).
        
        initial_gradient : list or 1D np.array with dimension = n_gates*n_params, optional
            Optional initial state vector (gradient matrix). It can be helpful
            to set this to the first measurement to increase convergence speed.
            If no initial state vector is given, a null vector is used.

        initial_covariance_matrix : 2D np.array with dimensions = (n_params*n_gates, n_params*n_gates), optional
            Optional initial covariance matrix.
            If no inital covariance matrix is given, a diagonal unit matrix is
            used.
        
        measurement_covariance_matrix : 2D np.array with dimensions = (n_params, n_params), optional
            Optional measurement noise/covariance matrix for the measurement of
            the parameters. You probably should specify it. If no measurement 
            covariance matrix is given, a diagonal unit matrix is used.
            
        process_noise : 2D np.array with dimensions = (n_params*n_gates, n_params*n_gates), optional
            Optional process noise matrix.
            If no process noise matrix is given, null matrix is used.
            
        alpha : float, optional
            Optional float value that is used for a fading memory effect. This
            value states how much the covariance increases in each prediction
            step because older measurements lose statistical significance 
            with time. You probably want it somewhere around 1.00-1.02.
            If no alpha is given, 1.00 is used (no fading memory).
        """
        
        self.n_gates = int(n_gates)
        self.n_params = int(n_params)
        
        # the dimension of our state vector is equal 
        # to the product of the number of gates and the number of parameters
        dim_x = self.n_gates*self.n_params
        
        # creating the KalmanFilter object
        self.filter = KalmanFilter(dim_x, self.n_params)
        
        # if no state transition function is given, 
        # we assume our system has no dynamics
        if state_transition_function is None:
            self.filter.F = np.eye(dim_x)
        else:
            self.filter.F = np.array(state_transition_function).reshape(dim_x, dim_x)
            
        # set initial state vector and converts to a np.array if needed
        if initial_gradient is not None:
            self.filter.x = np.array(initial_gradient).reshape(dim_x, 1)
        
        # set covariance matrix if needed
        if initial_covariance_matrix is not None:
            self.filter.P = np.array(initial_covariance_matrix).reshape(dim_x, dim_x)
            
        # set measurement and process covariance matrix
        if measurement_covariance_matrix is not None:
            self.filter.R = np.array(measurement_covariance_matrix).reshape(self.n_params, self.n_params)
        if process_noise is None:
            self.filter.Q = np.zeros((dim_x, dim_x))
        else:
            self.filter.Q = np.array(process_noise).reshape(dim_x, dim_x)
        
        # set alpha value (needed for fading memory filtering)
        self.filter.alpha = alpha

    def update(self, diff_volts, diff_params,
               measurement_covariance=None,
               process_covariance=None,
               predict=True):
        """
        Updates the current gradient matrix with new measurements.
        
        Parameters
        ----------

        diff_volts : list or 1D np.array with dimension = n_gates
            Vector of the voltage differences that were used to create the new
            measurement. 

        diff_params : list or 1D np.array with dimension = n_params
            Vector of the measured parameter differences.

        measurement_covariance : 2D np.array with dimensions = (n_params, n_params), optional
            Optional matrix of measurement covariance. If it is given, it will
            be used only for this update. Else self.filter.R is used.

        process_covariance : 2D np.array with dimensions = (n_params*n_gates, n_params*n_gates), optional
            Optional matrix of process covariance. If it is given, it will
            be used only for this update. Else self.filter.Q is used.
            Since it is used only in the prediction step, if predict is set
            to False, setting the Q-matrix will have no effect.

        predict : boolean, optional
            Boolean that states if the prediction step should be calculated.
            If the state transition function self.filter.F is a diagonal unit 
            matrix, that means the system has no expected dynamics, this will 
            only have an effect if the set alpha value is not equal to 1.
            If the alpha value is not equal to 1 we apply a 'fading memory' -
            effect. That means that older measurements lose significance
            and therefore the covariance increases by a given ratio each
            prediction step.
            If both the transition function is a diagonal unit matrix and
            alpha is set to 1, the prediciton step will have no effect at
            all and can be omitted (that means the predict value can be
            set to False).
        """
        
        diff_params = np.array(diff_params).reshape(self.n_params, 1)
        diff_volts = np.array(diff_volts).reshape(self.n_gates)
        
        if predict:
            self.filter.predict(Q=process_covariance)

        self.filter.update(diff_params, R=measurement_covariance, H=self.__createMatrixH(diff_volts))
    
    def __createMatrixH(self, diff_volts):
        """
        Creates the necessary H matrix (the so called measurement function)
        that is tailored to the voltage differences that were used in a 
        particular measurement.
        
        Parameters
        ----------

        diff_volts : list or 1D np.array with dimension = n_gates
            Vector of the voltage differences that were used to create the new
            measurement. 
        
            
        Returns
        ------
        
        H : 2D np.array with dimensions = (n_params,n_gates*n_params)
            The measurement function that will be used in an update step.
        """

        diff_volts = np.asarray(diff_volts)

        if diff_volts.shape != (self.n_gates, ):
            raise ValueError('Voltage differences have the wrong dimension')

        return scipy.linalg.block_diag(*[diff_volts]*self.n_params)
    
    @property
    def grad(self):
        """
        Returns the current gradient matrix.
        
        Returns
        -------

        grad : 2D np.array with dimension = (n_params, n_gates)
            The updated gradient matrix.
        """
        
        return self.filter.x.reshape(self.n_params, self.n_gates)
    
    @property
    def cov(self):
        """
        Returns the current covariance matrix of the gradient matrix.
        
        Returns
        -------

        cov : 2D np.array with dimensions = (n_params*n_gates, n_params*n_gates)
            The covariance matrix of the gradient matrix.
        """
        
        return self.filter.P
    
    @property
    def sugg_diff_volts(self):
        """
        Returns a suggestion for the next vector of voltage differences in the
        next measurement.
        
        Returns
        -------

        cov : 1D np.array with dimension = n_gates
            The suggested vector of voltage differences for the next
            measurement.
        """
        
        # evaluate eigenvalues and eigenvector of the covariance matrix
        w, v = np.linalg.eigh(self.filter.P)
        
        # split the eigenvector in parts (one part contains the actual vector,
        # all else are just null vectors). They will have the length of n_gates
        # after that. The eigenvector with the biggest eigenvalue is used,
        # because that is the vector with the greatest uncertainty                               
        z = np.split(v[:, np.argmax(w)], self.n_params)
        
        # just use the one vector which has a non trivial norm
        return next(i for i in z if np.linalg.norm(i) != 0)
