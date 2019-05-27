# qtune: Automated fine tuning and optimization
#
#   Copyright (C) 2019  Jonas Dedden and Simon S. Humpohl
#
#   This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation version 3 of the License.
#
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
################################################################################

# @email: julian.teske@rwth-aachen.de

import numpy as np
import scipy

from filterpy.kalman import KalmanFilter

from qtune.storage import HDF5Serializable


class KalmanGradient(metaclass=HDF5Serializable):
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
    def __init__(self, n_pos_dim: int, n_values: int, *,
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
        n_pos_dim : int
            Number of Gates in your system.
        
        n_values : int
            Number of Parameters in your system.
        
        state_transition_function : 2D np.array with dimensions = (n_values*n_pos_dim, n_values*n_pos_dim), optional
            Optional state transition function. If you know your system has a
            certain determinable  dynamic, you can specify it here to greatly
            increase filter performance. When no function is given, a diagonal
            unit matrix is used (that means the filter expects no dynamics).
        
        initial_gradient : list or 1D np.array with dimension = n_pos_dim*n_values, optional
            Optional initial state vector (gradient matrix). It can be helpful
            to set this to the first measurement to increase convergence speed.
            If no initial state vector is given, a null vector is used.

        initial_covariance_matrix : 2D np.array with dimensions = (n_values*n_pos_dim, n_values*n_pos_dim), optional
            Optional initial covariance matrix.
            If no inital covariance matrix is given, a diagonal unit matrix is
            used.
        
        measurement_covariance_matrix : 2D np.array with dimensions = (n_values, n_values), optional
            Optional measurement noise/covariance matrix for the measurement of
            the parameters. You probably should specify it. If no measurement 
            covariance matrix is given, a diagonal unit matrix is used.
            
        process_noise : 2D np.array with dimensions = (n_values*n_pos_dim, n_values*n_pos_dim), optional
            Optional process noise matrix.
            If no process noise matrix is given, null matrix is used.
            
        alpha : float, optional
            Optional float value that is used for a fading memory effect. This
            value states how much the covariance increases in each prediction
            step because older measurements lose statistical significance 
            with time. You probably want it somewhere around 1.00-1.02.
            If no alpha is given, 1.00 is used (no fading memory).
        """
        
        self.n_pos_dim = int(n_pos_dim)
        self.n_values = int(n_values)
        
        # the dimension of our state vector is equal 
        # to the product of the number of gates and the number of parameters
        dim_x = self.n_pos_dim*self.n_values
        
        # creating the KalmanFilter object
        self.filter = KalmanFilter(dim_x, self.n_values)
        
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
            self.filter.R = np.array(measurement_covariance_matrix).reshape(self.n_values, self.n_values)
        if process_noise is None:
            self.filter.Q = np.zeros((dim_x, dim_x))
        else:
            self.filter.Q = np.array(process_noise).reshape(dim_x, dim_x)
        
        # set alpha value (needed for fading memory filtering)
        self.filter.alpha = alpha

    def to_hdf5(self):
        return dict(n_pos_dim=self.n_pos_dim,
                    n_values=self.n_values,
                    state_transition_function=self.filter.F,
                    initial_gradient=self.grad,
                    initial_covariance_matrix=self.cov,
                    measurement_covariance_matrix=self.filter.R,
                    process_noise=self.filter.Q,
                    alpha=self.filter.alpha)

    def update(self, diff_position, diff_values,
               measurement_covariance=None,
               process_covariance=None,
               predict=True):
        """
        Updates the current gradient matrix with new measurements.
        
        Parameters
        ----------

        diff_position : list or 1D np.array with dimension = n_pos_dim
            Vector of the voltage differences that were used to create the new
            measurement. 

        diff_values : list or 1D np.array with dimension = n_values
            Vector of the measured parameter differences.

        measurement_covariance : 2D np.array with dimensions = (n_values, n_values), optional
            Optional matrix of measurement covariance. If it is given, it will
            be used only for this update. Else self.filter.R is used.

        process_covariance : 2D np.array with dimensions = (n_values*n_pos_dim, n_values*n_pos_dim), optional
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
        
        diff_values = np.array(diff_values).reshape(self.n_values, 1)
        diff_position = np.array(diff_position).reshape(self.n_pos_dim)
        
        if predict:
            self.filter.predict(Q=process_covariance)

        self.filter.update(diff_values, R=measurement_covariance, H=self.__createMatrixH(diff_position))
    
    def __createMatrixH(self, diff_position):
        """
        Creates the necessary H matrix (the so called measurement function)
        that is tailored to the voltage differences that were used in a 
        particular measurement.
        
        Parameters
        ----------

        diff_position : list or 1D np.array with dimension = n_pos_dim
            Vector of the voltage differences that were used to create the new
            measurement. 
        
            
        Returns
        ------
        
        H : 2D np.array with dimensions = (n_values,n_pos_dim*n_values)
            The measurement function that will be used in an update step.
        """

        diff_position = np.asarray(diff_position)

        if diff_position.shape != (self.n_pos_dim, ):
            raise ValueError('Voltage differences have the wrong dimension')

        return scipy.linalg.block_diag(*[diff_position]*self.n_values)
    
    @property
    def grad(self):
        """
        Returns the current gradient matrix.
        
        Returns
        -------

        grad : 2D np.array with dimension = (n_values, n_pos_dim)
            The updated gradient matrix.
        """
        
        return self.filter.x.reshape(self.n_values, self.n_pos_dim)
    
    @property
    def cov(self):
        """
        Returns the current covariance matrix of the gradient matrix.
        
        Returns
        -------

        cov : 2D np.array with dimensions = (n_values*n_pos_dim, n_values*n_pos_dim)
            The covariance matrix of the gradient matrix.
        """
        
        return self.filter.P
    
    @property
    def sugg_diff_position(self):
        """
        Returns a suggestion for the next vector of voltage differences in the
        next measurement.
        
        Returns
        -------

        cov : 1D np.array with dimension = n_pos_dim
            The suggested vector of voltage differences for the next
            measurement.
        """
        
        # evaluate eigenvalues and eigenvector of the covariance matrix
        w, v = np.linalg.eigh(self.filter.P)
        
        # split the eigenvector in parts (one part contains the actual vector,
        # all else are just null vectors). They will have the length of n_pos_dim
        # after that. The eigenvector with the biggest eigenvalue is used,
        # because that is the vector with the greatest uncertainty                               
        z = np.split(v[:, np.argmax(w)], self.n_values)
        
        # just use the one vector which has a non trivial norm
        return next(i for i in z if np.linalg.norm(i) != 0)
