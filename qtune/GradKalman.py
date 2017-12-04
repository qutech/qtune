# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 15:38:16 2017

@author: dedden
"""

import numpy as np
from filterpy.kalman import KalmanFilter

class GradKalmanFilter():
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
    def __init__(self, nGates, nParams, initF = None, initX = None, initP = None, initR = None, initQ = None, alpha = 1):
        """
        Create the Kalman filter. You have to specify the number of gates and
        parameters. Additionally, there are some initial conditions and 
        important functions such as the state transition function that you can
        specify but are optional.

        Parameters
        ----------
        nGates : int
            Number of Gates in your system.
        
        nParams : int
            Number of Parameters in your system.
        
        initF : 2D np.array with dimensions = (nParams*nGates, nParams*nGates), optional
            Optional state transition function. If you know your system has a
            certain determinable  dynamic, you can specify it here to greatly
            increase filter performance. When no function is given, a diagonal
            unit matrix is used (that means the filter expects no dynamic).
        
        initX : list or 1D np.array with dimension = nGates*nParams, optional
            Optional initial state vector (gradient matrix). It can be helpful
            to set this to the first measurement to increase convergence speed.
            If no initial state vector is given, a null vector is used.

        initP : 2D np.array with dimensions = (nParams*nGates, nParams*nGates), optional
            Optional initial covariance matrix.
            If no inital covariance matrix is given, a diagonal unit matrix is
            used.
        
        initR : 2D np.array with dimensions = (nParams, nParams), optional
            Optional measurement noise/covariance matrix for the measurement of
            the parameters. You probably should specify it. If no measurement 
            covariance matrix is given, a diagonal unit matrix is used.
            
        initQ : 2D np.array with dimensions = (nParams*nGates, nParams*nGates), optional
            Optional process noise matrix.
            If no process noise matrix is given, null matrix is used.
            
        alpha : float, optional
            Optional float value that is used for a fading memory effect. This
            value states how much the covariance increases in each prediction
            step because older measurements lose statistical significance 
            with time. You probably want it somewhere around 1.00-1.02.
            If no alpha is given, 1.00 is used (no fading memory).
        """
        
        self.nGates = int(nGates)
        self.nParams = int(nParams)
        
        # the dimension of our state vector is equal 
        # to the product of the number of gates and the number of parameters
        dim_x = self.nGates*self.nParams
        
        # creating the KalmanFilter object
        self.filter = KalmanFilter(dim_x, self.nParams)
        
        # if no state transition function is given, 
        # we assume our system has no dynamics
        if initF is None:
            self.filter.F = np.eye(dim_x)
        else:
            self.filter.F = np.array(initF).reshape(dim_x, dim_x)
            
        # set initial state vector and converts to a np.array if needed
        if not initX is None:
            self.filter.x = np.array(initX).reshape(dim_x, 1)
        
        # set covariance matrix if needed
        if not initP is None:
            self.filter.P = np.array(initP).reshape(dim_x, dim_x)
            
        # set measurement and process covariance matrix
        if not initR is None:
            self.filter.R = np.array(initR).reshape(self.nParams, self.nParams)
        if initQ is None:
            self.filter.Q = np.zeros((dim_x, dim_x))
        else:
            self.filter.Q = np.array(initQ).reshape(dim_x, dim_x)
        
        # set alpha value (needed for fading memory filtering)
        self.filter.alpha = alpha

    def update(self, dU, dT, R = None, Q = None, predict = True, hack = True, threshold = 30, factor = 300):
        """
        Updates the current gradient matrix with new measurements.
        
        Parameters
        ----------

        dU : list or 1D np.array with dimension = nGates
            Vector of the voltage differences that were used to create the new
            measurement. 

        dT : list or 1D np.array with dimension = nParams
            Vector of the measured parameter differences.

        R : 2D np.array with dimensions = (nParams, nParams), optional
            Optional matrix of measurement covariance. If it is given, it will
            be used only for this update. Else self.filter.R is used.

        Q : 2D np.array with dimensions = (nParams*nGates, nParams*nGates), optional
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
            prediciton step.
            If both the transition function is a diagonal unit matrix and
            alpha is set to 1, the prediciton step will have no effect at
            all and can be omitted (that means the predict value can be
            set to False).
        """
        
        dT = np.array(dT).reshape(self.nParams, 1)
        dU = np.array(dU).reshape(self.nGates)
        
        if predict:
            self.filter.predict(Q = Q)

        self.filter.update(dT, R = R, H = self.__createMatrixH(dU))
        
        """
        Following code is a hack that greatly increases the covariance
        if a possible jump has been detected.
        Detection occurs when the difference between estimation and the new
        measurement is greater then threshold*estimated error. The estimated
        error is already caluculated in filter.S. It is a projection of the
        state vector into measurement space using the current gate voltages.
        """
        
        if hack and max(abs(self.filter.y[:,0]/self.filter.S.diagonal())) > threshold:
            self.filter.P *= factor
    
    def __createMatrixH(self, dU):
        """
        Creates the necessary H matrix (the so called measurement function)
        that is tailored to the voltage differences that were used in a 
        particular measurement.
        
        Parameters
        ----------

        dU : list or 1D np.array with dimension = nGates
            Vector of the voltage differences that were used to create the new
            measurement. 
        
            
        Returns
        ------
        
        H : 2D np.array with dimensions = (nParams,nGates*nParams)
            The measurement function that will be used in an update step.
        """
        
        return np.array([[dU[j % self.nGates] if i*self.nGates<= j < (i+1)*self.nGates else 0 for j in range(self.nGates*self.nParams)] for i in range(self.nParams)])
    
    @property
    def grad(self):
        """
        Returns the current gradient matrix.
        
        Returns
        -------

        grad : 2D np.array with dimension = (nParams, nGates)
            The updated gradient matrix.
        """
        
        return self.filter.x.reshape(self.nParams, self.nGates)
    
    @property
    def cov(self):
        """
        Returns the current covariance matrix of the gradient matrix.
        
        Returns
        -------

        cov : 2D np.array with dimensions = (nParams*nGates, nParams*nGates)
            The covariance matrix of the gradient matrix.
        """
        
        return self.filter.P
    
    @property
    def sugg_dU(self):
        """
        Returns a suggestion for the next vector of voltage differences in the
        next measurement.
        
        Returns
        -------

        cov : 1D np.array with dimension = nGates
            The suggested vector of voltage differences for the next
            measurement.
        """
        
        # evaluate eigenvalues and eigenvector of the covariance matrix
        w, v = np.linalg.eigh(self.filter.P)
        
        # split the eigenvector in parts (one part contains the actual vector,
        # all else are just null vectors). They will have the length of nGates
        # after that. The eigenvector with the biggest eigenvalue is used,
        # because that is the vector with the greatest uncertainty                               
        z = np.split(v[:, np.argmax(w)], self.nParams)
        
        # just use the one vector which has a non trivial norm
        return [i for i in z if np.linalg.norm(i)!=0][0]