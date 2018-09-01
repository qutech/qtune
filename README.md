#Qtune Readme

The program package contains tools for the setup of a general optimization program. It is specifically designed for the 
automatic fine-tuning of semiconductor spin qubits based on gate defined quantum dots.  
The program must be completed by an interface to the physical backend where control 
parameters are set and target parameters are measured.  

##Interface to the Physical Backend

The interface is given by an **Experiment** class which can set voltage (or other control parameters) and conduct 
measurements. The definition of a specific measurement is stored in the **Measurement** class. An instance of
Measurement is given to the **Experiment**, which is conducting the actual measurement and returns the raw data.

##Target Parameter

The **Evaluator** class operates on the **Experiment** to measure a specific parameter. It contains a list of 
Measurements and 
an implementation of the analysis software required to extract the parameter from the raw data returned by the 
experiment. Each **Evaluator** represents the parameter it is evaluating.

###Interdependency

The class **ParameterTuner** represents a group of target parameters, which is tuned simultaneously. The 
**ParameterTuner** can use any set of gates to tune his group of parameters and it can slice the voltage corrections 
to restrict the step size so that the algorithm is less vulnerable to the non-linearity of target parameters.  
The **Autotuner** 
class handles the communication between the control parameters, set on the experiment and the 
groups of target parameters. It structures the groups of target parameters in an hierarchy, which expresses the physical
interdependency between the parameter groups.  
Consider for example a hierarchy consisting of three **ParameterTuners**:
1. Contrast in the Sensing Dot Signal
2. Chemical Potentials / Positions of the Charge Stability Diagram
3. Tunnel Couplings

All scans require a good contrast in the sensing dot for an accurate evaluation of the parameters. Therefore the 
contrast in the sensing dot signal is the lowest element in the hierarchy. The measurement of tunnel couplings requires
the knowledge of the positions of transitions in the charge diagram. If the chemical potentials change, the charge 
diagram is shifted, therefore the position of the charge diagram or the chemical potentials must be tuned before the 
tunnel couplings.  

The **Autotuner** works in iterations. In one iteration the **Autotuner** either measures target parameters, orders
the next voltages from a **ParameterTuner** or sets new voltages. Anytime the measured target parameters are accepted, 
the **Autotuner** goes one level up in the hierarchy and any time new control parameters are set, the **Autotuner** 
starts at the bottom of the tuning hierarchy. The **Autotuner** also controls the communication between the 
**ParameterTuners** by transmitting which parameters are already tuned. 

###Optimization

The voltage steps of each **ParameterTuner** are calculated by its member instance of the **Solver** class. This class 
can implement any optimization algorithm e.g. Nealder-Mead or Gauss-Newton algorithm. 
Gradient based **Solver** like the Gauss-Newton algorithm use the **GradientEstimator** class for calculation of the 
gradients of target parameters.  
The **GradientEstimator** subclasses implement different types of gradient estimation. The **KalmanGradientEstimator** 
implements the Kalman filter for gradients. This is an algorithm which calculates updates on the gradient by 
interpreting each measurement as finite difference measurement with respect to the last voltages. The accuracy of the
parameter evaluation is then compared to the uncertainty of the estimation of the gradient in order to find the 
most likely estimation of the gradient. The gradients can be calculated purely by Kalman updates or initially by finite
differences.

##Getting Started
See example_setup.md for a detailed tutorial 

##Deployment

All classes except for the experiment are serialized and stored in an HDF5 library. They can be reinitialized at any 
point during the tuning. This way, the program can be set back to any point during the tuning. The **History** class 
additionally saves all relevant information for the evaluation of the performance. The **History** class can plot the
gradients, last fits, control and target parameters.

The program is logging its activity and the user can chose how detailed the logging describes the current activity by 
setting the log level. For realtime plotting of parameters and gradients, the user can  couple the **History** and the 
**Autotuner** to the GUI. The GUI automatically stores the program data in the HDF5 library and lets the user
 start and stop the program conveniently. The program can also be ordered to execute only one iteration at a time. 


##Naming Convention

###Voltages
are used in the Evaluator class to describe the voltages on the gates in the experiment.

###Positions
are an abstraction of gate voltages in the Gradient and Solver classes. These classes
could not only be used for the tuning algorithm but they could be reused in any gradient 
based solving algorithm.

###Parameters
correspond to properties of the physical experiment. They are extracted from the measurement data 
by the Evaluator class and handed over to the ParameterTuner class.

###Values
are the abstraction of parameters in the Gradient and Solver classes.

###Options
describe the measurements in the Measurement class.