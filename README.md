#qtune Readme
The program package contains tools for the setup of a general optimization program. It is specifically designed for the 
automatic fine-tuning of semiconductor spin qubits based on gate defined quantum dots.  
An interface to the physical backend must be provided. With this backend, control 
parameters are set and target parameters are measured.   
Class names are written bold throughout the readme. UML class diagrams are inserted to show the dependencies of the
classes, and UML activity diagrams visualize function calls.
The package abbreviations are pd for pandas and np for numpy.
##Interface to the Physical Backend
The interface is given by an **Experiment** class which can set voltage (or other control parameters) and conduct 
measurements. The definition of a specific measurement is stored in the **Measurement** class. An instance of
Measurement is given to the **Experiment**, which is conducting the actual measurement and returns the raw data.
##Target Parameter
The **Evaluator** class operates on the **Experiment** class to measure a specific parameter. It contains a list of 
Measurements and 
an implementation of the analysis software required to extract the parameter from the raw data returned by the 
experiment. Each **Evaluator** represents the parameter it is evaluating. The **Evaluator** has acces to the 
**Experiment** to conduct scans saved in the Measurements.

[evaluation image]: docs/_static/resources/EvaluationParameter.png
[autotuner coordination]: docs/_static/resources/AutotunerCoordination.png
[newton solver gradient]: docs/_static/resources/NewtonSolverGradient.png
[tuner solver]: docs/_static/resources/TunerSolver.png
[autotuner flow]: docs/_static/resources/AutotunerFlow.png

![alt text][evaluation image =300x300]
###Interdependency

The class **ParameterTuner** represents a group of target parameters, which is tuned simultaneously. The 
**ParameterTuner** can use any set of gates to tune his group of parameters and it can slice the voltage corrections 
to restrict the step size so that the algorithm is less vulnerable to the non-linearity of target parameters.  
The **Autotuner** 
class handles the communication between the control parameters, set on the experiment and the 
groups of target parameters. It structures the ParameterTuners in an hierarchy, which expresses the physical
interdependency between the parameter groups.  

![alt text][autotuner coordination]

Consider for example a hierarchy consisting of three **ParameterTuners**:
1. Contrast in the Sensing Dot Signal
2. Chemical Potentials / Positions of the Charge Stability Diagram
3. Tunnel Couplings

All scans require a good contrast in the sensing dot for an accurate evaluation of the parameters. Therefore the 
contrast in the sensing dot signal is the lowest element in the hierarchy. The measurement of tunnel couplings requires
the positions of transitions in the charge diagram. If the chemical potentials change, the charge 
diagram is shifted, therefore the position of the charge diagram i.e. the chemical potentials must be tuned before the 
tunnel couplings.  

The **Autotuner** works in iterations. In one iteration the **Autotuner** either measures target parameters, orders
the next voltages from a **ParameterTuner** or sets new voltages. Anytime the measured target parameters are accepted, 
the **Autotuner** goes one level up in the hierarchy and any time new control parameters are set, the **Autotuner** 
starts at the bottom of the tuning hierarchy. The **Autotuner** also controls the communication between the 
**ParameterTuners** by transmitting which parameters are already tuned. 

![alt text][autotuner flow]
###Optimization

The voltage steps of each **ParameterTuner** are calculated by its member instance of the **Solver** class. This class 
can implement any optimization algorithm e.g. Nealder-Mead or Gauss-Newton algorithm. 
Gradient based **Solvers** like the Gauss-Newton algorithm use a instance of the **GradientEstimator** class for the
calculation of the 
gradients of target parameters.  

![alt text][tuner solver]

The **GradientEstimator** subclasses implement different types of gradient estimation. One example is the 
**KalmanGradientEstimator** which
implements the Kalman filter for gradients. This is an algorithm which calculates updates on the gradient by 
interpreting each measurement as finite difference measurement with respect to the last voltages. The accuracy of the
parameter evaluation is then compared to the uncertainty of the estimation of the gradient in order to find the 
most likely estimation of the gradient. The gradients can be calculated purely by Kalman updates or initially by finite
differences. If a **GradientEstimator** can not estimate the gradient in a certain direction with sufficient accuracy,
then he also suggests measurements in this direction. 

![alt text][newton solver gradient]

The crucial point in the optimization of non orthogonal systems is the ability to tune certain parameters without
changing the other ones. This requires communication between the **Solver** instances. Different **Solvers** can 
therefore share the same instances of the **GradientEstimators** so that they know the dependency of these parameters
on the gate voltages.  

Furthermore, the **Autotuner** communicates to the **ParameterTuners** which parameters are already tuned. A 
**ParameterTuner** can share this information with it's **Solver**. A **Solver** then calculates only update steps
in the nullspace of the gradients belonging to parameters which are tuned by another **ParameterTuner**. The 
**GradientEstimators** only determine their gradients in direction in which the tuned parameters are constant, since
only steps in these directions are executed for the tuning.
##Getting Started
The IPython notebook "setup_tutorial.ipynb" gives a detailed
tutorial for the setup of an automated fine-tuning program. The physical backend is replaced by a simulation to enable
the tutorial to be executed before the connection to an experiment. 
In this simulated experiment, a double quantum dot an a sensing dot are tuned. The tuning hierarchy is given by 1. the
sensing dot, 2. the positions of the charge diagram and 3. two parameters, being the inter dot tunnel coupling and the
singlet reload time. 

The gates of the sensing dot are assumed to have only an negligible effect on the positions and 
parameters. Therefore the **Solver** of the sensing dot is independet of the others. The other gates are simultaneously
tuning the positions and parameters. The **Solver** instances of the positions and parameters share all 
**GradientEstimators**.
##Features
###Storage
After each iteration of the Autotuner, the full state of all classes except for the experiment is serialized and stored 
in an HDF5 library. The full state of the program can be reinitialized from any iteration. This way, 
the program can be set back to any point during the tuning. The **History** class 
additionally saves all relevant information for the evaluation of the performance. The **History** class can plot the
gradients, last fits, control and target parameters.
###Logging
The program is logging its activity and the user can chose how detailed the logging describes the current activity by
setting the log level. For realtime plotting of parameters and gradients, the user can  couple the **History** and the
**Autotuner** to the GUI. The GUI automatically stores the program data in the HDF5 library and lets the user start and
stop the program conveniently. The program can also be ordered to execute only one iteration at a time.
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