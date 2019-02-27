# qtune Readme: Introduction
The qtune package contains tools for the setup of a general optimization program. It is originally designed for the 
automatic fine-tuning of semiconductor spin qubits based on gate defined quantum dots, but applicable to general 
optimization problems with dependent target parameters. 
An interface to the physical back-end must be provided. With this back-end, control 
parameters (here assumed to be voltages) are set and target parameters are measured.   
Class names are written **bold** and functions *cursive* throughout the readme. UML class diagrams are inserted to show 
the heritage and dependencies, and UML activity diagrams visualize function calls.
The package abbreviations are pd for pandas and np for numpy.

# Installation
qtune is compatible with Python 3.5+. 
For development we recommend cloning the git repository
[https://github.com/qutech/qtune](https://https://github.com/qutech/qtune) and installing by:

    python setup.py develop

It can also be installed as pip package:

    pip install qtune

# Interface of the Physical Back-End
The core features of this program package do not require a specific structure of the measurement software. This section 
concerns only the required interface of the physical back-end.
The **Experiment** class serves as abstraction of the physical experiment. It provides an interface to the control 
parameters with two functions called *read_gate_voltages*() and *set_gate_voltages*(new_voltages). The function
*set_gate_voltages*(new_voltages) returns the voltages it has actually set to the experiment, which is useful if the
hardware connected to the physical experiment uses a different floating point accuracy, or the **Experiment** is 
ordered to set voltages exceeding physical or safety limits.

The **Evaluator** class provides the function *evaluate*() which returns a fixed number of parameters and a measurement
error, which is interpreted as the variance of the evaluation.

# Proposed Measurement and Evaluation Structure
The implementation of a physical back-end, as contained in the qtune package, should be regarded as proposal.

The **Experiment** provides the function *measure*(**Measurement**), which receives an instance of the **Measurement** 
class and returns the raw data.
The **Measurement** contains a dictionary of data of any type used to define the physical measurement.
The **Evaluator** class calls the function **Experiment**.*measure*(**Measurement**) to initiate the physical 
measurements. It contains a list of 
**Measurements** and the analysis software required to extract the parameters from the raw data returned by the 
experiment. This could be for example a fitting function or an edge detection. 

[evaluation image]: docs/_static/resources/EvaluationParameter.png
[autotuner coordination]: docs/_static/resources/AutotunerCoordination.png
[newton solver gradient]: docs/_static/resources/NewtonSolverGradient.png
[tuner solver]: docs/_static/resources/TunerSolver.png
[autotuner flow]: docs/_static/resources/AutotunerFlow.png

![UML class diagram depicting the dependencies of the **Evaluator**. The **Measurements** stores the instructions
for the **Experiment** in the dictionary called options. When *evaluate*() is called on the Evaluator, it calls 
*measure*(**Measurement**) on the **Experiment**.][evaluation image]

# Parameter Tuning
This section describes how the dependency between parameters is taken into account.
The parameters are grouped by instances of the **ParameterTuner** class. Each group is tuned simultaneously, i.e. 
depends on the same set of distinct parameters. The dependencies are assumed always one directional and static. The
**Autotuner** structures the groups of parameters in an hierarchy, which is represented as list of **ParameterTuners**.

Consider the following example from the tuning of a quantum dot array.
Imagine the following hierarchy consisting of three groups of parameters i.e. three **ParameterTuners**:

1. Contrast in the Sensing Dot Signal
2. Chemical Potentials / Positions of the Charge Stability Diagram
3. Tunnel Couplings

All scans require a good contrast in the sensing dot for an accurate evaluation of the parameters. Therefore the 
contrast in the sensing dot signal is the first element in the hierarchy. The measurement of tunnel couplings requires
knowledge of the positions of transitions in the charge diagram. If the chemical potentials change, the charge 
diagram is shifted, therefore the position of the charge diagram i.e. the chemical potentials must be tuned before the 
tunnel couplings. 

![UML class diagram depicting the dependencies of the **Autotuner**. When the **Autotuner** calls 
*is_tuned*(current_voltages), the **ParameterTuner** calls *evaluate*() on its list of evaluator and returns True if
the parameter values are within the desired range. The **Autotuner** calls also *get_next_voltages*() and sets these
voltages on the experiment.][autotuner coordination]

A **ParameterTuner** suggests voltages to tune the parameters in his group. 
It can be restricted to use any set of gates. It can also slice the voltage corrections 
to restrict the step size so that the algorithm is less vulnerable to the non-linearity of the target parameters. 
The tuning of a group of parameters does ideally not detune the parameters which the group depends on i.e. which are 
higher in the hierarchy.

The **Autotuner** 
class handles the coordination between the groups of parameters in the following sense. It decides which group of 
parameters must currently be evaluated or tuned and calls the **ParameterTuner** to evaluate the corresponding
group of parameters or to suggest new voltages. It also sets the new voltages on the Experiment.
It works as finite-state machine as described in the UML activity diagram below. 

![UML activity diagram of the tuning on the level of the **Autotuner**. n is the current index in the tuning hierarchy. 
The index n is incremented every time every time the parameters of the **ParameterTuner** at index n is tuned. Otherwise
the voltages suggested by this **ParameterTuner** are set to the **Experiment** and the index is reset to 0.
][autotuner flow]

# Optimization Algorithms

The voltage steps of each **ParameterTuner** are calculated by its member instance of the **Solver** class. This class 
can implement any optimization algorithm e.g. Nelder-Mead or Gauss-Newton algorithm. 
Gradient based **Solvers** like the Gauss-Newton algorithm use a instance of the **GradientEstimator** class for the
calculation of the gradient of target parameter.  

![UML class diagram depicting the dependency between the **ParameterTuner** and the **Solver**. Any time the function
*is_tuned*() is called by the **Autotuner**, the **ParameterTuner** calls *evaluate*() and uses *update_after_step*() to
update the **Solver** with the measured values. When *get_next_voltages*() is called on the **ParameterTuner**, it calls
*suggest_next_position()* on the Solver.][tuner solver]

The **GradientEstimator** subclasses implement different methods for the gradient estimation. One example is the 
Kalman filter in the **KalmanGradientEstimator**. This is an algorithm which calculates updates on the gradient by 
interpreting each measurement as finite difference measurement with respect to the last voltages. The accuracy of the
parameter evaluation is then compared to the uncertainty of the estimation of the gradient in order to find the 
most likely gradient estimation. Thereby, the gradient estimation is described as multidimensional normal distribution,
defined by a mean and a covariance matrix. If the covariance becomes to large in a certain direction, the 
**KalmanGradientEstimator** suggests a tuning step in the direction of the maximal covariance. This tuning step does not
optimize any parameter but should be understood as finite difference measurement.

![UML class diagram depicting the dependencies between the **NewtonSolver** and various **GradientEstimator** 
subclasses. The subclasses **FiniteDifferenceGradientEstimator** and **KalmanGradientEstimator** implement the 
estimation of the gradient by finite difference measurements and updates with the Kalman filter respectively.
The class**SelfInitializingKalmanEstimator** combines the two approaches by calculating the initial gradient using 
finite differences and subsequently the Kalman filter for updates.][newton solver gradient]

The crucial point in the optimization of non orthogonal systems is the ability to tune certain parameters without
changing the other ones. This requires communication between the **Solver** instances. Different **Solvers** can 
therefore share the same instances of the **GradientEstimators** so that they know the dependency of these parameters
on the gate voltages.  

Furthermore, the **Autotuner** communicates which parameters are already tuned to the **ParameterTuners**. A 
**ParameterTuner** can share this information with it's **Solver**, which then calculates update steps
in the null space of the gradients belonging to parameters which are tuned by another **ParameterTuners**. 
A **Solver** also passes this information on to it's **GradientEstimators**, which calculate the gradients only in the 
mentioned null space.

# Getting Started
The IPython notebook "setup_tutorial.ipynb" gives a detailed
tutorial for the setup of an automated fine-tuning program. The physical back-end is replaced by a simulation to enable
the tutorial to be executed before the connection to an experiment. 
In this simulated experiment, a double quantum dot and a sensing dot are tuned. The tuning hierarchy is given by 

The **ParameterTuners** and **Solvers** which are used in the setup serve as an illustrative example.
They are structured in the tuning hierarchy:

1. the sensing dot 
2. the x and y position of the charge diagram
3. two parameters, being the inter dot tunnel coupling and the singlet reload time 

The gates of the sensing dot are assumed to have only an negligible effect on the positions and 
parameters. Therefore the **Solver** of the sensing dot is independent of the others. The other gates are simultaneously
tuning the positions and parameters. The positions and parameters are tuned by **ParameterTuners** restricted to the
same gates and their **Solver** instances share all **GradientEstimators**. The **GradientEstimators** belonging to the 
parameters estimate the gradients only in the null space of the gradients belonging to the positions.

# Features
## Storage
After each evaluation of parameters, change in voltages or estimation of gradients, 
the full state of all classes except for the experiment is serialized and stored 
in an HDF5 file. The full state of the program can be reinitialized from any library file. This way, 
the program can be set back to any point during the tuning. The **History** class 
additionally saves all relevant information for the evaluation of the performance. The **History** class can plot the
gradients, last fits, control and target parameters.
## GUI
For real-time plotting of parameters and gradients, the user can couple the **History** and the
**Autotuner** to the GUI. The GUI automatically stores the program data in the HDF5 library and lets the user start and
stop the program conveniently. The program can also be ordered to execute only one step at a time. The program is 
logging its activity and the user can chose how detailed the logging describes the current activity by
setting the log level. 
# Naming Convention
## Voltages
are used in the Evaluator class to describe the voltages on the gates in the experiment.
## Positions
are an abstraction of gate voltages in the Gradient and Solver classes. These classes
could not only be used for the tuning algorithm but they could be reused in any gradient 
based solving algorithm.
## Parameters
correspond to properties of the physical experiment. They are extracted from the measurement data 
by the Evaluator class and handed over to the ParameterTuner class.
## Values
are the abstraction of parameters in the Gradient and Solver classes.
## Options
describe the measurements in the Measurement class.

# License
    Copyright (c) 2017 and later, JARA-FIT Institute for Quantum Information,
    Forschungszentrum Jülich GmbH and RWTH Aachen University

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
