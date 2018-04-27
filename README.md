#Qtune Readme

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