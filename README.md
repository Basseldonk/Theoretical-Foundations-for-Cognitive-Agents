Theoretical-Foundations-for-Cognitive-Agents project on explainable AI.

Authors:	Bram van Asseldonk
		Loes Habermehl
		Rik van Lierop

This code was part of a project for the course Theoretical Foundations for Cognitive Agents in the master programme Artificial Intelligence at the Radboud University.

This code will asks users to label a set of pictures accoring to attractiveness. Once a set of pictures has been labeled, a MLP is trained on several objective
and subjective measures that represent the pictures and the corresponding labels. After the MLP is trained, the new MLPs are trained on the same data with one of
the features omitted using Recursive Feature Elemination (RFE). This results in a list of accuracies that can be compared to the MLP that was trained on all features
to figure out which features were most influential for the performance of the network. 

This code is not maintained any longer. Anyone is free to use this code as they see fit. 
