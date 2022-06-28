# pedestrian-intention-fuzzy-classifier
A research project repository containing algorithms to classify pedestrian obstruction probabilities into fuzzy logic states using 2D cameras 

## Next short-term goal(s) :

Predict pedestrian positions on frame based on collected data and compare with reality (show the path of the camera and that of the pedestrian intersecting with IRL test video)
- hardcode overlay and visualize 2D representation of 3D path [DONE]
- implement fuzzy logic [DONE]
- take the last nth (experiment) frame as last frame rather than immediately previous one
- record shoulder locations, predicted locations, apparent target height, past frame number from present

- implement 1D kalman filter for all varying values
- extend algorithm to multiple pedestrians

Figure out formulae to get collision probability from obtained variables.

Draw decision making automata & fuzzy logic map for collision prediction.




