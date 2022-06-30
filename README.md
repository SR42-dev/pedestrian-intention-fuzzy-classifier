# pedestrian-intention-fuzzy-classifier
A research project repository containing algorithms to classify pedestrian obstruction probabilities into fuzzy logic states using 2D cameras 

### Current week : 3

### Next short-term goal(s) :

- Predict pedestrian positions on frame based on collected data and compare with reality (show the path of the camera and that of the pedestrian intersecting with IRL test video)
    - Hardcode overlay and visualize 2D representation of 3D path [DONE]
    - Implement fuzzy logic [DONE]
    - Take the last nth (experiment) frame as last frame rather than immediately previous one [DONE]
    - Record shoulder locations, predicted locations, apparent target height, past frame number from present, error thresholds [DONE]
    - Find accuracies for various radii of prediction [DONE]
    - Implement euclidean distance between past locations to determine the direction [DONE]
    - Form the vectors for direction and get euclidean distance [DONE]
    - Last window and future window implementation.
    
    - Implement 1D kalman filter for all varying values
    - Extend algorithm to multiple pedestrians by multithreading
    
- Figure out formulae to get collision probability from obtained variables.
- Draw decision making automata & fuzzy logic map for collision prediction.

## Cheat sheets :

### Git commit & push cheat sheet for this repository:
- git status
- git add --all
- git status
- <Edit commitMessage.txt appropriately>
- git commit -F commitMessage.txt
- git status
- git log --all --decorate --oneline --graph

### Jupyter notebook cheat sheet :
- jupyter notebook <relative path name of the notebook>
- jupyter notebook --notebook-dir="<absolute path in case of directory switch>"




