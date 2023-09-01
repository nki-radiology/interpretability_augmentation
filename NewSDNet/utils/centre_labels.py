"""When training the classifier to classify the different centres the labels in the form of
integers must be contiguous, so we have to specify the different cases depending on the
out-of-distribution centre
"""

centres_labels = {
    "centre1": [5, 0, 1, 2, 3, 4],
    "centre2": [0, 5, 1, 2, 3, 4],
    "centre3": [0, 1, 5, 2, 3, 4],
    "centre4": [0, 1, 2, 5, 3, 4],
    "centre5": [0, 1, 2, 3, 5, 4],
    "centre6": [0, 1, 2, 3, 4, 5],
}
