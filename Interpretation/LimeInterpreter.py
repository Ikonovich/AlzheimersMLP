from NeuralNetwork import NeuralNetwork


class LimeInterpreter():
    # Implements LIME interpretation for image classifiers, as described in:
    #
    # Ribeiro, M. T., Singh, S., Guestrin, C. (2016, August 9). "why should I trust you?":
    # Explaining the predictions of any classifier.
    # Retrieved February 19, 2023,
    # from https://arxiv.org/abs/1602.04938

    def __init__(self, classifier: NeuralNetwork):
        self.classifier = classifier
