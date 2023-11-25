

class Layer:

    def __init__(self, network, input_size: int, output_size: int):
        self.input_size = input_size
        self.output_size = output_size
        self.input = None
        self.output = None
        self.loss = None
        self.delta = None
        self.network = None

    def forward(self, data: list[float]):
        pass

    def backward(self, loss: list[float]):
        pass
