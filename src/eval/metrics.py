import math

class Metrics:
    def __init__(self, window=50):
        self.window = window
        self.losses = []
        self.steps = []

    # updates the values of the losses and steps
    def update(self, step:int, loss: float):
        self.losses.append(loss)
        self.steps.append(step)

    def calculate_avg_loss(self):
        calculated_loss = sum(self.losses) / len(self.losses)
        if not self.losses:
            return None
        else:
            return calculated_loss

