import math

class Metrics:
    def __init__(self, window=50):
        self.window = window
        self.losses = []
        self.steps = []

    def save_losses(self, loss, step):
        self.losses.append(loss)
        self.steps.append(step)
    
    def calculate_avg_loss(self):
        return sum(self.losses) / len(self.losses)

    def calculate_moving_average(self):
        moving_averages = []
        for i in range(len(self.losses)):
            window = self.losses[i:i+self.window]
            moving_averages.append(sum(window)) / len(window)

        return moving_averages
