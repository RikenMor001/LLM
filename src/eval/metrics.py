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
        
        if len(self.losses) > self.window:
            self.losses.pop(0)
            self.steps.pop(0)

    def calculate_avg_loss(self):
        calculated_loss = sum(self.losses) / len(self.losses)
        if not self.losses:
            return None
        else:
            return calculated_loss

    # what is ppl?
    # ppl stands for perplexity, is a measure of how well a model 
    # predicts the next token in sequence
    def get_ppl(self):
        avg_loss = self.calculate_avg_loss()
        if avg_loss is None:
            return None
        else: 
            return math.exp(avg_loss)

    def log(self, step: int):
        avg_loss = self.calculate_avg_loss()
        ppl = self.get_ppl()

        if avg_loss is None:
            return None

        print(
            f"Step: {step} "
            f"Loss: {avg_loss:.4f} | "
            f"PPL: {ppl:.2f}"
        )