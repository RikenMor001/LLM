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
        if not self.losses:
            return None
        return sum(self.losses) / len(self.losses)

    # what is ppl?
    # ppl stands for perplexity, is a measure of how well a model 
    # predicts the next token in sequence
    def get_ppl(self):
        avg_loss = self.calculate_avg_loss()
        if avg_loss is None:
            return None
        else: 
            return math.exp(avg_loss)

    def log(self, step: int, lr: float = None, val_loss: float = None):
        avg_loss = self.calculate_avg_loss()
        ppl = self.get_ppl()

        if avg_loss is None:
            return

        msg = (
            f"Step {step} | "
            f"Loss: {avg_loss:.4f} | "
            f"PPL: {ppl:.2f}"
        )
        if lr is not None:
            msg += f" | LR: {lr:.2e}"
        print(msg)

        if val_loss is not None:
            val_ppl = math.exp(val_loss)
            print(f"         Val loss: {val_loss:.4f} | Val PPL: {val_ppl:.2f}")