from torch.optim import Optimizer

class AdamWMuonWrapper:
    def __init__(self, adam: Optimizer, muon: Optimizer):
        self.muon = muon
        self.adam = adam

    def zero_grad(self):
        self.muon.zero_grad()
        self.adam.zero_grad()

    def step(self):
        self.muon.step()
        self.adam.step()
