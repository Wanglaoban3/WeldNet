import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc = nn.Linear(2, 2, bias=False)

    def forward(self, x1, x2):
        y1 = self.fc(x1)
        y2 = self.fc(x2)
        y1 = F.normalize(y1)
        y2 = F.normalize(y2)
        sim = (y1 * y2).sum()

        return y1, y2, sim

if __name__ == "__main__":
    model = Model()
    x1 = torch.rand([1, 2])
    x2 = torch.rand([1, 2])
    optim = torch.optim.SGD(model.parameters(), lr=1e-1)
    for i in range(30):
        a1, a2, sim = model(x1, x2)
        loss = (-sim).exp().mean().log()
        loss.backward()
        optim.step()
        optim.zero_grad()