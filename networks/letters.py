import torch


class LettersNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 6, 5)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(6, 16, 5)
        self.conv3 = torch.nn.Conv2d(16, 32, 5)
        self.conv4 = torch.nn.Conv2d(32, 48, 5)
        self.fc1 = torch.nn.Linear(4800, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 3)
        self.dropout = torch.nn.Dropout(0.4)

    def forward(self, x):
        x = self.dropout(self.pool(torch.nn.functional.relu(self.conv1(x))))
        x = self.pool(torch.nn.functional.relu(self.conv2(x)))
        x = self.dropout(self.pool(torch.nn.functional.relu(self.conv3(x))))
        x = self.pool(torch.nn.functional.relu(self.conv4(x)))
        x = self.dropout(torch.flatten(x, 1)) # flatten all dimensions except batch
#         print(x.shape)
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.dropout(torch.nn.functional.relu(self.fc2(x)))
        x = self.fc3(x)
        x = torch.softmax(x, dim=1)
        return x