import torch
import torch.nn as nn
# Define a function to get batches of data

def get_batch(batch_size, data):
    for i in range(0, len(data), batch_size):
        yield data[i:i+batch_size]

# Train the network using batches


class ActionClassifier(nn.Module):
    def __init__(self, input_size, output_size):
        super(ActionClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

state_size = 32
numEpochs = 100
model = ActionClassifier(input_size=state_size, output_size=6)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the network
for epoch in range(num_epochs):
    for batch in get_batch(batch_size, training_data):
        inputs, labels = zip(*batch)
        inputs = torch.stack(inputs)
        labels = torch.tensor(labels)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()