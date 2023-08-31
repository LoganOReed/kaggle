import platform

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data


class FNet(nn.Module):
    def __init__(self):
        super(FNet, self).__init__()
        # 784 inputs connects to hidden layer with 600 nodes
        self.fc1 = nn.Linear(in_features=784, out_features=600)
        self.fc2 = nn.Linear(in_features=600, out_features=500)
        self.fc3 = nn.Linear(in_features=500, out_features=250)
        self.fc4 = nn.Linear(in_features=250, out_features=10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


def get_accuracy(predictions, true_labels):
    _, predicted = torch.max(predictions, 1)
    corrects = (predicted == true_labels).sum()
    accuracy = 100.0 * corrects / len(true_labels)
    return accuracy.item()


def training(data_loader, epochs, model, criterion, optimizer):
    train_accuracies, train_losses = [], []
    # set train mode
    model.train()

    for epoch in range(epochs):
        train_loss = 0
        train_accuracy = 0
        num_batch = 0

        # iterate over all batches
        for data, labels in data_loader:
            # zero the parameters gradient to not accumulate gradients from previous iteration
            optimizer.zero_grad()

            # put data into the model
            predictions = net(data)

            # calculate loss
            loss = criterion(predictions, labels)

            # calculate accuracy
            accurasy = get_accuracy(predictions, labels)

            # compute gradients
            loss.backward()

            # change the weights
            optimizer.step()

            num_batch += 1
            train_loss += loss.item()
            train_accuracy += accurasy

        epoch_accuracy = train_accuracy / num_batch
        epoch_loss = train_loss / num_batch
        train_accuracies.append(epoch_accuracy)
        train_losses.append(epoch_loss)

        print(
            "Epoch: {}/{} ".format(epoch + 1, epochs),
            "Training Loss: {:.4f} ".format(epoch_loss),
            "Training accuracy: {:.4f}".format(epoch_accuracy),
        )

    return train_accuracies, train_losses


def plot_img(data, label):
    """Prints a 3x3 array of handwritten digits."""
    _, axs = plt.subplots(3, 3)  # 9 images
    k = 0
    for i in range(3):
        for j in range(3):
            axs[i, j].imshow(data[k].astype("uint8").reshape(28, 28))  # plot image
            axs[i, j].set_ylabel("label:" + str(label[k].item()))  # print label
            k += 1
    plt.show()


def train_curves(epochs, train_losses, train_accuracies):
    iters = range(1, epochs + 1)
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(15, 5))
    fig.suptitle("Training Curve")
    ax1.plot(iters, train_losses)
    ax1.set_xlabel("Iterations")
    ax1.set_ylabel("Loss")
    ax2.plot(iters, train_accuracies, color="g")
    ax2.set_xlabel("Iterations")
    ax2.set_ylabel("Training Accuracy")
    plt.show()


torch.manual_seed(0)
# print to kitty
if platform.system() == "Linux":
    plt.switch_backend("module://matplotlib-backend-kitty")

# Get cpu, gpu or mps device for training.
device = "rocm" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

train_data = pd.read_csv("data/digit/train.csv")
test_data = pd.read_csv("data/digit/test.csv")

# Seperate the first column = result column
train_all = train_data.iloc[:, 1:].to_numpy()
# creates array of labels
train_all_label = train_data["label"].to_numpy()

# plot_img(train_all, train_all_label)
# convert numpy arrays to tensor
train = torch.as_tensor(train_all).type(torch.FloatTensor)  # type: ignore
train_label = torch.as_tensor(train_all_label)
test = torch.as_tensor(test_data.to_numpy()).type(torch.FloatTensor)  # type: ignore

# unique, counts_train = np.unique(train_label, return_counts=True)
# plt.subplot(1, 2, 1)
# plt.bar(unique, counts_train/len(train_label))
# unique, counts_val = np.unique(validation_label, return_counts=True)
# plt.subplot(1, 2, 2)
# plt.bar(unique, counts_val/len(validation_label))
# plt.show()


train_dataset = torch.utils.data.TensorDataset(train, train_label)
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)

epochs = 50

net = FNet()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=3e-4)

train_accuracies, train_losses = training(
    trainloader, epochs, net, criterion, optimizer
)

train_curves(epochs, train_losses, train_accuracies)

# set net in test (evaluation) mode
net.eval()

# get predictions for test data
test_predictions = net(test)

# to get class with the maximum score as prediction
_, test_predicted = torch.max(test_predictions.data, 1)

# Save results in the required format
output = pd.DataFrame({"ImageId": test_data.index + 1, "Label": test_predicted})
output.to_csv("data/digit/submission.csv", index=False)
output.head()
