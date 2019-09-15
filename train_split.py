"""train_split.py
BEFORE RUNNING THIS: run prepare_train.py

1. Train the neural net on the labeled data.
"""

import torch
from torch import nn
from torch import optim
from matplotlib import pyplot as plt
from model import Net

data_folder = "data/label/"
model_folder = "models/"
batch_size = 32
val_size = 100
n_epochs = 20
auxiliary = True

train_input = torch.load(data_folder + "imgs_expanded_split.pt")
train_target = torch.load(data_folder + "targets_expanded_split.pt")
mu, std = train_input.mean(), train_input.std()
train_input = train_input.sub(mu).div(std)
print(train_target.size())
idx = torch.randperm(len(train_input))
train_input = train_input[idx]
train_target = train_target[idx]

val_input = train_input[:val_size]
val_target = train_target[:val_size]
train_input = train_input[val_size:]
train_target = train_target[val_size:]

model = Net(batch_size=batch_size, auxiliary=auxiliary)
model.train()
optimizer = optim.Adam(model.parameters(), lr=1e-5)
criterion = nn.MSELoss()

# Comment in case you want to train from scratch
model.load_state_dict(torch.load(
    model_folder + "model_state_dict_cnn_2_aux.pt"))
optimizer.load_state_dict(
    torch.load(model_folder + "optim_state_dict_cnn_2_aux.pt"))

# Uncomment to freeze some weights
"""
count = 0
for child in model.features.children():
    count += 1
    if count > len(list(model.features.children())) - 2:
        print(child)
        for param in child.parameters():
            param.requires_grad = False
"""

avg_losses = []
val_losses = []
accs_train = []

for e in range(n_epochs):
    model.train()
    model.set_batch_size(batch_size)
    sum_loss = 0
    for b in range(0, train_input.size(0), batch_size):

        if b + batch_size > train_input.size(0):
            break

        if auxiliary:
            out_1, out_2, out_fin = model(train_input.narrow(0, b, batch_size))
            loss_1 = criterion(
                out_1, train_target[b:b + batch_size, 0, :].view(batch_size, -1))
            loss_2 = criterion(
                out_2, train_target[b:b + batch_size, 1, :].view(batch_size, -1))
            loss_fin = criterion(
                out_fin, train_target[b:b + batch_size].view(batch_size, -1))

            loss = loss_1 + loss_2 + loss_fin

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sum_loss += loss_fin.item()
        else:
            output = model(train_input.narrow(0, b, batch_size))
            target = train_target.narrow(0, b, batch_size)
            loss = criterion(output, target.view(batch_size, -1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sum_loss += loss.item()

    avg_loss = sum_loss / (train_input.size(0) / batch_size)

    model.eval()
    model.set_batch_size(val_size)
    if auxiliary:
        _, _, out_fin = model(val_input)
    else:
        out_fin = model(val_input)
    val_loss = criterion(out_fin, val_target.view(val_size, -1))

    avg_losses.append(avg_loss)
    val_losses.append(val_loss)

    model._update_progress((e + 1) / n_epochs, avg_loss, val_loss)

# Uncomment to see the training curves
"""
plt.plot(avg_losses[1:], label="Train Loss")
plt.plot(val_losses[1:], label="Val Loss")
plt.legend()
plt.yscale("log")
plt.show()
"""

torch.save(
    torch.tensor(avg_losses), model_folder + "avg_losses_cnn_2_aux_2_no-frozen.pt")
torch.save(
    model.state_dict(), model_folder + "model_state_dict_cnn_2_aux_2_no-frozen.pt")
torch.save(
    optimizer.state_dict(), model_folder + "optim_state_dict_cnn_2_aux_2_no-frozen.pt")
