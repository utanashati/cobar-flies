"""view_keypoints_test.py
BEFORE RUNNING THIS: get the data and the working model

Create an animation of a single fly throughout a single iteration.
"""

import torch
from model import Net
from matplotlib import pyplot as plt
import matplotlib.animation as animation

test_input = torch.load("data/orig/01_Pink_impTNT/01_pink_imptnt_01/img_split_sorted.pt")[0]
batch_size = test_input.size(0)

mu, std = test_input.mean(), test_input.std()
test_input = test_input.sub(mu).div(std)

train_target = torch.load("data/label/targets_expanded_split.pt")
print("Size", test_input[:batch_size].size())
model = Net(batch_size=batch_size)
model.load_state_dict(torch.load("models/model_state_dict_cnn_2_aux_2_no-frozen.pt"))

with torch.no_grad():
    model.eval()
    _, _, output = model(test_input[:batch_size])
    output = output.view(batch_size, -1)
print(output.size())

# Animation
fig = plt.figure(figsize=(6, 2))
plt.title("Neural Network Prediction Sample")
ims = []

for i in range(batch_size):
    im1 = plt.imshow(test_input[i, 0, :, :], cmap=plt.get_cmap("gray"))
    im2 = plt.scatter((output[i, 0]) * test_input.size(3), output[i, 1] * test_input.size(2),
        c=list(plt.rcParams['axes.prop_cycle'])[0]['color'])
    im3 = plt.scatter((output[i, 2]) * test_input.size(3), output[i, 3] * test_input.size(2),
        c=list(plt.rcParams['axes.prop_cycle'])[1]['color'])
    if i == 0:
        im2.set_label("head")
        im3.set_label("bottom")
    plt.legend()
    ims.append((im1, im2, im3,))
    #plt.show()

im_ani = animation.ArtistAnimation(fig, ims, interval=100, repeat_delay=3000,
                                   blit=True)
im_ani.save('pics/prediction_ani.mp4')
