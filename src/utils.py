import numpy as np
import matplotlib.pyplot as plt


def imshow(img, name="sample.png"):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
    plt.savefig(name)
