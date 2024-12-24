import numpy as np
import matplotlib.pyplot as plt

IMG_SIZE = 8
NUM_SAMPLES = 10

#Creates the test data with images of size IMG_SIZE x IMG_SIZE and labels -1 for vertical lines and +1 for horizontal lines
def create_data(num_images, size, bg_max_noise = 0.5, fg_max_noise = 0.1):
    images = []
    labels = []
    for i in range(0, num_images):
        #Generate some noisy background
        img = np.random.rand(size, size) * bg_max_noise
        #Create either a horizontal or a vertical line of length size/2 + 1
        line_length = int(size / 2 + 1)
        h = np.random.choice([0, 1])
        if h == 1:
            #Add a horizontal line
            x = np.random.choice(size - line_length)
            y = np.random.choice(size)
            for xi in range(x, x + line_length):
                img[y][xi] = 1 - np.random.rand(1) * fg_max_noise
            images.append(img)
            labels.append(1)
        else:
            #Add a vertical line
            x = np.random.choice(size)
            y = np.random.choice(size - line_length)
            for yi in range(y, y + line_length):
                img[yi][x] = 1 - np.random.rand(1) * fg_max_noise
            images.append(img)
            labels.append(-1)
    return images, labels

images, labels = create_data(10, IMG_SIZE)

for i in range(0, 4):
    plt.subplot(2,2,i+1)
    plt.imshow(images[i], cmap="gray")
    plt.title(labels[i])

plt.show()