import imageio
import glob
filenames = glob.glob('./data/*.jpg')
images = []
for filename in filenames:
    images.append(imageio.imread(filename))
imageio.mimsave('./assets/example.gif', images)