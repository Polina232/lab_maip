import json
import numpy as np
from matplotlib import pyplot as plt
from skimage.transform import AffineTransform, warp
from skimage.io import imread, imshow, show
from skimage.exposure import histogram

settings= {
    'parameter_0':'лаб_1.jpg',
    'parameter_1': 1.5, # scale x
    'parameter_2': 0.9, # scale y
    'parameter_3': -40, # translation x
    'parameter_4': -20, # translation y
    'parameter_5': 25, # rotation
}

# пишем в файл
with open('settings.json', 'w') as fp:
     json.dump(settings, fp)

# читаем из файла
with open('settings.json') as json_file:
    json_data = json.load(json_file)

num_1 = json_data['parameter_0']
num_2 = json_data['parameter_1']
num_3 = json_data['parameter_2']
num_4 = json_data['parameter_3']
num_5 = json_data['parameter_4']
num_6 = json_data['parameter_5']

fig = plt.figure(figsize=(12, 7))

fig.add_subplot(2, 2, 1)
imshow(num_1)

img_pepe = imread(num_1)
tform = AffineTransform(scale=[num_2,num_3], rotation=num_6, translation=[num_4,num_5])
img_pepe_tformed = warp(img_pepe, tform.inverse) #деформация изображения
fig.add_subplot(2, 2, 2)
img_pepe_tformed *= 255 # перобразование типа
imshow(img_pepe_tformed.astype(np.uint8))

hist_red_num_1, bins_red_num_1 = histogram(img_pepe[:, :, 0])
hist_green_num_1, bins_green_num_1 = histogram(img_pepe[:, :, 1])
hist_blue_num_1, bins_blue_num_1 = histogram(img_pepe[:, :, 2])
fig.add_subplot(2, 2, 3)
plt.ylabel('число отсчетов')
plt.xlabel('значение яркости')
plt.title('Гистограмма для исходного изображения')
plt.plot(bins_green_num_1, hist_green_num_1/(img_pepe.shape[0]*img_pepe.shape[1]), color='green', linestyle = '-', linewidth=1)
plt.plot(bins_red_num_1, hist_red_num_1/(img_pepe.shape[0]*img_pepe.shape[1]), color='red', linestyle = '-', linewidth=1)
plt.plot(bins_blue_num_1, hist_blue_num_1/(img_pepe.shape[0]*img_pepe.shape[1]), color='blue', linestyle = '-', linewidth=1)
plt.legend(['green', 'red', 'blue'])

hist_red_formed, bins_red_formed = histogram(img_pepe_tformed[:, :, 0])
hist_green_formed, bins_green_formed = histogram(img_pepe_tformed[:, :, 1])
hist_blue_formed, bins_blue_formed = histogram(img_pepe_tformed[:, :, 2])
fig.add_subplot(2, 2, 4)
plt.ylabel('число отсчетов')
plt.xlabel('значение яркости')
plt.title('Гистограмма для измененного изображения')
plt.plot(bins_green_formed, hist_green_formed/(img_pepe_tformed.shape[0]*img_pepe_tformed.shape[1]), color='green', linestyle = '-', linewidth=1)
plt.plot(bins_red_formed, hist_red_formed/(img_pepe_tformed.shape[0]*img_pepe_tformed.shape[1]), color='red', linestyle = '-', linewidth=1)
plt.plot(bins_blue_formed, hist_blue_formed/(img_pepe_tformed.shape[0]*img_pepe_tformed.shape[1]), color='blue', linestyle = '-', linewidth=1)
plt.legend(['green', 'red', 'blue'])

plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=3.0)

show()

