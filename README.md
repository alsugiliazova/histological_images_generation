## Синтез гистологических изображений.  
### В этом репозитории лежит код для курсовой/дипломной работы.  
Ссылка на используемый датасет: [Warwick](https://livecsmsu-my.sharepoint.com/personal/dsorokin_live_cs_msu_ru/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fdsorokin%5Flive%5Fcs%5Fmsu%5Fru%2FDocuments%2FHistology%20synthesis%2Fcw%2Ezip&parent=%2Fpersonal%2Fdsorokin%5Flive%5Fcs%5Fmsu%5Fru%2FDocuments%2FHistology%20synthesis&ga=1).  
Перед запуском ноутбука следует положить папку с названием cw в ту же директорию, что и ноутбук. 

**Проекты, которые использовались при выполнении дипломной работы:**
1. https://github.com/manicman1999/GAN256 - для генерации масок отдельных желез.
2. https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix - для генерации текстуры отдельной железы.
3. https://github.com/NVIDIA/pix2pixHD - для генерации текстуры фона.


Работа выполнялась в следующем порядке:
1. Генерация маски отдельной железы:
    1. Алгоритм SSM (https://github.com/alsugiliazova/histological_images_generation/blob/master/StatisticalShapeModels.ipynb)
    2. Нейросетевой метод генерации маски отдельной железы (https://github.com/manicman1999/GAN256)
    ![Иллюстрация к проекту](https://github.com/alsugiliazova/histological_images_generation/blob/master/images/masks_gan/gan_img_32.jpg)
    ![alt-text-1](https://github.com/alsugiliazova/histological_images_generation/blob/master/images/masks_gan/gan_img_5.jpg "title-1") ![alt-text-2](https://github.com/alsugiliazova/histological_images_generation/blob/master/images/masks_gan/gan_img_42.jpg "title-2")
2. Алгоритм расположения желез на фоне (https://github.com/alsugiliazova/histological_images_generation/blob/master/make_full_mask.ipynb)
![alt-text-1](https://github.com/alsugiliazova/histological_images_generation/blob/master/images/full_mask/22_mask.jpg "title-1") 
3. Генерация текстуры отдельной железы по маске (https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
![Иллюстрация к проекту](https://github.com/alsugiliazova/histological_images_generation/blob/master/images/pix2pix/img_1001_fake_B.png)
    ![alt-text-1](https://github.com/alsugiliazova/histological_images_generation/blob/master/images/pix2pix/img_1017_fake_B.png "title-1") ![alt-text-2](https://github.com/alsugiliazova/histological_images_generation/blob/master/images/pix2pix/img_1027_fake_B.png "title-2")
4. Генерация фона
    1. С использованием патчей из датасета (https://github.com/alsugiliazova/histological_images_generation/blob/master/background.ipynb)
    2. Нейросетевой метод генерации текстуры фона (https://github.com/NVIDIA/pix2pixHD)
![Иллюстрация к проекту](https://github.com/alsugiliazova/histological_images_generation/blob/master/images/full_image/21_synthesized_image.jpg)
![Иллюстрация к проекту](https://github.com/alsugiliazova/histological_images_generation/blob/master/images/full_image/22_synthesized_image.jpg)
