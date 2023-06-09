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
    ![alt-text-1]([image1.png](https://github.com/alsugiliazova/histological_images_generation/blob/master/images/masks_gan/gan_img_32.jpg) "title-1") ![alt-text-2](image2.png "title-2")
2. Алгоритм расположения желез на фоне (https://github.com/alsugiliazova/histological_images_generation/blob/master/make_full_mask.ipynb)
3. Генерация текстуры отдельной железы по маске (https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
4. Генерация фона
    1. С использованием патчей из датасета (https://github.com/alsugiliazova/histological_images_generation/blob/master/background.ipynb)
    2. Нейросетевой метод генерации текстуры фона (https://github.com/NVIDIA/pix2pixHD)
