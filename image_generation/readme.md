using **Pillow** image library 
https://pillow.readthedocs.io/en/stable/

this generator includes such options:
* _gaussian blur_
* _noise_
* _brightness_
* _grayscale_
* _sepia_
* _negative_
* _rotate_
* _flip (left - right)_

user enters number of images, noise factor 1/2, brightness factor 1/2, sepia depth and rotate degress and then it starts to work with images (e. g. 28):
1. **_blur_** on every base image (by the following exapmle now its 56 images)
2. **_noise 1_** on every base image (by the following exapmle now its 84 images)
3. **_noise 2_** on every base image (by the following exapmle now its 112 images)
4. **_brightness 1_** on every base image (by the following exapmle now its 140 images)
5. **_brightness 2_** on every base image (by the following exapmle now its 168 images)
6. **_grayscale_** on every base image (by the following exapmle now its 196 images)
7. **_sepia_** on every base image (by the following exapmle now its 224 images)
8. **_negative_** on every base image (by the following exapmle now its 252 images)
9. **_rotating_** every image (**not only base!**) (by the following exapmle now its 504 images)
10. **_flipping_** every image (**not only base!**) (by the following exapmle now its 1008 images)
