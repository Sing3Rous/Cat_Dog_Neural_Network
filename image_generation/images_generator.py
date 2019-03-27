import random
from PIL import Image, ImageDraw, ImageFilter

def blurGeneration(numOfImages, currentImageNum, blurRadius):
    for num in range(numOfImages):
        image = Image.open(str(num + 1) + '.jpg')

        bluredImage = image.filter(ImageFilter.GaussianBlur(blurRadius))
        bluredImage.save(str(currentImageNum) + '.jpg')
        currentImageNum += 1

def noiseGeneration(numOfImages, currentImageNum, factor):
    for num in range(numOfImages):
        image = Image.open(str(num + 1) + '.jpg')
        draw = ImageDraw.Draw(image)
        width = image.size[0]
        height = image.size[1]
        pix = image.load()

        for i in range(width):
            for j in range(height):
                rand = random.randint(-factor, factor)
                a = pix[i, j][0] + rand
                b = pix[i, j][1] + rand
                c = pix[i, j][2] + rand
                if (a < 0): a = 0
                if (b < 0): b = 0
                if (c < 0): c = 0
                if (a > 255): a = 255
                if (b > 255): b = 255
                if (c > 255): c = 255
                draw.point((i, j), (a, b, c))

        image.save(str(currentImageNum) + '.jpg')
        currentImageNum += 1

def brightnessGeneration(numOfImages, currentImageNum, factor):
    for num in range(numOfImages):
        image = Image.open(str(num + 1) + '.jpg')
        draw = ImageDraw.Draw(image)
        width = image.size[0]
        height = image.size[1]
        pix = image.load()

        for i in range(width):
            for j in range(height):
                a = pix[i, j][0] + factor
                b = pix[i, j][1] + factor
                c = pix[i, j][2] + factor
                if (a < 0): a = 0
                if (b < 0): b = 0
                if (c < 0): c = 0
                if (a > 255): a = 255
                if (b > 255): b = 255
                if (c > 255): c = 255
                draw.point((i, j), (a, b, c))

        image.save(str(currentImageNum) + '.jpg')
        currentImageNum += 1

def grayscaleGeneration(numOfImages, currentImageNum):
    for num in range(numOfImages):
        image = Image.open(str(num + 1) + '.jpg')
        draw = ImageDraw.Draw(image)
        width = image.size[0]
        height = image.size[1]
        pix = image.load()

        for i in range(width):
            for j in range(height):
                a, b, c = pix[i, j]
                S = (a + b + c) // 3
                draw.point((i, j), (S, S, S))

        image.save(str(currentImageNum) + '.jpg')
        currentImageNum += 1

def sepiaGeneration(numOfImages, currentImageNum, depth):
    for num in range(numOfImages):
        image = Image.open(str(num + 1) + '.jpg')
        draw = ImageDraw.Draw(image)
        width = image.size[0]
        height = image.size[1]
        pix = image.load()

        for i in range(width):
            for j in range(height):
                a, b, c = pix[i, j]
                S = (a + b + c) // 3
                a = S + depth * 2
                b = S + depth
                c = S
                if (a > 255): a = 255
                if (b > 255): b = 255
                if (c > 255): c = 255
                draw.point((i, j), (a, b, c))

        image.save(str(currentImageNum) + '.jpg')
        currentImageNum += 1

def negativeGeneration(numOfImages, currentImageNum):
    for num in range(numOfImages):
        image = Image.open(str(num + 1) + '.jpg')
        draw = ImageDraw.Draw(image)
        width = image.size[0]
        height = image.size[1]
        pix = image.load()

        for i in range(width):
            for j in range(height):
                a, b, c = pix[i, j]
                draw.point((i, j), (255 - a, 255 - b, 255 - c))

        image.save(str(currentImageNum) + '.jpg')
        currentImageNum += 1

def rotateGeneration(numOfImages, currentImageNum, degree):
    for num in range(numOfImages):
        image = Image.open(str(num + 1) + '.jpg')

        rotatedImage = image.rotate(degree)
        rotatedImage.save(str(currentImageNum) + '.jpg')
        currentImageNum += 1

def flipGeneration(numOfImages, currentImageNum):
    for num in range(numOfImages):
        image = Image.open(str(num + 1) + '.jpg')

        flipedImage = image.transpose(Image.FLIP_LEFT_RIGHT)
        flipedImage.save(str(currentImageNum) + '.jpg')
        currentImageNum += 1

print("Enter number of images:")
numOfImages = int(input())
print("Enter blur radius: (e. g. 3)")
blurRadius = int(input())
print("Enter noise factor 1: (e. g. 33)")
noiseFactor1 = int(input())
print("Enter noise factor 2: (e. g. 100)")
noiseFactor2 = int(input())
print("Enter brightness factor 1: (e. g. 100)")
brightnessFactor1 = int(input())
print("Enter brightness factor 2: (e. g. -100)")
brightnessFactor2 = int(input())
print("Enter sepia depth: (e. g. 30)")
sepiaDepth = int(input())
print("Enter rotate degrees: (e. g. 45)")
rotateDegrees = int(input())
currentImageNum = numOfImages + 1
print("Please, wait...")

blurGeneration(numOfImages, currentImageNum, blurRadius)
currentImageNum += numOfImages
noiseGeneration(numOfImages, currentImageNum, noiseFactor1)
currentImageNum += numOfImages
noiseGeneration(numOfImages, currentImageNum, noiseFactor2)
currentImageNum += numOfImages
brightnessGeneration(numOfImages, currentImageNum, brightnessFactor1)
currentImageNum += numOfImages
brightnessGeneration(numOfImages, currentImageNum, brightnessFactor2)
currentImageNum += numOfImages
grayscaleGeneration(numOfImages, currentImageNum)
currentImageNum += numOfImages
sepiaGeneration(numOfImages, currentImageNum, sepiaDepth)
currentImageNum += numOfImages
negativeGeneration(numOfImages, currentImageNum)
currentImageNum += numOfImages
numOfImages = currentImageNum - 1
rotateGeneration(numOfImages, currentImageNum, rotateDegrees)
currentImageNum = (numOfImages * 2) + 1
numOfImages = currentImageNum - 1
flipGeneration(numOfImages, currentImageNum)

print("Generating is over.")
