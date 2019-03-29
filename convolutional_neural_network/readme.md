for correct work your **data folder** must include 2 subfolders:
1. **generated** (e.g. _C:\Users\singe\Desktop\training\generated_)
    * this subfolder must include **pickles** subfolder with files **XTrain.pickle** and **yTrain.pickle**
2. **test** (e.g. _C:\Users\singe\Desktop\training\test_)
    * this subfolder must include **pickles** subfolder with files **XTest.pickle** and **yTest.pickle**

(for making .pickle files use _**image_preparator.py**_)

at training proccess there will be created **logs** subfolder 

at current moment the best result (**~93.48%**) was shown with such parameters:
* **3** convolutional layers
* **64** filters
* kernel size = **3**
* pool size = **2**
* **0** dense layers
* batch size = **8**
* **20** epochs