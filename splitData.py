import os
import random
import shutil
from itertools import islice

outputFolderPath ="datasets/SplitData"
inputFolderPath ="datasets/all"
splitRatio = {"train":0.7,"val":0.2,"test":0.1}
classes = ["fake", "real"]
try:
    shutil.rmtree(outputFolderPath)
except OSError as e:
    os.mkdir(outputFolderPath)

# -------Directories to create-----
os.makedirs(f"{outputFolderPath}/train/images",exist_ok=True)
os.makedirs(f"{outputFolderPath}/train/labels",exist_ok=True)
os.makedirs(f"{outputFolderPath}/val/images",exist_ok=True)
os.makedirs(f"{outputFolderPath}/val/labels",exist_ok=True)
os.makedirs(f"{outputFolderPath}/test/images",exist_ok=True)
os.makedirs(f"{outputFolderPath}/test/labels",exist_ok=True)

# -------Get the Names------
listNames = os.listdir(inputFolderPath)
print(listNames)
uniqueNames = []
for name in listNames:
    uniqueNames.append(name.split('.')[0])
uniqueNames = list(set(uniqueNames))

# -------Shuffle-----
random.shuffle(uniqueNames)
print(uniqueNames)

# -------Find the number of images for each folder-----
lenData = len(uniqueNames)
print(f'Total Images: {lenData}')
lenTrain = int(lenData*splitRatio['train'])
lenVal = int(lenData*splitRatio['val'])
lenTest = int(lenData*splitRatio['test'])
print(f'Total Images:{lenData}\n Split: {lenTrain},{lenVal},{lenTest}')

#----Put the remaining images in Training-----
if lenData != lenTrain + lenVal + lenTest:
    remaining = lenData - (lenTrain + lenVal + lenTest)
    lenTrain = remaining

# -------Split the list-----
lengthToSplit = (lenTrain, lenVal, lenTest)
Input = iter(uniqueNames)
Output = [list(islice(Input, elem)) for elem in lengthToSplit]
print(f'Total Images:{lenData}\n Split: {len(Output[0])},{len(Output[1])},{len(Output[2])}')

# -------Copy the file-----
sequence = ['train', 'val', 'test']
for i, out in enumerate(Output):
    for fileName in out:
        img_path = f"{inputFolderPath}/{fileName}.jpg"
        label_path = f"{inputFolderPath}/{fileName}.txt"

        if os.path.exists(img_path) and os.path.exists(label_path):
            shutil.copy(img_path, f"{outputFolderPath}/{sequence[i]}/images/{fileName}.jpg")
            shutil.copy(label_path, f"{outputFolderPath}/{sequence[i]}/labels/{fileName}.txt")
        else:
            print(f"⚠️ Skipping {fileName}: missing .jpg or .txt file")

print("Split Process Completed......")

#--------Creating Data.yaml file------
dataYaml= (f'path: {outputFolderPath}\n\
train : ../train/images\n\
val : ../val/images\n\
test : ../test/images\n\
\n\
nc: {len(classes)}\n\
names: {classes}')

f= open(f"{outputFolderPath}/data.yaml", 'w')
f.write(dataYaml)
f.close()

print("Data.yaml file created...")