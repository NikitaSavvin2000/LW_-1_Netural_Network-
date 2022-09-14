import numpy as np
from keras.datasets import mnist # импортирует датасет рукописных цифр
import matplotlib.pyplot as plt
from tqdm import tqdm # отображает индикатор выполнения
from torchvision import transforms # возвращает модифицированную версию изображения (тензор)
import torch.nn as nn # создает слои нейросети
from torch.utils.data import DataLoader,Dataset #обработка, загрузка
import torch
import torch.optim as optim # методы обновления весов исмещений
from torch.autograd import Variable #автодеференциирование (оборачивает тензор)


def add_noise(img,noise_type="gaussian"):
    # на входе изображение и тип обработки (по умолчанию Гаусса)
    row,col = img.shape 
    # img.shape имеет тип tuple (кортеж),и содержит 2 элемента 
    # (к-во строк и столбцов в img, т.е. (28,28)), которые присваиваются переменным row и col
    img = img.astype(np.float32) 
    # конвертируем входные данные изображения 
    # в 8-байтовый вещественный формат
    if noise_type == "gaussian":
        mean = -5.9 
        # математическое ожидание, 
        var = 35 
        # дисперсия, D
        sigma = var ** .5 
        # среднеквадратическое отклонение, , // здесь **.5 – корень 2-й степени
        noise = np.random.normal(mean, sigma, (row,col))
        # создание массива noise, размером row х col,заполненного
        # случайными числами, с плавающей точкой, из одномерного
        # нормального распределения Гаусса, с заданными  и 
        noise = noise.reshape(row,col)
        # контроль размеров массива 
        img = img + noise 
        # к каждому элементу массива img прибавляется элемент массива 
        # nose с идентичными номерами позиции
        return img
    if noise_type == "speckle":
        noise = np.random.randn(row,col) 
        # массив случайны чисел, размера row х col, отличный от массива 
        # в первом методе тем, что параметры распределения СВ 
        # заданы поумолчанию, т.е.  = 0 и  =1
        noise = noise.reshape(row,col)
        img = img + img * noise 
        # к каждому элементу массива img прибавляется его произведение 
        # с элементом массива nose с идентичными номерами позиции return img
        return img
    
(xtrain,ytrain),(xtest,ytest) = mnist.load_data()

print("No of training datapoints: {}\nNo of Test datapoints:{}".format(len(xtrain),len(xtest)))
noises=["gaussian","speckle"]
noise_ct=0
noise_id=0
traindata=np.zeros((60000,28,28))

for idx in tqdm(range(len(xtrain))):
    if noise_ct<(len(xtrain)/2):
        32
        noise_ct+=1
        traindata[idx] = add_noise(xtrain[idx], noise_type = noises[noise_id])
    else:
        print("\n{} noise addition completed to imag-es".format(noises[noise_id]))
    noise_id += 1
    noise_ct = 0
    print("\n{} noise addition completed to imag-es".format(noises[noise_id]))
    noise_ct = 0
    noise_id = 0
testdata = np.zeros((10000,28,28))
for idx in tqdm(range(len(xtest))):
if noise_ct < (len(xtest)/2):
noise_ct += 1
testdata[idx] = add_noise(xtest[idx], noise_type = noises[noise_id])
else:
print("\n{} noise addition completed to imag-es".format(noises[noise_id]))
noise_id += 1
noise_ct = 0
print("\n{} noise addition completed to imag-es".format(noises[noise_id]))
f, axes = plt.subplots(2,2)
f.set_figwidth(7)
f.set_figheight(7)
#showing images with gaussian noise
axes[0,0].imshow(xtrain[1100],cmap="gray")
axes[0,0].set_title("Original Image")
axes[1,0].imshow(traindata[1100],cmap='gray')
axes[1,0].set_title("Noised Image")
#showing images with speckle noise
axes[0,1].imshow(xtrain[31000],cmap='gray')
axes[0,1].set_title("Original Image")
axes[1,1].imshow(traindata[31000],cmap="gray")
axes[1,1].set_title("Noised Image")


class noisedDataset(Dataset):
    
    def __init__(self, datasetnoised, datasetclean, labels, transform):
        self.noise = datasetnoised # 1)
        self.clean = datasetclean # 2)
        self.labels = labels # 3)
        self.transform = transform
        
    def __len__(self):
        return len(self.noise)
    
    def __getitem__(self,idx):
        xNoise = self.noise[idx] # 1)
        xClean = self.clean[idx] # 2)
        y = self.labels[idx] # 3)
        if self.transform != None:
            xNoise = self.transform(xNoise) # 1)
            xClean = self.transform(xClean) # 2)
        return (xNoise,xClean,y)
    
    tsfms = transforms.Compose([transforms.ToTensor()])
    trainset = noisedDataset(traindata,xtrain,ytrain,tsfms)
    testset = noisedDataset(testdata,xtest,ytest,tsfms)
    batch_size = 32
    trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
    testloader = DataLoader(testset, batch_size=1, shuffle=True)


class denoising_model(nn.Module):

    def __init__(self):
        super(denoising_model,self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=7, stride=3),
            # padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=32),
            #nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(0.2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=64),
            nn.Dropout(0.2)
        )
        self.layer02 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size = 3, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features = 32),
            34
            #nn.Upsample(size = 13, mode='bilinear'),
            nn.Dropout(0.2)
        )
        self.layer01 = nn.Sequential(
            nn.ConvTranspose2d(32, 1, kernel_size=7, stride=3),
            nn.Sigmoid()
        )
        #self.drop_out = nn.Dropout()
        #self.fc1 = nn.Linear(7 * 7 * 64, 1000)
        #self.fc2 = nn.Linear(1000, 10)
        # layer1-2: датасет пропускается через кодер
        # layer01-02: датасет пропускается через декодер
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer02(out)
        out = self.layer01(out)
        return out

    if torch.cuda.is_available() == True:
        device = "cuda:0"
    else:
        device ="cpu"
        model=denoising_model().to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr = 0.003)
        epochs = 120
        l = len(trainloader)
        losslist = list()
        epochloss = 0
        running_loss = 0
        KOst = 0.0063
        
    for epoch in range(epochs):
        print("Entering Epoch: ",epoch)
        
    for i1, (dirty,clean,label) in tqdm(enumerate(trainloader)):
        dirty = dirty.view(dirty.size(0), 1, 28, 28).type(torch.FloatTensor)
        clean = clean.view(clean.size(0), 1, 28, 28).type(torch.FloatTensor)
        dirty,clean = dirty.to(device),clean.to(device)
        
    #-----------------Forward Pass---------------------
    output = model.forward(dirty)
    loss = criterion(output,clean)
    #-----------------Backward Pass--------------------

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    running_loss += loss.item()
    epochloss += loss.item()

    #-----------------Log-------------------------------
    sr_loss = running_loss/l
    losslist.append(sr_loss)
    running_loss=0
    print("======> epoch: {}/{}, Loss:{}".format(epoch,epochs, sr_loss))
    
    if sr_loss <= KOst: break
    
    plt.plot(range(len(losslist)), losslist)
    f, axes = plt.subplots(6,3,figsize = (20, 20))
    axes[0, 0].set_title("Original Image")
    axes[0, 1].set_title("Dirty Image")
    axes[0, 2].set_title("Cleaned Image")
    test_imgs = np.random.randint(0, 10000, size=6)
    print(test_imgs)
    
    for idx in range((6)):
        dirty = testset[test_imgs[idx]][0]
        clean = testset[test_imgs[idx]][1]
        label = testset[test_imgs[idx]][2]
        dirty = dirty.view(dirty.size(0), 1, 28, 28).type(torch.FloatTensor)
        dirty = dirty.to(device)
        output = model(dirty)
        output = output.view(1, 28, 28)
        output = output.permute(1, 2, 0).squeeze(2)
        output=output.detach().cpu().numpy
        dirty=dirty.view(1,28,28)
        dirty=dirty.permute(1,2,0).squeeze(2)
        dirty=dirty.detach().cpu().numpy()
        clean=clean.permute(1,2,0).squeeze(2)
        clean=clean.detach().cpu().numpy()
        axes[idx,0].imshow(clean,cmap="gray")
        axes[idx,1].imshow(dirty,cmap="gray")
        axes[idx,2].imshow(output,cmap="gray")
        
    from PIL import Image
    
    rw, cl = 28, 28
    n = 6
    name = []
    AryIzo = np.zeros((n, rw, cl))
    for k in range(n):
        name.append ("Izo" + str(k) + ".jpg")
        izo = Image.open(name[k])
        pix = izo.load()
        for i in range(rw):
            for j in range(cl):
                AryIzo[k,i,j] = (pix[i,j][0]+ pix[i,j][1] + pix[i,j][2])/3
            
    f, axes = plt.subplots(6, 3, figsize = (20, 20))
    axes[0, 0].set_title("Original Image")
    axes[0, 1].set_title("Dirty Image")
    axes[0, 2].set_title("Cleaned Image")
    test_imgs = np.random.randint(0, 10000, size = 6)
    
    for idx in range((6)):
        dirty = tsfms(np.transpose(AryIzo[idx]))
        clean = testset[test_imgs[idx]][1]
        label = testset[test_imgs[idx]][2]
        dirty = dirty.view(dirty.size(0), 1, 28, 28).type(torch.FloatTensor)
        dirty = dirty.to(device)
        output = model(dirty)
        output = output.view(1, 28, 28)
        output = output.permute(1, 2, 0).squeeze(2)
        output = output.detach().cpu().numpy()
        dirty = dirty.view(1, 28, 28)
        dirty = dirty.permute(1, 2, 0).squeeze(2)
        dirty = dirty.detach().cpu().numpy()
        clean = clean.permute(1, 2, 0).squeeze(2)
        clean = clean.detach().cpu().numpy()
    axes[idx, 0].imshow(clean,cmap = "gray")
    axes[idx, 1].imshow(dirty,cmap = "gray")
    [idx, 2].imshow(output,cmap = "gray")
    
import os

PATH = "PATHS101"
#"username/directory/lstmmodelgpu.pth"
torch.save(model.state_dict(),PATH)