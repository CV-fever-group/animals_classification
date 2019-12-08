import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch
from Multi_Network import *
from torchvision.transforms import transforms
from PIL import Image
import pandas as pd
import random
from torch import optim
from torch.optim import lr_scheduler
import copy


ROOT_DIR = '../Dataset/'
TRAIN_DIR = 'train/'
VAL_DIR = 'val/'
TRAIN_ANNO = 'Multi_train_annotation.csv'
VAL_ANNO = 'Multi_val_annotation.csv'
CLASSES = ['Mammals', 'Birds']
SPECIES = ['rabbits', 'rats', 'chickens']


class MyDataset():

    def __init__(self, root_dir, annotations_file, transform=None):

        self.root_dir = root_dir
        self.annotations_file = annotations_file
        self.transform = transform

        if not os.path.isfile(self.annotations_file):
            print(self.annotations_file + 'does not exist!')
        self.file_info = pd.read_csv(annotations_file, index_col=0)
        self.size = len(self.file_info)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        image_path = self.file_info['path'][idx]
        if not os.path.isfile(image_path):
            print(image_path + '  does not exist!')
            return None

        image = Image.open(image_path).convert('RGB')
        label_class = int(self.file_info.iloc[idx]['classes'])
        label_species = int(self.file_info.iloc[idx]['species'])

        sample = {'image': image, 'classes': label_class,'species': label_species}
        if self.transform:
            sample['image'] = self.transform(image)
        return sample

train_transforms = transforms.Compose([transforms.Resize((500, 500)),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       ])
val_transforms = transforms.Compose([transforms.Resize((500, 500)),
                                     transforms.ToTensor()
                                     ])

train_dataset = MyDataset(root_dir= ROOT_DIR + TRAIN_DIR,
                          annotations_file= TRAIN_ANNO,
                          transform=train_transforms)

test_dataset = MyDataset(root_dir= ROOT_DIR + VAL_DIR,
                         annotations_file= VAL_ANNO,
                         transform=val_transforms)

train_loader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(dataset=test_dataset)
data_loaders = {'train': train_loader, 'val': test_loader}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

def visualize_dataset():
    print(len(train_dataset))
    idx = random.randint(0, len(train_dataset))
    sample = train_loader.dataset[idx]
    print(idx, sample['image'].shape, CLASSES[sample['classes']],SPECIES[sample['species']])
    img = sample['image']
    plt.imshow(transforms.ToPILImage()(img))
    plt.show()
visualize_dataset()


#已是两个分类
def train_model(model, criterion, optimizer, scheduler, num_epochs=50):
    Loss_list = {'train': [], 'val': []}
    Accuracy_list_species = {'train': [], 'val': []}
    Accuracy_list_classes = {'train': [], 'val': []}

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc_classes = 0.0
    best_acc_species = 0.0
    best_acc = 0.0
    best_loss = 10000

    #epoch循环训练
    for epoch in range(num_epochs):
        print('epoch {}/{}'.format(epoch,num_epochs - 1))
        print('-*' * 10)

        # 每个epoch都有train(训练)和val(测试)两个阶段
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            corrects_classes = 0
            corrects_species = 0

            for idx,data in enumerate(data_loaders[phase]):
                #将数据存在gpu或cpu上
                inputs = data['image'].to(device)
                labels_classes = data['classes'].to(device)
                labels_species = data['species'].to(device)
                optimizer.zero_grad()

                #训练阶段
                with torch.set_grad_enabled(phase == 'train'):
                    x_classes,x_species = model(inputs)
                    x_classes = x_classes.view(-1, 2)
                    x_species = x_species.view(-1, 3)
                    _, preds_classes = torch.max(x_classes, 1)
                    _, preds_species = torch.max(x_species, 1)
                    #计算训练误差
                    loss = (criterion(x_species, labels_species)+criterion(x_classes, labels_classes))/2

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                corrects_classes += torch.sum(preds_classes == labels_classes)
                corrects_species += torch.sum(preds_species == labels_species)

            epoch_loss = running_loss / len(data_loaders[phase].dataset)
            Loss_list[phase].append(epoch_loss)

            epoch_acc_classes = corrects_classes.double() / len(data_loaders[phase].dataset)
            epoch_acc_species = corrects_species.double() / len(data_loaders[phase].dataset)

            Accuracy_list_classes[phase].append(100 * epoch_acc_classes)
            Accuracy_list_species[phase].append(100 * epoch_acc_species)
            print('{} Loss: {:.4f}  Acc_classes: {:.2%}  Acc_species: {:.2%}'
                  .format(phase, epoch_loss,epoch_acc_classes,epoch_acc_species))

            #测试阶段
            if phase == 'val':
                #如果当前epoch下的准确率总体提高或者误差下降，则认为当下的模型最优
                if epoch_acc_classes +  epoch_acc_species > best_acc or epoch_loss < best_loss:
                    best_acc_classes = epoch_acc_classes
                    best_acc_species = epoch_acc_species
                    best_acc = best_acc_classes + best_acc_species
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    print('Best_model:  classes Acc: {:.2%},  species Acc: {:.2%}'
                          .format(best_acc_classes,best_acc_species))

    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), 'best_model.pt')
    print('Best_model:  classes Acc: {:.2%},  species Acc: {:.2%}'
          .format(best_acc_classes,best_acc_species))
    return model, Loss_list,Accuracy_list_classes,Accuracy_list_species

net = resnet18(2,3)
network = net.to(device)
optimizer = torch.optim.SGD(network.parameters(), lr=0.01, momentum=0.9)
criterion = nn.CrossEntropyLoss()
exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1) # Decay LR by a factor of 0.1 every 1 epochs
model, Loss_list, Accuracy_list_classes, Accuracy_list_species = train_model(network, criterion, optimizer, exp_lr_scheduler, num_epochs=100)


x = range(0, 100)
y1 = Loss_list["val"]
y2 = Loss_list["train"]
plt.figure(figsize=(18,14))
plt.subplot(211)
plt.plot(x, y1, color="r", linestyle="-", marker="o", linewidth=1, label="val")
plt.plot(x, y2, color="b", linestyle="-", marker="o", linewidth=1, label="train")
plt.legend()
plt.title('train and val loss vs. epoches')
plt.ylabel('loss')
#plt.savefig("train and val loss vs epoches.jpg")

plt.subplot(212)
y3 = Accuracy_list_classes["train"]
y4 = Accuracy_list_classes["val"]
y5 = Accuracy_list_species["train"]
y6 = Accuracy_list_species["val"]
plt.plot(x, y3, color="y", linestyle="-", marker=".", linewidth=1, label="train_class")
plt.plot(x, y4, color="g", linestyle="-", marker=".", linewidth=1, label="val_class")
plt.plot(x, y5, color="r", linestyle="-", marker=".", linewidth=1, label="train_specy")
plt.plot(x, y6, color="b", linestyle="-", marker=".", linewidth=1, label="val_specy")
plt.legend()
plt.title('train and val classes_acc & Species_acc vs. epoches')
plt.ylabel('accuracy')
#plt.savefig("train and val Classes_acc vs epoches.jpg")

visualize_model(model)