import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import string
import torch.nn as nn
import torch.nn.functional as functional


def train_model(net, criterion, optimizer, trainloader, device, letters_amount, num_epochs=10):
    net.train(True)
    for epoch in range(num_epochs):
        loss = 0
        print(f"{epoch=}")
        for i, data in enumerate(trainloader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs).to(device)
            loss = 0
            for letter_num in range(letters_amount):
                loss += criterion(outputs[:, letter_num], labels[letter_num])
            loss.backward()
            optimizer.step()
        print(f'{loss=}')
    print('Finished Training')
    return net


class CaptchaDataset(Dataset):
    def __init__(self, dataframe, root_dir, transform=None, verbose=False):
        self.dataframe = dataframe
        self.root_dir = root_dir
        self.transform = transform
        self.verbose = verbose
        self.symbols = list(map(str, range(10))) + list(string.ascii_lowercase)
        self.symbol_to_id = {key: value for value, key in enumerate(self.symbols)}
        self.id_to_symbol = {value: key for value, key in enumerate(self.symbols)}

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        img_name = os.path.join(self.root_dir, self.dataframe['filename'][index])
        image = Image.open(img_name).convert('L')

        label = torch.tensor(np.array(
            [self.symbol_to_id[idx] for idx in self.dataframe['label'][index]]))

        if self.transform:
            image = self.transform(image)

        return image, label


class CaptchaSolverNet(nn.Module):
    def __init__(self, letters_amount):
        super(CaptchaSolverNet, self).__init__()
        self.letters_amount = letters_amount
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 128, kernel_size=(3, 6), padding=(1, 1))
        self.conv2 = nn.Conv2d(128, 64, kernel_size=(3, 6), padding=(1, 1))
        self.conv3 = nn.Conv2d(64, 32, kernel_size=(3, 6), padding=(1, 1))

        self.pool = nn.MaxPool2d(kernel_size=(2, 2))
        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(32)
        self.bn4 = nn.BatchNorm1d(512)

        # Full connected layers for each letter
        self.partials1 = torch.nn.ModuleList(tuple((
            nn.Linear(in_features=4224, out_features=512)
            for _ in range(letters_amount))))

        self.partials2 = torch.nn.ModuleList(tuple((
            nn.Linear(in_features=512, out_features=36)
            for _ in range(letters_amount))))

        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.conv1(x)
        x = functional.relu(x)
        x = self.pool(x)
        x = self.bn1(x)

        x = self.conv2(x)
        x = functional.relu(x)
        x = self.pool(x)
        x = self.bn2(x)

        x = self.conv3(x)
        x = functional.relu(x)
        x = self.pool(x)
        x = self.bn3(x)

        x = torch.flatten(x, 1)
        outputs = []
        for letter_num in range(self.letters_amount):
            letter = self.partials1[letter_num](x)
            letter = functional.relu(letter)
            letter = self.bn4(letter)
            letter = self.dropout(letter)

            letter = self.partials2[letter_num](letter)
            letter = functional.relu(letter)

            outputs.append(letter)
        result = torch.stack(outputs)
        return result
