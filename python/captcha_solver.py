from load_config import read_model_params
import torch
import utils
import neural_network_utils
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import os

CONFIG_PATH = r'config/config.yaml'
PARAMS = read_model_params(CONFIG_PATH)
PATH = PARAMS.input_filepath
LETTERS_AMOUNT = PARAMS.letters_amount
ROTATION_DEGREE = PARAMS.image_rotation_degree
NUM_EPOCHS = PARAMS.num_epochs
MODEL_PARAMETERS_FILE_PATH = PARAMS.model_parameters_filepath

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"

DEVICE = torch.device(device)

if __name__ == '__main__':
    transform_to_tensor = transforms.Compose([transforms.ToTensor()])

    names_train, names_test = utils.get_train_test_names(PATH)
    sample_mean, sample_std = utils.get_mean_std(neural_network_utils.CaptchaDataset, names_train,
                                                 PATH, transform_to_tensor)

    transform_to_tensor_and_norm_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(sample_mean, sample_std),
        transforms.RandomAffine(degrees=5)])

    transform_to_tensor_and_norm_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(sample_mean, sample_std)])

    trainset = neural_network_utils.CaptchaDataset(names_train, PATH, transform_to_tensor_and_norm_train)
    testset = neural_network_utils.CaptchaDataset(names_test, PATH, transform_to_tensor_and_norm_test,
                                                  verbose=True)

    batch_size = 24
    num_workers = 2

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=num_workers)

    testloader = torch.utils.data.DataLoader(testset, batch_size=1,
                                             shuffle=False, num_workers=num_workers)

    net = neural_network_utils.CaptchaSolverNet(letters_amount=LETTERS_AMOUNT)
    net = net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=1e-4, momentum=0.9)

    net = neural_network_utils.train_model(net, criterion, optimizer, trainloader,
                                           device=DEVICE, num_epochs=NUM_EPOCHS, letters_amount=LETTERS_AMOUNT)

    batch_size = 1
    num_workers = 1

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=num_workers)

    train_characters_error_rate = utils.character_error_rate(net, trainloader, device=DEVICE,
                                                             letters_amount=LETTERS_AMOUNT,
                                                             captcha_dataset=neural_network_utils.CaptchaDataset)

    test_characters_error_rate = utils.character_error_rate(net, testloader, device=DEVICE,
                                                            letters_amount=LETTERS_AMOUNT,
                                                            captcha_dataset=neural_network_utils.CaptchaDataset)

    print(f'Train characters error rate = {train_characters_error_rate}')
    print(f'Test characters error rate = {test_characters_error_rate}')
    torch.save(net.state_dict(), os.path.abspath(os.path.join(MODEL_PARAMETERS_FILE_PATH, 'result')))
