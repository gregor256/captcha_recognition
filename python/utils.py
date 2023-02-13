import os
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def get_train_test_names(root_dir, test_size=0.2):
    """
    Create train and test dataset with filenames and labels
    """
    picture_names = [[picture_name, picture_name[: -4]] for picture_name
                     in os.listdir(root_dir)]

    names_dataframe = pd.DataFrame(picture_names, columns=['filename', 'label'])
    names_train, names_test = train_test_split(names_dataframe,
                                               test_size=test_size,
                                               random_state=42)
    return names_train.reset_index(drop=True), names_test.reset_index(drop=True)


def get_mean_std(dataset_class, names_train, path, transform):
    """
    Get mean and std over all training dataset
    """
    dataset = dataset_class(names_train, path, transform)
    train_tensors = [dataset[i][0] for i in range(len(names_train))]
    train_tensors_stack = torch.stack(train_tensors)
    return train_tensors_stack.mean(), train_tensors_stack.std()


def tensor_to_word(detached_tensor, captcha_dataset):
    dataset = captcha_dataset(pd.DataFrame(), '')
    symbols_array = tuple(map(lambda x: dataset.id_to_symbol[int(x)],
                              tuple(detached_tensor)))
    return ''.join(map(str, symbols_array))


def character_error_rate(net, loader, captcha_dataset, letters_amount, device, verbose=False):
    """
    Model quality evaluation
    enable verbose to see word pairs: errors in recognition
    """
    net.train(False)
    with torch.no_grad():
        errors = 0
        for data in loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs[:, 0, :], 1)
            predicted, labels = predicted.detach().cpu().numpy(), labels.detach().cpu().numpy()[0]

            errors += np.count_nonzero(predicted - labels)
            if verbose:
                print("error:", tensor_to_word(predicted, captcha_dataset),
                      tensor_to_word(labels, captcha_dataset))

    return errors / (letters_amount * len(loader))
