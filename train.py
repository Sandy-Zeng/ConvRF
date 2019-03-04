from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from model import vggRF
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--mode', default='F',
                        help='convolution mode of VGGRF')
    parser.add_argument('--depth', type=int, default=3,
                        help='depth of VGG')
    parser.add_argument('--log', default='./log.txt',help='path to log file')
    args = parser.parse_args()
    use_cuda = torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    # kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=args.test_batch_size, shuffle=True)
    print('load data')

    model = vggRF(mode=args.mode,depth=args.depth).to(device)
    print('initialize model')

    classifier = RandomForestClassifier(n_estimators=100, min_samples_leaf=5,
                                             max_features='auto')

    #train the model
    training_size = len(train_loader.dataset)
    n_feature = 0
    with torch.no_grad():
        print('-'*20+'training'+'-'*20)
        feature_list = []
        target_list = []
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            feature = model.conv(x=data,conv=True)
            n_feature = feature.shape[-1]
            if len(feature_list) == 0:
                feature_list = feature.cpu().numpy()
                target_list = target.cpu().numpy()
            else:
                feature_list = np.concatenate((feature_list, feature.cpu().numpy()), axis=0)
                target_list = np.concatenate((target_list,target.cpu().numpy()),axis=0)
            # target_list += [target.cpu().numpy()]
        print(feature_list.shape)
        print(len(target_list))
        x = np.reshape(feature_list,(training_size,n_feature))
        y = np.reshape(target_list,(training_size,))
        classifier.fit(x,y)

    with torch.no_grad():
        print('-' * 20 + 'testing' + '-' * 20)
        running_corrects = 0.
        feature_list = []
        target_list = []
        test_size = len(test_loader.dataset)
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            feature = model.conv(x=data,conv=True)
            n_feature = feature.shape[-1]
            if len(feature_list) == 0:
                feature_list = feature.cpu().numpy()
                target_list = target.cpu().numpy()
            else:
                feature_list = np.concatenate((feature_list, feature.cpu().numpy()), axis=0)
                target_list = np.concatenate((target_list, target.cpu().numpy()), axis=0)
        print(feature_list.shape)
        print(len(target_list))
        test_x = np.reshape(feature_list, (test_size, n_feature))
        test_y = np.reshape(target_list, (test_size,))
        preds = classifier.predict(test_x)
        score = accuracy_score(test_y,preds)
        print(score)
        # for data, target in test_loader:
        #     data, target = data.to(device), target.to(device)
        #     preds = model.predict(data,conv=True)
        #     # print(preds)
        #     running_corrects += torch.sum(preds == target)
        #     score = accuracy_score(target,preds)
        #     print(score)
    # running_corrects = float(running_corrects)
    # print('Corrects',running_corrects)
    # print(len(test_loader.dataset))
    # acc = (running_corrects/len(test_loader.dataset))*100
    # print('Test Accuracy: %.2f'%(acc))

    f = open(args.log,'a')
    f.write(str(score))
    f.write('\n')
    f.close()

    # if (args.save_model):
    #     torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()