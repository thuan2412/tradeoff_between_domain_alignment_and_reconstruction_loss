import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
import pdb

import misc
import wide_resnet


class MLP(nn.Module):
    """Just an MLP"""
    def __init__(self, n_inputs, n_outputs, hparams):
        super(MLP, self).__init__()
        self.input = nn.Linear(n_inputs, hparams['mlp_width'])
        self.dropout = nn.Dropout(hparams['mlp_dropout'])
        self.hiddens = nn.ModuleList([
            nn.Linear(hparams['mlp_width'], hparams['mlp_width'])
            for _ in range(hparams['mlp_depth']-2)])
        self.output = nn.Linear(hparams['mlp_width'], n_outputs)
        self.n_outputs = n_outputs

    def forward(self, x):
        x = self.input(x)
        x = self.dropout(x)
        x = F.relu(x)
        for hidden in self.hiddens:
            x = hidden(x)
            x = self.dropout(x)
            x = F.relu(x)
        #### thuan added
        statistics = self.output(x)
        mu = statistics[:,:10] ## self.K is the size of the last layer
        std = F.softplus(statistics[:,:10]-5,beta=1) ### 10 classes, mean =5
        x = self.output(x)
        return mu, std, x


class Identity(nn.Module):
    def __init__(self, input_shape):
        super(Identity, self).__init__()
        # input_shape: (3, 28, 28)
        self.n_outputs = input_shape[0] * input_shape[1] * input_shape[2]

    def forward(self, x):
        bs = x.shape[0]
        return x.reshape(bs, -1)


class ResNet(torch.nn.Module):
    """ResNet with the softmax chopped off and the batchnorm frozen"""
    def __init__(self, input_shape, hparams):
        super(ResNet, self).__init__()
        if hparams['resnet18']:
            self.network = torchvision.models.resnet18(pretrained=True)
            self.n_outputs = 512
        else:
            self.network = torchvision.models.resnet50(pretrained=True)
            self.n_outputs = 2048

        # adapt number of channels
        nc = input_shape[0]
        if nc != 3:
            tmp = self.network.conv1.weight.data.clone()
            self.network.conv1 = nn.Conv2d(
                nc, 64, kernel_size=(7, 7),
                stride=(2, 2), padding=(3, 3), bias=False)

            for i in range(nc):
                self.network.conv1.weight.data[:, i, :, :] = tmp[:, i % 3, :, :]

        self.freeze_bn()
        self.hparams = hparams
        self.dropout = nn.Dropout(hparams['resnet_dropout'])

        # todo
        # del self.network.fc

    def forward(self, x):
        """Encode x into a feature vector of size n_outputs."""
        x = self.network.conv1(x)
        x = self.network.bn1(x)
        x = self.network.relu(x)
        x = self.network.maxpool(x)
        x = self.network.layer1(x)
        x = self.network.layer2(x)
        x = self.network.layer3(x)
        x = self.network.layer4(x)
        x = self.network.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return x

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super().train(mode)
        self.freeze_bn()

    def freeze_bn(self):
        for m in self.network.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()


class SqueezeLastTwo(nn.Module):
    """A module which squeezes the last two dimensions, ordinary squeeze can be a problem for batch size 1"""
    def __init__(self):
        super(SqueezeLastTwo, self).__init__()

    def forward(self, x):
        return x.view(x.shape[0], x.shape[1])

######### Thuan
######## this is the network for CS-MNIST
class xylMNIST_CNN(nn.Module):
    n_outputs = 128

    def __init__(self, input_shape):
        super(xylMNIST_CNN, self).__init__()
        # print("use xylMNIST_CNN")
        self.conv1 = nn.Conv2d(input_shape[0], 64, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 128, 3, 1, padding=1)
        #self.conv4 = nn.Conv2d(128, 128, 3, 1, padding=1)

        # I change from GroupNorm to BatchNorm
        self.bn0 = nn.BatchNorm2d(64)
        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(128)
        #self.bn3 = nn.BatchNorm2d(128)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.squeezeLastTwo = SqueezeLastTwo()

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn0(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.bn1(x)

        x = self.conv3(x)
        t = x
        x = F.relu(x)
        x = self.bn2(x)

        x = self.avgpool(x)

        x = self.squeezeLastTwo(x)

        # statistics = x
        # mu = statistics[:,:10] ## self.K is the size of the last layer
        # std = F.softplus(statistics[:,:10]-5,beta=1) ### 10 classes, mean =5
        
        return x  # (256, 128)




class MNIST_CNN(nn.Module):
    """
    Hand-tuned architecture for MNIST.
    Weirdness I've noticed so far with this architecture:
    - adding a linear layer after the mean-pool in features hurts
        RotatedMNIST-100 generalization severely.
    """
    n_outputs = 128

    def __init__(self, input_shape):
        super(MNIST_CNN, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 64, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 128, 3, 1, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, 1, padding=1)

        self.bn0 = nn.GroupNorm(8, 64)
        # self.bn0 = nn.BatchNorm2d(64)
        self.bn1 = nn.GroupNorm(8, 128)
        self.bn2 = nn.GroupNorm(8, 128)
        self.bn3 = nn.GroupNorm(8, 128)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn0(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.bn1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.bn2(x)

        x = self.conv4(x)
        x = F.relu(x)
        x = self.bn3(x)

        x = x.mean(dim=(2, 3))
        return x


class ContextNet(nn.Module):
    def __init__(self, input_shape):
        super(ContextNet, self).__init__()

        # Keep same dimensions
        padding = (5 - 1) // 2
        self.context_net = nn.Sequential(
            nn.Conv2d(input_shape[0], 64, 5, padding=padding),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 5, padding=padding),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 1, 5, padding=padding),
        )

    def forward(self, x):
        return self.context_net(x)


def Featurizer(input_shape, hparams):
    """Auto-select an appropriate featurizer for the given input shape."""
    if len(input_shape) == 1:
        return MLP(input_shape[0], 128, hparams)
    elif input_shape[1:3] == (28, 28):
        # todo
        if hparams["xylnn"]:
            return xylMNIST_CNN(input_shape)

            # return Identity(input_shape)
        return MNIST_CNN(input_shape)

    elif input_shape[1:3] == (32, 32):
        return wide_resnet.Wide_ResNet(input_shape, 16, 2, 0.)

    elif input_shape[1:3] == (64, 64):
        # print("Use Identity!!!")
        # return Identity(input_shape)
        # todo
        return wide_resnet.Wide_ResNet(input_shape, 16, 2, 0.)

    elif input_shape[1:3] == (224, 224):
        return ResNet(input_shape, hparams)
    else:
        print(input_shape)
        raise NotImplementedError
        
##### decoder network for small datasets ( Colored-MNIST  and Rorated-MNIST   )   
class Decoder(nn.Module):
    """
    From paper in search of loss in DG
    Hand-tuned architecture for MNIST.
    decoder output is 27 * 27
    """
    n_outputs = 128

    # input_shape should be the shape of one element
    def __init__(self, input_shape= 28):
        super(Decoder, self).__init__()
        self.deConv = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=(14, 14)),
            nn.ConvTranspose2d(128, 128, 3, 1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 128, 3, 1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 3, 1, padding=1)
        )
    def decoder(self, z):
        x = z.view(-1,128,1,1)
        x = self.deConv(x)
        return x

    def forward(self, x):
        out = self.decoder(x)
        return out