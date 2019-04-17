import numpy as np
import os
from skimage import io, transform
from pathlib import Path
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset


class Network(nn.Module):
    """Convolutional Neural Network with dropout"""
    def __init__(self, dropout_rate=0.0):
        super(Network, self).__init__()
        # Conv arguments were arbitrarily changed to require less model parameters
        self.conv1 = nn.Conv2d(3, 4, 8, stride=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(4, 10, 8, stride=2)
        # First linear argument is a function of image size and conv parameters
        self.fc1 = nn.Linear(10*12*12, 1000)
        self.fc2 = nn.Linear(1000, 500)
        self.fc3 = nn.Linear(500, 200)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 10*12*12)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


def predict_species(model_path, top_n):
    """
    Predict the species of the uploaded image
    Note: Requires the upload_img file having only one image
    :param model_path:
    :param top_n: number of top species to show
    :return: species prediction
    """
    net = Network(0.5)
    net.load_state_dict(torch.load(model_path))
    net.eval()

    # Load in the image in the necesarry form
    upload_root_dir = Path('uploaded_img')
    upload_img_lst = os.listdir('uploaded_img')
    upload_bird_dataset = BirdDataset(upload_root_dir, upload_img_lst,
                                      transform=transforms.Compose([
                                          Rescale(256),
                                          RandomCrop(224),
                                          ToTensor()
                                      ]))
    upload_tensor = upload_bird_dataset[0]['image'].float()
    upload_tensor = upload_tensor.reshape((1, 3, 224, 224))

    # Make the predictions
    predictions = net(upload_tensor)
    predictions = predictions.tolist()[0]

    most_likely = sorted(range(len(predictions)),
                         key=lambda i: predictions[i],
                         reverse=True)[:top_n]

    # TODO: Evalutate Predictions
    # return most_likely
    # print(most_likely)
    return [species for (species, i) in encoder.items() if i in most_likely]

# commit the sin of using the encoder as a global variable
pickle_in = open('models/encoder.pkl', 'rb')
encoder = pickle.load(pickle_in)

#####################################################
# Everything Below is for Data Transformation/Loading
#####################################################

class Rescale(object):
    """Rescale the image in a sample to a given size.
    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, species = sample['image'], sample['species']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)
        img = transform.resize(image, (new_h, new_w))

        return {'image': img, 'species': species}


class RandomCrop(object):
    """Crop randomly the image in a sample.
    Args:
        output_size (tuple or int): Desired output size. If int, square crop
        is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, species = sample['image'], sample['species']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                left: left + new_w]

        return {'image': image, 'species': species}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, species = sample['image'], sample['species']

        # swap color axis because
        # numpy iamge: HxWxC
        # torch image: CxHxW
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'species': species}


class BirdDataset(Dataset):
    def __init__(self, root_dir, img_lst, transform=None):
        """To be used when iterating over the
        files within the root directory"""
        self.root_dir = Path(root_dir)
        self.img_lst = img_lst
        self.transform = transform

    def create_encoder(self):
        """Convert all species names to integers"""
        encoding_dict = {}

        for img in self.img_lst:
            # relies on specific file naming pattern
            species = img.split('0')[0]

            if species not in encoding_dict.keys():
                encoding_dict[species] = len(encoding_dict)
        return encoding_dict

    def __len__(self):
        return len(self.img_lst)

    def __getitem__(self, idx: int) -> dict:
        """For any index in the range of img_lst
        it maps image to the image values
        and species to the type of bird"""
        img_path = self.root_dir / self.img_lst[idx]
        image = io.imread(img_path)

        # relies on specific file naming pattern
        species = self.img_lst[idx].split('0')[0]

        # naughty global variable: look to fix in future
        species = encoder[species]
        sample = {'image': image, 'species': species}

        if self.transform:
            sample = self.transform(sample)

        return sample

model_path = 'models/current_model.pt'
prediction = predict_species(model_path, 5)

# print(encoder.values())
# print(prediction)
print([species for (species, i) in encoder.items() if i in prediction])
# print(prediction)