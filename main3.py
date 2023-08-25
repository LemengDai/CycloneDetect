#!/usr/bin/env python
# coding: utf-8

# In[4]:


# get_ipython().system('pip3 install torch torchvision torchmetrics')


# In[5]:


import numpy as np


# In[6]:


import torch, torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch import nn
from torch.optim import Adam, SGD
from collections import OrderedDict
from torch.nn import functional as F


# Model Building

# In[7]:


class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """
        super(ConvLSTMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias
        
        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim, out_channels=4 * self.hidden_dim, kernel_size=self.kernel_size, padding=self.padding, bias=self.bias)
        
    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        
        combined = torch.cat([input_tensor, h_cur], dim=1)
        
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        
        return h_next, c_next
    
    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, width, height, device=self.conv.weight.device), torch.zeros(batch_size, self.hidden_dim, width, height, device=self.conv.weight.device))


# In[8]:


class ConvLSTM(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers, batch_first=False, bias=True, return_all_layers=False):
        """
        Parameters:
            input_dim: Number of channels in input
            hidden_dim: Number of hidden channels
            kernel_size: Size of kernel in convolutions
            num_layers: Number of LSTM layers stacked on each other
            batch_first: Whether or not dimension 0 is the batch or not
            bias: Bias or no bias in Convolution
            return_all_layers: Return the list of computations for all layers
        Input:
            A tensor of size B, T, W, H, C or T, B, W, H, C
        Output:
            A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
                0 - layer_output_list is the list of lists of length T of each output
                1 - last_state_list is the list of last states
                        each element of the list is a tuple (h, c) for hidden state and memory
        Example:
            >> x = torch.rand((32, 10, 64, 128, 128))
            >> convlstm = ConvLSTM(64, 16, 3, 1, True, True, False)
            >> _, last_states = convlstm(x)
            >> h = last_states[0][0]  # 0 for layer index, 0 for h index
        """
        super(ConvLSTM, self).__init__()
        
        self._check_kernel_size_consistency(kernel_size)
        
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')
            
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers
        
        cell_list = []
        for i in range(self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]
            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim, hidden_dim=self.hidden_dim[i], kernel_size=self.kernel_size[i], bias=self.bias))
        self.cell_list = nn.ModuleList(cell_list)
        
    def forward(self, input_tensor, hidden_state=None):
        """
        Parameters
        ----------
        input_tensor:
            5-D Tensor either of shape (t, b, w, h, c) or (b, t, w, h, c)

        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)
        
        # reshape the input tensor into (b, t, c, w, h)
        input_tensor = input_tensor.permute(0, 1, 4, 2, 3)
        b,_,_,w,h = input_tensor.size()
        
        if hidden_state is None:
            hidden_state = self._init_hidden(batch_size=b, image_size=(h, w))
            
        layer_output_list = []
        last_state_list = []
        
        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor
        
        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :], cur_state=[h, c])
                output_inner.append(h)
                
            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output
            
            layer_output_list.append(layer_output)
            last_state_list.append([h, c])
            
        # give the last layer
        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1]
            last_state_list = last_state_list[-1:]
        
        return layer_output_list
    
    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states
    
    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')
    
    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param
        


# In[9]:


INPUT_DIM = 3
OUTPUT_DIM = 1
BATCH_SIZE = 16
kernel_size = (5, 5)
# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# print(f"Using {device} device")


# In[10]:


model1 = ConvLSTM(input_dim=INPUT_DIM,
                 hidden_dim=[6, 12, OUTPUT_DIM],
                 kernel_size=(3, 3),
                 num_layers=3,
                 batch_first=True,
                 bias=True,
                 return_all_layers=False)


# In[11]:


class Seq2Seq(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, num_layers):
        super(Seq2Seq, self).__init__()
        self.sequential = nn.Sequential()
        self.sequential.add_module("conlstm1", ConvLSTM(input_dim, 64, kernel_size, 1))
        self.sequential.add_module("batchnorm1", nn.BatchNorm3d(num_features=64))
        
        for i in range(2, num_layers + 1):
            self.sequential.add_module(
                f"convlstm{i}", ConvLSTM(64, 64, kernel_size, 1)
                )
            self.sequential.add_module(
                f"batchnorm1{i}", nn.BatchNorm3d(num_features=64)
            )
            
        self.conv = nn.Conv2d(in_channels=64, out_channels=output_dim, kernel_size=kernel_size)
        
    def forward(self, input_tensor):
        output = self.sequential(input_tensor)
        output = self.conv(output[:,:,-1])
        return nn.Sigmoid(output)
        


# In[12]:


model2 = Seq2Seq(input_dim=INPUT_DIM, output_dim=OUTPUT_DIM,kernel_size=(5, 5), num_layers=2)


# In[13]:


convlstm_encoder_params = [
    [
        OrderedDict({'conv1_leaky_1': [INPUT_DIM, 8, 7]}),
        OrderedDict({'conv2_leaky_1': [64, 128, 5]}),
        OrderedDict({'conv3_leaky_1': [192, 192, 3]}),
    ],

    [
        ConvLSTM(8, 64, kernel_size, 1),
        ConvLSTM(128, 192, kernel_size, 1),
        ConvLSTM(192, 192, kernel_size, 1),
    ]
]

convlstm_decoder_params = [
    [
        OrderedDict({'deconv1_leaky_1': [192, 192, 4]}),
        OrderedDict({'deconv2_leaky_1': [128, 64, 5]}),
        OrderedDict({
            'deconv3_leaky_1': [64, 8, 7],
            'conv3_leaky_2': [8, 8, 3],
            'conv3_3': [8, OUTPUT_DIM, 1]
        }),
    ],

    [
        ConvLSTM(192, 128, kernel_size, 1),
        ConvLSTM(128, 192, kernel_size, 1),
        ConvLSTM(64, 8, kernel_size, 1),
    ]
]


# In[14]:


def make_layers(block):
    layers = []
    for layer_name, v in block.items():
        if 'pool' in layer_name:
            layer = nn.MaxPool2d(kernel_size=v[0], stride=v[1],
                                    padding=v[2])
            layers.append((layer_name, layer))
        elif 'deconv' in layer_name:
            transposeConv2d = nn.ConvTranspose2d(in_channels=v[0], out_channels=v[1],
                                                 kernel_size=v[2])
            layers.append((layer_name, transposeConv2d))
            if 'relu' in layer_name:
                layers.append(('relu_' + layer_name, nn.ReLU(inplace=True)))
            elif 'leaky' in layer_name:
                layers.append(('leaky_' + layer_name, nn.LeakyReLU(negative_slope=0.2, inplace=True)))
        elif 'conv' in layer_name:
            conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1],
                               kernel_size=v[2])
            layers.append((layer_name, conv2d))
            if 'relu' in layer_name:
                layers.append(('relu_' + layer_name, nn.ReLU(inplace=True)))
            elif 'leaky' in layer_name:
                layers.append(('leaky_' + layer_name, nn.LeakyReLU(negative_slope=0.2, inplace=True)))
        else:
            raise NotImplementedError

    return nn.Sequential(OrderedDict(layers))


# In[15]:


class Encoder(nn.Module):
    def __init__(self, subnets, rnns):
        super().__init__()
        assert len(subnets)==len(rnns)

        self.blocks = len(subnets)

        for index, (params, rnn) in enumerate(zip(subnets, rnns), 1):
            setattr(self, 'stage'+str(index), make_layers(params))
            setattr(self, 'rnn'+str(index), rnn)

    def forward_by_stage(self, input, subnet, rnn):
        input = subnet(input)
        outputs_stage, state_stage = rnn(input, None)
        
        return outputs_stage, state_stage
    
    def forward(self, input):
        hidden_states = []
        logging.debug(input.size())
        for i in range(1, self.blocks+1):
            input, state_stage = self.forward_by_stage(input, getattr(self, 'stage'+str(i)), getattr(self, 'rnn'+str(i)))
            hidden_states.append(state_stage)
        return tuple(hidden_states)


# In[16]:


class Decoder(nn.Module):
    def __init__(self, subnets, rnns):
        super().__init__()
        assert len(subnets) == len(rnns)

        self.blocks = len(subnets)

        for index, (params, rnn) in enumerate(zip(subnets, rnns)):
            setattr(self, 'rnn' + str(self.blocks-index), rnn)
            setattr(self, 'stage' + str(self.blocks-index), make_layers(params))

    def forward_by_stage(self, input, state, subnet, rnn):
        input, state_stage = rnn(input, state)
        input = subnet(input)
        return input

    def forward(self, hidden_states):
        input = self.forward_by_stage(None, hidden_states[-1], getattr(self, 'stage3'),
                                      getattr(self, 'rnn3'))
        for i in list(range(1, self.blocks))[::-1]:
            input = self.forward_by_stage(input, hidden_states[i-1], getattr(self, 'stage' + str(i)),
                                                       getattr(self, 'rnn' + str(i)))
        return input


# In[17]:


class EncoderDecoder(nn.Module):
    def __init__(self, encoder, forecaster):
        super().__init__()
        self.encoder = encoder
        self.forecaster = forecaster

    def forward(self, input):
        state = self.encoder(input)
        output = self.forecaster(state)
        return output


# In[18]:


encoder = Encoder(convlstm_encoder_params[0], convlstm_encoder_params[1])
decoder = Decoder(convlstm_decoder_params[0], convlstm_decoder_params[1])
model3 = EncoderDecoder(encoder, decoder)


# In[19]:


model = model2
print(model)


# Import Dataset

# In[20]:

label_train = np.load("data/hurricane_label_train.npy")
print(label_train.shape)

image_train = np.load("data/hurricane_image_train.npy")
print(image_train.shape)

# In[ ]:


image_train = np.reshape(image_train, (image_train.shape[0]*image_train.shape[1], 10, 128, 257, 6))
label_train = np.reshape(label_train, (label_train.shape[0]*label_train.shape[1], 10, 128, 257, 1))

print(image_train.shape)
print(label_train.shape)

# In[ ]:


image_test = np.load("data/hurricane_image_test.npy")
label_test = np.load("data/hurricane_label_test.npy")


# In[ ]:


image_test = np.reshape(image_test, (image_test.shape[0]*image_test.shape[1], 10, 128, 257, 3))
label_test = np.reshape(label_test, (label_test.shape[0]*label_test.shape[1], 10, 128, 257, 1))


# In[ ]:


print(image_test.shape)
print(label_test.shape)


# Data Parallelism

# In[21]:


import torch.multiprocessing as mp
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import os


# In[22]:


import argparse

parser = argparse.ArgumentParser(description='Tropical Cyclone Detection Model Training')
parser.add_argument('--lr', default=0.1, help='')
parser.add_argument('--batch_size', type=int, default=16, help='')
parser.add_argument('--max_epochs', type=int, default=5, help='')
parser.add_argument('--num_workers', type=int, default=0, help='')

parser.add_argument('--init_method', default='tcp://127.0.0.1:3456', type=str, help='')
parser.add_argument('--dist_backend', default='gloo', type=str, help='')
parser.add_argument('--world_size', default=1, type=int, help='')
parser.add_argument('--distributed', action='store_true', help='')
args = parser.parse_args()


# In[29]:


ngpus_per_node = torch.cuda.device_count()

local_rank = int(os.environ.get("SLURM_LOCALID"))
rank = int(os.environ.get("SLURM_NODEID"))*ngpus_per_node + local_rank

current_device = local_rank

torch.cuda.set_device(current_device)

print("From Rank: {}, ==> Initializing Process Group...".format(rank))

init_process_group(backend=args.dist_backend, init_method=args.init_method, world_size=args.world_size, rank=rank)
print("process group ready!")

print("From Rank: {}, ==> Making model...".format(rank))

model.cuda()
model = DDP(model, device_ids=[current_device])

print("From Rank: {}, ==> Preparing data...".format(rank))

train_sampler = DistributedSampler(image_train)
test_sampler = DistributedSampler(image_test)


# Prepare Dataset

# In[ ]:


class ClimateImageDataset(Dataset):
    def __init__(self, dataset, labels, transform=None, target_transform=None, test=False):
        self.ds_labels = labels
        self.dataset = dataset
        self.transform = transform
        self.target_transform = target_transform
        self.test = test

    def __len__(self):
        return len(self.ds_labels)

    def __getitem__(self, idx):
        if self.test:
            image = self.dataset[idx]
        else:
            image = self.dataset[idx, :, :, :, 1:4]
        label = self.ds_labels[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


# In[ ]:


training_data = ClimateImageDataset(image_train, label_train)
test_data = ClimateImageDataset(image_test, label_test, test=True)
train_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=(train_sampler is None), sampler=train_sampler)
test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=(test_sampler is None), sampler=test_sampler)


# In[19]:


learning_rate = 1e-3
epochs = 5
loss_fn = nn.MSELoss().cuda()
optimizer = Adam(model.parameters(), lr=learning_rate)


# Loss Functions

# In[24]:


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice

ALPHA = 0.8
GAMMA = 2

class FocalLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalLoss, self).__init__()

    def forward(self, inputs, targets, alpha=ALPHA, gamma=GAMMA, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #first compute binary cross-entropy 
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1-BCE_EXP)**gamma * BCE
                       
        return focal_loss
    
ALPHA = 0.5
BETA = 0.5

class TverskyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(TverskyLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1, alpha=ALPHA, beta=BETA):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()    
        FP = ((1-targets) * inputs).sum()
        FN = (targets * (1-inputs)).sum()
       
        Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)  
        
        return 1 - Tversky


# Training

# In[30]:

from torch import tensor

def train_loop(dataloader, model, loss_fn, optimizer, epoch, writer):
    size = len(dataloader.dataset)
    model.train()
    running_loss = 0
    for batch, (input, target) in enumerate(dataloader):
        
        input = input.cuda()
        target = target.cuda()

        pred = model(input)
        pred = pred.permute(0, 1, 3, 4, 2)
        loss = loss_fn(pred, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        if batch % 10 == 0:
            loss, current = loss.item(), (batch + 1) * len(input)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")
            writer.add_scalar('training loss',
                            running_loss / 10,
                            epoch * len(dataloader) + batch + 1)
            running_loss = 0
        
        
def test_loop(dataloader, model, loss_fn, epoch, writer, func):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss = 0
    correct = 0
    preds = []
    targets = []
    
    with torch.no_grad():
        for batch, (input, target) in enumerate(dataloader):
            
            input = input.cuda()
            target = target.cuda()

            [pred], _ = model(input)
            pred = pred.permute(0, 1, 3, 4, 2)
            
            loss = loss_fn(pred, target).item()
            test_loss += loss

            preds.append(pred)
            targets.append(target)
            writer.add_scalar('testing loss', loss, epoch * len(dataloader) + batch + 1)
    
    test_loss = test_loss / num_batches
    
    preds = torch.cat(preds, axis=0).cuda()
    targets = torch.cat(targets, axis=0).cuda()
    
    if func == Dice:
        metric = func(average="micro")
    else:
        metric = func(task="binary")
    
    metric.cuda()
    total_loss = loss_fn(preds, targets).item()
    targets = targets.to(torch.int64)
    score = metric(preds, targets)
    print(f"Test Error: \n : {metric.__class__.__name__}: {score:.5f}, Avg loss: {test_loss:>8f} \n")
    print(f"Total Loss: {total_loss:>8f}\n")


# In[28]:


# get_ipython().run_line_magic('pip', 'install tensorboard')


# In[28]:


from torch.utils.tensorboard import SummaryWriter
import torchmetrics
from torchmetrics.classification import Dice, Recall, Specificity, Accuracy, Precision, JaccardIndex, AveragePrecision
writer = SummaryWriter()

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_sampler.set_epoch(t)
    train_loop(train_dataloader, model, loss_fn, optimizer, t+1, writer)

    #save model at each checkpoint
    ckp = model.module.state_dict()
    PATH = "checkpoint.pt"
    torch.save(ckp, PATH)
    print(f"Epoch {t+1} | Training checkpoint saved at {PATH}")
    
    test_sampler.set_epoch(t)
    test_loop(test_dataloader, model, loss_fn, t+1, writer, JaccardIndex)

destroy_process_group()
print("Done")


# Evaluation

# In[29]:


SMOOTH = 1e-6

def iou_pytorch(outputs: torch.Tensor, labels: torch.Tensor):
    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x 1 x H x W shape
    outputs = outputs.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W
    
    intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum((1, 2))         # Will be zero if both are 0
    
    iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0
    
    thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds
    
    return thresholded  # Or thresholded.mean() if you are interested in average across the batch
