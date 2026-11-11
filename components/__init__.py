from components.main import initialize, read_params, outFunc
import os


from components.Algorithm import Algorithm
from components.Client import Client
from components.DataLoader import DataLoader
from components.Module import Module
from components.Metric import Metric
from components.seed import setup_seed

from components.metric.Correct import Correct

from components.model.CNN import CNN
from components.model.MLP import MLP
from components.model.ResNet import ResNet18, ResNet50, ResNet152


from components.algorithm.FedAvg.FedAvg import FedAvg
from components.algorithm.qFedAvg.qFedAvg import qFedAvg
from components.algorithm.AFL.AFL import AFL
from components.algorithm.FedMGDA_plus.FedMGDA_plus import FedMGDA_plus
from components.algorithm.DRFL.DRFL import DRFL
from components.algorithm.FedMDFG.FedMDFG import FedMDFG
from components.algorithm.FedLF.FedLF import FedLF
from components.algorithm.AdaFed.AdaFed import AdaFed
from components.algorithm.FairMOO.FairMOO import FairMOO
from components.algorithm.FC_MOO.FC_MOO import FC_MOO
from components.algorithm.FC_MOO.FC_MOO_sto import FC_MOO_sto

from components.dataloaders.separate_data import separate_data, create_data_pool
from components.dataloaders.DataLoader_cifar10_pat import DataLoader_cifar10_pat
from components.dataloaders.DataLoader_cifar10_dir import DataLoader_cifar10_dir
from components.dataloaders.DataLoader_fashion_pat import DataLoader_fashion_pat
from components.dataloaders.DataLoader_fashion_dir import DataLoader_fashion_dir
from components.dataloaders.DataLoader_cifar100_pat import DataLoader_cifar100_pat
from components.dataloaders.DataLoader_cifar100_pat1 import DataLoader_cifar100_pat1
from components.dataloaders.DataLoader_cifar100_dir import DataLoader_cifar100_dir


ALGORITHMS = {
    'FedAvg': FedAvg,
    'qFedAvg': qFedAvg,
    'AFL': AFL,
    'DRFL': DRFL,
    'FedMGDA_plus': FedMGDA_plus,
    'FedMDFG': FedMDFG,
    'FedLF': FedLF,
    'AdaFed': AdaFed,
    'FairMOO': FairMOO,
    'FC-MOO': FC_MOO,
    'FC-MOO-sto': FC_MOO_sto,
}


data_folder_path = os.path.dirname(os.path.abspath(__file__)) + '/data/'
if not os.path.exists(data_folder_path):
    os.makedirs(data_folder_path)


pool_folder_path = os.path.dirname(os.path.abspath(__file__)) + '/pool/'
if not os.path.exists(pool_folder_path):
    os.makedirs(pool_folder_path)
