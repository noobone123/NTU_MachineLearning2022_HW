"""
    Declare and Define some utility functions here 
"""
from myLib.Importer import *

def same_seed(seed):
    """
        Fixes random number generator seeds for reproducibility
    """
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # set seeds for all random generator we will use 
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_valid_split(data_set, valid_ratio, seed):
    """
        Split provided training data into training set and validation set
    """
    valid_set_size = int(valid_ratio * len(data_set))
    train_set_size = len(data_set) - valid_set_size
    train_set, valid_set = random_split(data_set, [train_set_size, valid_set_size], generator=torch.Generator().manual_seed(seed))

    return np.array(train_set), np.array(valid_set)
    

def predict():
    pass