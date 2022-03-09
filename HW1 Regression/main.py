from train_loop import *
from data_loader import *
from config import *
from model_test import *

def save(test_loader):
    model = model = My_Model(input_dim=input_dim).to(device)
    model.load_state_dict(torch.load(config['save_path']))
    preds = predict(test_loader, model, device)

    pred_file_path = "data/pred.csv"
    save_pred(preds, pred_file_path)


if __name__ == '__main__':
    train_loader, valid_loader, test_loader, input_dim = get_dataloader()
    model = My_Model(input_dim=input_dim).to(device)
    trainer(train_loader, valid_loader, model, config, device)

    print("============= Training Down =============")

    save(test_loader)

    
