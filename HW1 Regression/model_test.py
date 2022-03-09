from config import *
from nn_model import *
from data_loader import *

def predict(test_loader, model, device):
    model.eval()
    preds = []
    for x in tqdm(test_loader):
        x = x.to(device)
        with torch.no_grad():
            pred = model(x)
            preds.append(pred.detach().cpu())
    
    preds = torch.cat(preds, dim=0).numpy()
    return preds

def save_pred(preds, file):
    """
        Save predictions to specified file
    """
    with open(file, 'w') as fp:
        writer = csv.writer(fp)
        writer.writerow(['id', 'tested_positive'])
        for i, p in enumerate(preds):
            writer.writerow([i, p])
    return
