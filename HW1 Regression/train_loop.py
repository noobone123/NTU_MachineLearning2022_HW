from myLib.Importer import *
from myLib.Utility import *
from config import *
from nn_model import *
from data_loader import *

def trainer(train_loader, valid_loader, model, config, device):
    # Define loss function, do not modify this.
    criterion = nn.MSELoss(reduction='mean')

    # Define your optimization algorithm. 
    # TODO: Please check https://pytorch.org/docs/stable/optim.html to get more available algorithms.
    # TODO: L2 regularization (optimizer(weight decay...) or implement by your self).
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=1e-8)

    # Writer of tensorboard
    writer = SummaryWriter()

    if not os.path.isdir("./models"):
        os.mkdir("./models")

    n_epochs, best_loss, step, early_stop_count = config['n_epochs'], math.inf, 0, 0

    for epoch in range(n_epochs):
        # set model to train mode
        model.train()
        loss_record = []

        train_pbar = tqdm(train_loader, position=0, leave=True)

        for x, y in train_pbar:
            # set gradient to zero
            optimizer.zero_grad()
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward() # Compute gradient (Back Propagation)
            optimizer.step() # Update the model parameters

            step += 1
            loss_record.append(loss.detach().item())

            # Display current epoch number and loss on tqdm progress bar
            train_pbar.set_description(f"Epoch [{epoch+1}/{n_epochs}]")
            train_pbar.set_postfix({'loss': loss.detach().item()})

        mean_train_loss = sum(loss_record)/len(loss_record)
        writer.add_scalar("Loss/train", mean_train_loss, step)

        # Set model to evaluation mode
        model.eval()
        loss_record = []
        for x, y in valid_loader:
            x, y = x.to(device), y.to(device)
            # disable gradient calculation
            with torch.no_grad():
                pred = model(x)
                loss = criterion(pred, y)
            
            loss_record.append(loss.item())

        mean_valid_loss = sum(loss_record) / len(loss_record)
        print(f'Epoch [{epoch+1}/{n_epochs}]: Train loss: {mean_train_loss:.4f}, Valid loss: {mean_valid_loss:.4f}')
        writer.add_scalar("Loss/valid", mean_valid_loss, step)

        if mean_valid_loss < best_loss:
            best_loss = mean_valid_loss
            torch.save(model.state_dict(), config['save_path'])
            print("Saving model with loss {:.3f}...".format(best_loss))
            early_stop_count = 0
        else:
            early_stop_count += 1
        
        if early_stop_count >= config['early_stop']:
            print("Best loss is: {:.3f}...".format(best_loss))
            print("\n Model is not improving, halting the training session")
            return