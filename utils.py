import torch
import numpy as np

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"
    
def train(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    
    for batch in dataloader:
        timeseries_inputs, tabular_inputs, targets = batch
        
        timeseries_inputs = timeseries_inputs.to(device)
        tabular_inputs = tabular_inputs.to(device)
        targets = targets.to(device)
        
        outputs = model(timeseries_inputs, tabular_inputs)
        
        loss = criterion(outputs, targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(dataloader)

def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in dataloader:
            timeseries_inputs, tabular_inputs, targets = batch
            
            timeseries_inputs = timeseries_inputs.to(device)
            tabular_inputs = tabular_inputs.to(device)
            targets = targets.to(device)
            
            outputs = model(timeseries_inputs, tabular_inputs)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
    
    return total_loss / len(dataloader)


def fitting(epochs, model, train_dataloader, val_dataloader, criterion, optimizer, device, schedular, index):
    min_val_loss = float('inf')
    counter = 0
    patience = 20
    train_losses = []
    val_losses = []
    for epoch in range(1, epochs+1):
        train_loss = train(model=model, dataloader=train_dataloader, criterion=criterion, optimizer=optimizer, device=device)
        val_loss = validate(model=model, dataloader=val_dataloader, criterion=criterion, device=device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            counter = 0
            torch.save(model,f'./models/model_{index}.pth')
        else:
            counter += 1
            if counter >= patience:
                print(f'Early stopping triggered at epoch {epoch}')
                break
        schedular.step()
    train_losses = np.array(train_losses)
    val_losses = np.array(val_losses)
    print(f'{min_val_loss}')
    np.save(f'./models/train_losses_{index}', train_losses)
    np.save(f'./models/val_losses_{index}', val_losses)