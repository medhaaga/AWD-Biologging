import numpy as np

# Torch
import torch
import torch.nn.functional as F

def training_loop(model, optimizer, criterion, train_dataloader, device):
    """Training loop for CNN based behavior-prediction model

    Arguments
    ----------------
    model: torch.nn module
    optimizer: torch.optim object
    criterion: torch loss criterion
    train_dataloader: torch DataLoader
    device: torch.device object

    Returns
    ----------------
    train_loss: float
    """

    train_loss = 0

    for batch_X, batch_y in train_dataloader:

        # Forward pass
        outputs = model(batch_X.to(device))
        
        # Calculate the loss
        loss = criterion(outputs, batch_y.to(device))
        train_loss += loss.cpu().item()
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return train_loss

def eval_loop(model, criterion, X, y, device):

    with torch.no_grad():

        outputs = F.softmax(model(X.to(device)), dim=1)
        predictions = torch.argmax(outputs, dim=1)
        test_loss = criterion(outputs, y.to(device)).cpu().item()

    return test_loss, predictions.cpu(), outputs.cpu()

def multi_label_eval_loop(model, criterion, dataloader, device):
    """Evaluation loop for CNN based behavior-prediction model

    Arguments
    ----------------
    model: torch.nn module
    criterion: torch loss criterion
    dataloader: torch DataLoader
    device: torch.device object

    Returns
    ----------------
    loss: float
    true_labels: torch tensor B * 1; B = batch size
    predicted_labels: torch tensor B * 1; B = batch size
    scores: torch tensor B * K; B = batch size, K = number of classes
    """

    loss = 0
    true_labels, predicted_labels, scores = [], [], []
    with torch.no_grad():

        for inputs, labels in dataloader:

            outputs = model(inputs.to(device))
            predictions = torch.argmax(outputs, dim=1)
            loss += criterion(outputs, labels.to(device)).cpu().item()
            true_labels.append(torch.argmax(labels, dim=1).cpu().numpy())
            predicted_labels.append(predictions.cpu().numpy())
            scores.append(outputs.cpu().numpy())

        loss = loss/len(dataloader)
        true_labels = np.concatenate(true_labels)
        predicted_labels = np.concatenate(predicted_labels)
        scores = np.concatenate(scores)


    return loss, true_labels, predicted_labels, scores
