import numpy as np
import copy
# Torch
import torch
import torch.nn.functional as F
import time
import src.utils.io as utils_io
import tqdm as tqdm
from tqdm import trange

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

#############################################
###### Training & Saving 
##############################################

def train_run(model, optimizer, criterion, train_dataloader, val_dataloader, test_dataloader, args, device):

    epochs = args.num_epochs

    avg_train_losses, avg_test_losses = [], []
    best_val_loss = 100
    training_stats = []
    return_dict = {'model': None,
                    'training_stats': None,
                    'test_true_classes': None,
                    'test_predictions': None,
                    'test_scores': None,
                    'val_true_classes': None,
                    'val_predictions': None,
                    'val_scores': None
                    } 

    start_time = time.time()

    progress_bar = trange(epochs, desc="Initializing...")

    for epoch in progress_bar:

        model.train()

        t0 = time.time()

        total_train_loss = training_loop(model, optimizer, criterion, train_dataloader, device=device)

        t1 = time.time()

        model.eval()

        with torch.no_grad():
            val_loss, val_true_classes, val_predictions, val_scores = multi_label_eval_loop(model, criterion, val_dataloader, device=device)

        t2 = time.time()
        
        if val_loss < best_val_loss:

            # calculate test scores
            with torch.no_grad():
                test_loss, test_true_classes, test_predictions, test_scores = multi_label_eval_loop(model, criterion, test_dataloader, device=device)

            best_val_loss, best_val_predictions, best_val_scores = val_loss, val_predictions, val_scores
            best_model_state_dict = copy.deepcopy(model.state_dict())
            best_test_outputs = {
                'test_true_classes': test_true_classes,
                'test_predictions': test_predictions,
                'test_scores': test_scores
            }

            return_dict = {'model': model,
                            'test_true_classes': test_true_classes,
                            'test_predictions': test_predictions,
                            'test_scores': test_scores,
                            'val_true_classes': val_true_classes,
                            'val_predictions': best_val_predictions,
                            'val_scores': best_val_scores
                            } 

        
        # save train and test loss every 10 epochs 
        
        avg_train_loss = total_train_loss/len(train_dataloader)
        avg_train_losses.append(avg_train_loss)
        avg_test_losses.append(val_loss)

        progress_bar.set_description(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Best val Loss: {best_val_loss:.4f}")

        if args.verbose and (epoch == 0 or (epoch+1)%10 == 0):
            print("")
            print(f'========= Epoch {epoch+1}/{epochs} ==========')
            print(f"Average train loss: {avg_train_loss}")
            print(f" Average val loss: {val_loss}")    
            print(f" Best val loss: {best_val_loss}, best test loss: {test_loss}")   

        training_stats.append(
        {
            "epoch": epoch + 1,
            "Training Loss": avg_train_loss,
            "Validation Loss": val_loss,
            "Training Time": utils_io.format_time(t1 - t0),
            "Validation Time": utils_io.format_time(t2 - t1),
        }
        ) 

    end_time = time.time()
    print(f'Total training time: {utils_io.format_time(end_time-start_time)}')   

    # Restore best model state
    if best_model_state_dict is not None:
        model.load_state_dict(best_model_state_dict)

    return_dict.update({
        'model': model,
        'training_stats': training_stats,
        'test_true_classes': best_test_outputs['test_true_classes'],
        'test_predictions': best_test_outputs['test_predictions'],
        'test_scores': best_test_outputs['test_scores'],
        'val_true_classes': val_true_classes,
        'val_predictions': best_val_predictions,
        'val_scores': best_val_scores
    })

    return return_dict
