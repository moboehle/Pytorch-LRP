import numpy
import torch
from torch import nn

def predict(
    outputs,
    labels,
    all_preds,
    all_labels,
    prediction_type,
    criterion,
    **kwargs
    ):
    """ Predict according to loss and prediction type."""
    if prediction_type == "binary":
        if isinstance(criterion, nn.BCEWithLogitsLoss):
            all_preds, all_labels = bce_with_logits_inference(
                outputs,
                labels,
                all_preds,
                all_labels,
                **kwargs
            )
        elif isinstance(criterion, nn.BCELoss):
            all_preds, all_labels = bce_inference(
                outputs,
                labels,
                all_preds,
                all_labels,
                **kwargs
            )
    elif prediction_type == "classification":
        all_preds, all_labels = crossentropy_inference(
                outputs,
                labels,
                all_preds,
                all_labels
        )
    elif prediction_type == "regression":
        # TODO: test different loss functions
        all_preds, all_labels = regression_inference(
                outputs,
                labels,
                all_preds,
                all_labels
        )
    elif prediction_type == "reconstruction":
        # TODO: test different loss functions
        all_preds, all_labels = regression_inference(
                outputs,
                labels,
                all_preds,
                all_labels
        )
    elif prediction_type == "variational":
        # TODO: test different loss functions
        all_preds, all_labels = variational_inference(
                outputs,
                labels,
                all_preds,
                all_labels
        )
    else:
        raise NotImplementedError

    return all_preds, all_labels


def bce_with_logits_inference(
    outputs,
    labels,
    all_preds,
    all_labels,
    **kwargs
    ):
    sigmoid = torch.sigmoid(outputs)
    if kwargs["class_threshold"]:
        class_threshold = kwargs["class_threshold"]
    else:
        class_threshold = 0.5
    print
    predicted = sigmoid.data >= class_threshold
    for pred, label in zip(predicted, labels):
        all_preds.append(pred.cpu().item())
        all_labels.append(int(label.cpu().item()))
    return all_preds, all_labels

def bce_inference(
    outputs,
    labels,
    all_preds,
    all_labels,
    **kwargs
    ):
    if kwargs["class_threshold"]:
        class_threshold = kwargs["class_threshold"]
    else:
        class_threshold = 0.5
    predicted = outputs.data >= class_threshold
    for pred, label in zip(predicted, labels):
        all_preds.append(pred.cpu().item())
        all_labels.append(label.cpu().item())
    return all_preds, all_labels

def crossentropy_inference(
    outputs,
    labels,
    all_preds,
    all_labels,
    **kwargs
    ):
    _, predicted = torch.max(outputs.data, 1)
    for pred, label in zip(predicted, labels):
        all_preds.append(pred.cpu().item())
        all_labels.append(label.cpu().item())
    return all_preds, all_labels

def regression_inference(
    outputs,
    labels,
    all_preds,
    all_labels
    ):
    # Multi-head case
    # network returns a tuple of outputs
    if isinstance(outputs, (list,tuple)):
        predicted = [output.data for output in outputs]
        for head in range(len(predicted)):
            for j in range(len(predicted[head])):
                try:
                    all_preds[head].append(predicted[head][j].cpu().numpy()[0])
                    all_labels[head].append(labels[head][j].cpu().numpy()[0])
                except IndexError:
                    # create inner lists if needed
                    all_preds.append([predicted[head][j].cpu().numpy()[0]])
                    all_labels.append([labels[head][j].cpu().numpy()[0]])
        return all_preds, all_labels
    # Single-head case
    else:
        predicted = outputs[0].data
        # TODO: replace for loop with something faster
        for j in range(len(predicted)):
            try:
                all_preds.append(predicted[j].cpu().numpy().item())
                all_labels.append(labels[j].cpu().numpy().item())
            except:
                all_preds.append(predicted[j].cpu().numpy()[0])
                all_labels.append(labels[j].cpu().numpy()[0])
        return all_preds, all_labels

def variational_inference(
    outputs,
    labels,
    all_preds,
    all_labels
    ):
    """ Inference for variational autoencoders. """
    # VAE outputs reconstruction, mu and std
    # select reconstruction only
    outputs = outputs[0]
    predicted = outputs.data 
    # TODO: replace for loop with something faster
    for pred, label in zip(predicted, labels):
        all_preds.append(pred.cpu().item())
        all_labels.append(label.cpu().item())
    return all_preds, all_labels
