import time
import numpy as np
import torch
from torch import nn
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt
from nitorch.inference import predict
from nitorch.utils import *


class Trainer:
    def __init__(
            self,
            model,
            criterion,
            optimizer,
            scheduler=None,
            metrics=[],
            callbacks=[],
            training_time_callback=None,
            device=torch.device('cuda'),
            prediction_type="binary",
            **kwargs
    ):
        """ Main class for training.
        # Arguments
            model: neural network to train.
            criterion: loss function.
            optimizer: optimization function.
            scheduler: schedules the optimizer.
            metrics: list of metrics to report. Default is None.
            callbacks: list of callbacks to execute at the end of training epochs. Default is None.
            training_time_callback: a user-defined callback that executes the model.forward()
                and returns the output to the trainer.
                This can be used to perform debug during train time, Visualize features,
                call model.forward() with custom arguments, run multiple decoder networks etc.
                Default is None.
            class_threshold: classification threshold for binary
                classification. Default is 0.5.
            prediction_type: accepts one of ["binary", "classification",
                "regression", "reconstruction", "variational", "other"].
                This is used to determine output type.
            device: The device to use for training. Must be integer or
                    a torch.device object. By default, GPU with current
                    node is used.
        """
        if not isinstance(model, nn.Module):
            raise ValueError("Expects model type to be torch.nn.Module")
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.metrics = metrics
        self.callbacks = callbacks
        self.training_time_callback = training_time_callback
        if isinstance(device, int):
            self.device = torch.device("cuda:" + str(device))
        elif isinstance(device, torch.device):
            self.device = device
        else:
            raise ValueError("Device needs to be of type torch.device or \
                integer.")
        if "class_threshold" in kwargs.keys():
            self.class_threshold = kwargs["class_threshold"]
        else:
            self.class_threshold = None
        self.stop_training = False
        self.start_time = None
        self.prediction_type = prediction_type

    def train_model(
            self,
            train_loader,
            val_loader,
            inputs_key="image",
            labels_key="label",
            num_epochs=25,
            show_train_steps=25,
            show_validation_epochs=1
    ):
        """ Main function to train a network for one epoch.
        Args:
            train_loader: a pytorch Dataset iterator for training data
            val_loader: a pytorch Dataset iterator for validation data
            inputs_key, labels_key: The data returned by `train_loader` and `val_loader`can
                            either be a dict of format data_loader[X_key] = inputs and
                            data_loader[y_key] = labels or a list with data_loader[0] = inputs
                            and data_loader[1] = labels. The default keys are "image" and "label".
        """
        assert (show_validation_epochs < num_epochs) or (num_epochs == 1), "\
'show_validation_epochs' value should be less than 'num_epochs'"
        assert (show_train_steps>0) and (show_train_steps<=len(train_loader)),"\
'show_train_steps' value out of range. Must be > 0 and < len(train_loader)"

        val_metrics = dict()
        train_metrics = dict()

        self.start_time = time.time()
        self.best_metric = 0.0
        self.best_model = None

        for epoch in range(num_epochs):
            if self.stop_training:
                # TODO: check position of this
                print("Early stopping in epoch {}".format(epoch))
                return self.finish_training(train_metrics, val_metrics, epoch)
            else:
                # running_loss accumulates loss every 'show_train_steps' cycles until it must be printed.
                running_loss = np.array([])
                epoch_loss = 0.0
                if self.scheduler:
                    self.scheduler.step(epoch)

                # Reset all metrics related variables at the start of each epoch
                all_preds = []
                all_labels = []
                self.multi_batch_metrics = dict()
                # train mode
                self.model.train()

                for i, data in enumerate(train_loader):
                    try:
                        inputs, labels = data[inputs_key], data[labels_key]
                    except TypeError:
                        # if data does not come in dictionary, assume
                        # that data is ordered like [input, label]
                        try:
                            inputs, labels = data[0], data[1]
                        except TypeError:
                            raise TypeError
                    # in case of multi-input or output create a list
                    if isinstance(inputs, list):
                        inputs = [inp.to(self.device) for inp in inputs]
                    else:
                        inputs = inputs.to(self.device)
                    if isinstance(labels, list):
                        labels = [label.to(self.device) for label in labels]
                    else:
                        labels = labels.to(self.device)

                    # zero the parameter gradients
                    self.optimizer.zero_grad()
                    # forward + backward + optimize

                    if self.training_time_callback is not None:
                        outputs = self.training_time_callback(
                            inputs, 
                            labels,
                            i,
                            epoch
                        )
                    else:
                        outputs = self.model(inputs)

                    if self.prediction_type == "classification":
                        labels = labels.squeeze(1)
                    loss = self.criterion(outputs, labels)
                    loss.backward()

                    # enable the below commented code if you want to visualize the 
                    # gradient flow through the model during training
                    # plot_grad_flow(self.model.named_parameters())
                    self.optimizer.step()

                    # store results
                    all_preds, all_labels = predict(
                        outputs,
                        labels,
                        all_preds,
                        all_labels,
                        self.prediction_type,
                        self.criterion,
                        class_threshold=self.class_threshold
                    )
                    # update loss
                    running_loss= np.append(running_loss, loss.item())
                    epoch_loss += loss.item()
                    # print loss every X mini-batches
                    if (i % show_train_steps == 0) and (i != 0):
                        print(
                            "[%d, %5d] loss: %.5f"
                            % (epoch , i , 
                               running_loss.mean())
                        )
                        running_loss = np.array([]) #reset

                    # compute training metrics for X/2 mini-batches
                    # useful for large outputs (e.g. reconstructions)
                    if self.prediction_type == "reconstruction":
                        if i % int(show_train_steps/2) == 0:
                            self.estimate_metrics(
                                all_labels,
                                all_preds,
                            )
                            # TODO: test if del helps
                            all_labels = []
                            all_preds = []

                # report training metrics
                # weighted averages of metrics are computed over batches
                train_metrics = self._on_epoch_end(
                        train_metrics,
                        all_labels,
                        all_preds,
                        phase="train"
                    )
                epoch_loss /= len(train_loader)

                # add loss to metrics data
                if "loss" in train_metrics:
                    train_metrics["loss"].append(epoch_loss)
                else:
                    train_metrics["loss"] = [epoch_loss]

                #<end-of-training-cycle-loop>
            #<end-of-epoch-loop>

            # validate every x iterations
            if epoch % show_validation_epochs == 0:
                self.model.eval()
                validation_loss = 0.0
                all_preds = []
                all_labels = []
                self.multi_batch_metrics = dict()

                with torch.no_grad():
                    for i, data in enumerate(val_loader):
                        try:
                            inputs, labels = data[inputs_key], data[labels_key]
                        except TypeError:
                            # if data does not come in dictionary, assume
                            # that data is ordered like [input, label]
                            try:
                                inputs, labels = data[0], data[1]
                            except TypeError:
                                raise TypeError("Data not in correct \
                                 sequence format.")
                        # in case of multi-input or output create a list
                        if isinstance(inputs, list):
                            inputs = [inp.to(self.device) for inp in inputs]
                        else:
                            inputs = inputs.to(self.device)
                        if isinstance(labels, list):
                            labels = [label.to(self.device) for label in labels]
                        else:
                            labels = labels.to(self.device)

                        # forward pass only
                        if self.training_time_callback is not None:
                            outputs = self.training_time_callback(
                                inputs, 
                                labels,
                                1,  # dummy value
                                1  # dummy value
                            )
                        else:
                            outputs = self.model(inputs)

                        if self.prediction_type == "classification":
                            labels = labels.squeeze(1)
                        loss = self.criterion(outputs, labels)
                        # compute validation accuracy
                        all_preds, all_labels = predict(
                            outputs,
                            labels,
                            all_preds,
                            all_labels,
                            self.prediction_type,
                            self.criterion,
                            class_threshold=self.class_threshold
                        )

                        validation_loss += loss.item()

                        # compute training metrics for X/2 mini-batches
                        # useful for large outputs (e.g. reconstructions)
                        if self.prediction_type == "reconstruction":
                            if i % int(show_train_steps/2) == 0:
                                self.estimate_metrics(
                                    all_labels,
                                    all_preds,
                                )
                                # TODO: test if del helps
                                all_labels = []
                                all_preds = []

                    # report validation metrics
                    # weighted averages of metrics are computed over batches
                    val_metrics = self._on_epoch_end(
                        val_metrics,
                        all_labels,
                        all_preds,
                        phase="val"
                    )

                    validation_loss /= len(val_loader)
                    print("Val loss: {0:.6f}".format(validation_loss))
                    # add loss to metrics data
                    if "loss" in val_metrics:
                        val_metrics["loss"].append(validation_loss)
                    else:
                        val_metrics["loss"] = [validation_loss]
            if self.callbacks is not None:
                for callback in self.callbacks:
                    callback(self, epoch, val_metrics)
        # End training
        return self.finish_training(train_metrics, val_metrics, epoch)

    def finish_training(self, train_metrics, val_metrics, epoch):
        """
        End the training cyle, return a model and finish callbacks.
        """
        time_elapsed = int(time.time() - self.start_time)
        print("Total time elapsed: {}h:{}m:{}s".format(
            time_elapsed // 3600, (time_elapsed // 60) % 60, time_elapsed % 60))
        # execute final methods of callbacks
        if self.callbacks is not None:
            for callback in self.callbacks:
                # find all methods of the callback
                method_list = [
                    func
                    for func in dir(callback)
                    if (callable(getattr(callback, func))
                        and not func.startswith("__"))
                ]
                if "final" in method_list:
                    callback.final(trainer=self, epoch=epoch)
        # in case of no model selection, pick the last loss
        if self.best_metric == 0.0:
            self.best_metric = val_metrics["loss"][-1]
            self.best_model = self.model

        return (self.model,
                {
                    "train_metrics": train_metrics,
                    "val_metrics": val_metrics,
                    "best_model": self.best_model,
                    "best_metric": self.best_metric}
                )

    def visualize_training(self, report, metrics=None, save_fig_path=""):
        # Plot loss first
        plt.figure()
        plt.plot(report["train_metrics"]["loss"])
        plt.plot(report["val_metrics"]["loss"])
        plt.title("Loss during training")
        plt.legend(["Train", "Val"])
        if (save_fig_path):
            plt.savefig(save_fig_path)
        plt.show()
        if metrics is None:
            metrics = self.metrics
        for metric in metrics:
            plt.figure()
            plt.plot(report["train_metrics"][metric.__name__])
            plt.plot(report["val_metrics"][metric.__name__])
            plt.legend(["Train", "Val"])
            plt.title(metric.__name__)        
            if(save_fig_path):
                plt.savefig(save_fig_path+"_"+metric.__name__)
            plt.show()

    def evaluate_model(
            self,
            val_loader,
            additional_gpu=None,
            metrics=None,
            inputs_key="image",
            labels_key="label"
    ):
        # predict on the validation set
        """
        Predict on the validation set.
        # Arguments
            val_loader : data loader of the validation set
            additional_gpu : GPU number if evaluation should be done on
                separate GPU
            metrics: list of
        """
        all_preds = []
        all_labels = []

        self.model.eval()

        if additional_gpu is not None:
            device = additional_gpu
        else:
            device = self.device

        with torch.no_grad():
            for i, data in enumerate(val_loader):
                inputs, labels = data[inputs_key], data[labels_key]
                inputs = inputs.to(device)
                labels = labels.to(device)
                # forward + backward + optimize
                outputs = self.model(inputs)
                # run inference
                all_preds, all_labels = predict(
                    outputs,
                    labels,
                    all_preds,
                    all_labels,
                    self.prediction_type,
                    self.criterion,
                    class_threshold=self.class_threshold
                )

        # compute confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)

        # Visualize the confusion matrix
        classes = ["control", "patient"]
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = "d"
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(
                j,
                i,
                format(cm[i, j], fmt),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
            )
        plt.title("Confusion Matrix")
        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        plt.show()

        # print metrics
        if metrics is not None:
            for metric in metrics:
                if isinstance(all_preds[0], list):
                    print("{}: {}".format(metric.__name__, np.mean([metric(labels, preds) for preds,labels in zip(all_preds, all_labels)])))
                else:
                    print("{}: {}".format(metric.__name__, metric(all_labels, all_preds)))


        self.model.train()

    def report_metrics(
        self,
        metrics_dict,
        phase
        ):

        # report execution time only in training phase
        if (phase == "train"):
            time_elapsed = int(time.time() - self.start_time)
            print("Time elapsed: {}h:{}m:{}s".format(
                time_elapsed // 3600, (time_elapsed // 60) % 60, time_elapsed % 60))

        """ Store and report a list of metric functions. """
        for metric in self.metrics:
            # report everything but loss
            if metric.__name__ is not "loss":
                # weighted average over previous batches
                # weigh by the number of samples per batch and divide by
                # the total number of samples
                batch_results = np.zeros(shape=(
                    len(self.multi_batch_metrics["len_" + metric.__name__])))
                n_samples = 0
                for b_idx, batch_len in enumerate(
                    self.multi_batch_metrics["len_" + metric.__name__]
                    ):
                    batch_results[b_idx] = self.multi_batch_metrics[
                        metric.__name__][b_idx] * batch_len
                    n_samples += batch_len

                result = np.sum(batch_results) / n_samples

                if metric.__name__ in metrics_dict:
                    metrics_dict[metric.__name__].append(result)
                else:
                    metrics_dict[metric.__name__] = [result]
                # print result
                if isinstance(result, float):
                    print("{} {}: {:.2f} %".format(
                        phase, metric.__name__, result * 100))
                else:
                    print("{} {}: {} ".format(
                        phase, metric.__name__, str(result)))
        return metrics_dict

    def estimate_metrics(
        self,
        all_labels,
        all_preds
        ):
        """ Estimate a list of metric functions. """
        n_predictions = len(all_preds)

        for metric in self.metrics:
            # report everything but loss
            if metric.__name__ is not "loss":
                if isinstance(all_preds[0], list):
                    result = np.mean([metric(labels, preds) for preds,labels in zip(all_preds, all_labels)])
                else:
                    result = metric(all_labels, all_preds)
 
                if metric.__name__ in self.multi_batch_metrics:
                    self.multi_batch_metrics[metric.__name__].append(result)
                    self.multi_batch_metrics["len_" + metric.__name__].append(
                        n_predictions)
                else:
                    self.multi_batch_metrics[metric.__name__] = [result]
                    self.multi_batch_metrics["len_" + metric.__name__] = [n_predictions]

    def _on_epoch_end(
        self,
        metrics_dict,
        all_labels,
        all_preds,
        phase
        ):
        # check for unreported metrics
        if len(all_preds) > 0:
            self.estimate_metrics(
                    all_labels,
                    all_preds,
                )
            # TODO: test if del helps
            all_labels = []
            all_preds = []

        metrics_dict = self.report_metrics(
            metrics_dict,
            phase
        )

        return metrics_dict
        
