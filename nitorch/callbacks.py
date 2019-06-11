import os
import copy
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.mplot3d import Axes3D
# pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# nitorch
from nitorch.data import show_brain

class Callback:
    """
    Abstract class for callbacks.
    """

    def __init__(self):
        pass

    def __call__(self):
        pass

    def reset(self):
        pass

    def final(self, **kwargs):
        self.reset()

class ModelCheckpoint(Callback):
    """
    # TODO

    Arguments:
        path:
        num_iters: number of iterations after which to store the model.
            If set to -1, it will only store the last iteration's model.
        prepend: string to prepend the filename with.
        ignore_before: ignore early iterations.
        store_best: boolen whether to save the best model during
            training.
        store_best_metric: name of the metric to use for best model
            selection.
        mode: "max" or "min".
    """

    def __init__(
        self,
        path,
        retain_metric="loss",
        prepend="",
        num_iters=-1,
        ignore_before=0,
        store_best=False,
        mode="max"
        ):
        super().__init__()
        if os.path.isdir(path):
            self.path = path
        else:
            os.makedirs(path)
            self.path = path
        # end the prepended text with an underscore if it does not
        if not prepend.endswith("_") and prepend != "":
            prepend += "_"
        self.prepend = prepend
        self.num_iters = num_iters
        self.ignore_before = ignore_before
        self.best_model = None
        self.best_res = -1
        self.store_best = store_best
        self.retain_metric = retain_metric
        self.mode = mode

        print(self.path)

    def __call__(self, trainer, epoch, val_metrics):
        # do not store intermediate iterations
        print(self.path)
        if epoch >= self.ignore_before and epoch != 0:
            if not self.num_iters == -1:
            
                # counting epochs starts from 1; i.e. +1
                epoch += 1
                # store model recurrently if set
                if epoch % self.num_iters == 0:
                    name = self.prepend + "training_epoch_{}.h5".format(epoch)
                    full_path = os.path.join(self.path, name)
                    self.save_model(trainer, full_path)

            # store current model if improvement detected
            if self.store_best:
                current_res = 0
                try:
                    # check if value can be used directly or not
                    if isinstance(self.retain_metric, str):
                        current_res = val_metrics[self.retain_metric][-1]
                    else:
                        current_res = val_metrics[self.retain_metric.__name__][-1]
                except KeyError:
                    print("Couldn't find {} in validation metrics. Using \
                        loss instead.".format(self.retain_metric))
                    current_res = val_metrics["loss"][-1]
                    
                if self.has_improved(current_res):
                    self.best_res = current_res
                    self.best_model = deepcopy(trainer.model.state_dict())

    def reset(self):
        """
        Reset module after training.
        Useful for cross validation.
        """
        self.best_model = None
        self.best_res = -1

    def final(self, **kwargs):
        epoch = kwargs["epoch"] + 1
        if epoch >= self.ignore_before:
            name = self.prepend + "training_epoch_{}_FINAL.h5".format(epoch)
            full_path = os.path.join(self.path, name)
            self.save_model(kwargs["trainer"], full_path)
        else:
            print("Minimum iterations to store model not reached.")

        if self.best_model is not None:
            best_model = deepcopy(self.best_model)
            best_res = self.best_res
            print("Best result during training: {:.2f}. Saving model..".format(best_res))
            name = self.prepend + "BEST_ITERATION.h5"
            torch.save(best_model, os.path.join(self.path, name))
        self.reset()

    def save_model(self, trainer, full_path):
        print("Writing model to disk...")
        model = trainer.model.cpu()
        torch.save(model.state_dict(), full_path)
        if trainer.device is not None:
            trainer.model.cuda(trainer.device)

    def has_improved(self, res):
        if self.mode == "max":
            return res >= self.best_res
        elif self.mode == "min":
            # check if still standard value
            if self.best_res == -1:
                return True
            else:
                return res <= self.best_res
        else:
            raise NotImplementedError("Only modes 'min' and 'max' available")


class EarlyStopping(Callback):
    """ 
    Stop training when a monitored quantity has stopped improving.

    Arguments
        patience: number of iterations without improvement after which
            to stop
        retain_metric: the metric which you want to monitor
        mode: {min or max}; defines if you want to maximise or minimise
            your metric
        ignore_before: does not start the first window until this epoch.
            Can be useful when training spikes a lot in early epochs.
    """


    def __init__(self, patience, retain_metric, mode, ignore_before=0):
        self.patience = patience
        self.retain_metric = retain_metric
        self.mode = mode
        self.ignore_before = ignore_before
        self.best_res = -1
        # set to first iteration which is interesting
        self.best_epoch = self.ignore_before

    def __call__(self, trainer, epoch, val_metrics):
        if epoch >= self.ignore_before:
            if epoch - self.best_epoch < self.patience:
                if isinstance(self.retain_metric, str):
                    current_res = val_metrics[self.retain_metric][-1]
                else:
                    current_res = val_metrics[self.retain_metric.__name__][-1]
                if self.has_improved(current_res):
                    self.best_res = current_res
                    self.best_epoch = epoch
            else:
                # end training run
                trainer.stop_training = True

    def has_improved(self, res):
        if self.mode == "max":
            return res > self.best_res
        elif self.mode == "min":
            # check if still standard value
            if self.best_res == -1:
                return True
            else:
                return res < self.best_res
        else:
            raise NotImplementedError("Only modes 'min' and 'max' available")

    def reset(self):
        """ Resets after training. Useful for cross validation."""
        self.best_res = -1
        self.best_epoch = self.ignore_before

    def final(self, **kwargs):
        self.reset()

        
# Functions which can be used in custom-callbacks for visualizing 3D-features during training 
# (using the argument 'training_time_callback' in nitorch's Trainer class )
def visualize_feature_maps(features, return_fig=False):
    
    if(features.is_cuda):
        features = features.cpu().detach().numpy()

    num_features = len(features)
    plt.close('all')
    figsize=((num_features//8 + 5)*3 ,(num_features//8)*10 )
    fig = plt.figure(figsize=figsize)

    for i, f in enumerate(features, 1):            
        # normalize to range [0, 1] first as the values can be very small            
        if((f.max() - f.min()) != 0):
            f = (f - f.min()) / (f.max() - f.min())

            idxs = np.nonzero(f)
            vals = np.ravel(f[idxs])                
            if(len(vals)):
                # calculate the index where the mean value would lie
                mean_idx = np.average(idxs, axis = 1, weights=vals)
                # calculate the angel ratios for each non-zero val            
                angles = (mean_idx.reshape(-1,1) - idxs)
                angles = angles/ (np.max(abs(angles), axis=1).reshape(-1,1))    
            else: # if all values in f are zero, set dummy angle
                angles = [1, 1, 1]

#             print("values = ",vals)
            ax = fig.add_subplot(num_features//3+1, 3, i,
                                  projection='3d')
            ax.set_title("Feature-{} in the bottleneck".format(i))
            ax.quiver(*idxs
                      , angles[0]*vals, angles[1]*vals, angles[2]*vals
                     )    
            plt.grid()

        else:
            ax = fig.add_subplot(num_features//3+1, 3, i)
            ax.text(0.5, 0.5, "All values zero!", transform=ax.transAxes)
            plt.axis('off')
            
    plt.tight_layout()
    if return_fig:
        return fig


class CAE_VisualizeTraining(Callback):
    ''' 
    training_time_callback that prints the model dimensions,
    visualizes CAE encoder outputs, original image and reconstructed image
    during training.
    
    NOTE : The forward() function of the CAE model using this callback
    must return a (decoder_output, encoder_output) tuple.
    '''
    def __init__(self, model, max_train_iters, show_epochs_list=[], plotFeatures=True, plot_pdf_path="", cmap="nipy_spectral"):
        self.model = model
        self.max_train_iters = max_train_iters
        if plot_pdf_path is not None:
            assert isinstance(plot_pdf_path, str), "pp is not a path!"
        self.plot_pdf_path = plot_pdf_path
        assert isinstance(plotFeatures, bool), "plotFeatures not boolean object!"
        self.plotFeatures = plotFeatures
        assert isinstance(show_epochs_list, list), "show_epochs_list is not a list!"
        self.show_epochs_list = show_epochs_list
        self.cmap = cmap
        self.ave_grads = []
        self.layers = []
        # inform the model to also return the encoder output along with the decoder output
        try:
            if(isinstance(model, nn.DataParallel)): 
                model.module.set_return_encoder_out(True)
            else:
                model.set_return_encoder_out(True)
        except AttributeError:
            raise "The CAE model must implement a setter function 'set_return_encoder_out'\
 for a flag 'encoder_out' which when set to true, the forward() function using this callback \
must return a (decoder_output, encoder_output) tuple instead of just (encoder_output). See the CAE class in models.py for the framework."

    def __call__(self, inputs, labels, train_iter, epoch):
        debug = False
        visualize_training = False
        tmp_show_epoches_list = []

        # if show_epochs_list is empty, all epoches should be plotted. Therefore, add current epoch to the list
        if not self.show_epochs_list:
            tmp_show_epoches_list.append(epoch)
        else:
            tmp_show_epoches_list = self.show_epochs_list

        # check if epoch should be visualized
        if epoch in tmp_show_epoches_list:
            # print the model's parameter dimensions etc in the first iter
            if (train_iter == 0 and epoch == 0):
                debug = True
            # visualize training on the last iteration in that epoch
            elif(train_iter==1 and epoch==0) or (train_iter == self.max_train_iters):
                visualize_training = True

        # for nitorch models which have a 'debug' and 'visualize_training' switch in the
        # forward() method

        if(isinstance(self.model, nn.DataParallel)):
            self.model.module.set_debug(debug)
        else:
            self.model.set_debug(debug)

        outputs, encoder_out = self.model(inputs)
        
        if(visualize_training):
            # check if result should be plotted in PDF
            if self.plot_pdf_path != "":
                pp = PdfPages(os.path.join(self.plot_pdf_path, "training_epoch_" + str(epoch) + "_visualization.pdf"))
            else:
                pp = None
            
            # show only the first image in the batch
            if pp is None:
                # input image
                show_brain(inputs[0].squeeze().cpu().detach().numpy(),  draw_cross=False, cmap=self.cmap)
                plt.suptitle("Input image")
                plt.show()
                if(not torch.all(torch.eq(inputs[0],labels[0]))):
                    show_brain(labels[0].squeeze().cpu().detach().numpy(),  draw_cross = False, cmap=self.cmap)
                    plt.suptitle("Expected reconstruction")
                    plt.show()  
                # reconstructed image
                show_brain(outputs[0].squeeze().cpu().detach().numpy(),  draw_cross = False, cmap=self.cmap)
                plt.suptitle("Reconstructed Image")
                plt.show()
                # statistics
                print("\nStatistics of expected reconstruction:\n(min, max)=({:.4f}, {:.4f})\nmean={:.4f}\nstd={:.4f}".format(
                    labels[0].min(), labels[0].max(), labels[0].mean(), labels[0].std()))
                print("\nStatistics of Reconstructed image:\n(min, max)=({:.4f}, {:.4f})\nmean={:.4f}\nstd={:.4f}".format(
                    outputs[0].min(), outputs[0].max(), outputs[0].mean(), outputs[0].std()))   
                # feature maps
                visualize_feature_maps(encoder_out[0])
                plt.suptitle("Encoder output")
                plt.show()
            else:
                # input image
                fig = show_brain(inputs[0].squeeze().cpu().detach().numpy(),  draw_cross=False, return_fig=True,
                                 cmap=self.cmap)
                plt.suptitle("Input image")
                pp.savefig(fig)
                plt.close(fig)
                if(not torch.all(torch.eq(inputs[0],labels[0]))):
                    fig = show_brain(labels[0].squeeze().cpu().detach().numpy(),  draw_cross = False, cmap=self.cmap)
                    plt.suptitle("Expected reconstruction")
                    pp.savefig(fig)
                    plt.close(fig)
                # reconstructed image
                fig = show_brain(outputs[0].squeeze().cpu().detach().numpy(), draw_cross=False, return_fig=True, cmap=self.cmap)
                plt.suptitle("Reconstructed Image")
                pp.savefig(fig)
                plt.close(fig)
                # feature maps
                if self.plotFeatures:
                    fig = visualize_feature_maps(encoder_out[0], return_fig=True)
                    plt.suptitle("Encoder output")
                    pp.savefig(fig)
                    plt.close(fig)

            # close the PDF
            if pp is not None:
                pp.close()
 
        if(isinstance(self.model, nn.DataParallel)):
            self.model.module.set_debug(False)
        else:
            self.model.set_debug(False)

        return outputs
