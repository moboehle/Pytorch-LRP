# Layer-wise relevance propagation for explaining deep neural network decisions in MRI-based Alzheimer’s disease classification

**Moritz Böhle, Fabian Eitel, Martin Weygandt, and Kerstin Ritter**

**Preprint:** https://arxiv.org/abs/1903.07317

**Abstract:** Deep neural networks have led to state-of-the-art results in many medical imaging tasks including Alzheimer's disease (AD) detection based on structural magnetic resonance imaging (MRI) data. However, the network decisions are often perceived as being highly non-transparent making it difficult to apply these algorithms in clinical routine. In this study, we propose using layer-wise relevance propagation (LRP) to visualize convolutional neural network decisions for AD based on MRI data. Similarly to other visualization methods, LRP produces a heatmap in the input space indicating the importance / relevance of each voxel contributing to the final classification outcome. In contrast to susceptibility maps produced by guided backpropagation ("Which change in voxels would change the outcome most?"), the LRP method is able to directly highlight positive contributions to the network classification in the input space. In particular, we show that (1) the LRP method is very specific for individuals ("Why does this person have AD?") with high inter-patient variability, (2) there is very little relevance for AD in healthy controls and (3) areas that exhibit a lot of relevance correlate well with what is known from literature. To quantify the latter, we compute size-corrected metrics of the summed relevance per brain area, e.g. relevance density or relevance gain. Although these metrics produce very individual 'fingerprints' of relevance patterns for AD patients, a lot of importance is put on areas in the temporal lobe including the hippocampus. After discussing several limitations such as sensitivity towards the underlying model and computation parameters, we conclude that LRP might have a high potential to assist clinicians in explaining neural network decisions for diagnosing AD (and potentially other diseases) based on structural MRI data.

## Requirements

In order to run the code, standard pytorch packages and Python 3 are needed.
Moreover, add a settings.py file to the repo, containing the data paths and so forth as follows:

Please use the example settings.py with more information.

```python
settings = {
    "model_path": INSERT,
    "data_path": INSERT,
    "ADNI_DIR": INSERT,
    "train_h5": INSERT,
    "val_h5": INSERT,
    "holdout_h5": INSERT,
    "binary_brain_mask": "binary_brain_mask.nii.gz",
    "nmm_mask_path": "~/spm12/tpm/labels_Neuromorphometrics.nii",
    "nmm_mask_path_scaled": "nmm_mask_rescaled.nii"
}
```

With the "Evaluate GB and LRP" notebook, the heatmap results and the summed scores per area can be calculated.
The notebooks "Plotting result graphs" and "Plotting brain maps" can be used to calculate and plot the results according to the defined metrics and show the heatmaps of individual patient's brains and average heatmaps according to LRP and GB.

## Quickstart

You can use the visualization methods in this repository on your own model (PyTorch; for a Keras implementation, see heatmapping.org) like this:

```python
model = Net()
model.load_state_dict(torch.load("./mymodel"))
# Convert to innvestigate model
inn_model = InnvestigateModel(model, lrp_exponent=2,
                                method="e-rule",
                                beta=.5)
model_prediction, heatmap = inn_model.innvestigate(in_tensor=data)
```

`heatmap` contains the relevance heatmap. The methods should work for 2D and 3D images alike, see the MNIST example notebook or the LRP and GB evaluation notebook for an example with MRI images.

### Docker

To run [MNIST example.ipynb](./MNIST%20example.ipynb) in a Docker container (using only CPU) follow the steps below:

```sh
cd docker/
docker-compose up --detach
```

Visit [localhost:7700](http://localhost:7700) in your browser to open Jupyter.

## Code Structure

The repository consists of the general LRP wrapper ([innvestigator.py](innvestigator.py) and [inverter_util.py](inverter_util.py)), a simple example for applying the wrapper in the case of MNIST data, and the evaluation notebook for obtaining the heatmap results discussed in the article.

## Heatmaps

The methods for obtaining the heatmaps are shown in the notebook **Evaluate GB and LRP**

## Data

The MRI scans used for training are from the [Alzheimer Disease Neuroimaging Initiative (ADNI)](http://adni.loni.usc.edu/). The data is free but you need to apply for access on http://adni.loni.usc.edu/. Once you have an account, go [here](http://adni.loni.usc.edu/data-samples/access-data/) and log in. Settings.py gives information about the required data format.

## Citation

If you use our code, please cite us. The code is published under the BSD License 2.0.
