# Visualizing evidence for Alzheimer's disease
## Visualizing evidence for Alzheimer's disease in deep neural networks trained on structural MRI data

**Moritz Böhle, Fabian Eitel, Martin Weygandt, and Kerstin Ritter**

**Preprint:** Coming soon

**Abstract:** Deep neural networks have led to state-of-the-art results in many medical imaging tasks including Alzheimer's disease (AD) detection based on structural magnetic resonance imaging (MRI) data. However, the network decisions are often perceived as being highly non-transparent making it difficult to apply these algorithms in clinical routine. In this study, we propose using layer-wise relevance propagation (LRP) to visualize convolutional neural network decisions for AD based on MRI data. Similarly to other visualization methods, LRP produces a heatmap in the input space indicating the importance of each voxel contributing to the final classification outcome. In contrast to susceptibility maps produced by guided backpropagation ("Which change in voxels would change the outcome most?"), the LRP method is able to directly highlight positive contributions to the network classification in the input space. Thus, the highlighted areas can be interpreted as the 'positive evidence' used by the network for deciding whether an individual has AD. We find that this LRP-evidence indeed fulfills those expectations that one would have towards AD evidence: (1) it is very specific for individuals ("Why does this person have AD?") with high inter-patient variability, (2) there is very little evidence for AD in healthy controls and (3) areas that exhibit a lot of evidence correlate well with what is known from literature. To quantify the latter, we compute size-corrected metrics of the summed evidence per brain area, e.g. the 'evidence density' or 'evidence gain'. Although these metrics produce very individual 'fingerprints' of relevance patterns for AD patients, a lot of importance is put on areas in the temporal lobe including hippocampus and amygdala. We conclude that LRP provides a powerful tool for assisting clinicians in finding evidence for AD (and potentially other diseases) in structural MRI data. 


## Quickstart

You can use the visualization methods in this repository on your own model (PyTorch; for a Keras implementation, see heatmapping.org) like this:

    model = Net()
    model.load_state_dict(torch.load("./mymodel"))
    # Convert to innvestigate model
    inn_model = InnvestigateModel(model, lrp_exponent=2,
                                  method="e-rule",
                                  beta=.5)
    model_prediction, heatmap = inn_model.innvestigate(in_tensor=data)
    
`heatmap` contains the relevance heatmap. The methods should work for 2D and 3D images alike, see the MNIST example notebook.
    

## Code Structure

Coming soon.

## Trained Model and Heatmaps

Coming soon.

## Data

The MRI scans used for training are from the [Alzheimer Disease Neuroimaging Initiative (ADNI)](http://adni.loni.usc.edu/). The data is free but you need to apply for access on http://adni.loni.usc.edu/. Once you have an account, go [here](http://adni.loni.usc.edu/data-samples/access-data/) and log in. 



## Requirements

Coming soon.


## Citation

If you use our code, please cite us. 

  
