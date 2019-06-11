"""
Settings for re-running the experiments from the paper "Layer-wise
relevance propagation for explaining deep neural network decisions
in MRI-based Alzheimer’s disease classification".

Please note that you need to download the ADNI data from 
http://adni.loni.usc.edu/ and preprocess it using 
https://github.com/ANTsX/ANTs/blob/master/Scripts/antsRegistrationSyNQuick.sh

Please prepare the data, such that you will get three HDF5 files,
consisting of a training, a validation and a holdout (test) set.
Each HDF5 file is required to have 2 datasets, namely X and y,
containing the data matrix and label vector accordingly. We have
included the "Data Split ADNI.ipynb" file as a guideline for data splitting.
Please note that it is highly dependent on the format of your data storage
and needs to be individualized as such. 

Furthermore you will need SPM12 https://www.fil.ion.ucl.ac.uk/spm/software/spm12/
in order to access the Neuromorphometrics atlas.

Arguments:
    model_path: Path to the trained pytorch model parameters
    data_path: Path where the outputs will be stored and retrieved
    ADNI_DIR: Path to the root of your downloaded ADNI data
    train_h5: Path to the training set HDF5 file
    val_h5: Path to the validation set HDF5 file
    holdout_h5: Path to the holdout set HDF5 file
    binary_brain_mask: Path to the mask used for masking the images,
        included in the repository.
    nmm_mask_path: Path to the Neuromorphometrics mask. Needs to be
        acquired from SPM12. Typically located under 
        ~/spm12/tpm/labels_Neuromorphometrics.nii
    nmm_mask_path_scaled: Path to the rescaled Neuromorphometrics mask.


"""

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
