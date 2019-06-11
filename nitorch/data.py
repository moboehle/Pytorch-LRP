import numpy as np
import nibabel as nib
from scipy.ndimage.interpolation import zoom
import matplotlib.pyplot as plt
import nibabel
import os

def load_nifti(file_path, dtype=np.float32, incl_header=False, z_factor=None, mask=None):
    """
    Loads a volumetric image in nifti format (extensions .nii, .nii.gz etc.)
    as a 3D numpy.ndarray.
    
    Args:
        file_path: absolute path to the nifti file
        
        dtype(optional): datatype of the loaded numpy.ndarray
        
        incl_header(bool, optional): If True, the nifTI object of the 
        image is also returned.
        
        z_factor(float or sequence, optional): The zoom factor along the
        axes. If a float, zoom is the same for each axis. If a sequence,
        zoom should contain one value for each axis.
        
        mask(ndarray, optional): A mask with the same shape as the
        original image. If provided then the mask is element-wise
        multiplied with the image ndarray
    
    Returns:
        3D numpy.ndarray with axis order (saggital x coronal x axial)
    """
    
    img = nib.load(file_path)
    struct_arr = img.get_data().astype(dtype)
    
    # replace infinite values with 0
    if np.inf in struct_arr:
        struct_arr[struct_arr == np.inf] = 0.
    
    # replace NaN values with 0    
    if np.isnan(struct_arr).any() == True:
        struct_arr[np.isnan(struct_arr)] = 0.
        
    if mask is not None:
        struct_arr *= mask
        
    if z_factor is not None:
        struct_arr = zoom(struct_arr, z_factor)
    
    if incl_header:
        return struct_arr, img
    else:
        return struct_arr


def show_brain(img, cut_coords=None, 
               figsize=(10,5), cmap="nipy_spectral",
               draw_cross = True,
               return_fig = False
               ):
    """Displays 2D cross-sections of a 3D image along all 3 axis
    Arg:
        img: can be (1) 3-dimensional numpy.ndarray
                    (2) nibabel.Nifti1Image object
                    (3) path to the image file stored in nifTI format

        cut_coords (optional): The voxel coordinates
        of the axes where the cross-section cuts will be performed. 
        Should be a 3-tuple: (x, y, z). Default is the center = img_shape/2 
        
        figsize (optional): matplotlib figsize. Default is (10,5)
        cmap (optional): matplotlib colormap to be used
        
        draw_cross (optional): Draws horizontal and vertical lines which
        show where the cross-sections have been performed. D
        

        example:
            >>> show_brain(img, figsize=(7, 3), draw_cross=False)
            >>> plt.show()
        """
    
    if(isinstance(img, str) and os.path.isfile(img)):
        img_arr = load_nifti(img)
    elif(isinstance(img, nibabel.Nifti1Image)):
        img_arr = img.get_data()
        
    elif(isinstance(img, np.ndarray)):
        assert img.ndim == 3, "The numpy.ndarray must be 3-dimensional with shape (H x W x Z)"
        img_arr = img
    else:
        raise TypeError("Invalid type provided for 'img'- {}. \
Either provide a 3-dimensional numpy.ndarray of a MRI image or path to \
the image file stored as a nifTI format.".format(type(img)))
        
    # print(img_arr.shape)
    # img_arr = np.moveaxis(img_arr, 0, 1)
    # print(img_arr.shape)

    x_len, y_len, z_len = img_arr.shape
    # if cut_coordinates is not specified set it to the center of the image
    if(cut_coords == None):
        cut_coords = (x_len//2, y_len//2, z_len//2)

    f, ax = plt.subplots(nrows=1, ncols=3, figsize=figsize)

    ax[0].set_title("Saggital cross-section at x={}".format(cut_coords[0]))
    ax[0].imshow(
         np.rot90(img_arr[cut_coords[0],:,:]), cmap=cmap, aspect="equal")
    #draw cross
    if(draw_cross):
        ax[0].axvline(x=cut_coords[1], color='k', linewidth=1)
        ax[0].axhline(y=cut_coords[2], color='k', linewidth=1)

    ax[1].set_title("Coronal cross-section at y={}".format(cut_coords[1]))
    ax[1].imshow(
        np.rot90(img_arr[:,cut_coords[1],:]), cmap=cmap, aspect="equal")
    ax[1].text(0.05, 0.95,'L', 
        horizontalalignment='left', verticalalignment='top',
        transform=ax[1].transAxes
        , bbox=dict(facecolor='white')
        )
    ax[1].text(0.95, 0.95,'R', 
        horizontalalignment='right', verticalalignment='top'
        , transform=ax[1].transAxes
        , bbox=dict(facecolor='white')
        )
    #draw cross
    if(draw_cross):
        ax[1].axvline(x=cut_coords[0], color='k', linewidth=1)
        ax[1].axhline(y=cut_coords[2], color='k', linewidth=1)

    ax[2].set_title("Axial cross-section at z={}".format(cut_coords[2]))
    ax[2].imshow(
        np.rot90(img_arr[:,:,cut_coords[2]]), cmap=cmap, aspect="equal"
        )
    ax[2].text(0.05, 0.95,'L'
        , horizontalalignment='left', verticalalignment='top'
        , transform=ax[2].transAxes
        , bbox=dict(facecolor='white')
        )
    ax[2].text(0.95, 0.95,'R', 
        horizontalalignment='right', verticalalignment='top'
        , transform=ax[2].transAxes
        , bbox=dict(facecolor='white')
        )
    #draw cross
    if(draw_cross):
        ax[2].axvline(x=cut_coords[0], color='k', linewidth=1)
        ax[2].axhline(y=cut_coords[1], color='k', linewidth=1)

    plt.tight_layout()
    if return_fig:
        return f
