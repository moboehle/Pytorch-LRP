import numpy as np
import numbers
import torch
from scipy.ndimage.interpolation import rotate
from scipy.ndimage.interpolation import zoom


def normalize_float(x, min=-1):
    """ 
    Function that performs min-max normalization on a `numpy.ndarray` 
    matrix. 
    """
    if min == -1:
        norm = (2 * (x - np.min(x)) / (np.max(x) - np.min(x))) - 1
    elif min == 0:
        if np.max(x) == 0 and np.min(x) == 0:
            norm = x
        else:
            norm = (x - np.min(x)) / (np.max(x) - np.min(x))
    return norm


def normalize_float_torch(x, min=-1):
    '''
    Function that performs min-max normalization on a Pytorch tensor 
    matrix. Can also deal with Pytorch dictionaries where the data
    matrix key is 'image'.
    '''
    import torch
    if min == -1:
        norm = (2 * (x - torch.min(x)) / (torch.max(x) - torch.min(x))) - 1
    elif min == 0:
        if torch.max(x) == 0 and torch.min(x) == 0:
            norm = x
        else:    
            norm = (x - torch.min(x)) / (torch.max(x) - torch.min(x))
    return norm


def normalization_factors(data, train_idx, shape, mode="slice"):
    """ 
    Shape should be of length 3. 
    mode : either "slice" or "voxel" - defines the granularity of the 
    normalization. Voxelwise normalization does not work well with only
    linear registered data.
    """
    print("Computing the normalization factors of the training data..")
    if mode == "slice":
        axis = (0, 1, 2, 3)
    elif mode == "voxel":
        axis = 0
    else:
        raise NotImplementedError("Normalization mode unknown.")
    samples = np.zeros(
        [len(train_idx), 1, shape[0], shape[1], shape[2]], dtype=np.float32
    )
    for c, value in enumerate(train_idx):
        samples[c] = data[value]["image"].numpy()
    mean = np.mean(samples, axis=axis)
    std = np.std(samples, axis=axis)
    return np.squeeze(mean), np.squeeze(std)


class CenterCrop(object):
    """Crops the given 3D ndarray Image at the center.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w, d), a cube crop (size, size, size) is
            made.
    """
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size), int(size))
        else:
            self.size = np.asarray(size)
        assert len(self.size) == 3, "The `size` must be a tuple of length 3 but is \
length {}".format(len(self.size))
        
    def __call__(self, img):
        """
        Args:
            3D ndarray Image : Image to be cropped.
        Returns:
            3D ndarray Image: Cropped image.
        """     
        # if the 4th dimension of the image is the batch then ignore that dim 
        if len(img.shape) == 4:
            img_size = img.shape[1:]
        elif len(img.shape) == 3:
            img_size = img.shape
        else:
            raise ValueError("The size of the image can be either 3 dimension or 4\
dimension with one dimension as the batch size")
            
        # crop only if the size of the image is bigger than the size to be cropped to.
        if all(img_size >= self.size):
            slice_start = (img_size - self.size)//2
            slice_end = self.size + slice_start
            cropped = img[slice_start[0]:slice_end[0],
                          slice_start[1]:slice_end[1],
                          slice_start[2]:slice_end[2]
                         ]
            if len(img.shape) == 4:
                cropped = np.expand_dims(cropped, 0)
        else:
            cropped = img
        
        return cropped

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)
    

class Normalize(object):
    """
    Normalize tensor with first and second moments.
    By default will only normalize on non-zero voxels. Set 
    masked = False if this is undesired.
    """

    def __init__(self, mean, std=1, masked=True, eps=1e-10):
        self.mean = mean
        self.std = std
        self.masked = masked
        # set epsilon only if using std scaling
        self.eps = eps if np.all(std) != 1 else 0

    def __call__(self, image):
        if self.masked:
            image = self.zero_masked_transform(image)
        else:
            image = self.apply_transform(image)
        return image

    def denormalize(self, image):
        image = image * (self.std + self.eps) + self.mean
        return image

    def apply_transform(self, image):
        return (image - self.mean) / (self.std + self.eps)

    def zero_masked_transform(self, image):
        """ Only apply transform where input is not zero. """
        img_mask = image == 0
        # do transform
        image = self.apply_transform(image)
        image[img_mask] = 0.
        return image


class IntensityRescale:
    """
    Rescale image itensities between 0 and 1 for a single image.

    Arguments:
        masked: applies normalization only on non-zero voxels. Default
            is True.
        on_gpu: speed up computation by using GPU. Requires torch.Tensor
             instead of np.array. Default is False.
    """

    def __init__(self, masked=True, on_gpu=False):
        self.masked = masked
        self.on_gpu = on_gpu

    def __call__(self, image):
        if self.masked:
            image = self.zero_masked_transform(image)
        else:
            image = self.apply_transform(image)

        return image

    def apply_transform(self, image):
        if self.on_gpu:
            return normalize_float_torch(image, min=0)
        else:
            return normalize_float(image, min=0)

    def zero_masked_transform(self, image):
        """ Only apply transform where input is not zero. """
        img_mask = image == 0
        # do transform
        image = self.apply_transform(image)
        image[img_mask] = 0.
        return image


########################################################################
# Data augmentations
########################################################################

class ToTensor(object):
    """
    Convert ndarrays to Tensors.
    Expands channel axis
    # numpy image: H x W x Z
    # torch image: C x H x W x Z
    """

    def __call__(self, image):
        image = torch.from_numpy(image).unsqueeze(0)
        image = image.float()
        return image


class Flip:
    """
    Flip the input along a given axis.

    Arguments:
        axis: axis to flip over. Default is 0
        prob: probability to flip the image. Executes always when set to
             1. Default is 0.5
    """
    def __init__(self, axis=0, prob=0.5):
        self.axis = axis
        self.prob = prob

    def __call__(self, image):
        rand = np.random.uniform()
        if rand <= self.prob:
            augmented = np.flip(image, axis=self.axis).copy()
        else:
            augmented = image
        return augmented


class SagittalFlip(Flip):
    """
    Flip image along the sagittal axis (x-axis). 
    Expects input shape (X, Y, Z).
    """
    def __init__(self, prob=0.5):
        super().__init__(axis=0, prob=prob)
    
    def __call__(self, image):
        assert(len(image.shape) == 3)
        return super().__call__(image)

class CoronalFlip(Flip):
    """
    Flip image along the coronal axis (y-axis). 
    Expects input shape (X, Y, Z).
    """
    def __init__(self, prob=0.5):
        super().__init__(axis=1, prob=prob)

    def __call__(self, image):
        assert(len(image.shape) == 3)
        return super().__call__(image)


class AxialFlip(Flip):
    """
    Flip image along the axial axis (z-axis). 
    Expects input shape (X, Y, Z).
    """
    def __init__(self, prob=0.5):
        super().__init__(axis=2, prob=prob)

    def __call__(self, image):
        assert(len(image.shape) == 3)
        return super().__call__(image)


class Rotate:
    """ 
    Rotate the input along a given axis.

    Arguments:
        axis: axis to rotate. Default is 0
        deg: min and max rotation angles in degrees. Randomly rotates 
            within that range. Can be scalar, list or tuple. In case of 
            scalar it rotates between -abs(deg) and abs(deg). Default is
            (-3, 3).
    """
    def __init__(self, axis=0, deg=(-3, 3)):
        if axis == 0:
            self.axes = (1, 0)
        elif axis == 1:
            self.axes = (2, 1)
        elif axis == 2:
            self.axes = (0, 2)

        if isinstance(deg, tuple) or isinstance(deg, list):
            assert(len(deg) == 2)
            self.min_rot = np.min(deg)
            self.max_rot = np.max(deg)
        else:
            self.min_rot = -int(abs(deg))
            self.max_rot = int(abs(deg))

    def __call__(self, image):
        rand = np.random.randint(self.min_rot, self.max_rot + 1)
        augmented = rotate(
            image,
            angle=rand,
            axes=self.axes,
            reshape=False
            ).copy()
        return augmented


class SagittalRotate(Rotate):
    """
    Rotate image's sagittal axis (x-axis). 
    Expects input shape (X, Y, Z).
    """
    def __init__(self, deg=(-3, 3)):
        super().__init__(axis=0, deg=deg)


class CoronalRotate(Rotate):
    """
    Rotate image's coronal axis (y-axis). 
    Expects input shape (X, Y, Z).
    """
    def __init__(self, deg=(-3, 3)):
        super().__init__(axis=1, deg=deg)


class AxialRotate(Rotate):
    """
    Rotate image's axial axis (z-axis). 
    Expects input shape (X, Y, Z).
    """
    def __init__(self, deg=(-3, 3)):
        super().__init__(axis=2, deg=deg)


class Translate:
    """
    Translate the input along a given axis.

    Arguments:
        axis: axis to rotate. Default is 0
        dist: min and max translation distance in pixels. Randomly 
            translates within that range. Can be scalar, list or tuple. 
            In case of scalar it translates between -abs(dist) and 
            abs(dist). Default is (-3, 3).
    """
    def __init__(self, axis=0, dist=(-3, 3)):
        self.axis = axis

        if isinstance(dist, tuple) or isinstance(dist, list):
            assert(len(dist) == 2)
            self.min_trans = np.min(dist)
            self.max_trans = np.max(dist)
        else:
            self.min_trans = -int(abs(dist))
            self.max_trans = int(abs(dist))

    def __call__(self, image):
        rand = np.random.randint(self.min_trans, self.max_trans + 1)
        augmented = np.zeros_like(image)
        if self.axis == 0:
            if rand < 0:
                augmented[-rand:, :] = image[:rand, :]
            elif rand > 0:
                augmented[:-rand, :] = image[rand:, :]
            else:
                augmented = image
        elif self.axis == 1:
            if rand < 0:
                augmented[:,-rand:, :] = image[:,:rand, :]
            elif rand > 0:
                augmented[:,:-rand, :] = image[:,rand:, :]
            else:
                augmented = image
        elif self.axis == 2:
            if rand < 0:
                augmented[:,:,-rand:] = image[:,:,:rand]
            elif rand > 0:
                augmented[:,:,:-rand] = image[:,:,rand:]
            else:
                augmented = image
        return augmented


class SagittalTranslate(Translate):
    """
    Translate image along the sagittal axis (x-axis).
    Expects input shape (X, Y, Z).
    """
    def __init__(self, dist=(-3, 3)):
        super().__init__(axis=0, dist=dist)


class CoronalTranslate(Translate):
    """
    Translate image along the coronal axis (y-axis).
    Expects input shape (X, Y, Z).
    """
    def __init__(self, dist=(-3, 3)):
        super().__init__(axis=1, dist=dist)


class AxialTranslate(Translate):
    """
    Translate image along the axial axis (z-axis).
    Expects input shape (X, Y, Z).
    """
    def __init__(self, dist=(-3, 3)):
        super().__init__(axis=2, dist=dist)
        