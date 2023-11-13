from math import floor
from typing import Tuple
import torch
import torch.nn as nn
import torchvision.transforms as transforms

from exercise_code.model.hog import HoG


def sliding_window_detection(
    image: torch.Tensor, model: nn.Module, patch_size: Tuple[int, int], stride: int = 1
) -> torch.Tensor:
    
    grayscale_image = transforms.functional.rgb_to_grayscale(image).float()
    block_size = 8
    num_bins = 9
    H_p, W_p = patch_size[0], patch_size[1]
    ########################################################################
    # TODO:                                                                #
    # Perform person detection in a sliding window manner with HoG         #
    # features and a trained classification model. For this                #
    #                                                                      #
    # - get patches of the grayscale image by using the patch_size and     #
    # stride parameters. Be aware of the stride parameter and don't pad    #
    # the image.                                                           #
    #                                                                      #
    # - compute the HoG features for each patch. The function call is      #
    # hog_features = HoG(patches, block_size, num_bins), where             #
    # block_size and num_bins are hard coded to 8 and 9, respectively.     #
    # Use the HoG function on the batch of grey valued patches of shape    #
    # (B, patch_size[0], patch_size[1]) to compute the flattened HoG       #
    # features.                                                            #
    #                                                                      #
    # - use the model for prediction.                                      #
    #                                                                      #
    # - reshape the output of the model to have 2-spatial dimensions       #
    # and classification scores as values.                                 #
    #                                                                      #
    # Input:                                                               #
    # A single input image of size (3, H, W).                              #
    # The stride and patch_size for of the sliding window operation.       #
    # (With the provided parameters, the hog_features fit                  #
    # the expected input size of the classifier)                           #
    # The classification model, which takes a batch of flattened feature   #
    # patches (hog_features is already flattened)                          #
    # Output:                                                              #
    # An collection if classified patches. The ouput shape should be       #
    # (floor((H-(patch_size[0]-1)-1)/stride+1),                            #
    # floor((W-(patch_size[1]-1)-1)/stride+1)).                            #
    #                                                                      #
    ########################################################################
    
    # get patches of the grayscale image by using the patch_size and stride parameters
    patches = []
    for i in range(0, grayscale_image.shape[1] - H_p + 1, stride):
        for j in range(0, grayscale_image.shape[2] - W_p + 1, stride):
            patches.append(grayscale_image[:, i:i+H_p, j:j+W_p])
    
    
    patches = torch.stack(patches)  # (B, 1, H_p, W_p)
    patches = patches.squeeze(1)  # (B, H_p, W_p)

    hog_features = HoG(patches, block_size, num_bins)  # (B, 3780)

    prediction = model(hog_features)  # (B, 1)

    detection_image = prediction.reshape(floor((grayscale_image.shape[1]-(H_p-1)-1)/stride+1), floor((grayscale_image.shape[2]-(W_p-1)-1)/stride+1))
    
    

    ########################################################################
    #                           END OF YOUR CODE                           #
    ########################################################################

    return detection_image
