import torch


def fill_hog_bins(gradient_norm: torch.Tensor, gradient_angle: torch.Tensor, num_bins: int) -> torch.Tensor:
    assert gradient_norm.shape == gradient_angle.shape
    device = gradient_norm.device

    ########################################################################
    # TODO:                                                                #
    # Based on the given gradient norm and angle, fill the Histogram of    #
    # Orientatied Gradients bins.                                          #
    # For this, first determine the two bins a gradient should be part of  #
    # based on the gradient angle. Then, based on the distance to the bins #
    # fill the bins with a weighting of the gradient norm.                 #
    # Input:                                                               #
    # Both gradient_norm and gradient_angle have the shape (B, N), where   #
    # N is the flatten cell.                                               #
    # The angle is given in degrees with values in the range [0.0, 180.0). #
    # (the angles 0.0 and 180.0 are equivalent.)                           #
    # Output:                                                              #
    # The output is a histogram over the flattened cell with num_bins      #
    # quantized values and should have the shape (B, num_bins)             #
    #                                                                      #
    # NOTE: Keep in mind the cyclical nature of the gradient angle and     #
    # its effects on the bins.                                             #
    # NOTE: make sure, the histogram_of_oriented_gradients are on the same #
    # device as the gradient inputs. In general be mindful of the device   #
    # of the tensors.                                                      #
    # histogram_of_oriented_gradients = ...                                #
    ########################################################################

    histogram_of_oriented_gradients = torch.zeros((gradient_norm.shape[0], num_bins), device=device)
    bin_size = 180 / num_bins


    gradient_angle = gradient_angle % 180
    bin1 = (gradient_angle / bin_size).long() % num_bins
    bin2 = (bin1 + 1) % num_bins
    dist1 = 1 - (gradient_angle - bin1 * bin_size) / bin_size
    dist2 = 1 - (bin2 * bin_size - gradient_angle) / bin_size
    dist2[bin2 == 0] = 1 - (180 - gradient_angle)[bin2 == 0] / bin_size

    histogram_of_oriented_gradients.scatter_add_(1, bin1, gradient_norm * dist1)
    histogram_of_oriented_gradients.scatter_add_(1, bin2, gradient_norm * dist2)

    ########################################################################
    #                           END OF YOUR CODE                           #
    ########################################################################

    return histogram_of_oriented_gradients
