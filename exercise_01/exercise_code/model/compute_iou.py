import torch



def calculate_area(left, top, right, bottom):
    return max(0,(right - left)) * max(0,(bottom - top))

def compute_iou(bbox_1: torch.Tensor, bbox_2: torch.Tensor) -> torch.Tensor:
    assert bbox_1.shape == bbox_2.shape

    ########################################################################
    # TODO:                                                                #
    # Compute the intersection over union (IoU) for two batches of         #
    # bounding boxes, each of shape (B, 4). The result should be a tensor  #
    # of shape (B,).                                                       #
    # NOTE: the format of the bounding boxes is (ltrb), meaning            #
    # (left edge, top edge, right edge, bottom edge). Remember the         #
    # orientation of the image coordinates.                                #
    # NOTE: First calculate the intersection and use this to compute the   #
    # union                                                                #
    # iou = ...                                                            #
    ########################################################################

    iou = torch.zeros(bbox_1.shape[0])
    for i in range(bbox_1.shape[0]):
        left = max(bbox_1[i,0], bbox_2[i,0])
        right = min(bbox_1[i,2], bbox_2[i,2])
        top = max(bbox_1[i,1], bbox_2[i,1])
        bottom = min(bbox_1[i,3], bbox_2[i,3])
        intersection = calculate_area(left, top, right, bottom)
        union = calculate_area(bbox_1[i,0], bbox_1[i,1], bbox_1[i,2], bbox_1[i,3]) + calculate_area(bbox_2[i,0], bbox_2[i,1], bbox_2[i,2], bbox_2[i,3]) - intersection
        iou[i] = intersection / union

    ########################################################################
    #                           END OF YOUR CODE                           #
    ########################################################################

    return iou
