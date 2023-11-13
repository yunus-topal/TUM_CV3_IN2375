import torch

from exercise_code.model.compute_iou import compute_iou


def non_maximum_suppression(bboxes: torch.Tensor, scores: torch.Tensor, threshold: float) -> torch.Tensor:
    ########################################################################
    # TODO:                                                                #
    # Compute the non maximum suppression                                  #
    # Input:                                                               #
    # bounding boxes of shape B,4                                          #
    # scores of shape B                                                    #
    # threshold for iou: if the overlap is bigger, only keep one of the    #
    # bboxes                                                               #
    # Output:                                                              #
    # bounding boxes of shape B_,4                                         #
    ########################################################################
    bboxes_nms = torch.empty(0, 4)
    for i in range(bboxes.shape[0]):
        discard = False
        for j in range(bboxes.shape[0]):
            iou = compute_iou(bboxes[i].unsqueeze(0), bboxes[j].unsqueeze(0))
            if iou > threshold:
                if scores[i] < scores[j]:
                    discard = True
                    break
        
        if not discard:
            bboxes_nms = torch.cat((bboxes_nms, bboxes[i].unsqueeze(0)), dim=0)


    ########################################################################
    #                           END OF YOUR CODE                           #
    ########################################################################

    return bboxes_nms
