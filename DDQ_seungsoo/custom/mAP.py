class compute_map():
    def __init__(self):
        pass
    def compute_iou(self,pred_box, gt_box):
        # pred_box and gt_box are [x_min, y_min, x_max, y_max]
        
        # Intersection coordinates
        x_min = max(pred_box[0], gt_box[0])
        y_min = max(pred_box[1], gt_box[1])
        x_max = min(pred_box[2], gt_box[2])
        y_max = min(pred_box[3], gt_box[3])
        
        # Intersection area
        intersection = max(0, x_max - x_min) * max(0, y_max - y_min)
        
        # Predicted box area and Ground truth box area
        pred_area = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
        gt_area = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
        
        # Union area
        union = pred_area + gt_area - intersection
        
        # IoU calculation
        iou = intersection / union if union != 0 else 0
        return iou

    def compute_precision_recall(self,pred_boxes, gt_boxes, iou_threshold=0.5):
        tp, fp, fn = 0, 0, 0
        matched_gt = set()  # To keep track of matched ground truth boxes
        
        for pred_box in pred_boxes:
            best_iou = 0
            best_gt_idx = -1
            
            for i, gt_box in enumerate(gt_boxes):
                iou = self.compute_iou(pred_box, gt_box)
                if iou > best_iou and iou >= iou_threshold and i not in matched_gt:
                    best_iou = iou
                    best_gt_idx = i
            
            if best_gt_idx >= 0:
                tp += 1  # True positive
                matched_gt.add(best_gt_idx)
            else:
                fp += 1  # False positive
        
        # Remaining unmatched ground truth boxes are false negatives
        fn = len(gt_boxes) - len(matched_gt)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        return precision, recall

    def compute_average_precision(self,precisions, recalls):
        # Sort by recall
        recalls, precisions = zip(*sorted(zip(recalls, precisions)))

        ap = 0
        prev_recall = 0
        for p, r in zip(precisions, recalls):
            ap += p * (r - prev_recall)
            prev_recall = r
        
        return ap

    def calculate_map(self,pred_boxes_list, gt_boxes_list, iou_threshold=0.5):
        all_precisions = []
        all_recalls = []

        for pred_boxes, gt_boxes in zip(pred_boxes_list, gt_boxes_list):
            precision, recall = self.compute_precision_recall(pred_boxes, gt_boxes, iou_threshold)
            all_precisions.append(precision)
            all_recalls.append(recall)
        
        return self.compute_average_precision(all_precisions, all_recalls)
