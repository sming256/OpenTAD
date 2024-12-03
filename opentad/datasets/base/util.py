import numpy as np


def filter_same_annotation(annotation):
    gt_segments = []
    gt_labels = []
    gt_both = []
    for gt_segment, gt_label in zip(annotation["gt_segments"].tolist(), annotation["gt_labels"].tolist()):
        if (gt_segment, gt_label) not in gt_both:
            gt_segments.append(gt_segment)
            gt_labels.append(gt_label)
            gt_both.append((gt_segment, gt_label))
        else:
            continue

    annotation = dict(
        gt_segments=np.array(gt_segments, dtype=np.float32),
        gt_labels=np.array(gt_labels, dtype=np.int32),
    )
    return annotation


if __name__ == "__main__":
    anno1 = dict(gt_segments=np.array([[3, 5], [3, 6], [3, 5]]), gt_labels=np.array([0, 1, 0]))
    print(filter_same_annotation(anno1))
    # output should be:
    # 'gt_segments': array([[3., 5.], [3., 6.]], dtype=float32),
    # 'gt_labels': array([0, 1], dtype=int32)}

    anno2 = dict(gt_segments=np.array([[3, 5], [3, 6], [3, 5]]), gt_labels=np.array([0, 1, 2]))
    print(filter_same_annotation(anno2))
    # output should be:
    # 'gt_segments': array([[3., 5.], [3., 6.], [3., 5.]], dtype=float32),
    # 'gt_labels': array([0, 1, 2], dtype=int32)}

    anno3 = dict(gt_segments=np.array([[3, 5], [3, 5], [3, 5]]), gt_labels=np.array([0, 1, 1]))
    print(filter_same_annotation(anno3))
    # output should be:
    # 'gt_segments': array([[3., 5.], [3., 5.]], dtype=float32),
    # 'gt_labels': array([0, 1], dtype=int32)}
