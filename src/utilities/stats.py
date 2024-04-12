import numpy as np
from scipy import stats
from sklearn import metrics
import torch

def d_prime(auc):
    standard_normal = stats.norm()
    d_prime = standard_normal.ppf(auc) * np.sqrt(2.0)
    return d_prime

def calculate_stats(output, target):
    """Calculate statistics including mAP, AUC, etc.

    Args:
      output: 2d array, (samples_num, classes_num)
      target: 2d array, (samples_num, classes_num)

    Returns:
      stats: list of statistic of each class.
    """

    classes_num = target.shape[-1]
    stats = []

    # Accuracy, only used for single-label classification such as esc-50, not for multiple label one such as AudioSet
    acc = metrics.accuracy_score(np.argmax(target, 1), np.argmax(output, 1))

    # UAR, WAR and F1 Score
    uar = metrics.recall_score(np.argmax(target, 1), np.argmax(output, 1), average='macro')
    war = metrics.recall_score(np.argmax(target, 1), np.argmax(output, 1), average='weighted')
    f1 = metrics.f1_score(np.argmax(target, 1), np.argmax(output, 1), average='macro')
    
    # Print global performance metrics
    print('UAR is {:.2%}, in val func'.format(uar))
    print('WAR is {:.2%}'.format(war))
    print('F1 Score is {:.2%}'.format(f1))

    # Class-wise statistics
    for k in range(classes_num):

        # Average precision
        avg_precision = metrics.average_precision_score(
            target[:, k], output[:, k], average=None)

        # AUC
        try:
            auc = metrics.roc_auc_score(target[:, k], output[:, k], average=None)

            # Precisions, recalls
            (precisions, recalls, thresholds) = metrics.precision_recall_curve(
                target[:, k], output[:, k])

            # FPR, TPR
            (fpr, tpr, thresholds) = metrics.roc_curve(target[:, k], output[:, k])

            save_every_steps = 1000  # Sample statistics to reduce size
            dict = {'precisions': precisions[0::save_every_steps],
                    'recalls': recalls[0::save_every_steps],
                    'AP': avg_precision,
                    'fpr': fpr[0::save_every_steps],
                    'fnr': 1. - tpr[0::save_every_steps],
                    'auc': auc,
                    'acc': acc,  # note acc is not class-wise, just to keep consistent
                    'uar': uar,  # Include global UAR
                    'war': war,  # Include global WAR
                    'f1': f1    # Include global F1
                    }
        except ValueError as e:
            print('Error for class {}: {}'.format(k, e))
            dict = {'precisions': -1,
                    'recalls': -1,
                    'AP': -1,
                    'fpr': -1,
                    'fnr': -1,
                    'auc': -1,
                    'acc': acc,
                    'uar': uar,
                    'war': war,
                    'f1': f1
                    }
            print('class {:s} no true sample'.format(str(k)))
        
        stats.append(dict)

    return stats
