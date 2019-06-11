from sklearn.metrics import recall_score, roc_curve, auc

def specificity(y_true, y_pred):
    return recall_score(y_true, y_pred, pos_label=0)


def sensitivity(y_true, y_pred):
    return recall_score(y_true, y_pred, pos_label=1)


def balanced_accuracy(y_true, y_pred):
    spec = specificity(y_true, y_pred)
    sens = sensitivity(y_true, y_pred)
    return (spec + sens) / 2

def auc_score(y_true, y_pred):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    return auc(fpr, tpr)