import matplotlib.pyplot as plt

def plot_roc_curve(fpr, tpr, roc_auc, class_name='Class name', save_path=None):
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(class_name)
    plt.legend(loc="lower right")
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    return