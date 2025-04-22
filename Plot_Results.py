import matplotlib.pyplot as plt
import numpy as np
from prettytable import PrettyTable
import seaborn as sn
import pandas as pd
from itertools import cycle
from sklearn import metrics
from sklearn.metrics import roc_curve, confusion_matrix
import cv2 as cv

from Image_Reults import Image_Results1
from Segmentation_Evaluation import Segmentation_Evaluation


def addlabels(x,y):
    for i in range(len(x)):
        plt.text(i-0.20,np.round(y[i],3)/2,str(np.round(y[i],2))+'%',Bbox = dict(facecolor = 'white', alpha =.8))

def Plot_Results():
    Eval = np.load('Eval_all.npy', allow_pickle=True)
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1-Score', 'MCC']
    Graph_Term = [0,1,2, 3, 4, 5,6,7,8, 9]
    Algorithm = ['TERMS', 'DO-AAMENet', 'EOO-AAMENet', 'AVOA-AAMENet', 'AZOA-AAMENet', 'BF-AZO-AAMENet']
    Classifier = ['TERMS', 'LSTM', 'RNN','ResNet', 'AA-MENet','BF-AZO-AAMENet']

    value = Eval[4, :, 4:]
    value[:, :-1] = value[:, :-1] * 100
    Table = PrettyTable()
    Table.add_column(Algorithm[0], Terms)
    for j in range(len(Algorithm) - 1):
        Table.add_column(Algorithm[j + 1], value[j, :])
    print('-------------------------------------------------- Algorithm Comparison',
          '--------------------------------------------------')
    print(Table)

    Table = PrettyTable()
    Table.add_column(Classifier[0], Terms)
    for j in range(len(Classifier) - 1):
        Table.add_column(Classifier[j + 1], value[len(Algorithm) + j - 1, :])
    print('---------------------------------------------------Classifier Comparison',
          '--------------------------------------------------')
    print(Table)

    Eval = np.load('Eval_all.npy', allow_pickle=True)
    epoch = [100, 200, 300, 400, 500]
    for j in range(len(Graph_Term)):
        Graph = np.zeros((Eval.shape[0], Eval.shape[1]))
        for k in range(Eval.shape[0]):
            for l in range(Eval.shape[1]):
                if Graph_Term[j] == 9:
                    Graph[k, l] = Eval[k, l, Graph_Term[j] + 4]
                else:
                    Graph[k, l] = Eval[k, l, Graph_Term[j] + 4] * 100

        plt.plot(epoch, Graph[:, 5], '-.', color='#65fe08', linewidth=3, marker='*', markerfacecolor='blue', markersize=10,
                 label="LSTM")
        plt.plot(epoch, Graph[:, 6], '-.', color='#4e0550', linewidth=3, marker='*', markerfacecolor='red', markersize=10,
                 label="RNN")
        plt.plot(epoch, Graph[:, 7], '-.', color='#f70ffa', linewidth=3, marker='*', markerfacecolor='green', markersize=10,
                 label="ResNet")
        plt.plot(epoch, Graph[:, 8], '-.', color='#a8a495', linewidth=3, marker='*', markerfacecolor='yellow', markersize=10,
                 label="AA-MENet")
        plt.plot(epoch, Graph[:, 9], '-.', color='#004577', linewidth=3, marker='*', markerfacecolor='cyan', markersize=10,
                 label="BF-AZO-AAMENet")

        plt.xlabel('Epochs')
        plt.ylabel(Terms[Graph_Term[j]])
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
                   ncol=3, fancybox=True, shadow=True)
        path1 = "./Results/%s_line_1.png" % ( Terms[Graph_Term[j]])
        plt.savefig(path1)
        plt.show()

def plot_Method():
    Eval = np.load('Eval_all.npy', allow_pickle=True)
    x = np.arange(5)
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1-Score', 'MCC']
    for a in range(10):
        y = Eval[:, 4, a + 4] * 100
        if a == 0:
            markerline, stemlines, baseline = plt.stem(y, linefmt='darkturquoise', basefmt='grey', markerfmt='*')
        elif a == 1:
            markerline, stemlines, baseline = plt.stem(y, linefmt='orange', basefmt='grey', markerfmt='*')
        elif a == 2:
            markerline, stemlines, baseline = plt.stem(y, linefmt='darkslategrey', basefmt='grey', markerfmt='*')
        elif a == 3:
            markerline, stemlines, baseline = plt.stem(y, linefmt='maroon', basefmt='grey', markerfmt='*')
        elif a == 4:
            markerline, stemlines, baseline = plt.stem(y, linefmt='saddlebrown', basefmt='grey', markerfmt='*')
        elif a == 5:
            markerline, stemlines, baseline = plt.stem(y, linefmt='greenyellow', basefmt='grey', markerfmt='*')
        elif a == 6:
            markerline, stemlines, baseline = plt.stem(y, linefmt='dodgerblue', basefmt='grey', markerfmt='*')
        elif a == 7:
            markerline, stemlines, baseline = plt.stem(y, linefmt='crimson', basefmt='grey', markerfmt='*')
        elif a == 8:
            markerline, stemlines, baseline = plt.stem(y, linefmt='deeppink', basefmt='grey', markerfmt='*')
        else:
            markerline, stemlines, baseline = plt.stem(y, linefmt='darkturquoise', basefmt='grey', markerfmt='*')
        plt.setp(stemlines, 'linewidth', 8)
        plt.setp(baseline, 'linewidth', 8)
        plt.setp(markerline, markersize=10)
        markerline.set_markerfacecolor('None')
        addlabels(x, y)
        plt.xticks(x, ('100', '200', '300', '400', '500'))
        plt.ylabel(Terms[a])
        plt.xlabel('Epochs')
        # plt.title('PROPOSED')
        path1 = "./Results/Classifer_%s.png" % (str(a + 1))
        plt.savefig(path1)
        # plt.yticks(y + 0.25, ('a', 'b', 'c', 'd', 'e','f'))
        plt.show()

def Confusion_matrix():
    # Confusion Matrix
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_true, y_pred)
    sn.set(font_scale=1.2)  # Adjust font size for better readability

    # Use `annot=True` to display values in each cell
    sn.heatmap(cm, annot=True, fmt='d', cmap='Blues', annot_kws={"size": 16})

    # Add labels, title, and ticks
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.xticks(ticks=[0.5, 1.5], labels=['Negative', 'Positive'])
    plt.yticks(ticks=[0.5, 1.5], labels=['Negative', 'Positive'])

    # Show plot
    plt.show()

    # Eval = np.load('Eval_all.npy', allow_pickle=True)
    # value = Eval[3, 4, :5]
    # val = np.asarray([0, 1, 1])
    # data = {'y_Actual': [val.ravel()],
    #         'y_Predicted': [np.asarray(val).ravel()]
    #         }
    # df = pd.DataFrame(data, columns=['y_Actual', 'y_Predicted'])
    # confusion_matrix = pd.crosstab(df['y_Actual'][0], df['y_Predicted'][0], rownames=['Actual'], colnames=['Predicted'])
    # value = value.astype('int')
    #
    # confusion_matrix.values[0, 0] = value[1]
    # confusion_matrix.values[0, 1] = value[3]
    # confusion_matrix.values[1, 0] = value[2]
    # confusion_matrix.values[1, 1] = value[0]
    #
    # sn.heatmap(confusion_matrix, annot=True).set(title='Accuracy = ' + str(Eval[3, 4, 4] * 100)[:5] + '%')
    # sn.heatmap(confusion_matrix, annot=True, fmt='g')
    # sn.plotting_context()
    # path1 = './Results/Confusion.png'
    # plt.savefig(path1)
    # plt.show()

def Plot_ROC():
    lw = 2
    cls = ['LSTM', 'RNN', 'ResNet', 'AA-MENet','BF-AZO-AAMENet']
    colors1 = cycle(["plum", "red", "palegreen", "chocolate", "navy", ])
    colors2 = cycle(["hotpink", "plum", "chocolate", "navy", "red", "palegreen", "violet", "red"])
    for n in range(1):
        for i, color in zip(range(5), colors1):  # For all classifiers
            Predicted = np.load('roc_score.npy', allow_pickle=True)[n][i].astype('float')
            Actual = np.load('roc_act.npy', allow_pickle=True)[n][i].astype('int')
            false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(Actual[:, -1], Predicted[:, -1].ravel())

            auc = metrics.roc_auc_score(Actual[:, -1], Predicted[:,
                                                       -1].ravel())

            plt.plot(
                false_positive_rate1,
                true_positive_rate1,
                color=color,
                lw=lw,
                label="{0}".format(cls[i]),
            )

        plt.plot([0, 1], [0, 1], "k--", lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        path1 = "./Results/_roc_%s.png"  %(str(n+1))
        plt.savefig(path1)
        plt.show()

def Plot_Fitness():
    conv = np.load('Fitness.npy', allow_pickle=True)
    ind = np.argsort(conv[:, conv.shape[1] - 1])
    x = conv[ind[0], :].copy()
    y = conv[4, :].copy()
    conv[4, :] = x
    conv[ind[0], :] = y

    Statistics = ['BEST', 'WORST', 'MEAN', 'MEDIAN', 'STD']
    Algorithm = ['DOA-AAMENet', 'EOO-AAMENet', 'AVOA-AAMENet', 'AZOA-AAMENet', 'BF-AZO-AAMENet']

    Value = np.zeros((conv.shape[0], 5))
    for j in range(conv.shape[0]):
        Value[j, 0] = np.min(conv[j, :])
        Value[j, 1] = np.max(conv[j, :])
        Value[j, 2] = np.mean(conv[j, :])
        Value[j, 3] = np.median(conv[j, :])
        Value[j, 4] = np.std(conv[j, :])

    Table = PrettyTable()
    Table.add_column("ALGORITHMS", Statistics)
    for j in range(len(Algorithm)):
        Table.add_column(Algorithm[j], Value[j, :])
    print('--------------------------------------------------Statistical Analysis--------------------------------------------------')
    print(Table)

    iteration = np.arange(conv.shape[1])
    plt.plot(iteration, conv[0, :], color='r', linewidth=3, marker='>', markerfacecolor='blue', markersize=8,
             label="DOA-AAMENet")
    plt.plot(iteration, conv[1, :], color='g', linewidth=3, marker='>', markerfacecolor='red', markersize=8,
             label="EOO-AAMENet")
    plt.plot(iteration, conv[2, :], color='b', linewidth=3, marker='>', markerfacecolor='green', markersize=8,
             label="AVOA-AAMENet")
    plt.plot(iteration, conv[3, :], color='m', linewidth=3, marker='>', markerfacecolor='yellow', markersize=8,
             label="AZOA-AAMENet")
    plt.plot(iteration, conv[4, :], color='k', linewidth=3, marker='>', markerfacecolor='cyan', markersize=8,
             label="BF-AZO-AAMENet")
    plt.xlabel('Iteration')
    plt.ylabel('Cost Function')
    plt.legend(loc=1)
    path1 = "./Results/conv.png"
    plt.savefig(path1)
    plt.show()

def statistical_analysis(v):
    a = np.zeros((5))
    a[0] = np.min(v)
    a[1] = np.max(v)
    a[2] = np.mean(v)
    a[3] = np.median(v)
    a[4] = np.std(v)
    return a

if __name__ == '__main__':
    # Plot_Results()
    # plot_Method()
    Confusion_matrix()
    # Plot_ROC()
    # Plot_Fitness()

    # Image_Results1()
