import numpy as np
import matplotlib.pyplot as plt

def plot_dataset(X_train, X_test, y_train, y_test):

    plt.figure(figsize=(10,7.5))
    plt.title("Dataset")
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=50, label='Train')
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, s=50, marker='x',label='Test')
    plt.legend()
    plt.show()
    
def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy

def plot_contours(ax, clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

def plot_classification(input, target, model, partition):

    fig, ax = plt.subplots(figsize=(10,7.5))

    X0, X1 = input[:, 0], input[:, 1]
    xx, yy = make_meshgrid(X0, X1)

    plot_contours(ax, model, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)

    ax.scatter(X0, X1, c=target, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    ax.set_ylabel('PC2')
    ax.set_xlabel('PC1')
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(f'Decison surface for the {partition} partition')
    plt.show()