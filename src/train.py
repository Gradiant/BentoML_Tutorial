#%%
import bentoml

from sklearn.svm import SVC
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

from plot import plot_dataset, plot_classification

from loguru import logger

@logger.catch(reraise=True)
def train():

    logger.info("Loading dataset")
    iris = datasets.load_iris()
    X, y = iris.data, iris.target

    logger.info("Dimensionality reduction")
    pca = PCA(n_components=2)
    Xreduced = pca.fit_transform(X)
    X_train, X_test, y_train, y_test=train_test_split(Xreduced,y,test_size=0.30)

    logger.info("Showing dataset")
    plot_dataset(X_train, X_test, y_train, y_test)

    logger.info("Trainning")
    model = SVC(kernel='linear')
    model = model.fit(X_train, y_train)

    logger.info("Showing train boundaries")
    plot_classification(X_train, y_train, model, "train")
    logger.info("Showing train boundaries")
    plot_classification(X_test, y_test, model, "train")

    logger.info("Saving model to BentoML")
    bentoml.sklearn.save("tutorial_svm", model)

if __name__ == "__main__":

    train()
# %%