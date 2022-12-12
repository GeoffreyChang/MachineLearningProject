import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import Ridge
from helper_functions import *

folds = KFold(n_splits=10)
plt.style.use('ggplot')


if __name__ == "__main__":
    df = read_all_files(1)
    # df2 = read_file_no(2)
    df = df.sample(frac=1, random_state=12)
    features, target = get_features_and_target(df)
    # print((cross_val_score(Ridge(alpha=1.0), features, target, cv=folds)))
    r_scores = []
    for train_index, test_index in folds.split(df):
        x_train, x_test, y_train, y_test = features.iloc[[i for i in train_index]], \
                                           features.iloc[[i for i in test_index]], \
                                           target.iloc[[i for i in train_index]], \
                                           target.iloc[[i for i in test_index]]
        model = Ridge(alpha=1.0)
        model.fit(x_train, y_train)

        # Testing Against File 2
        # df2 = read_file_no(2)
        # features2, target2 = get_features_and_target(df2)
        # y_hat = model.predict(features2)
        # y_test = target2

        # Testing
        y_hat = model.predict(x_test)
        score = r2_score(y_test, y_hat)
        print("R square: %.3f" % score)
        r_scores.append(score)
        z_plot(y_hat, y_test)

        # Residuals Plot
        # plt.figure(figsize=(10, 6))
        # visualizer = ResidualsPlot(model)
        # visualizer.fit(x_train, y_train)
        # visualizer.score(x_test, y_test)
        # visualizer.show()
    print(np.mean(r_scores))
