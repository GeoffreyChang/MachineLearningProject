from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from helper_functions import *

folds = KFold(n_splits=5, shuffle=True, random_state=42)
plt.style.use('ggplot')


if __name__ == "__main__":
    separate = False

    if separate:
        df = read_all_files(1)
        df_norm = normalize_df(df)
        features1, target1 = get_features_and_target(df)
        features, target = get_features_and_target(df_norm)
    else:
        df = read_all_files()
        df.pop(9)
        df.pop(5)
        df = pd.concat(df)
        df = df.sample(frac=1, random_state=12)
        df = df.drop(["TIME", "S"], axis=1)
        df_norm = normalize_df(df)
        df_norm = df_norm.dropna()
        features, target = df_norm.iloc[:, :-1], df_norm["Z"]

    predicted_overall = []
    real_overall = []
    for train_index, test_index in folds.split(df_norm):
        x_train, x_test, y_train, y_test = features.iloc[train_index], \
                                           features.iloc[test_index], \
                                           target.iloc[train_index], \
                                           target.iloc[test_index]
        model = Ridge(alpha=1.0)
        model.fit(x_train, y_train)

        # Testing
        y_hat = model.predict(x_test)
        score = r2_score(y_test, y_hat)
        print("R square: %.3f" % score)
        predicted_overall.append(y_hat)
        real_overall.append(y_test)
    z_plot(predicted_overall, real_overall, split=False)
    # Residuals Plot
    # plt.figure(figsize=(10, 6))
    # visualizer = ResidualsPlot(model)
    # visualizer.fit(x_train, y_train)
    # visualizer.score(x_test, y_test)
    # visualizer.show()
