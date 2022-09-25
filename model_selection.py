from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split

gamma_list = [0.01, 0.005]
c_list = [0.1, 0.2, 2, 5, 7, 10]
hyper_param_comb = [{"gamma": g, "C": c} for g in gamma_list for c in c_list]
assert len(hyper_param_comb) == len(gamma_list) * len(c_list)

digits = datasets.load_digits()
x = digits.data
y = digits.target

metric=metrics.accuracy_score

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=0, shuffle=True)

create_model_and_h_params(hyper_param, model, x_train, x_test, y_train, y_test)

def create_model_and_h_params(hyper_param, model, x_train, y_train, x_dev, y_dev):
    metrics = [];
    for single_h_param in h_param_comb:
        svm_model = svm.SVC()
        svm_model.set_params(**single_h_param)
        svm_model.fit(x_train, y_train)
        predicted_train = svm_model.predict(x_test)
        train_metric = metric(y_pred=predicted_train, y_true=y_test)
        metrics.append(train_metric)

return metrics
