__author__ = 'boiko'
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt


def evaluate_arima_model(X, arima_order, steps):
    # prepare training dataset
    X = X.astype('float32')
    train_size = int(steps)
    train, test = X[0:train_size], X[train_size:]
    history = [x for x in train]
    # make predictions
    predictions = list()
    for t in range(len(test)):
        model = ARIMA(history, order=arima_order)
        # model_fit = model.fit(disp=0)
        model_fit = model.fit(trend='nc', disp=0)
        yhat = model_fit.forecast()[0]
        predictions.append(yhat)
        history.append(test[t])
    # calculate out of sample error
    rmse = sqrt(mean_squared_error(test, predictions))
    return rmse

# evaluate combinations of p, d and q values for an ARIMA model
def evaluate_models(dataset, steps, p_values, d_values, q_values):
    dataset = dataset.astype('float32')
    best_score, best_cfg = float("inf"), None
    p1,q1,d1 = 0,0,0
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p, d, q)
                try:
                    rmse = evaluate_arima_model(dataset, order,steps)
                    print(rmse)
                    if rmse < best_score:
                        p1 = p
                        q1 = q
                        d1 = d
                        best_score, best_cfg = rmse, order
                    print('ARIMA%s RMSE=%.3f' % (order, rmse))
                except:
                    continue
    print('Best ARIMA%s RMSE=%.3f' % (best_cfg, best_score))
    return p1, d1, q1


def StartProducingARIMAForecastValues(dataVals, p, d, q):
    model = ARIMA(dataVals, order=(p, d, q))
    model_fit = model.fit(disp=0)
    pred = model_fit.forecast()[0]
    return pred