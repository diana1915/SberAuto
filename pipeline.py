import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import make_column_selector
from sklearn.model_selection import train_test_split, StratifiedKFold
from catboost import *

import dill
from sklearn.preprocessing import StandardScaler



def filter_data(df):
    columns_to_drop = ['device_model']

    # Возвращаем копию датафрейма, inplace тут делать нельзя!
    return df.drop(columns_to_drop, axis=1)


def num_features(df2):
    import pandas as pd
    df = df2.copy()

    # Переведем разрешение экрана в числовой признак
    device_screen = df['device_screen_resolution'].apply(
        lambda x: int(x.split('x')[0]) * int(x.split('x')[1] if (x != '(not set)' or x != np.nan) else 0))
    df['device_screen_resolution'] = device_screen

    return df


def normal_data(df2):
    from sklearn.preprocessing import StandardScaler

    df = df2.copy()
    data = df['device_screen_resolution']
    q25 = data.quantile(0.25)
    q75 = data.quantile(0.75)
    iqr = q75 - q25
    boundaries = (q25 - 1.5 * iqr, q75 + 1.5 * iqr)
    df.loc[df['device_screen_resolution'] < boundaries[0], 'device_screen_resolution'] = round(boundaries[0])
    df.loc[df['device_screen_resolution'] > boundaries[1], 'device_screen_resolution'] = round(boundaries[1])

    scaler = StandardScaler()
    df['device_screen_resolution'] = scaler.fit_transform(df[['device_screen_resolution']])

    return df


def encode(df2):
    import pandas as pd
    from sklearn.impute import SimpleImputer

    df = df2.copy()

    imput = SimpleImputer(strategy='most_frequent')
    for i in df.columns:
        if i != 'device_screen_resolution' and i != 'target':
            df[i] = imput.fit_transform(df[i].values.reshape(-1, 1))[:, 0]

    # Заполним пропуски в категориальных переменных модой
    imput = SimpleImputer(missing_values=None, strategy='most_frequent')
    for i in df.columns:
        if i != 'device_screen_resolution' and i != 'target':
            df[i] = imput.fit_transform(df[i].values.reshape(-1, 1))[:, 0]

    return df


def new_features_cat(df2):
    df = df2.copy()

    all_cities = ['moscow', 'krasnogorsk', 'podolsk', 'khimki', 'mytishchi', 'balashikha', 'zheleznodorozhny',
                  'lyubertsy', 'korolev', 'krasnoarmeysk', 'elektrostal', 'orekhovo-zuevo', 'noginsk', 'sergiev posad',
                  'dmitrov', 'pushkino', 'ivanteevka', 'domodedovo', 'lobnya', 'klin', 'ramenskoye', 'shchyolkovo',
                  'odintsovo', 'vidnoye', 'reutov', 'tuchkovo', 'chekhov', 'dolgoprudny', 'zelenograd', 'khimki',
                  'pavlovsky posad', 'solnechnogorsk', 'shatura', 'khimki', 'krasnogorsk', 'naro-fominsk',
                  'orekhovo-zuevo', 'taldom', 'zvenigorod', 'kolomna', 'lytkarino', 'ramenskoye', 'serpukhov', 'dubna',
                  'pushkino', 'krasnoznamensk', 'kubinka', 'elektrogorsk', 'istra', 'ruzayevka', 'klimovsk',
                  'voskresensk', 'volokolamsk', 'zhukovsky', 'kotelniki', 'losino-petrovsky', 'zvyozdny gorodok',
                  'shakhovskaya', 'saint petersburg', 'vyborg', 'gatchina', 'kronshtadt', 'sosnovy bor', 'priozersk',
                  'lomonosov', 'podporozhye', 'sertolovo', 'volkhov', 'tikhvin', 'kirishi', 'kolpino', 'vsevolozhsk',
                  'sestroretsk', 'kirovsk', 'vyborg', 'pushkin', 'petergof', 'kingisepp', 'luga', 'slantsy', 'tosno',
                  'shlisselburg', 'boksitogorsk', 'pikalyovo', 'novaya ladoga', 'tikhvin', 'ivangorod', 'svetogorsk',
                  'sebezh', 'shakhovskaya']

    # если город это столица, то 1
    df['capital'] = df['geo_city'].apply(lambda x: 1 if (x.lower() == 'moscow') else 0)

    # если в городе есть СберАвтоподписка, то 1
    df['subscription'] = df['geo_city'].apply(lambda x: 1 if (x.lower() in all_cities) else 0)

    return df


def main():
    print('Loan Prediction Pipeline')

    df = pd.read_csv('data/data.csv', low_memory=False)
    df = df.drop(columns=['session_id', 'client_id', 'visit_date', 'visit_time', 'visit_number'])
    df = df.reset_index(drop=True)

    categorical_transformer = Pipeline(steps=[
        ('imputer1', SimpleImputer(strategy='most_frequent')),
        ('imputer2', SimpleImputer(strategy='most_frequent'))
    ])

    preprocessor2 = ColumnTransformer(transformers=[
        ('categorical', categorical_transformer, make_column_selector(dtype_include=object))
    ])

    preprocessor = Pipeline(steps=[
        ('filter', FunctionTransformer(filter_data)),
        ('num_features', FunctionTransformer(num_features)),
        ('normal', FunctionTransformer(normal_data)),
        ('inputer', FunctionTransformer(encode)),
        ('new_features', FunctionTransformer(new_features_cat))
    ])

    with open('preprocessor.pkl', 'wb') as file:
        dill.dump({
            'preprocessor': preprocessor,
        }, file)

    df = preprocessor.fit_transform(df)

    cat_features = [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11]

    X = df.drop(['target'], axis=1)
    y = df['target']

    model = CatBoostClassifier(
        loss_function='Logloss',
        learning_rate=0.3,
        depth=7,
        eval_metric='AUC',
        iterations=200,
        early_stopping_rounds=30,
        verbose=100,
    )  # параметры после применения grid_search

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    aucs = []
    precision_scores = []
    recall_scores = []
    f1_scores = []

    for train_index, valid_index in skf.split(X, y):
        X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
        y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

        train_data = Pool(data=X_train, label=y_train, cat_features=cat_features)
        valid_data = Pool(data=X_valid, label=y_valid, cat_features=cat_features)

        model.fit(train_data, eval_set=valid_data, use_best_model=True)

        preds_proba = model.predict_proba(X_valid)[:, 1]
        preds_class = (preds_proba > 0.5).astype(int)
        auc_pred = roc_auc_score(y_valid, preds_proba)
        precision = precision_score(y_valid, preds_class)
        recall = recall_score(y_valid, preds_class)
        f1 = f1_score(y_valid, preds_class)

        aucs.append(auc_pred)
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)

    print(f"Mean AUC: {np.mean(aucs)}")
    print(f"Mean Precision: {np.mean(precision_scores)}")
    print(f"Mean Recall: {np.mean(recall_scores)}")
    print(f"Mean F1 Score: {np.mean(f1_scores)}")

    with open('visit.pkl', 'wb') as file:
        dill.dump({
            'model': model,
            'metadata': {
                'name': 'Target action prediction model on SberAutosubscription',
                'autor': 'Simonyan Diana',
                'version': 1,
                'roс-auc': np.mean(aucs)
            }
        }, file)


if __name__ == '__main__':
    main()
