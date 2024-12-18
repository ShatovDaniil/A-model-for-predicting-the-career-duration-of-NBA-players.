
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from joblib import dump
import os
from DataPreprocessing import DataPreprocessing


#METODA Č.1 - HLAVNÍ METODA Gradient Boosting
def train_model(preprocessedData: object):
    """
        Funkce pro trénování modelu na základě předzpracovaných dat.

        Vstup:
        - preprocessedData (DataFrame): Data, která již prošla předzpracováním.
          Obsahuje:
            - Sloupec `target_5yrs`: Cílová proměnná s hodnotami "yes"/"no".
            - Ostatní sloupce: Znaky popisující sportovce.

        Proces:
        1. Převod hodnot v `target_5yrs` na binární formát:
           - "yes" → 1
           - "no" → 0
        2. Výběr relevantních znaků pro trénování modelu:
        3. Rozdělení dat na trénovací a testovací sady (80/20).
        4. Vytvoření a trénování modelu Gradient Boosting Classifier:
           - Parametry modelu:
             - `n_estimators=200`: Počet stromů.
             - `learning_rate=0.02`: Rychlost učení.
             - `max_depth=3`: Maximální hloubka stromů.
        5. Predikce výsledků na testovacích datech.
        6. Uložení naučeného modelu do souboru `trained_model.joblib`.

        Výstup:
        - model (GradientBoostingClassifier): Naučený model připravený k použití.

        """
    preprocessedData['target_5yrs'] = preprocessedData['target_5yrs'].map({'yes': 1, 'no': 0})
    X = preprocessedData.drop(columns=['target_5yrs', 'Var1', 'name', 'ft', 'ast', 'x3p_made', 'x3pa', 'x3p'])

    y = preprocessedData['target_5yrs']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    model = GradientBoostingClassifier(n_estimators=200,
    learning_rate=0.02,
    max_depth=3,
    random_state=42)

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)



    output_dir = r"C:\Users\Daniil\PycharmProjects\pythonProject\UIMprojekt"
    os.makedirs(output_dir, exist_ok=True)
    model_filename = os.path.join(output_dir, 'trained_model.joblib')
    dump(model, model_filename)

    print(f":{model_filename}")
    return model

filePath = "C:\\Users\\Daniil\\PycharmProjects\\pythonProject\\UIMprojekt"
inputData = pd.read_csv(f"{filePath}\\TrainNBAData.csv")
df = DataPreprocessing(inputData)
train_model(df)


