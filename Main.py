
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from DataPreprocessing import DataPreprocessing
from MyModel import MyModel
from GetScoreCareer import GetScoreCareer


def Main(filePath):
    # Funkce slouzi pro overeni klasifikacnich schopnosti navrzeneho modelu.

    # Vstup:     filePath:           Nazev slozky (textovy retezec) obsahujici data

    # Vystup:    se:                 Vysledna senzitivita modelu
    #            sp:                 Vysledna specificita modelu
    #            acc:                Vysledna presnost modelu
    #            fScore:             Vysledne F1 skore modelu
    #            ppv:                Pozitivni prediktivni hodnota
    #            confusionMatrix:    Matice zamen

    # Funkce:
    #            DataPreprocessing:  Funkce pro predzpracovani dat

    #            MyModel:            Funkce pro implementaci modelu. Nauceny model se bude nacitat z externiho souboru

    #            GetScoreCareer:     Funkce pro vyhodnoceni uspesnosti
    #            modelu.

    if os.path.isdir(filePath) == False:
        print("Wrong directory")
    # %% 1 - Nacteni dat
    inputData = pd.read_csv(f"{filePath}\\TrainNBAData.csv")
    numRecords = inputData.shape[0]
    confMatrix = np.zeros((2, 2))



        # %% 2 - Predzpracovani dat
    preprocessedData = DataPreprocessing(inputData)  # Do zpracovani vstupuji vsechny informace o hraci (krome poradoveho cisla,jmena,score)

        # %% 3 - Vybaveni natrenovaneho modelu
    outputClass = MyModel(preprocessedData.drop(columns=['target_5yrs', 'Var1', 'name', 'ft', 'ast', 'x3p_made', 'x3pa', 'x3p']))
        # plt.imshow()
    for idx in range(numRecords):
        targetClass = inputData.target_5yrs[idx]
        if targetClass == 'no':
            targetClass = 0
        else:
            targetClass = 1


        if outputClass[idx] == 0 or outputClass[idx] == 1:
            confMatrix[outputClass[idx], targetClass] += 1
        else:
            print('Invalid class number. Operation aborted.')
    se, sp, acc, ppv, fScore = GetScoreCareer(confMatrix)
    print(f"Sensitivity (Se): {se:.2f}")
    print(f"Specificity (Sp): {sp:.2f}")
    print(f"Accuracy (Acc): {acc:.2f}")
    print(f"Positive Predictive Value (PPV): {ppv:.2f}")
    print(f"F1 Score (FScore): {fScore:.2f}")

    return se, sp, acc, ppv, fScore, confMatrix


if __name__ == "__main__":
    filePath = "C:\\Users\\Daniil\\PycharmProjects\\pythonProject\\UIMprojekt"
    se, sp, acc, ppv, fScore, confMatrix = Main(filePath)