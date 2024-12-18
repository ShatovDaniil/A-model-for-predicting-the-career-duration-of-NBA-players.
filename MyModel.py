
from joblib import load

def MyModel(data):
#     Funkce slouzi k implementaci nauceneho modelu. Vas model bude ulozen v samostatne promenne a se spustenim se aplikuje
#     na vstupni data.

#Vstup:             data:           vstupni surova data reprezentujici 1
#                                   objekt

#Vystup:            output:         zarazeni objektu do tridy


    model_filename = "C:/Users/Daniil/PycharmProjects/pythonProject/UIMprojekt/trained_model.joblib"
    model = load(model_filename)
    output = model.predict(data)


    return output