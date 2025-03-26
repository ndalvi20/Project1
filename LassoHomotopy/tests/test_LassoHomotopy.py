import csv

import numpy

from model.LassoHomotopy import LassoHomotopyModel

def test_predict():
    model = LassoHomotopyModel(reg_param=0.1)
    data = []
    with open("small_test.csv", "r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append(row)

    X = numpy.array([[v for k,v in datum.items() if k.startswith('x')] for datum in data])
    y = numpy.array([[v for k,v in datum.items() if k=='y'] for datum in data])
    results = model.fit(X,y)
    preds = results.predict(X)
   # Check that the prediction shape matches the target shape
    assert preds.shape == y.shape
    # check that the predictions are within a plausible range
    assert numpy.all(preds > -1e6) and numpy.all(preds < 1e6)
