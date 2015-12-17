import numpy as np
from pybrain.datasets import ClassificationDataSet
from pybrain.utilities import percentError
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure import SoftmaxLayer

data_file = 'data.csv'

data = np.loadtxt(data_file, delimiter=',')
input_data = data[:, 0:-1]
target_data = data[:, -1]
target_data = target_data.reshape(-1, 1)

ds = ClassificationDataSet(input_data.shape[1], nb_classes=2)
ds.setField('input', input_data)
ds.setField('target', target_data)

tstdata, trndata = ds.splitWithProportion(0.25)

trndata._convertToOneOfMany()
tstdata._convertToOneOfMany()

print("Number of training patterns: ", len(trndata))
print("Input and output dimensions: ", trndata.indim, trndata.outdim)
print("First sample (input, target, class):")
print(trndata['input'][0], trndata['target'][0], trndata['class'][0])

hidden_neurons_count = 5
net = buildNetwork(trndata.indim, hidden_neurons_count, trndata.outdim,
                   outclass=SoftmaxLayer)

trainer = BackpropTrainer(net, dataset=trndata, verbose=True,
                          weightdecay=0.002)
trainer.trainOnDataset(trndata, 500)
trainer.testOnData(verbose=True)

out = net.activate(
    [1, 1, 18, 4, 2, 1049, 1, 2, 4, 2, 1, 4, 2, 21, 3, 1, 1, 3, 1, 1]
)
print(out)

out = net.activate(
    [0, 2, 48, 0, 10, 18424, 1, 3, 1, 2, 1, 2, 2, 32, 1, 2, 1, 4, 1, 2]
)
print(out)
