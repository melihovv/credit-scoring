import numpy as np
from pybrain.datasets import ClassificationDataSet
from pybrain.utilities import percentError
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer

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
net = buildNetwork(trndata.indim, hidden_neurons_count, trndata.outdim)

trainer = BackpropTrainer(net, dataset=trndata, momentum=0.1, verbose=True,
                          weightdecay=0.01)

for i in range(20):
    trainer.trainEpochs(1)
    trnresult = percentError(trainer.testOnClassData(),
        trndata['class'])
    tstresult = percentError(trainer.testOnClassData(
        dataset=tstdata), tstdata['class'])

    print("epoch: %4d" % trainer.totalepochs,
          "  train error: %5.2f%%" % trnresult,
          "  test error: %5.2f%%" % tstresult)
