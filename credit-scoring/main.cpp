#include <alglib/dataanalysis.h>
#include <QStringList>
#include <QString>
#include <QFile>
#include <cassert>
#include <tuple>

QList<QStringList> readCsv(const QString& fileName);

std::tuple<alglib::real_2d_array, alglib::real_2d_array>
prepareData(const QList<QStringList>& data, float ratio);

int main(int argc, char *argv[])
{
    QList<QStringList> rawData = readCsv("data.csv");
    alglib::real_2d_array trainData;
    alglib::real_2d_array testData;
    std::tie(trainData, testData) = prepareData(rawData, 0.75);

    assert(trainData.rows() > 0);

    int inputNumber = trainData.cols() -1;
    int numberOfClasses = 2;
    int numberOfHiddenLayers = 5;

    alglib::mlptrainer trainer;
    alglib::mlpcreatetrainercls(inputNumber, numberOfClasses, trainer);
    alglib::multilayerperceptron nn;
    alglib::mlpcreatec1(inputNumber, numberOfHiddenLayers, numberOfClasses, nn);
    alglib::mlpsetdataset(trainer, trainData, inputNumber);

    double wstep = 0.001;
    double decay = 0.001;
    alglib::ae_int_t restartsCount = 5;
    alglib::ae_int_t maxits = 100;

    alglib::mlpsetdecay(trainer, decay);
    alglib::mlpsetcond(trainer, wstep, maxits);

    alglib::mlpreport report;
    
    // cross-validation.
    //alglib::mlpkfoldcv(trainer, nn, 5, 10, report);

    alglib::mlptrainnetwork(trainer, nn, restartsCount, report);

    // calc error on test set.
    alglib::modelerrors modelErrors;
    alglib::mlpallerrorssubset(nn, testData, testData.rows(), 
        alglib::integer_1d_array(), -1, modelErrors);

    alglib::real_1d_array in = "[1,1,18,4,2,1049,1,2,4,2,1,4,2,21,3,1,1,3,1,1]";
    alglib::real_1d_array out;
    mlpprocess(nn, in, out);
    printf("%s\n", out.tostring(1).c_str());

    in = "[1,1,12,4,0,2122,1,3,3,3,1,2,1,39,3,1,2,2,2,1]";
    mlpprocess(nn, in, out);
    printf("%s\n", out.tostring(1).c_str());

    return 0;
}

QList<QStringList> readCsv(const QString& fileName)
{
    QFile file(fileName);
    if (!file.open(QIODevice::ReadOnly))
    {
        throw std::invalid_argument("Cannot read file " +
            fileName.toStdString());
    }

    QList<QStringList> wordList;
    while (!file.atEnd())
    {
        QString line = file.readLine();
        wordList.append(line.split(','));
    }

    return wordList;
}

std::tuple<alglib::real_2d_array, alglib::real_2d_array>
prepareData(const QList<QStringList>& data, float ratio)
{
    int rowsAmount = data.size() * ratio;
    if (rowsAmount < 1)
    {
        return std::make_tuple(
            alglib::real_2d_array(), alglib::real_2d_array());
    }
    const int colsAmount = data[0].size();

    double* rawTrainData = new double[rowsAmount * colsAmount];
    int index = 0;
    const auto lastRow = data.cbegin() + rowsAmount;
    auto iter = data.cbegin();
    for (; iter != lastRow; ++iter)
    {
        for (const auto col : *iter)
        {
            rawTrainData[index] = col.toDouble();
            ++index;
        }
    }

    alglib::real_2d_array trainData;
    trainData.setcontent(rowsAmount, colsAmount, rawTrainData);

    rowsAmount = data.size() - rowsAmount;

    double* rawTestData = new double[rowsAmount * colsAmount];
    index = 0;
    const auto end = data.cend();
    for (; iter != end; ++iter)
    {
        for (const auto col : *iter)
        {
            rawTestData[index] = col.toDouble();
            ++index;
        }
    }
    
    alglib::real_2d_array testData;
    testData.setcontent(rowsAmount, colsAmount, rawTestData);

    return std::make_tuple(trainData, testData);
}
