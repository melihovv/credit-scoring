using System;
using System.Collections.Generic;
using System.IO;
using AForge.Neuro;
using AForge.Neuro.Learning;

namespace credit_scoring
{
    class Program
    {
        const int INPUT_NUMBER = 5;

        static void Main(string[] args)
        {
            Tuple<double[][], double[][]> data = ReadData("data.csv");
            double[][] input = data.Item1;
            double[][] output = data.Item2;

            ActivationNetwork network = new ActivationNetwork(new SigmoidFunction(), INPUT_NUMBER, INPUT_NUMBER, 1);

            BackPropagationLearning teacher = new BackPropagationLearning(network);

            for (int i = 0; i < 10000; ++i)
            {
                double error = teacher.RunEpoch(input, output);

                if (error < 0.18)
                {
                    break;
                }
            }

            double[] result = network.Compute(new double[] {12, 2122, 1, 3, 3});
            Console.WriteLine(result[0]);
        }

        private static Tuple<double[][], double[][]> ReadData(string fileName)
        {
            var reader = new StreamReader(File.OpenRead(fileName));
            double[][] trainData = new double[1000][];
            double[][] resultColumn = new double[1000][];

            int i = 0;
            while (!reader.EndOfStream)
            {
                var line = reader.ReadLine();
                var columns = line.Split(',');

                int j = 0;
                trainData[i] = new double[INPUT_NUMBER];
                foreach (string column in columns)
                {
                    if (j == INPUT_NUMBER)
                    {
                        break;
                    }

                    trainData[i][j] = Double.Parse(column);
                    ++j;
                }

                resultColumn[i] = new double[1];
                resultColumn[i][0] = Double.Parse(columns[columns.Length - 1]);
                ++i;
            }

            return Tuple.Create(trainData, resultColumn);
        }
    }
}