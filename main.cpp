#include <vector>
#include <random>
#include <numeric>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <fstream>
#include <string>

#include <nlohmann/json.hpp>

#include "src/Activators/ReLu.cpp"
#include "src/Optimizers/SGD.cpp"
#include "src/Optimizers/AdaGrad.cpp"
#include "src/Activators/Sigmoid.cpp"
#include "src/Activators/Softmax.cpp"
#include "src/Losses/CategoricalCrossentropy.cpp"
#include "Model.hpp"

using json = nlohmann::json;

double random(const double &min, const double &max)
{
    std::random_device rd;
    std::default_random_engine generator(rd()); // rd() provides a random seed
    std::uniform_real_distribution<double> distribution(min, max);
    return distribution(generator);
}

std::tuple<Eigen::MatrixXd, Eigen::RowVectorXi> spiral_data(const size_t &points, const size_t &classes)
{
    Eigen::MatrixXd X(points * classes, 2);
    Eigen::RowVectorXi Y(points * classes);
    double r, t;
    for (size_t i = 0; i < classes; i++)
    {
        for (size_t j = 0; j < points; j++)
        {
            r = double(j) / double(points);
            t = i * 4 + (4 * r);
            X.row(i * points + j) = Eigen::RowVector2d(r * cos(t * 2.5), r * sin(t * 2.5)) + Eigen::RowVector2d(random(-0.15, 0.15), random(-0.15, 0.15));
            Y(i * points + j) = i;
        }
    }
    return std::make_tuple(X, Y);
}

int main()
{
    srand(time(NULL));

    std::tuple<Eigen::MatrixXd, Eigen::RowVectorXi> data = spiral_data(100, 3);

    Loss *loss = new CategoricalCrossentropy();

    Optimizer *sgd = new SGD(0.05, 1e-3, 0.9);
    Optimizer *adaGrad = new AdaGrad(1, 0, 1e-7);

    Model *model = new Model();
    model->setLoss(loss);
    model->setOptimizer(sgd);

    Layer *firstLayer = new Layer(2, 64, 0, 5e-4, 0, 5e-4);
    Activator *relu = new ReLu();
    Layer *secondLayer = new Layer(64, 3, 0, 0, 0, 0);
    Activator *smax = new Softmax();

    model->appendLayer(firstLayer, relu);
    model->appendLayer(secondLayer, smax);

    model->train(std::get<0>(data), std::get<1>(data), 10000, 100);

    delete (model);

    return 0;
}