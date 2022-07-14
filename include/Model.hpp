#pragma once

#include <vector>
#include <iostream>

#include "Layer.hpp"
#include "Activator.hpp"
#include "Loss.hpp"
#include "Optimizer.hpp"

class Model
{
private:
    std::vector<Layer *> layers;
    std::vector<Activator *> activators;
    Loss *loss;
    Optimizer *optimizer;

public:
    ~Model();
    void appendLayer(Layer *layer, Activator *activator);
    void setLoss(Loss *loss);
    void setOptimizer(Optimizer *optimizer);

    Eigen::MatrixXd forward(const Eigen::MatrixXd &input);
    void backward(const Eigen::MatrixXd &output, const Eigen::VectorXi &y);
    void train(const Eigen::MatrixXd &x, const Eigen::VectorXi &y, const int &iterations, const int &printGap);
    void saveModel();
};
