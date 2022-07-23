#pragma once

#include <vector>
#include <iostream>
#include <nlohmann/json.hpp>
#include <string>

#include "Layer.hpp"
#include "Activator.hpp"
#include "Loss.hpp"
#include "Optimizer.hpp"

using json = nlohmann::json;

class Model
{
private:
    std::vector<Layer *> layers;
    std::vector<Activator *> activators;
    Loss *loss;
    Optimizer *optimizer;
    double accuarcyTotal;
    double lossTotal;
    double testBatches;

public:
    Model();
    ~Model();
    void appendLayer(Layer *layer, Activator *activator);
    void setLoss(Loss *loss);
    void setOptimizer(Optimizer *optimizer);

    Eigen::MatrixXd forward(const Eigen::MatrixXd &input);
    void backward(const Eigen::MatrixXd &output, const Eigen::VectorXi &y);
    void train(const Eigen::MatrixXd &x, const Eigen::VectorXi &y, const int &iterations, const int &printGap);
    void test(const Eigen::MatrixXd &x, const Eigen::VectorXi &y);
    void evaluate(const Eigen::MatrixXd &x);
    void loadModel(json model);
    void saveModel();
};
