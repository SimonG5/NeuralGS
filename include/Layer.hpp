#pragma once

#include <Eigen/Dense>

class Layer
{
private:
    Eigen::MatrixXd inputs;

public:
    Eigen::MatrixXd optiWeightHelper;
    Eigen::MatrixXd weightCache;
    Eigen::RowVectorXd optiBiasHelper;
    Eigen::RowVectorXd biasCache;

    Eigen::MatrixXd output;

    Eigen::MatrixXd weights;
    Eigen::RowVectorXd biases;

    Eigen::MatrixXd dInputs;
    Eigen::MatrixXd dWeights;
    Eigen::RowVectorXd dBiases;

    double weightRegularizerOne;
    double weightRegularizerTwo;
    double biasRegularizerOne;
    double biasRegularizerTwo;

    Layer(int inputs, int neurons, double weightRegularizerOne, double weightRegularizerTwo, double biasRegularizerOne, double biasRegularizerTwo);
    Layer(const Eigen::MatrixXd &weights, const Eigen::RowVectorXd &biases);
    void forward(const Eigen::MatrixXd &inputs);
    void backward(const Eigen::MatrixXd &dValues);
    void setWeights(const Eigen::MatrixXd &newWeights);
    void setBiases(const Eigen::RowVectorXd &newBiases);
    Eigen::MatrixXd getWeights();
    Eigen::RowVectorXd getBiases();
    Eigen::MatrixXd getOutput();
};
