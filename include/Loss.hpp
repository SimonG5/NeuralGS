#pragma once

#include <cmath>
#include <Eigen/Dense>
#include "Layer.hpp"

class Loss
{
protected:
    Eigen::MatrixXd output;
    Eigen::MatrixXd dInputs;

    double clamp(const double &input)
    {
        if (input <= 0)
            return 1e-7;
        if (input >= 1)
            return 1 - 1e-7;
        return input;
    }

public:
    virtual double regularizationLoss(Layer *layer) = 0;
    virtual double forward(const Eigen::MatrixXd &inputs, const Eigen::VectorXi &yTrue) = 0;
    virtual void backward(const Eigen::MatrixXd &dValues, const Eigen::VectorXi &yTrue) = 0;

    Eigen::MatrixXd getOutput()
    {
        return this->output;
    }

    Eigen::MatrixXd getDInputs()
    {
        return this->dInputs;
    }
};
