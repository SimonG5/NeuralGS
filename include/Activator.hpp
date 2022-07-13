#pragma once

#include <Eigen/Dense>
#define MAX(a, b) a > b ? a : b

class Activator
{
protected:
    Eigen::MatrixXd inputs;
    Eigen::MatrixXd dInputs;
    Eigen::MatrixXd output;

public:
    virtual void forward(const Eigen::MatrixXd &inputs) = 0;
    virtual void backward(const Eigen::MatrixXd &dValues) = 0;

    Eigen::MatrixXd getOutput()
    {
        return this->output;
    }

    Eigen::MatrixXd getInputs()
    {
        return this->inputs;
    }

    Eigen::MatrixXd getDInputs()
    {
        return this->dInputs;
    }
};
