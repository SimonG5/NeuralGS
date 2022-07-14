#include "../include/Layer.hpp"
#include <iostream>

Layer::Layer(int inputs, int neurons, double weightRegularizerOne, double weightRegularizerTwo, double biasRegularizerOne, double biasRegularizerTwo)
{
    this->weights = Eigen::MatrixXd::Random(inputs, neurons);
    this->biases = Eigen::RowVectorXd::Zero(neurons);

    this->optiWeightHelper = Eigen::MatrixXd::Zero(inputs, neurons);
    this->weightCache = Eigen::MatrixXd::Zero(inputs, neurons);

    this->optiBiasHelper = Eigen::RowVectorXd::Zero(neurons);
    this->biasCache = Eigen::RowVectorXd::Zero(neurons);

    this->weightRegularizerOne = weightRegularizerOne;
    this->weightRegularizerTwo = weightRegularizerTwo;
    this->biasRegularizerOne = biasRegularizerOne;
    this->biasRegularizerTwo = biasRegularizerTwo;
}

Layer::Layer(const Eigen::MatrixXd &weights, const Eigen::RowVectorXd &biases)
{
    this->weights = weights;
    this->biases = biases;
}

void Layer::forward(const Eigen::MatrixXd &inputs)
{
    this->inputs = inputs;
    this->output = (inputs * this->weights).rowwise() + this->biases;
}

void Layer::backward(const Eigen::MatrixXd &dValues)
{
    this->dWeights = this->inputs.transpose() * dValues;
    this->dBiases = dValues.colwise().sum();

    if (this->weightRegularizerOne > 0)
    {
        Eigen::MatrixXd dLOne = Eigen::MatrixXd::Ones(this->weights.rows(), this->weights.cols());
        for (int r = 0; r < dLOne.rows(); r++)
        {
            for (int c = 0; c < dLOne.cols(); c++)
            {
                if (this->weights(r, c) < 0)
                    dLOne(r, c) = -1;
            }
        }
        this->dWeights += 2 * this->weightRegularizerOne * dLOne;
    }

    if (this->weightRegularizerTwo > 0)
    {
        this->dWeights += 2 * this->weightRegularizerTwo * this->weights;
    }

    if (this->biasRegularizerOne > 0)
    {
        Eigen::VectorXd dLOne = Eigen::VectorXd::Ones(this->biases.size());
        for (int i = 0; i < dLOne.size(); i++)
        {
            if (this->weights(i) < 0)
                dLOne(i) = -1;
        }
        this->dBiases += 2 * this->biasRegularizerOne * dLOne;
    }

    if (this->biasRegularizerTwo > 0)
    {
        this->dBiases += 2 * this->biasRegularizerTwo * this->biases;
    }

    this->dInputs = dValues * this->weights.transpose();
}

void Layer::setWeights(const Eigen::MatrixXd &newWeights)
{
    this->weights = newWeights;
}

void Layer::setBiases(const Eigen::RowVectorXd &newBiases)
{
    this->biases = newBiases;
}

Eigen::MatrixXd Layer::getWeights()
{
    return this->weights;
}

Eigen::RowVectorXd Layer::getBiases()
{
    return this->biases;
}

Eigen::MatrixXd Layer::getOutput()
{
    return this->output;
}