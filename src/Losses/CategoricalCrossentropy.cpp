#include "../../include/Loss.hpp"
#include <iostream>

class CategoricalCrossentropy : public Loss
{
public:
    double regularizationLoss(Layer *layer)
    {
        double regularization = 0;

        if (layer->weightRegularizerOne > 0)
            regularization += layer->weightRegularizerOne * layer->weights.array().abs().sum();

        if (layer->weightRegularizerTwo > 0)
        {
            Eigen::MatrixXd squaredMatrix = layer->weights;
            for (int r = 0; r < squaredMatrix.rows(); r++)
            {
                for (int c = 0; c < squaredMatrix.cols(); c++)
                    squaredMatrix(r, c) *= squaredMatrix(r, c);
            }
            regularization += layer->weightRegularizerTwo * squaredMatrix.sum();
        }

        if (layer->biasRegularizerOne > 0)
            regularization += layer->biasRegularizerOne * layer->biases.array().abs().sum();

        if (layer->biasRegularizerTwo > 0)
        {
            Eigen::RowVectorXd squaredRowVector = layer->biases;
            for (int r = 0; r < squaredRowVector.size(); r++)
                squaredRowVector(r) *= squaredRowVector(r);

            regularization += layer->biasRegularizerTwo * squaredRowVector.sum();
        }

        return regularization;
    }

    double forward(const Eigen::MatrixXd &inputs, const Eigen::VectorXi &yTrue)
    {
        this->output = inputs;
        Eigen::RowVectorXd loss(inputs.rows());
        for (int r = 0; r < inputs.rows(); r++)
        {
            loss(r) = -log(clamp(inputs(r, yTrue(r))));
        }
        return loss.mean();
    }

    void backward(const Eigen::MatrixXd &dValues, const Eigen::VectorXi &yTrue)
    {
        int samples = dValues.row(0).size();
        this->dInputs = Eigen::MatrixXd::Zero(yTrue.size(), samples);

        for (int r = 0; r < this->dInputs.rows(); r++)
        {
            this->dInputs(r, yTrue(r)) = (-1 / dValues(r, yTrue(r))) / samples;
        }
    }
};