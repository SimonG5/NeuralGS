#include "../../include/Loss.hpp"

class BinaryCrossentropy : public Loss
{
public:
    double regularizationLoss(Layer *layer)
    {
    }

    double forward(const Eigen::MatrixXd &inputs, const Eigen::VectorXi &yTrue)
    {
        Eigen::RowVectorXd clippedInput(inputs.rows());
        for (int r = 0; r < clippedInput.rows(); r++)
            clippedInput(r) = clamp(inputs(r));

        Eigen::RowVectorXd loss(inputs.rows());
        for (int r = 0; r < inputs.rows(); r++)
        {
            loss(r) = -(yTrue(r) * log(clippedInput(yTrue(r)))) + (1 - yTrue(r)) * log(1 - clippedInput(yTrue(r)));
        }

        return loss.mean();
    }

    void backward(const Eigen::MatrixXd &dValues, const Eigen::VectorXi &yTrue)
    {
    }
};
