#include "../../include/Activator.hpp"

class Softmax : public Activator
{
public:
    void forward(const Eigen::MatrixXd &inputs)
    {
        this->inputs = inputs;
        Eigen::MatrixXd output(inputs.rows(), inputs.cols());
        for (int r = 0; r < inputs.rows(); r++)
        {
            double maxVal = inputs.row(r).maxCoeff();
            for (int c = 0; c < inputs.cols(); c++)
            {
                output(r, c) = std::exp(inputs(r, c) - maxVal);
            }
        }
        for (int r = 0; r < inputs.rows(); r++)
        {
            double normalizeBase = output.row(r).sum();
            for (int c = 0; c < inputs.cols(); c++)
            {
                output(r, c) /= normalizeBase;
            }
        }
        this->output = output;
    }

    void backward(const Eigen::MatrixXd &dValues)
    {
        this->dInputs = Eigen::MatrixXd(dValues.rows(), dValues.cols());

        for (int r = 0; r < dValues.rows(); r++)
        {
            Eigen::MatrixXd digFlat = this->output.row(r).asDiagonal();

            Eigen::MatrixXd jacobianMatrix = digFlat - (this->output.row(r).transpose() * this->output.row(r));

            this->dInputs.row(r) = jacobianMatrix * dValues.row(r).transpose();
        }
    }
};