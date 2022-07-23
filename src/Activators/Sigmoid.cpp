#include "../../include/Activator.hpp"

class Sigmoid : public Activator
{
public:
    void forward(const Eigen::MatrixXd &inputs)
    {
        this->inputs = inputs;
        this->output = Eigen::MatrixXd::Zero(inputs.rows(), inputs.cols());
#pragma omp parallel for
        for (int r = 0; r < inputs.rows(); r++)
        {
            for (int c = 0; c < inputs.cols(); c++)
            {
                this->output(r, c) = 1 / (1 + std::exp(-inputs(r, c)));
            }
        }
    }

    void backward(const Eigen::MatrixXd &dValues)
    {
        Eigen::MatrixXd subtractedOutput(this->output.rows(), this->output.cols());

#pragma omp parallel for
        for (int r = 0; r < subtractedOutput.rows(); r++)
            for (int c = 0; c < subtractedOutput.cols(); c++)
                subtractedOutput(r, c) = 1 - this->output(r, c);

        this->dInputs = dValues.array() * subtractedOutput.array() * this->output.array();
    }
};
