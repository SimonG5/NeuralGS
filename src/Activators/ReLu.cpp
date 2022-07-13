#include "../../include/Activator.hpp"

class ReLu : public Activator
{
public:
    void forward(const Eigen::MatrixXd &inputs)
    {
        this->inputs = inputs;
        Eigen::MatrixXd output(inputs.rows(), inputs.cols());
        for (int r = 0; r < inputs.rows(); r++)
        {
            for (int c = 0; c < inputs.cols(); c++)
            {
                output(r, c) = MAX(0.0, inputs(r, c));
            }
        }
        this->output = output;
    }

    void backward(const Eigen::MatrixXd &dValues)
    {
        this->dInputs = dValues;
        for (int r = 0; r < dInputs.rows(); r++)
        {
            for (int c = 0; c < dInputs.cols(); c++)
            {
                if (this->inputs(r, c) <= 0)
                    this->dInputs(r, c) = 0;
            }
        }
    }
};
