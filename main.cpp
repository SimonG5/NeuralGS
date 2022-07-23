#include <string>
#include <vector>

#include "src/Activators/ReLu.cpp"
#include "src/Optimizers/Adam.cpp"
#include "src/Activators/Softmax.cpp"
#include "src/Losses/CategoricalCrossentropy.cpp"
#include "include/Model.hpp"

// Example usage
int main()
{
    srand(time(NULL));

    Loss *loss = new CategoricalCrossentropy();

    Optimizer *adam = new Adam(0.001, 1e-5, 1e-7, 0.9, 0.999);

    Model model;
    model.setLoss(loss);
    model.setOptimizer(adam);

    Layer *firstLayer = new Layer(30, 128, 0, 5e-4, 0, 5e-4);
    Activator *relu = new ReLu();
    Layer *secondLayer = new Layer(128, 128, 0, 5e-4, 0, 5e-4);
    Activator *reluTwo = new ReLu();
    Layer *thirdLayer = new Layer(128, 128, 0, 5e-4, 0, 5e-4);
    Activator *reluThree = new ReLu();
    Layer *fourtLayer = new Layer(128, 325, 0, 0, 0, 0);
    Activator *smax = new Softmax();

    model.appendLayer(firstLayer, relu);
    model.appendLayer(secondLayer, reluTwo);
    model.appendLayer(thirdLayer, reluThree);
    model.appendLayer(fourtLayer, smax);

    Eigen::MatrixXd x;
    Eigen::VectorXi y;

    int epochs = 10;
    int printGap = 1;

    model.train(x, y, epochs, printGap);

    return 0;
}