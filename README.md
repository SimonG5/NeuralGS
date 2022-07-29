# NeuralGS

NeuralGS is a neural network built in c++ with eigen3 as the underlying linear algebra libary. The network has no gpu accelerated methods as of yet. The network makes use of OpenMP to multithread heavy functions.

## Usage

Creating simple neural network model.

```bash
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
```

Training models, where dataset and labels are loaded in to the variabels x and y.
```bash
Eigen::MatrixXd x;
Eigen::VectorXi y;

int epochs = 10;
int printGap = 1;

model.train(x, y, epochs, printGap);
```

Saving and loading models
```bash
model.train(x, y, epochs, printGap);
model.saveModel();
//Loading
json savedFile
Model model;
model.loadModel(savedFile);
```
Testing and predicting with model
```bash
Eigen::MatrixXd testX;
Eigen::VectorXi testY;

model.test(x, y);
//Predicting
sampleInput = Eigen::MatrixXd(3,3);
sampleInput << 0.2,0.3,0.4,
               0.5,0.7,0.8,
               0.4,0.5,0.6;
model.evaluate(sampleInput)
```

## Dependencies
[Eigen3](https://gitlab.com/libeigen/eigen) - Linear algebra
[Nhloman::json](https://github.com/nlohmann/json) - Saving and loading models
[OpenMP](https://www.openmp.org/) - Multithreading
