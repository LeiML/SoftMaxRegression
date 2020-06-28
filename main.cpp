#include <Matrix.hpp>
#include "SoftMax.hpp"

int main(int argc, char **argv) {
    vector<string>data;
    read("../iris.data", data);
    Matrix sample = Matrix(data.size(), 4);
    Matrix label = Matrix(data.size(), 1);
    makeMatrix(data, sample, label);
    auto regression = SoftMaxRegression(0.005, 2000);
    regression.train(sample, label);
    return 0;
}
