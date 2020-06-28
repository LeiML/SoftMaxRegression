//
// Created by LeiLei on 2020/6/28.
//

#ifndef SoftMax_hpp
#define SoftMax_hpp
#include <Matrix.hpp>
#include <fstream>
#include <filesystem>
#include <string>
using namespace std::filesystem;
//读取文件
void read(const string& path, vector<string>&data);
//字符串分割
vector<string>split(const string&s, const string&c);
//字符串替换
string replace(const string &s, const string &oc, const string &nc);
//构建类别矩阵与样本矩阵
void makeMatrix(const vector<string>&data, Matrix & sample, Matrix& label);
//SoftMax的逻辑回归构建的类
class SoftMaxRegression{
public:
    //无参数构造函数
    SoftMaxRegression();
    //使用学习率作为参数构造
    explicit SoftMaxRegression(float rate);
    //使用循环迭代次数作为构造函数的参数
    explicit SoftMaxRegression(int cycle);
    //使用学习率、循环迭代次数
    SoftMaxRegression(float rate, int cycle);
    //进行数据的训练
    void train(Matrix& sample, Matrix& label);
private:
    //学习率，默认参数构造为0.01
    float rate;
    //循环迭代次数，默认为100次
    int cycles;
    //重新构建数据样本集
    virtual Matrix remake(Matrix & sample);
    //重新构建样本的标签数据集
    virtual Matrix rebuild(Matrix &label);
    //求出概率
    virtual Matrix softMax(Matrix&sample, Matrix&weight);
    //求出一行的最大值
    virtual int max(Matrix mat);
};

#endif //SoftMax_hpp
