//
// Created by LeiLei on 2020/6/28.
//

#include "SoftMax.hpp"
void read(const string& path, vector<string>&data){
    //判断文件是否存在
    try{
        if (!filesystem::exists(path))  throw MyException("【Error】the file is not exists");
    } catch (MyException & e) {
        cerr << e.what() << endl;
        return;
    }
    //读取文件
    fstream fs(path);
    string line;
    while(getline(fs, line))
        data.push_back(line);
}

vector<string>split(const string&s, const string&c){
    vector<string>result;
    int index = 0;
    string temp = s;
    while((index = temp.find(c))!=string::npos){
        result.push_back(temp.substr(0, index));
        temp = temp.substr(index+c.size(), temp.size()-index-c.size());
    }
    result.push_back(temp);
    return result;
}

string replace(const string &s, const string &oc, const string &nc){
    string result;
    int index = 0;
    string temp = s;
    while((index = temp.find(oc))!=string::npos){
        result += temp.substr(0, index) + nc;
        temp = temp.substr(index+oc.size(), temp.size()-index-oc.size());
    }
    result += temp;
    return result;
}

void makeMatrix(const vector<string>&data, Matrix & sample, Matrix& label){
    for(int i=0;i<data.size();i++){
        vector<string>temp = split(data.at(i), ",");
        for(int j=0;j<temp.size();j++){
            if (j == temp.size()-1)
                label.at(i, 0) = stoi(temp.at(j));
            else sample.at(i, j) = stod(temp.at(j));
        }
    }
}

SoftMaxRegression::SoftMaxRegression() : rate(0.01), cycles(100) {}

SoftMaxRegression::SoftMaxRegression(float rate) : rate(rate), cycles(100){}

SoftMaxRegression::SoftMaxRegression(int cycle) : rate(0.01), cycles(cycle){}

SoftMaxRegression::SoftMaxRegression(float rate, int cycle) : rate(rate), cycles(cycle){}

void SoftMaxRegression::train(Matrix &sample, Matrix &label) {
    Matrix samp = remake(sample);
    Matrix lab = rebuild(label);
    Matrix weight = Matrix::ones(samp.col, 3);
    for(int i=0;i<this->cycles;i++) {
        Matrix prob = this->softMax(samp, weight);
        Matrix error = prob - lab;
        weight = weight - samp.transpose().dot(error) * this->rate;
    }
    Matrix result = this->softMax(samp, weight);
    int count = 0;
    for(int i=0;i<result.row;i++){
        if (max(result.rows(i)) == max(lab.rows(i)))
            count ++;
    }
}

Matrix SoftMaxRegression::remake(Matrix & sample) {
    Matrix result = Matrix(sample.row, sample.col+1);
    for(int i=0;i<sample.row;i++){
        for (int j=0;j<result.col;j++)
            result.at(i, j) = j==0?1:sample.at(i, j-1);
    }
    return result;
}

Matrix SoftMaxRegression::rebuild(Matrix &label) {
    Matrix result = Matrix::zeros(label.row, 3);
    for(int i=0;i<label.row;i++){
        result.at(i, int(label.at(i, 0)-1)) = 1;
    }
    return result;
}

Matrix SoftMaxRegression::softMax(Matrix &sample, Matrix &weight) {
    Matrix data = sample.dot(weight).exp();
    Matrix result = Matrix(data.size());
    for(int i=0;i<data.row;i++){
        double sum = data.rows(i).transpose().sum();
        for(int j=0;j<data.col;j++)
            result.at(i, j) = data.at(i, j) / sum;
    }
    return result;
}

int SoftMaxRegression::max(Matrix mat) {
    double temp = -DBL_MAX;
    int j = 0;
    for(int i=0;i<mat.col;i++){
        if (mat.at(0, i) > temp) {
            temp = mat.at(0, i);
             j = i;
        }
    }
    return j;
}


