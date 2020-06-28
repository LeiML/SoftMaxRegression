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
//��ȡ�ļ�
void read(const string& path, vector<string>&data);
//�ַ����ָ�
vector<string>split(const string&s, const string&c);
//�ַ����滻
string replace(const string &s, const string &oc, const string &nc);
//��������������������
void makeMatrix(const vector<string>&data, Matrix & sample, Matrix& label);
//SoftMax���߼��ع鹹������
class SoftMaxRegression{
public:
    //�޲������캯��
    SoftMaxRegression();
    //ʹ��ѧϰ����Ϊ��������
    explicit SoftMaxRegression(float rate);
    //ʹ��ѭ������������Ϊ���캯���Ĳ���
    explicit SoftMaxRegression(int cycle);
    //ʹ��ѧϰ�ʡ�ѭ����������
    SoftMaxRegression(float rate, int cycle);
    //�������ݵ�ѵ��
    void train(Matrix& sample, Matrix& label);
private:
    //ѧϰ�ʣ�Ĭ�ϲ�������Ϊ0.01
    float rate;
    //ѭ������������Ĭ��Ϊ100��
    int cycles;
    //���¹�������������
    virtual Matrix remake(Matrix & sample);
    //���¹��������ı�ǩ���ݼ�
    virtual Matrix rebuild(Matrix &label);
    //�������
    virtual Matrix softMax(Matrix&sample, Matrix&weight);
    //���һ�е����ֵ
    virtual int max(Matrix mat);
};

#endif //SoftMax_hpp
