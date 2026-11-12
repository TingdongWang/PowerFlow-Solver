#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>
#include<iostream>
#include <fstream>
#include <string>
#include<vector>
#include<map>
#include<list>
#include<set>
#include "Platform.h"
#include<algorithm>
#include<complex>
#include <iomanip>
using namespace std;
typedef struct s0 {
    int bus_number = 0;
    int bus_type = 0;
    long double Pd = 0.0;         //结点注入有功
    long double Qd = 0.0;         //结点注入无功
    long double Gs = 0.0;
    long double Bs = 0.0;
    long double area = 0.0;
    long double Vm = 0.0;
    long double Va = 0.0;
    long double baseKV = 0.0;
    long double zone = 0.0;
    long double maxVm = 0.0;
    long double minVm = 0.0;
}Bus;
typedef struct s6 {
    double a[50];
    double all;
}cal_time;

typedef struct s1 {
    int fr = 0;
    int to = 0;
    long double r = 0.0;         //结点注入有功
    long double x = 0.0;         //结点注入无功
    long double b = 0.0;
    long double rateA = 0.0;
    long double rateB = 0.0;
    long double rateC = 0.0;
    long double _ratio = 0.0;
    long double angel = 0.0;
    long double status = 0.0;
    long double angmin = 0.0;
    long double angmax = 0.0;
}Line;

typedef struct gendata {
    long double Pg;
    long double Qg;
    long double Qmax;
    long double Qmin;
}Gen;

typedef struct s2 {
    int type = 0;
    long double mes_value = 0.0;
    long double add_error = 0.0;
    long double true_value = 0.0;
    long double weight = 0.0;
    int fr = 0;
    int to = 0;
    long double mes_index = 0.0;
}Mdata;

typedef struct s3 {
    Bus bus;
    vector<Line> lines;
    Gen gendata;
    pair<double, double> ans;
}node_Physics;

typedef struct s4 {
    //第一步初始化：得到Ya Yr 和 邻接结点
    //vector< pair <int, complex<double>>> Ya; //to,value的形式
    vector< pair <int, complex<double>>> Yr; //存放到vector中
    // 自导纳数据
    complex<double> Y;
    // 测量节点
    long double p = 0.0, q = 0.0, v = 0.0;
    long double ori_p = 0.0, ori_q = 0.0, ori_v = 0.0;
    map<int, double>H, M, K, L;
    //右边向量
    long double right_hp = 0.0;
    long double right_hq = 0.0;
}Node;


struct PerturbConfig {
    string name;
    // 相角扰动范围
    double theta_delta_min;
    double theta_delta_max;
    // 电压扰动范围
    double V_delta_min;
    double V_delta_max;
    // 电压安全约束
    double V_min;
    double V_max;
};

const PerturbConfig& choose_perturb_config(
    const PerturbConfig& cfg_small,
    const PerturbConfig& cfg_mid,
    const PerturbConfig& cfg_large,
    std::default_random_engine& generator)
{
    static std::uniform_real_distribution<double> dist01(0.0, 1.0);
    double r = dist01(generator);

    // return cfg_small;
    // 50% small, 30% mid, 20% large
    if (r < 0.60) return cfg_small;
    else if (r < 0.80) return cfg_mid;
    else return cfg_large;
}
/*
struct PerturbConfig {
    // 相角扰动范围
    double theta_delta_min = 0.0005;   // rad
    double theta_delta_max = 0.02;    // rad
    // 电压扰动范围
    double V_delta_min = 0.0005;       // pu
    double V_delta_max = 0.005;        // pu
    // 电压安全约束
    double V_min = 0.90;              // pu
    double V_max = 1.10;              // pu
};
*/
