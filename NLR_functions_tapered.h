#include<iostream>
#include<Eigen/Dense>

using namespace std;
using namespace Eigen;


MatrixXd function_yt(MatrixXd para, double area_1, double area_2, double len, double t_total, double ForceMax);
MatrixXd getJacobianMatrixt(MatrixXd para_est, MatrixXd deflection, MatrixXd ym, double area_1, double area_2, double len, double t_total, double ForceMax);
MatrixXd LevenbergMarquardtFitt(MatrixXd para_guess, MatrixXd deflection, MatrixXd ym, double area_1, double area_2, double len, double t_total, double ForceMax);
MatrixXd GaussNewtont(MatrixXd para_guess, MatrixXd deflection, MatrixXd ym, double area_1, double area_2, double len, double t_total, double ForceMax);
MatrixXd GradientDescentt(MatrixXd para_guess, MatrixXd deflection, MatrixXd ym, double area_1, double area_2, double len, double t_total, double ForceMax);
