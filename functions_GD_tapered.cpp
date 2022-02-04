#include<iostream>
#include<Eigen/Dense>
#include<iomanip>
#include<math.h>
#include"functions.h"
#include"fea_main.h"

using namespace std;
using namespace Eigen;

/*-----------------------------------------------------------------------------------------------------------------------------------------------------------*/

MatrixXd function_y(MatrixXd para, double area_1, double area_2, double len, double t_total, double ForceMax) {

	MatrixXd y = fea_main(para, area_1, area_2, len, t_total, ForceMax);

	return y;
};

/*-----------------------------------------------------------------------------------------------------------------------------------------------------------*/

MatrixXd getJacobianMatrix(MatrixXd para_est, MatrixXd deflection, MatrixXd ym, double area_1, double area_2, double len, double t_total, double ForceMax) {

	MatrixXd Jacobian_Matrix(ym.rows(), para_est.rows());
	MatrixXd y = function_y(para_est, area_1, area_2, len, t_total, ForceMax);
	MatrixXd y_deflected(ym.rows(), 1);

	for (int i = 0; i < para_est.rows(); i++) {

		para_est(i, 0) = para_est(i, 0) + deflection(i, 0);		/*Changing the parameters one by one */

		y_deflected = function_y(para_est, area_1, area_2, len, t_total, ForceMax);				/*Computing the deflected function arrray */
		for (int j = 0; j < ym.rows(); j++) {

			// [f(v, p + dp) - f(v, p) ] / [dp] 

			Jacobian_Matrix(j, i) = (y_deflected(j, 0) - y(j, 0)) / deflection(i, 0);
		}
		para_est(i, 0) = para_est(i, 0) - deflection(i, 0);		/*Bringing back the parametes to original value*/
	}
	return Jacobian_Matrix;
};

/*-----------------------------------------------------------------------------------------------------------------------------------------------------------*/


MatrixXd GradientDescent(MatrixXd para_guess, MatrixXd deflection, MatrixXd ym, double area_1, double area_2, double len, double t_total, double ForceMax) {

	cout << "-> Entered Gradient Descent\n";

	int npara = para_guess.rows(), ndata = ym.rows();

	MatrixXd d(ndata, 1);
	MatrixXd J(ndata, npara);
	double error,error_gd;

	//MatrixXd y_init = function_y(para_guess, input);
	
	double alpha =5e-3;
	MatrixXd para_est = para_guess;
	MatrixXd para_gd, y_est_gd, y_est;
	int maxiter = 1000, counter=0;
	
	while (counter < maxiter) {
		
		cout << "--> Iteration : " << counter << endl;

		if (counter == 0) {
			y_est = function_y(para_est, area_1, area_2, len, t_total, ForceMax);
			d = ym - y_est;
		}
		else {
			y_est = y_est_gd;
			para_est = para_gd;
			error = error_gd;
			d = ym - y_est;
		}

		J = getJacobianMatrix(para_est, deflection, ym, area_1, area_2, len, t_total, ForceMax);
		MatrixXd dp = alpha * J.transpose() * d;
		//cout << "dp: \n" << dp;
		para_gd = para_est + dp;
		y_est_gd= function_y(para_gd, area_1, area_2, len, t_total, ForceMax);
		MatrixXd d_gd = ym - y_est_gd;
		MatrixXd temp2 = d_gd.transpose() * d_gd;
		error_gd = temp2(0,0);
		cout << "err= " << error_gd << endl;
		if (error_gd > error && counter != 0) {
			cout << "Wrong Direction !" << endl;
			break;
		}

		if (error_gd < 1e-2) {
			counter = maxiter;
		}
		else counter++;
		
		
	}

	return para_gd;
};

/*-----------------------------------------------------------------------------------------------------------------------------------------------------------*/
