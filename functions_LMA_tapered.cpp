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
	
	MatrixXd y = fea_main(para,  area_1,  area_2,  len,  t_total,  ForceMax);
	
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


MatrixXd LevenbergMarquardtFit(MatrixXd para_guess, MatrixXd deflection, MatrixXd ym, double area_1, double area_2, double len, double t_total, double ForceMax) {

	cout << "-> Entered LevenbergMarquardtFit\n";

	//cout << "para_guess: " << para_guess << endl;
	/*cout << "deflection: " << deflection << endl;
	cout << "ym: " << ym << endl;
	cout << "input: " << input << endl;*/

	int npara = para_guess.rows(), ndata = ym.rows();

	MatrixXd IdentityMat = MatrixXd::Identity(npara, npara);
	MatrixXd H(npara, npara);
	MatrixXd d(ndata, 1);
	MatrixXd J(ndata, npara);
	double error,error_lm;

	MatrixXd y_init = function_y(para_guess, area_1, area_2, len, t_total, ForceMax);
	//cout << "yinit: " << y_init << endl;

	double lambda = 10;
	int updateJ = 1;
	MatrixXd para_est = para_guess;
	int maxiter = 100, counter=0;
	

	while (counter < maxiter) {
		
		cout << "--> Iteration : " << counter << endl;

		if (updateJ == 1) {
			//cout << "--> Inside IF" << endl;
			J = getJacobianMatrix(para_est, deflection, ym, area_1, area_2, len, t_total, ForceMax);
			//cout << "J: \n" << J << endl;
			MatrixXd y_est = function_y(para_est, area_1, area_2, len, t_total, ForceMax);
			d = ym - y_est;
			//cout << "d: \n" << d << endl;
			H = J.transpose() * J;
			//cout << "H: \n" << H << endl;

			if (counter == 0) {
				MatrixXd temp1 = d.transpose() * d;
				error = temp1(0,0);
				//cout << "error" << error << endl;
			}
		}

		MatrixXd H_lm = H + lambda * IdentityMat;
		MatrixXd dp = H_lm.inverse() * J.transpose() * d;
		cout << "dp: \n" << dp;
		MatrixXd para_lm = para_est + dp;
		MatrixXd y_est_lm = function_y(para_lm, area_1, area_2, len, t_total, ForceMax);
		MatrixXd d_lm = ym - y_est_lm;
		MatrixXd temp2 = d_lm.transpose() * d_lm;
		error_lm = temp2(0,0);

		if (error_lm < error) {
			lambda = lambda / 10;
			para_est = para_lm;
			error = error_lm;
			updateJ = 1;
			cout << "Lambda Decr. to: " << lambda << endl;
		}
		else {
			updateJ = 0;
			lambda = lambda * 10;
			cout << "Lambda Incr. to: " << lambda << endl;
		}

		if (dp.norm() < 1e-6) {
			counter = 100;
		}
		else counter++;
		
	}

	return para_est;
};

/*-----------------------------------------------------------------------------------------------------------------------------------------------------------*/
