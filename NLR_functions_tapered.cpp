#include<iostream>
#include<Eigen/Dense>
#include<iomanip>
#include<math.h>
#include"NLR_functions_tapered.h"
#include"fea_tapered_main.h"

using namespace std;
using namespace Eigen;

/*-----------------------------------------------------------------------------------------------------------------------------------------------------------*/

MatrixXd function_yt(MatrixXd para, double area_1, double area_2, double len, double t_total, double ForceMax) {

	MatrixXd y = fea_maint(para, area_1, area_2, len, t_total, ForceMax);

	return y;
};

/*-----------------------------------------------------------------------------------------------------------------------------------------------------------*/

MatrixXd getJacobianMatrixt(MatrixXd para_est, MatrixXd deflection, MatrixXd ym, double area_1, double area_2, double len, double t_total, double ForceMax) {

	MatrixXd Jacobian_Matrix(ym.rows(), para_est.rows());
	MatrixXd y = function_yt(para_est, area_1, area_2, len, t_total, ForceMax);
	MatrixXd y_deflected(ym.rows(), 1);

	for (int i = 0; i < para_est.rows(); i++) {

		para_est(i, 0) = para_est(i, 0) + deflection(i, 0);		/*Changing the parameters one by one */

		y_deflected = function_yt(para_est, area_1, area_2, len, t_total, ForceMax);				/*Computing the deflected function arrray */
		for (int j = 0; j < ym.rows(); j++) {

			// [f(v, p + dp) - f(v, p) ] / [dp] 

			Jacobian_Matrix(j, i) = (y_deflected(j, 0) - y(j, 0)) / deflection(i, 0);
		}
		para_est(i, 0) = para_est(i, 0) - deflection(i, 0);		/*Bringing back the parametes to original value*/
	}
	return Jacobian_Matrix;
};

/*-----------------------------------------------------------------------------------------------------------------------------------------------------------*/


MatrixXd LevenbergMarquardtFitt(MatrixXd para_guess, MatrixXd deflection, MatrixXd ym, double area_1, double area_2, double len, double t_total, double ForceMax) {

	//cout << "-> Entered LevenbergMarquardtFit\n";

	//cout << "para_guess: " << para_guess << endl;
	/*cout << "deflection: " << deflection << endl;
	cout << "ym: " << ym << endl;
	cout << "input: " << input << endl;*/

	int npara = para_guess.rows(), ndata = ym.rows();

	MatrixXd IdentityMat = MatrixXd::Identity(npara, npara);
	MatrixXd H(npara, npara);
	MatrixXd d(ndata, 1);
	MatrixXd J(ndata, npara);
	double error, error_lm,CHI2;

	MatrixXd y_init = function_yt(para_guess, area_1, area_2, len, t_total, ForceMax);
	//cout << "yinit: " << y_init << endl;

	double lambda = 10;
	int updateJ = 1;
	MatrixXd para_est = para_guess;
	int maxiter = 100, counter = 0;


	while (counter < maxiter) {

		cout << "--> Iteration : " << counter << endl;

		if (updateJ == 1) {
			//cout << "--> Inside IF" << endl;
			J = getJacobianMatrixt(para_est, deflection, ym, area_1, area_2, len, t_total, ForceMax);
			//cout << "J: \n" << J << endl;
			MatrixXd y_est = function_yt(para_est, area_1, area_2, len, t_total, ForceMax);
			d = ym - y_est;
			//cout << "d: \n" << d << endl;
			H = J.transpose() * J;
			//cout << "H: \n" << H << endl;

			if (counter == 0) {
				MatrixXd temp1 = d.transpose() * d;
				error = temp1(0, 0);
				//cout << "error" << error << endl;
			}
		}

		MatrixXd H_lm = H + lambda * IdentityMat;
		MatrixXd dp = H_lm.inverse() * J.transpose() * d;
		cout << "dp: \n" << dp;
		MatrixXd para_lm = para_est + dp;
		MatrixXd y_est_lm = function_yt(para_lm, area_1, area_2, len, t_total, ForceMax);
		MatrixXd d_lm = ym - y_est_lm;
		MatrixXd temp2 = d_lm.transpose() * d_lm;
		error_lm = temp2(0, 0);

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

		if (dp.norm() < 1e-4) {
			CHI2 = error;
			counter = maxiter;
		}
		else counter++;

	}
	cout << "Computation completed with CHI_SQUARE_ERROR = " << CHI2 << " and Lambda = " << lambda << " ." << endl;
	return para_est;
};

/*-----------------------------------------------------------------------------------------------------------------------------------------------------------*/

MatrixXd GaussNewtont(MatrixXd para_guess, MatrixXd deflection, MatrixXd ym, double area_1, double area_2, double len, double t_total, double ForceMax) {

	cout << "-> Entered Gauﬂ-Newton\n";

	//cout << "para_guess: " << para_guess << endl;
	/*cout << "deflection: " << deflection << endl;
	cout << "ym: " << ym << endl;
	cout << "input: " << input << endl;*/

	int npara = para_guess.rows(), ndata = ym.rows();

	//MatrixXd IdentityMat = MatrixXd::Identity(npara, npara);
	MatrixXd H(npara, npara);
	MatrixXd d(ndata, 1);
	MatrixXd J(ndata, npara);
	double error, error_gn, CHI2 ;

	MatrixXd y_init = function_yt(para_guess, area_1, area_2, len, t_total, ForceMax);
	//cout << "yinit: " << y_init << endl;

	//double lambda = 10;
	int updateJ = 1;
	MatrixXd para_est = para_guess;
	int maxiter = 1000, counter = 0;


	while (counter < maxiter) {

		cout << "--> Iteration : " << counter << endl;

		J = getJacobianMatrixt(para_est, deflection, ym, area_1, area_2, len, t_total, ForceMax);
		//cout << "J: \n" << J << endl;
		MatrixXd y_est = function_yt(para_est, area_1, area_2, len, t_total, ForceMax);
		d = ym - y_est;
		//cout << "d: \n" << d << endl;
		H = J.transpose() * J;
		cout << "H: \n" << H << endl;

		if (counter == 0) {
			MatrixXd temp1 = d.transpose() * d;
			error = temp1(0, 0);
			//cout << "error" << error << endl;
		}

		//cout << "Hinverse=\n" << H_lm.completeOrthogonalDecomposition().pseudoInverse() << endl;
		MatrixXd dp = H.completeOrthogonalDecomposition().pseudoInverse() * J.transpose() * d;
		cout << "dp: \n" << dp;
		MatrixXd para_gn = para_est + dp;
		MatrixXd y_est_gn = function_yt(para_gn, area_1, area_2, len, t_total, ForceMax);
		MatrixXd d_gn = ym - y_est_gn;
		MatrixXd temp2 = d_gn.transpose() * d_gn;
		error_gn = temp2(0, 0);

		para_est = para_gn;
		error = error_gn;

		if (dp.norm() < 1e-3) {
			CHI2 = error;
			counter = 1000;
		}
		else counter++;
		//cout << "Para:\n" << para_est << endl;
		cout << "dp:\n" << dp << endl;
	}

	cout << "Computation completed with CHI_SQUARE_ERROR = " << CHI2 << endl;
	return para_est;
};

MatrixXd GradientDescentt(MatrixXd para_guess, MatrixXd deflection, MatrixXd ym, double area_1, double area_2, double len, double t_total, double ForceMax) {

	cout << "-> Entered Gradient Descent\n";

	int npara = para_guess.rows(), ndata = ym.rows();

	MatrixXd d(ndata, 1);
	MatrixXd J(ndata, npara);
	double error, error_gd;

	//MatrixXd y_init = function_y(para_guess, input);

	double alpha = 3e-8;
	MatrixXd para_est = para_guess;
	MatrixXd para_gd, y_est_gd, y_est;
	int maxiter = 1000, counter = 0;

	while (counter < maxiter) {

		cout << "--> Iteration : " << counter << endl;

		if (counter == 0) {
			y_est = function_yt(para_est, area_1, area_2, len, t_total, ForceMax);
			d = ym - y_est;
		}
		else {
			y_est = y_est_gd;
			para_est = para_gd;
			error = error_gd;
			d = ym - y_est;
		}

		J = getJacobianMatrixt(para_est, deflection, ym, area_1, area_2, len, t_total, ForceMax);
		MatrixXd dp = alpha * J.transpose() * d;
		//cout << "dp: \n" << dp;
		para_gd = para_est + dp;
		y_est_gd = function_yt(para_gd, area_1, area_2, len, t_total, ForceMax);
		MatrixXd d_gd = ym - y_est_gd;
		MatrixXd temp2 = d_gd.transpose() * d_gd;
		error_gd = temp2(0, 0);
		cout << "err= " << error_gd << endl;

		//if (abs(error_gd - error) < 1 && error_gd < error) { alpha = alpha * 9.2; }

		if (error_gd >= error && counter < 60 && counter >1) {
			cout << "Wrong Direction !" << endl;
			para_gd = para_gd - dp;
			alpha =  alpha / 195;
			//break;
		}

		if ((error_gd >= error && counter > 60) || (error_gd > 100 && counter > 60) || (abs(error_gd - error) < 1e-1 && counter > 25)) {
			cout << "Wrong Direction ! Problem won't converge. Exiting Code. " << endl;
			break;
		}
		if (error_gd < 1) {
			counter = maxiter;
		}
		else counter++;
		

	}
	cout << "Computation completed with ERROR = " << error_gd << endl;
	return para_gd;
};

/*-----------------------------------------------------------------------------------------------------------------------------------------------------------*/
