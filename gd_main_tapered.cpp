#include<iostream>
#include<Eigen/Dense>
#include<iomanip>
#include<fstream>
#include<math.h>
#include<vector>
#include"src/rapidcsv.h"
#include"functions.h"
#include"plotme.h"

/*-----------------------------------------------------------------------------------------------------------------------------------------------------------*/

using namespace std;
using namespace Eigen;
using namespace rapidcsv;

/*-----------------------------------------------------------------------------------------------------------------------------------------------------------*/


void main() {

	cout << "Main program started..." << endl;

	
	Document dy("displacement.csv");

	vector<double> ycol = dy.GetColumn<double>(0);

	int data_size = ycol.size();

	MatrixXd y_measured(ycol.size(), 1);

	for (int u = 0; u < data_size; u++) {

		y_measured(u, 0) = ycol[u];
	}

	//PARAMETER 1 : Young's Modulus, PARAMETER 2 : Yield Stress, PARAMETER 3 : Viscosity
	MatrixXd parameters_guess{
		{119.77,242.0,18}  //Initial Guess for parameters: Scaled the first parameter by 1e3
	};

	MatrixXd initial_deflection{
		{1e-8,1e-8,1e-8}	//Initial deflection for parameters(to find Jacobian Matrix using finite difference scheme)
	};

	parameters_guess = parameters_guess.reshaped(parameters_guess.cols(), 1);
	initial_deflection = initial_deflection.reshaped(parameters_guess.rows(), 1);

	//INPUT DATA FOR FEA CODE
	double area_1 = 12;
	double area_2 = 4;
	double len = 180;
	double t_total = 0.025;
	double ForceMax = 4700;

	cout << "Data sourced, calling Gradient Descent. \n";

	MatrixXd Parameters = GradientDescent(parameters_guess, initial_deflection, y_measured, area_1, area_2, len, t_total, ForceMax);

	cout << "Parameters = \n" << Parameters << endl;

	ofstream myfitfile("final_fit.csv");

	MatrixXd final_fit = function_y(Parameters, area_1, area_2, len, t_total, ForceMax);

	for (int num = 0; num < final_fit.rows(); num++) {
		myfitfile << final_fit(num, 0) << endl;
	}

	myfitfile.close();
	cout << "Activating GNUPLOT using vcpkg and gnupuplot-iostream..\n";
	plot();


}
