#include<iostream>
#include<Eigen/Dense>
#include<iomanip>
#include<fstream>
#include<math.h>
#include<vector>
#include"src/rapidcsv.h"
#include"NLR_functions_2_ele.h"
#include"NLR_functions_tapered.h"
#include"plotme.h"

/*-----------------------------------------------------------------------------------------------------------------------------------------------------------*/

using namespace std;
using namespace Eigen;
using namespace rapidcsv;

/*-----------------------------------------------------------------------------------------------------------------------------------------------------------*/

void main() {

	cout << "Main program started...\n" << endl;

	MatrixXd initial_deflection{
		{1e-8,1e-8,1e-8}	//Initial deflection for parameters(to find Jacobian Matrix using finite difference scheme)
	};

	/*-------------------------------------------------------------------------------------------------------------------------------------------------------*/

	cout << "Sourcing Data for 2 Bar Problem..." << endl;
	Document dy2ele("displacement_2_bar.csv");
	vector<double> ycol2ele = dy2ele.GetColumn<double>(0);
	int data_size2ele = ycol2ele.size();
	MatrixXd y_measured_2_ele(ycol2ele.size(), 1);
	for (int u = 0; u < data_size2ele; u++) {
		y_measured_2_ele(u, 0) = ycol2ele[u];
	}
	
	//INPUT DATA FOR 2 Bar FEA CODE

	Document inp2bar("UserInputFile_PARA_2BarFEA.csv");
	vector<double> user_input_2bar = inp2bar.GetColumn<double>(0);

	double area_1 = user_input_2bar[1];
	double area_2 = user_input_2bar[2];
	double len_1 = user_input_2bar[3];
	double len_2 = user_input_2bar[4];
	double t_total = user_input_2bar[5];
	double ForceMax = user_input_2bar[0];

	//True Parameters:
	MatrixXd true_parameters_2ele{
		{user_input_2bar[12],user_input_2bar[13],user_input_2bar[14]}
	};

	cout << "Data sourced for 2 Bar Problem.\n" << endl;
	cout<<"Computing Parameters using Levenberg-Marquardt Algorithm... \n";

	//PARAMETER 1 : Young's Modulus, PARAMETER 2 : Yield Stress, PARAMETER 3 : Viscosity
	MatrixXd LMA_parameters_guess_2ele{
		{user_input_2bar[6],user_input_2bar[7],user_input_2bar[8]}  //Initial Guess for parameters: Scaled the first parameter by 1e3
	};

	LMA_parameters_guess_2ele = LMA_parameters_guess_2ele.reshaped(LMA_parameters_guess_2ele.cols(), 1);
	initial_deflection = initial_deflection.reshaped(LMA_parameters_guess_2ele.rows(), 1);
	MatrixXd Parameters_LMA_2ele = LevenbergMarquardtFit(LMA_parameters_guess_2ele, initial_deflection, y_measured_2_ele, area_1, area_2, len_1, len_2, t_total, ForceMax);
	cout << "\nParameters Obtained using LMA for 2 Element Problem = \n" << Parameters_LMA_2ele << endl;
	ofstream myfitfile1("LMA_fit_2ele.csv");
	MatrixXd final_fit_2ele1 = function_y(Parameters_LMA_2ele, area_1, area_2, len_1, len_2, t_total, ForceMax);
	for (int num = 0; num < final_fit_2ele1.rows(); num++) {
		myfitfile1 << final_fit_2ele1(num, 0) << endl;
	}
	myfitfile1.close();
	cout << "\n*************************************************************************\n" << endl;
	cout << "\nComputing Parameters using Gauss-Newton Algorithm... \n";
	//PARAMETER 1 : Young's Modulus, PARAMETER 2 : Yield Stress, PARAMETER 3 : Viscosity
	MatrixXd GN_parameters_guess_2ele{
		{user_input_2bar[6],user_input_2bar[7],user_input_2bar[8]}  //Initial Guess for parameters: Scaled the first parameter by 1e3
	};
	GN_parameters_guess_2ele = GN_parameters_guess_2ele.reshaped(GN_parameters_guess_2ele.cols(), 1);
	MatrixXd Parameters_GN_2ele = GaussNewton(GN_parameters_guess_2ele, initial_deflection, y_measured_2_ele, area_1, area_2, len_1, len_2, t_total, ForceMax);
	cout << "\nParameters Obtained using GN for 2 Element Problem = \n" << Parameters_GN_2ele << endl;
	ofstream myfitfile2("GN_fit_2ele.csv");
	MatrixXd final_fit_2ele2 = function_y(Parameters_GN_2ele, area_1, area_2, len_1, len_2, t_total, ForceMax);
	for (int num = 0; num < final_fit_2ele2.rows(); num++) {
		myfitfile2 << final_fit_2ele2(num, 0) << endl;
	}
	myfitfile2.close();

	cout << "\n*************************************************************************\n" << endl;
	cout << "\nComputing Parameters using Gradient-Descent Algorithm... \n";
	//PARAMETER 1 : Young's Modulus, PARAMETER 2 : Yield Stress, PARAMETER 3 : Viscosity
	MatrixXd GD_parameters_guess_2ele{
		{user_input_2bar[9],user_input_2bar[10],user_input_2bar[11]}  //Initial Guess for parameters: Scaled the first parameter by 1e3
	};
	GD_parameters_guess_2ele = GD_parameters_guess_2ele.reshaped(GD_parameters_guess_2ele.cols(), 1);
	MatrixXd Parameters_GD_2ele = GradientDescent(GD_parameters_guess_2ele, initial_deflection, y_measured_2_ele, area_1, area_2, len_1, len_2, t_total, ForceMax);
	cout << "\nParameters Obtained using GD for 2 Element Problem = \n" << Parameters_GD_2ele << endl;
	ofstream myfitfile3("GD_fit_2ele.csv");
	MatrixXd final_fit_2ele3 = function_y(Parameters_GD_2ele, area_1, area_2, len_1, len_2, t_total, ForceMax);
	for (int num = 0; num < final_fit_2ele3.rows(); num++) {
		myfitfile3 << final_fit_2ele3(num, 0) << endl;
	}
	myfitfile3.close();

	ofstream input2ele("InputFile_NLR_2BAR.txt");

	input2ele << "This file contains Input Details for Non-Linear Regression Code : \n" << endl;
	input2ele << "\nInitial Deflection for Finite Difference (for Jacobian) = " << initial_deflection << endl;
	input2ele << "\n2 Bar Problem Input for FEA Code:" << endl;
	input2ele << "\nA1=" << area_1 << ", A2=" << area_2 << ", L1=" << len_1 << ", L2=" << len_2 << ", t=" << t_total << ", F=" << ForceMax << endl;
	input2ele << "\n--------------------------------------------------------------------------\n" << endl;
	input2ele << "\nInitial Guess of Parameters for Levenberg Marquardt Algorithm: " << endl;
	input2ele << "\nYoung's Modulus = " << LMA_parameters_guess_2ele(0, 0) * 1e3 << " ; Yield Stress = " << LMA_parameters_guess_2ele(1, 0) << " ; Viscosity = " << LMA_parameters_guess_2ele(2, 0) << endl;
	input2ele << "\nInitial Guess of Parameters for Gauss Newton Algorithm: " << endl;
	input2ele << "\nYoung's Modulus = " << GN_parameters_guess_2ele(0, 0) * 1e3 << " ; Yield Stress = " << GN_parameters_guess_2ele(1, 0) << " ; Viscosity = " << GN_parameters_guess_2ele(2, 0) << endl;
	input2ele << "\nInitial Guess of Parameters for Gradient Descent Algorithm: " << endl;
	input2ele << "\nYoung's Modulus = " << GD_parameters_guess_2ele(0, 0) * 1e3 << " ; Yield Stress = " << GD_parameters_guess_2ele(1, 0) << " ; Viscosity = " << GD_parameters_guess_2ele(2, 0) << endl;

	input2ele.close();

	/*-------------------------------------------------------------------------------------------------------------------------------------------------------*/


	cout << "Sourcing Data for Tapered Problem..." << endl;
	Document dyt("displacement_tapered_10e.csv");
	vector<double> ycolt = dyt.GetColumn<double>(0);
	int data_sizet = ycolt.size();
	MatrixXd y_measured_t(ycol2ele.size(), 1);
	for (int u = 0; u < data_sizet; u++) {
		y_measured_t(u, 0) = ycolt[u];
	}

	//INPUT DATA FOR TAPERED BAR FEA CODE

	Document inpTaperedbar("UserInputFile_PARA_TaperedBarFEA.csv");
	vector<double> user_input_Tbar = inpTaperedbar.GetColumn<double>(0);

	double area_1t = user_input_Tbar[1];
	double area_2t = user_input_Tbar[2];
	double len = user_input_Tbar[3];
	double t_totalt = user_input_Tbar[4];
	double ForceMaxt = user_input_Tbar[0];

	MatrixXd true_parameters_t{
		{user_input_Tbar[12],user_input_Tbar[13],user_input_Tbar[14]}
	};

	cout << "Data sourced for Tapered Problem.\n" << endl;
	cout << "\nComputing Parameters using Levenberg-Marquardt Algorithm... \n";

	//PARAMETER 1 : Young's Modulus, PARAMETER 2 : Yield Stress, PARAMETER 3 : Viscosity
	MatrixXd LMA_parameters_guess_t{
		{user_input_Tbar[6],user_input_Tbar[7],user_input_Tbar[8]}  //Initial Guess for parameters: Scaled the first parameter by 1e3
	};
	LMA_parameters_guess_t = LMA_parameters_guess_t.reshaped(LMA_parameters_guess_t.cols(), 1);
	MatrixXd Parameters_LMA_t = LevenbergMarquardtFitt(LMA_parameters_guess_t, initial_deflection, y_measured_t, area_1t, area_2t, len, t_totalt, ForceMaxt);
	cout << "\nParameters Obtained using LMA for Tapered Problem = \n" << Parameters_LMA_t << endl;
	ofstream myfitfile4("LMA_fit_t.csv");
	MatrixXd final_fit_t4 = function_yt(Parameters_LMA_t,  area_1t,  area_2t,  len,  t_totalt,  ForceMaxt);
	for (int num = 0; num < final_fit_t4.rows(); num++) {
		myfitfile4 << final_fit_t4(num, 0) << endl;
	}
	myfitfile4.close();

	cout << "\n*************************************************************************\n" << endl;
	cout << "\nComputing Parameters using Gauss-Newton Algorithm... \n";
	//PARAMETER 1 : Young's Modulus, PARAMETER 2 : Yield Stress, PARAMETER 3 : Viscosity
	MatrixXd GN_parameters_guess_t{
		{user_input_Tbar[6],user_input_Tbar[7],user_input_Tbar[8]}  //Initial Guess for parameters: Scaled the first parameter by 1e3
	};
	GN_parameters_guess_t = GN_parameters_guess_t.reshaped(GN_parameters_guess_t.cols(), 1);
	MatrixXd Parameters_GN_t = GaussNewtont(GN_parameters_guess_t, initial_deflection, y_measured_t, area_1t, area_2t, len, t_totalt, ForceMaxt);
	cout << "\nParameters Obtained using GN for Tapered Problem = \n" << Parameters_GN_t << endl;
	ofstream myfitfile5("GN_fit_t.csv");
	MatrixXd final_fit_t5 = function_yt(Parameters_GN_t, area_1t, area_2t, len, t_totalt, ForceMaxt);
	for (int num = 0; num < final_fit_t5.rows(); num++) {
		myfitfile5 << final_fit_t5(num, 0) << endl;
	}
	myfitfile5.close();

	cout << "\n*************************************************************************\n" << endl;
	cout << "\nComputing Parameters using Gradient-Descent Algorithm... \n";
	//PARAMETER 1 : Young's Modulus, PARAMETER 2 : Yield Stress, PARAMETER 3 : Viscosity
	MatrixXd GD_parameters_guess_t{
		{user_input_Tbar[9],user_input_Tbar[10],user_input_Tbar[11]}  //Initial Guess for parameters: Scaled the first parameter by 1e3
	};
	GD_parameters_guess_t = GD_parameters_guess_t.reshaped(GD_parameters_guess_t.cols(), 1);
	MatrixXd Parameters_GD_t = GradientDescentt(GD_parameters_guess_t, initial_deflection, y_measured_t, area_1t, area_2t, len, t_totalt, ForceMaxt);
	cout << "\nParameters Obtained using GD for Tapered Problem = \n" << Parameters_GD_t << endl;
	ofstream myfitfile6("GD_fit_t.csv");
	MatrixXd final_fit_t6 = function_yt(Parameters_GD_t, area_1t, area_2t, len, t_totalt, ForceMaxt);
	for (int num = 0; num < final_fit_t5.rows(); num++) {
		myfitfile6 << final_fit_t6(num, 0) << endl;
	}
	myfitfile6.close();

	ofstream inputtap("InputFile_NLR_TAPERED.txt");

	inputtap << "This file contains the Input data for Non-Linear Regression Code: \n" << endl;
	inputtap << "\nInitial Deflection for Finite Difference (for Jacobian) = " << initial_deflection << endl;
	inputtap << "\nTapered Bar Problem Input for FEA Code:" << endl;
	inputtap << "\nA1=" << area_1t << ", A2=" << area_2t << ", L=" << len << ", t=" << t_totalt << ", F=" << ForceMaxt << endl;
	input2ele << "\n--------------------------------------------------------------------------\n" << endl;
	inputtap << "\nInitial Guess of Parameters for Levenberg Marquardt Algorithm: " << endl;
	inputtap << "\nYoung's Modulus = " << LMA_parameters_guess_t(0, 0) * 1e3 << " ; Yield Stress = " << LMA_parameters_guess_t(1, 0) << " ; Viscosity = " << LMA_parameters_guess_t(2, 0) << endl;
	inputtap << "\nInitial Guess of Parameters for Gauss Newton Algorithm: " << endl;
	inputtap << "\nYoung's Modulus = " << GN_parameters_guess_t(0, 0) * 1e3 << " ; Yield Stress = " << GN_parameters_guess_t(1, 0) << " ; Viscosity = " << GN_parameters_guess_t(2, 0) << endl;
	inputtap << "\nInitial Guess of Parameters for Gradient Descent Algorithm: " << endl;
	inputtap << "\nYoung's Modulus = " << GD_parameters_guess_t(0, 0) * 1e3 << " ; Yield Stress = " << GD_parameters_guess_t(1, 0) << " ; Viscosity = " << GD_parameters_guess_t(2, 0) << endl;

	inputtap.close();


	cout << "\n ********************************************************************************************* \n";
	cout << "\nParameters Obtained using LMA for 2 Bar Problem = \n" << Parameters_LMA_2ele << endl;
	cout << "\nParameters Obtained using GN for 2 Bar Problem = \n" << Parameters_GN_2ele << endl;
	cout << "\nParameters Obtained using GD for 2 Bar Problem = \n" << Parameters_GD_2ele << endl;
	cout << "\nParameters Obtained using LMA for Tapered Problem = \n" << Parameters_LMA_t << endl;
	cout << "\nParameters Obtained using GN for Tapered Problem = \n" << Parameters_GN_t << endl;
	cout << "\nParameters Obtained using GD for Tapered Problem = \n" << Parameters_GD_t << endl;

	ofstream output("OutputLog.txt");

	output << "\n ********************************************************************************************* \n";
	output<< "\nParameters Obtained using Levenberg-Marquardt Algorithm for 2 Bar Problem : "<< endl;
	output << "\nYoung's Modulus = " << Parameters_LMA_2ele(0, 0) * 1e3 << " ; Yield Stress = " << Parameters_LMA_2ele(1, 0) << " ; Viscosity = " << Parameters_LMA_2ele(2, 0) << endl;
	output << "\n ********************************************************************************************* \n";
	output << "\nParameters Obtained using Gauss_Newton Algorithm for 2 Bar Problem : " << endl;
	output << "\nYoung's Modulus = " << Parameters_GN_2ele(0, 0) * 1e3 << " ; Yield Stress = " << Parameters_GN_2ele(1, 0) << " ; Viscosity = " << Parameters_GN_2ele(2, 0) << endl;
	output << "\n ********************************************************************************************* \n";
	output << "\nParameters Obtained using Gradient-Descent Algorithm for 2 Bar Problem : " << endl;
	output << "\nYoung's Modulus = " << Parameters_GD_2ele(0, 0) * 1e3 << " ; Yield Stress = " << Parameters_GD_2ele(1, 0) << " ; Viscosity = " << Parameters_GD_2ele(2, 0) << endl;
	output << "\n ********************************************************************************************* \n";
	output << endl;
	output << "\n ********************************************************************************************* \n";
	output << "\nParameters Obtained using Levenberg-Marquardt Algorithm for Tapered Bar Problem : " << endl;
	output << "\nYoung's Modulus = " << Parameters_LMA_t(0, 0) * 1e3 << " ; Yield Stress = " << Parameters_LMA_t(1, 0) << " ; Viscosity = " << Parameters_LMA_t(2, 0) << endl;
	output << "\n ********************************************************************************************* \n";
	output << "\nParameters Obtained using Gauss_Newton Algorithm for Tapered Bar Problem : " << endl;
	output << "\nYoung's Modulus = " << Parameters_GN_t(0, 0) * 1e3 << " ; Yield Stress = " << Parameters_GN_t(1, 0) << " ; Viscosity = " << Parameters_GN_t(2, 0) << endl;
	output << "\n ********************************************************************************************* \n";
	output << "\nParameters Obtained using Gradient-Descent Algorithm for Tapered Bar Problem : " << endl;
	output << "\nYoung's Modulus = " << Parameters_GD_t(0, 0) * 1e3 << " ; Yield Stress = " << Parameters_GD_t(1, 0) << " ; Viscosity = " << Parameters_GD_t(2, 0) << endl;
	output << "\n ********************************************************************************************* \n";

	double error_2ele_LMA=0, error_2ele_GN = 0, error_2ele_GD = 0;
	double error_t_LMA = 0, error_t_GN = 0, error_t_GD = 0;
	//double e1, e2, e3;

	for (int v = 0; v < 3; v++) {

		error_2ele_LMA = error_2ele_LMA + abs(Parameters_LMA_2ele(v, 0) - true_parameters_2ele(0, v)) * 1e2 / true_parameters_2ele(0, v);
		error_2ele_GN = error_2ele_GN + abs(Parameters_GN_2ele(v, 0) - true_parameters_2ele(0, v)) * 1e2 / true_parameters_2ele(0, v);
		error_2ele_GD = error_2ele_GD + abs(Parameters_GD_2ele(v, 0) - true_parameters_2ele(0, v)) * 1e2 / true_parameters_2ele(0, v);
		error_t_LMA = error_t_LMA + abs(Parameters_LMA_t(v, 0) - true_parameters_t(0, v)) * 1e2 / true_parameters_t(0, v);
		error_t_GN = error_t_GN + abs(Parameters_GN_t(v, 0) - true_parameters_t(0, v)) * 1e2 / true_parameters_t(0, v);
		error_t_GD = error_t_GD + abs(Parameters_GD_t(v, 0) - true_parameters_t(0, v)) * 1e2 / true_parameters_t(0, v);
	}


	output << endl;
	output << endl;
	output << "\n ********************************************************************************************* \n" << endl;

	cout << "Average Error using LevenbergMarquardt Algorithm for 2 Bar Problem = " << error_2ele_LMA / 3 << " %" << endl;
	cout << "Average Error using Gauss Newton Algorithm for 2 Bar Problem = " << error_2ele_GN / 3 << " %" << endl;
	cout << "Average Error using Gradient Descent Algorithm for 2 Bar Problem = " << error_2ele_GD / 3 << " %" << endl;
	cout << "Average Error using LevenbergMarquardt Algorithm for Tapered Bar Problem = " << error_t_LMA / 3 << " %" << endl;
	cout << "Average Error using Gauss Newton Algorithm for Tapered Bar Problem = " << error_t_GN / 3 << " %" << endl;
	cout << "Average Error using Gradient Descent Algorithm for Tapered Bar Problem = " << error_t_GD / 3 << " %" << endl;
	
	output << endl;
	output << "Average Error using LevenbergMarquardt Algorithm for 2 Bar Problem = " << error_2ele_LMA / 3 << " %" << endl;
	output << "Average Error using Gauss Newton Algorithm for 2 Bar Problem = " << error_2ele_GN / 3 << " %" << endl;
	output << "Average Error using Gradient Descent Algorithm for 2 Bar Problem = " << error_2ele_GD / 3 << " %" << endl;
	output << "Average Error using LevenbergMarquardt Algorithm for Tapered Bar Problem = " << error_t_LMA / 3 << " %" << endl;
	output << "Average Error using Gauss Newton Algorithm for Tapered BAr Problem = " << error_t_GN / 3 << " %" << endl;
	output << "Average Error using Gradient Descent Algorithm for Tapered Bar Problem = " << error_t_GD / 3 << " %" << endl;

	output.close();

	cout << "Activating GNUPLOT using vcpkg and gnupuplot-iostream..\n";

	plot();
}