#include<iostream>
#include<vector>
#include"src/rapidcsv.h"
#include"gnuplot-iostream.h"
#include"plotme.h"

using namespace std;
using namespace rapidcsv;

void plot() {

	Gnuplot gp("\"C:\\Program Files\\gnuplot\\bin\\gnuplot.exe\"");
	Document dmeasured2ele("displacement_2_bar.csv");
	Document dmeasuredt("displacement_tapered_10e.csv");

	Document dfitGD_2ele("GD_fit_2ele.csv");
	Document dfitGN_2ele("GN_fit_2ele.csv");
	Document dfitLMA_2ele("LMA_fit_2ele.csv");

	Document dfitGD_t("GD_fit_t.csv");
	Document dfitGN_t("GN_fit_t.csv");
	Document dfitLMA_t("LMA_fit_t.csv");

	vector<double> getmeasured2ele = dmeasured2ele.GetColumn<double>(0);
	vector<double> getmeasuredt = dmeasuredt.GetColumn<double>(0);

	vector<double> getfit_GD_2ele = dfitGD_2ele.GetColumn<double>(0);
	vector<double> getfit_GN_2ele = dfitGN_2ele.GetColumn<double>(0);
	vector<double> getfit_LMA_2ele = dfitLMA_2ele.GetColumn<double>(0);

	vector<double> getfit_GD_t = dfitGD_t.GetColumn<double>(0);
	vector<double> getfit_GN_t = dfitGN_t.GetColumn<double>(0);
	vector<double> getfit_LMA_t = dfitLMA_t.GetColumn<double>(0);

	gp << "set multiplot layout 2,3 rowsfirst\n";
	gp << "set xlabel 'Time-Steps' font 'Times - Roman, 10'\n";
	gp << "set ylabel 'Displacement' font 'Times - Roman, 10'\n";
	gp << "set xtics font 'Arial, 7'\n";
	gp << "set ytics font 'Arial, 7'\n";

	gp << "set size 0.36, 0.5\n";
	gp << "set title 'LM-FIT vs Measurement:2 Element' font ',8\n";
	gp << "plot '-' with lines title 'Measured',"
		<< "'-' with lines title 'Fit'\n";

	gp.send(getmeasured2ele);
	gp.send(getfit_LMA_2ele);

	gp << "set size 0.36, 0.5\n";
	gp << "set title 'GN-FIT vs Measurement:2 Element' font ',8\n";
	gp << "plot '-' with lines title 'Measured',"
		<< "'-' with lines title 'Fit'\n";

	gp.send(getmeasured2ele);
	gp.send(getfit_GN_2ele);

	gp << "set size 0.36, 0.5\n";
	gp << "set title 'GD-FIT vs Measurement:2 Element' font ',8\n";
	gp << "plot '-' with lines title 'Measured',"
		<< "'-' with lines title 'Fit'\n";

	gp.send(getmeasured2ele);
	gp.send(getfit_GD_2ele);

	gp << "set size 0.36, 0.5\n";
	gp << "set title 'LM-FIT vs Measurement:Tapered Bar' font ',8\n";
	gp << "plot '-' with lines title 'Measured',"
		<< "'-' with lines title 'Fit'\n";

	gp.send(getmeasuredt);
	gp.send(getfit_LMA_t);

	gp << "set size 0.36, 0.5\n";
	gp << "set title 'GN-FIT vs Measurement:Tapered Bar' font ',8\n";
	gp << "plot '-' with lines title 'Measured',"
		<< "'-' with lines title 'Fit'\n";

	gp.send(getmeasuredt);
	gp.send(getfit_GN_t);

	gp << "set size 0.36, 0.5\n";
	gp << "set title 'GD-FIT vs Measurement:Tapered Bar' font ',8\n";
	gp << "plot '-' with lines title 'Measured',"
		<< "'-' with lines title 'Fit'\n";

	gp.send(getmeasuredt);
	gp.send(getfit_GD_t);


	cin.get();



};