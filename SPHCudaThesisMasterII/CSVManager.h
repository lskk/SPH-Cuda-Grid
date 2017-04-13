#ifndef CSVMANAGER_H
#define CSVMANAGER_H

#include <iostream>
#include <fstream>
#include <string>
#include <string.h>

using namespace std;

class CSVManager{
public:
	CSVManager(){};
	CSVManager(string _filename, string _limiter);
	void setFilename(string _filename);
	void setLimiter(string _limiter);
	void openCSV();
	void writeFile(int frame, double ms);
	void closeCSV();
private:
	string filename;
	ofstream outputFile;
	string limiter;
};

#endif