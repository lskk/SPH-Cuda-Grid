#include "CSVManager.h"

CSVManager::CSVManager(string _filename,string _limiter){
	filename = _filename;
	limiter = _limiter;
}

void CSVManager::setFilename(string _filename){
	filename = _filename;
}

void CSVManager::setLimiter(string _limiter){
	limiter = _limiter;
}

void CSVManager::openCSV(){
	outputFile.open(filename);
}

void CSVManager::writeFile(int frame, double ms){
	outputFile << frame << limiter << ms << endl;
	//cout << frame << limiter << ms << endl;
}

void CSVManager::closeCSV(){
	outputFile.close();
}