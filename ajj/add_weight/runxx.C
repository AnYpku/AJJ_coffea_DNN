#include "ajj.C"
#include <iostream>
#include <fstream>
using namespace std;

void runxx(TString year,TString dir) {
	ifstream infile("file"+year);
        float xs;TString infilename;

	while (!infile.eof()){
		infile>>infilename>>xs;
		if(infilename.Contains("end")==1)  break;
		if(infilename.Contains("root")==0) continue;

		TString outname="cutjet-out"+infilename;

		TFile *file1 =new TFile(dir+infilename);
		TTree *tree1 = (TTree*) file1->Get("Events");
		ajj m1(tree1,outname);
		cout<<"outname: "<<outname<<endl;
		TH1D*h1;float nevents=0;
		if(outname.Contains("EGamma")==0 && outname.Contains("Muon")==0){
			h1=(TH1D*)file1->Get("nEventsGenWeighted");
			nevents=h1->GetSum();
		}
		else nevents=1;
		cout<<dir<<infilename<<" -> "<<outname<<endl;
		m1.Loop(outname,nevents,xs,year);
		m1.endJob();

	}
}

int main(){
//	TString dir18 ="/home/pku/anying/cms/rootfiles/Ajj/";
//	TString dir17 ="/home/pku/anying/cms/rootfiles/ajj/2017/";
	TString dir16 ="/home/pku/anying/cms/rootfiles/Hugo/";
//	runxx("18",dir18);
//	runxx("17",dir17);
	runxx("16",dir16);
	return 1;
}

