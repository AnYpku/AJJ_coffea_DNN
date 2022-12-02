#include "chain_ajj.C"
#include <iostream>
#include <fstream>
#include <TChainElement.h>
using namespace std;

void runxx(TString year){
        ifstream file("file"+year);
	float xs;
        if(!file.is_open()){cout<<"can not open the file, break the code."<<endl;abort();}
        while(!file.eof()){ 
		TString dir;TString outname;
		file>>dir>>outname>>xs;
		if(dir.Contains("end")) break;
		cout<<dir<<"*.root "<<"-> ./"<<outname<<" "<<xs<<endl;
		TTree *tree1;
		TChain*chain = new TChain("Events","");
		chain->Add(dir+"*.root");
		cout<<"add root"<<endl;
		tree1 = chain;
		cout<<"get tree; "<<tree1->GetEntries()<<endl;

		float ntot=0;int i=0;float nevents=0;

		TChain*hist; TObjArray *fileElements;
                if( outname.Contains("EGamma")==0 && outname.Contains("Muon")==0 ){
			hist=new TChain("nEventsGenWeighted","");
			hist->Add(dir+"*.root");
			fileElements=hist->GetListOfFiles();
			TIter next(fileElements);
			TChainElement *chEl=0;
		
			while ( ( chEl=(TChainElement*)next() ) ) {
				TFile f=(chEl->GetTitle());
				TH1D*h1=(TH1D*)f.Get("nEventsGenWeighted");
				ntot=ntot+h1->GetSum();
				i++;
			}
			nevents = ntot;
		}
		else if(outname.Contains("EGamma")||outname.Contains("Muon")){
			nevents =1; xs=1;
		}
		chain_ajj m1(tree1,outname);
		cout<<outname<<" "<<nevents<<" "<<xs<<endl;
		m1.Loop(outname,nevents,xs,year);
		m1.endJob();
	}
}
int main(){
	runxx("18");
	return 1;
}
