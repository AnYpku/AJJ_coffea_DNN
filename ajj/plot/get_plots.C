#include "draw_style.C"
TH1D*get_hist(TString var, int nbins,float low, float high,TString cut,TString filename,TString year){
        TFile*file;TTree*tree;TChain*chain;
	TString weight;
	if(filename.Contains("EGamma")||filename.Contains("Muon")){
		chain=new TChain("Events","");
		chain->Add("/eos/cms/store/user/yian/cutjet-out"+filename+"*_"+year+".root");
		cout<<filename<<": number of files added in TChain "<<chain->GetNtrees()<<endl;
		tree=chain;
		weight="*scalef";
	}
	else{
		file=new TFile("/eos/cms/store/user/yian/cutjet-out"+filename+year+".root");
		tree=(TTree*)file->Get("Events");
		if(year.Contains("18"))
			weight="*scalef*puWeight";
		else
			weight="*scalef*puWeight*L1PreFiringWeight_Nom";
	}
	TString histname="hist_"+filename;
        TH1D*h1=new TH1D(histname,var,nbins,low,high);
	tree->Draw(var+">>"+histname,"("+cut+")"+weight,"goff");
        h1->SetBinContent(nbins,h1->GetBinContent(nbins)+h1->GetBinContent(nbins+1));
        h1->SetBinError(nbins,sqrt( pow(h1->GetBinError(nbins),2)+pow(h1->GetBinError(nbins+1),2) ) );
	cout<<filename<<" number of entries "<<tree->GetEntries()<<", number of yields "<<h1->GetSumOfWeights()<<endl;
        return h1;
}
void run(TString data,TString year,TString var,TString title,float nbins,float low,float high){
    vector<TString> filename={/*"ST","DiPhoton40ToInf,"TTJets"","WWinc","WZinc","ZZinc","EWKZ2Jets",*/"DYJets_NLO",data};
    vector<TString> fname={/*"t(#bar{t})W","#gamma#gamma","t#bar{t}","WW","WZ","ZZ","EWK Zjj",*/"DY NLO",data};
    const int nfile=filename.size();
    TH1D*histMC[nfile];TH1D*hdata;THStack*hs=new THStack("hs","");
    TString HLT; TString high_cut,low_cut;
    if(data.Contains("EGamma")){
	    HLT="HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL ==1";
	    if(year.Contains("16")){
//		    high_cut= "(vjj_isGood==1 && vjj_jj_m>500 && vjj_v_pt>200 &&  jet1pt>30 && jet2pt>30 && abs(Vmass-91)<15 )";
		    low_cut= "((nlepton>1 && abs(lepton_pdgId[0])==11 &&  abs(lepton_pdgId[1])==11 && lepton_charge[0]*lepton_charge[1]<0 && lepton_is_real[0]==1 && lepton_is_real[1]==1 && lep1pt>30 && lep2pt>25 && abs(lep1eta)<2.4 && abs(lep2eta)<2.4 && lepton_is_tight[0]==1 && lepton_is_tight[1]==1 && Vpt>75 && abs(Vmass-91)<15 && drj1l1>0.4 && drj2l1>0.4 && drj1l2>0.4 && drj2l2>0.4 && " + HLT +" && jet1pt>30 && jet2pt>30 && abs(jet1eta)<4.7 && abs(jet2eta)<4.7 && mjj>500 && Vpt>75 && abs(Vmass-91)<15 ))";
	    }
	    else{
//		    high_cut= "(vjj_isGood==1 && vjj_jj_m>500 && vjj_v_pt>200 && jet1pt>30 && jet2pt>30 && abs(Vmass-91)<15 )";
		    low_cut= "((nlepton>1 && abs(lepton_pdgId[0])==11 &&  abs(lepton_pdgId[1])==11 && lepton_charge[0]*lepton_charge[1]<0 && lepton_is_real[0]==1 && lepton_is_real[1]==1 && lep1pt>30 && lep2pt>25 && abs(lep1eta)<2.4 && abs(lep2eta)<2.4 && lepton_is_tight[0]==1 && lepton_is_tight[1]==1 && Vpt>75 && abs(Vmass-91)<15 && drj1l1>0.4 && drj2l1>0.4 && drj1l2>0.4 && drj2l2>0.4 && " + HLT +" && jet1pt>30 && jet2pt>30 && abs(jet1eta)<4.7 && abs(jet2eta)<4.7 && mjj>500 && Vpt>75 && abs(Vmass-91)<15 ))";
	    }
    }
    else if(data.Contains("Muon")){
//	    HLT="HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass3p8==1";
	    HLT="HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ==1";
	    if(year.Contains("16")){
//		    high_cut = "(vjj_isGood==1 && vjj_jj_m>500 && vjj_v_pt>175 && jet1pt>30 && jet2pt>30 && abs(Vmass-91)<15 )";
		    low_cut = "((nlepton>1 && abs(lepton_pdgId[0])==13 &&  abs(lepton_pdgId[1])==13 && lepton_charge[0]*lepton_charge[1]<0 && lep1pt>30 && lep2pt>25 && abs(lep1eta)<2.4 && abs(lep2eta)<2.4 && lepton_is_tight[0]==1 && lepton_is_tight[1]==1 && Vpt>75 && abs(Vmass-91)<15 && drj1l1>0.4 && drj2l1>0.4 && drj1l2>0.4 && drj2l2>0.4 && " + HLT +" && jet1pt>30 && jet2pt>30 && abs(jet1eta)<4.7 && abs(jet2eta)<4.7 && mjj>500 && Vpt>75 && abs(Vmass-91)<15 ))";
	    }
	    else{
//		    high_cut = "(vjj_isGood==1 && vjj_jj_m>500 && vjj_v_pt>200 && jet1pt>30 && jet2pt>30 && abs(Vmass-91)<15 )";
		    low_cut = "(nlepton>1 && abs(lepton_pdgId[0])==13 &&  abs(lepton_pdgId[1])==13 && lepton_charge[0]*lepton_charge[1]<0 && lepton_is_real[0]==1 && lepton_is_real[1]==1 && lep1pt>30 && lep2pt>25 && abs(lep1eta)<2.4 && abs(lep2eta)<2.4 && Vpt>75 && abs(Vmass-91)<15 && drj1l1>0.4 && drj2l1>0.4 && drj1l2>0.4 && drj2l2>0.4 && HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ==1 && jet1pt>30 && jet2pt>30 && abs(jet1eta)<4.7 && abs(jet2eta)<4.7 && mjj>500 && Vpt>200)";
	    }
    }
    cout<<data<<" "<<var<<", the selection is "<<low_cut<<endl;
    double lumi;
    if(year=="18") lumi=59.7;
    if(year=="17") lumi=41.5;
    if(year=="16pre") lumi=19.52;
    if(year=="16post") lumi=16.81;
    TFile*fout=new TFile("fout_"+data+"_"+year+".root","recreate");
    for(int i=0;i<nfile;i++){
	    if(filename[i].Contains("EGamma")==0 && filename[i].Contains("Muon")==0){
		    histMC[i]=get_hist(var,nbins,low,high,low_cut,filename[i],year);
                    histMC[i]->SetLineColor(i+2);
                    histMC[i]->SetFillColor(i+2);
                    histMC[i]->SetMarkerColor(i+2);
                    histMC[i]->SetMarkerSize(0);
		    histMC[i]->Scale(lumi);
//		    if(filename[i].Contains("DY") && filename[i].Contains("NLO"))
//			    histMC[i]->Scale(5765.4/6077.22);
//		    if(filename[i].Contains("DY") && filename[i].Contains("NLO")==0)
//			    histMC[i]->Scale(6077.22/5321.);
		    cout<<filename[i]<<" "<<histMC[i]->GetSumOfWeights()<<endl;
		    hs->Add(histMC[i]);
		    fout->cd();
		    histMC[i]->Write();
	    }
	    else    hdata=get_hist(var,nbins,low,high,low_cut,filename[i],year);
    }
    fout->cd();
    hdata->Write();
    cout<<"Data: "<<hdata->GetSumOfWeights()<<endl;
    hs->Write();
    draw(hdata,hs,filename,fname,title,data,year,Form("%.2f",lumi));
    fout->Close();
}
int get_plots(){
	vector<TString> title={"p_{T}^{Z}","m_{Z}","#eta^{Z}","#phi^{Z}","p_{T}^{j1}","#eta^{j1}","#phi^{j1}","p_{T}^{j2}","#eta^{j2}","#phi^{j2}","m_{jj}","#Delta#eta_{jj}","#Delta R_{jj}"};
	vector<TString> var={"Vpt","Vmass","Veta","Vphi","jet1pt","jet1eta","jet1phi","jet2pt","jet2eta","jet2phi","mjj","detajj","drjj"};
	vector<float> high={300,110,2.5, 5,800, 5,5 ,500,5,5 ,2000,10,10};
	vector<float> low ={75, 75,-2.5,-5, 30,-5,-5,30,-5,-5,500 ,0 ,0 };
	vector<float> nbins={40,40,30,30,30,30,30,30,30,30,30,30,30};
	vector<TString> data={"Muon","EGamma"};
	for(int i=0;i<data.size()-1;i++){
//		for(int k=0;k<4;k++){
		for(int k=0;k<var.size();k++){
//			if(var[k].Contains("npvs")==0) continue;
//			run(data[i],"16pre",var[k],title[k],nbins[k],low[k],high[k]);
//			run(data[i],"16post",var[k],title[k],nbins[k],low[k],high[k]);
			run(data[i],"18",var[k],title[k],nbins[k],low[k],high[k]);
//			run(data[i],"17",var[k],title[k],nbins[k],low[k],high[k]);
		}
	}

	return 1;
}
