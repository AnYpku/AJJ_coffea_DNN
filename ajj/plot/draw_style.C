#include "CMS_lumi.C"
void draw(TH1D*hdata,THStack* hs,vector<TString> filename,vector<TString> fname,TString title,TString data, TString year,string lumiValue){
	gStyle->SetOptStat(0);
	TCanvas* cv = new TCanvas("cv","",600,600);
	TPad*fPads1 = new TPad("pad1", "", 0.00, 0.20, 0.99, 0.99);
	TPad*fPads2 = new TPad("pad2", "", 0.00, 0.00, 0.99, 0.20);
	fPads1->SetFillColor(0);
	fPads1->SetLineColor(0);
	fPads2->SetFillColor(0);
	fPads2->SetLineColor(0);
	fPads1->SetBottomMargin(0.012);
	fPads2->SetTopMargin(0.005);
	fPads2->SetBottomMargin(0.5);
	fPads1->Draw();
	fPads2->Draw();

	fPads1->cd();
        fPads1->SetLogy(false);
        double max=hs->GetMaximum();
        hs->SetMinimum(0.1);
        hs->SetMaximum(max*5);
	hs->Draw("HIST ");
	cout<<"draw mc"<<endl;
        hs->GetYaxis()->SetLabelSize(0.035); 
        hs->GetXaxis()->SetLabelSize(0.00); 
        hs->GetYaxis()->SetTitleOffset(1.4);
	hs->GetYaxis()->SetTitleSize(0.04);
        hs->GetYaxis()->SetTitle("Events/bin");

	TLegend* leg1 = new TLegend(0.52, 0.5, 0.82, 0.89);
        TLegend* leg2 = new TLegend(0.15, 0.5, 0.45, 0.89);
        leg1->SetTextSize(0.035);
	leg1->SetFillColor(kWhite);
	leg1->SetLineColor(kWhite);
	leg2->SetFillColor(kWhite);
	leg2->SetLineColor(kWhite);
        leg2->SetTextSize(0.035);
	const int nhist=hs->GetNhists();
	for(int i=0;i<hs->GetNhists();i++){
		TH1D*temp= (TH1D*)hs->GetStack()->FindObject("hist_"+filename[i]);
                TString Label = fname[i];
                if(i<hs->GetNhists()/2)leg1->AddEntry(temp,Label);
		else leg2->AddEntry(temp,Label);
	}       
	double yieldsDataErr=0;
	double yieldsData = hdata->IntegralAndError(0,hdata->GetNbinsX(),yieldsDataErr);
	char yData[100];sprintf(yData,"%0.f",yieldsData);
	char yDataErr[100];sprintf(yDataErr,"%0.f",yieldsDataErr);
	TString samplesData = "Data";
	TString LabelData = samplesData +" ["+ yData+"#pm"+yDataErr+"]";

        TH1D* htemp = (TH1D*)(hs->GetStack()->Last());
        htemp->SetBinErrorOption(TH1::kPoisson);
        const int binsize = htemp->GetNbinsX();

        TLatex latex;
        latex.SetTextSize(0.1);
        latex.SetTextFont(42);
        double poi_X=htemp->GetBinLowEdge(htemp->GetNbinsX()/3*2);
        TString channel;TString data_type=hdata->GetName();
        if(data_type.Contains("Muon"))  channel="#mu#mu";
        else if(data_type.Contains("Ele") || data_type.Contains("EGamma"))
                channel="#font[132]{ee}";
        latex.DrawLatex(poi_X,max*5*0.5,channel);

        Double_t x[binsize];
        Double_t y[binsize];
        Double_t xerror_l[binsize];
        Double_t xerror_r[binsize];
        Double_t yerror_u[binsize];
        Double_t yerror_d[binsize];
        for(int i=0;i<binsize;i++)
        {       
                x[i]=htemp->GetBinCenter(i+1);
                y[i]=htemp->GetBinContent(i+1);
                xerror_l[i]=0.5*htemp->GetBinWidth(i+1);
                xerror_r[i]=0.5*htemp->GetBinWidth(i+1);
                yerror_u[i]=htemp->GetBinErrorUp(i+1);
                yerror_d[i]=htemp->GetBinErrorLow(i+1);
                if(htemp->GetBinContent(i+1)==0)
                {yerror_u[i]=0.;yerror_d[i]=0.;}
        }
        TGraphAsymmErrors* gr = new TGraphAsymmErrors(binsize, x, y, xerror_l,xerror_r, yerror_d, yerror_u);
        gr->SetFillColor(1);
        gr->SetFillStyle(3005);
        gr->Draw("SAME 2");// MC stat unc.
	cout<<"draw mc stat"<<endl;

	cout<<"test data hist "<<hdata->GetEntries()<<endl;
	TGraphAsymmErrors * g = new TGraphAsymmErrors(hdata);
	for (int i = 0; i < g->GetN(); ++i) {
		double x=hdata->GetXaxis()->GetBinCenter(i+1);
		double y=hdata->GetBinContent(i+1);
		double ey=hdata->GetBinError(i+1);
		g->SetPoint(i,x,y);
		g->SetPointEXlow(i, 0);
		g->SetPointEXhigh(i, 0);
		g->SetPointEYlow(i, ey);
		g->SetPointEYhigh(i, ey);
	}
	g->SetFillColor(kBlack);
	g->SetLineColor(1);
	g->SetMarkerStyle(20);
	g->SetMarkerSize(1.0);
	g->Draw("SAME E P"); // data 
	cout<<"draw data"<<endl;
	hdata->SetMarkerStyle(20);

	double yieldsMCErr=0;
	double yieldsMC = htemp->IntegralAndError(0,htemp->GetNbinsX(),yieldsMCErr);
	char yMC[100];sprintf(yMC,"%0.f",yieldsMC);
	char yMCErr[100];sprintf(yMCErr,"%0.f",yieldsMCErr);
	TString samplesMC = "Pred.";
	TString LabelMC = samplesMC +" ["+ yMC+"#pm"+yMCErr+"]";

        leg1->AddEntry(hdata,LabelData,"ep");
        leg2->AddEntry(gr,LabelMC);
	leg1->Draw();
	leg2->Draw();

        fPads1->Update();

	fPads2->cd();
	TH1D*nominal=(TH1D*)htemp->Clone("nominal");
	TH1D*nomNoErr=(TH1D*)nominal->Clone("nomNoErr");
	nominal->SetMarkerSize(0);
	nominal->SetFillColor(1);
	nominal->SetLineColor(1);
	nominal->SetFillStyle(3005);
	for (int i = 1; i<= nomNoErr->GetNbinsX(); ++i){nomNoErr->SetBinError(i,0);}
	nominal->Divide(nomNoErr);
	nominal->SetTitle("");
	nominal->Draw("E2");

	const int nbins=hdata->GetNbinsX();
	float nsigma_x[nbins],nsigma_y[nbins],nsigma_ex[nbins],nsigma_ey[nbins];
	for (int ibin=0; ibin!= hdata->GetNbinsX();ibin++) {
		double Data = hdata->GetBinContent(ibin + 1);
		double Bkg = htemp->GetBinContent(ibin + 1);
		double eData = hdata->GetBinError(ibin + 1);
		double eBkg = htemp->GetBinError(ibin + 1);
		double x = hdata->GetBinCenter(ibin + 1);
		nsigma_x[ibin] = x;
		nsigma_ex[ibin] = hdata->GetBinWidth(ibin+1)/2.;
		if (Data !=0 && Bkg !=0 ){
			nsigma_y[ibin] = Data / Bkg;
		        nsigma_ey[ibin] = sqrt((eData*eData/(Bkg*Bkg)) + (Data*Data*eBkg*eBkg/(Bkg*Bkg*Bkg*Bkg)));
		}
		else{
			nsigma_y[ibin] = 0;
			nsigma_ey[ibin] = 0;
		}
	}
	TGraph *nsigmaGraph = new TGraphErrors(nbins,nsigma_x, nsigma_y,nsigma_ex,nsigma_ey);
	nominal->SetTitle("");
        double thisYmin = 0.5;double thisYmax = 1.6;
	nominal->GetYaxis()->SetRangeUser(thisYmin, thisYmax);
	nominal->GetYaxis()->SetNdivisions(404);
	nominal->GetYaxis()->SetTitle("Data/Pred.");
	nominal->GetYaxis()->SetTitleOffset(0.3);
	nominal->GetXaxis()->SetLabelOffset(0.04);
	nominal->GetYaxis()->SetTitleSize(0.14);
	nominal->GetYaxis()->SetLabelSize(0.15);
	nominal->GetXaxis()->SetLabelSize(0.15);
	nominal->GetXaxis()->SetTitleSize(0.20);
	nominal->GetXaxis()->SetTitle(title+" [GeV]");
	nominal->GetXaxis()->SetLimits(htemp->GetXaxis()->GetXmin(),htemp->GetXaxis()->GetXmax());
	nominal->GetXaxis()->SetRangeUser(htemp->GetXaxis()->GetXmin(),htemp->GetXaxis()->GetXmax());
	nominal->GetXaxis()->SetTitleOffset(0.9);
	nsigmaGraph->SetMarkerStyle(20);
	nsigmaGraph->SetMarkerSize(1.0);
	nsigmaGraph->SetMarkerColor(kBlack);
	nsigmaGraph->Draw("same pe");

	TLine* line = new TLine(htemp->GetXaxis()->GetXmin(), 1,htemp->GetXaxis()->GetXmax(), 1);
	line->SetLineColor(2);
	line->SetLineStyle(2);
	line->Draw();
	fPads2->Update();
	CMS_lumi(fPads1, 4, 0, lumiValue);
        TString name=htemp->GetTitle();
	cv->SaveAs("./"+name+"_"+data+year+".pdf");

	fPads1->cd();
	fPads1->SetLogy(true);
	hs->SetMaximum(max*10*10000);
        cv->SetLogy(true);
        cv->Update();
        cv->Modified(true);
        cv->Draw();
	cv->SaveAs("./LOG_"+name+"_"+data+year+".pdf");
}
