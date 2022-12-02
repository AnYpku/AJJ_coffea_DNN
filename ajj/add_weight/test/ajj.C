#define ajj_cxx
#include "ajj.h"
#include <TH2.h>
#include <TStyle.h>
#include <TCanvas.h>
#include <iostream>
#include "TLorentzVector.h"
#include "TRandom.h"
#define Pi 3.1415926

void ajj::Loop(TString name,float nevents,float xs,TString year)
{
// METHOD2: replace line
//    fChain->GetEntry(jentry);       //read all branches
//by  b_branchname->GetEntry(ientry); //read only this branch
   if (fChain == 0) return;

   Long64_t nentries = fChain->GetEntriesFast();

   Long64_t nbytes = 0, nb = 0;
   TLorentzVector Zp4, photonp4, jet1p4, jet2p4,lep1p4,lep2p4;
   Bool_t BSL=0;

   for (Long64_t jentry=0; jentry<nentries;jentry++) {
      Long64_t ientry = LoadTree(jentry);
      if (ientry < 0) break;
      nb = fChain->GetEntry(jentry);   nbytes += nb;
      // if (Cut(ientry) < 0) continue;
      set_init();

      if(name.Contains("EGamma") || name.Contains("Muon"))
              var["scalef"]=1;
      else
              var["scalef"]=1000.*xs/float(nevents)*Generator_weight/fabs(Generator_weight);

      if(jentry%10000==0)
              cout<<jentry<<" "<<nentries<<", scalef "<<var["scalef"]<<endl;

      var["channel"]=1;
      var["Vpt"]=vjj_v_pt;var["Veta"]=vjj_v_eta;
      var["Vphi"]=vjj_v_phi;var["Vmass"]=vjj_v_m;
      var["jet1pt"]=vjj_lead_pt;var["jet1eta"]=vjj_lead_eta;var["jet1phi"]=vjj_lead_phi;
      var["jet2pt"]=vjj_sublead_pt;var["jet2eta"]=vjj_sublead_eta;var["jet2phi"]=vjj_sublead_phi;
      var["jet1mass"]=vjj_lead_m; var["jet2mass"]=vjj_sublead_m;
      var["mjj"]=vjj_jj_m;
      var["detajj"] = fabs(var["jet1eta"]-var["jet2eta"]);
      var["drjj"]=delta_R(var["jet1eta"],var["jet1phi"],var["jet2eta"],var["jet2phi"]);

      ExTree->Fill(); 

   }
}
Float_t ajj::delta_R(Float_t eta1, Float_t phi1, Float_t eta2, Float_t phi2)
{
        Float_t dp = phi1-phi2;
        if(std::fabs(dp) > Pi) dp = 2*Pi - std::fabs(dp);
        Float_t dr = std::sqrt((eta1-eta2)*(eta1-eta2)+dp*dp);
        return dr;
}
void ajj::set_init()
{
        for(int i=0;i<treename.size();i++){
            var[treename[i]]=-10;
        }
}
  
