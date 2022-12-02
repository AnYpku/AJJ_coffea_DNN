#define chain_ajj_cxx
#include <TH2.h>
#include <TStyle.h>
#include <TCanvas.h>
#include <iostream>
#include "chain_ajj.h"
#include "TLorentzVector.h"
#include "TRandom.h"
#define Pi 3.1415926

void chain_ajj::Loop(TString name,float nevents,float xs,TString year)
{
//       To read only selected branches, Insert statements like:
// METHOD1:
//    fChain->SetBranchStatus("*",0);  // disable all branches
//    fChain->SetBranchStatus("branchname",1);  // activate branchname
// METHOD2: replace line
//    fChain->GetEntry(jentry);       //read all branches
//by  b_branchname->GetEntry(ientry); //read only this branch
   if (fChain == 0) return;

   Long64_t nentries = fChain->GetEntriesFast();
   Long64_t nbytes = 0, nb = 0;

   TLorentzVector Zp4, photonp4, jet1p4, jet2p4,lep1p4,lep2p4;
   Bool_t BSL=0;int count=0;

//   nentries=10000;
   for (Long64_t jentry=0; jentry<nentries;jentry++) {
      Long64_t ientry = LoadTree(jentry);
      if (ientry < 0) break;
      nb = fChain->GetEntry(jentry);   nbytes += nb;
      // if (Cut(ientry) < 0) continue;
      set_init();

      if(name.Contains("Muon") || name.Contains("EGamma"))
	      var["scalef"]=1;
      else
	      var["scalef"]=1000.*xs/float(nevents)*Generator_weight/fabs(Generator_weight);

      if(nlepton==1){
              var["channel"]=1;var["lep1pt"]=lepton_corrected_pt[0];var["lep1eta"]=lepton_eta[0];
	      var["lep1phi"]=lepton_phi[0];var["lep1mass"]=lepton_mass[0];
	      lep1p4.SetPtEtaPhiM(var["lep1pt"],var["lep1eta"],var["lep1phi"],var["lep1mass"]);
      }
      else if(nlepton>1){
              var["channel"]=2;
	      var["lep1pt"]=lepton_corrected_pt[0];var["lep1eta"]=lepton_eta[0];var["lep1phi"]=lepton_phi[0];
	      var["lep2pt"]=lepton_corrected_pt[1];var["lep2eta"]=lepton_eta[1];var["lep2phi"]=lepton_phi[1];
	      var["lep1mass"]=lepton_mass[0];var["lep2mass"]=lepton_mass[1];
	      lep1p4.SetPtEtaPhiM(var["lep1pt"],var["lep1eta"],var["lep1phi"],var["lep1mass"]);
	      lep2p4.SetPtEtaPhiM(var["lep2pt"],var["lep2eta"],var["lep2phi"],var["lep2mass"]);
              var["Vpt"]=(lep1p4+lep2p4).Pt();var["Veta"]=(lep1p4+lep2p4).Eta();
              var["Vphi"]=(lep1p4+lep2p4).Phi();var["Vmass"]=(lep1p4+lep2p4).M();
      }
      else
	      var["channel"]=0;
      if(njets25>1){
              int jet1_index=0;float Firstpt=0;
              for(int i=0;i<njet;i++){
                      if(jet_pt_nom[i]>Firstpt) {
                              Firstpt=jet_pt_nom[i];
                              jet1_index=i;
                      }
              }
              int jet2_index=0;float Secondpt=0;
              for(int i=0;i<njet;i++){
                      if(i==jet1_index) continue;
                      if(jet_pt_nom[i]>Secondpt) {
                              Secondpt=jet_pt_nom[i];
                              jet2_index=i;
                      }
              }
              var["jet1pt"]=jet_pt_nom[jet1_index];var["jet1eta"]=jet_eta[jet1_index];var["jet1phi"]=jet_phi[jet1_index];
              var["jet2pt"]=jet_pt_nom[jet2_index];var["jet2eta"]=jet_eta[jet2_index];var["jet2phi"]=jet_phi[jet2_index];
              var["jet1mass"]=jet_mass_nom[jet1_index]; var["jet2mass"]=jet_mass_nom[jet2_index];
	      jet1p4.SetPtEtaPhiM(var["jet1pt"],var["jet1eta"],var["jet1phi"],var["jet1mass"]);
	      jet2p4.SetPtEtaPhiM(var["jet2pt"],var["jet2eta"],var["jet2phi"],var["jet2mass"]);
              var["mjj_nom"]=(jet1p4+jet2p4).M();
	      var["detajj"] = fabs(var["jet1eta"]-var["jet2eta"]);
              var["drjj"]=delta_R(var["jet1eta"],var["jet1phi"],var["jet2eta"],var["jet2phi"]);
	      if(nlepton==1){ 
		      var["drj1l1"]=delta_R(var["lep1eta"],var["lep1phi"],var["jet1eta"],var["jet1phi"]);
		      var["drj2l1"]=delta_R(var["lep1eta"],var["lep1phi"],var["jet2eta"],var["jet2phi"]);
	      }
	      if(nlepton>1){ 
		      var["drj1l1"]=delta_R(var["lep1eta"],var["lep1phi"],var["jet1eta"],var["jet1phi"]);
		      var["drj2l1"]=delta_R(var["lep1eta"],var["lep1phi"],var["jet2eta"],var["jet2phi"]);
		      var["drj1l2"]=delta_R(var["lep2eta"],var["lep2phi"],var["jet1eta"],var["jet1phi"]);
		      var["drj2l2"]=delta_R(var["lep2eta"],var["lep2phi"],var["jet2eta"],var["jet2phi"]);
	      }
      }
      if(nphoton>0){
	      var["photonet"]=photon_pt[0];var["photoneta"]=photon_eta[0];
	      var["photonphi"]=photon_phi[0];var["photonmass"]=photon_mass[0];
	      var["photonMCmatch"]=photon_genPartIdx[0];
	      photonp4.SetPtEtaPhiM(var["photonet"],var["photoneta"],var["photonphi"],var["photonmass"]);
	      var["zepp"]=fabs( photonp4.Rapidity() - (jet1p4.Rapidity()+jet2p4.Rapidity())/2 );
	      var["dphi"]=fabs( var["photonphi"]-(jet1p4+jet2p4).Phi() );
	      var["drj1a"]=delta_R(var["jet1eta"],var["jet1phi"],var["photoneta"],var["photonphi"]);
	      var["drj2a"]=delta_R(var["jet2eta"],var["jet2phi"],var["photoneta"],var["photonphi"]);
	      if(nlepton==1){
		      var["drl1a"]=delta_R(var["lep1eta"],var["lep1phi"],var["photoneta"],var["photonphi"]);
		      var["ml1g"]=(lep1p4+photonp4).M();
	      }
	      if(nlepton>1){
		      var["drl1a"]=delta_R(var["lep1eta"],var["lep1phi"],var["photoneta"],var["photonphi"]);
		      var["drl2a"]=delta_R(var["lep2eta"],var["lep2phi"],var["photoneta"],var["photonphi"]);
		      var["ml1g"]=(lep1p4+photonp4).M();var["ml2g"]=(lep2p4+photonp4).M();
		      var["mllg"]=(lep1p4+lep2p4+photonp4).M();
	      }
      }
 
      if(jentry%10000==0) cout<<name<<" "<<jentry<<" "<<nentries<<" "<<mjj<<" "<<var["mjj_nom"]<<endl;  

      BSL = var["jet1pt"]>30 && var["jet2pt"]>30 && fabs(var["jet1eta"])<4.7 && fabs(var["jet2eta"])<4.7 && var["mjj_nom"]>500; 
      if(nlepton==1)
	      BSL = BSL && (var["lep1pt"]>20 && fabs(var["lep1eta"])<2.5);
      if(nlepton>1)
	      BSL = BSL && (var["lep1pt"]>20  &&  fabs(var["lep1eta"])<2.5 && var["lep2pt"]>20 && fabs(var["lep2eta"])<2.5 && var["Vpt"]>75);
      if(nphoton>0)
	      BSL = BSL && (var["photonet"]>75);
      if(!BSL)
	      continue;
      ExTree->Fill();
      count++;
   }
   cout<<name<<": after the BSL cut, there are "<<count<<" events left"<<endl;
}
Float_t chain_ajj::delta_R(Float_t eta1, Float_t phi1, Float_t eta2, Float_t phi2)
{       
        Float_t dp = phi1-phi2;
        if(std::fabs(dp) > Pi) dp = 2*Pi - std::fabs(dp);
        Float_t dr = std::sqrt((eta1-eta2)*(eta1-eta2)+dp*dp);
        return dr;
}
void chain_ajj::set_init()
{
	for(int i=0;i<treename.size();i++){
            var[treename[i]]=-10;
	}
}
