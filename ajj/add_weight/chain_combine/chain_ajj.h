//////////////////////////////////////////////////////////
// This class has been automatically generated on
// Mon Sep 26 18:50:53 2022 by ROOT version 6.14/09
// from TChain Events/
//////////////////////////////////////////////////////////

#ifndef chain_ajj_h
#define chain_ajj_h

#include <TROOT.h>
#include <TChain.h>
#include <TFile.h>
#include <vector>
using namespace std;

// Header file for the classes stored in the TTree if any.

class chain_ajj {
public :
   TTree          *fChain;   //!pointer to the analyzed TTree or TChain
   Int_t           fCurrent; //!current Tree number in a TChain

// Fixed size dimensions of array or collections stored in the TTree if any.

   // Declaration of leaf types
   UInt_t          run;
   UInt_t          luminosityBlock;
   ULong64_t       event;
   Float_t         btagWeight_CSVV2;
   Float_t         btagWeight_DeepCSVB;
   UInt_t          nCorrT1METJet;
   Float_t         CorrT1METJet_area[28];   //[nCorrT1METJet]
   Float_t         CorrT1METJet_eta[28];   //[nCorrT1METJet]
   Float_t         CorrT1METJet_muonSubtrFactor[28];   //[nCorrT1METJet]
   Float_t         CorrT1METJet_phi[28];   //[nCorrT1METJet]
   Float_t         CorrT1METJet_rawPt[28];   //[nCorrT1METJet]
   UInt_t          nElectron;
   Float_t         Electron_dEscaleDown[7];   //[nElectron]
   Float_t         Electron_dEscaleUp[7];   //[nElectron]
   Float_t         Electron_dEsigmaDown[7];   //[nElectron]
   Float_t         Electron_dEsigmaUp[7];   //[nElectron]
   Float_t         Electron_deltaEtaSC[7];   //[nElectron]
   Float_t         Electron_dr03EcalRecHitSumEt[7];   //[nElectron]
   Float_t         Electron_dr03HcalDepth1TowerSumEt[7];   //[nElectron]
   Float_t         Electron_dr03TkSumPt[7];   //[nElectron]
   Float_t         Electron_dr03TkSumPtHEEP[7];   //[nElectron]
   Float_t         Electron_dxy[7];   //[nElectron]
   Float_t         Electron_dxyErr[7];   //[nElectron]
   Float_t         Electron_dz[7];   //[nElectron]
   Float_t         Electron_dzErr[7];   //[nElectron]
   Float_t         Electron_eCorr[7];   //[nElectron]
   Float_t         Electron_eInvMinusPInv[7];   //[nElectron]
   Float_t         Electron_energyErr[7];   //[nElectron]
   Float_t         Electron_eta[7];   //[nElectron]
   Float_t         Electron_hoe[7];   //[nElectron]
   Float_t         Electron_ip3d[7];   //[nElectron]
   Float_t         Electron_jetPtRelv2[7];   //[nElectron]
   Float_t         Electron_jetRelIso[7];   //[nElectron]
   Float_t         Electron_mass[7];   //[nElectron]
   Float_t         Electron_miniPFRelIso_all[7];   //[nElectron]
   Float_t         Electron_miniPFRelIso_chg[7];   //[nElectron]
   Float_t         Electron_mvaFall17V2Iso[7];   //[nElectron]
   Float_t         Electron_mvaFall17V2noIso[7];   //[nElectron]
   Float_t         Electron_pfRelIso03_all[7];   //[nElectron]
   Float_t         Electron_pfRelIso03_chg[7];   //[nElectron]
   Float_t         Electron_phi[7];   //[nElectron]
   Float_t         Electron_pt[7];   //[nElectron]
   Float_t         Electron_r9[7];   //[nElectron]
   Float_t         Electron_scEtOverPt[7];   //[nElectron]
   Float_t         Electron_sieie[7];   //[nElectron]
   Float_t         Electron_sip3d[7];   //[nElectron]
   Float_t         Electron_mvaTTH[7];   //[nElectron]
   Int_t           Electron_charge[7];   //[nElectron]
   Int_t           Electron_cutBased[7];   //[nElectron]
   Int_t           Electron_jetIdx[7];   //[nElectron]
   Int_t           Electron_pdgId[7];   //[nElectron]
   Int_t           Electron_photonIdx[7];   //[nElectron]
   Int_t           Electron_tightCharge[7];   //[nElectron]
   Int_t           Electron_vidNestedWPBitmap[7];   //[nElectron]
   Int_t           Electron_vidNestedWPBitmapHEEP[7];   //[nElectron]
   Bool_t          Electron_convVeto[7];   //[nElectron]
   Bool_t          Electron_cutBased_HEEP[7];   //[nElectron]
   Bool_t          Electron_isPFcand[7];   //[nElectron]
   UChar_t         Electron_jetNDauCharged[7];   //[nElectron]
   UChar_t         Electron_lostHits[7];   //[nElectron]
   Bool_t          Electron_mvaFall17V2Iso_WP80[7];   //[nElectron]
   Bool_t          Electron_mvaFall17V2Iso_WP90[7];   //[nElectron]
   Bool_t          Electron_mvaFall17V2Iso_WPL[7];   //[nElectron]
   Bool_t          Electron_mvaFall17V2noIso_WP80[7];   //[nElectron]
   Bool_t          Electron_mvaFall17V2noIso_WP90[7];   //[nElectron]
   Bool_t          Electron_mvaFall17V2noIso_WPL[7];   //[nElectron]
   UChar_t         Electron_seedGain[7];   //[nElectron]
   UInt_t          nFsrPhoton;
   UInt_t          nGenJet;
   Float_t         GenJet_eta[22];   //[nGenJet]
   Float_t         GenJet_mass[22];   //[nGenJet]
   Float_t         GenJet_phi[22];   //[nGenJet]
   Float_t         GenJet_pt[22];   //[nGenJet]
   UInt_t          nGenPart;
   Float_t         GenPart_eta[123];   //[nGenPart]
   Float_t         GenPart_mass[123];   //[nGenPart]
   Float_t         GenPart_phi[123];   //[nGenPart]
   Float_t         GenPart_pt[123];   //[nGenPart]
   Int_t           GenPart_genPartIdxMother[123];   //[nGenPart]
   Int_t           GenPart_pdgId[123];   //[nGenPart]
   Int_t           GenPart_status[123];   //[nGenPart]
   Int_t           GenPart_statusFlags[123];   //[nGenPart]
   Float_t         Generator_binvar;
   Float_t         Generator_scalePDF;
   Float_t         Generator_weight;
   Float_t         Generator_x1;
   Float_t         Generator_x2;
   Float_t         Generator_xpdf1;
   Float_t         Generator_xpdf2;
   Int_t           Generator_id1;
   Int_t           Generator_id2;
   Float_t         GenVtx_x;
   Float_t         GenVtx_y;
   Float_t         GenVtx_z;
   UInt_t          nGenVisTau;
   Float_t         genWeight;
   Float_t         LHEWeight_originalXWGTUP;
   UInt_t          nLHEPdfWeight;
   Float_t         LHEPdfWeight[103];   //[nLHEPdfWeight]
   UInt_t          nLHEReweightingWeight;
   Float_t         LHEReweightingWeight[1];   //[nLHEReweightingWeight]
   UInt_t          nLHEScaleWeight;
   Float_t         LHEScaleWeight[9];   //[nLHEScaleWeight]
   UInt_t          nPSWeight;
   Float_t         PSWeight[4];   //[nPSWeight]
   UInt_t          nIsoTrack;
   UInt_t          nJet;
   Float_t         Jet_area[24];   //[nJet]
   Float_t         Jet_btagCSVV2[24];   //[nJet]
   Float_t         Jet_btagDeepB[24];   //[nJet]
   Float_t         Jet_btagDeepCvB[24];   //[nJet]
   Float_t         Jet_btagDeepCvL[24];   //[nJet]
   Float_t         Jet_btagDeepFlavB[24];   //[nJet]
   Float_t         Jet_btagDeepFlavCvB[24];   //[nJet]
   Float_t         Jet_btagDeepFlavCvL[24];   //[nJet]
   Float_t         Jet_btagDeepFlavQG[24];   //[nJet]
   Float_t         Jet_chEmEF[24];   //[nJet]
   Float_t         Jet_chFPV0EF[24];   //[nJet]
   Float_t         Jet_chHEF[24];   //[nJet]
   Float_t         Jet_eta[24];   //[nJet]
   Float_t         Jet_hfsigmaEtaEta[24];   //[nJet]
   Float_t         Jet_hfsigmaPhiPhi[24];   //[nJet]
   Float_t         Jet_mass[24];   //[nJet]
   Float_t         Jet_muEF[24];   //[nJet]
   Float_t         Jet_muonSubtrFactor[24];   //[nJet]
   Float_t         Jet_neEmEF[24];   //[nJet]
   Float_t         Jet_neHEF[24];   //[nJet]
   Float_t         Jet_phi[24];   //[nJet]
   Float_t         Jet_pt[24];   //[nJet]
   Float_t         Jet_puIdDisc[24];   //[nJet]
   Float_t         Jet_qgl[24];   //[nJet]
   Float_t         Jet_rawFactor[24];   //[nJet]
   Float_t         Jet_bRegCorr[24];   //[nJet]
   Float_t         Jet_bRegRes[24];   //[nJet]
   Float_t         Jet_cRegCorr[24];   //[nJet]
   Float_t         Jet_cRegRes[24];   //[nJet]
   Int_t           Jet_electronIdx1[24];   //[nJet]
   Int_t           Jet_electronIdx2[24];   //[nJet]
   Int_t           Jet_hfadjacentEtaStripsSize[24];   //[nJet]
   Int_t           Jet_hfcentralEtaStripSize[24];   //[nJet]
   Int_t           Jet_jetId[24];   //[nJet]
   Int_t           Jet_muonIdx1[24];   //[nJet]
   Int_t           Jet_muonIdx2[24];   //[nJet]
   Int_t           Jet_nElectrons[24];   //[nJet]
   Int_t           Jet_nMuons[24];   //[nJet]
   Int_t           Jet_puId[24];   //[nJet]
   UChar_t         Jet_nConstituents[24];   //[nJet]
   Float_t         L1PreFiringWeight_Dn;
   Float_t         L1PreFiringWeight_ECAL_Dn;
   Float_t         L1PreFiringWeight_ECAL_Nom;
   Float_t         L1PreFiringWeight_ECAL_Up;
   Float_t         L1PreFiringWeight_Muon_Nom;
   Float_t         L1PreFiringWeight_Muon_StatDn;
   Float_t         L1PreFiringWeight_Muon_StatUp;
   Float_t         L1PreFiringWeight_Muon_SystDn;
   Float_t         L1PreFiringWeight_Muon_SystUp;
   Float_t         L1PreFiringWeight_Nom;
   Float_t         L1PreFiringWeight_Up;
   UChar_t         LHE_Njets;
   UInt_t          nLHEPart;
   UInt_t          nLowPtElectron;
   Float_t         GenMET_phi;
   Float_t         GenMET_pt;
   Float_t         MET_MetUnclustEnUpDeltaX;
   Float_t         MET_MetUnclustEnUpDeltaY;
   Float_t         MET_covXX;
   Float_t         MET_covXY;
   Float_t         MET_covYY;
   Float_t         MET_phi;
   Float_t         MET_pt;
   Float_t         MET_significance;
   Float_t         MET_sumEt;
   Float_t         MET_sumPtUnclustered;
   UInt_t          nMuon;
   Float_t         Muon_dxy[15];   //[nMuon]
   Float_t         Muon_dxyErr[15];   //[nMuon]
   Float_t         Muon_dxybs[15];   //[nMuon]
   Float_t         Muon_dz[15];   //[nMuon]
   Float_t         Muon_dzErr[15];   //[nMuon]
   Float_t         Muon_eta[15];   //[nMuon]
   Float_t         Muon_ip3d[15];   //[nMuon]
   Float_t         Muon_jetPtRelv2[15];   //[nMuon]
   Float_t         Muon_jetRelIso[15];   //[nMuon]
   Float_t         Muon_mass[15];   //[nMuon]
   Float_t         Muon_miniPFRelIso_all[15];   //[nMuon]
   Float_t         Muon_miniPFRelIso_chg[15];   //[nMuon]
   Float_t         Muon_pfRelIso03_all[15];   //[nMuon]
   Float_t         Muon_pfRelIso03_chg[15];   //[nMuon]
   Float_t         Muon_pfRelIso04_all[15];   //[nMuon]
   Float_t         Muon_phi[15];   //[nMuon]
   Float_t         Muon_pt[15];   //[nMuon]
   Float_t         Muon_ptErr[15];   //[nMuon]
   Float_t         Muon_segmentComp[15];   //[nMuon]
   Float_t         Muon_sip3d[15];   //[nMuon]
   Float_t         Muon_softMva[15];   //[nMuon]
   Float_t         Muon_tkRelIso[15];   //[nMuon]
   Float_t         Muon_tunepRelPt[15];   //[nMuon]
   Float_t         Muon_mvaLowPt[15];   //[nMuon]
   Float_t         Muon_mvaTTH[15];   //[nMuon]
   Int_t           Muon_charge[15];   //[nMuon]
   Int_t           Muon_jetIdx[15];   //[nMuon]
   Int_t           Muon_nStations[15];   //[nMuon]
   Int_t           Muon_nTrackerLayers[15];   //[nMuon]
   Int_t           Muon_pdgId[15];   //[nMuon]
   Int_t           Muon_tightCharge[15];   //[nMuon]
   Int_t           Muon_fsrPhotonIdx[15];   //[nMuon]
   UChar_t         Muon_highPtId[15];   //[nMuon]
   Bool_t          Muon_highPurity[15];   //[nMuon]
   Bool_t          Muon_inTimeMuon[15];   //[nMuon]
   Bool_t          Muon_isGlobal[15];   //[nMuon]
   Bool_t          Muon_isPFcand[15];   //[nMuon]
   Bool_t          Muon_isStandalone[15];   //[nMuon]
   Bool_t          Muon_isTracker[15];   //[nMuon]
   UChar_t         Muon_jetNDauCharged[15];   //[nMuon]
   Bool_t          Muon_looseId[15];   //[nMuon]
   Bool_t          Muon_mediumId[15];   //[nMuon]
   Bool_t          Muon_mediumPromptId[15];   //[nMuon]
   UChar_t         Muon_miniIsoId[15];   //[nMuon]
   UChar_t         Muon_multiIsoId[15];   //[nMuon]
   UChar_t         Muon_mvaId[15];   //[nMuon]
   UChar_t         Muon_mvaLowPtId[15];   //[nMuon]
   UChar_t         Muon_pfIsoId[15];   //[nMuon]
   UChar_t         Muon_puppiIsoId[15];   //[nMuon]
   Bool_t          Muon_softId[15];   //[nMuon]
   Bool_t          Muon_softMvaId[15];   //[nMuon]
   Bool_t          Muon_tightId[15];   //[nMuon]
   UChar_t         Muon_tkIsoId[15];   //[nMuon]
   Bool_t          Muon_triggerIdLoose[15];   //[nMuon]
   UInt_t          nPhoton;
   Float_t         Photon_dEscaleDown[9];   //[nPhoton]
   Float_t         Photon_dEscaleUp[9];   //[nPhoton]
   Float_t         Photon_dEsigmaDown[9];   //[nPhoton]
   Float_t         Photon_dEsigmaUp[9];   //[nPhoton]
   Float_t         Photon_eCorr[9];   //[nPhoton]
   Float_t         Photon_energyErr[9];   //[nPhoton]
   Float_t         Photon_eta[9];   //[nPhoton]
   Float_t         Photon_hoe[9];   //[nPhoton]
   Float_t         Photon_mass[9];   //[nPhoton]
   Float_t         Photon_mvaID[9];   //[nPhoton]
   Float_t         Photon_mvaID_Fall17V1p1[9];   //[nPhoton]
   Float_t         Photon_pfRelIso03_all[9];   //[nPhoton]
   Float_t         Photon_pfRelIso03_chg[9];   //[nPhoton]
   Float_t         Photon_phi[9];   //[nPhoton]
   Float_t         Photon_pt[9];   //[nPhoton]
   Float_t         Photon_r9[9];   //[nPhoton]
   Float_t         Photon_sieie[9];   //[nPhoton]
   Int_t           Photon_charge[9];   //[nPhoton]
   Int_t           Photon_cutBased[9];   //[nPhoton]
   Int_t           Photon_cutBased_Fall17V1Bitmap[9];   //[nPhoton]
   Int_t           Photon_electronIdx[9];   //[nPhoton]
   Int_t           Photon_jetIdx[9];   //[nPhoton]
   Int_t           Photon_pdgId[9];   //[nPhoton]
   Int_t           Photon_vidNestedWPBitmap[9];   //[nPhoton]
   Bool_t          Photon_electronVeto[9];   //[nPhoton]
   Bool_t          Photon_isScEtaEB[9];   //[nPhoton]
   Bool_t          Photon_isScEtaEE[9];   //[nPhoton]
   Bool_t          Photon_mvaID_WP80[9];   //[nPhoton]
   Bool_t          Photon_mvaID_WP90[9];   //[nPhoton]
   Bool_t          Photon_pixelSeed[9];   //[nPhoton]
   UChar_t         Photon_seedGain[9];   //[nPhoton]
   Float_t         Pileup_nTrueInt;
   Float_t         Pileup_pudensity;
   Float_t         Pileup_gpudensity;
   Int_t           Pileup_nPU;
   Int_t           Pileup_sumEOOT;
   Int_t           Pileup_sumLOOT;
   Float_t         PuppiMET_phi;
   Float_t         PuppiMET_phiJERDown;
   Float_t         PuppiMET_phiJERUp;
   Float_t         PuppiMET_phiJESDown;
   Float_t         PuppiMET_phiJESUp;
   Float_t         PuppiMET_phiUnclusteredDown;
   Float_t         PuppiMET_phiUnclusteredUp;
   Float_t         PuppiMET_pt;
   Float_t         PuppiMET_ptJERDown;
   Float_t         PuppiMET_ptJERUp;
   Float_t         PuppiMET_ptJESDown;
   Float_t         PuppiMET_ptJESUp;
   Float_t         PuppiMET_ptUnclusteredDown;
   Float_t         PuppiMET_ptUnclusteredUp;
   Float_t         PuppiMET_sumEt;
   Float_t         RawMET_phi;
   Float_t         RawMET_pt;
   Float_t         RawMET_sumEt;
   Float_t         RawPuppiMET_phi;
   Float_t         RawPuppiMET_pt;
   Float_t         RawPuppiMET_sumEt;
   Float_t         fixedGridRhoFastjetAll;
   Float_t         fixedGridRhoFastjetCentral;
   Float_t         fixedGridRhoFastjetCentralCalo;
   Float_t         fixedGridRhoFastjetCentralChargedPileUp;
   Float_t         fixedGridRhoFastjetCentralNeutral;
   UInt_t          nGenDressedLepton;
   Float_t         GenDressedLepton_eta[4];   //[nGenDressedLepton]
   Float_t         GenDressedLepton_mass[4];   //[nGenDressedLepton]
   Float_t         GenDressedLepton_phi[4];   //[nGenDressedLepton]
   Float_t         GenDressedLepton_pt[4];   //[nGenDressedLepton]
   Int_t           GenDressedLepton_pdgId[4];   //[nGenDressedLepton]
   Bool_t          GenDressedLepton_hasTauAnc[4];   //[nGenDressedLepton]
   UInt_t          nGenIsolatedPhoton;
   Float_t         GenIsolatedPhoton_eta[3];   //[nGenIsolatedPhoton]
   Float_t         GenIsolatedPhoton_mass[3];   //[nGenIsolatedPhoton]
   Float_t         GenIsolatedPhoton_phi[3];   //[nGenIsolatedPhoton]
   Float_t         GenIsolatedPhoton_pt[3];   //[nGenIsolatedPhoton]
   Float_t         PV_ndof;
   Float_t         PV_x;
   Float_t         PV_y;
   Float_t         PV_z;
   Float_t         PV_chi2;
   Float_t         PV_score;
   Int_t           PV_npvs;
   Int_t           PV_npvsGood;
   Int_t           Electron_genPartIdx[7];   //[nElectron]
   UChar_t         Electron_genPartFlav[7];   //[nElectron]
   Int_t           GenJet_partonFlavour[22];   //[nGenJet]
   UChar_t         GenJet_hadronFlavour[22];   //[nGenJet]
   Float_t         GenVtx_t0;
   Int_t           Jet_genJetIdx[24];   //[nJet]
   Int_t           Jet_hadronFlavour[24];   //[nJet]
   Int_t           Jet_partonFlavour[24];   //[nJet]
   Int_t           Muon_genPartIdx[15];   //[nMuon]
   UChar_t         Muon_genPartFlav[15];   //[nMuon]
   Int_t           Photon_genPartIdx[9];   //[nPhoton]
   UChar_t         Photon_genPartFlav[9];   //[nPhoton]
   Float_t         MET_fiducialGenPhi;
   Float_t         MET_fiducialGenPt;
   UChar_t         Electron_cleanmask[7];   //[nElectron]
   UChar_t         Jet_cleanmask[24];   //[nJet]
   UChar_t         Muon_cleanmask[15];   //[nMuon]
   UChar_t         Photon_cleanmask[9];   //[nPhoton]
   Bool_t          Flag_HBHENoiseFilter;
   Bool_t          Flag_HBHENoiseIsoFilter;
   Bool_t          Flag_CSCTightHaloFilter;
   Bool_t          Flag_CSCTightHaloTrkMuUnvetoFilter;
   Bool_t          Flag_CSCTightHalo2015Filter;
   Bool_t          Flag_globalTightHalo2016Filter;
   Bool_t          Flag_globalSuperTightHalo2016Filter;
   Bool_t          Flag_HcalStripHaloFilter;
   Bool_t          Flag_hcalLaserEventFilter;
   Bool_t          Flag_EcalDeadCellTriggerPrimitiveFilter;
   Bool_t          Flag_EcalDeadCellBoundaryEnergyFilter;
   Bool_t          Flag_ecalBadCalibFilter;
   Bool_t          Flag_goodVertices;
   Bool_t          Flag_eeBadScFilter;
   Bool_t          Flag_ecalLaserCorrFilter;
   Bool_t          Flag_trkPOGFilters;
   Bool_t          Flag_chargedHadronTrackResolutionFilter;
   Bool_t          Flag_muonBadTrackFilter;
   Bool_t          Flag_BadChargedCandidateFilter;
   Bool_t          Flag_BadPFMuonFilter;
   Bool_t          Flag_BadPFMuonDzFilter;
   Bool_t          Flag_hfNoisyHitsFilter;
   Bool_t          Flag_BadChargedCandidateSummer16Filter;
   Bool_t          Flag_BadPFMuonSummer16Filter;
   Bool_t          Flag_trkPOG_manystripclus53X;
   Bool_t          Flag_trkPOG_toomanystripclus53X;
   Bool_t          Flag_trkPOG_logErrorTooManyClusters;
   Bool_t          Flag_METFilters;
   Bool_t          HLT_Ele27_Ele37_CaloIdL_MW;
   Bool_t          HLT_Mu27_Ele37_CaloIdL_MW;
   Bool_t          HLT_Mu37_Ele27_CaloIdL_MW;
   Bool_t          HLT_Mu37_TkMu27;
   Bool_t          HLT_DoublePhoton33_CaloIdL;
   Bool_t          HLT_DoublePhoton70;
   Bool_t          HLT_DoublePhoton85;
   Bool_t          HLT_Ele27_WPTight_Gsf;
   Bool_t          HLT_Ele28_WPTight_Gsf;
   Bool_t          HLT_Ele30_WPTight_Gsf;
   Bool_t          HLT_Ele32_WPTight_Gsf;
   Bool_t          HLT_Ele35_WPTight_Gsf;
   Bool_t          HLT_Ele35_WPTight_Gsf_L1EGMT;
   Bool_t          HLT_Ele38_WPTight_Gsf;
   Bool_t          HLT_Ele40_WPTight_Gsf;
   Bool_t          HLT_Ele32_WPTight_Gsf_L1DoubleEG;
   Bool_t          HLT_Ele24_eta2p1_WPTight_Gsf_LooseChargedIsoPFTauHPS30_eta2p1_CrossL1;
   Bool_t          HLT_Ele24_eta2p1_WPTight_Gsf_MediumChargedIsoPFTauHPS30_eta2p1_CrossL1;
   Bool_t          HLT_Ele24_eta2p1_WPTight_Gsf_TightChargedIsoPFTauHPS30_eta2p1_CrossL1;
   Bool_t          HLT_Ele24_eta2p1_WPTight_Gsf_LooseChargedIsoPFTauHPS30_eta2p1_TightID_CrossL1;
   Bool_t          HLT_Ele24_eta2p1_WPTight_Gsf_MediumChargedIsoPFTauHPS30_eta2p1_TightID_CrossL1;
   Bool_t          HLT_Ele24_eta2p1_WPTight_Gsf_TightChargedIsoPFTauHPS30_eta2p1_TightID_CrossL1;
   Bool_t          HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL;
   Bool_t          HLT_Mu19_TrkIsoVVL_Mu9_TrkIsoVVL;
   Bool_t          HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ;
   Bool_t          HLT_Mu19_TrkIsoVVL_Mu9_TrkIsoVVL_DZ;
   Bool_t          HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass8;
   Bool_t          HLT_Mu19_TrkIsoVVL_Mu9_TrkIsoVVL_DZ_Mass8;
   Bool_t          HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass3p8;
   Bool_t          HLT_Mu19_TrkIsoVVL_Mu9_TrkIsoVVL_DZ_Mass3p8;
   Bool_t          HLT_PFJet15;
   Bool_t          HLT_PFJet25;
   Bool_t          HLT_PFJet40;
   Bool_t          HLT_PFJet60;
   Bool_t          HLT_PFJet80;
   Bool_t          HLT_PFJet140;
   Bool_t          HLT_PFJet200;
   Bool_t          HLT_PFJet260;
   Bool_t          HLT_PFJet320;
   Bool_t          HLT_PFJet400;
   Bool_t          HLT_PFJet450;
   Bool_t          HLT_PFJet500;
   Bool_t          HLT_PFJet550;
   Bool_t          HLT_PFJetFwd15;
   Bool_t          HLT_PFJetFwd25;
   Bool_t          HLT_PFJetFwd40;
   Bool_t          HLT_PFJetFwd60;
   Bool_t          HLT_PFJetFwd80;
   Bool_t          HLT_PFJetFwd140;
   Bool_t          HLT_PFJetFwd200;
   Bool_t          HLT_PFJetFwd260;
   Bool_t          HLT_PFJetFwd320;
   Bool_t          HLT_PFJetFwd400;
   Bool_t          HLT_PFJetFwd450;
   Bool_t          HLT_PFJetFwd500;
   Bool_t          HLT_PFMET110_PFMHT110_IDTight;
   Bool_t          HLT_PFMET120_PFMHT120_IDTight;
   Bool_t          HLT_PFMET130_PFMHT130_IDTight;
   Bool_t          HLT_PFMET140_PFMHT140_IDTight;
   Bool_t          HLT_PFMET100_PFMHT100_IDTight_CaloBTagDeepCSV_3p1;
   Bool_t          HLT_PFMET110_PFMHT110_IDTight_CaloBTagDeepCSV_3p1;
   Bool_t          HLT_PFMET120_PFMHT120_IDTight_CaloBTagDeepCSV_3p1;
   Bool_t          HLT_PFMET130_PFMHT130_IDTight_CaloBTagDeepCSV_3p1;
   Bool_t          HLT_PFMET140_PFMHT140_IDTight_CaloBTagDeepCSV_3p1;
   Bool_t          HLT_PFMET120_PFMHT120_IDTight_PFHT60;
   Bool_t          HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_PFHT60;
   Bool_t          HLT_PFMETTypeOne120_PFMHT120_IDTight_PFHT60;
   Bool_t          HLT_PFMETTypeOne110_PFMHT110_IDTight;
   Bool_t          HLT_PFMETTypeOne120_PFMHT120_IDTight;
   Bool_t          HLT_PFMETTypeOne130_PFMHT130_IDTight;
   Bool_t          HLT_PFMETTypeOne140_PFMHT140_IDTight;
   Bool_t          HLT_PFMETNoMu110_PFMHTNoMu110_IDTight;
   Bool_t          HLT_PFMETNoMu120_PFMHTNoMu120_IDTight;
   Bool_t          HLT_PFMETNoMu130_PFMHTNoMu130_IDTight;
   Bool_t          HLT_PFMETNoMu140_PFMHTNoMu140_IDTight;
   Bool_t          HLT_PFMET200_NotCleaned;
   Bool_t          HLT_PFMET200_HBHECleaned;
   Bool_t          HLT_PFMET250_HBHECleaned;
   Bool_t          HLT_PFMET300_HBHECleaned;
   Bool_t          HLT_PFMET200_HBHE_BeamHaloCleaned;
   Bool_t          HLT_PFMETTypeOne200_HBHE_BeamHaloCleaned;
   Bool_t          HLT_MET105_IsoTrk50;
   Bool_t          HLT_MET120_IsoTrk50;
   Bool_t          HLT_DoublePFJets40_CaloBTagDeepCSV_p71;
   Bool_t          HLT_DoublePFJets100_CaloBTagDeepCSV_p71;
   Bool_t          HLT_DoublePFJets200_CaloBTagDeepCSV_p71;
   Bool_t          HLT_DoublePFJets350_CaloBTagDeepCSV_p71;
   Bool_t          HLT_DoublePFJets116MaxDeta1p6_DoubleCaloBTagDeepCSV_p71;
   Bool_t          HLT_DoublePFJets128MaxDeta1p6_DoubleCaloBTagDeepCSV_p71;
   Bool_t          HLT_Photon300_NoHE;
   Bool_t          HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ;
   Bool_t          HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ_PFDiJet30;
   Bool_t          HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ_CaloDiJet30;
   Bool_t          HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ_PFDiJet30_PFBtagDeepCSV_1p5;
   Bool_t          HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ_CaloDiJet30_CaloBtagDeepCSV_1p5;
   Bool_t          HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL;
   Bool_t          HLT_Ele15_Ele8_CaloIdL_TrackIdL_IsoVL;
   Bool_t          HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ;
   Bool_t          HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL;
   Bool_t          HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_DZ;
   Bool_t          HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL;
   Bool_t          HLT_Mu12_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL;
   Bool_t          HLT_Mu12_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ;
   Bool_t          HLT_Photon20;
   Bool_t          HLT_Photon33;
   Bool_t          HLT_Photon50;
   Bool_t          HLT_Photon75;
   Bool_t          HLT_Photon90;
   Bool_t          HLT_Photon120;
   Bool_t          HLT_Photon150;
   Bool_t          HLT_Photon175;
   Bool_t          HLT_Photon200;
   Bool_t          HLT_Photon100EB_TightID_TightIso;
   Bool_t          HLT_Photon110EB_TightID_TightIso;
   Bool_t          HLT_Photon120EB_TightID_TightIso;
   Bool_t          HLT_Photon100EBHE10;
   Bool_t          HLT_Photon100EEHE10;
   Bool_t          HLT_Photon100EE_TightID_TightIso;
   Bool_t          HLT_Photon50_R9Id90_HE10_IsoM;
   Bool_t          HLT_Photon75_R9Id90_HE10_IsoM;
   Bool_t          HLT_Photon75_R9Id90_HE10_IsoM_EBOnly_CaloMJJ300_PFJetsMJJ400DEta3;
   Bool_t          HLT_Photon75_R9Id90_HE10_IsoM_EBOnly_CaloMJJ400_PFJetsMJJ600DEta3;
   Bool_t          HLT_Photon90_R9Id90_HE10_IsoM;
   Bool_t          HLT_Photon120_R9Id90_HE10_IsoM;
   Bool_t          HLT_Photon165_R9Id90_HE10_IsoM;
   Bool_t          HLT_Photon90_CaloIdL_PFHT700;
   Bool_t          HLT_Diphoton30_22_R9Id_OR_IsoCaloId_AND_HE_R9Id_Mass90;
   Bool_t          HLT_Diphoton30_22_R9Id_OR_IsoCaloId_AND_HE_R9Id_Mass95;
   Bool_t          HLT_Diphoton30PV_18PV_R9Id_AND_IsoCaloId_AND_HE_R9Id_PixelVeto_Mass55;
   Bool_t          HLT_Diphoton30PV_18PV_R9Id_AND_IsoCaloId_AND_HE_R9Id_NoPixelVeto_Mass55;
   Bool_t          HLT_Photon35_TwoProngs35;
   Bool_t          HLT_Ele30_eta2p1_WPTight_Gsf_CentralPFJet35_EleCleaned;
   Bool_t          HLT_Ele28_eta2p1_WPTight_Gsf_HT150;
   Bool_t          HLT_Ele28_HighEta_SC20_Mass55;
   Bool_t          HLT_DoubleIsoMu20_eta2p1;
   Bool_t          HLT_Ele23_CaloIdL_TrackIdL_IsoVL_PFJet30;
   Bool_t          HLT_Ele23_CaloIdM_TrackIdM_PFJet30;
   Bool_t          HLT_Ele50_CaloIdVT_GsfTrkIdT_PFJet165;
   Bool_t          HLT_Ele115_CaloIdVT_GsfTrkIdT;
   Bool_t          HLT_Ele135_CaloIdVT_GsfTrkIdT;
   Bool_t          HLT_Ele145_CaloIdVT_GsfTrkIdT;
   Bool_t          HLT_Ele200_CaloIdVT_GsfTrkIdT;
   Bool_t          HLT_Ele250_CaloIdVT_GsfTrkIdT;
   Bool_t          HLT_Ele300_CaloIdVT_GsfTrkIdT;
   Bool_t          HLT_Photon60_R9Id90_CaloIdL_IsoL;
   Bool_t          HLT_Photon60_R9Id90_CaloIdL_IsoL_DisplacedIdL;
   Bool_t          HLT_Photon60_R9Id90_CaloIdL_IsoL_DisplacedIdL_PFHT350MinPFJet15;
   Bool_t          HLT_AK4PFJet30;
   Bool_t          HLT_AK4PFJet50;
   Bool_t          HLT_AK4PFJet80;
   Bool_t          HLT_AK4PFJet100;
   Bool_t          HLT_AK4PFJet120;
   Bool_t          HLT_SinglePhoton10_Eta3p1ForPPRef;
   Bool_t          HLT_SinglePhoton20_Eta3p1ForPPRef;
   Bool_t          HLT_SinglePhoton30_Eta3p1ForPPRef;
   Bool_t          HLT_Photon20_HoverELoose;
   Bool_t          HLT_Photon30_HoverELoose;
   Bool_t          HLT_Photon50_R9Id90_HE10_IsoM_EBOnly_PFJetsMJJ300DEta3_PFMET50;
   Bool_t          HLT_Photon75_R9Id90_HE10_IsoM_EBOnly_PFJetsMJJ300DEta3;
   Bool_t          HLT_Photon75_R9Id90_HE10_IsoM_EBOnly_PFJetsMJJ600DEta3;
   Bool_t          HLT_PFMET100_PFMHT100_IDTight_PFHT60;
   Bool_t          HLT_PFMETNoMu100_PFMHTNoMu100_IDTight_PFHT60;
   Bool_t          HLT_PFMETTypeOne100_PFMHT100_IDTight_PFHT60;
   Bool_t          HLT_Diphoton30_18_R9IdL_AND_HE_AND_IsoCaloId_NoPixelVeto_Mass55;
   Bool_t          HLT_Diphoton30_18_R9IdL_AND_HE_AND_IsoCaloId_NoPixelVeto;
   Float_t         Muon_corrected_pt[15];   //[nMuon]
   Float_t         Muon_correctedUp_pt[15];   //[nMuon]
   Float_t         Muon_correctedDown_pt[15];   //[nMuon]
   UInt_t          nLepton;
   Float_t         Lepton_jetPtRelv2[18];   //[nLepton]
   UChar_t         Lepton_jetNDauCharged[18];   //[nLepton]
   Int_t           Lepton_jetIdx[18];   //[nLepton]
   Int_t           Lepton_photonIdx[18];   //[nLepton]
   Float_t         Lepton_jetRelIso[18];   //[nLepton]
   UInt_t          nlepton;
   Float_t         lepton_mass[2];   //[nlepton]
   Bool_t          lepton_isStandalone[2];   //[nlepton]
   Bool_t          lepton_mvaFall17V2noIso_WPL[2];   //[nlepton]
   Float_t         lepton_dxyErr[2];   //[nlepton]
   Float_t         lepton_dr03TkSumPt[2];   //[nlepton]
   Bool_t          lepton_mediumPromptId[2];   //[nlepton]
   Float_t         lepton_mvaLowPt[2];   //[nlepton]
   UChar_t         lepton_mvaLowPtId[2];   //[nlepton]
   Bool_t          lepton_mediumId[2];   //[nlepton]
   UChar_t         lepton_mvaId[2];   //[nlepton]
   Float_t         lepton_dxy[2];   //[nlepton]
   Float_t         lepton_softMva[2];   //[nlepton]
   Float_t         lepton_energyErr[2];   //[nlepton]
   Int_t           lepton_pdgId[2];   //[nlepton]
   Bool_t          lepton_inTimeMuon[2];   //[nlepton]
   UChar_t         lepton_genPartFlav[2];   //[nlepton]
   Bool_t          lepton_triggerIdLoose[2];   //[nlepton]
   Float_t         lepton_hoe[2];   //[nlepton]
   Float_t         lepton_miniPFRelIso_all[2];   //[nlepton]
   Float_t         lepton_correctedUp_pt[2];   //[nlepton]
   Float_t         lepton_sip3d[2];   //[nlepton]
   Int_t           lepton_nTrackerLayers[2];   //[nlepton]
   UChar_t         lepton_tkIsoId[2];   //[nlepton]
   UChar_t         lepton_miniIsoId[2];   //[nlepton]
   UChar_t         lepton_multiIsoId[2];   //[nlepton]
   Float_t         lepton_pfRelIso03_all[2];   //[nlepton]
   Float_t         lepton_dr03EcalRecHitSumEt[2];   //[nlepton]
   UChar_t         lepton_highPtId[2];   //[nlepton]
   Bool_t          lepton_softMvaId[2];   //[nlepton]
   Bool_t          lepton_isGlobal[2];   //[nlepton]
   Float_t         lepton_miniPFRelIso_chg[2];   //[nlepton]
   Float_t         lepton_jetRelIso[2];   //[nlepton]
   Int_t           lepton_genPartIdx[2];   //[nlepton]
   Int_t           lepton_vidNestedWPBitmap[2];   //[nlepton]
   Float_t         lepton_ip3d[2];   //[nlepton]
   Float_t         lepton_jetPtRelv2[2];   //[nlepton]
   Int_t           lepton_photonIdx[2];   //[nlepton]
   Float_t         lepton_pfRelIso03_chg[2];   //[nlepton]
   Float_t         lepton_mvaFall17V2Iso[2];   //[nlepton]
   UChar_t         lepton_lostHits[2];   //[nlepton]
   Float_t         lepton_dr03TkSumPtHEEP[2];   //[nlepton]
   Float_t         lepton_tunepRelPt[2];   //[nlepton]
   Float_t         lepton_dxybs[2];   //[nlepton]
   Bool_t          lepton_cutBased_HEEP[2];   //[nlepton]
   Float_t         lepton_pt[2];   //[nlepton]
   Bool_t          lepton_softId[2];   //[nlepton]
   Bool_t          lepton_mvaFall17V2noIso_WP90[2];   //[nlepton]
   UChar_t         lepton_puppiIsoId[2];   //[nlepton]
   Float_t         lepton_mvaTTH[2];   //[nlepton]
   Int_t           lepton_charge[2];   //[nlepton]
   Int_t           lepton_vidNestedWPBitmapHEEP[2];   //[nlepton]
   Bool_t          lepton_isPFcand[2];   //[nlepton]
   Float_t         lepton_correctedDown_pt[2];   //[nlepton]
   Float_t         lepton_sieie[2];   //[nlepton]
   Float_t         lepton_dr03HcalDepth1TowerSumEt[2];   //[nlepton]
   Bool_t          lepton_highPurity[2];   //[nlepton]
   Float_t         lepton_dz[2];   //[nlepton]
   Float_t         lepton_scEtOverPt[2];   //[nlepton]
   Bool_t          lepton_isTracker[2];   //[nlepton]
   Float_t         lepton_tkRelIso[2];   //[nlepton]
   Bool_t          lepton_tightId[2];   //[nlepton]
   Float_t         lepton_pfRelIso04_all[2];   //[nlepton]
   Float_t         lepton_dzErr[2];   //[nlepton]
   Float_t         lepton_phi[2];   //[nlepton]
   UChar_t         lepton_pfIsoId[2];   //[nlepton]
   UChar_t         lepton_jetNDauCharged[2];   //[nlepton]
   Float_t         lepton_eInvMinusPInv[2];   //[nlepton]
   Float_t         lepton_dEsigmaDown[2];   //[nlepton]
   Float_t         lepton_mvaFall17V2noIso[2];   //[nlepton]
   Int_t           lepton_tightCharge[2];   //[nlepton]
   Int_t           lepton_fsrPhotonIdx[2];   //[nlepton]
   Float_t         lepton_eCorr[2];   //[nlepton]
   Bool_t          lepton_mvaFall17V2Iso_WP80[2];   //[nlepton]
   Bool_t          lepton_looseId[2];   //[nlepton]
   Int_t           lepton_cutBased[2];   //[nlepton]
   Float_t         lepton_dEscaleUp[2];   //[nlepton]
   Bool_t          lepton_mvaFall17V2Iso_WPL[2];   //[nlepton]
   Float_t         lepton_ptErr[2];   //[nlepton]
   Float_t         lepton_deltaEtaSC[2];   //[nlepton]
   Float_t         lepton_r9[2];   //[nlepton]
   UChar_t         lepton_seedGain[2];   //[nlepton]
   Float_t         lepton_eta[2];   //[nlepton]
   Bool_t          lepton_mvaFall17V2Iso_WP90[2];   //[nlepton]
   Int_t           lepton_nStations[2];   //[nlepton]
   Float_t         lepton_corrected_pt[2];   //[nlepton]
   Float_t         lepton_dEsigmaUp[2];   //[nlepton]
   Int_t           lepton_jetIdx[2];   //[nlepton]
   Float_t         lepton_dEscaleDown[2];   //[nlepton]
   Bool_t          lepton_mvaFall17V2noIso_WP80[2];   //[nlepton]
   Bool_t          lepton_convVeto[2];   //[nlepton]
   UChar_t         lepton_cleanmask[2];   //[nlepton]
   Float_t         lepton_segmentComp[2];   //[nlepton]
   Bool_t          lepton_is_tight[2];   //[nlepton]
   Bool_t          lepton_is_real[2];   //[nlepton]
   UInt_t          nphoton;
   Bool_t          photon_electronVeto[3];   //[nphoton]
   Int_t           photon_genPartIdx[3];   //[nphoton]
   UChar_t         photon_genPartFlav[3];   //[nphoton]
   UChar_t         photon_cleanmask[3];   //[nphoton]
   Int_t           photon_cutBased[3];   //[nphoton]
   Int_t           photon_cutBased_Fall17V1Bitmap[3];   //[nphoton]
   Int_t           photon_jetIdx[3];   //[nphoton]
   Int_t           photon_vidNestedWPBitmap[3];   //[nphoton]
   Float_t         photon_dEscaleUp[3];   //[nphoton]
   Float_t         photon_dEsigmaDown[3];   //[nphoton]
   Float_t         photon_eta[3];   //[nphoton]
   Float_t         photon_hoe[3];   //[nphoton]
   Float_t         photon_mass[3];   //[nphoton]
   Float_t         photon_mvaID[3];   //[nphoton]
   Float_t         photon_mvaID_Fall17V1p1[3];   //[nphoton]
   Float_t         photon_pfRelIso03_all[3];   //[nphoton]
   Float_t         photon_pfRelIso03_chg[3];   //[nphoton]
   Float_t         photon_dEscaleDown[3];   //[nphoton]
   Int_t           photon_electronIdx[3];   //[nphoton]
   Int_t           photon_pdgId[3];   //[nphoton]
   Float_t         photon_dEsigmaUp[3];   //[nphoton]
   Float_t         photon_eCorr[3];   //[nphoton]
   Float_t         photon_energyErr[3];   //[nphoton]
   Float_t         photon_phi[3];   //[nphoton]
   Float_t         photon_pt[3];   //[nphoton]
   Float_t         photon_r9[3];   //[nphoton]
   Float_t         photon_sieie[3];   //[nphoton]
   Int_t           photon_charge[3];   //[nphoton]
   Bool_t          photon_isScEtaEB[3];   //[nphoton]
   Bool_t          photon_mvaID_WP80[3];   //[nphoton]
   Bool_t          photon_mvaID_WP90[3];   //[nphoton]
   Bool_t          photon_pixelSeed[3];   //[nphoton]
   UChar_t         photon_seedGain[3];   //[nphoton]
   Bool_t          photon_isScEtaEE[3];   //[nphoton]
   UInt_t          njet;
   Float_t         jet_area[11];   //[njet]
   Float_t         jet_btagCSVV2[11];   //[njet]
   Float_t         jet_btagDeepB[11];   //[njet]
   Float_t         jet_btagDeepCvB[11];   //[njet]
   Float_t         jet_btagDeepCvL[11];   //[njet]
   Float_t         jet_btagDeepFlavB[11];   //[njet]
   Float_t         jet_btagDeepFlavCvB[11];   //[njet]
   Float_t         jet_btagDeepFlavCvL[11];   //[njet]
   Float_t         jet_btagDeepFlavQG[11];   //[njet]
   Float_t         jet_chEmEF[11];   //[njet]
   Float_t         jet_chFPV0EF[11];   //[njet]
   Float_t         jet_chHEF[11];   //[njet]
   Float_t         jet_eta[11];   //[njet]
   Float_t         jet_hfsigmaEtaEta[11];   //[njet]
   Float_t         jet_hfsigmaPhiPhi[11];   //[njet]
   Float_t         jet_mass[11];   //[njet]
   Float_t         jet_muEF[11];   //[njet]
   Float_t         jet_muonSubtrFactor[11];   //[njet]
   Float_t         jet_neEmEF[11];   //[njet]
   Float_t         jet_neHEF[11];   //[njet]
   Float_t         jet_phi[11];   //[njet]
   Float_t         jet_pt[11];   //[njet]
   Float_t         jet_puIdDisc[11];   //[njet]
   Float_t         jet_qgl[11];   //[njet]
   Float_t         jet_rawFactor[11];   //[njet]
   Float_t         jet_bRegCorr[11];   //[njet]
   Int_t           jet_partonFlavour[11];   //[njet]
   Float_t         jet_bRegRes[11];   //[njet]
   Float_t         jet_cRegCorr[11];   //[njet]
   Int_t           jet_electronIdx1[11];   //[njet]
   Int_t           jet_electronIdx2[11];   //[njet]
   Int_t           jet_hfadjacentEtaStripsSize[11];   //[njet]
   Int_t           jet_hfcentralEtaStripSize[11];   //[njet]
   Int_t           jet_muonIdx1[11];   //[njet]
   Int_t           jet_nMuons[11];   //[njet]
   Int_t           jet_puId[11];   //[njet]
   UChar_t         jet_cleanmask[11];   //[njet]
   Int_t           jet_genJetIdx[11];   //[njet]
   Int_t           jet_hadronFlavour[11];   //[njet]
   Float_t         jet_cRegRes[11];   //[njet]
   Int_t           jet_jetId[11];   //[njet]
   Int_t           jet_muonIdx2[11];   //[njet]
   Int_t           jet_nElectrons[11];   //[njet]
   UChar_t         jet_nConstituents[11];   //[njet]
   Float_t         Jet_pt_raw[24];   //[nJet]
   Float_t         Jet_pt_nom[24];   //[nJet]
   Float_t         Jet_mass_raw[24];   //[nJet]
   Float_t         Jet_mass_nom[24];   //[nJet]
   Float_t         Jet_corr_JEC[24];   //[nJet]
   Float_t         Jet_corr_JER[24];   //[nJet]
   Float_t         MET_T1_pt;
   Float_t         MET_T1_phi;
   Float_t         MET_T1Smear_pt;
   Float_t         MET_T1Smear_phi;
   Float_t         Jet_pt_jerUp[24];   //[nJet]
   Float_t         Jet_mass_jerUp[24];   //[nJet]
   Float_t         MET_T1_pt_jerUp;
   Float_t         MET_T1_phi_jerUp;
   Float_t         MET_T1Smear_pt_jerUp;
   Float_t         MET_T1Smear_phi_jerUp;
   Float_t         Jet_pt_jesTotalUp[24];   //[nJet]
   Float_t         Jet_mass_jesTotalUp[24];   //[nJet]
   Float_t         MET_T1_pt_jesTotalUp;
   Float_t         MET_T1_phi_jesTotalUp;
   Float_t         MET_T1Smear_pt_jesTotalUp;
   Float_t         MET_T1Smear_phi_jesTotalUp;
   Float_t         Jet_pt_jesHEMIssueUp[24];   //[nJet]
   Float_t         Jet_mass_jesHEMIssueUp[24];   //[nJet]
   Float_t         MET_T1_pt_jesHEMIssueUp;
   Float_t         MET_T1_phi_jesHEMIssueUp;
   Float_t         MET_T1Smear_pt_jesHEMIssueUp;
   Float_t         MET_T1Smear_phi_jesHEMIssueUp;
   Float_t         MET_T1_pt_unclustEnUp;
   Float_t         MET_T1_phi_unclustEnUp;
   Float_t         MET_T1Smear_pt_unclustEnUp;
   Float_t         MET_T1Smear_phi_unclustEnUp;
   Float_t         Jet_pt_jerDown[24];   //[nJet]
   Float_t         Jet_mass_jerDown[24];   //[nJet]
   Float_t         MET_T1_pt_jerDown;
   Float_t         MET_T1_phi_jerDown;
   Float_t         MET_T1Smear_pt_jerDown;
   Float_t         MET_T1Smear_phi_jerDown;
   Float_t         Jet_pt_jesTotalDown[24];   //[nJet]
   Float_t         Jet_mass_jesTotalDown[24];   //[nJet]
   Float_t         MET_T1_pt_jesTotalDown;
   Float_t         MET_T1_phi_jesTotalDown;
   Float_t         MET_T1Smear_pt_jesTotalDown;
   Float_t         MET_T1Smear_phi_jesTotalDown;
   Float_t         Jet_pt_jesHEMIssueDown[24];   //[nJet]
   Float_t         Jet_mass_jesHEMIssueDown[24];   //[nJet]
   Float_t         MET_T1_pt_jesHEMIssueDown;
   Float_t         MET_T1_phi_jesHEMIssueDown;
   Float_t         MET_T1Smear_pt_jesHEMIssueDown;
   Float_t         MET_T1Smear_phi_jesHEMIssueDown;
   Float_t         MET_T1_pt_unclustEnDown;
   Float_t         MET_T1_phi_unclustEnDown;
   Float_t         MET_T1Smear_pt_unclustEnDown;
   Float_t         MET_T1Smear_phi_unclustEnDown;
   Float_t         lumi;
   Int_t           photon_selection;
   Int_t           n_loose_mu;
   Int_t           n_loose_ele;
   Int_t           n_tight_mu;
   Int_t           n_tight_ele;
   Bool_t          HEM_issue;
   Float_t         mjj;
   Int_t           n_leptons;
   Float_t         jet_pt_jesTotalDown[11];   //[njet]
   Float_t         jet_mass_jesTotalDown[11];   //[njet]
   Float_t         jet_pt_jesHEMIssueDown[11];   //[njet]
   Float_t         jet_mass_jesHEMIssueDown[11];   //[njet]
   Float_t         jet_pt_jerDown[11];   //[njet]
   Float_t         jet_mass_jerDown[11];   //[njet]
   Float_t         jet_pt_jesTotalUp[11];   //[njet]
   Float_t         jet_mass_jesTotalUp[11];   //[njet]
   Float_t         jet_pt_jesHEMIssueUp[11];   //[njet]
   Float_t         jet_pt_jerUp[11];   //[njet]
   Float_t         jet_mass_jerUp[11];   //[njet]
   Float_t         jet_mass_jesHEMIssueUp[11];   //[njet]
   Float_t         jet_pt_raw[11];   //[njet]
   Float_t         jet_pt_nom[11];   //[njet]
   Float_t         jet_mass_raw[11];   //[njet]
   Float_t         jet_corr_JEC[11];   //[njet]
   Float_t         jet_mass_nom[11];   //[njet]
   Float_t         jet_corr_JER[11];   //[njet]
   Int_t           n_photon;
   Int_t           photon_isprompt;
   Int_t           photon_gen_matching;
   Int_t           npu;
   Float_t         ntruepu;
   Int_t           npvs;
   Int_t           njets25;
   Float_t         puWeight;
   Float_t         puWeightUp;
   Float_t         puWeightDown;

   // List of branches
   TBranch        *b_run;   //!
   TBranch        *b_luminosityBlock;   //!
   TBranch        *b_event;   //!
   TBranch        *b_btagWeight_CSVV2;   //!
   TBranch        *b_btagWeight_DeepCSVB;   //!
   TBranch        *b_nCorrT1METJet;   //!
   TBranch        *b_CorrT1METJet_area;   //!
   TBranch        *b_CorrT1METJet_eta;   //!
   TBranch        *b_CorrT1METJet_muonSubtrFactor;   //!
   TBranch        *b_CorrT1METJet_phi;   //!
   TBranch        *b_CorrT1METJet_rawPt;   //!
   TBranch        *b_nElectron;   //!
   TBranch        *b_Electron_dEscaleDown;   //!
   TBranch        *b_Electron_dEscaleUp;   //!
   TBranch        *b_Electron_dEsigmaDown;   //!
   TBranch        *b_Electron_dEsigmaUp;   //!
   TBranch        *b_Electron_deltaEtaSC;   //!
   TBranch        *b_Electron_dr03EcalRecHitSumEt;   //!
   TBranch        *b_Electron_dr03HcalDepth1TowerSumEt;   //!
   TBranch        *b_Electron_dr03TkSumPt;   //!
   TBranch        *b_Electron_dr03TkSumPtHEEP;   //!
   TBranch        *b_Electron_dxy;   //!
   TBranch        *b_Electron_dxyErr;   //!
   TBranch        *b_Electron_dz;   //!
   TBranch        *b_Electron_dzErr;   //!
   TBranch        *b_Electron_eCorr;   //!
   TBranch        *b_Electron_eInvMinusPInv;   //!
   TBranch        *b_Electron_energyErr;   //!
   TBranch        *b_Electron_eta;   //!
   TBranch        *b_Electron_hoe;   //!
   TBranch        *b_Electron_ip3d;   //!
   TBranch        *b_Electron_jetPtRelv2;   //!
   TBranch        *b_Electron_jetRelIso;   //!
   TBranch        *b_Electron_mass;   //!
   TBranch        *b_Electron_miniPFRelIso_all;   //!
   TBranch        *b_Electron_miniPFRelIso_chg;   //!
   TBranch        *b_Electron_mvaFall17V2Iso;   //!
   TBranch        *b_Electron_mvaFall17V2noIso;   //!
   TBranch        *b_Electron_pfRelIso03_all;   //!
   TBranch        *b_Electron_pfRelIso03_chg;   //!
   TBranch        *b_Electron_phi;   //!
   TBranch        *b_Electron_pt;   //!
   TBranch        *b_Electron_r9;   //!
   TBranch        *b_Electron_scEtOverPt;   //!
   TBranch        *b_Electron_sieie;   //!
   TBranch        *b_Electron_sip3d;   //!
   TBranch        *b_Electron_mvaTTH;   //!
   TBranch        *b_Electron_charge;   //!
   TBranch        *b_Electron_cutBased;   //!
   TBranch        *b_Electron_jetIdx;   //!
   TBranch        *b_Electron_pdgId;   //!
   TBranch        *b_Electron_photonIdx;   //!
   TBranch        *b_Electron_tightCharge;   //!
   TBranch        *b_Electron_vidNestedWPBitmap;   //!
   TBranch        *b_Electron_vidNestedWPBitmapHEEP;   //!
   TBranch        *b_Electron_convVeto;   //!
   TBranch        *b_Electron_cutBased_HEEP;   //!
   TBranch        *b_Electron_isPFcand;   //!
   TBranch        *b_Electron_jetNDauCharged;   //!
   TBranch        *b_Electron_lostHits;   //!
   TBranch        *b_Electron_mvaFall17V2Iso_WP80;   //!
   TBranch        *b_Electron_mvaFall17V2Iso_WP90;   //!
   TBranch        *b_Electron_mvaFall17V2Iso_WPL;   //!
   TBranch        *b_Electron_mvaFall17V2noIso_WP80;   //!
   TBranch        *b_Electron_mvaFall17V2noIso_WP90;   //!
   TBranch        *b_Electron_mvaFall17V2noIso_WPL;   //!
   TBranch        *b_Electron_seedGain;   //!
   TBranch        *b_nFsrPhoton;   //!
   TBranch        *b_nGenJet;   //!
   TBranch        *b_GenJet_eta;   //!
   TBranch        *b_GenJet_mass;   //!
   TBranch        *b_GenJet_phi;   //!
   TBranch        *b_GenJet_pt;   //!
   TBranch        *b_nGenPart;   //!
   TBranch        *b_GenPart_eta;   //!
   TBranch        *b_GenPart_mass;   //!
   TBranch        *b_GenPart_phi;   //!
   TBranch        *b_GenPart_pt;   //!
   TBranch        *b_GenPart_genPartIdxMother;   //!
   TBranch        *b_GenPart_pdgId;   //!
   TBranch        *b_GenPart_status;   //!
   TBranch        *b_GenPart_statusFlags;   //!
   TBranch        *b_Generator_binvar;   //!
   TBranch        *b_Generator_scalePDF;   //!
   TBranch        *b_Generator_weight;   //!
   TBranch        *b_Generator_x1;   //!
   TBranch        *b_Generator_x2;   //!
   TBranch        *b_Generator_xpdf1;   //!
   TBranch        *b_Generator_xpdf2;   //!
   TBranch        *b_Generator_id1;   //!
   TBranch        *b_Generator_id2;   //!
   TBranch        *b_GenVtx_x;   //!
   TBranch        *b_GenVtx_y;   //!
   TBranch        *b_GenVtx_z;   //!
   TBranch        *b_nGenVisTau;   //!
   TBranch        *b_genWeight;   //!
   TBranch        *b_LHEWeight_originalXWGTUP;   //!
   TBranch        *b_nLHEPdfWeight;   //!
   TBranch        *b_LHEPdfWeight;   //!
   TBranch        *b_nLHEReweightingWeight;   //!
   TBranch        *b_LHEReweightingWeight;   //!
   TBranch        *b_nLHEScaleWeight;   //!
   TBranch        *b_LHEScaleWeight;   //!
   TBranch        *b_nPSWeight;   //!
   TBranch        *b_PSWeight;   //!
   TBranch        *b_nIsoTrack;   //!
   TBranch        *b_nJet;   //!
   TBranch        *b_Jet_area;   //!
   TBranch        *b_Jet_btagCSVV2;   //!
   TBranch        *b_Jet_btagDeepB;   //!
   TBranch        *b_Jet_btagDeepCvB;   //!
   TBranch        *b_Jet_btagDeepCvL;   //!
   TBranch        *b_Jet_btagDeepFlavB;   //!
   TBranch        *b_Jet_btagDeepFlavCvB;   //!
   TBranch        *b_Jet_btagDeepFlavCvL;   //!
   TBranch        *b_Jet_btagDeepFlavQG;   //!
   TBranch        *b_Jet_chEmEF;   //!
   TBranch        *b_Jet_chFPV0EF;   //!
   TBranch        *b_Jet_chHEF;   //!
   TBranch        *b_Jet_eta;   //!
   TBranch        *b_Jet_hfsigmaEtaEta;   //!
   TBranch        *b_Jet_hfsigmaPhiPhi;   //!
   TBranch        *b_Jet_mass;   //!
   TBranch        *b_Jet_muEF;   //!
   TBranch        *b_Jet_muonSubtrFactor;   //!
   TBranch        *b_Jet_neEmEF;   //!
   TBranch        *b_Jet_neHEF;   //!
   TBranch        *b_Jet_phi;   //!
   TBranch        *b_Jet_pt;   //!
   TBranch        *b_Jet_puIdDisc;   //!
   TBranch        *b_Jet_qgl;   //!
   TBranch        *b_Jet_rawFactor;   //!
   TBranch        *b_Jet_bRegCorr;   //!
   TBranch        *b_Jet_bRegRes;   //!
   TBranch        *b_Jet_cRegCorr;   //!
   TBranch        *b_Jet_cRegRes;   //!
   TBranch        *b_Jet_electronIdx1;   //!
   TBranch        *b_Jet_electronIdx2;   //!
   TBranch        *b_Jet_hfadjacentEtaStripsSize;   //!
   TBranch        *b_Jet_hfcentralEtaStripSize;   //!
   TBranch        *b_Jet_jetId;   //!
   TBranch        *b_Jet_muonIdx1;   //!
   TBranch        *b_Jet_muonIdx2;   //!
   TBranch        *b_Jet_nElectrons;   //!
   TBranch        *b_Jet_nMuons;   //!
   TBranch        *b_Jet_puId;   //!
   TBranch        *b_Jet_nConstituents;   //!
   TBranch        *b_L1PreFiringWeight_Dn;   //!
   TBranch        *b_L1PreFiringWeight_ECAL_Dn;   //!
   TBranch        *b_L1PreFiringWeight_ECAL_Nom;   //!
   TBranch        *b_L1PreFiringWeight_ECAL_Up;   //!
   TBranch        *b_L1PreFiringWeight_Muon_Nom;   //!
   TBranch        *b_L1PreFiringWeight_Muon_StatDn;   //!
   TBranch        *b_L1PreFiringWeight_Muon_StatUp;   //!
   TBranch        *b_L1PreFiringWeight_Muon_SystDn;   //!
   TBranch        *b_L1PreFiringWeight_Muon_SystUp;   //!
   TBranch        *b_L1PreFiringWeight_Nom;   //!
   TBranch        *b_L1PreFiringWeight_Up;   //!
   TBranch        *b_LHE_Njets;   //!
   TBranch        *b_nLHEPart;   //!
   TBranch        *b_nLowPtElectron;   //!
   TBranch        *b_GenMET_phi;   //!
   TBranch        *b_GenMET_pt;   //!
   TBranch        *b_MET_MetUnclustEnUpDeltaX;   //!
   TBranch        *b_MET_MetUnclustEnUpDeltaY;   //!
   TBranch        *b_MET_covXX;   //!
   TBranch        *b_MET_covXY;   //!
   TBranch        *b_MET_covYY;   //!
   TBranch        *b_MET_phi;   //!
   TBranch        *b_MET_pt;   //!
   TBranch        *b_MET_significance;   //!
   TBranch        *b_MET_sumEt;   //!
   TBranch        *b_MET_sumPtUnclustered;   //!
   TBranch        *b_nMuon;   //!
   TBranch        *b_Muon_dxy;   //!
   TBranch        *b_Muon_dxyErr;   //!
   TBranch        *b_Muon_dxybs;   //!
   TBranch        *b_Muon_dz;   //!
   TBranch        *b_Muon_dzErr;   //!
   TBranch        *b_Muon_eta;   //!
   TBranch        *b_Muon_ip3d;   //!
   TBranch        *b_Muon_jetPtRelv2;   //!
   TBranch        *b_Muon_jetRelIso;   //!
   TBranch        *b_Muon_mass;   //!
   TBranch        *b_Muon_miniPFRelIso_all;   //!
   TBranch        *b_Muon_miniPFRelIso_chg;   //!
   TBranch        *b_Muon_pfRelIso03_all;   //!
   TBranch        *b_Muon_pfRelIso03_chg;   //!
   TBranch        *b_Muon_pfRelIso04_all;   //!
   TBranch        *b_Muon_phi;   //!
   TBranch        *b_Muon_pt;   //!
   TBranch        *b_Muon_ptErr;   //!
   TBranch        *b_Muon_segmentComp;   //!
   TBranch        *b_Muon_sip3d;   //!
   TBranch        *b_Muon_softMva;   //!
   TBranch        *b_Muon_tkRelIso;   //!
   TBranch        *b_Muon_tunepRelPt;   //!
   TBranch        *b_Muon_mvaLowPt;   //!
   TBranch        *b_Muon_mvaTTH;   //!
   TBranch        *b_Muon_charge;   //!
   TBranch        *b_Muon_jetIdx;   //!
   TBranch        *b_Muon_nStations;   //!
   TBranch        *b_Muon_nTrackerLayers;   //!
   TBranch        *b_Muon_pdgId;   //!
   TBranch        *b_Muon_tightCharge;   //!
   TBranch        *b_Muon_fsrPhotonIdx;   //!
   TBranch        *b_Muon_highPtId;   //!
   TBranch        *b_Muon_highPurity;   //!
   TBranch        *b_Muon_inTimeMuon;   //!
   TBranch        *b_Muon_isGlobal;   //!
   TBranch        *b_Muon_isPFcand;   //!
   TBranch        *b_Muon_isStandalone;   //!
   TBranch        *b_Muon_isTracker;   //!
   TBranch        *b_Muon_jetNDauCharged;   //!
   TBranch        *b_Muon_looseId;   //!
   TBranch        *b_Muon_mediumId;   //!
   TBranch        *b_Muon_mediumPromptId;   //!
   TBranch        *b_Muon_miniIsoId;   //!
   TBranch        *b_Muon_multiIsoId;   //!
   TBranch        *b_Muon_mvaId;   //!
   TBranch        *b_Muon_mvaLowPtId;   //!
   TBranch        *b_Muon_pfIsoId;   //!
   TBranch        *b_Muon_puppiIsoId;   //!
   TBranch        *b_Muon_softId;   //!
   TBranch        *b_Muon_softMvaId;   //!
   TBranch        *b_Muon_tightId;   //!
   TBranch        *b_Muon_tkIsoId;   //!
   TBranch        *b_Muon_triggerIdLoose;   //!
   TBranch        *b_nPhoton;   //!
   TBranch        *b_Photon_dEscaleDown;   //!
   TBranch        *b_Photon_dEscaleUp;   //!
   TBranch        *b_Photon_dEsigmaDown;   //!
   TBranch        *b_Photon_dEsigmaUp;   //!
   TBranch        *b_Photon_eCorr;   //!
   TBranch        *b_Photon_energyErr;   //!
   TBranch        *b_Photon_eta;   //!
   TBranch        *b_Photon_hoe;   //!
   TBranch        *b_Photon_mass;   //!
   TBranch        *b_Photon_mvaID;   //!
   TBranch        *b_Photon_mvaID_Fall17V1p1;   //!
   TBranch        *b_Photon_pfRelIso03_all;   //!
   TBranch        *b_Photon_pfRelIso03_chg;   //!
   TBranch        *b_Photon_phi;   //!
   TBranch        *b_Photon_pt;   //!
   TBranch        *b_Photon_r9;   //!
   TBranch        *b_Photon_sieie;   //!
   TBranch        *b_Photon_charge;   //!
   TBranch        *b_Photon_cutBased;   //!
   TBranch        *b_Photon_cutBased_Fall17V1Bitmap;   //!
   TBranch        *b_Photon_electronIdx;   //!
   TBranch        *b_Photon_jetIdx;   //!
   TBranch        *b_Photon_pdgId;   //!
   TBranch        *b_Photon_vidNestedWPBitmap;   //!
   TBranch        *b_Photon_electronVeto;   //!
   TBranch        *b_Photon_isScEtaEB;   //!
   TBranch        *b_Photon_isScEtaEE;   //!
   TBranch        *b_Photon_mvaID_WP80;   //!
   TBranch        *b_Photon_mvaID_WP90;   //!
   TBranch        *b_Photon_pixelSeed;   //!
   TBranch        *b_Photon_seedGain;   //!
   TBranch        *b_Pileup_nTrueInt;   //!
   TBranch        *b_Pileup_pudensity;   //!
   TBranch        *b_Pileup_gpudensity;   //!
   TBranch        *b_Pileup_nPU;   //!
   TBranch        *b_Pileup_sumEOOT;   //!
   TBranch        *b_Pileup_sumLOOT;   //!
   TBranch        *b_PuppiMET_phi;   //!
   TBranch        *b_PuppiMET_phiJERDown;   //!
   TBranch        *b_PuppiMET_phiJERUp;   //!
   TBranch        *b_PuppiMET_phiJESDown;   //!
   TBranch        *b_PuppiMET_phiJESUp;   //!
   TBranch        *b_PuppiMET_phiUnclusteredDown;   //!
   TBranch        *b_PuppiMET_phiUnclusteredUp;   //!
   TBranch        *b_PuppiMET_pt;   //!
   TBranch        *b_PuppiMET_ptJERDown;   //!
   TBranch        *b_PuppiMET_ptJERUp;   //!
   TBranch        *b_PuppiMET_ptJESDown;   //!
   TBranch        *b_PuppiMET_ptJESUp;   //!
   TBranch        *b_PuppiMET_ptUnclusteredDown;   //!
   TBranch        *b_PuppiMET_ptUnclusteredUp;   //!
   TBranch        *b_PuppiMET_sumEt;   //!
   TBranch        *b_RawMET_phi;   //!
   TBranch        *b_RawMET_pt;   //!
   TBranch        *b_RawMET_sumEt;   //!
   TBranch        *b_RawPuppiMET_phi;   //!
   TBranch        *b_RawPuppiMET_pt;   //!
   TBranch        *b_RawPuppiMET_sumEt;   //!
   TBranch        *b_fixedGridRhoFastjetAll;   //!
   TBranch        *b_fixedGridRhoFastjetCentral;   //!
   TBranch        *b_fixedGridRhoFastjetCentralCalo;   //!
   TBranch        *b_fixedGridRhoFastjetCentralChargedPileUp;   //!
   TBranch        *b_fixedGridRhoFastjetCentralNeutral;   //!
   TBranch        *b_nGenDressedLepton;   //!
   TBranch        *b_GenDressedLepton_eta;   //!
   TBranch        *b_GenDressedLepton_mass;   //!
   TBranch        *b_GenDressedLepton_phi;   //!
   TBranch        *b_GenDressedLepton_pt;   //!
   TBranch        *b_GenDressedLepton_pdgId;   //!
   TBranch        *b_GenDressedLepton_hasTauAnc;   //!
   TBranch        *b_nGenIsolatedPhoton;   //!
   TBranch        *b_GenIsolatedPhoton_eta;   //!
   TBranch        *b_GenIsolatedPhoton_mass;   //!
   TBranch        *b_GenIsolatedPhoton_phi;   //!
   TBranch        *b_GenIsolatedPhoton_pt;   //!
   TBranch        *b_PV_ndof;   //!
   TBranch        *b_PV_x;   //!
   TBranch        *b_PV_y;   //!
   TBranch        *b_PV_z;   //!
   TBranch        *b_PV_chi2;   //!
   TBranch        *b_PV_score;   //!
   TBranch        *b_PV_npvs;   //!
   TBranch        *b_PV_npvsGood;   //!
   TBranch        *b_Electron_genPartIdx;   //!
   TBranch        *b_Electron_genPartFlav;   //!
   TBranch        *b_GenJet_partonFlavour;   //!
   TBranch        *b_GenJet_hadronFlavour;   //!
   TBranch        *b_GenVtx_t0;   //!
   TBranch        *b_Jet_genJetIdx;   //!
   TBranch        *b_Jet_hadronFlavour;   //!
   TBranch        *b_Jet_partonFlavour;   //!
   TBranch        *b_Muon_genPartIdx;   //!
   TBranch        *b_Muon_genPartFlav;   //!
   TBranch        *b_Photon_genPartIdx;   //!
   TBranch        *b_Photon_genPartFlav;   //!
   TBranch        *b_MET_fiducialGenPhi;   //!
   TBranch        *b_MET_fiducialGenPt;   //!
   TBranch        *b_Electron_cleanmask;   //!
   TBranch        *b_Jet_cleanmask;   //!
   TBranch        *b_Muon_cleanmask;   //!
   TBranch        *b_Photon_cleanmask;   //!
   TBranch        *b_Flag_HBHENoiseFilter;   //!
   TBranch        *b_Flag_HBHENoiseIsoFilter;   //!
   TBranch        *b_Flag_CSCTightHaloFilter;   //!
   TBranch        *b_Flag_CSCTightHaloTrkMuUnvetoFilter;   //!
   TBranch        *b_Flag_CSCTightHalo2015Filter;   //!
   TBranch        *b_Flag_globalTightHalo2016Filter;   //!
   TBranch        *b_Flag_globalSuperTightHalo2016Filter;   //!
   TBranch        *b_Flag_HcalStripHaloFilter;   //!
   TBranch        *b_Flag_hcalLaserEventFilter;   //!
   TBranch        *b_Flag_EcalDeadCellTriggerPrimitiveFilter;   //!
   TBranch        *b_Flag_EcalDeadCellBoundaryEnergyFilter;   //!
   TBranch        *b_Flag_ecalBadCalibFilter;   //!
   TBranch        *b_Flag_goodVertices;   //!
   TBranch        *b_Flag_eeBadScFilter;   //!
   TBranch        *b_Flag_ecalLaserCorrFilter;   //!
   TBranch        *b_Flag_trkPOGFilters;   //!
   TBranch        *b_Flag_chargedHadronTrackResolutionFilter;   //!
   TBranch        *b_Flag_muonBadTrackFilter;   //!
   TBranch        *b_Flag_BadChargedCandidateFilter;   //!
   TBranch        *b_Flag_BadPFMuonFilter;   //!
   TBranch        *b_Flag_BadPFMuonDzFilter;   //!
   TBranch        *b_Flag_hfNoisyHitsFilter;   //!
   TBranch        *b_Flag_BadChargedCandidateSummer16Filter;   //!
   TBranch        *b_Flag_BadPFMuonSummer16Filter;   //!
   TBranch        *b_Flag_trkPOG_manystripclus53X;   //!
   TBranch        *b_Flag_trkPOG_toomanystripclus53X;   //!
   TBranch        *b_Flag_trkPOG_logErrorTooManyClusters;   //!
   TBranch        *b_Flag_METFilters;   //!
   TBranch        *b_HLT_Ele27_Ele37_CaloIdL_MW;   //!
   TBranch        *b_HLT_Mu27_Ele37_CaloIdL_MW;   //!
   TBranch        *b_HLT_Mu37_Ele27_CaloIdL_MW;   //!
   TBranch        *b_HLT_Mu37_TkMu27;   //!
   TBranch        *b_HLT_DoublePhoton33_CaloIdL;   //!
   TBranch        *b_HLT_DoublePhoton70;   //!
   TBranch        *b_HLT_DoublePhoton85;   //!
   TBranch        *b_HLT_Ele27_WPTight_Gsf;   //!
   TBranch        *b_HLT_Ele28_WPTight_Gsf;   //!
   TBranch        *b_HLT_Ele30_WPTight_Gsf;   //!
   TBranch        *b_HLT_Ele32_WPTight_Gsf;   //!
   TBranch        *b_HLT_Ele35_WPTight_Gsf;   //!
   TBranch        *b_HLT_Ele35_WPTight_Gsf_L1EGMT;   //!
   TBranch        *b_HLT_Ele38_WPTight_Gsf;   //!
   TBranch        *b_HLT_Ele40_WPTight_Gsf;   //!
   TBranch        *b_HLT_Ele32_WPTight_Gsf_L1DoubleEG;   //!
   TBranch        *b_HLT_Ele24_eta2p1_WPTight_Gsf_LooseChargedIsoPFTauHPS30_eta2p1_CrossL1;   //!
   TBranch        *b_HLT_Ele24_eta2p1_WPTight_Gsf_MediumChargedIsoPFTauHPS30_eta2p1_CrossL1;   //!
   TBranch        *b_HLT_Ele24_eta2p1_WPTight_Gsf_TightChargedIsoPFTauHPS30_eta2p1_CrossL1;   //!
   TBranch        *b_HLT_Ele24_eta2p1_WPTight_Gsf_LooseChargedIsoPFTauHPS30_eta2p1_TightID_CrossL1;   //!
   TBranch        *b_HLT_Ele24_eta2p1_WPTight_Gsf_MediumChargedIsoPFTauHPS30_eta2p1_TightID_CrossL1;   //!
   TBranch        *b_HLT_Ele24_eta2p1_WPTight_Gsf_TightChargedIsoPFTauHPS30_eta2p1_TightID_CrossL1;   //!
   TBranch        *b_HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL;   //!
   TBranch        *b_HLT_Mu19_TrkIsoVVL_Mu9_TrkIsoVVL;   //!
   TBranch        *b_HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ;   //!
   TBranch        *b_HLT_Mu19_TrkIsoVVL_Mu9_TrkIsoVVL_DZ;   //!
   TBranch        *b_HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass8;   //!
   TBranch        *b_HLT_Mu19_TrkIsoVVL_Mu9_TrkIsoVVL_DZ_Mass8;   //!
   TBranch        *b_HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass3p8;   //!
   TBranch        *b_HLT_Mu19_TrkIsoVVL_Mu9_TrkIsoVVL_DZ_Mass3p8;   //!
   TBranch        *b_HLT_PFJet15;   //!
   TBranch        *b_HLT_PFJet25;   //!
   TBranch        *b_HLT_PFJet40;   //!
   TBranch        *b_HLT_PFJet60;   //!
   TBranch        *b_HLT_PFJet80;   //!
   TBranch        *b_HLT_PFJet140;   //!
   TBranch        *b_HLT_PFJet200;   //!
   TBranch        *b_HLT_PFJet260;   //!
   TBranch        *b_HLT_PFJet320;   //!
   TBranch        *b_HLT_PFJet400;   //!
   TBranch        *b_HLT_PFJet450;   //!
   TBranch        *b_HLT_PFJet500;   //!
   TBranch        *b_HLT_PFJet550;   //!
   TBranch        *b_HLT_PFJetFwd15;   //!
   TBranch        *b_HLT_PFJetFwd25;   //!
   TBranch        *b_HLT_PFJetFwd40;   //!
   TBranch        *b_HLT_PFJetFwd60;   //!
   TBranch        *b_HLT_PFJetFwd80;   //!
   TBranch        *b_HLT_PFJetFwd140;   //!
   TBranch        *b_HLT_PFJetFwd200;   //!
   TBranch        *b_HLT_PFJetFwd260;   //!
   TBranch        *b_HLT_PFJetFwd320;   //!
   TBranch        *b_HLT_PFJetFwd400;   //!
   TBranch        *b_HLT_PFJetFwd450;   //!
   TBranch        *b_HLT_PFJetFwd500;   //!
   TBranch        *b_HLT_PFMET110_PFMHT110_IDTight;   //!
   TBranch        *b_HLT_PFMET120_PFMHT120_IDTight;   //!
   TBranch        *b_HLT_PFMET130_PFMHT130_IDTight;   //!
   TBranch        *b_HLT_PFMET140_PFMHT140_IDTight;   //!
   TBranch        *b_HLT_PFMET100_PFMHT100_IDTight_CaloBTagDeepCSV_3p1;   //!
   TBranch        *b_HLT_PFMET110_PFMHT110_IDTight_CaloBTagDeepCSV_3p1;   //!
   TBranch        *b_HLT_PFMET120_PFMHT120_IDTight_CaloBTagDeepCSV_3p1;   //!
   TBranch        *b_HLT_PFMET130_PFMHT130_IDTight_CaloBTagDeepCSV_3p1;   //!
   TBranch        *b_HLT_PFMET140_PFMHT140_IDTight_CaloBTagDeepCSV_3p1;   //!
   TBranch        *b_HLT_PFMET120_PFMHT120_IDTight_PFHT60;   //!
   TBranch        *b_HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_PFHT60;   //!
   TBranch        *b_HLT_PFMETTypeOne120_PFMHT120_IDTight_PFHT60;   //!
   TBranch        *b_HLT_PFMETTypeOne110_PFMHT110_IDTight;   //!
   TBranch        *b_HLT_PFMETTypeOne120_PFMHT120_IDTight;   //!
   TBranch        *b_HLT_PFMETTypeOne130_PFMHT130_IDTight;   //!
   TBranch        *b_HLT_PFMETTypeOne140_PFMHT140_IDTight;   //!
   TBranch        *b_HLT_PFMETNoMu110_PFMHTNoMu110_IDTight;   //!
   TBranch        *b_HLT_PFMETNoMu120_PFMHTNoMu120_IDTight;   //!
   TBranch        *b_HLT_PFMETNoMu130_PFMHTNoMu130_IDTight;   //!
   TBranch        *b_HLT_PFMETNoMu140_PFMHTNoMu140_IDTight;   //!
   TBranch        *b_HLT_PFMET200_NotCleaned;   //!
   TBranch        *b_HLT_PFMET200_HBHECleaned;   //!
   TBranch        *b_HLT_PFMET250_HBHECleaned;   //!
   TBranch        *b_HLT_PFMET300_HBHECleaned;   //!
   TBranch        *b_HLT_PFMET200_HBHE_BeamHaloCleaned;   //!
   TBranch        *b_HLT_PFMETTypeOne200_HBHE_BeamHaloCleaned;   //!
   TBranch        *b_HLT_MET105_IsoTrk50;   //!
   TBranch        *b_HLT_MET120_IsoTrk50;   //!
   TBranch        *b_HLT_DoublePFJets40_CaloBTagDeepCSV_p71;   //!
   TBranch        *b_HLT_DoublePFJets100_CaloBTagDeepCSV_p71;   //!
   TBranch        *b_HLT_DoublePFJets200_CaloBTagDeepCSV_p71;   //!
   TBranch        *b_HLT_DoublePFJets350_CaloBTagDeepCSV_p71;   //!
   TBranch        *b_HLT_DoublePFJets116MaxDeta1p6_DoubleCaloBTagDeepCSV_p71;   //!
   TBranch        *b_HLT_DoublePFJets128MaxDeta1p6_DoubleCaloBTagDeepCSV_p71;   //!
   TBranch        *b_HLT_Photon300_NoHE;   //!
   TBranch        *b_HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ;   //!
   TBranch        *b_HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ_PFDiJet30;   //!
   TBranch        *b_HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ_CaloDiJet30;   //!
   TBranch        *b_HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ_PFDiJet30_PFBtagDeepCSV_1p5;   //!
   TBranch        *b_HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ_CaloDiJet30_CaloBtagDeepCSV_1p5;   //!
   TBranch        *b_HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL;   //!
   TBranch        *b_HLT_Ele15_Ele8_CaloIdL_TrackIdL_IsoVL;   //!
   TBranch        *b_HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ;   //!
   TBranch        *b_HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL;   //!
   TBranch        *b_HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_DZ;   //!
   TBranch        *b_HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL;   //!
   TBranch        *b_HLT_Mu12_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL;   //!
   TBranch        *b_HLT_Mu12_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ;   //!
   TBranch        *b_HLT_Photon20;   //!
   TBranch        *b_HLT_Photon33;   //!
   TBranch        *b_HLT_Photon50;   //!
   TBranch        *b_HLT_Photon75;   //!
   TBranch        *b_HLT_Photon90;   //!
   TBranch        *b_HLT_Photon120;   //!
   TBranch        *b_HLT_Photon150;   //!
   TBranch        *b_HLT_Photon175;   //!
   TBranch        *b_HLT_Photon200;   //!
   TBranch        *b_HLT_Photon100EB_TightID_TightIso;   //!
   TBranch        *b_HLT_Photon110EB_TightID_TightIso;   //!
   TBranch        *b_HLT_Photon120EB_TightID_TightIso;   //!
   TBranch        *b_HLT_Photon100EBHE10;   //!
   TBranch        *b_HLT_Photon100EEHE10;   //!
   TBranch        *b_HLT_Photon100EE_TightID_TightIso;   //!
   TBranch        *b_HLT_Photon50_R9Id90_HE10_IsoM;   //!
   TBranch        *b_HLT_Photon75_R9Id90_HE10_IsoM;   //!
   TBranch        *b_HLT_Photon75_R9Id90_HE10_IsoM_EBOnly_CaloMJJ300_PFJetsMJJ400DEta3;   //!
   TBranch        *b_HLT_Photon75_R9Id90_HE10_IsoM_EBOnly_CaloMJJ400_PFJetsMJJ600DEta3;   //!
   TBranch        *b_HLT_Photon90_R9Id90_HE10_IsoM;   //!
   TBranch        *b_HLT_Photon120_R9Id90_HE10_IsoM;   //!
   TBranch        *b_HLT_Photon165_R9Id90_HE10_IsoM;   //!
   TBranch        *b_HLT_Photon90_CaloIdL_PFHT700;   //!
   TBranch        *b_HLT_Diphoton30_22_R9Id_OR_IsoCaloId_AND_HE_R9Id_Mass90;   //!
   TBranch        *b_HLT_Diphoton30_22_R9Id_OR_IsoCaloId_AND_HE_R9Id_Mass95;   //!
   TBranch        *b_HLT_Diphoton30PV_18PV_R9Id_AND_IsoCaloId_AND_HE_R9Id_PixelVeto_Mass55;   //!
   TBranch        *b_HLT_Diphoton30PV_18PV_R9Id_AND_IsoCaloId_AND_HE_R9Id_NoPixelVeto_Mass55;   //!
   TBranch        *b_HLT_Photon35_TwoProngs35;   //!
   TBranch        *b_HLT_Ele30_eta2p1_WPTight_Gsf_CentralPFJet35_EleCleaned;   //!
   TBranch        *b_HLT_Ele28_eta2p1_WPTight_Gsf_HT150;   //!
   TBranch        *b_HLT_Ele28_HighEta_SC20_Mass55;   //!
   TBranch        *b_HLT_DoubleIsoMu20_eta2p1;   //!
   TBranch        *b_HLT_Ele23_CaloIdL_TrackIdL_IsoVL_PFJet30;   //!
   TBranch        *b_HLT_Ele23_CaloIdM_TrackIdM_PFJet30;   //!
   TBranch        *b_HLT_Ele50_CaloIdVT_GsfTrkIdT_PFJet165;   //!
   TBranch        *b_HLT_Ele115_CaloIdVT_GsfTrkIdT;   //!
   TBranch        *b_HLT_Ele135_CaloIdVT_GsfTrkIdT;   //!
   TBranch        *b_HLT_Ele145_CaloIdVT_GsfTrkIdT;   //!
   TBranch        *b_HLT_Ele200_CaloIdVT_GsfTrkIdT;   //!
   TBranch        *b_HLT_Ele250_CaloIdVT_GsfTrkIdT;   //!
   TBranch        *b_HLT_Ele300_CaloIdVT_GsfTrkIdT;   //!
   TBranch        *b_HLT_Photon60_R9Id90_CaloIdL_IsoL;   //!
   TBranch        *b_HLT_Photon60_R9Id90_CaloIdL_IsoL_DisplacedIdL;   //!
   TBranch        *b_HLT_Photon60_R9Id90_CaloIdL_IsoL_DisplacedIdL_PFHT350MinPFJet15;   //!
   TBranch        *b_HLT_AK4PFJet30;   //!
   TBranch        *b_HLT_AK4PFJet50;   //!
   TBranch        *b_HLT_AK4PFJet80;   //!
   TBranch        *b_HLT_AK4PFJet100;   //!
   TBranch        *b_HLT_AK4PFJet120;   //!
   TBranch        *b_HLT_SinglePhoton10_Eta3p1ForPPRef;   //!
   TBranch        *b_HLT_SinglePhoton20_Eta3p1ForPPRef;   //!
   TBranch        *b_HLT_SinglePhoton30_Eta3p1ForPPRef;   //!
   TBranch        *b_HLT_Photon20_HoverELoose;   //!
   TBranch        *b_HLT_Photon30_HoverELoose;   //!
   TBranch        *b_HLT_Photon50_R9Id90_HE10_IsoM_EBOnly_PFJetsMJJ300DEta3_PFMET50;   //!
   TBranch        *b_HLT_Photon75_R9Id90_HE10_IsoM_EBOnly_PFJetsMJJ300DEta3;   //!
   TBranch        *b_HLT_Photon75_R9Id90_HE10_IsoM_EBOnly_PFJetsMJJ600DEta3;   //!
   TBranch        *b_HLT_PFMET100_PFMHT100_IDTight_PFHT60;   //!
   TBranch        *b_HLT_PFMETNoMu100_PFMHTNoMu100_IDTight_PFHT60;   //!
   TBranch        *b_HLT_PFMETTypeOne100_PFMHT100_IDTight_PFHT60;   //!
   TBranch        *b_HLT_Diphoton30_18_R9IdL_AND_HE_AND_IsoCaloId_NoPixelVeto_Mass55;   //!
   TBranch        *b_HLT_Diphoton30_18_R9IdL_AND_HE_AND_IsoCaloId_NoPixelVeto;   //!
   TBranch        *b_Muon_corrected_pt;   //!
   TBranch        *b_Muon_correctedUp_pt;   //!
   TBranch        *b_Muon_correctedDown_pt;   //!
   TBranch        *b_nLepton;   //!
   TBranch        *b_Lepton_jetPtRelv2;   //!
   TBranch        *b_Lepton_jetNDauCharged;   //!
   TBranch        *b_Lepton_jetIdx;   //!
   TBranch        *b_Lepton_photonIdx;   //!
   TBranch        *b_Lepton_jetRelIso;   //!
   TBranch        *b_nlepton;   //!
   TBranch        *b_lepton_mass;   //!
   TBranch        *b_lepton_isStandalone;   //!
   TBranch        *b_lepton_mvaFall17V2noIso_WPL;   //!
   TBranch        *b_lepton_dxyErr;   //!
   TBranch        *b_lepton_dr03TkSumPt;   //!
   TBranch        *b_lepton_mediumPromptId;   //!
   TBranch        *b_lepton_mvaLowPt;   //!
   TBranch        *b_lepton_mvaLowPtId;   //!
   TBranch        *b_lepton_mediumId;   //!
   TBranch        *b_lepton_mvaId;   //!
   TBranch        *b_lepton_dxy;   //!
   TBranch        *b_lepton_softMva;   //!
   TBranch        *b_lepton_energyErr;   //!
   TBranch        *b_lepton_pdgId;   //!
   TBranch        *b_lepton_inTimeMuon;   //!
   TBranch        *b_lepton_genPartFlav;   //!
   TBranch        *b_lepton_triggerIdLoose;   //!
   TBranch        *b_lepton_hoe;   //!
   TBranch        *b_lepton_miniPFRelIso_all;   //!
   TBranch        *b_lepton_correctedUp_pt;   //!
   TBranch        *b_lepton_sip3d;   //!
   TBranch        *b_lepton_nTrackerLayers;   //!
   TBranch        *b_lepton_tkIsoId;   //!
   TBranch        *b_lepton_miniIsoId;   //!
   TBranch        *b_lepton_multiIsoId;   //!
   TBranch        *b_lepton_pfRelIso03_all;   //!
   TBranch        *b_lepton_dr03EcalRecHitSumEt;   //!
   TBranch        *b_lepton_highPtId;   //!
   TBranch        *b_lepton_softMvaId;   //!
   TBranch        *b_lepton_isGlobal;   //!
   TBranch        *b_lepton_miniPFRelIso_chg;   //!
   TBranch        *b_lepton_jetRelIso;   //!
   TBranch        *b_lepton_genPartIdx;   //!
   TBranch        *b_lepton_vidNestedWPBitmap;   //!
   TBranch        *b_lepton_ip3d;   //!
   TBranch        *b_lepton_jetPtRelv2;   //!
   TBranch        *b_lepton_photonIdx;   //!
   TBranch        *b_lepton_pfRelIso03_chg;   //!
   TBranch        *b_lepton_mvaFall17V2Iso;   //!
   TBranch        *b_lepton_lostHits;   //!
   TBranch        *b_lepton_dr03TkSumPtHEEP;   //!
   TBranch        *b_lepton_tunepRelPt;   //!
   TBranch        *b_lepton_dxybs;   //!
   TBranch        *b_lepton_cutBased_HEEP;   //!
   TBranch        *b_lepton_pt;   //!
   TBranch        *b_lepton_softId;   //!
   TBranch        *b_lepton_mvaFall17V2noIso_WP90;   //!
   TBranch        *b_lepton_puppiIsoId;   //!
   TBranch        *b_lepton_mvaTTH;   //!
   TBranch        *b_lepton_charge;   //!
   TBranch        *b_lepton_vidNestedWPBitmapHEEP;   //!
   TBranch        *b_lepton_isPFcand;   //!
   TBranch        *b_lepton_correctedDown_pt;   //!
   TBranch        *b_lepton_sieie;   //!
   TBranch        *b_lepton_dr03HcalDepth1TowerSumEt;   //!
   TBranch        *b_lepton_highPurity;   //!
   TBranch        *b_lepton_dz;   //!
   TBranch        *b_lepton_scEtOverPt;   //!
   TBranch        *b_lepton_isTracker;   //!
   TBranch        *b_lepton_tkRelIso;   //!
   TBranch        *b_lepton_tightId;   //!
   TBranch        *b_lepton_pfRelIso04_all;   //!
   TBranch        *b_lepton_dzErr;   //!
   TBranch        *b_lepton_phi;   //!
   TBranch        *b_lepton_pfIsoId;   //!
   TBranch        *b_lepton_jetNDauCharged;   //!
   TBranch        *b_lepton_eInvMinusPInv;   //!
   TBranch        *b_lepton_dEsigmaDown;   //!
   TBranch        *b_lepton_mvaFall17V2noIso;   //!
   TBranch        *b_lepton_tightCharge;   //!
   TBranch        *b_lepton_fsrPhotonIdx;   //!
   TBranch        *b_lepton_eCorr;   //!
   TBranch        *b_lepton_mvaFall17V2Iso_WP80;   //!
   TBranch        *b_lepton_looseId;   //!
   TBranch        *b_lepton_cutBased;   //!
   TBranch        *b_lepton_dEscaleUp;   //!
   TBranch        *b_lepton_mvaFall17V2Iso_WPL;   //!
   TBranch        *b_lepton_ptErr;   //!
   TBranch        *b_lepton_deltaEtaSC;   //!
   TBranch        *b_lepton_r9;   //!
   TBranch        *b_lepton_seedGain;   //!
   TBranch        *b_lepton_eta;   //!
   TBranch        *b_lepton_mvaFall17V2Iso_WP90;   //!
   TBranch        *b_lepton_nStations;   //!
   TBranch        *b_lepton_corrected_pt;   //!
   TBranch        *b_lepton_dEsigmaUp;   //!
   TBranch        *b_lepton_jetIdx;   //!
   TBranch        *b_lepton_dEscaleDown;   //!
   TBranch        *b_lepton_mvaFall17V2noIso_WP80;   //!
   TBranch        *b_lepton_convVeto;   //!
   TBranch        *b_lepton_cleanmask;   //!
   TBranch        *b_lepton_segmentComp;   //!
   TBranch        *b_lepton_is_tight;   //!
   TBranch        *b_lepton_is_real;   //!
   TBranch        *b_nphoton;   //!
   TBranch        *b_photon_electronVeto;   //!
   TBranch        *b_photon_genPartIdx;   //!
   TBranch        *b_photon_genPartFlav;   //!
   TBranch        *b_photon_cleanmask;   //!
   TBranch        *b_photon_cutBased;   //!
   TBranch        *b_photon_cutBased_Fall17V1Bitmap;   //!
   TBranch        *b_photon_jetIdx;   //!
   TBranch        *b_photon_vidNestedWPBitmap;   //!
   TBranch        *b_photon_dEscaleUp;   //!
   TBranch        *b_photon_dEsigmaDown;   //!
   TBranch        *b_photon_eta;   //!
   TBranch        *b_photon_hoe;   //!
   TBranch        *b_photon_mass;   //!
   TBranch        *b_photon_mvaID;   //!
   TBranch        *b_photon_mvaID_Fall17V1p1;   //!
   TBranch        *b_photon_pfRelIso03_all;   //!
   TBranch        *b_photon_pfRelIso03_chg;   //!
   TBranch        *b_photon_dEscaleDown;   //!
   TBranch        *b_photon_electronIdx;   //!
   TBranch        *b_photon_pdgId;   //!
   TBranch        *b_photon_dEsigmaUp;   //!
   TBranch        *b_photon_eCorr;   //!
   TBranch        *b_photon_energyErr;   //!
   TBranch        *b_photon_phi;   //!
   TBranch        *b_photon_pt;   //!
   TBranch        *b_photon_r9;   //!
   TBranch        *b_photon_sieie;   //!
   TBranch        *b_photon_charge;   //!
   TBranch        *b_photon_isScEtaEB;   //!
   TBranch        *b_photon_mvaID_WP80;   //!
   TBranch        *b_photon_mvaID_WP90;   //!
   TBranch        *b_photon_pixelSeed;   //!
   TBranch        *b_photon_seedGain;   //!
   TBranch        *b_photon_isScEtaEE;   //!
   TBranch        *b_njet;   //!
   TBranch        *b_jet_area;   //!
   TBranch        *b_jet_btagCSVV2;   //!
   TBranch        *b_jet_btagDeepB;   //!
   TBranch        *b_jet_btagDeepCvB;   //!
   TBranch        *b_jet_btagDeepCvL;   //!
   TBranch        *b_jet_btagDeepFlavB;   //!
   TBranch        *b_jet_btagDeepFlavCvB;   //!
   TBranch        *b_jet_btagDeepFlavCvL;   //!
   TBranch        *b_jet_btagDeepFlavQG;   //!
   TBranch        *b_jet_chEmEF;   //!
   TBranch        *b_jet_chFPV0EF;   //!
   TBranch        *b_jet_chHEF;   //!
   TBranch        *b_jet_eta;   //!
   TBranch        *b_jet_hfsigmaEtaEta;   //!
   TBranch        *b_jet_hfsigmaPhiPhi;   //!
   TBranch        *b_jet_mass;   //!
   TBranch        *b_jet_muEF;   //!
   TBranch        *b_jet_muonSubtrFactor;   //!
   TBranch        *b_jet_neEmEF;   //!
   TBranch        *b_jet_neHEF;   //!
   TBranch        *b_jet_phi;   //!
   TBranch        *b_jet_pt;   //!
   TBranch        *b_jet_puIdDisc;   //!
   TBranch        *b_jet_qgl;   //!
   TBranch        *b_jet_rawFactor;   //!
   TBranch        *b_jet_bRegCorr;   //!
   TBranch        *b_jet_partonFlavour;   //!
   TBranch        *b_jet_bRegRes;   //!
   TBranch        *b_jet_cRegCorr;   //!
   TBranch        *b_jet_electronIdx1;   //!
   TBranch        *b_jet_electronIdx2;   //!
   TBranch        *b_jet_hfadjacentEtaStripsSize;   //!
   TBranch        *b_jet_hfcentralEtaStripSize;   //!
   TBranch        *b_jet_muonIdx1;   //!
   TBranch        *b_jet_nMuons;   //!
   TBranch        *b_jet_puId;   //!
   TBranch        *b_jet_cleanmask;   //!
   TBranch        *b_jet_genJetIdx;   //!
   TBranch        *b_jet_hadronFlavour;   //!
   TBranch        *b_jet_cRegRes;   //!
   TBranch        *b_jet_jetId;   //!
   TBranch        *b_jet_muonIdx2;   //!
   TBranch        *b_jet_nElectrons;   //!
   TBranch        *b_jet_nConstituents;   //!
   TBranch        *b_Jet_pt_raw;   //!
   TBranch        *b_Jet_pt_nom;   //!
   TBranch        *b_Jet_mass_raw;   //!
   TBranch        *b_Jet_mass_nom;   //!
   TBranch        *b_Jet_corr_JEC;   //!
   TBranch        *b_Jet_corr_JER;   //!
   TBranch        *b_MET_T1_pt;   //!
   TBranch        *b_MET_T1_phi;   //!
   TBranch        *b_MET_T1Smear_pt;   //!
   TBranch        *b_MET_T1Smear_phi;   //!
   TBranch        *b_Jet_pt_jerUp;   //!
   TBranch        *b_Jet_mass_jerUp;   //!
   TBranch        *b_MET_T1_pt_jerUp;   //!
   TBranch        *b_MET_T1_phi_jerUp;   //!
   TBranch        *b_MET_T1Smear_pt_jerUp;   //!
   TBranch        *b_MET_T1Smear_phi_jerUp;   //!
   TBranch        *b_Jet_pt_jesTotalUp;   //!
   TBranch        *b_Jet_mass_jesTotalUp;   //!
   TBranch        *b_MET_T1_pt_jesTotalUp;   //!
   TBranch        *b_MET_T1_phi_jesTotalUp;   //!
   TBranch        *b_MET_T1Smear_pt_jesTotalUp;   //!
   TBranch        *b_MET_T1Smear_phi_jesTotalUp;   //!
   TBranch        *b_Jet_pt_jesHEMIssueUp;   //!
   TBranch        *b_Jet_mass_jesHEMIssueUp;   //!
   TBranch        *b_MET_T1_pt_jesHEMIssueUp;   //!
   TBranch        *b_MET_T1_phi_jesHEMIssueUp;   //!
   TBranch        *b_MET_T1Smear_pt_jesHEMIssueUp;   //!
   TBranch        *b_MET_T1Smear_phi_jesHEMIssueUp;   //!
   TBranch        *b_MET_T1_pt_unclustEnUp;   //!
   TBranch        *b_MET_T1_phi_unclustEnUp;   //!
   TBranch        *b_MET_T1Smear_pt_unclustEnUp;   //!
   TBranch        *b_MET_T1Smear_phi_unclustEnUp;   //!
   TBranch        *b_Jet_pt_jerDown;   //!
   TBranch        *b_Jet_mass_jerDown;   //!
   TBranch        *b_MET_T1_pt_jerDown;   //!
   TBranch        *b_MET_T1_phi_jerDown;   //!
   TBranch        *b_MET_T1Smear_pt_jerDown;   //!
   TBranch        *b_MET_T1Smear_phi_jerDown;   //!
   TBranch        *b_Jet_pt_jesTotalDown;   //!
   TBranch        *b_Jet_mass_jesTotalDown;   //!
   TBranch        *b_MET_T1_pt_jesTotalDown;   //!
   TBranch        *b_MET_T1_phi_jesTotalDown;   //!
   TBranch        *b_MET_T1Smear_pt_jesTotalDown;   //!
   TBranch        *b_MET_T1Smear_phi_jesTotalDown;   //!
   TBranch        *b_Jet_pt_jesHEMIssueDown;   //!
   TBranch        *b_Jet_mass_jesHEMIssueDown;   //!
   TBranch        *b_MET_T1_pt_jesHEMIssueDown;   //!
   TBranch        *b_MET_T1_phi_jesHEMIssueDown;   //!
   TBranch        *b_MET_T1Smear_pt_jesHEMIssueDown;   //!
   TBranch        *b_MET_T1Smear_phi_jesHEMIssueDown;   //!
   TBranch        *b_MET_T1_pt_unclustEnDown;   //!
   TBranch        *b_MET_T1_phi_unclustEnDown;   //!
   TBranch        *b_MET_T1Smear_pt_unclustEnDown;   //!
   TBranch        *b_MET_T1Smear_phi_unclustEnDown;   //!
   TBranch        *b_lumi;   //!
   TBranch        *b_photon_selection;   //!
   TBranch        *b_n_loose_mu;   //!
   TBranch        *b_n_loose_ele;   //!
   TBranch        *b_n_tight_mu;   //!
   TBranch        *b_n_tight_ele;   //!
   TBranch        *b_HEM_issue;   //!
   TBranch        *b_mjj; 
   TBranch        *b_n_leptons;   //!
   TBranch        *b_jet_pt_jesTotalDown;   //!
   TBranch        *b_jet_mass_jesTotalDown;   //!
   TBranch        *b_jet_pt_jesHEMIssueDown;   //!
   TBranch        *b_jet_mass_jesHEMIssueDown;   //!
   TBranch        *b_jet_pt_jerDown;   //!
   TBranch        *b_jet_mass_jerDown;   //!
   TBranch        *b_jet_pt_jesTotalUp;   //!
   TBranch        *b_jet_mass_jesTotalUp;   //!
   TBranch        *b_jet_pt_jesHEMIssueUp;   //!
   TBranch        *b_jet_pt_jerUp;   //!
   TBranch        *b_jet_mass_jerUp;   //!
   TBranch        *b_jet_mass_jesHEMIssueUp;   //!
   TBranch        *b_jet_pt_raw;   //!
   TBranch        *b_jet_pt_nom;   //!
   TBranch        *b_jet_mass_raw;   //!
   TBranch        *b_jet_corr_JEC;   //!
   TBranch        *b_jet_mass_nom;   //!
   TBranch        *b_jet_corr_JER;   //!
   TBranch        *b_n_photon;   //!
   TBranch        *b_photon_isprompt;   //!
   TBranch        *b_photon_gen_matching;   //!
   TBranch        *b_npu;   //!
   TBranch        *b_ntruepu;   //!
   TBranch        *b_npvs;   //!
   TBranch        *b_njets25;   //!
   TBranch        *b_puWeight;   //!
   TBranch        *b_puWeightUp;   //!
   TBranch        *b_puWeightDown;   //!

   TString m_dataset; 
   chain_ajj(TTree *tree=0,TString dataset="");
   virtual ~chain_ajj();
   virtual Int_t    Cut(Long64_t entry);
   virtual Int_t    GetEntry(Long64_t entry);
   virtual Long64_t LoadTree(Long64_t entry);
   virtual void     Init(TTree *tree);
   virtual void     Loop(TString name,float nevents,float xs, TString year);
   virtual Float_t  delta_R(Float_t eta1, Float_t phi1, Float_t eta2, Float_t phi2);
   virtual void     set_init();
   virtual void     endJob();
   virtual Bool_t   Notify();
   virtual void     Show(Long64_t entry = -1);

 private:
   TTree *ExTree;TFile *fout;
   float channel;
   float lep1pt,lep2pt,lep1eta,lep2eta,lep1phi,lep2phi,lep1mass,lep2mass;
   float photonet,photoneta,photonphi,photonmass;
   float Vpt,Veta,Vphi,Vmass;
   float jet1pt,jet2pt,jet1eta,jet2eta,jet1phi,jet2phi,jet1mass,jet2mass;
   float drjj,drj1a,drj2a,drl1a,drl2a,drj1l1,drj1l2,drj2l1,drj2l2;
   float mjj_nom,detajj,zepp,dphi;
   map<TString, float> var;
   vector<TString> treename={"channel","lep1pt","lep2pt","lep1eta","lep2eta","lep1phi","lep2phi","lep1mass","lep2mass","photonet","photoneta","photonphi","photonmass","photonMCmatch","ml1g","ml2g","mllg","Vpt","Veta","Vphi","Vmass","jet1pt","jet2pt","jet1eta","jet2eta","jet1phi","jet2phi","jet1mass","jet2mass","drjj","drj1a","drj2a","drl1a","drl2a","drj1l1","drj1l2","drj2l1","drj2l2","mjj_nom","detajj","zepp","dphi","scalef","photon_id_scale","photon_veto_scale","ele1_id_scale","ele2_id_scale","ele1_reco_scale","ele2_reco_scale","muon1_id_scale","muon2_id_scale","muon1_iso_scale","muon2_iso_scale","HLT_scale"};

};

#endif

#ifdef chain_ajj_cxx
chain_ajj::chain_ajj(TTree *tree,TString dataset) : fChain(0) 
{
// if parameter tree is not specified (or zero), connect the file
// used to generate this class and read the Tree.
   if (tree == 0) {

#ifdef SINGLE_TREE
      // The following code should be used if you want this class to access
      // a single tree instead of a chain
      TFile *f = (TFile*)gROOT->GetListOfFiles()->FindObject("Memory Directory");
      if (!f || !f->IsOpen()) {
         f = new TFile("Memory Directory");
      }
      f->GetObject("Events",tree);

#else // SINGLE_TREE

      // The following code should be used if you want this class to access a chain
      // of trees.
      TChain * chain = new TChain("Events","");
      chain->Add("WW18.root/Events");
      chain->Add("WZ18.root/Events");
      tree = chain;
#endif // SINGLE_TREE

   }
   m_dataset=dataset;
   Init(tree);
}

chain_ajj::~chain_ajj()
{
   if (!fChain) return;
   delete fChain->GetCurrentFile();
}

Int_t chain_ajj::GetEntry(Long64_t entry)
{
// Read contents of entry.
   if (!fChain) return 0;
   return fChain->GetEntry(entry);
}
Long64_t chain_ajj::LoadTree(Long64_t entry)
{
// Set the environment to read one entry
   if (!fChain) return -5;
   Long64_t centry = fChain->LoadTree(entry);
   if (centry < 0) return centry;
   if (fChain->GetTreeNumber() != fCurrent) {
      fCurrent = fChain->GetTreeNumber();
      Notify();
   }
   return centry;
}

void chain_ajj::Init(TTree *tree)
{
   // The Init() function is called when the selector needs to initialize
   // a new tree or chain. Typically here the branch addresses and branch
   // pointers of the tree will be set.
   // It is normally not necessary to make changes to the generated
   // code, but the routine can be extended by the user if needed.
   // Init() will be called many times when running on PROOF
   // (once per file to be processed).

   // Set branch addresses and branch pointers
   if (!tree) return;
   fChain = tree;
   fCurrent = -1;
   fChain->SetMakeClass(1);
//   fout = new TFile("./"+m_dataset, "RECREATE");
   fout = new TFile("/eos/cms/store/user/yian/"+m_dataset, "RECREATE");
   ExTree = fChain->CloneTree(0);
   for(int i=0;i<treename.size();i++){
	   ExTree->Branch(treename[i], &var[treename[i]], treename[i]+"/F");
   }

   fChain->SetBranchAddress("run", &run, &b_run);
   fChain->SetBranchAddress("luminosityBlock", &luminosityBlock, &b_luminosityBlock);
   fChain->SetBranchAddress("event", &event, &b_event);
   fChain->SetBranchAddress("btagWeight_CSVV2", &btagWeight_CSVV2, &b_btagWeight_CSVV2);
   fChain->SetBranchAddress("btagWeight_DeepCSVB", &btagWeight_DeepCSVB, &b_btagWeight_DeepCSVB);
   fChain->SetBranchAddress("nCorrT1METJet", &nCorrT1METJet, &b_nCorrT1METJet);
   fChain->SetBranchAddress("CorrT1METJet_area", CorrT1METJet_area, &b_CorrT1METJet_area);
   fChain->SetBranchAddress("CorrT1METJet_eta", CorrT1METJet_eta, &b_CorrT1METJet_eta);
   fChain->SetBranchAddress("CorrT1METJet_muonSubtrFactor", CorrT1METJet_muonSubtrFactor, &b_CorrT1METJet_muonSubtrFactor);
   fChain->SetBranchAddress("CorrT1METJet_phi", CorrT1METJet_phi, &b_CorrT1METJet_phi);
   fChain->SetBranchAddress("CorrT1METJet_rawPt", CorrT1METJet_rawPt, &b_CorrT1METJet_rawPt);
   fChain->SetBranchAddress("nElectron", &nElectron, &b_nElectron);
   fChain->SetBranchAddress("Electron_dEscaleDown", Electron_dEscaleDown, &b_Electron_dEscaleDown);
   fChain->SetBranchAddress("Electron_dEscaleUp", Electron_dEscaleUp, &b_Electron_dEscaleUp);
   fChain->SetBranchAddress("Electron_dEsigmaDown", Electron_dEsigmaDown, &b_Electron_dEsigmaDown);
   fChain->SetBranchAddress("Electron_dEsigmaUp", Electron_dEsigmaUp, &b_Electron_dEsigmaUp);
   fChain->SetBranchAddress("Electron_deltaEtaSC", Electron_deltaEtaSC, &b_Electron_deltaEtaSC);
   fChain->SetBranchAddress("Electron_dr03EcalRecHitSumEt", Electron_dr03EcalRecHitSumEt, &b_Electron_dr03EcalRecHitSumEt);
   fChain->SetBranchAddress("Electron_dr03HcalDepth1TowerSumEt", Electron_dr03HcalDepth1TowerSumEt, &b_Electron_dr03HcalDepth1TowerSumEt);
   fChain->SetBranchAddress("Electron_dr03TkSumPt", Electron_dr03TkSumPt, &b_Electron_dr03TkSumPt);
   fChain->SetBranchAddress("Electron_dr03TkSumPtHEEP", Electron_dr03TkSumPtHEEP, &b_Electron_dr03TkSumPtHEEP);
   fChain->SetBranchAddress("Electron_dxy", Electron_dxy, &b_Electron_dxy);
   fChain->SetBranchAddress("Electron_dxyErr", Electron_dxyErr, &b_Electron_dxyErr);
   fChain->SetBranchAddress("Electron_dz", Electron_dz, &b_Electron_dz);
   fChain->SetBranchAddress("Electron_dzErr", Electron_dzErr, &b_Electron_dzErr);
   fChain->SetBranchAddress("Electron_eCorr", Electron_eCorr, &b_Electron_eCorr);
   fChain->SetBranchAddress("Electron_eInvMinusPInv", Electron_eInvMinusPInv, &b_Electron_eInvMinusPInv);
   fChain->SetBranchAddress("Electron_energyErr", Electron_energyErr, &b_Electron_energyErr);
   fChain->SetBranchAddress("Electron_eta", Electron_eta, &b_Electron_eta);
   fChain->SetBranchAddress("Electron_hoe", Electron_hoe, &b_Electron_hoe);
   fChain->SetBranchAddress("Electron_ip3d", Electron_ip3d, &b_Electron_ip3d);
   fChain->SetBranchAddress("Electron_jetPtRelv2", Electron_jetPtRelv2, &b_Electron_jetPtRelv2);
   fChain->SetBranchAddress("Electron_jetRelIso", Electron_jetRelIso, &b_Electron_jetRelIso);
   fChain->SetBranchAddress("Electron_mass", Electron_mass, &b_Electron_mass);
   fChain->SetBranchAddress("Electron_miniPFRelIso_all", Electron_miniPFRelIso_all, &b_Electron_miniPFRelIso_all);
   fChain->SetBranchAddress("Electron_miniPFRelIso_chg", Electron_miniPFRelIso_chg, &b_Electron_miniPFRelIso_chg);
   fChain->SetBranchAddress("Electron_mvaFall17V2Iso", Electron_mvaFall17V2Iso, &b_Electron_mvaFall17V2Iso);
   fChain->SetBranchAddress("Electron_mvaFall17V2noIso", Electron_mvaFall17V2noIso, &b_Electron_mvaFall17V2noIso);
   fChain->SetBranchAddress("Electron_pfRelIso03_all", Electron_pfRelIso03_all, &b_Electron_pfRelIso03_all);
   fChain->SetBranchAddress("Electron_pfRelIso03_chg", Electron_pfRelIso03_chg, &b_Electron_pfRelIso03_chg);
   fChain->SetBranchAddress("Electron_phi", Electron_phi, &b_Electron_phi);
   fChain->SetBranchAddress("Electron_pt", Electron_pt, &b_Electron_pt);
   fChain->SetBranchAddress("Electron_r9", Electron_r9, &b_Electron_r9);
   fChain->SetBranchAddress("Electron_scEtOverPt", Electron_scEtOverPt, &b_Electron_scEtOverPt);
   fChain->SetBranchAddress("Electron_sieie", Electron_sieie, &b_Electron_sieie);
   fChain->SetBranchAddress("Electron_sip3d", Electron_sip3d, &b_Electron_sip3d);
   fChain->SetBranchAddress("Electron_mvaTTH", Electron_mvaTTH, &b_Electron_mvaTTH);
   fChain->SetBranchAddress("Electron_charge", Electron_charge, &b_Electron_charge);
   fChain->SetBranchAddress("Electron_cutBased", Electron_cutBased, &b_Electron_cutBased);
   fChain->SetBranchAddress("Electron_jetIdx", Electron_jetIdx, &b_Electron_jetIdx);
   fChain->SetBranchAddress("Electron_pdgId", Electron_pdgId, &b_Electron_pdgId);
   fChain->SetBranchAddress("Electron_photonIdx", Electron_photonIdx, &b_Electron_photonIdx);
   fChain->SetBranchAddress("Electron_tightCharge", Electron_tightCharge, &b_Electron_tightCharge);
   fChain->SetBranchAddress("Electron_vidNestedWPBitmap", Electron_vidNestedWPBitmap, &b_Electron_vidNestedWPBitmap);
   fChain->SetBranchAddress("Electron_vidNestedWPBitmapHEEP", Electron_vidNestedWPBitmapHEEP, &b_Electron_vidNestedWPBitmapHEEP);
   fChain->SetBranchAddress("Electron_convVeto", Electron_convVeto, &b_Electron_convVeto);
   fChain->SetBranchAddress("Electron_cutBased_HEEP", Electron_cutBased_HEEP, &b_Electron_cutBased_HEEP);
   fChain->SetBranchAddress("Electron_isPFcand", Electron_isPFcand, &b_Electron_isPFcand);
   fChain->SetBranchAddress("Electron_jetNDauCharged", Electron_jetNDauCharged, &b_Electron_jetNDauCharged);
   fChain->SetBranchAddress("Electron_lostHits", Electron_lostHits, &b_Electron_lostHits);
   fChain->SetBranchAddress("Electron_mvaFall17V2Iso_WP80", Electron_mvaFall17V2Iso_WP80, &b_Electron_mvaFall17V2Iso_WP80);
   fChain->SetBranchAddress("Electron_mvaFall17V2Iso_WP90", Electron_mvaFall17V2Iso_WP90, &b_Electron_mvaFall17V2Iso_WP90);
   fChain->SetBranchAddress("Electron_mvaFall17V2Iso_WPL", Electron_mvaFall17V2Iso_WPL, &b_Electron_mvaFall17V2Iso_WPL);
   fChain->SetBranchAddress("Electron_mvaFall17V2noIso_WP80", Electron_mvaFall17V2noIso_WP80, &b_Electron_mvaFall17V2noIso_WP80);
   fChain->SetBranchAddress("Electron_mvaFall17V2noIso_WP90", Electron_mvaFall17V2noIso_WP90, &b_Electron_mvaFall17V2noIso_WP90);
   fChain->SetBranchAddress("Electron_mvaFall17V2noIso_WPL", Electron_mvaFall17V2noIso_WPL, &b_Electron_mvaFall17V2noIso_WPL);
   fChain->SetBranchAddress("Electron_seedGain", Electron_seedGain, &b_Electron_seedGain);
   fChain->SetBranchAddress("nFsrPhoton", &nFsrPhoton, &b_nFsrPhoton);
   fChain->SetBranchAddress("nGenJet", &nGenJet, &b_nGenJet);
   fChain->SetBranchAddress("GenJet_eta", GenJet_eta, &b_GenJet_eta);
   fChain->SetBranchAddress("GenJet_mass", GenJet_mass, &b_GenJet_mass);
   fChain->SetBranchAddress("GenJet_phi", GenJet_phi, &b_GenJet_phi);
   fChain->SetBranchAddress("GenJet_pt", GenJet_pt, &b_GenJet_pt);
   fChain->SetBranchAddress("nGenPart", &nGenPart, &b_nGenPart);
   fChain->SetBranchAddress("GenPart_eta", GenPart_eta, &b_GenPart_eta);
   fChain->SetBranchAddress("GenPart_mass", GenPart_mass, &b_GenPart_mass);
   fChain->SetBranchAddress("GenPart_phi", GenPart_phi, &b_GenPart_phi);
   fChain->SetBranchAddress("GenPart_pt", GenPart_pt, &b_GenPart_pt);
   fChain->SetBranchAddress("GenPart_genPartIdxMother", GenPart_genPartIdxMother, &b_GenPart_genPartIdxMother);
   fChain->SetBranchAddress("GenPart_pdgId", GenPart_pdgId, &b_GenPart_pdgId);
   fChain->SetBranchAddress("GenPart_status", GenPart_status, &b_GenPart_status);
   fChain->SetBranchAddress("GenPart_statusFlags", GenPart_statusFlags, &b_GenPart_statusFlags);
   fChain->SetBranchAddress("Generator_binvar", &Generator_binvar, &b_Generator_binvar);
   fChain->SetBranchAddress("Generator_scalePDF", &Generator_scalePDF, &b_Generator_scalePDF);
   fChain->SetBranchAddress("Generator_weight", &Generator_weight, &b_Generator_weight);
   fChain->SetBranchAddress("Generator_x1", &Generator_x1, &b_Generator_x1);
   fChain->SetBranchAddress("Generator_x2", &Generator_x2, &b_Generator_x2);
   fChain->SetBranchAddress("Generator_xpdf1", &Generator_xpdf1, &b_Generator_xpdf1);
   fChain->SetBranchAddress("Generator_xpdf2", &Generator_xpdf2, &b_Generator_xpdf2);
   fChain->SetBranchAddress("Generator_id1", &Generator_id1, &b_Generator_id1);
   fChain->SetBranchAddress("Generator_id2", &Generator_id2, &b_Generator_id2);
   fChain->SetBranchAddress("GenVtx_x", &GenVtx_x, &b_GenVtx_x);
   fChain->SetBranchAddress("GenVtx_y", &GenVtx_y, &b_GenVtx_y);
   fChain->SetBranchAddress("GenVtx_z", &GenVtx_z, &b_GenVtx_z);
   fChain->SetBranchAddress("nGenVisTau", &nGenVisTau, &b_nGenVisTau);
   fChain->SetBranchAddress("genWeight", &genWeight, &b_genWeight);
   fChain->SetBranchAddress("LHEWeight_originalXWGTUP", &LHEWeight_originalXWGTUP, &b_LHEWeight_originalXWGTUP);
   fChain->SetBranchAddress("nLHEPdfWeight", &nLHEPdfWeight, &b_nLHEPdfWeight);
   fChain->SetBranchAddress("LHEPdfWeight", LHEPdfWeight, &b_LHEPdfWeight);
   fChain->SetBranchAddress("nLHEReweightingWeight", &nLHEReweightingWeight, &b_nLHEReweightingWeight);
   fChain->SetBranchAddress("LHEReweightingWeight", &LHEReweightingWeight, &b_LHEReweightingWeight);
   fChain->SetBranchAddress("nLHEScaleWeight", &nLHEScaleWeight, &b_nLHEScaleWeight);
   fChain->SetBranchAddress("LHEScaleWeight", LHEScaleWeight, &b_LHEScaleWeight);
   fChain->SetBranchAddress("nPSWeight", &nPSWeight, &b_nPSWeight);
   fChain->SetBranchAddress("PSWeight", PSWeight, &b_PSWeight);
   fChain->SetBranchAddress("nIsoTrack", &nIsoTrack, &b_nIsoTrack);
   fChain->SetBranchAddress("nJet", &nJet, &b_nJet);
   fChain->SetBranchAddress("Jet_area", Jet_area, &b_Jet_area);
   fChain->SetBranchAddress("Jet_btagCSVV2", Jet_btagCSVV2, &b_Jet_btagCSVV2);
   fChain->SetBranchAddress("Jet_btagDeepB", Jet_btagDeepB, &b_Jet_btagDeepB);
   fChain->SetBranchAddress("Jet_btagDeepCvB", Jet_btagDeepCvB, &b_Jet_btagDeepCvB);
   fChain->SetBranchAddress("Jet_btagDeepCvL", Jet_btagDeepCvL, &b_Jet_btagDeepCvL);
   fChain->SetBranchAddress("Jet_btagDeepFlavB", Jet_btagDeepFlavB, &b_Jet_btagDeepFlavB);
   fChain->SetBranchAddress("Jet_btagDeepFlavCvB", Jet_btagDeepFlavCvB, &b_Jet_btagDeepFlavCvB);
   fChain->SetBranchAddress("Jet_btagDeepFlavCvL", Jet_btagDeepFlavCvL, &b_Jet_btagDeepFlavCvL);
   fChain->SetBranchAddress("Jet_btagDeepFlavQG", Jet_btagDeepFlavQG, &b_Jet_btagDeepFlavQG);
   fChain->SetBranchAddress("Jet_chEmEF", Jet_chEmEF, &b_Jet_chEmEF);
   fChain->SetBranchAddress("Jet_chFPV0EF", Jet_chFPV0EF, &b_Jet_chFPV0EF);
   fChain->SetBranchAddress("Jet_chHEF", Jet_chHEF, &b_Jet_chHEF);
   fChain->SetBranchAddress("Jet_eta", Jet_eta, &b_Jet_eta);
   fChain->SetBranchAddress("Jet_hfsigmaEtaEta", Jet_hfsigmaEtaEta, &b_Jet_hfsigmaEtaEta);
   fChain->SetBranchAddress("Jet_hfsigmaPhiPhi", Jet_hfsigmaPhiPhi, &b_Jet_hfsigmaPhiPhi);
   fChain->SetBranchAddress("Jet_mass", Jet_mass, &b_Jet_mass);
   fChain->SetBranchAddress("Jet_muEF", Jet_muEF, &b_Jet_muEF);
   fChain->SetBranchAddress("Jet_muonSubtrFactor", Jet_muonSubtrFactor, &b_Jet_muonSubtrFactor);
   fChain->SetBranchAddress("Jet_neEmEF", Jet_neEmEF, &b_Jet_neEmEF);
   fChain->SetBranchAddress("Jet_neHEF", Jet_neHEF, &b_Jet_neHEF);
   fChain->SetBranchAddress("Jet_phi", Jet_phi, &b_Jet_phi);
   fChain->SetBranchAddress("Jet_pt", Jet_pt, &b_Jet_pt);
   fChain->SetBranchAddress("Jet_puIdDisc", Jet_puIdDisc, &b_Jet_puIdDisc);
   fChain->SetBranchAddress("Jet_qgl", Jet_qgl, &b_Jet_qgl);
   fChain->SetBranchAddress("Jet_rawFactor", Jet_rawFactor, &b_Jet_rawFactor);
   fChain->SetBranchAddress("Jet_bRegCorr", Jet_bRegCorr, &b_Jet_bRegCorr);
   fChain->SetBranchAddress("Jet_bRegRes", Jet_bRegRes, &b_Jet_bRegRes);
   fChain->SetBranchAddress("Jet_cRegCorr", Jet_cRegCorr, &b_Jet_cRegCorr);
   fChain->SetBranchAddress("Jet_cRegRes", Jet_cRegRes, &b_Jet_cRegRes);
   fChain->SetBranchAddress("Jet_electronIdx1", Jet_electronIdx1, &b_Jet_electronIdx1);
   fChain->SetBranchAddress("Jet_electronIdx2", Jet_electronIdx2, &b_Jet_electronIdx2);
   fChain->SetBranchAddress("Jet_hfadjacentEtaStripsSize", Jet_hfadjacentEtaStripsSize, &b_Jet_hfadjacentEtaStripsSize);
   fChain->SetBranchAddress("Jet_hfcentralEtaStripSize", Jet_hfcentralEtaStripSize, &b_Jet_hfcentralEtaStripSize);
   fChain->SetBranchAddress("Jet_jetId", Jet_jetId, &b_Jet_jetId);
   fChain->SetBranchAddress("Jet_muonIdx1", Jet_muonIdx1, &b_Jet_muonIdx1);
   fChain->SetBranchAddress("Jet_muonIdx2", Jet_muonIdx2, &b_Jet_muonIdx2);
   fChain->SetBranchAddress("Jet_nElectrons", Jet_nElectrons, &b_Jet_nElectrons);
   fChain->SetBranchAddress("Jet_nMuons", Jet_nMuons, &b_Jet_nMuons);
   fChain->SetBranchAddress("Jet_puId", Jet_puId, &b_Jet_puId);
   fChain->SetBranchAddress("Jet_nConstituents", Jet_nConstituents, &b_Jet_nConstituents);
   fChain->SetBranchAddress("L1PreFiringWeight_Dn", &L1PreFiringWeight_Dn, &b_L1PreFiringWeight_Dn);
   fChain->SetBranchAddress("L1PreFiringWeight_ECAL_Dn", &L1PreFiringWeight_ECAL_Dn, &b_L1PreFiringWeight_ECAL_Dn);
   fChain->SetBranchAddress("L1PreFiringWeight_ECAL_Nom", &L1PreFiringWeight_ECAL_Nom, &b_L1PreFiringWeight_ECAL_Nom);
   fChain->SetBranchAddress("L1PreFiringWeight_ECAL_Up", &L1PreFiringWeight_ECAL_Up, &b_L1PreFiringWeight_ECAL_Up);
   fChain->SetBranchAddress("L1PreFiringWeight_Muon_Nom", &L1PreFiringWeight_Muon_Nom, &b_L1PreFiringWeight_Muon_Nom);
   fChain->SetBranchAddress("L1PreFiringWeight_Muon_StatDn", &L1PreFiringWeight_Muon_StatDn, &b_L1PreFiringWeight_Muon_StatDn);
   fChain->SetBranchAddress("L1PreFiringWeight_Muon_StatUp", &L1PreFiringWeight_Muon_StatUp, &b_L1PreFiringWeight_Muon_StatUp);
   fChain->SetBranchAddress("L1PreFiringWeight_Muon_SystDn", &L1PreFiringWeight_Muon_SystDn, &b_L1PreFiringWeight_Muon_SystDn);
   fChain->SetBranchAddress("L1PreFiringWeight_Muon_SystUp", &L1PreFiringWeight_Muon_SystUp, &b_L1PreFiringWeight_Muon_SystUp);
   fChain->SetBranchAddress("L1PreFiringWeight_Nom", &L1PreFiringWeight_Nom, &b_L1PreFiringWeight_Nom);
   fChain->SetBranchAddress("L1PreFiringWeight_Up", &L1PreFiringWeight_Up, &b_L1PreFiringWeight_Up);
   fChain->SetBranchAddress("LHE_Njets", &LHE_Njets, &b_LHE_Njets);
   fChain->SetBranchAddress("nLHEPart", &nLHEPart, &b_nLHEPart);
   fChain->SetBranchAddress("nLowPtElectron", &nLowPtElectron, &b_nLowPtElectron);
   fChain->SetBranchAddress("GenMET_phi", &GenMET_phi, &b_GenMET_phi);
   fChain->SetBranchAddress("GenMET_pt", &GenMET_pt, &b_GenMET_pt);
   fChain->SetBranchAddress("MET_MetUnclustEnUpDeltaX", &MET_MetUnclustEnUpDeltaX, &b_MET_MetUnclustEnUpDeltaX);
   fChain->SetBranchAddress("MET_MetUnclustEnUpDeltaY", &MET_MetUnclustEnUpDeltaY, &b_MET_MetUnclustEnUpDeltaY);
   fChain->SetBranchAddress("MET_covXX", &MET_covXX, &b_MET_covXX);
   fChain->SetBranchAddress("MET_covXY", &MET_covXY, &b_MET_covXY);
   fChain->SetBranchAddress("MET_covYY", &MET_covYY, &b_MET_covYY);
   fChain->SetBranchAddress("MET_phi", &MET_phi, &b_MET_phi);
   fChain->SetBranchAddress("MET_pt", &MET_pt, &b_MET_pt);
   fChain->SetBranchAddress("MET_significance", &MET_significance, &b_MET_significance);
   fChain->SetBranchAddress("MET_sumEt", &MET_sumEt, &b_MET_sumEt);
   fChain->SetBranchAddress("MET_sumPtUnclustered", &MET_sumPtUnclustered, &b_MET_sumPtUnclustered);
   fChain->SetBranchAddress("nMuon", &nMuon, &b_nMuon);
   fChain->SetBranchAddress("Muon_dxy", Muon_dxy, &b_Muon_dxy);
   fChain->SetBranchAddress("Muon_dxyErr", Muon_dxyErr, &b_Muon_dxyErr);
   fChain->SetBranchAddress("Muon_dxybs", Muon_dxybs, &b_Muon_dxybs);
   fChain->SetBranchAddress("Muon_dz", Muon_dz, &b_Muon_dz);
   fChain->SetBranchAddress("Muon_dzErr", Muon_dzErr, &b_Muon_dzErr);
   fChain->SetBranchAddress("Muon_eta", Muon_eta, &b_Muon_eta);
   fChain->SetBranchAddress("Muon_ip3d", Muon_ip3d, &b_Muon_ip3d);
   fChain->SetBranchAddress("Muon_jetPtRelv2", Muon_jetPtRelv2, &b_Muon_jetPtRelv2);
   fChain->SetBranchAddress("Muon_jetRelIso", Muon_jetRelIso, &b_Muon_jetRelIso);
   fChain->SetBranchAddress("Muon_mass", Muon_mass, &b_Muon_mass);
   fChain->SetBranchAddress("Muon_miniPFRelIso_all", Muon_miniPFRelIso_all, &b_Muon_miniPFRelIso_all);
   fChain->SetBranchAddress("Muon_miniPFRelIso_chg", Muon_miniPFRelIso_chg, &b_Muon_miniPFRelIso_chg);
   fChain->SetBranchAddress("Muon_pfRelIso03_all", Muon_pfRelIso03_all, &b_Muon_pfRelIso03_all);
   fChain->SetBranchAddress("Muon_pfRelIso03_chg", Muon_pfRelIso03_chg, &b_Muon_pfRelIso03_chg);
   fChain->SetBranchAddress("Muon_pfRelIso04_all", Muon_pfRelIso04_all, &b_Muon_pfRelIso04_all);
   fChain->SetBranchAddress("Muon_phi", Muon_phi, &b_Muon_phi);
   fChain->SetBranchAddress("Muon_pt", Muon_pt, &b_Muon_pt);
   fChain->SetBranchAddress("Muon_ptErr", Muon_ptErr, &b_Muon_ptErr);
   fChain->SetBranchAddress("Muon_segmentComp", Muon_segmentComp, &b_Muon_segmentComp);
   fChain->SetBranchAddress("Muon_sip3d", Muon_sip3d, &b_Muon_sip3d);
   fChain->SetBranchAddress("Muon_softMva", Muon_softMva, &b_Muon_softMva);
   fChain->SetBranchAddress("Muon_tkRelIso", Muon_tkRelIso, &b_Muon_tkRelIso);
   fChain->SetBranchAddress("Muon_tunepRelPt", Muon_tunepRelPt, &b_Muon_tunepRelPt);
   fChain->SetBranchAddress("Muon_mvaLowPt", Muon_mvaLowPt, &b_Muon_mvaLowPt);
   fChain->SetBranchAddress("Muon_mvaTTH", Muon_mvaTTH, &b_Muon_mvaTTH);
   fChain->SetBranchAddress("Muon_charge", Muon_charge, &b_Muon_charge);
   fChain->SetBranchAddress("Muon_jetIdx", Muon_jetIdx, &b_Muon_jetIdx);
   fChain->SetBranchAddress("Muon_nStations", Muon_nStations, &b_Muon_nStations);
   fChain->SetBranchAddress("Muon_nTrackerLayers", Muon_nTrackerLayers, &b_Muon_nTrackerLayers);
   fChain->SetBranchAddress("Muon_pdgId", Muon_pdgId, &b_Muon_pdgId);
   fChain->SetBranchAddress("Muon_tightCharge", Muon_tightCharge, &b_Muon_tightCharge);
   fChain->SetBranchAddress("Muon_fsrPhotonIdx", Muon_fsrPhotonIdx, &b_Muon_fsrPhotonIdx);
   fChain->SetBranchAddress("Muon_highPtId", Muon_highPtId, &b_Muon_highPtId);
   fChain->SetBranchAddress("Muon_highPurity", Muon_highPurity, &b_Muon_highPurity);
   fChain->SetBranchAddress("Muon_inTimeMuon", Muon_inTimeMuon, &b_Muon_inTimeMuon);
   fChain->SetBranchAddress("Muon_isGlobal", Muon_isGlobal, &b_Muon_isGlobal);
   fChain->SetBranchAddress("Muon_isPFcand", Muon_isPFcand, &b_Muon_isPFcand);
   fChain->SetBranchAddress("Muon_isStandalone", Muon_isStandalone, &b_Muon_isStandalone);
   fChain->SetBranchAddress("Muon_isTracker", Muon_isTracker, &b_Muon_isTracker);
   fChain->SetBranchAddress("Muon_jetNDauCharged", Muon_jetNDauCharged, &b_Muon_jetNDauCharged);
   fChain->SetBranchAddress("Muon_looseId", Muon_looseId, &b_Muon_looseId);
   fChain->SetBranchAddress("Muon_mediumId", Muon_mediumId, &b_Muon_mediumId);
   fChain->SetBranchAddress("Muon_mediumPromptId", Muon_mediumPromptId, &b_Muon_mediumPromptId);
   fChain->SetBranchAddress("Muon_miniIsoId", Muon_miniIsoId, &b_Muon_miniIsoId);
   fChain->SetBranchAddress("Muon_multiIsoId", Muon_multiIsoId, &b_Muon_multiIsoId);
   fChain->SetBranchAddress("Muon_mvaId", Muon_mvaId, &b_Muon_mvaId);
   fChain->SetBranchAddress("Muon_mvaLowPtId", Muon_mvaLowPtId, &b_Muon_mvaLowPtId);
   fChain->SetBranchAddress("Muon_pfIsoId", Muon_pfIsoId, &b_Muon_pfIsoId);
   fChain->SetBranchAddress("Muon_puppiIsoId", Muon_puppiIsoId, &b_Muon_puppiIsoId);
   fChain->SetBranchAddress("Muon_softId", Muon_softId, &b_Muon_softId);
   fChain->SetBranchAddress("Muon_softMvaId", Muon_softMvaId, &b_Muon_softMvaId);
   fChain->SetBranchAddress("Muon_tightId", Muon_tightId, &b_Muon_tightId);
   fChain->SetBranchAddress("Muon_tkIsoId", Muon_tkIsoId, &b_Muon_tkIsoId);
   fChain->SetBranchAddress("Muon_triggerIdLoose", Muon_triggerIdLoose, &b_Muon_triggerIdLoose);
   fChain->SetBranchAddress("nPhoton", &nPhoton, &b_nPhoton);
   fChain->SetBranchAddress("Photon_dEscaleDown", Photon_dEscaleDown, &b_Photon_dEscaleDown);
   fChain->SetBranchAddress("Photon_dEscaleUp", Photon_dEscaleUp, &b_Photon_dEscaleUp);
   fChain->SetBranchAddress("Photon_dEsigmaDown", Photon_dEsigmaDown, &b_Photon_dEsigmaDown);
   fChain->SetBranchAddress("Photon_dEsigmaUp", Photon_dEsigmaUp, &b_Photon_dEsigmaUp);
   fChain->SetBranchAddress("Photon_eCorr", Photon_eCorr, &b_Photon_eCorr);
   fChain->SetBranchAddress("Photon_energyErr", Photon_energyErr, &b_Photon_energyErr);
   fChain->SetBranchAddress("Photon_eta", Photon_eta, &b_Photon_eta);
   fChain->SetBranchAddress("Photon_hoe", Photon_hoe, &b_Photon_hoe);
   fChain->SetBranchAddress("Photon_mass", Photon_mass, &b_Photon_mass);
   fChain->SetBranchAddress("Photon_mvaID", Photon_mvaID, &b_Photon_mvaID);
   fChain->SetBranchAddress("Photon_mvaID_Fall17V1p1", Photon_mvaID_Fall17V1p1, &b_Photon_mvaID_Fall17V1p1);
   fChain->SetBranchAddress("Photon_pfRelIso03_all", Photon_pfRelIso03_all, &b_Photon_pfRelIso03_all);
   fChain->SetBranchAddress("Photon_pfRelIso03_chg", Photon_pfRelIso03_chg, &b_Photon_pfRelIso03_chg);
   fChain->SetBranchAddress("Photon_phi", Photon_phi, &b_Photon_phi);
   fChain->SetBranchAddress("Photon_pt", Photon_pt, &b_Photon_pt);
   fChain->SetBranchAddress("Photon_r9", Photon_r9, &b_Photon_r9);
   fChain->SetBranchAddress("Photon_sieie", Photon_sieie, &b_Photon_sieie);
   fChain->SetBranchAddress("Photon_charge", Photon_charge, &b_Photon_charge);
   fChain->SetBranchAddress("Photon_cutBased", Photon_cutBased, &b_Photon_cutBased);
   fChain->SetBranchAddress("Photon_cutBased_Fall17V1Bitmap", Photon_cutBased_Fall17V1Bitmap, &b_Photon_cutBased_Fall17V1Bitmap);
   fChain->SetBranchAddress("Photon_electronIdx", Photon_electronIdx, &b_Photon_electronIdx);
   fChain->SetBranchAddress("Photon_jetIdx", Photon_jetIdx, &b_Photon_jetIdx);
   fChain->SetBranchAddress("Photon_pdgId", Photon_pdgId, &b_Photon_pdgId);
   fChain->SetBranchAddress("Photon_vidNestedWPBitmap", Photon_vidNestedWPBitmap, &b_Photon_vidNestedWPBitmap);
   fChain->SetBranchAddress("Photon_electronVeto", Photon_electronVeto, &b_Photon_electronVeto);
   fChain->SetBranchAddress("Photon_isScEtaEB", Photon_isScEtaEB, &b_Photon_isScEtaEB);
   fChain->SetBranchAddress("Photon_isScEtaEE", Photon_isScEtaEE, &b_Photon_isScEtaEE);
   fChain->SetBranchAddress("Photon_mvaID_WP80", Photon_mvaID_WP80, &b_Photon_mvaID_WP80);
   fChain->SetBranchAddress("Photon_mvaID_WP90", Photon_mvaID_WP90, &b_Photon_mvaID_WP90);
   fChain->SetBranchAddress("Photon_pixelSeed", Photon_pixelSeed, &b_Photon_pixelSeed);
   fChain->SetBranchAddress("Photon_seedGain", Photon_seedGain, &b_Photon_seedGain);
   fChain->SetBranchAddress("Pileup_nTrueInt", &Pileup_nTrueInt, &b_Pileup_nTrueInt);
   fChain->SetBranchAddress("Pileup_pudensity", &Pileup_pudensity, &b_Pileup_pudensity);
   fChain->SetBranchAddress("Pileup_gpudensity", &Pileup_gpudensity, &b_Pileup_gpudensity);
   fChain->SetBranchAddress("Pileup_nPU", &Pileup_nPU, &b_Pileup_nPU);
   fChain->SetBranchAddress("Pileup_sumEOOT", &Pileup_sumEOOT, &b_Pileup_sumEOOT);
   fChain->SetBranchAddress("Pileup_sumLOOT", &Pileup_sumLOOT, &b_Pileup_sumLOOT);
   fChain->SetBranchAddress("PuppiMET_phi", &PuppiMET_phi, &b_PuppiMET_phi);
   fChain->SetBranchAddress("PuppiMET_phiJERDown", &PuppiMET_phiJERDown, &b_PuppiMET_phiJERDown);
   fChain->SetBranchAddress("PuppiMET_phiJERUp", &PuppiMET_phiJERUp, &b_PuppiMET_phiJERUp);
   fChain->SetBranchAddress("PuppiMET_phiJESDown", &PuppiMET_phiJESDown, &b_PuppiMET_phiJESDown);
   fChain->SetBranchAddress("PuppiMET_phiJESUp", &PuppiMET_phiJESUp, &b_PuppiMET_phiJESUp);
   fChain->SetBranchAddress("PuppiMET_phiUnclusteredDown", &PuppiMET_phiUnclusteredDown, &b_PuppiMET_phiUnclusteredDown);
   fChain->SetBranchAddress("PuppiMET_phiUnclusteredUp", &PuppiMET_phiUnclusteredUp, &b_PuppiMET_phiUnclusteredUp);
   fChain->SetBranchAddress("PuppiMET_pt", &PuppiMET_pt, &b_PuppiMET_pt);
   fChain->SetBranchAddress("PuppiMET_ptJERDown", &PuppiMET_ptJERDown, &b_PuppiMET_ptJERDown);
   fChain->SetBranchAddress("PuppiMET_ptJERUp", &PuppiMET_ptJERUp, &b_PuppiMET_ptJERUp);
   fChain->SetBranchAddress("PuppiMET_ptJESDown", &PuppiMET_ptJESDown, &b_PuppiMET_ptJESDown);
   fChain->SetBranchAddress("PuppiMET_ptJESUp", &PuppiMET_ptJESUp, &b_PuppiMET_ptJESUp);
   fChain->SetBranchAddress("PuppiMET_ptUnclusteredDown", &PuppiMET_ptUnclusteredDown, &b_PuppiMET_ptUnclusteredDown);
   fChain->SetBranchAddress("PuppiMET_ptUnclusteredUp", &PuppiMET_ptUnclusteredUp, &b_PuppiMET_ptUnclusteredUp);
   fChain->SetBranchAddress("PuppiMET_sumEt", &PuppiMET_sumEt, &b_PuppiMET_sumEt);
   fChain->SetBranchAddress("RawMET_phi", &RawMET_phi, &b_RawMET_phi);
   fChain->SetBranchAddress("RawMET_pt", &RawMET_pt, &b_RawMET_pt);
   fChain->SetBranchAddress("RawMET_sumEt", &RawMET_sumEt, &b_RawMET_sumEt);
   fChain->SetBranchAddress("RawPuppiMET_phi", &RawPuppiMET_phi, &b_RawPuppiMET_phi);
   fChain->SetBranchAddress("RawPuppiMET_pt", &RawPuppiMET_pt, &b_RawPuppiMET_pt);
   fChain->SetBranchAddress("RawPuppiMET_sumEt", &RawPuppiMET_sumEt, &b_RawPuppiMET_sumEt);
   fChain->SetBranchAddress("fixedGridRhoFastjetAll", &fixedGridRhoFastjetAll, &b_fixedGridRhoFastjetAll);
   fChain->SetBranchAddress("fixedGridRhoFastjetCentral", &fixedGridRhoFastjetCentral, &b_fixedGridRhoFastjetCentral);
   fChain->SetBranchAddress("fixedGridRhoFastjetCentralCalo", &fixedGridRhoFastjetCentralCalo, &b_fixedGridRhoFastjetCentralCalo);
   fChain->SetBranchAddress("fixedGridRhoFastjetCentralChargedPileUp", &fixedGridRhoFastjetCentralChargedPileUp, &b_fixedGridRhoFastjetCentralChargedPileUp);
   fChain->SetBranchAddress("fixedGridRhoFastjetCentralNeutral", &fixedGridRhoFastjetCentralNeutral, &b_fixedGridRhoFastjetCentralNeutral);
   fChain->SetBranchAddress("nGenDressedLepton", &nGenDressedLepton, &b_nGenDressedLepton);
   fChain->SetBranchAddress("GenDressedLepton_eta", GenDressedLepton_eta, &b_GenDressedLepton_eta);
   fChain->SetBranchAddress("GenDressedLepton_mass", GenDressedLepton_mass, &b_GenDressedLepton_mass);
   fChain->SetBranchAddress("GenDressedLepton_phi", GenDressedLepton_phi, &b_GenDressedLepton_phi);
   fChain->SetBranchAddress("GenDressedLepton_pt", GenDressedLepton_pt, &b_GenDressedLepton_pt);
   fChain->SetBranchAddress("GenDressedLepton_pdgId", GenDressedLepton_pdgId, &b_GenDressedLepton_pdgId);
   fChain->SetBranchAddress("GenDressedLepton_hasTauAnc", GenDressedLepton_hasTauAnc, &b_GenDressedLepton_hasTauAnc);
   fChain->SetBranchAddress("nGenIsolatedPhoton", &nGenIsolatedPhoton, &b_nGenIsolatedPhoton);
   fChain->SetBranchAddress("GenIsolatedPhoton_eta", GenIsolatedPhoton_eta, &b_GenIsolatedPhoton_eta);
   fChain->SetBranchAddress("GenIsolatedPhoton_mass", GenIsolatedPhoton_mass, &b_GenIsolatedPhoton_mass);
   fChain->SetBranchAddress("GenIsolatedPhoton_phi", GenIsolatedPhoton_phi, &b_GenIsolatedPhoton_phi);
   fChain->SetBranchAddress("GenIsolatedPhoton_pt", GenIsolatedPhoton_pt, &b_GenIsolatedPhoton_pt);
   fChain->SetBranchAddress("PV_ndof", &PV_ndof, &b_PV_ndof);
   fChain->SetBranchAddress("PV_x", &PV_x, &b_PV_x);
   fChain->SetBranchAddress("PV_y", &PV_y, &b_PV_y);
   fChain->SetBranchAddress("PV_z", &PV_z, &b_PV_z);
   fChain->SetBranchAddress("PV_chi2", &PV_chi2, &b_PV_chi2);
   fChain->SetBranchAddress("PV_score", &PV_score, &b_PV_score);
   fChain->SetBranchAddress("PV_npvs", &PV_npvs, &b_PV_npvs);
   fChain->SetBranchAddress("PV_npvsGood", &PV_npvsGood, &b_PV_npvsGood);
   fChain->SetBranchAddress("Electron_genPartIdx", Electron_genPartIdx, &b_Electron_genPartIdx);
   fChain->SetBranchAddress("Electron_genPartFlav", Electron_genPartFlav, &b_Electron_genPartFlav);
   fChain->SetBranchAddress("GenJet_partonFlavour", GenJet_partonFlavour, &b_GenJet_partonFlavour);
   fChain->SetBranchAddress("GenJet_hadronFlavour", GenJet_hadronFlavour, &b_GenJet_hadronFlavour);
   fChain->SetBranchAddress("GenVtx_t0", &GenVtx_t0, &b_GenVtx_t0);
   fChain->SetBranchAddress("Jet_genJetIdx", Jet_genJetIdx, &b_Jet_genJetIdx);
   fChain->SetBranchAddress("Jet_hadronFlavour", Jet_hadronFlavour, &b_Jet_hadronFlavour);
   fChain->SetBranchAddress("Jet_partonFlavour", Jet_partonFlavour, &b_Jet_partonFlavour);
   fChain->SetBranchAddress("Muon_genPartIdx", Muon_genPartIdx, &b_Muon_genPartIdx);
   fChain->SetBranchAddress("Muon_genPartFlav", Muon_genPartFlav, &b_Muon_genPartFlav);
   fChain->SetBranchAddress("Photon_genPartIdx", Photon_genPartIdx, &b_Photon_genPartIdx);
   fChain->SetBranchAddress("Photon_genPartFlav", Photon_genPartFlav, &b_Photon_genPartFlav);
   fChain->SetBranchAddress("MET_fiducialGenPhi", &MET_fiducialGenPhi, &b_MET_fiducialGenPhi);
   fChain->SetBranchAddress("MET_fiducialGenPt", &MET_fiducialGenPt, &b_MET_fiducialGenPt);
   fChain->SetBranchAddress("Electron_cleanmask", Electron_cleanmask, &b_Electron_cleanmask);
   fChain->SetBranchAddress("Jet_cleanmask", Jet_cleanmask, &b_Jet_cleanmask);
   fChain->SetBranchAddress("Muon_cleanmask", Muon_cleanmask, &b_Muon_cleanmask);
   fChain->SetBranchAddress("Photon_cleanmask", Photon_cleanmask, &b_Photon_cleanmask);
   fChain->SetBranchAddress("Flag_HBHENoiseFilter", &Flag_HBHENoiseFilter, &b_Flag_HBHENoiseFilter);
   fChain->SetBranchAddress("Flag_HBHENoiseIsoFilter", &Flag_HBHENoiseIsoFilter, &b_Flag_HBHENoiseIsoFilter);
   fChain->SetBranchAddress("Flag_CSCTightHaloFilter", &Flag_CSCTightHaloFilter, &b_Flag_CSCTightHaloFilter);
   fChain->SetBranchAddress("Flag_CSCTightHaloTrkMuUnvetoFilter", &Flag_CSCTightHaloTrkMuUnvetoFilter, &b_Flag_CSCTightHaloTrkMuUnvetoFilter);
   fChain->SetBranchAddress("Flag_CSCTightHalo2015Filter", &Flag_CSCTightHalo2015Filter, &b_Flag_CSCTightHalo2015Filter);
   fChain->SetBranchAddress("Flag_globalTightHalo2016Filter", &Flag_globalTightHalo2016Filter, &b_Flag_globalTightHalo2016Filter);
   fChain->SetBranchAddress("Flag_globalSuperTightHalo2016Filter", &Flag_globalSuperTightHalo2016Filter, &b_Flag_globalSuperTightHalo2016Filter);
   fChain->SetBranchAddress("Flag_HcalStripHaloFilter", &Flag_HcalStripHaloFilter, &b_Flag_HcalStripHaloFilter);
   fChain->SetBranchAddress("Flag_hcalLaserEventFilter", &Flag_hcalLaserEventFilter, &b_Flag_hcalLaserEventFilter);
   fChain->SetBranchAddress("Flag_EcalDeadCellTriggerPrimitiveFilter", &Flag_EcalDeadCellTriggerPrimitiveFilter, &b_Flag_EcalDeadCellTriggerPrimitiveFilter);
   fChain->SetBranchAddress("Flag_EcalDeadCellBoundaryEnergyFilter", &Flag_EcalDeadCellBoundaryEnergyFilter, &b_Flag_EcalDeadCellBoundaryEnergyFilter);
   fChain->SetBranchAddress("Flag_ecalBadCalibFilter", &Flag_ecalBadCalibFilter, &b_Flag_ecalBadCalibFilter);
   fChain->SetBranchAddress("Flag_goodVertices", &Flag_goodVertices, &b_Flag_goodVertices);
   fChain->SetBranchAddress("Flag_eeBadScFilter", &Flag_eeBadScFilter, &b_Flag_eeBadScFilter);
   fChain->SetBranchAddress("Flag_ecalLaserCorrFilter", &Flag_ecalLaserCorrFilter, &b_Flag_ecalLaserCorrFilter);
   fChain->SetBranchAddress("Flag_trkPOGFilters", &Flag_trkPOGFilters, &b_Flag_trkPOGFilters);
   fChain->SetBranchAddress("Flag_chargedHadronTrackResolutionFilter", &Flag_chargedHadronTrackResolutionFilter, &b_Flag_chargedHadronTrackResolutionFilter);
   fChain->SetBranchAddress("Flag_muonBadTrackFilter", &Flag_muonBadTrackFilter, &b_Flag_muonBadTrackFilter);
   fChain->SetBranchAddress("Flag_BadChargedCandidateFilter", &Flag_BadChargedCandidateFilter, &b_Flag_BadChargedCandidateFilter);
   fChain->SetBranchAddress("Flag_BadPFMuonFilter", &Flag_BadPFMuonFilter, &b_Flag_BadPFMuonFilter);
   fChain->SetBranchAddress("Flag_BadPFMuonDzFilter", &Flag_BadPFMuonDzFilter, &b_Flag_BadPFMuonDzFilter);
   fChain->SetBranchAddress("Flag_hfNoisyHitsFilter", &Flag_hfNoisyHitsFilter, &b_Flag_hfNoisyHitsFilter);
   fChain->SetBranchAddress("Flag_BadChargedCandidateSummer16Filter", &Flag_BadChargedCandidateSummer16Filter, &b_Flag_BadChargedCandidateSummer16Filter);
   fChain->SetBranchAddress("Flag_BadPFMuonSummer16Filter", &Flag_BadPFMuonSummer16Filter, &b_Flag_BadPFMuonSummer16Filter);
   fChain->SetBranchAddress("Flag_trkPOG_manystripclus53X", &Flag_trkPOG_manystripclus53X, &b_Flag_trkPOG_manystripclus53X);
   fChain->SetBranchAddress("Flag_trkPOG_toomanystripclus53X", &Flag_trkPOG_toomanystripclus53X, &b_Flag_trkPOG_toomanystripclus53X);
   fChain->SetBranchAddress("Flag_trkPOG_logErrorTooManyClusters", &Flag_trkPOG_logErrorTooManyClusters, &b_Flag_trkPOG_logErrorTooManyClusters);
   fChain->SetBranchAddress("Flag_METFilters", &Flag_METFilters, &b_Flag_METFilters);
   fChain->SetBranchAddress("HLT_Ele27_Ele37_CaloIdL_MW", &HLT_Ele27_Ele37_CaloIdL_MW, &b_HLT_Ele27_Ele37_CaloIdL_MW);
   fChain->SetBranchAddress("HLT_Mu27_Ele37_CaloIdL_MW", &HLT_Mu27_Ele37_CaloIdL_MW, &b_HLT_Mu27_Ele37_CaloIdL_MW);
   fChain->SetBranchAddress("HLT_Mu37_Ele27_CaloIdL_MW", &HLT_Mu37_Ele27_CaloIdL_MW, &b_HLT_Mu37_Ele27_CaloIdL_MW);
   fChain->SetBranchAddress("HLT_Mu37_TkMu27", &HLT_Mu37_TkMu27, &b_HLT_Mu37_TkMu27);
   fChain->SetBranchAddress("HLT_DoublePhoton33_CaloIdL", &HLT_DoublePhoton33_CaloIdL, &b_HLT_DoublePhoton33_CaloIdL);
   fChain->SetBranchAddress("HLT_DoublePhoton70", &HLT_DoublePhoton70, &b_HLT_DoublePhoton70);
   fChain->SetBranchAddress("HLT_DoublePhoton85", &HLT_DoublePhoton85, &b_HLT_DoublePhoton85);
   fChain->SetBranchAddress("HLT_Ele27_WPTight_Gsf", &HLT_Ele27_WPTight_Gsf, &b_HLT_Ele27_WPTight_Gsf);
   fChain->SetBranchAddress("HLT_Ele28_WPTight_Gsf", &HLT_Ele28_WPTight_Gsf, &b_HLT_Ele28_WPTight_Gsf);
   fChain->SetBranchAddress("HLT_Ele30_WPTight_Gsf", &HLT_Ele30_WPTight_Gsf, &b_HLT_Ele30_WPTight_Gsf);
   fChain->SetBranchAddress("HLT_Ele32_WPTight_Gsf", &HLT_Ele32_WPTight_Gsf, &b_HLT_Ele32_WPTight_Gsf);
   fChain->SetBranchAddress("HLT_Ele35_WPTight_Gsf", &HLT_Ele35_WPTight_Gsf, &b_HLT_Ele35_WPTight_Gsf);
   fChain->SetBranchAddress("HLT_Ele35_WPTight_Gsf_L1EGMT", &HLT_Ele35_WPTight_Gsf_L1EGMT, &b_HLT_Ele35_WPTight_Gsf_L1EGMT);
   fChain->SetBranchAddress("HLT_Ele38_WPTight_Gsf", &HLT_Ele38_WPTight_Gsf, &b_HLT_Ele38_WPTight_Gsf);
   fChain->SetBranchAddress("HLT_Ele40_WPTight_Gsf", &HLT_Ele40_WPTight_Gsf, &b_HLT_Ele40_WPTight_Gsf);
   fChain->SetBranchAddress("HLT_Ele32_WPTight_Gsf_L1DoubleEG", &HLT_Ele32_WPTight_Gsf_L1DoubleEG, &b_HLT_Ele32_WPTight_Gsf_L1DoubleEG);
   fChain->SetBranchAddress("HLT_Ele24_eta2p1_WPTight_Gsf_LooseChargedIsoPFTauHPS30_eta2p1_CrossL1", &HLT_Ele24_eta2p1_WPTight_Gsf_LooseChargedIsoPFTauHPS30_eta2p1_CrossL1, &b_HLT_Ele24_eta2p1_WPTight_Gsf_LooseChargedIsoPFTauHPS30_eta2p1_CrossL1);
   fChain->SetBranchAddress("HLT_Ele24_eta2p1_WPTight_Gsf_MediumChargedIsoPFTauHPS30_eta2p1_CrossL1", &HLT_Ele24_eta2p1_WPTight_Gsf_MediumChargedIsoPFTauHPS30_eta2p1_CrossL1, &b_HLT_Ele24_eta2p1_WPTight_Gsf_MediumChargedIsoPFTauHPS30_eta2p1_CrossL1);
   fChain->SetBranchAddress("HLT_Ele24_eta2p1_WPTight_Gsf_TightChargedIsoPFTauHPS30_eta2p1_CrossL1", &HLT_Ele24_eta2p1_WPTight_Gsf_TightChargedIsoPFTauHPS30_eta2p1_CrossL1, &b_HLT_Ele24_eta2p1_WPTight_Gsf_TightChargedIsoPFTauHPS30_eta2p1_CrossL1);
   fChain->SetBranchAddress("HLT_Ele24_eta2p1_WPTight_Gsf_LooseChargedIsoPFTauHPS30_eta2p1_TightID_CrossL1", &HLT_Ele24_eta2p1_WPTight_Gsf_LooseChargedIsoPFTauHPS30_eta2p1_TightID_CrossL1, &b_HLT_Ele24_eta2p1_WPTight_Gsf_LooseChargedIsoPFTauHPS30_eta2p1_TightID_CrossL1);
   fChain->SetBranchAddress("HLT_Ele24_eta2p1_WPTight_Gsf_MediumChargedIsoPFTauHPS30_eta2p1_TightID_CrossL1", &HLT_Ele24_eta2p1_WPTight_Gsf_MediumChargedIsoPFTauHPS30_eta2p1_TightID_CrossL1, &b_HLT_Ele24_eta2p1_WPTight_Gsf_MediumChargedIsoPFTauHPS30_eta2p1_TightID_CrossL1);
   fChain->SetBranchAddress("HLT_Ele24_eta2p1_WPTight_Gsf_TightChargedIsoPFTauHPS30_eta2p1_TightID_CrossL1", &HLT_Ele24_eta2p1_WPTight_Gsf_TightChargedIsoPFTauHPS30_eta2p1_TightID_CrossL1, &b_HLT_Ele24_eta2p1_WPTight_Gsf_TightChargedIsoPFTauHPS30_eta2p1_TightID_CrossL1);
   fChain->SetBranchAddress("HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL", &HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL, &b_HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL);
   fChain->SetBranchAddress("HLT_Mu19_TrkIsoVVL_Mu9_TrkIsoVVL", &HLT_Mu19_TrkIsoVVL_Mu9_TrkIsoVVL, &b_HLT_Mu19_TrkIsoVVL_Mu9_TrkIsoVVL);
   fChain->SetBranchAddress("HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ", &HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ, &b_HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ);
   fChain->SetBranchAddress("HLT_Mu19_TrkIsoVVL_Mu9_TrkIsoVVL_DZ", &HLT_Mu19_TrkIsoVVL_Mu9_TrkIsoVVL_DZ, &b_HLT_Mu19_TrkIsoVVL_Mu9_TrkIsoVVL_DZ);
   fChain->SetBranchAddress("HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass8", &HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass8, &b_HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass8);
   fChain->SetBranchAddress("HLT_Mu19_TrkIsoVVL_Mu9_TrkIsoVVL_DZ_Mass8", &HLT_Mu19_TrkIsoVVL_Mu9_TrkIsoVVL_DZ_Mass8, &b_HLT_Mu19_TrkIsoVVL_Mu9_TrkIsoVVL_DZ_Mass8);
   fChain->SetBranchAddress("HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass3p8", &HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass3p8, &b_HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass3p8);
   fChain->SetBranchAddress("HLT_Mu19_TrkIsoVVL_Mu9_TrkIsoVVL_DZ_Mass3p8", &HLT_Mu19_TrkIsoVVL_Mu9_TrkIsoVVL_DZ_Mass3p8, &b_HLT_Mu19_TrkIsoVVL_Mu9_TrkIsoVVL_DZ_Mass3p8);
   fChain->SetBranchAddress("HLT_PFJet15", &HLT_PFJet15, &b_HLT_PFJet15);
   fChain->SetBranchAddress("HLT_PFJet25", &HLT_PFJet25, &b_HLT_PFJet25);
   fChain->SetBranchAddress("HLT_PFJet40", &HLT_PFJet40, &b_HLT_PFJet40);
   fChain->SetBranchAddress("HLT_PFJet60", &HLT_PFJet60, &b_HLT_PFJet60);
   fChain->SetBranchAddress("HLT_PFJet80", &HLT_PFJet80, &b_HLT_PFJet80);
   fChain->SetBranchAddress("HLT_PFJet140", &HLT_PFJet140, &b_HLT_PFJet140);
   fChain->SetBranchAddress("HLT_PFJet200", &HLT_PFJet200, &b_HLT_PFJet200);
   fChain->SetBranchAddress("HLT_PFJet260", &HLT_PFJet260, &b_HLT_PFJet260);
   fChain->SetBranchAddress("HLT_PFJet320", &HLT_PFJet320, &b_HLT_PFJet320);
   fChain->SetBranchAddress("HLT_PFJet400", &HLT_PFJet400, &b_HLT_PFJet400);
   fChain->SetBranchAddress("HLT_PFJet450", &HLT_PFJet450, &b_HLT_PFJet450);
   fChain->SetBranchAddress("HLT_PFJet500", &HLT_PFJet500, &b_HLT_PFJet500);
   fChain->SetBranchAddress("HLT_PFJet550", &HLT_PFJet550, &b_HLT_PFJet550);
   fChain->SetBranchAddress("HLT_PFJetFwd15", &HLT_PFJetFwd15, &b_HLT_PFJetFwd15);
   fChain->SetBranchAddress("HLT_PFJetFwd25", &HLT_PFJetFwd25, &b_HLT_PFJetFwd25);
   fChain->SetBranchAddress("HLT_PFJetFwd40", &HLT_PFJetFwd40, &b_HLT_PFJetFwd40);
   fChain->SetBranchAddress("HLT_PFJetFwd60", &HLT_PFJetFwd60, &b_HLT_PFJetFwd60);
   fChain->SetBranchAddress("HLT_PFJetFwd80", &HLT_PFJetFwd80, &b_HLT_PFJetFwd80);
   fChain->SetBranchAddress("HLT_PFJetFwd140", &HLT_PFJetFwd140, &b_HLT_PFJetFwd140);
   fChain->SetBranchAddress("HLT_PFJetFwd200", &HLT_PFJetFwd200, &b_HLT_PFJetFwd200);
   fChain->SetBranchAddress("HLT_PFJetFwd260", &HLT_PFJetFwd260, &b_HLT_PFJetFwd260);
   fChain->SetBranchAddress("HLT_PFJetFwd320", &HLT_PFJetFwd320, &b_HLT_PFJetFwd320);
   fChain->SetBranchAddress("HLT_PFJetFwd400", &HLT_PFJetFwd400, &b_HLT_PFJetFwd400);
   fChain->SetBranchAddress("HLT_PFJetFwd450", &HLT_PFJetFwd450, &b_HLT_PFJetFwd450);
   fChain->SetBranchAddress("HLT_PFJetFwd500", &HLT_PFJetFwd500, &b_HLT_PFJetFwd500);
   fChain->SetBranchAddress("HLT_PFMET110_PFMHT110_IDTight", &HLT_PFMET110_PFMHT110_IDTight, &b_HLT_PFMET110_PFMHT110_IDTight);
   fChain->SetBranchAddress("HLT_PFMET120_PFMHT120_IDTight", &HLT_PFMET120_PFMHT120_IDTight, &b_HLT_PFMET120_PFMHT120_IDTight);
   fChain->SetBranchAddress("HLT_PFMET130_PFMHT130_IDTight", &HLT_PFMET130_PFMHT130_IDTight, &b_HLT_PFMET130_PFMHT130_IDTight);
   fChain->SetBranchAddress("HLT_PFMET140_PFMHT140_IDTight", &HLT_PFMET140_PFMHT140_IDTight, &b_HLT_PFMET140_PFMHT140_IDTight);
   fChain->SetBranchAddress("HLT_PFMET100_PFMHT100_IDTight_CaloBTagDeepCSV_3p1", &HLT_PFMET100_PFMHT100_IDTight_CaloBTagDeepCSV_3p1, &b_HLT_PFMET100_PFMHT100_IDTight_CaloBTagDeepCSV_3p1);
   fChain->SetBranchAddress("HLT_PFMET110_PFMHT110_IDTight_CaloBTagDeepCSV_3p1", &HLT_PFMET110_PFMHT110_IDTight_CaloBTagDeepCSV_3p1, &b_HLT_PFMET110_PFMHT110_IDTight_CaloBTagDeepCSV_3p1);
   fChain->SetBranchAddress("HLT_PFMET120_PFMHT120_IDTight_CaloBTagDeepCSV_3p1", &HLT_PFMET120_PFMHT120_IDTight_CaloBTagDeepCSV_3p1, &b_HLT_PFMET120_PFMHT120_IDTight_CaloBTagDeepCSV_3p1);
   fChain->SetBranchAddress("HLT_PFMET130_PFMHT130_IDTight_CaloBTagDeepCSV_3p1", &HLT_PFMET130_PFMHT130_IDTight_CaloBTagDeepCSV_3p1, &b_HLT_PFMET130_PFMHT130_IDTight_CaloBTagDeepCSV_3p1);
   fChain->SetBranchAddress("HLT_PFMET140_PFMHT140_IDTight_CaloBTagDeepCSV_3p1", &HLT_PFMET140_PFMHT140_IDTight_CaloBTagDeepCSV_3p1, &b_HLT_PFMET140_PFMHT140_IDTight_CaloBTagDeepCSV_3p1);
   fChain->SetBranchAddress("HLT_PFMET120_PFMHT120_IDTight_PFHT60", &HLT_PFMET120_PFMHT120_IDTight_PFHT60, &b_HLT_PFMET120_PFMHT120_IDTight_PFHT60);
   fChain->SetBranchAddress("HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_PFHT60", &HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_PFHT60, &b_HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_PFHT60);
   fChain->SetBranchAddress("HLT_PFMETTypeOne120_PFMHT120_IDTight_PFHT60", &HLT_PFMETTypeOne120_PFMHT120_IDTight_PFHT60, &b_HLT_PFMETTypeOne120_PFMHT120_IDTight_PFHT60);
   fChain->SetBranchAddress("HLT_PFMETTypeOne110_PFMHT110_IDTight", &HLT_PFMETTypeOne110_PFMHT110_IDTight, &b_HLT_PFMETTypeOne110_PFMHT110_IDTight);
   fChain->SetBranchAddress("HLT_PFMETTypeOne120_PFMHT120_IDTight", &HLT_PFMETTypeOne120_PFMHT120_IDTight, &b_HLT_PFMETTypeOne120_PFMHT120_IDTight);
   fChain->SetBranchAddress("HLT_PFMETTypeOne130_PFMHT130_IDTight", &HLT_PFMETTypeOne130_PFMHT130_IDTight, &b_HLT_PFMETTypeOne130_PFMHT130_IDTight);
   fChain->SetBranchAddress("HLT_PFMETTypeOne140_PFMHT140_IDTight", &HLT_PFMETTypeOne140_PFMHT140_IDTight, &b_HLT_PFMETTypeOne140_PFMHT140_IDTight);
   fChain->SetBranchAddress("HLT_PFMETNoMu110_PFMHTNoMu110_IDTight", &HLT_PFMETNoMu110_PFMHTNoMu110_IDTight, &b_HLT_PFMETNoMu110_PFMHTNoMu110_IDTight);
   fChain->SetBranchAddress("HLT_PFMETNoMu120_PFMHTNoMu120_IDTight", &HLT_PFMETNoMu120_PFMHTNoMu120_IDTight, &b_HLT_PFMETNoMu120_PFMHTNoMu120_IDTight);
   fChain->SetBranchAddress("HLT_PFMETNoMu130_PFMHTNoMu130_IDTight", &HLT_PFMETNoMu130_PFMHTNoMu130_IDTight, &b_HLT_PFMETNoMu130_PFMHTNoMu130_IDTight);
   fChain->SetBranchAddress("HLT_PFMETNoMu140_PFMHTNoMu140_IDTight", &HLT_PFMETNoMu140_PFMHTNoMu140_IDTight, &b_HLT_PFMETNoMu140_PFMHTNoMu140_IDTight);
   fChain->SetBranchAddress("HLT_PFMET200_NotCleaned", &HLT_PFMET200_NotCleaned, &b_HLT_PFMET200_NotCleaned);
   fChain->SetBranchAddress("HLT_PFMET200_HBHECleaned", &HLT_PFMET200_HBHECleaned, &b_HLT_PFMET200_HBHECleaned);
   fChain->SetBranchAddress("HLT_PFMET250_HBHECleaned", &HLT_PFMET250_HBHECleaned, &b_HLT_PFMET250_HBHECleaned);
   fChain->SetBranchAddress("HLT_PFMET300_HBHECleaned", &HLT_PFMET300_HBHECleaned, &b_HLT_PFMET300_HBHECleaned);
   fChain->SetBranchAddress("HLT_PFMET200_HBHE_BeamHaloCleaned", &HLT_PFMET200_HBHE_BeamHaloCleaned, &b_HLT_PFMET200_HBHE_BeamHaloCleaned);
   fChain->SetBranchAddress("HLT_PFMETTypeOne200_HBHE_BeamHaloCleaned", &HLT_PFMETTypeOne200_HBHE_BeamHaloCleaned, &b_HLT_PFMETTypeOne200_HBHE_BeamHaloCleaned);
   fChain->SetBranchAddress("HLT_MET105_IsoTrk50", &HLT_MET105_IsoTrk50, &b_HLT_MET105_IsoTrk50);
   fChain->SetBranchAddress("HLT_MET120_IsoTrk50", &HLT_MET120_IsoTrk50, &b_HLT_MET120_IsoTrk50);
   fChain->SetBranchAddress("HLT_DoublePFJets40_CaloBTagDeepCSV_p71", &HLT_DoublePFJets40_CaloBTagDeepCSV_p71, &b_HLT_DoublePFJets40_CaloBTagDeepCSV_p71);
   fChain->SetBranchAddress("HLT_DoublePFJets100_CaloBTagDeepCSV_p71", &HLT_DoublePFJets100_CaloBTagDeepCSV_p71, &b_HLT_DoublePFJets100_CaloBTagDeepCSV_p71);
   fChain->SetBranchAddress("HLT_DoublePFJets200_CaloBTagDeepCSV_p71", &HLT_DoublePFJets200_CaloBTagDeepCSV_p71, &b_HLT_DoublePFJets200_CaloBTagDeepCSV_p71);
   fChain->SetBranchAddress("HLT_DoublePFJets350_CaloBTagDeepCSV_p71", &HLT_DoublePFJets350_CaloBTagDeepCSV_p71, &b_HLT_DoublePFJets350_CaloBTagDeepCSV_p71);
   fChain->SetBranchAddress("HLT_DoublePFJets116MaxDeta1p6_DoubleCaloBTagDeepCSV_p71", &HLT_DoublePFJets116MaxDeta1p6_DoubleCaloBTagDeepCSV_p71, &b_HLT_DoublePFJets116MaxDeta1p6_DoubleCaloBTagDeepCSV_p71);
   fChain->SetBranchAddress("HLT_DoublePFJets128MaxDeta1p6_DoubleCaloBTagDeepCSV_p71", &HLT_DoublePFJets128MaxDeta1p6_DoubleCaloBTagDeepCSV_p71, &b_HLT_DoublePFJets128MaxDeta1p6_DoubleCaloBTagDeepCSV_p71);
   fChain->SetBranchAddress("HLT_Photon300_NoHE", &HLT_Photon300_NoHE, &b_HLT_Photon300_NoHE);
   fChain->SetBranchAddress("HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ", &HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ, &b_HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ);
   fChain->SetBranchAddress("HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ_PFDiJet30", &HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ_PFDiJet30, &b_HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ_PFDiJet30);
   fChain->SetBranchAddress("HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ_CaloDiJet30", &HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ_CaloDiJet30, &b_HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ_CaloDiJet30);
   fChain->SetBranchAddress("HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ_PFDiJet30_PFBtagDeepCSV_1p5", &HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ_PFDiJet30_PFBtagDeepCSV_1p5, &b_HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ_PFDiJet30_PFBtagDeepCSV_1p5);
   fChain->SetBranchAddress("HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ_CaloDiJet30_CaloBtagDeepCSV_1p5", &HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ_CaloDiJet30_CaloBtagDeepCSV_1p5, &b_HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ_CaloDiJet30_CaloBtagDeepCSV_1p5);
   fChain->SetBranchAddress("HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL", &HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL, &b_HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL);
   fChain->SetBranchAddress("HLT_Ele15_Ele8_CaloIdL_TrackIdL_IsoVL", &HLT_Ele15_Ele8_CaloIdL_TrackIdL_IsoVL, &b_HLT_Ele15_Ele8_CaloIdL_TrackIdL_IsoVL);
   fChain->SetBranchAddress("HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ", &HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ, &b_HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ);
   fChain->SetBranchAddress("HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL", &HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL, &b_HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL);
   fChain->SetBranchAddress("HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_DZ", &HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_DZ, &b_HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_DZ);
   fChain->SetBranchAddress("HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL", &HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL, &b_HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL);
   fChain->SetBranchAddress("HLT_Mu12_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL", &HLT_Mu12_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL, &b_HLT_Mu12_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL);
   fChain->SetBranchAddress("HLT_Mu12_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ", &HLT_Mu12_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ, &b_HLT_Mu12_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ);
   fChain->SetBranchAddress("HLT_Photon20", &HLT_Photon20, &b_HLT_Photon20);
   fChain->SetBranchAddress("HLT_Photon33", &HLT_Photon33, &b_HLT_Photon33);
   fChain->SetBranchAddress("HLT_Photon50", &HLT_Photon50, &b_HLT_Photon50);
   fChain->SetBranchAddress("HLT_Photon75", &HLT_Photon75, &b_HLT_Photon75);
   fChain->SetBranchAddress("HLT_Photon90", &HLT_Photon90, &b_HLT_Photon90);
   fChain->SetBranchAddress("HLT_Photon120", &HLT_Photon120, &b_HLT_Photon120);
   fChain->SetBranchAddress("HLT_Photon150", &HLT_Photon150, &b_HLT_Photon150);
   fChain->SetBranchAddress("HLT_Photon175", &HLT_Photon175, &b_HLT_Photon175);
   fChain->SetBranchAddress("HLT_Photon200", &HLT_Photon200, &b_HLT_Photon200);
   fChain->SetBranchAddress("HLT_Photon100EB_TightID_TightIso", &HLT_Photon100EB_TightID_TightIso, &b_HLT_Photon100EB_TightID_TightIso);
   fChain->SetBranchAddress("HLT_Photon110EB_TightID_TightIso", &HLT_Photon110EB_TightID_TightIso, &b_HLT_Photon110EB_TightID_TightIso);
   fChain->SetBranchAddress("HLT_Photon120EB_TightID_TightIso", &HLT_Photon120EB_TightID_TightIso, &b_HLT_Photon120EB_TightID_TightIso);
   fChain->SetBranchAddress("HLT_Photon100EBHE10", &HLT_Photon100EBHE10, &b_HLT_Photon100EBHE10);
   fChain->SetBranchAddress("HLT_Photon100EEHE10", &HLT_Photon100EEHE10, &b_HLT_Photon100EEHE10);
   fChain->SetBranchAddress("HLT_Photon100EE_TightID_TightIso", &HLT_Photon100EE_TightID_TightIso, &b_HLT_Photon100EE_TightID_TightIso);
   fChain->SetBranchAddress("HLT_Photon50_R9Id90_HE10_IsoM", &HLT_Photon50_R9Id90_HE10_IsoM, &b_HLT_Photon50_R9Id90_HE10_IsoM);
   fChain->SetBranchAddress("HLT_Photon75_R9Id90_HE10_IsoM", &HLT_Photon75_R9Id90_HE10_IsoM, &b_HLT_Photon75_R9Id90_HE10_IsoM);
   fChain->SetBranchAddress("HLT_Photon75_R9Id90_HE10_IsoM_EBOnly_CaloMJJ300_PFJetsMJJ400DEta3", &HLT_Photon75_R9Id90_HE10_IsoM_EBOnly_CaloMJJ300_PFJetsMJJ400DEta3, &b_HLT_Photon75_R9Id90_HE10_IsoM_EBOnly_CaloMJJ300_PFJetsMJJ400DEta3);
   fChain->SetBranchAddress("HLT_Photon75_R9Id90_HE10_IsoM_EBOnly_CaloMJJ400_PFJetsMJJ600DEta3", &HLT_Photon75_R9Id90_HE10_IsoM_EBOnly_CaloMJJ400_PFJetsMJJ600DEta3, &b_HLT_Photon75_R9Id90_HE10_IsoM_EBOnly_CaloMJJ400_PFJetsMJJ600DEta3);
   fChain->SetBranchAddress("HLT_Photon90_R9Id90_HE10_IsoM", &HLT_Photon90_R9Id90_HE10_IsoM, &b_HLT_Photon90_R9Id90_HE10_IsoM);
   fChain->SetBranchAddress("HLT_Photon120_R9Id90_HE10_IsoM", &HLT_Photon120_R9Id90_HE10_IsoM, &b_HLT_Photon120_R9Id90_HE10_IsoM);
   fChain->SetBranchAddress("HLT_Photon165_R9Id90_HE10_IsoM", &HLT_Photon165_R9Id90_HE10_IsoM, &b_HLT_Photon165_R9Id90_HE10_IsoM);
   fChain->SetBranchAddress("HLT_Photon90_CaloIdL_PFHT700", &HLT_Photon90_CaloIdL_PFHT700, &b_HLT_Photon90_CaloIdL_PFHT700);
   fChain->SetBranchAddress("HLT_Diphoton30_22_R9Id_OR_IsoCaloId_AND_HE_R9Id_Mass90", &HLT_Diphoton30_22_R9Id_OR_IsoCaloId_AND_HE_R9Id_Mass90, &b_HLT_Diphoton30_22_R9Id_OR_IsoCaloId_AND_HE_R9Id_Mass90);
   fChain->SetBranchAddress("HLT_Diphoton30_22_R9Id_OR_IsoCaloId_AND_HE_R9Id_Mass95", &HLT_Diphoton30_22_R9Id_OR_IsoCaloId_AND_HE_R9Id_Mass95, &b_HLT_Diphoton30_22_R9Id_OR_IsoCaloId_AND_HE_R9Id_Mass95);
   fChain->SetBranchAddress("HLT_Diphoton30PV_18PV_R9Id_AND_IsoCaloId_AND_HE_R9Id_PixelVeto_Mass55", &HLT_Diphoton30PV_18PV_R9Id_AND_IsoCaloId_AND_HE_R9Id_PixelVeto_Mass55, &b_HLT_Diphoton30PV_18PV_R9Id_AND_IsoCaloId_AND_HE_R9Id_PixelVeto_Mass55);
   fChain->SetBranchAddress("HLT_Diphoton30PV_18PV_R9Id_AND_IsoCaloId_AND_HE_R9Id_NoPixelVeto_Mass55", &HLT_Diphoton30PV_18PV_R9Id_AND_IsoCaloId_AND_HE_R9Id_NoPixelVeto_Mass55, &b_HLT_Diphoton30PV_18PV_R9Id_AND_IsoCaloId_AND_HE_R9Id_NoPixelVeto_Mass55);
   fChain->SetBranchAddress("HLT_Photon35_TwoProngs35", &HLT_Photon35_TwoProngs35, &b_HLT_Photon35_TwoProngs35);
   fChain->SetBranchAddress("HLT_Ele30_eta2p1_WPTight_Gsf_CentralPFJet35_EleCleaned", &HLT_Ele30_eta2p1_WPTight_Gsf_CentralPFJet35_EleCleaned, &b_HLT_Ele30_eta2p1_WPTight_Gsf_CentralPFJet35_EleCleaned);
   fChain->SetBranchAddress("HLT_Ele28_eta2p1_WPTight_Gsf_HT150", &HLT_Ele28_eta2p1_WPTight_Gsf_HT150, &b_HLT_Ele28_eta2p1_WPTight_Gsf_HT150);
   fChain->SetBranchAddress("HLT_Ele28_HighEta_SC20_Mass55", &HLT_Ele28_HighEta_SC20_Mass55, &b_HLT_Ele28_HighEta_SC20_Mass55);
   fChain->SetBranchAddress("HLT_DoubleIsoMu20_eta2p1", &HLT_DoubleIsoMu20_eta2p1, &b_HLT_DoubleIsoMu20_eta2p1);
   fChain->SetBranchAddress("HLT_Ele23_CaloIdL_TrackIdL_IsoVL_PFJet30", &HLT_Ele23_CaloIdL_TrackIdL_IsoVL_PFJet30, &b_HLT_Ele23_CaloIdL_TrackIdL_IsoVL_PFJet30);
   fChain->SetBranchAddress("HLT_Ele23_CaloIdM_TrackIdM_PFJet30", &HLT_Ele23_CaloIdM_TrackIdM_PFJet30, &b_HLT_Ele23_CaloIdM_TrackIdM_PFJet30);
   fChain->SetBranchAddress("HLT_Ele50_CaloIdVT_GsfTrkIdT_PFJet165", &HLT_Ele50_CaloIdVT_GsfTrkIdT_PFJet165, &b_HLT_Ele50_CaloIdVT_GsfTrkIdT_PFJet165);
   fChain->SetBranchAddress("HLT_Ele115_CaloIdVT_GsfTrkIdT", &HLT_Ele115_CaloIdVT_GsfTrkIdT, &b_HLT_Ele115_CaloIdVT_GsfTrkIdT);
   fChain->SetBranchAddress("HLT_Ele135_CaloIdVT_GsfTrkIdT", &HLT_Ele135_CaloIdVT_GsfTrkIdT, &b_HLT_Ele135_CaloIdVT_GsfTrkIdT);
   fChain->SetBranchAddress("HLT_Ele145_CaloIdVT_GsfTrkIdT", &HLT_Ele145_CaloIdVT_GsfTrkIdT, &b_HLT_Ele145_CaloIdVT_GsfTrkIdT);
   fChain->SetBranchAddress("HLT_Ele200_CaloIdVT_GsfTrkIdT", &HLT_Ele200_CaloIdVT_GsfTrkIdT, &b_HLT_Ele200_CaloIdVT_GsfTrkIdT);
   fChain->SetBranchAddress("HLT_Ele250_CaloIdVT_GsfTrkIdT", &HLT_Ele250_CaloIdVT_GsfTrkIdT, &b_HLT_Ele250_CaloIdVT_GsfTrkIdT);
   fChain->SetBranchAddress("HLT_Ele300_CaloIdVT_GsfTrkIdT", &HLT_Ele300_CaloIdVT_GsfTrkIdT, &b_HLT_Ele300_CaloIdVT_GsfTrkIdT);
   fChain->SetBranchAddress("HLT_Photon60_R9Id90_CaloIdL_IsoL", &HLT_Photon60_R9Id90_CaloIdL_IsoL, &b_HLT_Photon60_R9Id90_CaloIdL_IsoL);
   fChain->SetBranchAddress("HLT_Photon60_R9Id90_CaloIdL_IsoL_DisplacedIdL", &HLT_Photon60_R9Id90_CaloIdL_IsoL_DisplacedIdL, &b_HLT_Photon60_R9Id90_CaloIdL_IsoL_DisplacedIdL);
   fChain->SetBranchAddress("HLT_Photon60_R9Id90_CaloIdL_IsoL_DisplacedIdL_PFHT350MinPFJet15", &HLT_Photon60_R9Id90_CaloIdL_IsoL_DisplacedIdL_PFHT350MinPFJet15, &b_HLT_Photon60_R9Id90_CaloIdL_IsoL_DisplacedIdL_PFHT350MinPFJet15);
   fChain->SetBranchAddress("HLT_AK4PFJet30", &HLT_AK4PFJet30, &b_HLT_AK4PFJet30);
   fChain->SetBranchAddress("HLT_AK4PFJet50", &HLT_AK4PFJet50, &b_HLT_AK4PFJet50);
   fChain->SetBranchAddress("HLT_AK4PFJet80", &HLT_AK4PFJet80, &b_HLT_AK4PFJet80);
   fChain->SetBranchAddress("HLT_AK4PFJet100", &HLT_AK4PFJet100, &b_HLT_AK4PFJet100);
   fChain->SetBranchAddress("HLT_AK4PFJet120", &HLT_AK4PFJet120, &b_HLT_AK4PFJet120);
   fChain->SetBranchAddress("HLT_SinglePhoton10_Eta3p1ForPPRef", &HLT_SinglePhoton10_Eta3p1ForPPRef, &b_HLT_SinglePhoton10_Eta3p1ForPPRef);
   fChain->SetBranchAddress("HLT_SinglePhoton20_Eta3p1ForPPRef", &HLT_SinglePhoton20_Eta3p1ForPPRef, &b_HLT_SinglePhoton20_Eta3p1ForPPRef);
   fChain->SetBranchAddress("HLT_SinglePhoton30_Eta3p1ForPPRef", &HLT_SinglePhoton30_Eta3p1ForPPRef, &b_HLT_SinglePhoton30_Eta3p1ForPPRef);
   fChain->SetBranchAddress("HLT_Photon20_HoverELoose", &HLT_Photon20_HoverELoose, &b_HLT_Photon20_HoverELoose);
   fChain->SetBranchAddress("HLT_Photon30_HoverELoose", &HLT_Photon30_HoverELoose, &b_HLT_Photon30_HoverELoose);
   fChain->SetBranchAddress("HLT_Photon50_R9Id90_HE10_IsoM_EBOnly_PFJetsMJJ300DEta3_PFMET50", &HLT_Photon50_R9Id90_HE10_IsoM_EBOnly_PFJetsMJJ300DEta3_PFMET50, &b_HLT_Photon50_R9Id90_HE10_IsoM_EBOnly_PFJetsMJJ300DEta3_PFMET50);
   fChain->SetBranchAddress("HLT_Photon75_R9Id90_HE10_IsoM_EBOnly_PFJetsMJJ300DEta3", &HLT_Photon75_R9Id90_HE10_IsoM_EBOnly_PFJetsMJJ300DEta3, &b_HLT_Photon75_R9Id90_HE10_IsoM_EBOnly_PFJetsMJJ300DEta3);
   fChain->SetBranchAddress("HLT_Photon75_R9Id90_HE10_IsoM_EBOnly_PFJetsMJJ600DEta3", &HLT_Photon75_R9Id90_HE10_IsoM_EBOnly_PFJetsMJJ600DEta3, &b_HLT_Photon75_R9Id90_HE10_IsoM_EBOnly_PFJetsMJJ600DEta3);
   fChain->SetBranchAddress("HLT_PFMET100_PFMHT100_IDTight_PFHT60", &HLT_PFMET100_PFMHT100_IDTight_PFHT60, &b_HLT_PFMET100_PFMHT100_IDTight_PFHT60);
   fChain->SetBranchAddress("HLT_PFMETNoMu100_PFMHTNoMu100_IDTight_PFHT60", &HLT_PFMETNoMu100_PFMHTNoMu100_IDTight_PFHT60, &b_HLT_PFMETNoMu100_PFMHTNoMu100_IDTight_PFHT60);
   fChain->SetBranchAddress("HLT_PFMETTypeOne100_PFMHT100_IDTight_PFHT60", &HLT_PFMETTypeOne100_PFMHT100_IDTight_PFHT60, &b_HLT_PFMETTypeOne100_PFMHT100_IDTight_PFHT60);
   fChain->SetBranchAddress("HLT_Diphoton30_18_R9IdL_AND_HE_AND_IsoCaloId_NoPixelVeto_Mass55", &HLT_Diphoton30_18_R9IdL_AND_HE_AND_IsoCaloId_NoPixelVeto_Mass55, &b_HLT_Diphoton30_18_R9IdL_AND_HE_AND_IsoCaloId_NoPixelVeto_Mass55);
   fChain->SetBranchAddress("HLT_Diphoton30_18_R9IdL_AND_HE_AND_IsoCaloId_NoPixelVeto", &HLT_Diphoton30_18_R9IdL_AND_HE_AND_IsoCaloId_NoPixelVeto, &b_HLT_Diphoton30_18_R9IdL_AND_HE_AND_IsoCaloId_NoPixelVeto);
   fChain->SetBranchAddress("Muon_corrected_pt", Muon_corrected_pt, &b_Muon_corrected_pt);
   fChain->SetBranchAddress("Muon_correctedUp_pt", Muon_correctedUp_pt, &b_Muon_correctedUp_pt);
   fChain->SetBranchAddress("Muon_correctedDown_pt", Muon_correctedDown_pt, &b_Muon_correctedDown_pt);
   fChain->SetBranchAddress("nLepton", &nLepton, &b_nLepton);
   fChain->SetBranchAddress("Lepton_jetPtRelv2", Lepton_jetPtRelv2, &b_Lepton_jetPtRelv2);
   fChain->SetBranchAddress("Lepton_jetNDauCharged", Lepton_jetNDauCharged, &b_Lepton_jetNDauCharged);
   fChain->SetBranchAddress("Lepton_jetIdx", Lepton_jetIdx, &b_Lepton_jetIdx);
   fChain->SetBranchAddress("Lepton_photonIdx", Lepton_photonIdx, &b_Lepton_photonIdx);
   fChain->SetBranchAddress("Lepton_jetRelIso", Lepton_jetRelIso, &b_Lepton_jetRelIso);
   fChain->SetBranchAddress("nlepton", &nlepton, &b_nlepton);
   fChain->SetBranchAddress("lepton_mass", lepton_mass, &b_lepton_mass);
   fChain->SetBranchAddress("lepton_isStandalone", lepton_isStandalone, &b_lepton_isStandalone);
   fChain->SetBranchAddress("lepton_mvaFall17V2noIso_WPL", lepton_mvaFall17V2noIso_WPL, &b_lepton_mvaFall17V2noIso_WPL);
   fChain->SetBranchAddress("lepton_dxyErr", lepton_dxyErr, &b_lepton_dxyErr);
   fChain->SetBranchAddress("lepton_dr03TkSumPt", lepton_dr03TkSumPt, &b_lepton_dr03TkSumPt);
   fChain->SetBranchAddress("lepton_mediumPromptId", lepton_mediumPromptId, &b_lepton_mediumPromptId);
   fChain->SetBranchAddress("lepton_mvaLowPt", lepton_mvaLowPt, &b_lepton_mvaLowPt);
   fChain->SetBranchAddress("lepton_mvaLowPtId", lepton_mvaLowPtId, &b_lepton_mvaLowPtId);
   fChain->SetBranchAddress("lepton_mediumId", lepton_mediumId, &b_lepton_mediumId);
   fChain->SetBranchAddress("lepton_mvaId", lepton_mvaId, &b_lepton_mvaId);
   fChain->SetBranchAddress("lepton_dxy", lepton_dxy, &b_lepton_dxy);
   fChain->SetBranchAddress("lepton_softMva", lepton_softMva, &b_lepton_softMva);
   fChain->SetBranchAddress("lepton_energyErr", lepton_energyErr, &b_lepton_energyErr);
   fChain->SetBranchAddress("lepton_pdgId", lepton_pdgId, &b_lepton_pdgId);
   fChain->SetBranchAddress("lepton_inTimeMuon", lepton_inTimeMuon, &b_lepton_inTimeMuon);
   fChain->SetBranchAddress("lepton_genPartFlav", lepton_genPartFlav, &b_lepton_genPartFlav);
   fChain->SetBranchAddress("lepton_triggerIdLoose", lepton_triggerIdLoose, &b_lepton_triggerIdLoose);
   fChain->SetBranchAddress("lepton_hoe", lepton_hoe, &b_lepton_hoe);
   fChain->SetBranchAddress("lepton_miniPFRelIso_all", lepton_miniPFRelIso_all, &b_lepton_miniPFRelIso_all);
   fChain->SetBranchAddress("lepton_correctedUp_pt", lepton_correctedUp_pt, &b_lepton_correctedUp_pt);
   fChain->SetBranchAddress("lepton_sip3d", lepton_sip3d, &b_lepton_sip3d);
   fChain->SetBranchAddress("lepton_nTrackerLayers", lepton_nTrackerLayers, &b_lepton_nTrackerLayers);
   fChain->SetBranchAddress("lepton_tkIsoId", lepton_tkIsoId, &b_lepton_tkIsoId);
   fChain->SetBranchAddress("lepton_miniIsoId", lepton_miniIsoId, &b_lepton_miniIsoId);
   fChain->SetBranchAddress("lepton_multiIsoId", lepton_multiIsoId, &b_lepton_multiIsoId);
   fChain->SetBranchAddress("lepton_pfRelIso03_all", lepton_pfRelIso03_all, &b_lepton_pfRelIso03_all);
   fChain->SetBranchAddress("lepton_dr03EcalRecHitSumEt", lepton_dr03EcalRecHitSumEt, &b_lepton_dr03EcalRecHitSumEt);
   fChain->SetBranchAddress("lepton_highPtId", lepton_highPtId, &b_lepton_highPtId);
   fChain->SetBranchAddress("lepton_softMvaId", lepton_softMvaId, &b_lepton_softMvaId);
   fChain->SetBranchAddress("lepton_isGlobal", lepton_isGlobal, &b_lepton_isGlobal);
   fChain->SetBranchAddress("lepton_miniPFRelIso_chg", lepton_miniPFRelIso_chg, &b_lepton_miniPFRelIso_chg);
   fChain->SetBranchAddress("lepton_jetRelIso", lepton_jetRelIso, &b_lepton_jetRelIso);
   fChain->SetBranchAddress("lepton_genPartIdx", lepton_genPartIdx, &b_lepton_genPartIdx);
   fChain->SetBranchAddress("lepton_vidNestedWPBitmap", lepton_vidNestedWPBitmap, &b_lepton_vidNestedWPBitmap);
   fChain->SetBranchAddress("lepton_ip3d", lepton_ip3d, &b_lepton_ip3d);
   fChain->SetBranchAddress("lepton_jetPtRelv2", lepton_jetPtRelv2, &b_lepton_jetPtRelv2);
   fChain->SetBranchAddress("lepton_photonIdx", lepton_photonIdx, &b_lepton_photonIdx);
   fChain->SetBranchAddress("lepton_pfRelIso03_chg", lepton_pfRelIso03_chg, &b_lepton_pfRelIso03_chg);
   fChain->SetBranchAddress("lepton_mvaFall17V2Iso", lepton_mvaFall17V2Iso, &b_lepton_mvaFall17V2Iso);
   fChain->SetBranchAddress("lepton_lostHits", lepton_lostHits, &b_lepton_lostHits);
   fChain->SetBranchAddress("lepton_dr03TkSumPtHEEP", lepton_dr03TkSumPtHEEP, &b_lepton_dr03TkSumPtHEEP);
   fChain->SetBranchAddress("lepton_tunepRelPt", lepton_tunepRelPt, &b_lepton_tunepRelPt);
   fChain->SetBranchAddress("lepton_dxybs", lepton_dxybs, &b_lepton_dxybs);
   fChain->SetBranchAddress("lepton_cutBased_HEEP", lepton_cutBased_HEEP, &b_lepton_cutBased_HEEP);
   fChain->SetBranchAddress("lepton_pt", lepton_pt, &b_lepton_pt);
   fChain->SetBranchAddress("lepton_softId", lepton_softId, &b_lepton_softId);
   fChain->SetBranchAddress("lepton_mvaFall17V2noIso_WP90", lepton_mvaFall17V2noIso_WP90, &b_lepton_mvaFall17V2noIso_WP90);
   fChain->SetBranchAddress("lepton_puppiIsoId", lepton_puppiIsoId, &b_lepton_puppiIsoId);
   fChain->SetBranchAddress("lepton_mvaTTH", lepton_mvaTTH, &b_lepton_mvaTTH);
   fChain->SetBranchAddress("lepton_charge", lepton_charge, &b_lepton_charge);
   fChain->SetBranchAddress("lepton_vidNestedWPBitmapHEEP", lepton_vidNestedWPBitmapHEEP, &b_lepton_vidNestedWPBitmapHEEP);
   fChain->SetBranchAddress("lepton_isPFcand", lepton_isPFcand, &b_lepton_isPFcand);
   fChain->SetBranchAddress("lepton_correctedDown_pt", lepton_correctedDown_pt, &b_lepton_correctedDown_pt);
   fChain->SetBranchAddress("lepton_sieie", lepton_sieie, &b_lepton_sieie);
   fChain->SetBranchAddress("lepton_dr03HcalDepth1TowerSumEt", lepton_dr03HcalDepth1TowerSumEt, &b_lepton_dr03HcalDepth1TowerSumEt);
   fChain->SetBranchAddress("lepton_highPurity", lepton_highPurity, &b_lepton_highPurity);
   fChain->SetBranchAddress("lepton_dz", lepton_dz, &b_lepton_dz);
   fChain->SetBranchAddress("lepton_scEtOverPt", lepton_scEtOverPt, &b_lepton_scEtOverPt);
   fChain->SetBranchAddress("lepton_isTracker", lepton_isTracker, &b_lepton_isTracker);
   fChain->SetBranchAddress("lepton_tkRelIso", lepton_tkRelIso, &b_lepton_tkRelIso);
   fChain->SetBranchAddress("lepton_tightId", lepton_tightId, &b_lepton_tightId);
   fChain->SetBranchAddress("lepton_pfRelIso04_all", lepton_pfRelIso04_all, &b_lepton_pfRelIso04_all);
   fChain->SetBranchAddress("lepton_dzErr", lepton_dzErr, &b_lepton_dzErr);
   fChain->SetBranchAddress("lepton_phi", lepton_phi, &b_lepton_phi);
   fChain->SetBranchAddress("lepton_pfIsoId", lepton_pfIsoId, &b_lepton_pfIsoId);
   fChain->SetBranchAddress("lepton_jetNDauCharged", lepton_jetNDauCharged, &b_lepton_jetNDauCharged);
   fChain->SetBranchAddress("lepton_eInvMinusPInv", lepton_eInvMinusPInv, &b_lepton_eInvMinusPInv);
   fChain->SetBranchAddress("lepton_dEsigmaDown", lepton_dEsigmaDown, &b_lepton_dEsigmaDown);
   fChain->SetBranchAddress("lepton_mvaFall17V2noIso", lepton_mvaFall17V2noIso, &b_lepton_mvaFall17V2noIso);
   fChain->SetBranchAddress("lepton_tightCharge", lepton_tightCharge, &b_lepton_tightCharge);
   fChain->SetBranchAddress("lepton_fsrPhotonIdx", lepton_fsrPhotonIdx, &b_lepton_fsrPhotonIdx);
   fChain->SetBranchAddress("lepton_eCorr", lepton_eCorr, &b_lepton_eCorr);
   fChain->SetBranchAddress("lepton_mvaFall17V2Iso_WP80", lepton_mvaFall17V2Iso_WP80, &b_lepton_mvaFall17V2Iso_WP80);
   fChain->SetBranchAddress("lepton_looseId", lepton_looseId, &b_lepton_looseId);
   fChain->SetBranchAddress("lepton_cutBased", lepton_cutBased, &b_lepton_cutBased);
   fChain->SetBranchAddress("lepton_dEscaleUp", lepton_dEscaleUp, &b_lepton_dEscaleUp);
   fChain->SetBranchAddress("lepton_mvaFall17V2Iso_WPL", lepton_mvaFall17V2Iso_WPL, &b_lepton_mvaFall17V2Iso_WPL);
   fChain->SetBranchAddress("lepton_ptErr", lepton_ptErr, &b_lepton_ptErr);
   fChain->SetBranchAddress("lepton_deltaEtaSC", lepton_deltaEtaSC, &b_lepton_deltaEtaSC);
   fChain->SetBranchAddress("lepton_r9", lepton_r9, &b_lepton_r9);
   fChain->SetBranchAddress("lepton_seedGain", lepton_seedGain, &b_lepton_seedGain);
   fChain->SetBranchAddress("lepton_eta", lepton_eta, &b_lepton_eta);
   fChain->SetBranchAddress("lepton_mvaFall17V2Iso_WP90", lepton_mvaFall17V2Iso_WP90, &b_lepton_mvaFall17V2Iso_WP90);
   fChain->SetBranchAddress("lepton_nStations", lepton_nStations, &b_lepton_nStations);
   fChain->SetBranchAddress("lepton_corrected_pt", lepton_corrected_pt, &b_lepton_corrected_pt);
   fChain->SetBranchAddress("lepton_dEsigmaUp", lepton_dEsigmaUp, &b_lepton_dEsigmaUp);
   fChain->SetBranchAddress("lepton_jetIdx", lepton_jetIdx, &b_lepton_jetIdx);
   fChain->SetBranchAddress("lepton_dEscaleDown", lepton_dEscaleDown, &b_lepton_dEscaleDown);
   fChain->SetBranchAddress("lepton_mvaFall17V2noIso_WP80", lepton_mvaFall17V2noIso_WP80, &b_lepton_mvaFall17V2noIso_WP80);
   fChain->SetBranchAddress("lepton_convVeto", lepton_convVeto, &b_lepton_convVeto);
   fChain->SetBranchAddress("lepton_cleanmask", lepton_cleanmask, &b_lepton_cleanmask);
   fChain->SetBranchAddress("lepton_segmentComp", lepton_segmentComp, &b_lepton_segmentComp);
   fChain->SetBranchAddress("lepton_is_tight", lepton_is_tight, &b_lepton_is_tight);
   fChain->SetBranchAddress("lepton_is_real", lepton_is_real, &b_lepton_is_real);
   fChain->SetBranchAddress("nphoton", &nphoton, &b_nphoton);
   fChain->SetBranchAddress("photon_electronVeto", photon_electronVeto, &b_photon_electronVeto);
   fChain->SetBranchAddress("photon_genPartIdx", photon_genPartIdx, &b_photon_genPartIdx);
   fChain->SetBranchAddress("photon_genPartFlav", photon_genPartFlav, &b_photon_genPartFlav);
   fChain->SetBranchAddress("photon_cleanmask", photon_cleanmask, &b_photon_cleanmask);
   fChain->SetBranchAddress("photon_cutBased", photon_cutBased, &b_photon_cutBased);
   fChain->SetBranchAddress("photon_cutBased_Fall17V1Bitmap", photon_cutBased_Fall17V1Bitmap, &b_photon_cutBased_Fall17V1Bitmap);
   fChain->SetBranchAddress("photon_jetIdx", photon_jetIdx, &b_photon_jetIdx);
   fChain->SetBranchAddress("photon_vidNestedWPBitmap", photon_vidNestedWPBitmap, &b_photon_vidNestedWPBitmap);
   fChain->SetBranchAddress("photon_dEscaleUp", photon_dEscaleUp, &b_photon_dEscaleUp);
   fChain->SetBranchAddress("photon_dEsigmaDown", photon_dEsigmaDown, &b_photon_dEsigmaDown);
   fChain->SetBranchAddress("photon_eta", photon_eta, &b_photon_eta);
   fChain->SetBranchAddress("photon_hoe", photon_hoe, &b_photon_hoe);
   fChain->SetBranchAddress("photon_mass", photon_mass, &b_photon_mass);
   fChain->SetBranchAddress("photon_mvaID", photon_mvaID, &b_photon_mvaID);
   fChain->SetBranchAddress("photon_mvaID_Fall17V1p1", photon_mvaID_Fall17V1p1, &b_photon_mvaID_Fall17V1p1);
   fChain->SetBranchAddress("photon_pfRelIso03_all", photon_pfRelIso03_all, &b_photon_pfRelIso03_all);
   fChain->SetBranchAddress("photon_pfRelIso03_chg", photon_pfRelIso03_chg, &b_photon_pfRelIso03_chg);
   fChain->SetBranchAddress("photon_dEscaleDown", photon_dEscaleDown, &b_photon_dEscaleDown);
   fChain->SetBranchAddress("photon_electronIdx", photon_electronIdx, &b_photon_electronIdx);
   fChain->SetBranchAddress("photon_pdgId", photon_pdgId, &b_photon_pdgId);
   fChain->SetBranchAddress("photon_dEsigmaUp", photon_dEsigmaUp, &b_photon_dEsigmaUp);
   fChain->SetBranchAddress("photon_eCorr", photon_eCorr, &b_photon_eCorr);
   fChain->SetBranchAddress("photon_energyErr", photon_energyErr, &b_photon_energyErr);
   fChain->SetBranchAddress("photon_phi", photon_phi, &b_photon_phi);
   fChain->SetBranchAddress("photon_pt", photon_pt, &b_photon_pt);
   fChain->SetBranchAddress("photon_r9", photon_r9, &b_photon_r9);
   fChain->SetBranchAddress("photon_sieie", photon_sieie, &b_photon_sieie);
   fChain->SetBranchAddress("photon_charge", photon_charge, &b_photon_charge);
   fChain->SetBranchAddress("photon_isScEtaEB", photon_isScEtaEB, &b_photon_isScEtaEB);
   fChain->SetBranchAddress("photon_mvaID_WP80", photon_mvaID_WP80, &b_photon_mvaID_WP80);
   fChain->SetBranchAddress("photon_mvaID_WP90", photon_mvaID_WP90, &b_photon_mvaID_WP90);
   fChain->SetBranchAddress("photon_pixelSeed", photon_pixelSeed, &b_photon_pixelSeed);
   fChain->SetBranchAddress("photon_seedGain", photon_seedGain, &b_photon_seedGain);
   fChain->SetBranchAddress("photon_isScEtaEE", photon_isScEtaEE, &b_photon_isScEtaEE);
   fChain->SetBranchAddress("njet", &njet, &b_njet);
   fChain->SetBranchAddress("jet_area", jet_area, &b_jet_area);
   fChain->SetBranchAddress("jet_btagCSVV2", jet_btagCSVV2, &b_jet_btagCSVV2);
   fChain->SetBranchAddress("jet_btagDeepB", jet_btagDeepB, &b_jet_btagDeepB);
   fChain->SetBranchAddress("jet_btagDeepCvB", jet_btagDeepCvB, &b_jet_btagDeepCvB);
   fChain->SetBranchAddress("jet_btagDeepCvL", jet_btagDeepCvL, &b_jet_btagDeepCvL);
   fChain->SetBranchAddress("jet_btagDeepFlavB", jet_btagDeepFlavB, &b_jet_btagDeepFlavB);
   fChain->SetBranchAddress("jet_btagDeepFlavCvB", jet_btagDeepFlavCvB, &b_jet_btagDeepFlavCvB);
   fChain->SetBranchAddress("jet_btagDeepFlavCvL", jet_btagDeepFlavCvL, &b_jet_btagDeepFlavCvL);
   fChain->SetBranchAddress("jet_btagDeepFlavQG", jet_btagDeepFlavQG, &b_jet_btagDeepFlavQG);
   fChain->SetBranchAddress("jet_chEmEF", jet_chEmEF, &b_jet_chEmEF);
   fChain->SetBranchAddress("jet_chFPV0EF", jet_chFPV0EF, &b_jet_chFPV0EF);
   fChain->SetBranchAddress("jet_chHEF", jet_chHEF, &b_jet_chHEF);
   fChain->SetBranchAddress("jet_eta", jet_eta, &b_jet_eta);
   fChain->SetBranchAddress("jet_hfsigmaEtaEta", jet_hfsigmaEtaEta, &b_jet_hfsigmaEtaEta);
   fChain->SetBranchAddress("jet_hfsigmaPhiPhi", jet_hfsigmaPhiPhi, &b_jet_hfsigmaPhiPhi);
   fChain->SetBranchAddress("jet_mass", jet_mass, &b_jet_mass);
   fChain->SetBranchAddress("jet_muEF", jet_muEF, &b_jet_muEF);
   fChain->SetBranchAddress("jet_muonSubtrFactor", jet_muonSubtrFactor, &b_jet_muonSubtrFactor);
   fChain->SetBranchAddress("jet_neEmEF", jet_neEmEF, &b_jet_neEmEF);
   fChain->SetBranchAddress("jet_neHEF", jet_neHEF, &b_jet_neHEF);
   fChain->SetBranchAddress("jet_phi", jet_phi, &b_jet_phi);
   fChain->SetBranchAddress("jet_pt", jet_pt, &b_jet_pt);
   fChain->SetBranchAddress("jet_puIdDisc", jet_puIdDisc, &b_jet_puIdDisc);
   fChain->SetBranchAddress("jet_qgl", jet_qgl, &b_jet_qgl);
   fChain->SetBranchAddress("jet_rawFactor", jet_rawFactor, &b_jet_rawFactor);
   fChain->SetBranchAddress("jet_bRegCorr", jet_bRegCorr, &b_jet_bRegCorr);
   fChain->SetBranchAddress("jet_partonFlavour", jet_partonFlavour, &b_jet_partonFlavour);
   fChain->SetBranchAddress("jet_bRegRes", jet_bRegRes, &b_jet_bRegRes);
   fChain->SetBranchAddress("jet_cRegCorr", jet_cRegCorr, &b_jet_cRegCorr);
   fChain->SetBranchAddress("jet_electronIdx1", jet_electronIdx1, &b_jet_electronIdx1);
   fChain->SetBranchAddress("jet_electronIdx2", jet_electronIdx2, &b_jet_electronIdx2);
   fChain->SetBranchAddress("jet_hfadjacentEtaStripsSize", jet_hfadjacentEtaStripsSize, &b_jet_hfadjacentEtaStripsSize);
   fChain->SetBranchAddress("jet_hfcentralEtaStripSize", jet_hfcentralEtaStripSize, &b_jet_hfcentralEtaStripSize);
   fChain->SetBranchAddress("jet_muonIdx1", jet_muonIdx1, &b_jet_muonIdx1);
   fChain->SetBranchAddress("jet_nMuons", jet_nMuons, &b_jet_nMuons);
   fChain->SetBranchAddress("jet_puId", jet_puId, &b_jet_puId);
   fChain->SetBranchAddress("jet_cleanmask", jet_cleanmask, &b_jet_cleanmask);
   fChain->SetBranchAddress("jet_genJetIdx", jet_genJetIdx, &b_jet_genJetIdx);
   fChain->SetBranchAddress("jet_hadronFlavour", jet_hadronFlavour, &b_jet_hadronFlavour);
   fChain->SetBranchAddress("jet_cRegRes", jet_cRegRes, &b_jet_cRegRes);
   fChain->SetBranchAddress("jet_jetId", jet_jetId, &b_jet_jetId);
   fChain->SetBranchAddress("jet_muonIdx2", jet_muonIdx2, &b_jet_muonIdx2);
   fChain->SetBranchAddress("jet_nElectrons", jet_nElectrons, &b_jet_nElectrons);
   fChain->SetBranchAddress("jet_nConstituents", jet_nConstituents, &b_jet_nConstituents);
   fChain->SetBranchAddress("Jet_pt_raw", Jet_pt_raw, &b_Jet_pt_raw);
   fChain->SetBranchAddress("Jet_pt_nom", Jet_pt_nom, &b_Jet_pt_nom);
   fChain->SetBranchAddress("Jet_mass_raw", Jet_mass_raw, &b_Jet_mass_raw);
   fChain->SetBranchAddress("Jet_mass_nom", Jet_mass_nom, &b_Jet_mass_nom);
   fChain->SetBranchAddress("Jet_corr_JEC", Jet_corr_JEC, &b_Jet_corr_JEC);
   fChain->SetBranchAddress("Jet_corr_JER", Jet_corr_JER, &b_Jet_corr_JER);
   fChain->SetBranchAddress("MET_T1_pt", &MET_T1_pt, &b_MET_T1_pt);
   fChain->SetBranchAddress("MET_T1_phi", &MET_T1_phi, &b_MET_T1_phi);
   fChain->SetBranchAddress("MET_T1Smear_pt", &MET_T1Smear_pt, &b_MET_T1Smear_pt);
   fChain->SetBranchAddress("MET_T1Smear_phi", &MET_T1Smear_phi, &b_MET_T1Smear_phi);
   fChain->SetBranchAddress("Jet_pt_jerUp", Jet_pt_jerUp, &b_Jet_pt_jerUp);
   fChain->SetBranchAddress("Jet_mass_jerUp", Jet_mass_jerUp, &b_Jet_mass_jerUp);
   fChain->SetBranchAddress("MET_T1_pt_jerUp", &MET_T1_pt_jerUp, &b_MET_T1_pt_jerUp);
   fChain->SetBranchAddress("MET_T1_phi_jerUp", &MET_T1_phi_jerUp, &b_MET_T1_phi_jerUp);
   fChain->SetBranchAddress("MET_T1Smear_pt_jerUp", &MET_T1Smear_pt_jerUp, &b_MET_T1Smear_pt_jerUp);
   fChain->SetBranchAddress("MET_T1Smear_phi_jerUp", &MET_T1Smear_phi_jerUp, &b_MET_T1Smear_phi_jerUp);
   fChain->SetBranchAddress("Jet_pt_jesTotalUp", Jet_pt_jesTotalUp, &b_Jet_pt_jesTotalUp);
   fChain->SetBranchAddress("Jet_mass_jesTotalUp", Jet_mass_jesTotalUp, &b_Jet_mass_jesTotalUp);
   fChain->SetBranchAddress("MET_T1_pt_jesTotalUp", &MET_T1_pt_jesTotalUp, &b_MET_T1_pt_jesTotalUp);
   fChain->SetBranchAddress("MET_T1_phi_jesTotalUp", &MET_T1_phi_jesTotalUp, &b_MET_T1_phi_jesTotalUp);
   fChain->SetBranchAddress("MET_T1Smear_pt_jesTotalUp", &MET_T1Smear_pt_jesTotalUp, &b_MET_T1Smear_pt_jesTotalUp);
   fChain->SetBranchAddress("MET_T1Smear_phi_jesTotalUp", &MET_T1Smear_phi_jesTotalUp, &b_MET_T1Smear_phi_jesTotalUp);
   fChain->SetBranchAddress("Jet_pt_jesHEMIssueUp", Jet_pt_jesHEMIssueUp, &b_Jet_pt_jesHEMIssueUp);
   fChain->SetBranchAddress("Jet_mass_jesHEMIssueUp", Jet_mass_jesHEMIssueUp, &b_Jet_mass_jesHEMIssueUp);
   fChain->SetBranchAddress("MET_T1_pt_jesHEMIssueUp", &MET_T1_pt_jesHEMIssueUp, &b_MET_T1_pt_jesHEMIssueUp);
   fChain->SetBranchAddress("MET_T1_phi_jesHEMIssueUp", &MET_T1_phi_jesHEMIssueUp, &b_MET_T1_phi_jesHEMIssueUp);
   fChain->SetBranchAddress("MET_T1Smear_pt_jesHEMIssueUp", &MET_T1Smear_pt_jesHEMIssueUp, &b_MET_T1Smear_pt_jesHEMIssueUp);
   fChain->SetBranchAddress("MET_T1Smear_phi_jesHEMIssueUp", &MET_T1Smear_phi_jesHEMIssueUp, &b_MET_T1Smear_phi_jesHEMIssueUp);
   fChain->SetBranchAddress("MET_T1_pt_unclustEnUp", &MET_T1_pt_unclustEnUp, &b_MET_T1_pt_unclustEnUp);
   fChain->SetBranchAddress("MET_T1_phi_unclustEnUp", &MET_T1_phi_unclustEnUp, &b_MET_T1_phi_unclustEnUp);
   fChain->SetBranchAddress("MET_T1Smear_pt_unclustEnUp", &MET_T1Smear_pt_unclustEnUp, &b_MET_T1Smear_pt_unclustEnUp);
   fChain->SetBranchAddress("MET_T1Smear_phi_unclustEnUp", &MET_T1Smear_phi_unclustEnUp, &b_MET_T1Smear_phi_unclustEnUp);
   fChain->SetBranchAddress("Jet_pt_jerDown", Jet_pt_jerDown, &b_Jet_pt_jerDown);
   fChain->SetBranchAddress("Jet_mass_jerDown", Jet_mass_jerDown, &b_Jet_mass_jerDown);
   fChain->SetBranchAddress("MET_T1_pt_jerDown", &MET_T1_pt_jerDown, &b_MET_T1_pt_jerDown);
   fChain->SetBranchAddress("MET_T1_phi_jerDown", &MET_T1_phi_jerDown, &b_MET_T1_phi_jerDown);
   fChain->SetBranchAddress("MET_T1Smear_pt_jerDown", &MET_T1Smear_pt_jerDown, &b_MET_T1Smear_pt_jerDown);
   fChain->SetBranchAddress("MET_T1Smear_phi_jerDown", &MET_T1Smear_phi_jerDown, &b_MET_T1Smear_phi_jerDown);
   fChain->SetBranchAddress("Jet_pt_jesTotalDown", Jet_pt_jesTotalDown, &b_Jet_pt_jesTotalDown);
   fChain->SetBranchAddress("Jet_mass_jesTotalDown", Jet_mass_jesTotalDown, &b_Jet_mass_jesTotalDown);
   fChain->SetBranchAddress("MET_T1_pt_jesTotalDown", &MET_T1_pt_jesTotalDown, &b_MET_T1_pt_jesTotalDown);
   fChain->SetBranchAddress("MET_T1_phi_jesTotalDown", &MET_T1_phi_jesTotalDown, &b_MET_T1_phi_jesTotalDown);
   fChain->SetBranchAddress("MET_T1Smear_pt_jesTotalDown", &MET_T1Smear_pt_jesTotalDown, &b_MET_T1Smear_pt_jesTotalDown);
   fChain->SetBranchAddress("MET_T1Smear_phi_jesTotalDown", &MET_T1Smear_phi_jesTotalDown, &b_MET_T1Smear_phi_jesTotalDown);
   fChain->SetBranchAddress("Jet_pt_jesHEMIssueDown", Jet_pt_jesHEMIssueDown, &b_Jet_pt_jesHEMIssueDown);
   fChain->SetBranchAddress("Jet_mass_jesHEMIssueDown", Jet_mass_jesHEMIssueDown, &b_Jet_mass_jesHEMIssueDown);
   fChain->SetBranchAddress("MET_T1_pt_jesHEMIssueDown", &MET_T1_pt_jesHEMIssueDown, &b_MET_T1_pt_jesHEMIssueDown);
   fChain->SetBranchAddress("MET_T1_phi_jesHEMIssueDown", &MET_T1_phi_jesHEMIssueDown, &b_MET_T1_phi_jesHEMIssueDown);
   fChain->SetBranchAddress("MET_T1Smear_pt_jesHEMIssueDown", &MET_T1Smear_pt_jesHEMIssueDown, &b_MET_T1Smear_pt_jesHEMIssueDown);
   fChain->SetBranchAddress("MET_T1Smear_phi_jesHEMIssueDown", &MET_T1Smear_phi_jesHEMIssueDown, &b_MET_T1Smear_phi_jesHEMIssueDown);
   fChain->SetBranchAddress("MET_T1_pt_unclustEnDown", &MET_T1_pt_unclustEnDown, &b_MET_T1_pt_unclustEnDown);
   fChain->SetBranchAddress("MET_T1_phi_unclustEnDown", &MET_T1_phi_unclustEnDown, &b_MET_T1_phi_unclustEnDown);
   fChain->SetBranchAddress("MET_T1Smear_pt_unclustEnDown", &MET_T1Smear_pt_unclustEnDown, &b_MET_T1Smear_pt_unclustEnDown);
   fChain->SetBranchAddress("MET_T1Smear_phi_unclustEnDown", &MET_T1Smear_phi_unclustEnDown, &b_MET_T1Smear_phi_unclustEnDown);
   fChain->SetBranchAddress("lumi", &lumi, &b_lumi);
   fChain->SetBranchAddress("photon_selection", &photon_selection, &b_photon_selection);
   fChain->SetBranchAddress("n_loose_mu", &n_loose_mu, &b_n_loose_mu);
   fChain->SetBranchAddress("n_loose_ele", &n_loose_ele, &b_n_loose_ele);
   fChain->SetBranchAddress("n_tight_mu", &n_tight_mu, &b_n_tight_mu);
   fChain->SetBranchAddress("n_tight_ele", &n_tight_ele, &b_n_tight_ele);
   fChain->SetBranchAddress("HEM_issue", &HEM_issue, &b_HEM_issue);
   fChain->SetBranchAddress("mjj", &mjj, &b_mjj);
   fChain->SetBranchAddress("n_leptons", &n_leptons, &b_n_leptons);
   fChain->SetBranchAddress("jet_pt_jesTotalDown", jet_pt_jesTotalDown, &b_jet_pt_jesTotalDown);
   fChain->SetBranchAddress("jet_mass_jesTotalDown", jet_mass_jesTotalDown, &b_jet_mass_jesTotalDown);
   fChain->SetBranchAddress("jet_pt_jesHEMIssueDown", jet_pt_jesHEMIssueDown, &b_jet_pt_jesHEMIssueDown);
   fChain->SetBranchAddress("jet_mass_jesHEMIssueDown", jet_mass_jesHEMIssueDown, &b_jet_mass_jesHEMIssueDown);
   fChain->SetBranchAddress("jet_pt_jerDown", jet_pt_jerDown, &b_jet_pt_jerDown);
   fChain->SetBranchAddress("jet_mass_jerDown", jet_mass_jerDown, &b_jet_mass_jerDown);
   fChain->SetBranchAddress("jet_pt_jesTotalUp", jet_pt_jesTotalUp, &b_jet_pt_jesTotalUp);
   fChain->SetBranchAddress("jet_mass_jesTotalUp", jet_mass_jesTotalUp, &b_jet_mass_jesTotalUp);
   fChain->SetBranchAddress("jet_pt_jesHEMIssueUp", jet_pt_jesHEMIssueUp, &b_jet_pt_jesHEMIssueUp);
   fChain->SetBranchAddress("jet_pt_jerUp", jet_pt_jerUp, &b_jet_pt_jerUp);
   fChain->SetBranchAddress("jet_mass_jerUp", jet_mass_jerUp, &b_jet_mass_jerUp);
   fChain->SetBranchAddress("jet_mass_jesHEMIssueUp", jet_mass_jesHEMIssueUp, &b_jet_mass_jesHEMIssueUp);
   fChain->SetBranchAddress("jet_pt_raw", jet_pt_raw, &b_jet_pt_raw);
   fChain->SetBranchAddress("jet_pt_nom", jet_pt_nom, &b_jet_pt_nom);
   fChain->SetBranchAddress("jet_mass_raw", jet_mass_raw, &b_jet_mass_raw);
   fChain->SetBranchAddress("jet_corr_JEC", jet_corr_JEC, &b_jet_corr_JEC);
   fChain->SetBranchAddress("jet_mass_nom", jet_mass_nom, &b_jet_mass_nom);
   fChain->SetBranchAddress("jet_corr_JER", jet_corr_JER, &b_jet_corr_JER);
   fChain->SetBranchAddress("n_photon", &n_photon, &b_n_photon);
   fChain->SetBranchAddress("photon_isprompt", &photon_isprompt, &b_photon_isprompt);
   fChain->SetBranchAddress("photon_gen_matching", &photon_gen_matching, &b_photon_gen_matching);
   fChain->SetBranchAddress("npu", &npu, &b_npu);
   fChain->SetBranchAddress("ntruepu", &ntruepu, &b_ntruepu);
   fChain->SetBranchAddress("npvs", &npvs, &b_npvs);
   fChain->SetBranchAddress("njets25", &njets25, &b_njets25);
   fChain->SetBranchAddress("puWeight", &puWeight, &b_puWeight);
   fChain->SetBranchAddress("puWeightUp", &puWeightUp, &b_puWeightUp);
   fChain->SetBranchAddress("puWeightDown", &puWeightDown, &b_puWeightDown);

   Notify();
}

Bool_t chain_ajj::Notify()
{
   // The Notify() function is called when a new file is opened. This
   // can be either for a new TTree in a TChain or when when a new TTree
   // is started when using PROOF. It is normally not necessary to make changes
   // to the generated code, but the routine can be extended by the
   // user if needed. The return value is currently not used.

   return kTRUE;
}

void chain_ajj::Show(Long64_t entry)
{
// Print contents of entry.
// If entry is not specified, print current entry
   if (!fChain) return;
   fChain->Show(entry);
}
void chain_ajj::endJob() {
   fout->cd();
   ExTree->Write();
   fout->Write();
   fout->Close();
   delete fout;
}
Int_t chain_ajj::Cut(Long64_t entry)
{
// This function may be called from Loop.
// returns  1 if entry is accepted.
// returns -1 otherwise.
   return 1;
}
#endif // #ifdef chain_ajj_cxx
