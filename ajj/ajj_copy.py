import uproot
import glob
import numpy as np
#import coffea.hist
import hist
from coffea import lookup_tools
from coffea import processor
from coffea.lookup_tools.dense_lookup import dense_lookup
from coffea.jetmet_tools import JECStack, CorrectedJetsFactory
from coffea.nanoevents import NanoEventsFactory, BaseSchema, NanoAODSchema
from coffea.nanoevents.methods import candidate
from coffea.lumi_tools import LumiMask
from coffea.lookup_tools import extractor
from coffea.util import save

import awkward as ak
import numpy as np
import xgboost as xgb
import numba
import hist
import math
import json
import argparse
import pprint
import correctionlib
import matplotlib.pyplot as plt
import pandas
from tqdm import tqdm
import os

#os.environ['HOME']='/afs/cern.ch/user/y/yian/'

parser = argparse.ArgumentParser()

parser.add_argument('--year',dest='year',default='2018')
parser.add_argument('--samples',dest='samples',default='samples.json')
#parser.add_argument('--samples',dest='samples',default='inputfile.txt')
parser.add_argument('--basedir',dest='basedir',default='/afs/cern.ch/user/y/yian/work/DESY_pro/ajj/')
parser.add_argument('--outfile',dest='outfile',type=str,default='outfile.coffea')
parser.add_argument('--nproc',dest='nproc',type=int,default='10')
args = parser.parse_args()
year = args.year

if year == '2016pre' or year == '2016post':
    lumimask = LumiMask(args.basedir+'/lumimasks/Cert_271036-284044_13TeV_Legacy2016_Collisions16_JSON.txt')
elif year == '2017':
    lumimask = LumiMask(args.basedir+'/lumimasks/Cert_294927-306462_13TeV_UL2017_Collisions17_GoldenJSON.txt')
elif year == '2018':
    lumimask = LumiMask(args.basedir+'/lumimasks/Cert_314472-325175_13TeV_Legacy2018_Collisions18_JSON.txt')
else:
    assert(0)

ext = extractor()

if year == '2016pre':
    ext.add_weight_sets(['electronrecosf EGamma_SF2D '+args.basedir+'/data/EGM2D_UL2016preVFP.root'])
    ext.add_weight_sets(['electronrecosfunc EGamma_SF2D_error '+args.basedir+'/data/EGM2D_UL2016preVFP.root'])
    ext.add_weight_sets(['electronidsf EGamma_SF2D '+args.basedir+'/data/Ele_Medium_preVFP_EGM2D.root'])
    ext.add_weight_sets(['electronidsfunc EGamma_SF2D_error '+args.basedir+'/data/Ele_Medium_preVFP_EGM2D.root'])
    ext.add_weight_sets(['electronhltsf EGamma_SF2D '+args.basedir+'/data/electron_hlt_sfs_2016.root'])
    ext.add_weight_sets(['electronhltsfunc EGamma_SF2D_error '+args.basedir+'/data/electron_hlt_sfs_2016.root'])
    ext.add_weight_sets(['muonidsf NUM_TightID_DEN_TrackerMuons_abseta_pt '+args.basedir+'/data/Efficiencies_muon_generalTracks_Z_Run2016_UL_HIPM_ID.root'])
    ext.add_weight_sets(['muonidsfunc NUM_TightID_DEN_TrackerMuons_abseta_pt_error '+args.basedir+'/data/Efficiencies_muon_generalTracks_Z_Run2016_UL_HIPM_ID.root'])
    ext.add_weight_sets(['muonisosf NUM_TightRelIso_DEN_TightIDandIPCut_abseta_pt '+args.basedir+'/data/Efficiencies_muon_generalTracks_Z_Run2016_UL_HIPM_ISO.root'])
    ext.add_weight_sets(['muonisosfunc NUM_TightRelIso_DEN_TightIDandIPCut_abseta_pt_error '+args.basedir+'/data/Efficiencies_muon_generalTracks_Z_Run2016_UL_HIPM_ISO.root'])
    ext.add_weight_sets(['muonhltsf NUM_IsoMu24_or_IsoTkMu24_DEN_CutBasedIdTight_and_PFIsoTight_abseta_pt '+args.basedir+'/data/Efficiencies_muon_generalTracks_Z_Run2016_UL_HIPM_SingleMuonTriggers.root'])
    ext.add_weight_sets(['muonhltsfunc NUM_IsoMu24_or_IsoTkMu24_DEN_CutBasedIdTight_and_PFIsoTight_abseta_pt_error '+args.basedir+'/data/Efficiencies_muon_generalTracks_Z_Run2016_UL_HIPM_SingleMuonTriggers.root'])
    ext.add_weight_sets(['* * '+args.basedir+'/data/2016pre/Summer19UL16APV_V7_MC_L1FastJet_AK4PFchs.jec.txt'])
    ext.add_weight_sets(['* * '+args.basedir+'/data/2016pre/Summer19UL16APV_V7_MC_L2Relative_AK4PFchs.jec.txt'])
    ext.add_weight_sets(['* * '+args.basedir+'/data/2016pre/Summer19UL16APV_V7_MC_L3Absolute_AK4PFchs.jec.txt'])
    ext.add_weight_sets(['* * '+args.basedir+'/data/2016pre/Summer19UL16APV_V7_MC_L2L3Residual_AK4PFchs.jec.txt'])
    ext.add_weight_sets(['* * '+args.basedir+'/data/2016pre/Summer20UL16APV_JRV3_MC_PtResolution_AK4PFchs.jr.txt'])
    ext.add_weight_sets(['* * '+args.basedir+'/data/2016pre/Summer19UL16APV_V7_MC_Uncertainty_AK4PFchs.junc.txt'])
    ext.add_weight_sets(['* * '+args.basedir+'/data/2016pre/Summer20UL16APV_JRV3_MC_SF_AK4PFchs.jersf.txt'])
elif year == '2016post':
    ext.add_weight_sets(['electronrecosf EGamma_SF2D '+args.basedir+'/data/EGM2D_UL2016postVFP.root'])
    ext.add_weight_sets(['electronrecosfunc EGamma_SF2D_error '+args.basedir+'/data/EGM2D_UL2016postVFP.root'])
    ext.add_weight_sets(['electronidsf EGamma_SF2D '+args.basedir+'/data/Ele_Medium_postVFP_EGM2D.root'])
    ext.add_weight_sets(['electronidsfunc EGamma_SF2D_error '+args.basedir+'/data/Ele_Medium_postVFP_EGM2D.root'])
    ext.add_weight_sets(['electronhltsf EGamma_SF2D '+args.basedir+'/data/electron_hlt_sfs_2016.root'])
    ext.add_weight_sets(['electronhltsfunc EGamma_SF2D_error '+args.basedir+'/data/electron_hlt_sfs_2016.root'])
    ext.add_weight_sets(['muonidsf NUM_TightID_DEN_TrackerMuons_abseta_pt '+args.basedir+'/data/Efficiencies_muon_generalTracks_Z_Run2016_UL_ID.root'])
    ext.add_weight_sets(['muonidsfunc NUM_TightID_DEN_TrackerMuons_abseta_pt_error '+args.basedir+'/data/Efficiencies_muon_generalTracks_Z_Run2016_UL_ID.root'])
    ext.add_weight_sets(['muonisosf NUM_TightRelIso_DEN_TightIDandIPCut_abseta_pt '+args.basedir+'/data/Efficiencies_muon_generalTracks_Z_Run2016_UL_ISO.root'])
    ext.add_weight_sets(['muonisosfunc NUM_TightRelIso_DEN_TightIDandIPCut_abseta_pt_error '+args.basedir+'/data/Efficiencies_muon_generalTracks_Z_Run2016_UL_ISO.root'])
    ext.add_weight_sets(['muonhltsf NUM_IsoMu24_or_IsoTkMu24_DEN_CutBasedIdTight_and_PFIsoTight_abseta_pt '+args.basedir+'/data/Efficiencies_muon_generalTracks_Z_Run2016_UL_SingleMuonTriggers.root'])
    ext.add_weight_sets(['muonhltsfunc NUM_IsoMu24_or_IsoTkMu24_DEN_CutBasedIdTight_and_PFIsoTight_abseta_pt_error '+args.basedir+'/data/Efficiencies_muon_generalTracks_Z_Run2016_UL_SingleMuonTriggers.root'])
    ext.add_weight_sets(['* * '+args.basedir+'/data/2016post/Summer19UL16_V7_MC_L1FastJet_AK4PFchs.jec.txt'])
    ext.add_weight_sets(['* * '+args.basedir+'/data/2016post/Summer19UL16_V7_MC_L2Relative_AK4PFchs.jec.txt'])
    ext.add_weight_sets(['* * '+args.basedir+'/data/2016post/Summer19UL16_V7_MC_L3Absolute_AK4PFchs.jec.txt'])
    ext.add_weight_sets(['* * '+args.basedir+'/data/2016post/Summer19UL16_V7_MC_L2L3Residual_AK4PFchs.jec.txt'])
    ext.add_weight_sets(['* * '+args.basedir+'/data/2016post/Summer20UL16_JRV3_MC_PtResolution_AK4PFchs.jr.txt'])
    ext.add_weight_sets(['* * '+args.basedir+'/data/2016post/Summer19UL16_V7_MC_Uncertainty_AK4PFchs.junc.txt'])
    ext.add_weight_sets(['* * '+args.basedir+'/data/2016post/Summer20UL16_JRV3_MC_SF_AK4PFchs.jersf.txt'])
elif year == '2017':
    ext.add_weight_sets(['electronrecosf EGamma_SF2D '+args.basedir+'/data/EGM2D_UL2017.root'])
    ext.add_weight_sets(['electronrecosfunc EGamma_SF2D_error '+args.basedir+'/data/EGM2D_UL2017.root'])
    ext.add_weight_sets(['electronidsf EGamma_SF2D '+args.basedir+'/data/EGM2D_Medium_UL17.root'])
    ext.add_weight_sets(['electronidsfunc EGamma_SF2D_error '+args.basedir+'/data/EGM2D_Medium_UL17.root'])
    ext.add_weight_sets(['electronhltsf EGamma_SF2D '+args.basedir+'/data/electron_hlt_sfs_2017.root'])
    ext.add_weight_sets(['electronhltsfunc EGamma_SF2D_error '+args.basedir+'/data/electron_hlt_sfs_2017.root'])
    ext.add_weight_sets(['muonidsf NUM_TightID_DEN_TrackerMuons_abseta_pt '+args.basedir+'/data/Efficiencies_muon_generalTracks_Z_Run2017_UL_ID.root'])
    ext.add_weight_sets(['muonidsfunc NUM_TightID_DEN_TrackerMuons_abseta_pt_error '+args.basedir+'/data/Efficiencies_muon_generalTracks_Z_Run2017_UL_ID.root'])
    ext.add_weight_sets(['muonisosf NUM_TightRelIso_DEN_TightIDandIPCut_abseta_pt '+args.basedir+'/data/Efficiencies_muon_generalTracks_Z_Run2017_UL_ISO.root'])
    ext.add_weight_sets(['muonisosfunc NUM_TightRelIso_DEN_TightIDandIPCut_abseta_pt_error '+args.basedir+'/data/Efficiencies_muon_generalTracks_Z_Run2017_UL_ISO.root'])
    ext.add_weight_sets(['muonhltsf NUM_IsoMu27_DEN_CutBasedIdTight_and_PFIsoTight_abseta_pt '+args.basedir+'/data/Efficiencies_muon_generalTracks_Z_Run2017_UL_SingleMuonTriggers.root'])
    ext.add_weight_sets(['muonhltsfunc NUM_IsoMu27_DEN_CutBasedIdTight_and_PFIsoTight_abseta_pt_error '+args.basedir+'/data/Efficiencies_muon_generalTracks_Z_Run2017_UL_SingleMuonTriggers.root'])
    ext.add_weight_sets(['* * '+args.basedir+'/data/2017/Summer19UL17_V5_MC_L1FastJet_AK4PFchs.jec.txt'])
    ext.add_weight_sets(['* * '+args.basedir+'/data/2017/Summer19UL17_V5_MC_L2Relative_AK4PFchs.jec.txt'])
    ext.add_weight_sets(['* * '+args.basedir+'/data/2017/Summer19UL17_V5_MC_L3Absolute_AK4PFchs.jec.txt'])
    ext.add_weight_sets(['* * '+args.basedir+'/data/2017/Summer19UL17_V5_MC_L2L3Residual_AK4PFchs.jec.txt'])
    ext.add_weight_sets(['* * '+args.basedir+'/data/2017/Summer19UL17_JRV2_MC_PtResolution_AK4PFchs.jr.txt'])
    ext.add_weight_sets(['* * '+args.basedir+'/data/2017/Summer19UL17_V5_MC_Uncertainty_AK4PFchs.junc.txt'])
    ext.add_weight_sets(['* * '+args.basedir+'/data/2017/Summer19UL17_JRV2_MC_SF_AK4PFchs.jersf.txt'])
elif year == '2018':
    ext.add_weight_sets(['electronrecosf EGamma_SF2D '+args.basedir+'/data/EGM2D_UL2018.root'])
    ext.add_weight_sets(['electronrecosfunc EGamma_SF2D_error '+args.basedir+'/data/EGM2D_UL2018.root'])
    ext.add_weight_sets(['electronidsf EGamma_SF2D '+args.basedir+'/data/Ele_Medium_EGM2D.root'])
    ext.add_weight_sets(['electronidsfunc EGamma_SF2D_error '+args.basedir+'/data/Ele_Medium_EGM2D.root'])
    ext.add_weight_sets(['electronhltsf EGamma_SF2D '+args.basedir+'/data/electron_hlt_sfs_2018.root'])
    ext.add_weight_sets(['electronhltsfunc EGamma_SF2D_error '+args.basedir+'/data/electron_hlt_sfs_2018.root'])
    ext.add_weight_sets(['muonidsf NUM_TightID_DEN_TrackerMuons_abseta_pt '+args.basedir+'/data/Efficiencies_muon_generalTracks_Z_Run2018_UL_ID.root'])
    ext.add_weight_sets(['muonidsfunc NUM_TightID_DEN_TrackerMuons_abseta_pt_error '+args.basedir+'/data/Efficiencies_muon_generalTracks_Z_Run2018_UL_ID.root'])
    ext.add_weight_sets(['muonisosf NUM_TightRelIso_DEN_TightIDandIPCut_abseta_pt '+args.basedir+'/data/Efficiencies_muon_generalTracks_Z_Run2018_UL_ISO.root'])
    ext.add_weight_sets(['muonisosfunc NUM_TightRelIso_DEN_TightIDandIPCut_abseta_pt_error '+args.basedir+'/data/Efficiencies_muon_generalTracks_Z_Run2018_UL_ISO.root'])
    ext.add_weight_sets(['muonhltsf NUM_IsoMu24_DEN_CutBasedIdTight_and_PFIsoTight_abseta_pt '+args.basedir+'/data/Efficiencies_muon_generalTracks_Z_Run2018_UL_SingleMuonTriggers.root'])
    ext.add_weight_sets(['muonhltsfunc NUM_IsoMu24_DEN_CutBasedIdTight_and_PFIsoTight_abseta_pt_error '+args.basedir+'/data/Efficiencies_muon_generalTracks_Z_Run2018_UL_SingleMuonTriggers.root'])
    ext.add_weight_sets(['* * '+args.basedir+'/data/2018/Summer19UL18_V5_MC_L1FastJet_AK4PFchs.jec.txt'])
    ext.add_weight_sets(['* * '+args.basedir+'/data/2018/Summer19UL18_V5_MC_L2Relative_AK4PFchs.jec.txt'])
    ext.add_weight_sets(['* * '+args.basedir+'/data/2018/Summer19UL18_V5_MC_L3Absolute_AK4PFchs.jec.txt'])
    ext.add_weight_sets(['* * '+args.basedir+'/data/2018/Summer19UL18_V5_MC_L2L3Residual_AK4PFchs.jec.txt'])
    ext.add_weight_sets(['* * '+args.basedir+'/data/2018/Summer19UL18_JRV2_MC_PtResolution_AK4PFchs.jr.txt'])
    ext.add_weight_sets(['* * '+args.basedir+'/data/2018/Summer19UL18_V5_MC_Uncertainty_AK4PFchs.junc.txt'])
    ext.add_weight_sets(['* * '+args.basedir+'/data/2018/Summer19UL18_JRV2_MC_SF_AK4PFchs.jersf.txt'])


ext.add_weight_sets(['pileup ratio_{} '.format(year.split('pre')[0].split('post')[0])+args.basedir+'/data/pileup.root'])
ext.add_weight_sets(['pileup_up ratio_{}_up '.format(year.split('pre')[0].split('post')[0])+args.basedir+'/data/pileup.root'])
ext.add_weight_sets(['pileup_down ratio_{}_down '.format(year.split('pre')[0].split('post')[0])+args.basedir+'/data/pileup.root'])

ext.finalize()

evaluator = ext.make_evaluator()

if year == '2016pre':
    jec_stack_names = ['Summer19UL16APV_V7_MC_L1FastJet_AK4PFchs','Summer19UL16APV_V7_MC_L2Relative_AK4PFchs','Summer19UL16APV_V7_MC_L3Absolute_AK4PFchs','Summer19UL16APV_V7_MC_L2L3Residual_AK4PFchs','Summer20UL16APV_JRV3_MC_PtResolution_AK4PFchs','Summer19UL16APV_V7_MC_Uncertainty_AK4PFchs','Summer20UL16APV_JRV3_MC_SF_AK4PFchs']
elif year == '2016post':
    jec_stack_names = ['Summer19UL16_V7_MC_L1FastJet_AK4PFchs','Summer19UL16_V7_MC_L2Relative_AK4PFchs','Summer19UL16_V7_MC_L3Absolute_AK4PFchs','Summer19UL16_V7_MC_L2L3Residual_AK4PFchs','Summer20UL16_JRV3_MC_PtResolution_AK4PFchs','Summer19UL16_V7_MC_Uncertainty_AK4PFchs','Summer20UL16_JRV3_MC_SF_AK4PFchs']
elif year == '2017':
    jec_stack_names = ['Summer19UL17_V5_MC_L1FastJet_AK4PFchs','Summer19UL17_V5_MC_L2Relative_AK4PFchs','Summer19UL17_V5_MC_L3Absolute_AK4PFchs','Summer19UL17_V5_MC_L2L3Residual_AK4PFchs','Summer19UL17_JRV2_MC_PtResolution_AK4PFchs','Summer19UL17_V5_MC_Uncertainty_AK4PFchs','Summer19UL17_JRV2_MC_SF_AK4PFchs']
elif year == '2018':
    jec_stack_names = ['Summer19UL18_V5_MC_L1FastJet_AK4PFchs','Summer19UL18_V5_MC_L2Relative_AK4PFchs','Summer19UL18_V5_MC_L3Absolute_AK4PFchs','Summer19UL18_V5_MC_L2L3Residual_AK4PFchs','Summer19UL18_JRV2_MC_PtResolution_AK4PFchs','Summer19UL18_V5_MC_Uncertainty_AK4PFchs','Summer19UL18_JRV2_MC_SF_AK4PFchs']

jec_inputs = {name: evaluator[name] for name in jec_stack_names}

jec_stack = JECStack(jec_inputs)

hists_for_fit = ['sel3_bdtscore_binning1','sel6_vbfdijetmass_binning1']

hist_definitions = [
    ['sel1_boson_pt','Z pT',20,70,400],
    ['sel1_boson_mass','Z mass',40,70,110],
    ['sel1_mjj','mjj',20,500,2000],
    ['sel2_boson_pt','Z pT',20,70,400],
    ['sel2_boson_mass','Z mass',40,70,110],
    ['sel2_mjj','mjj',21,500,2000],
]

class ajjProcessor(processor.ProcessorABC):
    def __init__(self):
        self._accumulator = processor.dict_accumulator({
            'sumw': processor.defaultdict_accumulator(float),
            'sumsign': processor.defaultdict_accumulator(float),
            'nevents': processor.defaultdict_accumulator(float),
        })

        for hist_definition in hist_definitions:
            self._accumulator[hist_definition[0]] = hist.Hist(hist.axis.StrCategory([], growth=True, name="dataset", label="Primary dataset"),hist.axis.Regular(hist_definition[2], hist_definition[3], hist_definition[4], name="variable"),storage="weight")


    @property
    def accumulator(self):
        return self._accumulator

    def process(self, events):

        def fill_histogram(weights,variables,dataset,selections,output):
            
            for sel in selections:
                for variable in variables.keys():
                        print(variable,' ',sel+'_'+variable)
                        if 'nom' in weights.keys():
                            output[sel+'_'+variable][dataset].fill(
                                dataset=dataset,
                                variable=variables[variable],
                                weight=weights['nom']
                            )
                                
        output = self.accumulator
        dataset = events.metadata['dataset']
        print('!!!!!!!!!!!!######### dataset ',dataset)

        if dataset not in ['singleelectron','singlemuon','egamma']:
            output['sumw'][dataset] += ak.sum(events.Generator.weight)
            output['sumsign'][dataset] += ak.sum(np.sign(events.Generator.weight))
        output['nevents'][dataset] += len(events)

        if dataset in ['singleelectron','singlemuon','egamma']:
            events = events[lumimask(events.run,events.luminosityBlock)]

        events = events[(events.PuppiMET.pt > 30) | (events.PuppiMET.ptJERUp > 30) | (events.PuppiMET.ptJESUp > 30)]    

        if year == '2016pre' or year == '2016post':
            if dataset == 'singlemuon':
                events = events[events.HLT.IsoTkMu24 | events.HLT.IsoMu24]
            elif dataset == 'singleelectron':
                events = events[events.HLT.Ele27_WPTight_Gsf & ~(events.HLT.IsoTkMu24 | events.HLT.IsoMu24)]
            else:    
                events = events[events.HLT.IsoTkMu24 | events.HLT.IsoMu24 | events.HLT.Ele27_WPTight_Gsf]
            muon_pt_cut = 26
            electron_pt_cut = 30
        elif year == '2017':
            if dataset == 'singlemuon':
                events = events[events.HLT.IsoMu27]
            elif dataset == 'singleelectron':
                events = events[events.HLT.Ele32_WPTight_Gsf_L1DoubleEG & ~events.HLT.IsoMu27]
            else:
                events = events[events.HLT.IsoMu27 | events.HLT.Ele32_WPTight_Gsf_L1DoubleEG]
            muon_pt_cut = 30
            electron_pt_cut = 35
        elif year == '2018':
            if dataset == 'singlemuon':
                events = events[events.HLT.IsoMu24]
            elif dataset == 'egamma':    
                events = events[events.HLT.Ele32_WPTight_Gsf & ~events.HLT.IsoMu24]
            else:
                events = events[events.HLT.IsoMu24 |events.HLT.Ele32_WPTight_Gsf]
            muon_pt_cut = 26
            electron_pt_cut = 35

        if year == '2016pre' or year == '2016post':
            events = events[events.Flag.goodVertices & events.Flag.globalSuperTightHalo2016Filter & events.Flag.HBHENoiseFilter &  events.Flag.HBHENoiseIsoFilter & events.Flag.EcalDeadCellTriggerPrimitiveFilter & events.Flag.BadPFMuonFilter & events.Flag.BadPFMuonDzFilter & events.Flag.eeBadScFilter]
        elif year == '2017' or year == '2018':
            events = events[events.Flag.goodVertices & events.Flag.globalSuperTightHalo2016Filter & events.Flag.HBHENoiseFilter & events.Flag.HBHENoiseIsoFilter & events.Flag.EcalDeadCellTriggerPrimitiveFilter & events.Flag.BadPFMuonFilter & events.Flag.BadPFMuonDzFilter & events.Flag.eeBadScFilter & events.Flag.ecalBadCalibFilter]

        events = events[ (ak.num(events.Jet) > 1) ]
 
        if dataset not in ['singleelectron','singlemuon','egamma']:

            name_map = jec_stack.blank_name_map
            name_map['JetPt'] = 'pt'
            name_map['JetMass'] = 'mass'
            name_map['JetEta'] = 'eta'
            name_map['JetA'] = 'area'

            jets = events.Jet
        
            jets['pt_raw'] = (1 - jets['rawFactor']) * jets['pt']
            jets['mass_raw'] = (1 - jets['rawFactor']) * jets['mass']
            jets['pt_gen'] = ak.values_astype(ak.fill_none(jets.matched_gen.pt, 0), np.float32)
            jets['rho'] = ak.broadcast_arrays(events.fixedGridRhoFastjetAll, jets.pt)[0]
            name_map['ptGenJet'] = 'pt_gen'
            name_map['ptRaw'] = 'pt_raw'
            name_map['massRaw'] = 'mass_raw'
            name_map['Rho'] = 'rho'

            events_cache = events.caches[0]

            jet_factory = CorrectedJetsFactory(name_map, jec_stack)
            corrected_jets = jet_factory.build(jets, lazy_cache=events_cache)    

            jet_pt = corrected_jets.pt
            jet_pt_jesup = corrected_jets.JES_jes.up.pt
            jet_pt_jerup = corrected_jets.JER.up.pt
            jet_pt_jesdn = corrected_jets.JES_jes.down.pt
            jet_pt_jerdn = corrected_jets.JER.down.pt

        else:    

            corrected_jets = events.Jet
            jet_pt = events.Jet.pt
            jet_pt_jesup = events.Jet.pt
            jet_pt_jerup = events.Jet.pt
            jet_pt_jesdn = events.Jet.pt
            jet_pt_jerdn = events.Jet.pt

        tight_muons = events.Muon[events.Muon.tightId & (events.Muon.pfRelIso04_all < 0.15) & (events.Muon.pt > 10) & (abs(events.Muon.eta) < 2.4)]
        tight_electrons = events.Electron[(events.Electron.pt > 10) & (events.Electron.mvaFall17V2Iso_WP80) & (abs(events.Electron.eta + events.Electron.deltaEtaSC) < 2.5) & (((abs(events.Electron.dz) < 0.1) & (abs(events.Electron.dxy) < 0.05) & (abs(events.Electron.eta + events.Electron.deltaEtaSC) < 1.479)) | ((abs(events.Electron.dz) < 0.2) & (abs(events.Electron.dxy) < 0.1) & (abs(events.Electron.eta + events.Electron.deltaEtaSC) > 1.479)))]
        tight_photons=events.Photon[(ak.num(events.Photon)>0) & (events.Photon.pt>70) & (events.Photon.cutBased==3) & (events.Photon.pixelSeed==0) ]
        tight_jets=corrected_jets[(corrected_jets.jetId==6) & (corrected_jets.pt>30) & (abs(corrected_jets.eta)<4.7)]
        tight_jets_jesdn=corrected_jets[(corrected_jets.jetId==6) & (jet_pt_jesdn>30) & (abs(corrected_jets.eta)<4.7)]
        tight_jets_jerdn=corrected_jets[(corrected_jets.jetId==6) & (jet_pt_jerdn>30) & (abs(corrected_jets.eta)<4.7)]
        tight_jets_jesup=corrected_jets[(corrected_jets.jetId==6) & (jet_pt_jesup>30) & (abs(corrected_jets.eta)<4.7)]
        tight_jets_jerup=corrected_jets[(corrected_jets.jetId==6) & (jet_pt_jerup>30) & (abs(corrected_jets.eta)<4.7)]
        
        tight_jets = tight_jets[(ak.num(tight_jets)>1)]
        tight_muons= tight_muons[(ak.num(tight_jets)>1)]
        tight_electrons= tight_muons[(ak.num(tight_jets)>1)]
        basejet_cut = ( (tight_jets[:,0]+tight_jets[:,1]).mass > 500 )

### For mm
        CRmm = ( (ak.num(tight_muons)>1) & (ak.num(tight_jets)>1) ) 
        sel1_muons=tight_muons[CRmm & basejet_cut]
        sel1_events=events[CRmm & basejet_cut]
        sel1_jets=tight_jets[CRmm & basejet_cut]

        CRmm_cut = ( (abs((sel1_muons[:,0]+sel1_muons[:,1]).mass-91)<15) & ((sel1_muons[:,0]+sel1_muons[:,1]).pt>70) )
        sel1_muons=sel1_muons[CRmm_cut] 
        sel1_jets=sel1_jets[CRmm_cut]
        sel1_events=sel1_events[CRmm_cut]

### For ee 
        CRee = ( (ak.num(tight_electrons)>1) & (ak.num(tight_jets)>1) ) 
        sel2_eles  = tight_electrons[CRee & basejet_cut]
        sel2_events=events[CRee & basejet_cut]
        sel2_jets  =tight_jets[CRee & basejet_cut]
      
        CRee_cut= ( (abs((sel2_eles[:,0]+sel2_eles[:,1]).mass-91)<15) & ((sel2_eles[:,0]+sel2_eles[:,1]).pt>70) )
        sel2_eles = sel2_eles[CRee_cut]
        sel2_jets = sel2_jets[CRee_cut]
        sel2_events = sel2_events[CRee_cut]

        sel1_weight = {}
        sel1_variable = {}

        sel1_variable['boson_mass']=(sel1_muons[:,0]+sel1_muons[:,1]).mass
        sel1_variable['boson_pt']=(sel1_muons[:,0]+sel1_muons[:,1]).pt
        sel1_variable['mjj']=(sel1_jets[:,0]+sel1_jets[:,1]).mass
#        sel2_variable['boson_mass']=(sel2_eles[:,0]+sel2_eles[:,1]).mass
#        sel2_variable['boson_pt']=(sel2_eles[:,0]+sel2_eles[:,1]).pt
#        sel2_variable['mjj']=(sel2_jets[:,0]+sel2_jets[:,1]).mass

#        if dataset == 'data':
        if dataset in ['singleelectron','singlemuon','egamma']:
           sel1_weight['nom'] = np.ones(len(sel1_events))
        else:
           gen_weight = np.sign(sel1_events.Generator.weight)
           sel1_puweight = evaluator['pileup'](sel1_events.Pileup.nTrueInt)
           sel1_muonidsf = evaluator['muonidsf'](abs(sel1_muons[:,0].eta), sel1_muons[:,0].pt)*evaluator['muonidsf'](abs(sel1_muons[:,1].eta), sel1_muons[:,1].pt)
           sel1_muonisosf = evaluator['muonisosf'](abs(sel1_muons[:,0].eta), sel1_muons[:,0].pt)*evaluator['muonisosf'](abs(sel1_muons[:,1].eta), sel1_muons[:,1].pt)
           sel1_weight['nom'] = gen_weight*sel1_puweight*sel1_events.L1PreFiringWeight.Nom*sel1_muonidsf*sel1_muonisosf
    
        fill_histogram(sel1_weight,sel1_variable,dataset,['sel1'],output)

        return output

    def postprocess(self, accumulator):
#        return accumulator
        pass

f_samples=open(args.basedir+'/'+args.samples)

samples=json.loads(f_samples.read())

for sample in samples:
    f = open(args.basedir+'/filelists/'+year+'/'+samples[sample]['filelist'])
    frac = samples[sample]['frac']
    mod = samples[sample]['mod']
    residues = samples[sample]['residues']
    samples[sample] = []
    filelist = f.read().rstrip('\n').split('\n')
    if mod != None:
        for i in range(int(len(filelist)*frac)):
            if i % mod in residues:
                samples[sample].append(filelist[i])
    else:
        samples[sample] = filelist[0:int(frac*len(filelist))]

#result = processor.run_uproot_job(
#    samples,
#    'Events',
#    ajjProcessor(),
#    processor.futures_executor,
#    {'schema': NanoAODSchema},
##    {'schema': NanoAODSchema, 'workers': args.nproc},
#    chunksize=10000000,
#)

run = processor.Runner(
    executor = processor.FuturesExecutor(compression=None, workers=args.nproc),
    schema=NanoAODSchema,
    chunksize=100_000,
    # maxchunks=10,  # total 676 chunks
)
result = run(
    samples,
    "Events",
    processor_instance=ajjProcessor(),
)

print(result)
print('\n')
for key in result['nevents'].keys():
    print('result[\'nevents\'][\'{}\'] = {}'.format(key,result['nevents'][key]))

save(result,args.outfile)
