// main01.cc is a part of the PYTHIA event generator.
// Copyright (C) 2009 Torbjorn Sjostrand.
// PYTHIA is licenced under the GNU GPL version 2, see COPYING for details.
// Please respect the MCnet Guidelines, see GUIDELINES for details.


#include "TFile.h"
#include "Pythia8/Pythia.h"
#include "TH1D.h"
#include "TH2D.h"
#include "TNtuple.h"
#include "TMath.h"
#include "TCanvas.h"
#include "TLorentzVector.h"
#include "TGraph.h"
#include "TTree.h"

#include "Pythia8/HeavyIons.h"
#include <fastjet/JetDefinition.hh>
#include <fastjet/PseudoJet.hh>
#include <fastjet/ClusterSequence.hh>
#include <vector>

//#include <boost/foreach.hpp>

using namespace Pythia8; 
using namespace std;

static const double pi = acos(-1.0);

int main() {

  int com_energy = 5020;
  int minpthat = 5;
  int nevents = 1000;
  double min_jet_pt = 5;
  double max_abs_y = 2.1;
  int index = 1;
  bool doHI = false;
  int ionPID = 2212;

  bool write_ntuple = true;

  std::vector<double> vjetch2pt;
  std::vector<double> vjetch2eta;
  std::vector<double> vjetch2phi;
  std::vector<double> vjetch4pt;
  std::vector<double> vjetch4eta;
  std::vector<double> vjetch4phi;
  std::vector<double> vjetch6pt;
  std::vector<double> vjetch6eta;
  std::vector<double> vjetch6phi;
  std::vector<double> vjetpt;
  std::vector<double> vjetphi;
  std::vector<double> vjeteta;
  std::vector<double> vjety;
  std::vector<double> vjete;
  std::vector<double> vmult_EPD_pt;
  std::vector<double> vmult_EPD_eta;
  std::vector<double> vmult_EPD_phi;

  double x1, x2, weight;


  //jet_tree->Branch("vjetmult",&vjetmult);

//  cout << "center of mass energy and min pt (both in GeV), nevents, index, heavy ion, ion" << endl;
//  cin >> com_energy >> minpthat >> nevents >> index >> doHI >> ionPID;
cout << "center of mass energy, min pt (previous both in Gev), nevents, run number" << endl;
cin >> com_energy >> minpthat >> nevents >> index; // what this does is ask for changes to com, minpt, nevents, run number. and heavyion, and then you input it
 
  if(com_energy==5020){
     min_jet_pt = 20;
     max_abs_y = 4.9;
  }
  if(com_energy==5360){
     min_jet_pt = 20;
     max_abs_y = 2.1; //changing this on 10/30/25  I think that the rapidity cut for the real plots for 5.02 TeV might have actually been 4.9
  }
  if(com_energy==8160){
     min_jet_pt = 30;
     max_abs_y = 4.9;
  }
  if(com_energy==200){
     min_jet_pt = 5;
     max_abs_y = 4.9;
  }

  TTree *jet_tree = new TTree("jet_tree","Tree with vectors");
  jet_tree->Branch("vjetch2pt",&vjetch2pt);
  jet_tree->Branch("vjetch2eta",&vjetch2eta);
  jet_tree->Branch("vjetch2phi",&vjetch2phi);
  jet_tree->Branch("vjetch4pt",&vjetch4pt);
  jet_tree->Branch("vjetch4eta",&vjetch4eta);
  jet_tree->Branch("vjetch4phi",&vjetch4phi);
  jet_tree->Branch("vjetch6pt",&vjetch6pt);
  jet_tree->Branch("vjetch6eta",&vjetch6eta);
  jet_tree->Branch("vjetch6phi",&vjetch6phi);
  jet_tree->Branch("vjetpt",&vjetpt);
  jet_tree->Branch("vjeteta",&vjeteta);
  jet_tree->Branch("vjetphi",&vjetphi);
  jet_tree->Branch("vjety",&vjety);
  jet_tree->Branch("vjete",&vjete);
  jet_tree->Branch("x1",&x1);
  jet_tree->Branch("x2",&x2);
  jet_tree->Branch("weight",&weight);
  if(com_energy == 200){
     jet_tree->Branch("ch_EPD_pt",&vmult_EPD_pt);
     jet_tree->Branch("ch_EPD_eta",&vmult_EPD_eta);
     jet_tree->Branch("ch_EPD_phi",&vmult_EPD_phi);
  }

  cout << "min jet pt " << min_jet_pt << " " << max_abs_y << " " << com_energy << endl;
  ostringstream tmp1, tmp2;
  tmp1 << "Beams:eCM = " << com_energy;
  tmp2 << "PhaseSpace:pTHatMin = " << minpthat;
  // Generator. Process selection. LHC initialization. Histogram.
  //Pythia pythia("/Users/sickles/work/comp/pythia/pythia8308/share/Pythia8/xmldoc");
  Pythia pythia("/home/keeg/pythia/pythia8317");
  pythia.readString(tmp1.str());
  pythia.readString("HardQCD:all = on");    
  pythia.readString(tmp2.str());  
  pythia.readString("Random:setseed = on");
  pythia.readString("Random:seed = 0");

  if(doHI){
     ostringstream tmp3, tmp4;
     tmp3 << "Beams:idA = " << ionPID;
     tmp4 << "Beams:idB = " << ionPID;
     pythia.readString("HeavyIon:mode = 1");
     pythia.readString(tmp3.str());
     pythia.readString(tmp4.str());
  }
  pythia.readString("Tune:pp = 21");
  pythia.init();


  TH1D *hnjets = new TH1D("hnjets","hnjets",100,0,500);
  TH1D *hnevents = new TH1D("hnevents","hnevents",1,0,2);
  TH1D *hnch = new TH1D("hnch","hnch",40,0,200);
  TH2D *hchjet = new TH2D("hchjet","hchjet",80,0,400,80,0,400);
  hnjets->Sumw2();
  hnevents->Sumw2();
  hnch->Sumw2();

  double min_et_threshold = 0.1;


  fastjet::JetDefinition *fJetAlgorithm2 = new fastjet::JetDefinition(fastjet::antikt_algorithm,0.2,fastjet::Best);
  fastjet::JetDefinition *fJetAlgorithm4 = new fastjet::JetDefinition(fastjet::antikt_algorithm,0.4,fastjet::Best);
  fastjet::JetDefinition *fJetAlgorithm6 = new fastjet::JetDefinition(fastjet::antikt_algorithm,0.6,fastjet::Best);

  // Begin event loop. Generate event. Skip if error. List first one.
  for (int iEvent = 0; iEvent < nevents; ++iEvent) {
     hnevents->Fill(1);
    if (!pythia.next()) continue;
    if (iEvent < 1) {pythia.info.list(); pythia.event.list();} 
    weight = pythia.info.weight();
    x1 = pythia.info.x1();
    x2 = pythia.info.x2();
    vector<fastjet::PseudoJet> particles;
    vector<fastjet::PseudoJet> chparticles;
    for (int i = 0; i < pythia.event.size(); ++i){ // defining constants and stuff
       double pt = sqrt(pythia.event[i].px()*pythia.event[i].px() + pythia.event[i].py()*pythia.event[i].py());
       double p = sqrt(pt*pt + pythia.event[i].pz()*pythia.event[i].pz());
       double eta = 0.5*log((p+pythia.event[i].pz())/(p-pythia.event[i].pz()));
       double x = pythia.event[i].xProd();
       double y = pythia.event[i].yProd();
       double z = pythia.event[i].zProd();
       double px = pythia.event[i].px();
       double py = pythia.event[i].py();
       double pz = pythia.event[i].pz();
       double e = pythia.event[i].e();
       double et = pythia.event[i].e()/TMath::CosH(eta);
       double phi = TMath::ATan2(py,px); //is this the right phi definition

       int pid = pythia.event[i].id();
       int abs_pid = fabs(pid);

       bool isFinal = pythia.event[i].isFinal();
       bool isCharged = pythia.event[i].isCharged();
       //probably should take out muons at some point...
       if(!isFinal || (abs_pid > 11 && abs_pid <19))continue; //put some particle acceptor here
       if(isCharged && isFinal && abs(eta)<2.4){
	  hnch->Fill(pt);
	  fastjet::PseudoJet pseudoJet(px,py,pz,e);
	  pseudoJet.set_user_index(i);
	  chparticles.push_back(pseudoJet);
       }

       if(isCharged && isFinal && abs(eta)<4.9 && abs(eta)>2.1){
	  vmult_EPD_eta.push_back(eta);
	  vmult_EPD_pt.push_back(pt);
	  vmult_EPD_phi.push_back(phi);
       }

       fastjet::PseudoJet pseudoJet(px,py,pz,e);
       pseudoJet.set_user_index(i);
       particles.push_back(pseudoJet);
    }
    fastjet::ClusterSequence truthjetFinder(particles,*fJetAlgorithm4);
    vector<fastjet::PseudoJet> truthJets = truthjetFinder.inclusive_jets();
    unsigned int ntruthjets = truthJets.size();

    fastjet::ClusterSequence chjetFinder2(chparticles,*fJetAlgorithm2);
    vector<fastjet::PseudoJet> chjets2 = chjetFinder2.inclusive_jets();

    fastjet::ClusterSequence chjetFinder4(chparticles,*fJetAlgorithm4);
    vector<fastjet::PseudoJet> chjets4 = chjetFinder4.inclusive_jets();

    fastjet::ClusterSequence chjetFinder6(chparticles,*fJetAlgorithm6);
    vector<fastjet::PseudoJet> chjets6 = chjetFinder6.inclusive_jets();
    for(fastjet::PseudoJet fastJet: truthJets){ ///////////// do this but for the jet constituints. 
        double jet_pt = fastJet.perp();
	double jet_eta = fastJet.pseudorapidity();
	double jet_y = fastJet.rapidity();
	double jet_phi = fastJet.phi();
	double jet_e = fastJet.e();
	if(jet_pt < min_jet_pt  || fabs(jet_eta) > max_abs_y)continue;
	hnjets->Fill(jet_pt);
	if(write_ntuple){
  	   vjetpt.push_back(jet_pt);
	   vjetphi.push_back(jet_phi);
	   vjeteta.push_back(jet_eta);
	   vjety.push_back(jet_y);
	   vjete.push_back(jet_e);
	   //vjetmult.push_back(fastJet.constituents().size());
	}
	for(fastjet::PseudoJet aconstituent: fastJet.constituents()){
//	   cout << aconstituent.perp() << " " << aconstituent.m() << " " << jet_pt << endl;
	   bool accept=false;
	   if(aconstituent.perp() < 10.0)continue;
	   if(aconstituent.m() > 0.138 && aconstituent.m() < 0.141)accept=true;
	   if(aconstituent.m() > 0.493 && aconstituent.m() < 0.494)accept=true;
	   if(aconstituent.m() > 0.938 && aconstituent.m() < 0.939)accept=true;
	   if(accept){
	      hchjet->Fill(aconstituent.perp(),jet_pt);
	   }
		 
	}

    }

    for(fastjet::PseudoJet fastJet: chjets2){
        double jet_pt = fastJet.perp();
	double jet_eta = fastJet.pseudorapidity();
	double jet_y = fastJet.rapidity();
	double jet_phi = fastJet.phi();
	double jet_e = fastJet.e();
	if(jet_pt < min_jet_pt  || fabs(jet_eta) > max_abs_y)continue;
	if(write_ntuple){
  	   vjetch2pt.push_back(jet_pt);
	   vjetch2phi.push_back(jet_phi);
	   vjetch2eta.push_back(jet_eta);
	}
    }
    for(fastjet::PseudoJet fastJet: chjets4){
        double jet_pt = fastJet.perp();
	double jet_eta = fastJet.pseudorapidity();
	double jet_y = fastJet.rapidity();
	double jet_phi = fastJet.phi();
	double jet_e = fastJet.e();
	if(jet_pt < min_jet_pt  || fabs(jet_eta) > max_abs_y)continue;
	if(write_ntuple){
  	   vjetch4pt.push_back(jet_pt);
	   vjetch4phi.push_back(jet_phi);
	   vjetch4eta.push_back(jet_eta);
	}
    }
    for(fastjet::PseudoJet fastJet: chjets6){
        double jet_pt = fastJet.perp();
	double jet_eta = fastJet.pseudorapidity();
	double jet_y = fastJet.rapidity();
	double jet_phi = fastJet.phi();
	double jet_e = fastJet.e();
	if(jet_pt < min_jet_pt  || fabs(jet_eta) > max_abs_y)continue;
	if(write_ntuple){
  	   vjetch6pt.push_back(jet_pt);
	   vjetch6phi.push_back(jet_phi);
	   vjetch6eta.push_back(jet_eta);
	}
    }

    if(write_ntuple){
       jet_tree->Fill();
    }
    vjetch2pt.clear();
    vjetch4pt.clear();
    vjetch6pt.clear();
    vjetch2phi.clear();
    vjetch4phi.clear();
    vjetch6phi.clear();
    vjetch2eta.clear();
    vjetch4eta.clear();
    vjetch6eta.clear();
    vjetpt.clear();
    vjetphi.clear();
    vjeteta.clear();
    vjety.clear();
    vjete.clear();
    vmult_EPD_eta.clear();
    vmult_EPD_pt.clear();
    vmult_EPD_phi.clear();
  // End of event loop. Statistics. Histogram. Done.
  }
//  TFile *fout = new TFile("bdecay_tune5_kt175_2.root","RECREATE");
  ostringstream name;
  //name << "projection_files/com_" << com_energy << "_minpt_" << minpthat << "_evts_" << nevents << "_" << index <<  "_HI" << doHI << "_" << ionPID << ".root";
  name << "/mnt/e/data/purejets_com_" << com_energy << "_minpt_" << minpthat << "_evts_" << nevents << "_" << index <<  "_HI" << doHI << "_" << ionPID << ".root";
  TFile *fout = new TFile(name.str().c_str(),"RECREATE");

  TGraph *gcross_section = new TGraph();
  gcross_section->SetPoint(0,0,pythia.info.sigmaGen());
  gcross_section->SetName("gcross_section");
  gcross_section->SetTitle("gcross_section (mb)");
  gcross_section->Write();
  hnjets->Write();
  hnch->Write();
  hnevents->Write();
  
  if(write_ntuple)jet_tree->Write();
  hchjet->Write();
  fout->Close();
  pythia.stat();

  return 0;
}
