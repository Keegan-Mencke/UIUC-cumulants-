#include "Pythia8/Pythia.h"
#include "TFile.h"
#include "TTree.h"

using namespace Pythia8;
using namespace std;

int main() {

  // Initialize Pythia
  Pythia pythia;
  pythia.readString("Print:errors = 0");
  // Beam settings: Pb-Pb at 5.02 TeV
  pythia.readString("Beams:idA = 1000822080");  // Lead
  pythia.readString("Beams:idB = 1000822080");  // Lead
  pythia.readString("Beams:eCM = 5020.");       // center-of-mass energy

  // Turn on QCD (soft + hard)
  pythia.readString("HeavyIon:mode = 1"); //Angantyr

  // Initialize generator
  pythia.init();

  // Create ROOT file
  TFile *outfile = new TFile("/mnt/e/data/50k_pbpb.root", "RECREATE");

  // Variables to store
 
  int eventNumber;
  vector<float> pt;
  vector<float> eta;
  vector<float> phi;
  double weight;

  // Create TTree
  TTree *tree = new TTree("tree", "Particle kinematics");

  tree->Branch("event", &eventNumber, "event/I");
  tree->Branch("pt", &pt );
  tree->Branch("eta", &eta);
  tree->Branch("phi", &phi);
  tree->Branch("weight", &weight);

  // Number of events
  int nEvents = 500;

  // Event loop
  for (int iEvent = 0; iEvent < nEvents; ++iEvent) {

    if (!pythia.next()) continue;

    eventNumber = iEvent;
    pt.clear();
    phi.clear();
    eta.clear();

    // Loop over particles in event
    for (int i = 0; i < pythia.event.size(); ++i) {

      // only final state particles
      if (!pythia.event[i].isFinal()) continue;
      if (!pythia.event[i].isCharged()) continue;
      if (abs(pythia.event[i].eta())> 3.0) continue;

      pt.push_back(pythia.event[i].pT());
      eta.push_back(pythia.event[i].eta());
      phi.push_back(pythia.event[i].phi());
      weight = pythia.info.weight();
      
    }
   tree->Fill();
   if (iEvent % 100 ==0 && iEvent >0) {
                cout<< "Processed" << iEvent << "events" << endl;
                tree->AutoSave("SaveSelf");
                                                              }
  }

  // Write output
  tree->Write("", TObject::kOverwrite);
  outfile->Close();

  // Print statistics
  pythia.stat();

  return 0;
}

