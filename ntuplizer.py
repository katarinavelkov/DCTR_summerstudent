import ROOT
import sys

ROOT.EnableImplicitMT()

# Load NanoAOD file
infile=sys.argv[1]
print(f"Processing {infile}")
outfile = "/eos/cms/store/group/cmst3/user/sesanche/SummerStudent/" + infile.split("/")[-1]
file_path = f"root://xrootd-cms.infn.it/{infile}"
isSignal=f"{sys.argv[2]}"


rdf = ROOT.RDataFrame("Events", file_path)

# Utility C++ functions
ROOT.gInterpreter.Declare("""
#include <ROOT/RVec.hxx>
using namespace ROOT::VecOps;

RVec<int> pad_int(int n, int val) {
    return RVec<int>(n, val);
}

RVec<int> concat_int(const RVec<int>& v1, const RVec<int>& v2, const RVec<int>& v3, const RVec<int>& v4) {
    RVec<int> out = v1;
    out.insert(out.end(), v2.begin(), v2.end());
    out.insert(out.end(), v3.begin(), v3.end());
    out.insert(out.end(), v4.begin(), v4.end());
    return out;
}

RVec<float> concat_float(const RVec<float>& v1, const RVec<float>& v2, const RVec<float>& v3, const RVec<float>& v4) {
    RVec<float> out = v1;
    out.insert(out.end(), v2.begin(), v2.end());
    out.insert(out.end(), v3.begin(), v3.end());
    out.insert(out.end(), v4.begin(), v4.end());
    return out;
}
""")



# Neutrinos
rdf = rdf.Define("Neutrino_mask", "abs(GenPart_pdgId) == 12 || abs(GenPart_pdgId) == 14 || abs(GenPart_pdgId) == 16") \
         .Define("Neutrino_pt",   "GenPart_pt[Neutrino_mask]") \
         .Define("Neutrino_eta",  "GenPart_eta[Neutrino_mask]") \
         .Define("Neutrino_phi",  "GenPart_phi[Neutrino_mask]") \
         .Define("Neutrino_mass", "GenPart_mass[Neutrino_mask]") \
         .Define("Neutrino_charge", "(GenPart_pdgId[Neutrino_mask] > 0)*2-1")
rdf = rdf.Define("Neutrino_charge_int", "ROOT::VecOps::RVec<int>(Neutrino_charge.begin(), Neutrino_charge.end())")
rdf = rdf.Define("isSignal", f"{isSignal}")

# Ensure all charges are ints
rdf = rdf.Define("GenDressedLepton_charge", "(GenDressedLepton_pdgId > 0)*2-1")\
         .Define("GenDressedLepton_charge_int", "ROOT::VecOps::RVec<int>(GenDressedLepton_charge.begin(), GenDressedLepton_charge.end())")\
         .Define("GenVisTau_charge_int", "ROOT::VecOps::RVec<int>(GenVisTau_charge.begin(), GenVisTau_charge.end())")

# Predefine one-hot flags
rdf = rdf.Define("GenJet_flag",     "pad_int(nGenJet, 1)") \
         .Define("GenLepton_flag",  "pad_int(nGenDressedLepton, 1)") \
         .Define("GenVisTau_flag",     "pad_int(nGenVisTau, 1)") \
         .Define("GenNu_flag",      "pad_int(Neutrino_pt.size(), 1)") \
         .Define("ZeroJet_flag",    "pad_int(nGenJet, 0)") \
         .Define("ZeroLepton_flag", "pad_int(nGenDressedLepton, 0)") \
         .Define("ZeroTau_flag",    "pad_int(nGenVisTau, 0)") \
         .Define("ZeroNu_flag",     "pad_int(Neutrino_pt.size(), 0)")

# Merge GenObjects
rdf = rdf.Define("GenObject_pt",    "concat_float(GenJet_pt, GenDressedLepton_pt, GenVisTau_pt, Neutrino_pt)") \
         .Define("GenObject_eta",   "concat_float(GenJet_eta, GenDressedLepton_eta, GenVisTau_eta, Neutrino_eta)") \
         .Define("GenObject_phi",   "concat_float(GenJet_phi, GenDressedLepton_phi, GenVisTau_phi, Neutrino_phi)") \
         .Define("GenObject_mass",  "concat_float(GenJet_mass, GenDressedLepton_mass, GenVisTau_mass, Neutrino_mass)") \
         .Define("GenObject_charge","concat_int(pad_int(nGenJet, 0), GenDressedLepton_charge_int, GenVisTau_charge_int, Neutrino_charge_int)") \
         .Define("GenObject_isJet",     "concat_int(GenJet_flag, ZeroLepton_flag, ZeroTau_flag, ZeroNu_flag)") \
         .Define("GenObject_isLepton",  "concat_int(ZeroJet_flag, GenLepton_flag, ZeroTau_flag, ZeroNu_flag)") \
         .Define("GenObject_isTau",     "concat_int(ZeroJet_flag, ZeroLepton_flag, GenVisTau_flag, ZeroNu_flag)") \
         .Define("GenObject_isNeutrino","concat_int(ZeroJet_flag, ZeroLepton_flag, ZeroTau_flag, GenNu_flag)")

rdf = rdf.Define("nGenObject", "GenObject_pt.size()")
# Save snapshot
columns = [
    "nGenObject",
    "GenObject_pt", "GenObject_eta", "GenObject_phi", "GenObject_mass", "GenObject_charge",
    "GenObject_isJet", "GenObject_isLepton", "GenObject_isTau", "GenObject_isNeutrino",
    "isSignal", 'genWeight'
]
rdf.Snapshot("GenObjectTree", f"{outfile}.root", columns)
