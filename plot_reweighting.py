import ROOT

# Open your ROOT file
f = ROOT.TFile.Open("nn_output.root")
tree = f.Get("Events")

# Expressions and cuts
draws = [
    ("score_label_sample1", "label_sample1 == 1", ROOT.kRed,  "label=1"),
    ("score_label_sample1", "label_sample1 == 0", ROOT.kBlue, "label=0"),
    ("score_label_sample1", "(label_sample1 == 0)*(nn_output/(1-nn_output))", ROOT.kGreen+2, "label ==0, reweighted to label ==1"),
]

# Canvas
c = ROOT.TCanvas("c", "Histograms", 800, 600)

hists = []
for i, (expr, cut, color, label) in enumerate(draws):
    hname = f"h{i}"
    drawopt = "hist norm" + (",same" if i > 0 else "")
    tree.Draw(f"{expr}>>{hname}(50,0.4,0.6)", cut, drawopt)
    h = ROOT.gPad.GetPrimitive(hname)
    h.SetLineColor(color)
    h.SetLineWidth(2)
    hists.append((h, label))

# Legend
leg = ROOT.TLegend(0.6, 0.7, 0.88, 0.88)
for h, label in hists:
    leg.AddEntry(h, label, "l")
leg.Draw()

c.Update()
c.SaveAs("histograms.png")