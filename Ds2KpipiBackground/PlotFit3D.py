import rootpy.ROOT as ROOT
from rootpy.plotting import Hist1D, Hist2D
from root_numpy import root2array

ROOT.gROOT.ProcessLine(".x ../lhcbstyle.C")
ROOT.gStyle.SetPadRightMargin(0.05)
ROOT.gStyle.SetPadLeftMargin(0.21)
ROOT.gStyle.SetTitleOffset(1.4, "Y")
ROOT.gROOT.GetColor(3).SetRGB(0., 0.6, 0.)

c = ROOT.TCanvas("c", "", 600, 500)

toy_sig = root2array("test_tuple.root", branches = ["mprime", "thetaprime"], selection = "abs(md-1.97)<0.05")
toy_sb  = root2array("test_tuple.root", branches = ["mprime", "thetaprime"], selection = "(md<1.97-0.05)||(md>1.97+0.05)")
fit_sig = root2array("fit_result_3d.root", branches = ["mprime", "thetaprime"], selection = "abs(md-1.97)<0.05")
#fit_sig = root2array("fit_result.root", branches = ["mprime", "thetaprime"], selection = "md>1.97+0.05")

hsig = Hist1D(100, 0., 1.)
hsb  = Hist1D(100, 0., 1.)
hfit = Hist1D(100, 0., 1.)

hsig.fill_array(toy_sig['mprime'])
hsb.fill_array(toy_sb['mprime'])
hfit.fill_array(fit_sig['mprime'])

hsig.SetMarkerSize(0.5)
hsig.Draw("e")
hsig.GetXaxis().SetTitle("m'")
hsig.GetYaxis().SetTitle("Entries / (0.01)")
hsb.Scale(hsig.GetSumOfWeights()/hsb.GetSumOfWeights())
hsb.SetLineColor(6)
hsb.SetLineWidth(3)
hsb.SetLineStyle(7)
hsb.Draw("hist same")
hfit.Scale(hsig.GetSumOfWeights()/hfit.GetSumOfWeights())
hfit.SetLineColor(3)
hfit.SetLineWidth(2)
hfit.SetLineStyle(1)
hfit.Draw("hist same")
hsig.Draw("e same")

legend = ROOT.TLegend(0.33,0.23,0.65,0.40)
legend.AddEntry(hsig, "Signal region", "pe")
legend.AddEntry(hsb,  "Sidebands", "l")
legend.AddEntry(hfit, "Fit result", "l")
legend.Draw()

c.Print("fit_result_3d_m.pdf")

tsig = Hist1D(100, 0., 1.)
tsb = Hist1D(100, 0., 1.)
tfit = Hist1D(100, 0., 1.)

tsig.fill_array(toy_sig['thetaprime'])
tsb.fill_array(toy_sb['thetaprime'])
tfit.fill_array(fit_sig['thetaprime'])

tsig.SetMarkerSize(0.5)
tsig.Draw("e")
tsig.GetXaxis().SetTitle("#theta'")
tsig.GetYaxis().SetTitle("Entries / (0.01)")
tsb.Scale(tsig.GetSumOfWeights()/tsb.GetSumOfWeights())
tsb.SetLineColor(6)
tsb.SetLineWidth(3)
tsb.SetLineStyle(7)
tsb.Draw("hist same")
tfit.Scale(tsig.GetSumOfWeights()/tfit.GetSumOfWeights())
tfit.SetLineColor(3)
tfit.SetLineWidth(2)
tfit.SetLineStyle(1)
tfit.Draw("hist same")
tsig.Draw("e same")

legend = ROOT.TLegend(0.33,0.23,0.65,0.40)
legend.AddEntry(tsig, "Signal region", "pe")
legend.AddEntry(tsb,  "Sidebands", "l")
legend.AddEntry(tfit, "Fit result", "l")
legend.Draw()

c.Print("fit_result_3d_theta.pdf")
