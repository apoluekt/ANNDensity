import rootpy.ROOT as ROOT
from rootpy.plotting import Hist1D, Hist2D
from root_numpy import root2array

ROOT.gROOT.ProcessLine(".x ../lhcbstyle.C")
ROOT.gStyle.SetPadRightMargin(0.05)
ROOT.gStyle.SetPadLeftMargin(0.21)
ROOT.gStyle.SetTitleOffset(1.4, "Y")
ROOT.gROOT.GetColor(3).SetRGB(0., 0.6, 0.)

c = ROOT.TCanvas("c", "", 500, 450)

toy_sig = root2array("test_tuple.root", branches = ["mprime", "thetaprime"])
fit_sig = root2array("eff_fit_result.root", branches = ["mprime", "thetaprime"])

hsig = Hist1D(100, 0., 1.)
hfit = Hist1D(100, 0., 1.)

hsig.fill_array(toy_sig['mprime'])
hfit.fill_array(fit_sig['mprime'])

hsig2d = Hist2D(50, 0., 1., 50, 0., 1.)
hfit2d = Hist2D(50, 0., 1., 50, 0., 1.)
hsig2d.fill_array(toy_sig.view((float, len(toy_sig.dtype.names))))
hfit2d.fill_array(fit_sig.view((float, len(fit_sig.dtype.names))))
hfit2d.Scale(hsig2d.GetSumOfWeights()/hfit2d.GetSumOfWeights())
sqdiff = (hfit2d-hsig2d)**2/hfit2d
print(sqdiff.GetSumOfWeights())

hsig.SetMinimum(0.)
hsig.SetMarkerSize(0.5)
hsig.Draw("e")
hsig.GetXaxis().SetTitle("m'")
hsig.GetYaxis().SetTitle("Entries / (0.01)")
hfit.Scale(hsig.GetSumOfWeights()/hfit.GetSumOfWeights())
hfit.SetLineColor(3)
hfit.SetLineWidth(2)
hfit.SetLineStyle(1)
hfit.Draw("hist ][ same")
hsig.Draw("e same")

legend = ROOT.TLegend(0.33,0.23,0.65,0.40)
legend.AddEntry(hsig, "Simulation", "pe")
legend.AddEntry(hfit, "Fit result", "l")
legend.Draw()

c.Print("eff_fit_result_m.pdf")

tsig = Hist1D(100, 0., 1.)
tfit = Hist1D(100, 0., 1.)

tsig.fill_array(toy_sig['thetaprime'])
tfit.fill_array(fit_sig['thetaprime'])

tsig.SetMinimum(0.)
tsig.SetMarkerSize(0.5)
tsig.Draw("e")
tsig.GetXaxis().SetTitle("#theta'")
tsig.GetYaxis().SetTitle("Entries / (0.01)")
tfit.Scale(tsig.GetSumOfWeights()/tfit.GetSumOfWeights())
tfit.SetLineColor(3)
tfit.SetLineWidth(2)
tfit.SetLineStyle(1)
tfit.Draw("hist ][ same")
tsig.Draw("e same")

legend = ROOT.TLegend(0.33,0.23,0.65,0.40)
legend.AddEntry(tsig, "Simulation", "pe")
legend.AddEntry(tfit, "Fit result", "l")
legend.Draw()

c.Print("eff_fit_result_theta.pdf")
