//----------https://github.com/smaruyam/usercode/blob/master/TriggerDataAna/TriggerDataAna/test/drawEff.cc
void CMS_lumi(TPad* pad, int iPeriod, int iPosX, string lumi) {
	TString cmsText = "CMS";
	float cmsTextFont = 60; // default is helvetic-bold
	bool writeExtraText = true;
//	TString extraText = "   Preliminary";
	TString extraText = "";
	float extraTextFont = 52; // default is helvetica-italics
// text sizes and text offsets with respect to the top frame
// in unit of the top margin size
	float lumiTextSize = 0.6;
	float lumiTextOffset = 0.2;
	float cmsTextSize = 0.75;
	float cmsTextOffset = 0.1; // only used in outOfFrame version
	float relPosX = 0.045;
	float relPosY = 0.035;
	float relExtraDY = 1.2;
// ratio of "CMS" and extra text size
	float extraOverCmsTextSize = 0.76;
	TString lumi_13TeV = lumi + " fb^{-1}";
//TString lumi_8TeV = "19.7 fb^{-1}";
	TString lumi_8TeV = "18.5 fb^{-1}";
	TString lumi_7TeV = "5.1 fb^{-1}";
// set lumi style
	int H_ref = 600;
	int W_ref = 800;
// references for T, B, L, R
// float
	float T = 0.08 * H_ref;
	float B = 0.12 * H_ref;
	float L = 0.12 * W_ref;
	float R = 0.04 * W_ref;
//
	bool outOfFrame = false;
	if (iPosX / 10 == 0) {
		outOfFrame = true;
	}
	int alignY_ = 3;
	int alignX_ = 2;
	if (iPosX / 10 == 0)
		alignX_ = 1;
	if (iPosX == 0)
		alignX_ = 1;
	if (iPosX == 0)
		alignY_ = 1;
	if (iPosX / 10 == 1)
		alignX_ = 1;
	if (iPosX / 10 == 2)
		alignX_ = 2;
	if (iPosX / 10 == 3)
		alignX_ = 3;
	if (iPosX == 0)
		relPosX = 0.12;
	int align_ = 10 * alignX_ + alignY_;
	float l = pad->GetLeftMargin();
	float t = pad->GetTopMargin();
	float r = pad->GetRightMargin();
	float b = pad->GetBottomMargin();
	pad->cd();
	TString lumiText;
	if (iPeriod == 1) {
		lumiText += lumi_7TeV;
		lumiText += " (7 TeV)";
	} else if (iPeriod == 2) {
		lumiText += lumi_8TeV;
		lumiText += " (8 TeV)";
	} else if (iPeriod == 3) {
		lumiText = lumi_8TeV;
		lumiText += " (8 TeV)";
		lumiText += " + ";
		lumiText += lumi_7TeV;
		lumiText += " (7 TeV)";
	} else if (iPeriod == 4) {
		lumiText += lumi_13TeV;
		lumiText += " (13 TeV)";
	} else if (iPeriod == 7) {
		if (outOfFrame)
			lumiText += "#scale[0.85]{";
		lumiText += lumi_13TeV;
		lumiText += " (13 TeV)";
		lumiText += " + ";
		lumiText += lumi_8TeV;
		lumiText += " (8 TeV)";
		lumiText += " + ";
		lumiText += lumi_7TeV;
		lumiText += " (7 TeV)";
		if (outOfFrame)
			lumiText += "}";
	} else if (iPeriod == 12) {
		lumiText += "8 TeV";
	}
	std::cout << lumiText << endl;
	TLatex latex;
	latex.SetNDC();
	latex.SetTextAngle(0);
	latex.SetTextColor(kBlack);
	float extraTextSize = extraOverCmsTextSize * cmsTextSize;
	latex.SetTextFont(42);
	latex.SetTextAlign(31);
	latex.SetTextSize(lumiTextSize * t);
	latex.DrawLatex(1 - r, 1 - t + lumiTextOffset * t, lumiText);
	if (outOfFrame) {
		latex.SetTextFont(cmsTextFont);
		latex.SetTextAlign(11);
		latex.SetTextSize(cmsTextSize * t);
		latex.DrawLatex(l, 1 - t + lumiTextOffset * t, cmsText);
	}
	pad->cd();
	float posX_ = 0;
	if (iPosX % 10 <= 1) {
		posX_ = l + relPosX * (1 - l - r);
	} else if (iPosX % 10 == 2) {
		posX_ = l + 0.5 * (1 - l - r);
	} else if (iPosX % 10 == 3) {
		posX_ = 1 - r - relPosX * (1 - l - r);
	}
	float posY_ = 1 - t - relPosY * (1 - t - b);
	if (!outOfFrame) {
		latex.SetTextFont(cmsTextFont);
		latex.SetTextSize(cmsTextSize * t);
		latex.SetTextAlign(align_);
		latex.DrawLatex(posX_, posY_, cmsText);
		if (writeExtraText) {
			latex.SetTextFont(extraTextFont);
			latex.SetTextAlign(align_);
			latex.SetTextSize(extraTextSize * t);
			latex.DrawLatex(posX_, posY_ - relExtraDY * cmsTextSize * t,
					extraText);
		}
	} else if (writeExtraText) {
		if (iPosX == 0) {
			posX_ = l + relPosX * (1 - l - r);
			posY_ = 1 - t + lumiTextOffset * t;
		}
		latex.SetTextFont(extraTextFont);
		latex.SetTextSize(extraTextSize * t);
		latex.SetTextAlign(align_);
		latex.DrawLatex(posX_, posY_, extraText);
	}
	return;
}
