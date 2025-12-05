# delimit ;
set more off;
cd "/home/f1yxs02/nas03/f1yxs02/Data/LPS";
use "/home/f1yxs02/nas03/f1yxs02/Data/LPS/loan_level_data_oneobs.dta", clear; 
# delimit;
gen y=close_year;
drop if fips==.; 
drop if fips==42101 | fips==42003;
cd "/home/f1yxs02/nas03/f1yxs02/past_experience_project";
# delimit;
merge m:1 zipcode using listofbothzipcode;
gen dual_exploration=0;
replace dual_exploration=1 if _merge==3;
keep if _merge==1;
drop if orig_rate_miss==1 | frm_miss==1;
gen nodrilling=0;
replace nodrilling=1 if fracking==0 & nonfracking==0;
gen FICO580=0;
gen FICO620=0;
gen FICO660=0;
gen FICO700=0;
replace FICO580=1 if fico_orig_win<580;
replace FICO620=1 if fico_orig_win >=580 & fico_orig_win< 620;
replace FICO660=1 if fico_orig_win >=620 & fico_orig_win<660;
replace FICO700=1 if fico_orig_win >=660 & fico_orig_win<700;
drop if document_miss==1;
*replace low_doc=0 if low_doc ==.;
# delimit;
replace prepay_penalty=0 if prepay_penalty==.;
replace refi=0 if refi==.;
replace dti_ratio=0 if dti_ratio==.;

keep d_90_orig_24 d_90_orig_36 fracking nonfracking nodrilling fico_orig_win FICO* dti_ratio_win ltv_ratio_win orig_rate_win orig_amt_win ltv_ratio_80 low_doc frm balloon io jumbo optarm pmi prepay_penalty refi term_30 y;

local display_controls fico_orig_win dti_ratio_win ltv_ratio_win orig_rate_win orig_amt_win y d_90_orig_24 d_90_orig_36 ltv_ratio_80 FICO580 FICO620 FICO660 FICO700 low_doc frm balloon io jumbo optarm pmi prepay_penalty refi term_30 fracking nonfracking nodrilling fracking_GSE GSE nonfracking_GSE ;

rename d_90_orig_24 Default24;
rename d_90_orig_36 Default36;
rename fico_orig_win FICO;
rename dti_ratio_win DTI;
rename ltv_ratio_win LTV;
rename orig_rate_win RATE;
rename orig_amt_win LOAN_AMOUNT;
rename ltv_ratio_80 LTV80;
rename low_doc LOW_DOC;
rename frm FRM;
rename balloon BALLOON;
rename io IO;
rename jumbo JUMBO;
rename optarm OPT_ARM;
rename pmi PMI;
rename prepay_penalty PENALTY;
rename refi REFI;
rename term_30 TERM30;
rename fracking FRACK_ZIP;
rename nonfracking CONV_ZIP;
rename nodrilling NO_DRILLING;
*rename fracking_GSE FRACKxGSE;
*rename nonfracking_GSE CONVxGSE;
rename y ORIG_YEAR;
gen LOG_AMOUNT =log(LOAN_AMOUNT);
drop if LOG_AMOUNT==.;

# delimit;
label var Default24 `"Indicator variable=1 if loan default within 24 months of origination (Default24)"';
label var Default36 `"Indicator variable=1 if loan default within 36 months of origination (Default36)"';
label var FICO `"Borrower FICO score at origination (FICO)"';
label var DTI `"Borrower Debt to Income Ratio at origination (DTI)"';
label var LTV `"Loan to value ratio at origination (LTV)"';
label var RATE `"Rate of borrowing at origination (RATE)"';
label var LOAN_AMOUNT `"Amount of loan at origination (LOAN_AMOUNT)"';
label var LOG_AMOUNT `"Log loan amount at origination (LOG_AMOUNT)"';
label var LTV80 `"Indicator variable=1 if LTV=80 (LTV80)"';
*label var DOC_MISS `"Indicator variable=1 if no document type is missing (DOC_MISS)"';
label var LOW_DOC `"Indicator variable=1 if the loan is is low-document (LOW_DOC)"';
label var FRM `"Indicator variable for fixed rate mortgage (FRM)"';
label var BALLOON `"Indicator variable=1 if loan has a baloon payment (BALLOON)"';
label var IO `"Indicator variable for interest only loan (IO)"';
label var JUMBO `"Indicator variable for Jumbo loan (JUMBO)"';
label var OPT_ARM `"Indicator variable for optional adjusted rate mortgage loan (OPT_ARM)"';
label var PMI `"Indicator variable=1 if loan has binding mortgage insurance (PMI)"';
label var PENALTY `"Indicator variable=1 if loan has prepayment-penalty (PENALT)"';
label var REFI `"Indicator variable=1 if loan purpose is refinance (REFI)"';
label var TERM30 `"Indicator variable=1 if loan term=30 years (TERM30)"';
*label var GSE `"Indicator variable=1 if loan is secured by GSE (GSE)"';
label var FRACK_ZIP `"Indicator variable=1 if property is located in fracking zipcode (FRACK)"';
label var CONV_ZIP `"Indicator variable=1 if property is located in conventional drilling zipcode (CONV)"';
label var NO_DRILLING `"Indicator variable=1 if property is located in no-drilling zipcode (NO_DRILLING)"';
*label var FRACKxGSE `"Interactions term of fracking and GSE (FRACKxGSE)"';
*label var CONVxGSE `"Interactions term of Conventional and GSE (CONVxGSE)"';
label var ORIG_YEAR  `"Year of origination (ORIG_YEAR)"';
label var FICO580 `"Indicator Variable if borrower FICO is below 580 (FICO580)"';
label var FICO620 `"Indicator Variable if borrower FICO is between 580 and 620 (FICO620)"';
label var FICO660 `"Indicator Variable if borrower FICO is between 620 and 660 (FICO660)"';
label var FICO700 `"Indicator Variable if borrower FICO is between 660 and 700 (FICO700)"';

# delimit;
preserve;


cd "/home/f1yxs02/nas03/f1yxs02/oct1draft";
# delimit ;
local display_controls FICO DTI LTV RATE LOAN_AMOUNT ORIG_YEAR Default24 Default36 LTV80 FICO580 FICO620 FICO660 FICO700 LOW_DOC FRM BALLOON IO JUMBO OPT_ARM PMI PENALTY REFI TERM30 FRACK_ZIP CONV_ZIP NO_DRILLING;

*local display_controls fico_orig_win dti_ratio_win ltv_ratio_win orig_rate_win orig_amt_win y d_90_orig_24 d_90_orig_36 ltv_ratio_80 FICO580 FICO620 FICO660 FICO700 low_doc frm balloon io jumbo optarm pmi prepay_penalty refi term_30 fracking nonfracking nodrilling fracking_GSE GSE nonfracking_GSE ;

outreg2 using summarystat0411.xls if ORIG_YEAR<=2011, sum(log) replace label eqkeep(N mean sd ) sortvar(`display_controls') addnote(Descriptive Statistics for the 1,114,173 loans originated in PA from 2004 to 2012);
bysort FRACK_ZIP: outreg2 using summarystatfrack0411.xls if ORIG_YEAR<=2011, sum(log) replace  label eqkeep(N mean sd) sortvar(`display_controls') addnote(Descriptive statistics for the loans. Column 1 includes loans for properties located in non-Fracking zipcodes. Column 1 includes loans for properties located in Fracking zipcodes.);
bysort CONV_ZIP: outreg2 using summarystatnonfrack0411.xls if ORIG_YEAR<=2011, sum(log) replace  label eqkeep(N mean sd) sortvar(`display_controls') addnote(Descriptive statistics for the loans. Column 1 includes loans for properties located in non-conventional drilling zipcodes. Column 1 includes loans for properties located in conventional drilling zipcodes.);
*bysort NO_DRILLING: outreg2 using summarystatdrilling0411.xls if ORIG_YEAR<=2011, sum(log) replace  label eqkeep(N mean sd) sortvar(`display_controls') addnote(Descriptive statistics for the loans. Column 1 includes loans for properties located in non-conventional drilling zipcodes. Column 1 includes loans for properties located in conventional drilling zipcodes.);


/*****0406
# delimit ;
set more off;
outreg2 using summarystat0406.xls if ORIG_YEAR<=2006, sum(log) replace label eqkeep( N mean sd) sortvar(`display_controls') addnote(Dummy variables not shown);
bysort FRACK_ZIP: outreg2 using summarystatfrack0406.xls if ORIG_YEAR<=2006, sum(log) replace sortvar(`display_controls') label eqkeep(N mean sd)  addnote(Descriptive statistics for the loans. Column 1 includes loans for properties located in non-Fracking zipcodes. Column 1 includes loans for properties located in Fracking zipcodes.);
bysort CONV_ZIP: outreg2 using summarystatnonfrack0406.xls if ORIG_YEAR<=2006, sum(log) replace  sortvar(`display_controls') label eqkeep(N mean sd)  addnote(Descriptive statistics for the loans. Column 1 includes loans for properties located in non-conventional drilling zipcodes. Column 1 includes loans for properties located in conventional drilling zipcodes.);
bysort NO_DRILLING: outreg2 using summarystatdrilling0406.xls if ORIG_YEAR<=2006, sum(log) replace  label eqkeep(N mean sd) sortvar(`display_controls') addnote(Descriptive statistics for the loans. Column 1 includes loans for properties located in non-conventional drilling zipcodes. Column 1 includes loans for properties located in conventional drilling zipcodes.);
*/

***0407
# delimit ;
set more off;
local display_controls FICO DTI LTV RATE LOAN_AMOUNT ORIG_YEAR Default24 Default36 LTV80 FICO580 FICO620 FICO660 FICO700 LOW_DOC FRM BALLOON IO JUMBO OPT_ARM PMI PENALTY REFI TERM30 FRACK_ZIP CONV_ZIP NO_DRILLING;

outreg2 using summarystat0407.xls if ORIG_YEAR<=2007, sum(log) replace label eqkeep( N mean sd ) sortvar(`display_controls')  addnote(Dummy variables not shown);
bysort FRACK_ZIP: outreg2 using summarystatfrack0407.xls if ORIG_YEAR<=2007, sum(log) replace  sortvar(`display_controls')  label eqkeep(N mean sd) addnote(Descriptive statistics for the loans. Column 1 includes loans for properties located in non-Fracking zipcodes. Column 1 includes loans for properties located in Fracking zipcodes.);
bysort CONV_ZIP: outreg2 using summarystatnonfrack0407.xls if ORIG_YEAR<=2007, sum(log) replace sortvar(`display_controls')  label eqkeep(N mean sd)  addnote(Descriptive statistics for the loans. Column 1 includes loans for properties located in non-conventional drilling zipcodes. Column 1 includes loans for properties located in conventional drilling zipcodes.);
*bysort NO_DRILLING: outreg2 using summarystatdrilling0407.xls if ORIG_YEAR<=2007, sum(log) replace sortvar(`display_controls')  label eqkeep(N mean sd) addnote(Descriptive statistics for the loans. Column 1 includes loans for properties located in non-conventional drilling zipcodes. Column 1 includes loans for properties located in conventional drilling zipcodes.);


*****0812 774,702 
# delimit ;
set more off;
outreg2 using summarystat0812.xls if ORIG_YEAR<=2011 & ORIG_YEAR>=2008, sum(log) replace  label eqkeep(N mean sd ) sortvar(display_controls) addnote(Descriptive Statistics for the 774,702 loans originated in PA from 2008 to 2012);
bysort FRACK_ZIP: outreg2 using summarystatfrack0811.xls if ORIG_YEAR<=2011 & ORIG_YEAR>=2008, sum(log) replace sortvar(display_controls)  label eqkeep(N mean sd)  addnote(Descriptive statistics for the loans. Column 1 includes loans for properties located in non-Fracking zipcodes. Column 1 includes loans for properties located in Fracking zipcodes.);
bysort CONV_ZIP: outreg2 using summarystatnonfrack0811.xls if ORIG_YEAR<=2011 & ORIG_YEAR>=2008, sum(log) replace sortvar(display_controls) label eqkeep(N mean sd) addnote(Descriptive statistics for the loans. Column 1 includes loans for properties located in non-conventional drilling zipcodes. Column 1 includes loans for properties located in conventional drilling zipcodes.);
*bysort NO_DRILLING: outreg2 using summarystatdrilling0811.xls if ORIG_YEAR<=2011 & ORIG_YEAR>=2008, sum(log) replace sortvar(display_controls) label eqkeep(N mean sd) addnote(Descriptive statistics for the loans. Column 1 includes loans for properties located in non-conventional drilling zipcodes. Column 1 includes loans for properties located in conventional drilling zipcodes.);

*restore;
*keep d_90_orig_24 d_90_orig_36 ltv_ratio_80 FICO580 FICO620 FICO660 FICO700 low_doc frm balloon io jumbo optarm pmi prepay_penalty refi term_30 y fracking nonfracking nodrilling fracking_GSE GSE nonfracking_GSE;
