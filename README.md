<div align="center">
<h1><img src='/sundry/1f52c.gif' width="35px"> DiagUS <img src='/sundry/1f4a1.gif' width="35px"></h1>
<h3>Non-invasive Differentiation of Uterine Sarcomas from Leiomyomas Using Multiparametric MRI Radiomics: A Multicenter Validation Study</h3>



<!--
### [Project Page]() | [Paper link]()
-->

</div>

## <img src='/sundry/1f43c.gif' width="30px"> News

* **``:** We released our inference code. Paper/Project pages are coming soon. Please stay tuned! <img src='/sundry/1f379.gif' width="25px">

## <img src='/sundry/1f4f0.gif' width="30px"> Abstract
Uterine sarcomas are uncommon yet highly aggressive tumours that can closely mimic benign fibroids on MRI, complicating preoperative distinction. Misdiagnosis may lead to inappropriate surgical management and a 20–25% reduction in 5-year overall survival. Radiomics enables extraction of quantitative imaging features that are imperceptible to the human eye. We evaluated a radiomics-based machine learning model—Sarcoma vs Uterine Fibroids Radiomics AI (SUFRAI)—which integrates intratumoural and peritumoural features from multiparametric MRI in a multicentre cohort.
Methods
In this retrospective study, 520 patients (193 sarcomas and 327 fibroids) from six tertiary hospitals were included. Radiomic features were extracted from T2-weighted and diffusion-weighted imaging, with a 5-mm peritumoral margin applied. Feature selection used Pearson correlation, maximum Relevance Minimum Redundancy(mRMR), and Least Absolute Shrinkage and Selection Operator (LASSO). Three models (intratumoral, peritumoral, and combined) were trained with a support vector machine classifier and assessed by Receiver Operating Characteristic curve (ROC) analysis in training, internal testing, and four external validation sets. Interobserver reproducibility of segmentation was evaluated with intraclass correlation coefficients.
Findings
The combined model achieved the highest performance, with AUCs of 0.89 in training, 0·87 in testing, and 0·75–0·79 across external cohorts. Dual-sequence features from T2-weighted and diffusion-weighted images captured complementary anatomical and microstructural information. SHAP analysis highlighted two DWI-derived features as the most predictive. The integrated model showed superior generalizability compared with single-region or single-sequence models.
Interpretation
In the largest multicenter radiomics cohort for uterine sarcoma diagnosis to date, the SUFRAI model significantly improved non-invasive preoperative differentiation of sarcomas from fibroids. Combining intratumoral and peritumoral features with multiparametric MRI enhances accuracy, robustness, and interpretability, supporting its potential clinical utility.


## <img src='/sundry/1f9f3.gif' width="30px"> Environment Setups

* python 3.7
* See 'requirements.txt' for Python libraries required

```shell
conda create -n DiagUS python=3.7
conda activate DiagUS
# cd /xx/xx/DiagUS
pip install -r requirements.txt
```


## <img src='/sundry/1f5c2-fe0f.gif' width="30px"> Model Checkpoints
model weight in the /DiagUS/model.

## <img src='/sundry/听诊器.gif' width="30px"> Our Dataset 
We have provided sample data in the data directory.

## <img src='/sundry/1f3ac.gif' width="30px"> Inference Demo
You can visualize our inference results and evaluation metrics using the following commands.:

*  inference
```shell
python test.py 

```




<!--
## Acknowledgement


## Citation
-->
