negative_sample_generation_prompt="""
## CPIC QA Dataset Generator
You are a pharmacogenomics expert creating structured QA datasets that CAN'T BE FOUND IN CPIC guidelines. Follow this protocol:

You be given a csv file specifying the EXISTING CPIC guideline for certain pair of drugs and genes,
You should select any pair of drug/gene that DON'T have CPIC guideline.
This is for negative Question & Answer pair generation.

#### Question Generation Requirements
- **Exclusively generate drug-gene interaction questions**
- **Question Structure**:
  Design questions to extract specific clinical decision-making information. Below are the required question types with purposes, content targets, and examples:

  1. **Specific Dosing Guidance**  
     - *Purpose*: Extract precise drug dosage values and adjustment protocols  
     - *Content to Search*: Dosage numbers, adjustment protocols, clinical recommendations  
     - *Example*: "What is the recommended efavirenz dose for a CYP2B6 poor metabolizer?"  

  2. **Clinical Risk Stratification**  
     - *Purpose*: Obtain quantified adverse reaction risk data  
     - *Content to Search*: Risk multipliers, incidence rates, toxicity thresholds  
     - *Example*: "What is the increased risk of CNS adverse effects for a CYP2B6 poor metabolizer?"  

  3. **Pharmacokinetics/Pharmacodynamics Data**  
     - *Purpose*: Gather therapeutic drug monitoring values  
     - *Content to Search*: Plasma concentration ranges, therapeutic windows  
     - *Example*: "What is the suggested therapeutic range for plasma efavirenz concentrations?"  

  4. **Genotype-Phenotype Correspondence**  
     - *Purpose*: Clarify genotype to metabolic phenotype mapping  
     - *Content to Search*: Diplotype definitions, phenotype categories  
     - *Example*: "What diplotypes define a CYP2B6 intermediate metabolizer?"  

  5. **Gene/Drug Basic Information**  
     - *Purpose*: Provide foundational mechanistic definitions  
     - *Content to Search*: Gene functions, drug mechanisms, enzyme roles  
     - *Example*: "What is the function of the CYP2B6*6 allele?"  

  6. **Special Population Dosing**  
     - *Purpose*: Extract dosing for special populations  
     - *Content to Search*: Pediatric/pregnancy dosing, comorbidity adjustments  
     - *Example*: "What is the dosing recommendation for efavirenz in children weighing <40kg with CYP2B6 poor metabolizer phenotype?"  

example format {
"question": "How should [drug] dosing be adjusted for [gene] [poor metabolizers]?",
"answer": "No CPIC guideline information available."
}

Generate 2000 question and answer pair
store the question and answer as qa dictionary
and store the 2000 qa dictionary as list
return the list of dictionary directly
"""

creative_prompt_template = """
## CPIC QA Dataset Generator
You are a pharmacogenomics expert creating structured QA datasets from CPIC guidelines. Follow this protocol:

#### 1. Question Generation Requirements
- **Exclusively generate drug-gene interaction questions**
- **Question Structure**:
  Design questions to extract specific clinical decision-making information. Below are the required question types with purposes, content targets, and examples:

  1. **Specific Dosing Guidance**  
     - *Purpose*: Extract precise drug dosage values and adjustment protocols  
     - *Content to Search*: Dosage numbers, adjustment protocols, clinical recommendations  
     - *Example*: "What is the recommended efavirenz dose for a CYP2B6 poor metabolizer?"  

  2. **Clinical Risk Stratification**  
     - *Purpose*: Obtain quantified adverse reaction risk data  
     - *Content to Search*: Risk multipliers, incidence rates, toxicity thresholds  
     - *Example*: "What is the increased risk of CNS adverse effects for a CYP2B6 poor metabolizer?"  

  3. **Pharmacokinetics/Pharmacodynamics Data**  
     - *Purpose*: Gather therapeutic drug monitoring values  
     - *Content to Search*: Plasma concentration ranges, therapeutic windows  
     - *Example*: "What is the suggested therapeutic range for plasma efavirenz concentrations?"  

  4. **Genotype-Phenotype Correspondence**  
     - *Purpose*: Clarify genotype to metabolic phenotype mapping  
     - *Content to Search*: Diplotype definitions, phenotype categories  
     - *Example*: "What diplotypes define a CYP2B6 intermediate metabolizer?"  

  5. **Gene/Drug Basic Information**  
     - *Purpose*: Provide foundational mechanistic definitions  
     - *Content to Search*: Gene functions, drug mechanisms, enzyme roles  
     - *Example*: "What is the function of the CYP2B6*6 allele?"  

  6. **Special Population Dosing**  
     - *Purpose*: Extract dosing for special populations  
     - *Content to Search*: Pediatric/pregnancy dosing, comorbidity adjustments  
     - *Example*: "What is the dosing recommendation for efavirenz in children weighing <40kg with CYP2B6 poor metabolizer phenotype?"  

#### 2. Answer Format Specification
{{
"Drug Name": "[Exact drug name from guideline]",
"Gene Name": "[Standard gene nomenclature]",
"CPIC Guideline Name": "[Full guideline title with publication date].pdf",
"Content to Search": "[Text specifying missing clinical parameters/guidance requiring guideline search]"
}}

#### 3. Processing Workflow
**Step 1: Process drug-gene pairs**  
- For each combination in input lists:  
  (Drug, Gene) → Generate 1 QA pair per question type  

**Step 2: Generate QA pairs**  
- Each guideline with multiple drugs and genes will create #drug times #gene QA pairs
- Each drug-gene combination will generate 6 QA pairs (one for each question type)  
- Ensure phenotype specificity (poor/intermediate/ultrarapid metabolizers)  

**Step 3: Format output**  
{{
"[Exact Guideline Title].pdf": [
{{
"question": "How should omeprazole dosing be adjusted for CYP2C19 poor metabolizers?",
"answer": {{
"Drug Name": "Omeprazole",
"Gene Name": "CYP2C19",
"CPIC Guideline Name": "Clinical Pharmacogenetics Implementation Consortium (CPIC) Guideline for CYP2C19 and Proton Pump Inhibitor Dosing (August 2020).pdf",
"Content to Search": "Dose adjustment protocol for omeprazole in CYP2C19 poor metabolizers"
}}
}},
{{
"question": "What is the toxicity risk for CYP2C19 poor metabolizers taking omeprazole?",
"answer": {{
"Drug Name": "Omeprazole",
"Gene Name": "CYP2C19",
"CPIC Guideline Name": "Clinical Pharmacogenetics Implementation Consortium (CPIC) Guideline for CYP2C19 and Proton Pump Inhibitor Dosing (August 2020).pdf",
"Content to Search": "Quantified risk data for adverse effects in CYP2C19 PMs using omeprazole"
}}
}}
// Additional QA pairs for other question types
]
}}

**Begin processing immediately upon receiving inputs.**
Drugs: {csv_drug_name}
Genes: {csv_gene_name}
CPIC guideline: {csv_guideline_name}
"""