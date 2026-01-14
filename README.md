# NED-Tree: Bridging the Semantic Gap with Nonlinear Element Decomposition Tree for LLM Nonlinear Optimization Modeling

Automating the translation of Operations Research (OR) problems from natural language to executable models is a critical challenge. While Large Language Models (LLMs) have shown promise in linear tasks, they suffer from severe performance degradation in real-world nonlinear scenarios due to semantic misalignment between mathematical formulations and solver codes, as well as unstable information extraction. In this study, we introduce **NED-Tree**, a systematic framework designed to bridge the semantic gap. **NED-Tree** employs (a) a sentence-by-sentence extraction strategy to ensure robust parameter mapping and traceability; and (b) a recursive tree-based structure that adaptively decomposes complex nonlinear terms into solver-compatible sub-elements. Additionally, we present **NEXTOR**, a novel benchmark specifically designed for complex nonlinear, extensive-constraint OR problems. Experiments across 10 benchmarks demonstrate that **NED-Tree** establishes a new state-of-the-art with 72.51% average accuracy, **NED-Tree** is the first framework that drives LLMs to resolve nonlinear modeling difficulties through element decomposition, achieving alignment between modeling semantics and code semantics. The **NED-Tree** framework and benchmark are accessible in the anonymous repository. https://anonymous.4open.science/r/NORA-NEXTOR.

## Figure 1: The Nonlinear Semantic Gap in LLM Optimization Modeling

![motivation](pic/S1_new_motivation.pdf)

## Figure 2: LLM Nonlinear Optimization Modeling in Existing Methods.

 (a) Accuracy of 4 categories. Category L (Linear): Linear baselines; Category A (Non-quadratic Powers): Involving high-order power terms; Category B (Fractional/Rational): Introducing complex ratios or average cost constraints; Category C (Logic/Indicator):
Containing piecewise functions or conditional costs.

(b) Proportion of 3 types of errors: type I: modeling semantic errors, type II: nonlinear code semantic errors, type III: other code writing errors

![observation](pic/S3_observation.pdf)

## Figure 3: NEDTree framework.

Our approach comprises three parts: (a) Sentence-by-Sentence Extraction, (b)Mapping from Modeling Semantic to NED-Tree, and (c) Mapping from NED-Tree to Coding Semantic. The aim is to align modeling semantics with code semantics.

![framework](pic/S4_framework.pdf)

## Figure 4: Ablation study of NEDTree on NEXTOR benchmark. 

he left line is a line graph showing the difference between the module in the ablation study and NEDTree. The further to the left, the greater the difference.

![AblationStudy](pic/S5_AblationStudy.pdf)

## Figure 5: Case Study

![case_study](pic/S5_case_study.pdf)

