# Solving Nonlinear and Complex-Constraint Operations Research Problems with Context-based LLM Agent

Automating the translation of complex Operations Research (OR) problems from natural language descriptions into numerical solutions remains a significant challenge. While large language models (LLMs) show promise, current fine-tuning methods are often prohibitively expensive and exhibit poor generalization, whereas prompt-based approaches struggle with robust information extraction and nonlinear processing. To address these limitations, we propose NORA, a novel framework employing a context-based LLM agent to solve OR problems with nonlinearity and complex constraints. NORA enhances accuracy and traceability through sentence-by-sentence extraction and enforces strict alignment between the problem model and generated code—critical for handling nonlinearity and ensuring code reliability. Furthermore, we introduce NEXTOR, a benchmark of OR problems featuring intricate nonlinearities and extensive constraints, designed to enable rigorous evaluation and overcome dataset limitations. Experiments across 10 benchmarks establish NORA as state-of-the-art, achieving 72.51% average accuracy—surpassing the best fine-tuning model by 13.02% and the leading non-fine-tuning model by 6.27%. The NORA framework and NEXTOR benchmark are publicly accessible in the anonymous repository: https://anonymous.4open.science/r/NORA-NEXTOR.

## Figure 1: NORA Framework.

(a) How the problem is solved in NORA, from problem to final answer. (b) NORA ’s Architecture. OR
problem 𝑝 becomes extracted information from Long-Search-Agent(P means for parameters extraction, E means for elements
extraction). then, result transforms to math model from Model Agent and AlignModel Agent. Third, result transforms to
python code from CodeSolver, and transforms to numerical answer 𝑎𝑛𝑠 by solver(GUROBI). (c) NORA’s core strategies, including
Sentence-by-sentence Extraction, 3-Step Model Alignment and CodeSolver and Errors Advice.

![contribution](pic\0801NORAframework.png)

## Figure 2: NEXTOR Synthesis Method. 

This synthesis method first (a) collects data from three primary channels, and then proceeds

![framework](pic\0801NEXTORSynthesisMethod.png)



## Figure 3: Ablation study of NORA on NEXTOR benchmark.

 The left line is a line graph showing the difference between the module
in the ablation study and NORA. The further to the left, the greater the difference.

![AblationStudy](D:\LLMProject\git_NORA\NExT\pic\AblationStudy.png)

## Figure 4: NEXTOR’s Statistics.

 (a) Problem type distribution, where HP, FP and ELP are nonlinear programming involving
high-order powers (greater than 2), fractions & fractional powers, and exponentials & logarithms, respectively. (b) Question
length distribution. (c) Variable numbers distribution.

![0801NEXTORStatistics](pic\0801NEXTORStatistics.png)

## Algorithm 1 CodeSolver: Generate, Execute and Repair

<img src="pic\0801CodeSolver.png" alt="sysMethod" style="zoom:40%;" />

## Table 1: Comparison of NEXTOR with other benchmarks. 

HP, FP, and ELP are nonlinear programming involving high-order
powers (greater than 2), fractions & fractional powers, and exponentials & logarithms, respectively. ✔ indicates coverage of
this aspect, and ✘ indicates that this aspect is not covered. Multimodality means that the benchmark has multiple languages,
character images and mermaid drawings. Redundant Content indicates the noise content and background.

![result](pic\0801NEXTORcomparison.png)

## Table 2: Comparison of Accuracy (pass@1) metric between NORA and other methods.

Underlined results indicate results that are second only to the SOTA, while bold results indicate the current SOTA. †: †Results are cited from their original papers [1, 10, 13, 15, 27, 29]. ‡: ComplexOR’s GT answer is based on LLMOPT’s open source repository[13]. OPTIBench and OPTMATH’s best models have no weights released. IR(Improvement Rate): + indicates that NORA outperforms the best performing method in this category, and - indicates not. IR(with NFT) is based on the best of all non-fine-tuning methods. IR(with FT) is based on the best accuracy of all fine-tuning methods.

![0801NORAResult](pic\0801NORAResult.png)

## Table 3: Comparison of AC and PR metrics between linear and nonlinear tasks on NEXTOR Benchmark.

<img src="pic\0801NEXTORACPR.png" alt="0801NEXTORACPR" style="zoom:40%;" />

