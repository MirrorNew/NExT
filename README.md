# NORA: NExT OR Agent

Automating the translation of complex Operations Research (OR) problems from natural language to numerical solutions is a major challenge. While Large Language Models (LLMs) have shown promise, current static prompt-based methods struggle with long-text, complex and nonlinear problems, while fine-tuning is prohibitively expensive. We introduce **NORA**, a novel, context-based multi-agent framework that designed to overcome these limitations. NORA utilizes a dynamic context with specialized agents to manage long-text element extraction (Long-Search-Agent) and ensure model-code alignment for nonlinearities (Align-Model-Agent and RE-Code-Agent). Additionally, to address the deficiencies of existing benchmarks and facilitate evaluation, we also developed **NExTOR**, a benchmark for complex nonlinear and extensive-constraint OR problems. Experiments on 10 benchmarks establish NORA as the new state-of-the-art with 72.51\% average accuracy. This result is 7\% higher than prompt-based methods and matches the performance of fine-tuned models in 80\% of cases without costly training. NORA and NExTOR are available in anonymous repository https://github.com/MirrorNew/NExT.

## Fig 1: Contributions

 (a)Comparison of NExTOR with Other Benchmarks. HP, FP, and ELP are nonlinear programming involving high-order
powers (greater than 2), fractions & fractional powers, and exponentials & logarithms, respectively. ✓ indicates coverage of this
aspect, and a horizontal line indicates that this aspect is not covered.(b) NExTOR’s Statistics.(c) NORA and Shared Context
Mechanism.

![contribution](D:\LLMProject\git_NORA\NExT\pic\contribution.png)

## Fig 2: NORA Framework

![framework](D:\LLMProject\git_NORA\NExT\pic\framework.png)

## Fig 3: NETA

NExTOR synthesis method. This synthesis method first collect data from 3 primary channel, and comprises 3 core
step:(b)Forward step, (c)NExT step and (d)Verification step, based on 4 Guiding Principles

![sysMethod](D:\LLMProject\git_NORA\NExT\pic\sysMethod.png)

## Table 1: Main Result

Accuracy (pass@1) comparison across multiple datasets and methods.

![result](D:\LLMProject\git_NORA\NExT\pic\result.png)

## Figure 4: Ablation Study.

The left line is a line graph showing the difference between the module in the ablation study andNORA. The further to the left, the greater the difference.

![AblationStudy](D:\LLMProject\git_NORA\NExT\pic\AblationStudy.png)