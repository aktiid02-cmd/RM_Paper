\documentclass[conference]{IEEEtran}
\IEEEoverridecommandlockouts
%\DeclareUnicodeCharacter{2212}{\textminus}
\usepackage{placeins}
\usepackage{tabularx}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage[utf8]{inputenc}
\usepackage{amsmath,amssymb,amsfonts}

%avoid caption and subcaptions, it changes table captions, rather use minipage for subfigures
%\usepackage{caption} 
%\usepackage{subcaption} 

\usepackage{textcomp}
\usepackage{xcolor}
\usepackage{textgreek}
\usepackage{algorithmic}
\usepackage{algorithm}
\usepackage{graphicx}
\usepackage[numbers]{natbib}
%\usepackage{cite} % REMOVE or KEEP COMMENTED - conflicts with natbib
\usepackage{hyperref}
\usepackage{url}
\usepackage{fancyhdr}
\usepackage{float}
\usepackage{flushend}

\usepackage{soul}
\sethlcolor{green!20} 

\usepackage{comment}
\usepackage{multirow}
\usepackage{tabularx}
\usepackage{array}
\usepackage{longtable}
\usepackage{afterpage}
\usepackage{stfloats}
\usepackage{booktabs}
\usepackage{balance}
\usepackage{gensymb}
\usepackage{makecell}
\usepackage{siunitx}
\newcolumntype{R}[1]{S[table-format=#1]}

\def\BibTeX{{\rm B\kern-.05em{\sc i\kern-.025em b}\kern-.08em
    T\kern-.1667em\lower.7ex\hbox{E}\kern-.125emX}}

% ================= PAGE-SAVING LATEX TUNING =================
\setlength{\textfloatsep}{8pt plus 2pt minus 2pt}
\setlength{\floatsep}{6pt plus 2pt minus 2pt}
\setlength{\intextsep}{6pt plus 2pt minus 2pt}
\setlength{\dbltextfloatsep}{8pt plus 2pt minus 2pt}
\setlength{\dblfloatsep}{6pt plus 2pt minus 2pt}
\renewcommand{\arraystretch}{0.95} % Tightens table row spacing slightly
% ============================================================

\begin{document}
\title{Evaluating SHAP Explanation Consistency Across Ensemble and Neural Models for software defect prediction under multicollinearity
}

\makeatletter
\newcommand{\linebreakand}{%
  \end{@IEEEauthorhalign}
  \hfill\mbox{}\par
  \mbox{}\hfill\begin{@IEEEauthorhalign}
}
\makeatother
\begin{comment}
    

\author{\IEEEauthorblockN{Shah Jalal Dip} %1\textsuperscript{st}
\IEEEauthorblockA{\textit{Department of Computer Science} \\
\textit{American International University-Bangladesh}\\
Dhaka, Bangladesh \\
23-54777-3@student.aiub.edu}
\and
\IEEEauthorblockN{Tazrean Maisa Silvy} %2\textsuperscript{nd} 
\IEEEauthorblockA{\textit{Department of Computer Science} \\
\textit{American International University-Bangladesh}\\
Dhaka, Bangladesh \\
23-53263-3@student.aiub.edu}
\linebreakand
\IEEEauthorblockN{Linda Maria Gomes} %3\textsuperscript{rd} 
\IEEEauthorblockA{\textit{Department of Computer Science} \\
\textit{American International University-Bangladesh}\\
Dhaka, Bangladesh \\
23-52979-3@student.aiub.edu}
\and
\IEEEauthorblockN{Md. Avi} %4\textsuperscript{th} 
\IEEEauthorblockA{\textit{Department of Computer Science} \\
\textit{American International University-Bangladesh}\\
Dhaka, Bangladesh \\
23-53476-3@student.aiub.edu}
\linebreakand
\IEEEauthorblockN{Al Israfil Niloy} %3\textsuperscript{rd} 
\IEEEauthorblockA{\textit{Department of Computer Science} \\
\textit{American International University-Bangladesh}\\
Dhaka, Bangladesh \\
22-49033-3@student.aiub.edu}
\and
\IEEEauthorblockN{Kamruddin Nur} %6\textsuperscript{th} 
\IEEEauthorblockA{\textit{Department of Computer Science} \\
\textit{American International University-Bangladesh}\\
Dhaka, Bangladesh \\
kamruddin@aiub.edu}

}
\end{comment}
% % Configure header and footer
% \fancypagestyle{firstpage}{
%     \fancyhf{} % Clear all header and footer fields
%     % Header
%     \fancyhead[L]{2025 International Conference on \ldots , date of conference, place of conference, city, country}
%     % Footer
%     \fancyfoot[L]{979-8-3503-5750-9/25/\$31.00 ©2025 IEEE}
%     \renewcommand{\headrulewidth}{0pt} % Remove header line
%     \renewcommand{\footrulewidth}{0pt} % Remove footer line
% }

\maketitle
%\thispagestyle{firstpage}

\begin{abstract}
Software Defect Prediction (SDP) optimizes quality assurance by identifying fault-prone modules, but modern classifiers are evaluated solely on accuracy, leaving the consistency of their explanations unverified under multicollinearity. This study addresses this gap by investigating explanation consistency across structurally diverse machine learning models. We evaluate Random Forest, XGBoost, LightGBM, and a Multi-Layer Perceptron on a cleaned, deduplicated NASA JM1 dataset under leakage-free cross-validation, proposing a VIF-guided mitigation called RF-RID. Predictive metrics cluster closely (F1 0.371 to 0.446), but SHAP explanations diverge substantially, with MLP being the least consistent partner (CECI as low as 0.566). RF-RID has a high explanation alignment with the complete RF on the shared feature space (Spearman correlation (SC) = 0.958, CECI = 0.946) while preserving predictive performance (MCC to 0.2793). This proves that accuracy does not guarantee explanation stability, and VIF-guided pruning provides a practical mechanism to deliver trustworthy, auditable explanations for industry quality assurance.
\end{abstract}

\begin{IEEEkeywords}
 software defect prediction, SHAP, ensemble learning, explainability, multicollinearity 
 % \hl{(5 to 6 keywords)}
\end{IEEEkeywords}

\section{Introduction}
As software systems grow increasingly complex, early Software Defect Prediction (SDP) becomes essential for optimizing quality assurance budgets and reducing post-release debugging costs \cite{bib11, bib12}. While modern ensemble and neural classifiers are highly effective, existing comparative studies evaluate them strictly on classification accuracy \cite{bib14}, leaving the consistency of their explainable AI (XAI) feature attributions such as those generated via SHAP completely unexamined \cite{bib13}. Compounding this gap, standard NASA datasets like JM1 contain severe feature multicollinearity and a high rate of duplicate records (32.5\%) that can leak across train/test splits, silently inflating accuracy in prior literature that lacks rigorous deduplication \cite{bib1}. \\
To address these data-quality and explanation stability challenges, this study develops a leakage-free evaluation protocol across four structurally diverse classifiers (Random Forest, XGBoost, LightGBM, and an MLP), unifies global and local SHAP agreements via a novel Cross-model Explanation Consistency Index (CECI), and proposes a VIF-guided feature removal mitigation (RF-RID) to stabilize Random Forest's explanations under collinearity.

The key contributions of this study are as follows:
\begin{enumerate}
\item \textbf{Leakage-aware software defect prediction:} We propose an algorithmic framework to systematically conduct exact-duplicate removal, label-conflict detection and strict cross-validated sampling to avoid data leakage in the process of imbalance handling. 
\item \textbf{VIF guided RF-RID feature redundancy reduction:} We introduce a Variance Inflation Factor-inspired feature decomposition approach for Random Forest models (RF-RID) that limits feature redundancy, preserves MCC and F1 scores, and makes the explanations of the features more stable. 
\item \textbf{CECI metric for explanation consistency evaluation:} We introduce the Cross-model Explanation Consistency Index (CECI), a quantitative measure that is composed of both global and local feature alignment to measure agreement between SHAP explanations of the same feature across models. 
\item \textbf{Empirical Evidence:} We demonstrate that predictive accuracy is not enough for trustworthy explanations; that models with a little less accuracy can provide a much more consistent and reliable explanation rationale. 
\end{enumerate}
The remainder of this paper is organized as follows. Section \ref{sec:RW} reviews related work. Section \ref{sec:PS} presents the methodology, including data-quality analysis, model development, RF-RID, and the CECI framework. Section \ref{sec:ERD} presents the experimental results. Section \ref{SEC: limF} discusses the limitations and future work.
Finally, Section \ref{sec:con} concludes the paper. 

\section{Related Works} \label{sec:RW}

Software defect and bug prediction have been extensively studied using both traditional machine learning (ML) and deep learning (DL) methods. Recently, researchers have investigated NASA metric datasets, ensemble methods, feature selection, and hybrid intelligent software methods to enhance prediction accuracy, tackle computational complexity, and resolve class imbalance before software deployment.

%\citeauthor{bib1} \cite{bib1} 
 A machine learning-based approach was proposed for predicting software bugs specifically on the NASA JM1 dataset \citep{bib1}. Using an 80:20 train-test split without any duplicate-removal step, they compared seven ML models (Na\"ive Bayes, Decision Tree, Random Forest, SVM, Logistic Regression, ANN, and KNN); Random Forest was the best performer at 81\% accuracy. Because JM1 is known to contain a substantial fraction of duplicate rows, a plain random split of this kind risks leaking near-identical instances across the train/test boundary, which can inflate reported accuracy independently of genuine model quality a concern our data-quality analysis (Section~\ref{subsec:dataquality}) directly investigates and quantifies for the same dataset. MLP, CNN, and LSTM architectures were tested through a seven-phase pipeline with statistical validation on twelve NASA MDP datasets, with LSTM achieving 93.5\% accuracy at roughly double the training time of MLP \citep{bib2}. Eight ML/DL algorithms were compared on a large unified bug dataset with extreme class imbalance, finding LSTM strongest overall but all models constrained to low binary F1 \citep{bib3}. IECGA, a genetic-algorithm feature-selected soft-voting ensemble was proposed reaching 95.1\% accuracy on PC1 but sensitive to parameter tuning on other datasets \citep{bib4}.RF, SVM, Na\"ive Bayes, ANN, and CNN were combined with hybrid SMOTE-Tomek sampling on JM1 alone, without examining interpretability or cross-project generalizability  \citep{bib7} .Opt-aiNet feature selection was applied across five NASA datasets without addressing class imbalance directly or considering instance-level explanation behavior \citep{bib6}.

In general, the reviewed literature indicates that SDP performance can be enhanced with advanced techniques, but persistent challenges extreme class imbalance, unexamined duplicate/conflicting instances, a lack of statistical validation, and the absence of any explanation-consistency analysis remain prevalent. These gaps motivate the leakage-aware, explanation-aware evaluation protocol proposed here. 

The summary of the related works is presented in Table~\ref{tab:related_work}. 

\begin{table*}[ht]
\centering
\caption{Summary of related studies on software defect prediction}
\label{tab:related_work}
\resizebox{\textwidth}{!}{%
\begin{tabular}{c c p{4.2cm} p{3.6cm} p{3.2cm} p{4.6cm}}
\toprule
\textbf{Ref.} & \textbf{Year} & \textbf{Method / Algorithm} & \textbf{Dataset} & \textbf{Results} & \textbf{Limitations} \\
\midrule
{\cite{bib1}} & 2024 & RF, SVM, Decision Tree, Na\"ive Bayes, Logistic Regression, ANN, KNN (80:20 split) & NASA JM1 & Random Forest: 81\% accuracy (best) & No duplicate/leakage analysis; traditional ML only \\
{\cite{bib4}} & 2024 & GA-based feature selection + soft-voting ensemble (RF, SVM, NB) -- IECGA & NASA PC1 & accuracy: 95.1\% & GA sensitive to parameter tuning \\
{\cite{bib7}} & 2024 & RF, SVM, NB, ANN, CNN + hybrid SMOTE-Tomek sampling & NASA JM1 & RF: 82.3\% accuracy, F1: 0.898 & Single dataset; no interpretability analysis \\
{\cite{bib2}} & 2025 & MLP, CNN, LSTM; 7-phase pipeline with statistical validation & NASA MDP datasets & LSTM: 93.5\% accuracy; F1: 93.6\%; ROC-AUC: 0.947 & LSTM training time $\approx$2$\times$ MLP; unsuitable for real-time deployment \\
{\cite{bib6}} & 2021 & Opt-aiNet feature selection + SVM, KNN, NB, DT, LDA, RF & NASA datasets & Decision Tree: 94.82\% accuracy & No deep learning; imbalance not addressed \\
\bottomrule
\end{tabular}}
\end{table*}

%\begin{figure*}
	%\centering
	%\includegraphics[width=\linewidth]{model.png}
	%\caption{Workflow of the proposed model training validation, and testing on the dataset.}
	%\label{ProposedModel}
%\end{figure*}

\section{Methodology}\label{sec:PS}
This study aimed to combine feature redundancy management, machine learning classification, post-hoc explainability and quantitative explanation consistency metrics to provide trustworthy software defect prediction. The entire process has been segmented into six phases. The workflow of this study is presented in fig. \ref{ProposedModel}. 
\begin{figure*} [ht]
	\centering
	\includegraphics[width=\linewidth]{workflow.jpg}
	\caption{Workflow of the proposed model training validation, and testing on the dataset.}
	\label{ProposedModel}
\end{figure*}
\subsection{Datasets and Data-Quality Analysis}
\label{subsec:dataquality}

We evaluated our framework on the NASA MDP JM1 dataset which contains 13,204 rows with 21 static code attributes (Halstead, McCabe, and lines-of-code metrics) and a binary defect label. Before modeling, we executed a rigorous data-quality pass:
\begin{enumerate}
\item Exact-Duplicate Detection: We identified and removed 4,296 identical rows (32.5\% of the raw dataset) to prevent cross-split leakage.
\item Label-Conflict Detection: We isolated 88 feature-vector groups (176 rows) with identical feature values but conflicting labels; these are flagged and retained as they represent an irreducible noise ceiling.
\end{enumerate}

Deduplication yields 8,908 unique rows and shifts the defect rate from 15.9\% (raw) to 22.5\% (cleaned), indicating that duplicates were disproportionately non-defective. This deduplication is critical: standard 80:20 splits on undeduplicated files \cite{bib1} allow duplicates to leak across splits, inflating performance. The cleaned dataset is split 80:20 through stratified sampling into a development set (7,126 rows) and a locked independent test set (1,782 rows).

\subsection{Data Preprocessing and Imbalance Handling}
\label{subsec:preprocessing}

Missing values are median-imputed, and features are standardized using training-fold statistics to avoid leakage. To handle the 77.5\%/22.5\% class imbalance, prevent synthetic sample leakage, SMOTE was only used within the training partition of each fold in cross-validation. No oversampling of validation or test samples was done. The best hyperparameters were chosen, and the final models were retrained on the entire development set, with the use of SMOTE only on the development training set and tested on the locked test set without using SMOTE.

\subsection{Baseline Models}
\label{subsec:ensemble_framework}

We trained and evaluated four diverse architectures: Random Forest (RF), XGBoost, LightGBM, and a Multi-Layer Perceptron (MLP). Hyperparameters are optimized via grid search inside cross-validation only: 
\begin{enumerate}
\item Random Forest: Ensemble of decision trees based on the whole feature set, which are measured for performance before feature redundancy decomposition.($n_{\text{estimators}}{=}300$, $\text{max\_depth}{=}10$), 
\item XGBoost: Gradient-boosted decision tree algorithm with  ($\text{learning\_rate}{=}0.01$, $\text{max\_depth}{=}3$, $n_{\text{estimators}}{=}200$), 
\item LightGBM: Leaf-wise Gradient Boosting framework with  ($\text{num\_leaves}{=}63$, $\text{max\_depth}{=}20$)
\item MLP: A feedforward neural network with single hidden layer of 50 neurons using $tanh$ as an activation function and Adam for optimization. 
\end{enumerate}
The final models are retrained on the entire development set and evaluated once on the locked test set at a 0.50 threshold.

\subsection{Proposed RF-RID Model}
\label{subsec:rfrid}
The collinear feature splits issues in tree-based ensembles are addressed by proposing Random Forest with Redundancy Identification and Decomposition (RF-RID). There are four sequential steps in the workflow:  
\begin{enumerate}
\item VIF Calculation: For each feature $j$ compute the Variance Inflation Factor ($\text{VIF}_j$) as follows: $$\text{VIF}_j = \frac{1}{1 - R_j^2}$$ where $R_j^2$ is the coefficient of determination derived when feature $j$ is regressed against all other features.
 \item Iterative Feature Reduction: Remove the attribute with the highest VIF above some threshold until the attributes are no longer multicollinear, leaving 12 attributes, after pruning ( \texttt{e}, \texttt{n}, \texttt{v}, \texttt{b}, \texttt{branchCount}, \texttt{total\_Opnd}, \texttt{lOCode}, \texttt{total\_Op}, \texttt{loc}) 
\item Model Training: Train Random Forest ensemble on the non-redundant subset of the features. 
\item Stability Evaluation: Compare various prediction metrics with the stability of attributions from full-feature baselines.
\end{enumerate}
 RF-RID is evaluated on the locked test set, and its SHAP explanations are compared with the full RF's explanations (restricted to the same 12 features) to prevent feature-set size from confounding consistency scores.

\subsection{SHAP Explainability and Consistency Analysis}
\label{subsec:shap_analysis}
SHAP (Shapley Additive Explanations) is used as a post-hoc model-agnostic explanation method to generate local feature attributions and global orderings of feature importance. We introduced the Cross-model Explanation Consistency Index (CECI) to numerically measure the agreement between the explanations produced by different model pairs $(A, B)$. The global and local attribution dimensions are separated first and then the unified score is computed: 
Global Component ($\text{CECI}_{\text{global}}$): The mean of the normalized global Spearman rank correlation ($\hat{\rho}^{\text{glob}}$) and normalized global cosine similarity ($\hat{c}^{\text{glob}}$) between global feature importance vectors:$$\text{CECI}_{\text{global}} = \frac{\hat{\rho}^{\text{glob}} + \hat{c}^{\text{glob}}}{2}$$
Local Component ($\text{CECI}_{\text{local}}$): The mean of the normalized local Spearman rank correlation ($\hat{\rho}^{\text{loc}}$) and normalized local cosine similarity ($\hat{c}^{\text{loc}}$) across instance-level attribution vectors:$$\text{CECI}_{\text{local}} = \frac{\hat{\rho}^{\text{loc}} + \hat{c}^{\text{loc}}}{2}$$
Final CECI Score: The unweighted average of the global and local components:$$\text{CECI}_{A,B} = \frac{\text{CECI}_{\text{global}} + \text{CECI}_{\text{local}}}{2}$$

\subsection{Statistical Validation and Robustness}
For each model pair, local Spearman correlations across the 300 test instances ( drawn once form the locked tet set to bound kernelSHAP cost) are evaluated against zero via a Wilcoxon signed-rank test with Holm-Bonferroni correction. Standard errors and robustness are assessed using bootstrap resampling (5,000 resamples of local observations with 95\% percentile confidence intervals) and leave-one-pair-out sensitivity analysis.

\section{Results and Discussions } \label{sec:ERD}

\subsection{Predictive Performance}
\label{subsec:predictive_performance}

Table~\ref{tab:model_performance} reports performance on the locked test set. RF-RID attains the highest MCC (0.279), RF the highest ROC-AUC (0.707)
(Fig.~\ref{fig:ceci_heatmap}), XGBoost the peak F1-score (0.446) and Recall
(0.491), and LightGBM the highest Accuracy (0.763) and Precision (0.460). While ranking abilities (PR-AUC: 0.405--0.421) are highly comparable
(Fig.~\ref{fig:rfrid_shap}), these absolute metrics are lower than undeduplicated literature studies. This gap is directly
attributable to our leakage-free split (Section~\ref{subsec:dataquality}), which prevents duplicate rows from inflating performance on both sides of the boundary.

\begin{table}
\centering
\caption{Comparison of base line models and Proposed RF-RID on the locked JM1 test set}
\label{tab:model_performance}
\resizebox{\linewidth}{!}{%
\begin{tabular}{lcccccc}
\toprule
\textbf{Model} & \textbf{Acc} & \textbf{Prec} & \textbf{Rec} & \textbf{F1} & \textbf{MCC} & \textbf{AUC} \\
\midrule
MLP    & 0.7160 & 0.3851 & 0.4389 & 0.4103 & 0.2251 & 0.6736 \\
LightGBM & {0.7626} & {0.4596} & 0.3117 & 0.3715 & 0.2384 & 0.6878 \\
XGBoost  & 0.7256 & 0.4087 & {0.4913} & {0.4462} & 0.2678 & 0.6940 \\
RF     & 0.7480 & 0.4391 & 0.4314 & 0.4352 & {0.2731} & {0.7073} \\
\textbf{RF-RID(Proposed)} & \textbf{0.7508} & \textbf{0.4450} & \textbf{0.4339} & \textbf{0.4394} & \textbf{0.2793} & \textbf{0.7026}\\
\bottomrule
\end{tabular}}
\end{table}

Crucially, no model dominates across all metrics, and F1-scores cluster within a narrow 0.075 range and remain effectively tied on predicitive performance. because predictive metrics alone cannot single out an optional model, expalaination consistency becomes the decisive criterion.

\subsection{Multicollinearity in the JM1 Feature Space}
Table~\ref{tab:multicollinearity_pairs} summarizes the collinear relationships within the JM1 dataset. Halstead effort ($e$) and time estimate ($t$) share an exact definitional identity ($t=e/18$, $r>0.99999$, VIF $\approx 4.48 \times 10^{13}$), as do volume ($v$) and bug estimate ($b$) ($b=v/3000$, $r=0.9996$). Additionally, length ($n$) overlaps near-perfectly with total operators ($r=0.995$). Overall, nine of the 21 features exhibit severe collinearity (VIF $>25$).


\begin{table}
\centering
\caption{Selected feature pairs with severe multicollinearity (VIF and Pearson $r$).}
\label{tab:multicollinearity_pairs}
\resizebox{\linewidth}{!}{%
\begin{tabular}{llcl}
\toprule
\textbf{Feature 1} & \textbf{Feature 2} & \textbf{$r$} & \textbf{Relationship} \\
\midrule
$e$ (effort) & $t$ (time est.) & 1.0000 & $t = e/18$ \\
$v$ (volume) & $b$ (bug est.) & 0.9996 & $b = v/3000$ \\
$n$ (length) & total\_Op & 0.9954 & near-linear overlap \\
$v_g$ (cyclomatic) & branchCount & 0.9816 & shared decision logic \\
loc & lOCode & 0.9223 & overlapping line counts \\
\bottomrule
\end{tabular}}
\end{table}

\subsection{Proposed RF-RID Performance and Attribution Stability}
The purpose of RF-RID is not to outperform all baselines on every predictive
metric, but to serve as an explainability-oriented mitigation that addresses the attribution instabilities caused by multicollinearity. RF-RID reduces model complexity by removing nine redundant attributes and improves the F1-score (0.4394) and MCC (0.2793) of Random Forest. Most importantly, over the shared feature space it yields highly consistent explanations relative to the full Random Forest, with a global Spearman rank correlation of $\rho_g = 0.958$ and a CECI of 0.946, confirming that VIF-guided redundancy removal reduces mitigates collinearity without degrading predictive power. The corresponding confusion matrix on the locked test set is shown in Fig.~\ref{fig:vif}, and a full comparison with related studies is given in Table~\ref{tab:comparison_work}.

\begin{table}[t]
\centering
\caption{Comparison with related software defect prediction studies}
\label{tab:comparison_work}
\setlength{\tabcolsep}{3pt}
\renewcommand{\arraystretch}{1.2}
\scriptsize
\begin{tabular}{@{}l >{\raggedright\arraybackslash}p{1.75cm} l >{\raggedright\arraybackslash}p{1.35cm} c >{\raggedright\arraybackslash}p{1.55cm}@{}}
\toprule
\textbf{Study} & \textbf{Method} & \textbf{Data} & \textbf{Best Result} & \textbf{XAI} & \textbf{Leakage Control} \\
\midrule
\cite{bib1} & RF, SVM, DT, NB, LR, ANN, KNN & JM1 & RF: 81\% Acc. & No & No dup.\ analysis \\
\cite{bib7} & RF, SVM, NB, ANN, CNN + SMOTE--Tomek & JM1 & RF: 82.3\% Acc.; F1: 0.898 & No & Limited \\
\cite{bib2} & MLP, CNN, LSTM & MDP suite & LSTM: 93.5\% Acc. & Part. & Stat.\ valid.\ only \\
\cite{bib6} & Opt-aiNet + ML & NASA (5) & DT: 94.82\% Acc. & No & Not addressed \\
\midrule
\textbf{Proposed}& \textbf{RF-RID + SHAP + CECI} & \textbf{JM1} & \textbf{MCC: 0.279; F1: 0.439} & \textbf{Yes} & \textbf{Dedup.\ + LF-CV} \\
\bottomrule
\end{tabular}
\end{table}
\begin{figure}[ht]
\centering
\includegraphics[width=\linewidth]{matrix.png}
\caption{Confusion matrix of RF-RID on the locked JM1 test set ($\text{threshold} = 0.50$)}
\label{fig:vif}
\end{figure}
\begin{figure}[ht]
\centering
\includegraphics[width=\linewidth]{precision_recall_curves.png}
\caption{Precision-Recall curves for baseline RF ($\text{AP} = 0.421$) and RF-RID ($\text{AP} = 0.410$)}
\label{fig:rfrid_shap}
\end{figure}

\subsection{Cross-Model Explanation Consistency (CECI)}
\label{subsec:ceci}
Table~\ref{tab:ceci_consistency} details the pairwise CECI metrics for comparison of the RF-RID with baseline models. The results indicate high attribution alignment with the full RF with a global Spearman correlation of $\rho_g = 0.958$, a global cosine similarity of $S_g = 0.979$ and a CECI of 0.946. Both statements are true: VIF-guided redundancy removal will remove collinearity without changing the decision logic that underlies the Random Forest. The agreement is consistent across architectures, with RF-RID having similar performance as tree-based baselines (CECI = 0.809 vs. XGBoost; CECI = 0.798 vs. LightGBM), and it is quite different from the neural MLP model (CECI = 0.566; global $\rho_g = -0.021$). The explanation variance is mostly explained by the architectural differences and not by the accuracy of the classification which can be confirmed by Wilcoxon signed ranked tests ($p < 10^{-14}$).

\begin{table}[t]
\centering
\caption{ SHAP Explanation Consistency (CECI) between RF-RID and Baseline Models. Metrics: global Spearman ($\rho_g$), global cosine ($S_g$), local Spearman ($\bar{\rho}_l$), local cosine ($\bar{S}_l$).}
\label{tab:ceci_consistency}
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lccccc}
\toprule
\textbf{vs.} & $\rho_g$ & $S_g$ & $\bar{\rho}_l$ & $\bar{S}_l$ & \textbf{CECI} \\
\midrule
MLP  & $-0.021$ & 0.652 & $-0.028$ & $-0.076$ & 0.566 \\
LGBM & 0.748 & 0.952 & 0.331 & 0.351 & 0.798 \\
XGB  & 0.776 & 0.740 & 0.499 & 0.458 & 0.809 \\
\textbf{RF}   & \textbf{0.958} & \textbf{0.979} & \textbf{0.786} & \textbf{0.846} & \textbf{0.946} \\
\bottomrule
\end{tabular}
\end{table}

\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{roc_curves.png}
\caption{ROC curves comparing baseline RF ($\text{AUC} = 0.707$) and proposed RF-RID ($\text{AUC} = 0.703$).}
\label{fig:ceci_heatmap}
\end{figure}

\subsection{Intra-Model Stability}
We compared the intra-model self-consistency between multiple repetitions of SHAP executions with the Intra-Model Stability Index (IMSI). Full Random Forest exhibits high self-consistency ($\text{IMSI} = 0.920$), followed by XGBoost (0.904), MLP (0.880), and LightGBM (0.866). Different from self-consistency across runs and even though the two models share only 12 features, RF-RID shows high attribution consensus with baseline full RF (CECI = 0.946, global Spearman $\rho = 0.958$). Reporting IMSI and CECI separately makes it clear that there are two aspects of trustworthy XAI; that of intra-model stability and that of cross-variant alignment.

\subsection{Performance–Consistency Decoupling}
Comparing Table~\ref{tab:model_performance} and Table~\ref{tab:ceci_consistency} confirms our core finding: predictive accuracy and explanation consistency are decoupled. LightGBM leads on Accuracy and Precision yet is not RF-RID's most consistent partner (CECI = 0.798), while the less accurate XGBoost aligns more closely (CECI = 0.809). The MLP matches the tree ensembles on predictive metrics but is the least consistent of all (CECI = 0.566). Selecting models on accuracy alone, without verifying explanation stability, thus risks deploying one whose attributions are volatile and inconsistent with equally accurate alternatives.

\section{Limitations} \label{SEC: limF}
Our framework enhanced the rigor with which software defect prediction is evaluated, but comes with some drawbacks. First, experiments are conducted only on the NASA MDP JM1 dataset and not on the cross-project generalizability of diverse software repositories is not verified. Second, the model is tested at a fixed classification threshold of 0.50; in real operational settings this could be problematic because of the need for cost sensitivity. Third, leaving the 176 label-conflicting instances in place will leave natural data noise, but provide an error floor that cannot be overcome. Lastly, instead of comparing post-hoc SHAP attributions to human expert rationales for bugs, they are compared to each model.

\section{Conclusion and Future Work}\label{sec:con}
We evaluated four structurally diverse classifiers (RF, XGBoost, LightGBM, and
MLP) alongside our proposed RF-RID framework on a deduplicated, leakage-free
NASA JM1 dataset, and introduced the Cross-model Explanation Consistency Index
(CECI) to quantify SHAP attribution agreement across models. Our central finding
is that predictive accuracy does not imply explanation trustworthiness: although
predictive metrics cluster tightly across all models, their explanation
rationales diverge sharply, most notably between tree-based ensembles and the
neural MLP. By pruning nine collinear metrics, RF-RID preserves the predictive
performance of the full Random Forest (comparable MCC) while providing very aligned attributions in terms of both the global Spearman ρ (0.958) and the CECI (0.946) on the shared feature space with the full Random Forest, demonstrating that VIF-guided redundancy removal yields more
auditable explanations while maintaining comparable predictive performance.

Future work will extend CECI to additional repositories (CM1, KC1, PC1, and
PROMISE), integrate dynamic decision-threshold optimization within
cross-validation folds, and incorporate human-in-the-loop validation to assess
developer trust in the stabilized SHAP attributions.
\begin{comment}
\section*{Acknowledgment}
The authors express their sincere gratitude to the Ubiquitous, Cloud, and Human-Computer Interaction (UCH) Research Group, Department of Computer Science, American International University-Bangladesh for supporting this research.
\end{comment}
\bibliographystyle{IEEEtranN}
\bibliography{refs}
\balance

\end{document}
