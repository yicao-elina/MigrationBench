from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
old = (ROOT/'source_snapshot/response_to_reviewers.tex').read_text()
comments = {}
for m in re.finditer(r'\\subsection\*\{([AC]\d+)\.(.*?)\}\s*.*?\\begin\{quote\}(.*?)\\end\{quote\}', old, re.S):
    comment = m[3].strip().replace(r'\\\path',r'\path').replace(r'\\\texttt',r'\texttt')
    comments[m[1]] = (m[2].strip(), comment)
assert set(comments)=={'A1','A2','A3','A4','A5','A6','A7','C8','C9','C10','C11','C12','C13'}

responses = {
'A1': r"""
We agree that the original comparison was ambiguous. The 0.336~eV reference (0.34~eV after rounding) belongs to a 61-atom, one-Cr, five-image G1 set. The earlier 0.41~eV result belongs to an 82-atom, two-Cr, twelve-image record with inconsistent calculators and a serialization defect. We have withdrawn the 0.41~eV claim and the judgement based on it. These were not two evaluations of the same path.

A final raw-record check also corrected an erroneous zero in the supplied seed matrix: Scratch-600K seed 123/within-QL self has a finite peak of 2.351~eV and is nonconverged. Seed 234/in-gap self is also nonconverged. Both corrections are supported by matching model SHA256 identities, and the revised figure marks all five nonconverged entries with crosses.

The revised paper defines the operators explicitly and presents fixed and self-relaxed profiles side by side in main Fig.~\ref{M-fig:profiles}. Fixed scoring compares energy assignments to common geometries. Self-relaxed NEB also changes the geometries and therefore tests model-driven path generation. A small optimized peak can result from sampling another path or basin; it does not establish a more accurate physical barrier. The canonical main-figure in-gap reference is 1.5243~eV, a different image set from G1.

\textbf{Takeaway.} The original ranking is withdrawn. The revised comparison separates energy calibration from path generation, and warning-bearing DFT references are described as image-set diagnostics. Converged, same-path electronic-structure validation would still be needed for quantitative kinetics.
\location{Main Secs.~\ref{M-sec:migration} and \ref{M-sec:operators}; Figs.~\ref{M-fig:profiles}--\ref{M-fig:taskmatrix}.}{SI Secs.~\ref{SI-sec:matrix_status} and \ref{SI-sec:legacy_paths}; Fig.~\ref{SI-fig:legacy_operator_audit}.}
""",
'A2': r"""
We agree that the frame shuffle did not rule out trajectory-adjacent leakage. The grouped protocol assigns complete source trajectories to train, validation or test. The 600~K partition contains 1,786/311/267 configurations.

For the 25 legacy NEB images, the declared training files have no exact global or Cr-local match and no SOAP cosine distance at or below $10^{-4}$. Minimum SOAP distances are $5.4\times10^{-4}$ for the full training set and $6.4\times10^{-4}$ for the 89-frame subset. The minimum composition-compatible local RMSD is 2.06~\AA, above the 0.05~\AA\ threshold. The RMSD comparison has few composition-compatible candidates, so it is not presented as a universal coverage certificate; the SOAP check covers the audited image set.

\textbf{The 0.16~eV value does not survive this control.} The grouped, three-seed FT-600K G1 discrepancy is $0.47\pm0.12$~eV. It replaces the original single-run estimate. These G1 controls are kept separate from the newer canonical paths; they are not silently reused as an overlap audit for every later image.
\location{Main Sec.~\ref{M-sec:models}, blue sentences [R1-A2] and [R1-C10].}{SI Sec.~\ref{SI-sec:overlap}, Table~\ref{SI-tab:a2_grouped_audit_summary} and Fig.~\ref{SI-fig:grouped_overlap}.}
""",
'A3': r"""
We have replaced the chance explanation with seed-resolved evidence. Three training seeds are reported for every contemporary trained condition in the main task matrix. For the canonical fixed within-QL images, Scratch-600K has a mean absolute reference discrepancy of $0.45\pm0.05$~eV. Its comparatively small discrepancy is reproducible across these three initializations, so we no longer dismiss it as chance.

The separate legacy control reaches the same narrower conclusion on Q1: the Scratch discrepancy is $0.51\pm0.11$~eV. Q2 and Q3 are more seed-dependent ($0.72\pm0.55$ and $1.77\pm1.00$~eV). All standard deviations are sample standard deviations over training seeds. They are not confidence intervals over materials or transition paths.

\textbf{Takeaway.} A repeatable fixed-image energy result does not establish reliable dynamics or a correct model-generated path. The same scratch training conditions fail rapidly in MD. We report that distinction and retain nonconverged pathway values as diagnostics, rather than calling them kinetic predictions.
\location{Main Sec.~\ref{M-sec:migration}, blue sentence [R1-A3]; Figs.~\ref{M-fig:taskmatrix} and \ref{M-fig:md}.}{SI Tables~\ref{SI-tab:current_neb_errors} and \ref{SI-tab:scratch5_path_resolved_descriptive}; Sec.~\ref{SI-sec:matrix_status}.}
""",
'A4': r"""
We have adopted the narrower scope. The title, abstract and Discussion now describe process-specific reliability for the tested MACE routes in Cr-doped \ce{Sb2Te3}. We do not claim a generalizable standard, a universal ordering of strategies, or a quantitative rule based on distance from the foundation pretraining distribution.

The supplied Cu/Li/Na path figure and the six-system endpoint-perturbation study remain in the SI as additional diagnostics. They do not provide a second fully validated migration benchmark, and we do not use them to justify broad transferability claims.

\textbf{Takeaway.} The study demonstrates a candidate way to test a model for a chosen process in one primary material. Broader validation and active learning are stated as future work.
\location{Main Introduction and final paragraph of Sec.~\ref{M-sec:discussion}, marked [R1-A4].}{SI Fig.~\ref{SI-fig:other_dopants} and Sec.~\ref{SI-sec:endpoint_perturbation_campaign}.}
""",
'A5': r"""
We agree that the target-data budget must be controlled. The revised primary 600~K comparison uses \textbf{the same 1,786 training configurations}, with the same 311/267 validation/test partitions and three seeds, for the full scratch and fine-tuned routes. This removes a difference in target-data count from that comparison. Architecture and optimization still differ between the registered routes, so we do not attribute the entire difference to pretraining alone.

We also correct the description of the existing Scratch-5\% run. It uses 89 configurations, not approximately 1,000, and is therefore \textbf{not} a data-matched control for the grouped FT model. Its G1 discrepancy of $0.68\pm0.20$~eV is retained only as a separate data-volume sensitivity result. We have removed the earlier argument that an exactly matched subset would necessarily break trajectory grouping.

\textbf{Takeaway.} The matched full-data pair addresses the budget comparison at the revised 1,786-frame budget. The 89-frame run cannot isolate the value of pretraining, and no architecture-controlled causal estimate or universal fine-tuning advantage is claimed.
\location{Main Sec.~\ref{M-sec:models}, blue sentence [R1-A5].}{SI Secs.~\ref{SI-sec:training} and \ref{SI-sec:overlap}; Tables~\ref{SI-tab:current_pointwise} and \ref{SI-tab:a2_grouped_audit_summary}.}
""",
'A6': r"""
We have narrowed both analyses. The trajectory projections use recorded energy, force and structural summaries, not native MACE hidden states. We now state this directly, report current original-space metrics and projection sensitivity, and discuss energy offsets, feature scale, separately fitted SOAP PCA bases and unequal trajectory horizons. For the current 927-frame matrix, the raw silhouette is 0.063 and the standardized value is $-0.059$; the displayed $t$-SNE score is 0.117.

The new SHAP figure explains \textbf{surrogate predictions of absolute energy error per atom} on a common 267-frame trajectory. The separate five-fold diagnostic $R^2$ values are $0.981\pm0.003$ (FT-600K), $0.973\pm0.003$ (FT-MultiT), $0.976\pm0.006$ (Scratch-600K) and $0.952\pm0.022$ (Scratch-MultiT). The final TreeSHAP model is fitted on the full sample; its near-unit in-sample fit is not offered as generalization evidence. Shuffled folds from one trajectory also do not test transfer to an independent trajectory.

The current rankings vary across training seeds and do not support a single dominant Cr--Cr causal channel. The SI coordinate-perturbation campaign contains 846 native-MACE endpoint relaxations over six dopant--host systems. It tests local response to actual atomic displacement, but does not isolate a SOAP channel or explain the migration discrepancy.

\textbf{Takeaway.} The projections describe sampled populations, and SHAP describes an error surrogate. Neither is presented as a mechanistic explanation of MACE.
\location{Main Secs.~\ref{M-sec:representations}--\ref{M-sec:shap}; Figs.~\ref{M-fig:representation}--\ref{M-fig:shap}.}{SI Secs.~\ref{SI-sec:representation}--\ref{SI-sec:shap}; Tables~\ref{SI-tab:current_original_space}--\ref{SI-tab:current_shap_fidelity}; endpoint appendix.}
""",
'A7': r"""
The Introduction now discusses all three requested neighbouring studies. Alghamdi et al.\ show successful zero-shot/lightly adapted MACE diffusion modelling in LiF; Bheemaguli et al.\ distinguish migration-energy and path-geometry performance; Deng et al.\ establish systematic softening as prior work~\citep{alghamdi2025maceLiF,bheemaguli2026mlipBarriers,deng2025softening}. Our large within-QL overestimates are not described as a new demonstration that softening exists.

We also corrected the statement of the contribution relative to our own report~\citep{cao2025migration}. That report already contained migration, sliding and representation analyses. Those topics are \textbf{not} claimed as new here. The revision adds or strengthens trajectory-grouped controls, additional seeds and checkpoints, explicit operator definitions, failure-horizon accounting, current common-frame SHAP analysis and original-space checks. It also withdraws the incompatible 0.41~eV result and replaces the unreproduced 0.16~eV estimate.

\textbf{Takeaway.} The contribution is a more carefully controlled and more narrowly interpreted reliability comparison, rather than the introduction of already reported task types.
\location{Main Sec.~\ref{M-sec:introduction}, related-work and prior-report paragraphs; marker [R1-A7].}{The corresponding controls and corrections are documented in SI Secs.~\ref{SI-sec:overlap}, \ref{SI-sec:legacy_paths}, \ref{SI-sec:representation} and \ref{SI-sec:shap}.}
""",
'C8': r"""
We confirm the exporter defect. Both affected scripts have been changed to fail closed when real QE forces are unavailable. The collector refuses a force-bearing training export without validated forces; the extended-XYZ exporter raises an error whenever forces are requested, which is its command-line default, and writes energy and geometry only when called with \texttt{-{}-no-forces}. No fabricated zero-force array is an accepted substitute.

The hash-bound provenance audit covers 9,163 records from the declared historical and grouped training splits. Exactly nine records have identically zero forces, all labelled one-atom Cr/Sb/Te isolated-atom energy references. There are zero unclassified and zero NEB-derived zero-force training records in those audited files. The frame manifest, machine-readable summary and audit executable accompany the revision, together with the corrected exporters.

\textbf{Takeaway.} The defect was real, but the audited training lineage contains no fabricated NEB-force frames. This is a dataset-specific conclusion, not a guarantee inferred from a model family name.
\location{Main Sec.~\ref{M-sec:models}, blue sentence [R1-C8].}{SI Sec.~\ref{SI-sec:force_audit}; accompanying zero-force manifest and corrected exporters.}
""",
'C9': r"""
We have removed the asymmetric short-run transport comparison. The revised MD figure uses recorded steps with a 1~fs timestep and a 200-step output interval. All runs are evaluated as continuations of the common 200~ps state towards a 1,000~ps target clock. A completed continuation is therefore 800~ps, not an independently generated 1~ns trajectory.

The current status table also changes the conclusion: the three foundation checkpoints and FT-600K seed 234 reach the target. The remaining five fine-tuned and six scratch runs terminate at recorded thresholds. Their actual horizons are shown in the SI; they are not extended or assigned a common long-time transport coefficient. Training seeds characterize model variation, while the common momenta/MD seed do not supply independent thermal replicas.

\textbf{Takeaway.} The revised result is an observed stability, structure and mobility comparison under a common continuation protocol. The earlier diffusion/conductivity ranking and the earlier statement that all fine-tuned runs were stable are removed. No heat-current record supports a thermal-conductivity claim.
\location{Main Sec.~\ref{M-sec:md}, Fig.~\ref{M-fig:md} and blue sentence [R1-C9].}{SI Sec.~\ref{SI-sec:md} and Table~\ref{SI-tab:md_horizons}.}
""",
'C10': r"""
The grouped comparisons now assign complete source trajectories to train, validation or test. The revised main parity figure reports grouped train and held-out test data separately, and its caption distinguishes unique configurations from the three-seed pooled prediction count. The original frame-level shuffle is not used as the independent generalization estimate.

The 600~K data-matched pair uses 1,786/311/267 train/validation/test configurations. Full uncentred test errors and model/data hashes accompany the figure. Cluster-centering in the energy display subtracts the same DFT offset from both axes and is disclosed as a visualization transformation, including the use of test-only cluster means; it is not a fitted deployment calibration.

\textbf{Takeaway.} Same-trajectory temporal leakage is controlled by grouping. Geometric similarity between different source trajectories is a separate question, for which the legacy path overlap audit has an explicitly limited scope.
\location{Main Secs.~\ref{M-sec:pointwise} and \ref{M-sec:models}, Fig.~\ref{M-fig:parity}, marker [R1-C10].}{SI Secs.~\ref{SI-sec:training} and \ref{SI-sec:overlap}; Table~\ref{SI-tab:current_pointwise}.}
""",
'C11': r"""
We agree. Panel d of the supplied main projection figure still reports a \textbf{two-dimensional $t$-SNE score}; its caption now says so explicitly. It is not relabelled as an original-space metric.

We independently computed the requested original-space diagnostics on the current 927-frame dataset, removing the inactive column. The Euclidean silhouette is 0.0627, cosine silhouette 0.0609 and standardized Euclidean silhouette $-0.0592$. The common 30-bin neighbour histogram gives 0.0564. These are the current values; the older 404-frame values of 0.33 and $-0.02$ are not reused for this figure. The SI also reports a $t$-SNE sensitivity check at three perplexities; under PCA initialization the three nominal seeds give identical embeddings, so the check spans three distinct settings.

\textbf{Takeaway.} The scale dependence does not support strong latent-separation claims. The new original-space table and audit figure, rather than the projection score, ground the revised interpretation.
\location{Main Sec.~\ref{M-sec:representations}, marker [R1-C11], and Fig.~\ref{M-fig:representation} caption.}{SI Table~\ref{SI-tab:current_original_space}, Table~\ref{SI-tab:current_tsne_sensitivity} and Fig.~\ref{SI-fig:current_feature_audit}.}
""",
'C12': r"""
We confirm the inactive column and correct the scope of the earlier response. The newly supplied 15-model reconstruction inherited a 6,192-column feature grammar containing the same all-zero placeholder. We therefore do not repeat the earlier statement that every reconstruction had already removed it.

The current executable audit verifies that the final column is identically zero and removes it before every recomputed distance and projection, leaving 6,191 features. A constant zero coordinate contributes nothing to Euclidean or cosine distance, so it carries no force-response information. The supplied image is retained as a descriptive reconstruction with that provenance disclosed. No force-sensitivity result is claimed, and the separate SHAP workflow uses computed SOAP/structural descriptors.

\textbf{Takeaway.} The placeholder is explicitly excluded from the current quantitative workflow and is not presented as a physical diagnostic.
\location{Main Sec.~\ref{M-sec:diagnostics}, blue sentence [R1-C12]; Fig.~\ref{M-fig:representation} caption.}{SI Sec.~\ref{SI-sec:representation}; executable current-feature audit.}
""",
'C13': r"""
The empty-stub criticism was valid for the reviewed snapshot. The revision does not rely on that advertised API. The accompanying package supplies the actual executable SHAP analysis, the common 267-frame XYZ, all 12 aligned evaluation tables, feature-label mapping, per-seed metrics and attributions, and the renderer for the selected top-20 figure. The current-feature audit and its input identity are supplied separately.

We reran the supplied analysis from the common XYZ and the 12 evaluation tables. All four checked outputs reproduce numerically: 12 per-seed metric rows, 1,628 family-feature rows, 4,884 seed-feature rows and 3,204 frame-aligned rows have identical column definitions and zero maximum numerical difference from the current source tables. The machine-readable comparison accompanies the revision.

The current common-frame, four-condition analysis replaces the earlier three-model, 150-frame reconstruction. We do not mix the earlier total-energy-eV surrogate metrics or its 30-seed rankings into the current meV-per-atom result. The Methods and SI specify the 407 features, separate diagnostic and final estimators, and the limited interpretation of the shuffled-fold check.

\textbf{Takeaway.} Reproducibility is tied to the nonempty workflows and figure-specific source tables included with this revision, rather than to an unimplemented benchmark package or a claim that one historical commit reproduces every current artifact.
\location{Main Code availability, marker [R1-C13], and Sec.~\ref{M-sec:diagnostics}.}{SI Sec.~\ref{SI-sec:shap}; accompanying executable source and figure data.}
"""
}

prefix = r"""\documentclass[11pt]{article}
\usepackage[a4paper,margin=23mm]{geometry}
\usepackage[T1]{fontenc}
\usepackage{lmodern,microtype,amsmath,amssymb,booktabs,array,xcolor}
\usepackage[version=4]{mhchem}
\usepackage{xurl}
\usepackage[numbers,sort&compress]{natbib}
\usepackage{xr-hyper}
\usepackage[colorlinks=true,linkcolor=blue,citecolor=blue,urlcolor=blue]{hyperref}
\providecommand{\sitab}[1]{Supplementary Table~\ref{SI-#1}}
\providecommand{\sifig}[1]{Supplementary Fig.~\ref{SI-#1}}
\providecommand{\file}[1]{\texttt{\detokenize{#1}}}
\externaldocument[M-]{revision1/sn-article-final}
\externaldocument[SI-]{revision1/sn-article-SI}
\usepackage{fancyhdr,needspace}
\pagestyle{fancy}\fancyhf{}\lhead{\small Response to Reviewer 1}\rhead{\small MigrationBench}\cfoot{\thepage}
\setlength{\headheight}{14pt}
\setlength{\parindent}{0pt}\setlength{\parskip}{.5em}
\newcommand{\location}[2]{\par{\small\textbf{Where to find the change.} #1\par #2}\par}
\begin{document}
\begin{center}
{\Large\bfseries Response to Reviewer 1}\\[4pt]
{\large When Does a Foundation Model Help?}\\
Process-Specific Reliability of Machine-Learned Interatomic Potentials
\end{center}

Thank you for identifying both the scientific ambiguities and the implementation issues. We have reorganized the paper around seven main figures and rewritten the interpretation around process-specific reliability. The revised comparison separates pointwise error, fixed-image energy assignments, model-relaxed paths and observed MD horizons. It no longer assumes that fine-tuning is universally preferable.

Several earlier claims required correction: the 0.16~eV estimate is not reproduced under grouped retraining; the incompatible 0.41~eV record is withdrawn; the 89-frame control is not called data-matched; and the newest MD data do not show all fine-tuned runs reaching the target. We also replace mechanistic representation claims with descriptive diagnostics. Warning-bearing DFT references remain an explicit limitation, so no validated activation-energy or rate claim is made for them.

Responses below quote each comment and give the direct correction and its location. In the marked manuscript, only the most directly relevant sentences are blue and carry [R1-A1]--[R1-C13] identifiers. Section, figure and table references below are resolved from the revised main text and SI.
"""
parts=[prefix]
for idx in ['A1','A2','A3','A4','A5','A6','A7','C8','C9','C10','C11','C12','C13']:
    title,comment=comments[idx]
    parts += [r'\Needspace{11\baselineskip}\section*{'+idx+'. '+title+'}',r'\begin{quote}\small '+comment+r'\end{quote}',
              r'\textbf{Response.} '+responses[idx]]
parts += [r'{\small\setlength{\bibsep}{2pt}\bibliographystyle{unsrtnat}',r'\bibliography{revision1/references-camera-ready}}',r'\end{document}']
(ROOT.parents[1]/'paper/revision1/response_to_reviewers.tex').write_text('\n\n'.join(parts)+'\n')
print('Wrote 13 direct responses with original comments.')
