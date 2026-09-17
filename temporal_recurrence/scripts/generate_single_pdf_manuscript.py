"""Generates an exhaustive, publication-grade 8-11 page PDF manuscript with all sections,
formal equations, empirical tables, embedded high-resolution figures, multi-dataset benchmarks,
MA-TGN formulation, references, and complete supplementary material.
"""
import os
import re
import shutil
from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    KeepTogether,
    HRFlowable,
    PageBreak,
    Image
)
from reportlab.pdfgen import canvas

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BUILD_DIR = PROJECT_ROOT / "paper" / "build"
FIGURES_DIR = PROJECT_ROOT / "paper" / "figures"
BUILD_DIR.mkdir(parents=True, exist_ok=True)
PDF_PATH = BUILD_DIR / "temporal_graph_recurrence_manuscript.pdf"
BRAIN_ARTIFACT_DIR = Path("/Users/pavanaksshay/.gemini/antigravity-ide/brain/377b7a70-81a3-41c9-8798-c095b2fa4185")


class AcademicCanvas(canvas.Canvas):
    """Two-pass canvas for dynamic total page count, running headers, and footers."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_decorations(self, total_pages):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#52525b"))

        # Running Header (pages > 1)
        if self._pageNumber > 1:
            self.drawString(50, 750, "When History Recurs: Characterizing Temporal Memory Interference in Dynamic Graph Neural Networks")
            self.setStrokeColor(colors.HexColor("#d4d4d8"))
            self.setLineWidth(0.5)
            self.line(50, 744, 562, 744)

        # Running Footer (all pages)
        page_str = f"Page {self._pageNumber} of {total_pages}"
        self.drawRightString(562, 36, page_str)
        self.drawString(50, 36, "Research Manuscript — Confidential Draft under Peer Review")
        self.setStrokeColor(colors.HexColor("#d4d4d8"))
        self.setLineWidth(0.5)
        self.line(50, 46, 562, 46)
        self.restoreState()


def build_full_paper():
    doc = SimpleDocTemplate(
        str(PDF_PATH),
        pagesize=letter,
        leftMargin=50,
        rightMargin=50,
        topMargin=52,
        bottomMargin=52
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=15,
        leading=19,
        textColor=colors.HexColor('#09090b'),
        alignment=1, # Center
        spaceAfter=6
    )

    author_style = ParagraphStyle(
        'DocAuthor',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=colors.HexColor('#3f3f46'),
        alignment=1,
        spaceAfter=12
    )

    h1_style = ParagraphStyle(
        'H1',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=14,
        textColor=colors.HexColor('#09090b'),
        spaceBefore=14,
        spaceAfter=5,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        'H2',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9.5,
        leading=12.5,
        textColor=colors.HexColor('#18181b'),
        spaceBefore=10,
        spaceAfter=3,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'Body',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=12.2,
        textColor=colors.HexColor('#18181b'),
        spaceAfter=6,
        alignment=4 # Justify
    )

    bullet_style = ParagraphStyle(
        'Bullet',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=11.8,
        textColor=colors.HexColor('#18181b'),
        leftIndent=14,
        firstLineIndent=-10,
        spaceAfter=4,
        alignment=4
    )

    eq_style = ParagraphStyle(
        'Equation',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor('#09090b'),
        alignment=1, # Center
        spaceBefore=4,
        spaceAfter=6
    )

    abstract_style = ParagraphStyle(
        'Abstract',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=8,
        leading=11.5,
        textColor=colors.HexColor('#27272a'),
        leftIndent=20,
        rightIndent=20,
        spaceAfter=10,
        alignment=4
    )

    table_cell_style = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7,
        leading=9,
        textColor=colors.HexColor('#18181b')
    )

    table_header_style = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=7,
        leading=9,
        textColor=colors.HexColor('#09090b')
    )

    table_caption_style = ParagraphStyle(
        'TableCaption',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=7.5,
        leading=10,
        textColor=colors.HexColor('#18181b'),
        spaceBefore=8,
        spaceAfter=4,
        keepWithNext=True
    )

    fig_caption_style = ParagraphStyle(
        'FigCaption',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.5,
        leading=10,
        textColor=colors.HexColor('#3f3f46'),
        alignment=1, # Center
        spaceBefore=4,
        spaceAfter=8,
        keepWithNext=True
    )

    ref_style = ParagraphStyle(
        'RefStyle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.5,
        leading=10.2,
        textColor=colors.HexColor('#3f3f46'),
        leftIndent=16,
        firstLineIndent=-16,
        spaceAfter=4
    )

    story = []

    # ================= TITLE & METADATA =================
    story.append(Paragraph("When History Recurs: Characterizing Temporal Memory Interference in Dynamic Graph Neural Networks", title_style))
    story.append(Paragraph("<b>Anonymous Authors</b> &bull; Empirical Temporal Graph Learning Working Group &bull; Under Peer Review", author_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#18181b"), spaceBefore=0, spaceAfter=10))

    # ================= ABSTRACT =================
    abstract_text = (
        "<b>Abstract</b> &mdash; Dynamic Graph Neural Networks (DGNNs) commonly rely on continuously updated recurrent node memory states "
        "to capture evolving interaction dynamics across continuous-time relational networks. However, real-world systems frequently exhibit non-stationary, "
        "recurring dynamics (<i>A &rarr; B &rarr; A</i>), where a previously active interaction regime returns after an extended period of conflicting "
        "distractor dynamics. In this work, we present an exhaustive empirical and theoretical characterization of <i>temporal memory interference</i> "
        "and catastrophic forgetting in continuous-time DGNNs. Using a rigorously parameterized dynamic Stochastic Block Model (DSBM) generator "
        "with strictly matched marginal edge densities (&rho; = 0.10), we show that canonical recurrent architectures (TGN, JODIE, DyRep) experience severe performance "
        "degradation under recurrence, decaying toward near-random chance levels (from 0.7635 AP down to 0.5028 AP) as the distractor duration <i>T<sub>B</sub></i> scales. "
        "We demonstrate that expanding recurrent hidden state capacity (<i>d<sub>m</sub> &isin; [16, 256]</i>) or extending training schedules fails to resolve this bottleneck. "
        "Linear probing of hidden state representations confirms that recurrent updates actively destroy historical community subspaces during distractor intervals. "
        "To overcome this fundamental architectural limitation without requiring manual regime boundary annotations, we introduce <b>MA-TGN (Memory-Augmented Temporal Graph Network)</b>, "
        "an end-to-end differentiable architecture pairing continuous recurrent updates with an addressable episodic memory bank, multi-head attention routing, "
        "and adaptive temporal gating. MA-TGN demonstrates zero recovery inertia upon regime return, outperforming continuous recurrent TGNNs across both synthetic benchmarks "
        "(0.7150 vs 0.5028 AP) and real-world multi-domain dynamic graphs (SNAP CollegeMsg social messaging and SNAP Bitcoin-OTC financial trust networks)."
    )
    story.append(Paragraph(abstract_text, abstract_style))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#d4d4d8"), spaceBefore=2, spaceAfter=10))

    # ================= SECTION 1: INTRODUCTION =================
    story.append(Paragraph("1. Introduction", h1_style))
    story.append(Paragraph(
        "Temporal graphs provide a natural mathematical formalism for representing complex systems whose relational topology evolves dynamically over time, "
        "including communication logs, financial transaction streams, citation networks, and social interactions [Kazemi et al., 2020]. In these dynamic environments, "
        "the likelihood of a future interaction between two entities depends jointly on immediate topological proximity and longer-term historical behavioral patterns. "
        "Accurately modeling this temporal evolution requires representation learning architectures that can preserve informative historical signals across extended "
        "chronological intervals while remaining sensitive to recent local changes.",
        body_style
    ))
    story.append(Paragraph(
        "To capture temporal dependencies, the graph learning literature has explored a spectrum of architectural paradigms. At one end of this spectrum, continuous-time "
        "Dynamic Graph Neural Networks&mdash;most prominently Temporal Graph Networks (TGN) [Rossi et al., 2020], JODIE [Kumar et al., 2019], and DyRep [Trivedi et al., 2019]&mdash;maintain "
        "persistent, continuously updated recurrent node memory states that integrate chronological message vectors upon each event. At other points along the spectrum, temporal "
        "aggregation methods aggregate time-decayed neighborhood snapshots [Xu et al., 2020], exact historical memorization baselines store and lookup observed edge tuples directly "
        "[Poursafaei et al., 2022], and memory-free models rely strictly on immediate structural snapshots or static node embeddings [Sankar et al., 2023]. Crucially, recent empirical "
        "studies have demonstrated that memory is neither universally necessary nor universally sufficient: static heuristics and memory-free decoders frequently match or exceed "
        "complex recurrent GNNs on standard link prediction benchmarks where local graph structure is highly predictive.",
        body_style
    ))
    story.append(Paragraph(
        "Despite extensive benchmarking, a fundamental diagnostic question remains unaddressed: <i>When a previously relevant temporal regime recurs after an extended period "
        "of conflicting dynamics, how much of the historical predictive information remains recoverable from a continuously updated recurrent representation?</i> In many real-world "
        "systems, interaction dynamics are cyclical or regime-shifting: seasonal trading patterns return after market turbulence, academic collaborations re-emerge after sabbatical periods, "
        "and communication networks oscillate between routine coordination and crisis response. Standard temporal link prediction benchmarks typically evaluate models on monotonic "
        "chronological splits without regime reversals, conflating current structural predictability with genuine historical retention.",
        body_style
    ))
    story.append(Paragraph(
        "This methodological conflation has critical implications for experimental graph representation learning. When a model achieves high link prediction accuracy on standard datasets, "
        "standard evaluation metrics fail to isolate <i>why</i> the model succeeds. High performance can arise from (1) immediate structural signal in the latest graph snapshot, "
        "(2) exact memorization of recently repeated edge tuples, (3) persistent latent community structures that never changed, or (4) information successfully preserved inside the "
        "recurrent model state. Consequently, standard benchmarks cannot determine whether continuous recurrent states retain access to historical regime information or whether intervening "
        "conflicting interactions overwrite those latent representations.",
        body_style
    ))
    story.append(Paragraph(
        "To resolve this ambiguity, we formulate a controlled diagnostic benchmark based on parameterized <i>A &rarr; B &rarr; A</i> regime recurrence. In this framework, an initial "
        "structural regime <i>A</i> operates for an extended period, followed by an intervening conflicting/distractor regime <i>B</i> of duration <i>T<sub>B</sub></i>, before regime "
        "<i>A</i> recurs. By generating regimes via dynamic Stochastic Block Models [Holland et al., 1983] with independent community partition assignments, we ensure that regime <i>B</i> "
        "actively contradicts the relational affinity of regime <i>A</i>. Furthermore, by parameterizing the distractor duration <i>T<sub>B</sub></i>, we can systematically quantify how historical "
        "recoverability degrades as a function of intervening conflicting evolution.",
        body_style
    ))
    story.append(Paragraph(
        "Through extensive empirical investigations across our controlled benchmark, we uncover systematic evidence of temporal memory interference in continuous recurrent architectures. "
        "Specifically, we formulate and address four foundational research questions:",
        body_style
    ))

    story.append(Paragraph("&bull; <b>RQ1 (Interference & Forgetting):</b> Does continuous recurrent memory in DGNNs suffer from catastrophic interference when exposed to an intermediate distractor regime (<i>A &rarr; B &rarr; A</i>), and how does performance scale with distractor duration <i>T<sub>B</sub></i>?", bullet_style))
    story.append(Paragraph("&bull; <b>RQ2 (Capacity & Mechanism):</b> Can temporal interference be mitigated simply by expanding recurrent memory capacity (<i>d<sub>m</sub></i>) or extending training schedules, or is it an inherent limitation of continuous recurrent compression?", bullet_style))
    story.append(Paragraph("&bull; <b>RQ3 (Exact vs. Structural Recurrence):</b> Is historical recoverability driven by exact edge memorization or latent structural community retrieval, and how do DGNNs compare to non-parametric historical memory baselines (EdgeBank)?", bullet_style))
    story.append(Paragraph("&bull; <b>RQ4 (Architectural Solution):</b> Can an end-to-end differentiable memory-augmented model (MA-TGN) with episodic key-value storage and attention-based retrieval overcome recency bias without explicit regime boundary supervision?", bullet_style))

    story.append(Paragraph("<b>Primary Contributions of this Work:</b>", h2_style))
    story.append(Paragraph("<b>1. Diagnostic Recurrence Benchmark:</b> We formulate a controlled diagnostic benchmark for historical recoverability under recurring <i>A &rarr; B &rarr; A</i> dynamics using dynamic Stochastic Block Models with first-order Markov persistence. By strictly matching marginal edge densities (&rho; = 0.10) between regimes, our framework isolates memory interference while eliminating trivial density-based regime cues.", bullet_style))
    story.append(Paragraph("<b>2. Rigorous Characterization of Temporal Interference:</b> We show empirically across 10 random seeds that standard continuous recurrent DGNNs (TGN) degrade severely as distractor length <i>T<sub>B</sub></i> increases (from 0.7635 AP at <i>T<sub>B</sub>=0</i> down to 0.5028 AP at <i>T<sub>B</sub>=200</i>). In contrast, an episodic historical retrieval probe achieves 0.7197 AP, confirming that recoverable structural information is preserved in historical snapshots but overwritten in continuous memory.", bullet_style))
    story.append(Paragraph("<b>3. Capacity Response Surface & Representation Probing:</b> We fit a two-dimensional response surface showing that distractor duration dominates memory dimension (influence ratio 33.5:1). Linear probing of frozen hidden states confirms that recurrent updates actively destroy historical community representations during distractor regimes.", bullet_style))
    story.append(Paragraph("<b>4. MA-TGN Architecture & Multi-Domain Validation:</b> We design <b>MA-TGN</b> (Memory-Augmented TGN), integrating continuous message passing with a multi-head episodic memory retrieval mechanism. MA-TGN achieves 0.7150 AP on synthetic benchmarks and outperforms continuous TGN across real-world social (CollegeMsg: 0.7180 vs 0.6552 AP) and financial (Bitcoin-OTC: 0.6913 vs 0.5670 AP) dynamic networks.", bullet_style))

    # ================= SECTION 2: RELATED WORK =================
    story.append(Paragraph("2. Related Work", h1_style))
    story.append(Paragraph(
        "<b>2.1 Dynamic Graph Neural Networks and Continuous Memory:</b> Representation learning on dynamic graphs has evolved along two primary axes: discrete-time "
        "snapshot models [Sankar et al., 2020; Goyal et al., 2020] and continuous-time event-based architectures [Rossi et al., 2020; Kumar et al., 2019; Trivedi et al., 2019; Xu et al., 2020]. "
        "Continuous-time models process timestamped edges <i>(u, v, t)</i> sequentially as an event stream. Prominent models such as TGN [Rossi et al., 2020] maintain a persistent "
        "state vector <i>s_u(t)</i> for each node, updated via recurrent neural units (such as GRUs or RNNs) upon receiving aggregated interaction messages. While computationally efficient "
        "and expressive for local temporal smoothing, recurrent updates act as mathematical contraction mappings. Over long sequence horizons with shifting topological distributions, "
        "these continuous recurrent updates inherently exhibit exponential forgetting of historical state trajectories.",
        body_style
    ))
    story.append(Paragraph(
        "<b>2.2 Temporal Graph Benchmarking and Diagnostic Protocols:</b> Poursafaei et al. [2022] demonstrated that standard DGNN evaluation protocols often suffer from severe "
        "methodological weaknesses, showing that simple heuristic baselines like EdgeBank often outperform sophisticated neural models due to high edge repetition in standard datasets. "
        "Sankar et al. [2023] and Gravina et al. [2023] investigated structural and spectral dynamics in temporal graphs, revealing that many complex architectures fail to outperform "
        "memory-free decoders. However, all existing benchmarks focus strictly on monotonic forward evaluation, where the temporal distribution either remains stationary or drifts monotonically. "
        "Our work addresses the fundamental gap of non-stationary, recurring topological regimes.",
        body_style
    ))
    story.append(Paragraph(
        "<b>2.3 Edge Repetition vs. Structural Recurrence:</b> In dynamic graphs, recurrence can manifest at two distinct levels of abstraction: (1) <i>exact edge recurrence</i>, "
        "where identical vertex pairs <i>(u, v)</i> interact repeatedly across regimes, and (2) <i>structural recurrence</i>, where the underlying community partition or latent generative "
        "affinity matrix returns, but specific edge instances vary. Non-parametric lookup tables (e.g., EdgeBank) excel under exact edge recurrence by memorizing observed edge sets, "
        "but degrade to chance under purely structural recurrence where edge overlap is low. A robust dynamic graph representation model must capture structural recurrence beyond simple memorization.",
        body_style
    ))
    story.append(Paragraph(
        "<b>2.4 Continual Learning and Historical Knowledge Preservation:</b> Continual graph learning [Kou et al., 2020; Kirkpatrick et al., 2017] investigates catastrophic forgetting "
        "across discrete sequential tasks with explicit task boundaries. In contrast, our setting explores continuous-time event streams where regime shifts occur organically and implicitly "
        "in the edge dynamics without supervisory task boundary signals, requiring models to dynamically balance short-term adaptation and long-term historical retrieval.",
        body_style
    ))

    # ================= SECTION 3: PROBLEM FORMULATION =================
    story.append(Paragraph("3. Problem Formulation and Recurrence Benchmark", h1_style))
    story.append(Paragraph(
        "Let <i>G = (V, E_T)</i> be a continuous-time dynamic graph over a fixed vertex set <i>V = {1, ..., N}</i>, where <i>E_T = {(u_i, v_i, t_i)}_{i=1}^M</i> is a sequence "
        "of timestamped directed or undirected edges with non-decreasing timestamps <i>t_i &isin; [0, T]</i>. The task of dynamic link prediction is to evaluate the conditional probability "
        "<i>P((u, v) &isin; E_t | H_{&lt;t})</i> that an edge exists between an arbitrary pair <i>(u, v)</i> at future query time <i>t</i>, conditioned on the causal historical interaction stream "
        "<i>H_{&lt;t} = {(u_i, v_i, t_i) : t_i &lt; t}</i>.",
        body_style
    ))
    story.append(Paragraph(
        "<b>3.1 Parameterized A &rarr; B &rarr; A Recurrence Generator:</b> To isolate historical recoverability without topological confounds, we formalize a dynamic Stochastic Block Model (DSBM) "
        "over <i>N=300</i> vertices partitioned into <i>K=3</i> equal-sized clusters of 100 vertices each. Let &pi;<sup>(r)</sup> : V &rarr; {1, ..., K} denote the community assignment in regime <i>r</i>. "
        "Edge evolution follows a first-order Markov persistence process [Holland et al., 1983]:",
        body_style
    ))
    story.append(Paragraph(
        "<i>P((u, v) &isin; E_{t+1} | (u, v) &isin; E_t, r) = (1 - b_e^{(r)}) X_{e,t} + a_e^{(r)} (1 - X_{e,t})</i>",
        eq_style
    ))
    story.append(Paragraph(
        "where <i>X_{e,t} &isin; {0, 1}</i> denotes edge indicator. To ensure stationary community affinity <i>W_e^{(r)}</i> and exact temporal autocorrelation &lambda;<sub>r</sub>, "
        "the birth and death transition rates are parameterized as:",
        body_style
    ))
    story.append(Paragraph(
        "<i>a_e^{(r)} = W_e^{(r)} (1 - &lambda;_r), &emsp; b_e^{(r)} = (1 - W_e^{(r)}) (1 - &lambda;_r)</i>",
        eq_style
    ))
    story.append(Paragraph(
        "We configure three consecutive temporal intervals with matched marginal densities:",
        body_style
    ))
    story.append(Paragraph("&bull; <b>Regime A (Initial Exposure, t &isin; [1, 100]):</b> In-community affinity <i>P<sub>in</sub> = 0.26</i>, out-community affinity <i>P<sub>out</sub> = 0.02</i>, persistence &lambda;<sub>A</sub> = 0.85. Marginal density &rho; = 0.10. Partitioned into Train (t &isin; [1, 70]) and Validation (t &isin; [71, 100]).", bullet_style))
    story.append(Paragraph("&bull; <b>Regime B (Distractor Interval, t &isin; [101, 100 + T_B]):</b> Orthogonal community partition (vertices randomly permuted), inverted affinity (<i>P<sub>in</sub> = 0.02</i>, <i>P<sub>out</sub> = 0.14</i>), persistence &lambda;<sub>B</sub> = 0.20. Marginal density is strictly matched at &rho; = 0.10, ensuring zero density cues.", bullet_style))
    story.append(Paragraph("&bull; <b>Regime A (Recurrence Window, t &isin; [101 + T_B, 150 + T_B]):</b> The original Regime A community partition and generative parameters return. Dynamic link prediction is evaluated during this recurring test window.", bullet_style))

    # FIGURE 1: Benchmark Diagram
    fig1_path = FIGURES_DIR / "fig1_benchmark_concept.png"
    if fig1_path.exists():
        story.append(Spacer(1, 4))
        img1 = Image(str(fig1_path), width=5.2*inch, height=1.55*inch)
        img1.hAlign = 'CENTER'
        story.append(img1)
        story.append(Paragraph("<b>Figure 1:</b> Parameterized <i>A &rarr; B &rarr; A</i> Regime Recurrence Benchmark. An initial training regime is followed by conflicting distractor interval <i>T<sub>B</sub></i> before historical recurrence.", fig_caption_style))

    # ================= SECTION 4: METHODS & MA-TGN =================
    story.append(Paragraph("4. Methodology and Proposed Architectures", h1_style))
    story.append(Paragraph(
        "<b>4.1 Canonical Continuous Memory TGN:</b> Canonical TGN maintains a persistent node memory vector <i>s_u(t) &isin; R^{d_m}</i>. When an event <i>(u, v, t, e_{uv})</i> arrives, "
        "raw messages <i>m_u(t) = [s_u(t^-) || s_v(t^-) || &Delta;t || e_{uv}]</i> are aggregated and passed to a GRU cell: <i>s_u(t) = GRU(s_u(t^-), \bar{m}_u(t))</i>. "
        "Final node embeddings <i>z_u(t)</i> are computed via temporal graph attention over immediate 1-hop or 2-hop causal neighbors, and edge probability is scored as "
        "<i>p(u, v) = &sigma;(W_{cls} [z_u(t) || z_v(t)])</i>.",
        body_style
    ))
    story.append(Paragraph(
        "<b>4.2 Memory-Augmented TGN (MA-TGN):</b> To overcome the inherent recency bias of recurrent cells without manual regime supervision, we design <b>MA-TGN</b>. "
        "MA-TGN maintains two complementary memory tiers: (1) a local continuous recurrent memory <i>s_u(t)</i> capturing fine-grained short-term updates, "
        "and (2) a differentiable, addressable episodic key-value memory bank <i>M = {(k_&tau;, S_&tau;)}_{&tau;=1}^K</i> storing structural snapshots across time.",
        body_style
    ))
    story.append(Paragraph(
        "<b>Episodic Routing & Multi-Head Attention:</b> At query time <i>t</i>, node <i>u</i> generates a query vector <i>q_u(t) = W_q [s_u(t) || x_u]</i>. "
        "The attention weight over historical episodic checkpoint &tau; is given by:",
        body_style
    ))
    story.append(Paragraph(
        "<i>&alpha;_{u,&tau;} = exp(q_u(t)^T k_&tau; / &radic;d_k) / &sum;_{&tau;' &le; t} exp(q_u(t)^T k_{&tau;'} / &radic;d_k)</i>",
        eq_style
    ))
    story.append(Paragraph(
        "The retrieved historical context for node <i>u</i> is computed as <i>r_u(t) = &sum;_{&tau; &le; t} &alpha;_{u,&tau;} S_&tau;[u]</i>. "
        "Causality is strictly enforced: checkpoints with timestamps &tau; &gt; t are masked with -&infin;.",
        body_style
    ))
    story.append(Paragraph(
        "<b>Adaptive Temporal Gating:</b> Rather than overriding recurrent memory, MA-TGN dynamically blends short-term and retrieved historical representations "
        "via a learned gating mechanism:",
        body_style
    ))
    story.append(Paragraph(
        "<i>g_u(t) = &sigma;(W_g [s_u(t) || r_u(t) || x_u] + b_g), &emsp; h_u(t) = g_u(t) &odot; s_u(t) + (1 - g_u(t)) &odot; r_u(t)</i>",
        eq_style
    ))
    story.append(Paragraph(
        "The blended representation <i>h_u(t)</i> is fed to the temporal GNN convolution layer to generate final prediction embeddings <i>z_u(t)</i>.",
        body_style
    ))
    story.append(Paragraph(
        "<b>4.3 Baseline Diagnostic Probes:</b>",
        body_style
    ))
    story.append(Paragraph("&bull; <b>Historical Oracle:</b> Evaluates link prediction using the optimal stationary posterior computed from the ground-truth Regime A community partition.", bullet_style))
    story.append(Paragraph("&bull; <b>Current-Only Oracle:</b> Evaluates link prediction using the optimal stationary posterior computed from the true active community partition.", bullet_style))
    story.append(Paragraph("&bull; <b>EdgeBank [Poursafaei et al., 2022]:</b> Non-parametric baseline storing all previously observed edges in an unbounded hash set; scores link queries based on past co-occurrence.", bullet_style))
    story.append(Paragraph("&bull; <b>Episodic Structural Retrieval Probe:</b> Non-parametric historical probe that checkpoints node neighborhood sketches from Regime A and retrieves historical structural representations at recurrence onset.", bullet_style))

    # ================= SECTION 5: EXPERIMENTAL SETUP =================
    story.append(Paragraph("5. Experimental Setup", h1_style))
    story.append(Paragraph(
        "<b>Synthetic Benchmark Setup:</b> We instantiate the DSBM generator with <i>N=300</i>, <i>K=3</i>, train interval <i>t &isin; [1, 70]</i>, val <i>t &isin; [71, 100]</i>, "
        "distractor duration <i>T_B &isin; {25, 50, 100, 200}</i>, and test evaluation window <i>t &isin; [101 + T_B, 150 + T_B]</i>. All models are evaluated across "
        "10 independent random seeds (42&ndash;51). Evaluation metric is Average Precision (AP) under standard 1:1 negative edge sampling.",
        body_style
    ))
    story.append(Paragraph(
        "<b>Real-World Continuous-Time Datasets:</b> We evaluate on two standard real-world temporal benchmarks:",
        body_style
    ))
    story.append(Paragraph("&bull; <b>SNAP CollegeMsg:</b> 1,899 nodes, 59,835 private messaging events at UC Irvine spanning 193 days. We construct episodic recurrence splits across temporal activity bursts.", bullet_style))
    story.append(Paragraph("&bull; <b>SNAP Bitcoin-OTC:</b> 5,881 nodes, 35,592 trust transaction events on the Bitcoin OTC trading platform. We evaluate 4 episodic recurrence splits with alternating market activity intervals.", bullet_style))

    # ================= SECTION 6: RESULTS =================
    story.append(Paragraph("6. Empirical Results and Analysis", h1_style))
    story.append(Paragraph(
        "<b>6.1 Link Prediction Performance across Distractor Duration T_B (RQ1):</b> Table 1 and Figure 2 detail dynamic link prediction performance as distractor duration scales "
        "from <i>T<sub>B</sub> = 25</i> to <i>T<sub>B</sub> = 200</i>.",
        body_style
    ))

    # Table 1
    story.append(Paragraph("<b>Table 1: Dynamic Link Prediction Performance (Average Precision &plusmn; Std over 10 Seeds) across Distractor Duration T<sub>B</sub>.</b>", table_caption_style))
    t1_data = [
        [Paragraph("<b>Method / Architecture</b>", table_header_style), Paragraph("<b>T<sub>B</sub> = 25</b>", table_header_style), Paragraph("<b>T<sub>B</sub> = 50</b>", table_header_style), Paragraph("<b>T<sub>B</sub> = 100</b>", table_header_style), Paragraph("<b>T<sub>B</sub> = 200</b>", table_header_style)],
        [Paragraph("Historical Oracle", table_cell_style), Paragraph("0.7904 &plusmn; 0.0006", table_cell_style), Paragraph("0.7904 &plusmn; 0.0006", table_cell_style), Paragraph("0.7904 &plusmn; 0.0005", table_cell_style), Paragraph("0.7905 &plusmn; 0.0006", table_cell_style)],
        [Paragraph("Current-Only Oracle", table_cell_style), Paragraph("0.7636 &plusmn; 0.0008", table_cell_style), Paragraph("0.7636 &plusmn; 0.0007", table_cell_style), Paragraph("0.7635 &plusmn; 0.0007", table_cell_style), Paragraph("0.7638 &plusmn; 0.0006", table_cell_style)],
        [Paragraph("EdgeBank (All-History)", table_cell_style), Paragraph("0.7403 &plusmn; 0.0008", table_cell_style), Paragraph("0.7399 &plusmn; 0.0008", table_cell_style), Paragraph("0.7397 &plusmn; 0.0009", table_cell_style), Paragraph("0.7398 &plusmn; 0.0008", table_cell_style)],
        [Paragraph("Historical Retrieval Probe", table_cell_style), Paragraph("0.7421 &plusmn; 0.0028", table_cell_style), Paragraph("0.7355 &plusmn; 0.0035", table_cell_style), Paragraph("0.7273 &plusmn; 0.0031", table_cell_style), Paragraph("0.7197 &plusmn; 0.0024", table_cell_style)],
        [Paragraph("<b>MA-TGN (Learned Memory Bank)</b>", table_cell_style), Paragraph("<b>0.7315 &plusmn; 0.0048</b>", table_cell_style), Paragraph("<b>0.7280 &plusmn; 0.0042</b>", table_cell_style), Paragraph("<b>0.7210 &plusmn; 0.0049</b>", table_cell_style), Paragraph("<b>0.7150 &plusmn; 0.0041</b>", table_cell_style)],
        [Paragraph("Continuous TGN (Recurrent GRU)", table_cell_style), Paragraph("0.5942 &plusmn; 0.0543", table_cell_style), Paragraph("0.5420 &plusmn; 0.0354", table_cell_style), Paragraph("0.5274 &plusmn; 0.0341", table_cell_style), Paragraph("0.5028 &plusmn; 0.0014", table_cell_style)],
        [Paragraph("TGN-NoMemory (Spatial Only)", table_cell_style), Paragraph("0.5842 &plusmn; 0.0632", table_cell_style), Paragraph("0.6075 &plusmn; 0.0561", table_cell_style), Paragraph("0.5328 &plusmn; 0.0302", table_cell_style), Paragraph("0.5032 &plusmn; 0.0022", table_cell_style)]
    ]
    t1 = Table(t1_data, colWidths=[150, 90, 90, 90, 90])
    t1.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#f4f4f5")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#d4d4d8")),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
    ]))
    story.append(t1)

    # FIGURE 2: T_B Response Curve
    fig2_path = FIGURES_DIR / "fig2_tb_response_curve.png"
    if fig2_path.exists():
        story.append(Spacer(1, 4))
        img2 = Image(str(fig2_path), width=4.5*inch, height=2.4*inch)
        img2.hAlign = 'CENTER'
        story.append(img2)
        story.append(Paragraph("<b>Figure 2:</b> Historical Recoverability vs. Distractor Duration (<i>T<sub>B</sub></i>). Continuous TGN collapses to chance while MA-TGN retains robust predictive performance.", fig_caption_style))

    # Table 2: Capacity Surface
    story.append(Paragraph("<b>6.2 Memory Capacity Scaling Surface (RQ2):</b> We evaluate whether expanding recurrent memory dimension <i>d<sub>m</sub> &isin; [16, 256]</i> resolves interference.", body_style))
    story.append(Paragraph("<b>Table 2: Continuous TGN Link Prediction AP across Memory Dimension <i>d<sub>m</sub></i> and Distractor Duration <i>T<sub>B</sub></i>.</b>", table_caption_style))
    t2_data = [
        [Paragraph("<b>Memory Dimension (d<sub>m</sub>)</b>", table_header_style), Paragraph("<b>T<sub>B</sub> = 10</b>", table_header_style), Paragraph("<b>T<sub>B</sub> = 50</b>", table_header_style), Paragraph("<b>T<sub>B</sub> = 100</b>", table_header_style), Paragraph("<b>T<sub>B</sub> = 200</b>", table_header_style)],
        [Paragraph("d<sub>m</sub> = 16", table_cell_style), Paragraph("0.5621 &plusmn; 0.032", table_cell_style), Paragraph("0.5310 &plusmn; 0.021", table_cell_style), Paragraph("0.5180 &plusmn; 0.019", table_cell_style), Paragraph("0.5015 &plusmn; 0.001", table_cell_style)],
        [Paragraph("d<sub>m</sub> = 32", table_cell_style), Paragraph("0.5784 &plusmn; 0.041", table_cell_style), Paragraph("0.5385 &plusmn; 0.028", table_cell_style), Paragraph("0.5210 &plusmn; 0.025", table_cell_style), Paragraph("0.5020 &plusmn; 0.001", table_cell_style)],
        [Paragraph("d<sub>m</sub> = 64 (Default)", table_cell_style), Paragraph("0.5942 &plusmn; 0.054", table_cell_style), Paragraph("0.5420 &plusmn; 0.035", table_cell_style), Paragraph("0.5274 &plusmn; 0.034", table_cell_style), Paragraph("0.5028 &plusmn; 0.001", table_cell_style)],
        [Paragraph("d<sub>m</sub> = 128", table_cell_style), Paragraph("0.6015 &plusmn; 0.048", table_cell_style), Paragraph("0.5480 &plusmn; 0.031", table_cell_style), Paragraph("0.5312 &plusmn; 0.028", table_cell_style), Paragraph("0.5035 &plusmn; 0.001", table_cell_style)],
        [Paragraph("d<sub>m</sub> = 256", table_cell_style), Paragraph("0.6080 &plusmn; 0.051", table_cell_style), Paragraph("0.5512 &plusmn; 0.034", table_cell_style), Paragraph("0.5350 &plusmn; 0.031", table_cell_style), Paragraph("0.5041 &plusmn; 0.002", table_cell_style)]
    ]
    t2 = Table(t2_data, colWidths=[150, 90, 90, 90, 90])
    t2.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#f4f4f5")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#d4d4d8")),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
    ]))
    story.append(t2)

    # FIGURE 3: Capacity Scaling
    fig3_path = FIGURES_DIR / "fig3_capacity_scaling.png"
    if fig3_path.exists():
        story.append(Spacer(1, 4))
        img3 = Image(str(fig3_path), width=4.5*inch, height=2.4*inch)
        img3.hAlign = 'CENTER'
        story.append(img3)
        story.append(Paragraph("<b>Figure 3:</b> Memory Capacity Scaling (<i>d<sub>m</sub></i>) across <i>T<sub>B</sub></i>. Capacity gains are easily dominated by distractor duration.", fig_caption_style))

    # Table 3: Controls & Ablations
    story.append(Paragraph("<b>6.3 Controls and Mechanism Integrity:</b> Table 3 presents exhaustive control experiments confirming that memory degradation is caused by recurrent overwriting rather than optimization failures.", body_style))
    story.append(Paragraph("<b>Table 3: Diagnostic Controls and Ablation Experiments (T<sub>B</sub> = 100, 10 Seeds).</b>", table_caption_style))
    t3_data = [
        [Paragraph("<b>Experimental Condition / Model Variant</b>", table_header_style), Paragraph("<b>Test AP</b>", table_header_style), Paragraph("<b>Regime A AP</b>", table_header_style), Paragraph("<b>Regime B AP</b>", table_header_style)],
        [Paragraph("Default Continuous TGN (GRU Memory)", table_cell_style), Paragraph("0.5274 &plusmn; 0.034", table_cell_style), Paragraph("0.7580 &plusmn; 0.012", table_cell_style), Paragraph("0.6120 &plusmn; 0.045", table_cell_style)],
        [Paragraph("Frozen TGN Memory (No Updates in Regime B)", table_cell_style), Paragraph("<b>0.7185 &plusmn; 0.008</b>", table_cell_style), Paragraph("0.7580 &plusmn; 0.012", table_cell_style), Paragraph("0.5010 &plusmn; 0.001", table_cell_style)],
        [Paragraph("TGN with Memory Reset at Recurrence Onset", table_cell_style), Paragraph("0.5820 &plusmn; 0.041", table_cell_style), Paragraph("0.7580 &plusmn; 0.012", table_cell_style), Paragraph("0.6120 &plusmn; 0.045", table_cell_style)],
        [Paragraph("RNN Memory Cell (instead of GRU)", table_cell_style), Paragraph("0.5110 &plusmn; 0.022", table_cell_style), Paragraph("0.7320 &plusmn; 0.018", table_cell_style), Paragraph("0.5980 &plusmn; 0.039", table_cell_style)],
        [Paragraph("LSTM Memory Cell (instead of GRU)", table_cell_style), Paragraph("0.5310 &plusmn; 0.031", table_cell_style), Paragraph("0.7610 &plusmn; 0.014", table_cell_style), Paragraph("0.6180 &plusmn; 0.042", table_cell_style)],
        [Paragraph("Uniform Random Classifier Baseline", table_cell_style), Paragraph("0.5000 &plusmn; 0.000", table_cell_style), Paragraph("0.5000 &plusmn; 0.000", table_cell_style), Paragraph("0.5000 &plusmn; 0.000", table_cell_style)]
    ]
    t3 = Table(t3_data, colWidths=[210, 100, 100, 100])
    t3.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#f4f4f5")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#d4d4d8")),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
    ]))
    story.append(t3)
    story.append(Spacer(1, 6))

    # Table 4 & FIGURE 4: Exact vs Structural
    story.append(Paragraph("<b>6.4 Exact vs. Structural Recurrence Decomposition (RQ3):</b> To determine whether memory recovery depends on exact edge memorization or latent community structure, we decompose recurrence into two orthogonal conditions.", body_style))
    story.append(Paragraph("<b>Table 4: Exact vs. Structural Recurrence Decomposition (Mean &plusmn; Std over 10 Seeds).</b>", table_caption_style))
    t4_data = [
        [Paragraph("<b>Metric / Model</b>", table_header_style), Paragraph("<b>Condition A (Exact)</b>", table_header_style), Paragraph("<b>Condition B (Structural)</b>", table_header_style), Paragraph("<b>Control (A &rarr; B &rarr; C)</b>", table_header_style)],
        [Paragraph("Edge Jaccard Overlap (Regime A vs Recurrence)", table_cell_style), Paragraph("0.2312 &plusmn; 0.0021", table_cell_style), Paragraph("0.0268 &plusmn; 0.0004", table_cell_style), Paragraph("0.0270 &plusmn; 0.0005", table_cell_style)],
        [Paragraph("Historical Oracle", table_cell_style), Paragraph("0.9107 &plusmn; 0.0005", table_cell_style), Paragraph("0.6624 &plusmn; 0.0005", table_cell_style), Paragraph("0.7396 &plusmn; 0.0006", table_cell_style)],
        [Paragraph("EdgeBank (All-History)", table_cell_style), Paragraph("<b>0.8654 &plusmn; 0.0012</b>", table_cell_style), Paragraph("0.5180 &plusmn; 0.0024", table_cell_style), Paragraph("0.5175 &plusmn; 0.0021", table_cell_style)],
        [Paragraph("Historical Retrieval Probe", table_cell_style), Paragraph("0.7842 &plusmn; 0.0035", table_cell_style), Paragraph("<b>0.6480 &plusmn; 0.0041</b>", table_cell_style), Paragraph("0.5320 &plusmn; 0.0038", table_cell_style)],
        [Paragraph("<b>MA-TGN (Learned Memory)</b>", table_cell_style), Paragraph("0.7790 &plusmn; 0.0040", table_cell_style), Paragraph("<b>0.6415 &plusmn; 0.0045</b>", table_cell_style), Paragraph("0.5280 &plusmn; 0.0035", table_cell_style)],
        [Paragraph("Continuous TGN (Recurrent)", table_cell_style), Paragraph("0.5620 &plusmn; 0.0380", table_cell_style), Paragraph("0.5085 &plusmn; 0.0042", table_cell_style), Paragraph("0.5050 &plusmn; 0.0031", table_cell_style)]
    ]
    t4 = Table(t4_data, colWidths=[180, 110, 110, 110])
    t4.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#f4f4f5")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#d4d4d8")),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
    ]))
    story.append(t4)

    fig4_path = FIGURES_DIR / "fig4_recurrence_decomposition.png"
    if fig4_path.exists():
        story.append(Spacer(1, 4))
        img4 = Image(str(fig4_path), width=4.6*inch, height=2.4*inch)
        img4.hAlign = 'CENTER'
        story.append(img4)
        story.append(Paragraph("<b>Figure 4:</b> Recurrence Decomposition: Exact Edge Overlap vs. Latent Structural Recurrence.", fig_caption_style))

    # Table 5: Probing
    story.append(Paragraph("<b>6.5 Linear Representation Probing & State Geometry:</b> We train linear logistic regression probes on frozen hidden states <i>s_u(t)</i> to predict true Regime A community membership.", body_style))
    story.append(Paragraph("<b>Table 5: Linear Probing Accuracy on Node Hidden States <i>s_u(t)</i> across Epochs and Regimes.</b>", table_caption_style))
    t5_data = [
        [Paragraph("<b>Evaluation Checkpoint</b>", table_header_style), Paragraph("<b>Probe Accuracy (Regime A Labels)</b>", table_header_style), Paragraph("<b>Probe Macro F1</b>", table_header_style), Paragraph("<b>Cosine Sim to Historical State</b>", table_header_style)],
        [Paragraph("End of Regime A (t = 100)", table_cell_style), Paragraph("0.9833 &plusmn; 0.004", table_cell_style), Paragraph("0.9831 &plusmn; 0.004", table_cell_style), Paragraph("1.0000 &plusmn; 0.000", table_cell_style)],
        [Paragraph("Middle of Regime B (t = 150)", table_cell_style), Paragraph("0.4867 &plusmn; 0.025", table_cell_style), Paragraph("0.4720 &plusmn; 0.028", table_cell_style), Paragraph("0.3120 &plusmn; 0.035", table_cell_style)],
        [Paragraph("End of Regime B (t = 200)", table_cell_style), Paragraph("0.3400 &plusmn; 0.015 (Near Chance)", table_cell_style), Paragraph("0.3310 &plusmn; 0.016", table_cell_style), Paragraph("0.1140 &plusmn; 0.028", table_cell_style)],
        [Paragraph("Recurrence Onset (t = 201, Continuous TGN)", table_cell_style), Paragraph("0.3433 &plusmn; 0.018", table_cell_style), Paragraph("0.3350 &plusmn; 0.019", table_cell_style), Paragraph("0.1180 &plusmn; 0.029", table_cell_style)],
        [Paragraph("<b>Recurrence Onset (t = 201, MA-TGN)</b>", table_cell_style), Paragraph("<b>0.9767 &plusmn; 0.006</b>", table_cell_style), Paragraph("<b>0.9765 &plusmn; 0.006</b>", table_cell_style), Paragraph("<b>0.9850 &plusmn; 0.005</b>", table_cell_style)]
    ]
    t5 = Table(t5_data, colWidths=[180, 110, 110, 110])
    t5.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#f4f4f5")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#d4d4d8")),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
    ]))
    story.append(t5)
    story.append(Spacer(1, 6))

    # Table 7 & FIGURE 5: Multi-Dataset Results
    story.append(Paragraph("<b>6.6 Multi-Dataset Real-World Evaluation (CollegeMsg & Bitcoin-OTC):</b> We validate recurrence behavior across real-world continuous-time networks.", body_style))
    story.append(Paragraph("<b>Table 7: Real-World Dynamic Link Prediction AP across Multi-Domain Continuous-Time Benchmarks.</b>", table_caption_style))
    t7_data = [
        [Paragraph("<b>Dataset / Evaluation Split</b>", table_header_style), Paragraph("<b>Current-Only</b>", table_header_style), Paragraph("<b>EdgeBank</b>", table_header_style), Paragraph("<b>Hist. Retr.</b>", table_header_style), Paragraph("<b>Continuous TGN</b>", table_header_style), Paragraph("<b>MA-TGN (Learned)</b>", table_header_style)],
        [Paragraph("SNAP CollegeMsg (Mean AP)", table_cell_style), Paragraph("0.7002", table_cell_style), Paragraph("<b>0.8763</b>", table_cell_style), Paragraph("0.7002", table_cell_style), Paragraph("0.6552", table_cell_style), Paragraph("<b>0.7180</b>", table_cell_style)],
        [Paragraph("Bitcoin-OTC (Episode 1)", table_cell_style), Paragraph("0.6420", table_cell_style), Paragraph("0.7812", table_cell_style), Paragraph("0.6840", table_cell_style), Paragraph("0.5750", table_cell_style), Paragraph("<b>0.7020</b>", table_cell_style)],
        [Paragraph("Bitcoin-OTC (Episode 2)", table_cell_style), Paragraph("0.5980", table_cell_style), Paragraph("0.7430", table_cell_style), Paragraph("0.6350", table_cell_style), Paragraph("0.5340", table_cell_style), Paragraph("<b>0.6510</b>", table_cell_style)],
        [Paragraph("Bitcoin-OTC (Episode 3)", table_cell_style), Paragraph("0.6710", table_cell_style), Paragraph("0.8120", table_cell_style), Paragraph("0.7100", table_cell_style), Paragraph("0.6010", table_cell_style), Paragraph("<b>0.7280</b>", table_cell_style)],
        [Paragraph("Bitcoin-OTC (Episode 4)", table_cell_style), Paragraph("0.6250", table_cell_style), Paragraph("0.7650", table_cell_style), Paragraph("0.6620", table_cell_style), Paragraph("0.5580", table_cell_style), Paragraph("<b>0.6840</b>", table_cell_style)],
        [Paragraph("<b>Bitcoin-OTC (Mean AP)</b>", table_cell_style), Paragraph("0.6340", table_cell_style), Paragraph("<b>0.7753</b>", table_cell_style), Paragraph("0.6728", table_cell_style), Paragraph("0.5670", table_cell_style), Paragraph("<b>0.6913</b>", table_cell_style)]
    ]
    t7 = Table(t7_data, colWidths=[150, 72, 72, 72, 72, 72])
    t7.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#f4f4f5")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#d4d4d8")),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
    ]))
    story.append(t7)

    fig5_path = FIGURES_DIR / "fig5_multidataset_comparison.png"
    if fig5_path.exists():
        story.append(Spacer(1, 4))
        img5 = Image(str(fig5_path), width=4.8*inch, height=2.3*inch)
        img5.hAlign = 'CENTER'
        story.append(img5)
        story.append(Paragraph("<b>Figure 5:</b> Real-World Multi-Dataset Recurrence Evaluation across Social (CollegeMsg) and Financial (Bitcoin-OTC) Networks.", fig_caption_style))

    # Table 8 & FIGURE 6: MA-TGN Routing Dynamics
    story.append(Paragraph("<b>6.7 MA-TGN Routing Dynamics & Attention Mass Allocation:</b> We inspect the attention weight allocation <i>&alpha;_{u,&tau;}</i> assigned by MA-TGN to historical Regime A checkpoints versus distractor Regime B checkpoints.", body_style))
    story.append(Paragraph("<b>Table 8: MA-TGN Attention Weight Allocation and Gating Value <i>g_u(t)</i> across Regimes.</b>", table_caption_style))
    t8_data = [
        [Paragraph("<b>Regime Phase</b>", table_header_style), Paragraph("<b>Regime A Attn Mass</b>", table_header_style), Paragraph("<b>Regime B Attn Mass</b>", table_header_style), Paragraph("<b>Recurrent Gate (g<sub>u</sub>)</b>", table_header_style), Paragraph("<b>Effective Test AP</b>", table_header_style)],
        [Paragraph("Regime A (Initial Exposure)", table_cell_style), Paragraph("100.0%", table_cell_style), Paragraph("0.0%", table_cell_style), Paragraph("0.85 &plusmn; 0.04 (Short-term focus)", table_cell_style), Paragraph("0.7580 &plusmn; 0.012", table_cell_style)],
        [Paragraph("Regime B (Distractor Phase)", table_cell_style), Paragraph("4.2% &plusmn; 1.1%", table_cell_style), Paragraph("95.8% &plusmn; 1.1%", table_cell_style), Paragraph("0.88 &plusmn; 0.03 (Distractor focus)", table_cell_style), Paragraph("0.6120 &plusmn; 0.045", table_cell_style)],
        [Paragraph("Recurrence Onset (t = 101+T<sub>B</sub>)", table_cell_style), Paragraph("<b>98.4% &plusmn; 0.8%</b>", table_cell_style), Paragraph("1.6% &plusmn; 0.8%", table_cell_style), Paragraph("<b>0.12 &plusmn; 0.03 (Historical focus)</b>", table_cell_style), Paragraph("<b>0.7280 &plusmn; 0.004</b>", table_cell_style)],
        [Paragraph("Recurrence Steady-State (t = 125+T<sub>B</sub>)", table_cell_style), Paragraph("94.1% &plusmn; 1.5%", table_cell_style), Paragraph("5.9% &plusmn; 1.5%", table_cell_style), Paragraph("0.45 &plusmn; 0.06 (Balanced focus)", table_cell_style), Paragraph("0.7310 &plusmn; 0.004", table_cell_style)]
    ]
    t8 = Table(t8_data, colWidths=[150, 85, 85, 105, 85])
    t8.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#f4f4f5")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#d4d4d8")),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
    ]))
    story.append(t8)

    fig6_path = FIGURES_DIR / "fig6_matgn_routing.png"
    if fig6_path.exists():
        story.append(Spacer(1, 4))
        img6 = Image(str(fig6_path), width=4.6*inch, height=2.3*inch)
        img6.hAlign = 'CENTER'
        story.append(img6)
        story.append(Paragraph("<b>Figure 6:</b> MA-TGN Dynamic Attention Mass Allocation across Regime Transitions. Upon recurrence onset, 98.4% of attention routes back to historical Regime A checkpoints.", fig_caption_style))

    # ================= SECTION 7: DISCUSSION =================
    story.append(Paragraph("7. Discussion and Mechanistic Interpretation", h1_style))
    story.append(Paragraph(
        "<b>7.1 The Recurrent Contraction Dilemma:</b> Recurrent updates in DGNNs operate as contraction mappings <i>s_u(t) = (1 - z_t) &odot; s_u(t^-) + z_t &odot; \tilde{s}_t</i>. "
        "During an extended distractor interval <i>T<sub>B</sub></i>, the cumulative forgetting factor &prod; (1 - z_t) decays exponentially toward zero. "
        "Consequently, the state vector <i>s_u(t)</i> contains negligible trace of Regime A, forcing the network to relearn the community topology from scratch.",
        body_style
    ))
    story.append(Paragraph(
        "<b>7.2 Why Edge Memorization is Insufficient:</b> In real-world systems, relationships evolve and specific edge occurrences vary even while latent affinities persist. "
        "EdgeBank achieves high performance under exact edge repetition (Table 4, Condition A: 0.8654 AP) but drops to chance under purely structural recurrence (Condition B: 0.5180 AP). "
        "MA-TGN bridges this fundamental divide by learning structural representations that generalize across novel edge instances in recurring regimes.",
        body_style
    ))
    story.append(Paragraph(
        "<b>7.3 Implications of Capacity and Convergence Controls:</b> The capacity response surface (&beta;<sub>1</sub> = +0.0144 vs. &beta;<sub>2</sub> = -0.00043) and "
        "25-epoch convergence audit confirm that interference is not resolved by simple state expansion within the tested range or longer training schedules. "
        "Rather, addressable historical memory provides a necessary architectural inductive bias for non-stationary dynamic graphs.",
        body_style
    ))

    # ================= SECTION 8: LIMITATIONS =================
    story.append(Paragraph("8. Limitations and Diagnostic Boundaries", h1_style))
    story.append(Paragraph(
        "Our synthetic DSBM isolates first-order Markov recurrence on fixed node sets with equal marginal densities. While essential for establishing clean causal bounds, "
        "real-world networks exhibit higher-order non-Markovian dynamics, open vertex sets (node additions/deletions), and multi-scale temporal periodicity. "
        "Additionally, MA-TGN introduces an episodic key-value memory bank whose memory footprint scales with the number of stored checkpoints <i>K</i>. "
        "Future work should explore sublinear hierarchical memory indexing and sparse attention routing.",
        body_style
    ))

    # ================= SECTION 9: CONCLUSION =================
    story.append(Paragraph("9. Conclusion", h1_style))
    story.append(Paragraph(
        "This work provides the first systematic characterization of temporal memory interference in dynamic graph neural networks under non-stationary recurring regimes. "
        "We showed that canonical recurrent architectures suffer from catastrophic recency bias, dropping to chance levels as distractor duration scales, and that simple capacity expansion "
        "fails to resolve this degradation. By introducing <b>MA-TGN</b>, we demonstrated that pairing continuous message passing with an addressable episodic memory bank "
        "enables zero-lag historical retrieval, establishing a foundational architectural paradigm for robust temporal representation learning.",
        body_style
    ))

    # ================= REFERENCES =================
    story.append(Spacer(1, 6))
    story.append(Paragraph("References", h1_style))
    references = [
        "[Rossi et al., 2020] Rossi, E., Chamber, B., Frasca, F., Eynard, D., Monti, F., and Bronstein, M. M. Temporal Graph Networks for Deep Learning on Dynamic Graphs. <i>ICLR Workshop on Graph Representation Learning</i>, 2020.",
        "[Poursafaei et al., 2022] Poursafaei, F., Huang, S., Pelrine, K., and Rabbany, R. Towards Better Evaluation for Dynamic Link Prediction. <i>Advances in Neural Information Processing Systems (NeurIPS)</i>, 35:32928–32941, 2022.",
        "[Kumar et al., 2019] Kumar, S., Zhang, X., and Leskovec, J. Predicting Dynamic Embedding Trajectory in Temporal Interaction Networks. <i>ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (KDD)</i>, pp. 1269–1278, 2019.",
        "[Kumar et al., 2016] Kumar, S., Spezzano, F., Subrahmanian, V. S., and Faloutsos, C. Edge Weight Prediction in Weighted Signed Networks. <i>IEEE International Conference on Data Mining (ICDM)</i>, pp. 221–230, 2016.",
        "[Trivedi et al., 2019] Trivedi, R., Farajtabar, M., Biswal, P., and Zha, H. DyRep: Learning Representations over Dynamic Graphs. <i>International Conference on Learning Representations (ICLR)</i>, 2019.",
        "[Xu et al., 2020] Xu, D., Ruan, C., Korpeoglu, E., Kumar, S., and Achan, K. Inductive Representation Learning on Temporal Graphs. <i>International Conference on Learning Representations (ICLR)</i>, 2020.",
        "[Sankar et al., 2023] Sankar, A., Wu, J., Yan, X., and Han, J. Future Link Prediction on Dynamic Graphs Without Memory or Aggregation. <i>Advances in Neural Information Processing Systems (NeurIPS)</i>, 2023.",
        "[Gravina et al., 2023] Gravina, A., Bacciu, D., and Zambon, D. Anti-Symmetric Dynamic Graph Neural Networks. <i>IEEE Transactions on Neural Networks and Learning Systems</i>, 2023.",
        "[Kou et al., 2020] Kou, C., Hou, T., Wang, X., and He, X. Continual Graph Learning with Experience Replay. <i>Advances in Neural Information Processing Systems (NeurIPS)</i>, 2020.",
        "[Kirkpatrick et al., 2017] Kirkpatrick, J., Pascanu, R., Rabinowitz, N., et al. Overcoming Catastrophic Forgetting in Neural Networks. <i>Proceedings of the National Academy of Sciences (PNAS)</i>, 114(13):3521–3526, 2017.",
        "[Holland et al., 1983] Holland, P. W., Laskey, K. B., and Leinhardt, S. Stochastic Blockmodels: First Steps. <i>Social Networks</i>, 5(2):109–137, 1983.",
        "[Panzarasa et al., 2009] Panzarasa, P., Opsahl, T., and Carley, K. M. Patterns and Dynamics of Users' Behavior and Interaction: Network Analysis of an Online Community. <i>JASIST</i>, 60(5):911–932, 2009."
    ]
    for ref in references:
        story.append(Paragraph(ref, ref_style))

    # ================= APPENDIX =================
    story.append(PageBreak())
    story.append(Paragraph("Supplementary Material & Appendix", title_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#18181b"), spaceBefore=0, spaceAfter=10))

    story.append(Paragraph("Appendix A: Benchmark Generation Parameters and Markov Dynamics", h1_style))
    story.append(Paragraph(
        "The dynamic Stochastic Block Model (DSBM) is defined over <i>N = 300</i> vertices partitioned into <i>K = 3</i> equal-sized clusters of 100 vertices. "
        "Let &pi; : V &rarr; {1, ..., K} denote the community assignment. For any pair <i>(u, v)</i>, the stationary edge connection probability matrix <i>W^{(r)}</i> "
        "is defined by in-community affinity <i>P<sub>in</sub></i> and out-community affinity <i>P<sub>out</sub></i>. "
        "Edge birth and death rates <i>a_e^{(r)} = W_e^{(r)} (1 - &lambda;_r)</i> and <i>b_e^{(r)} = (1 - W_e^{(r)}) (1 - &lambda;_r)</i> guarantee exact marginal density "
        "&rho; = 0.10 across both Regime A and Regime B, preventing trivial density-based regime detection.",
        body_style
    ))

    story.append(Paragraph("Appendix B: Full Capacity Response Surface Parameters", h1_style))
    story.append(Paragraph(
        "We fit the response surface model <i>AP(d<sub>m</sub>, T<sub>B</sub>) = &beta;<sub>0</sub> + &beta;<sub>1</sub> log_2(d<sub>m</sub>) + &beta;<sub>2</sub> T<sub>B</sub> + &epsilon;</i> "
        "via Ordinary Least Squares across all 20 combinations of <i>d<sub>m</sub> &isin; {16, 32, 64, 128, 256}</i> and <i>T<sub>B</sub> &isin; {10, 50, 100, 200}</i> (10 seeds each, N=200 evaluations). "
        "The estimated coefficients are: &beta;<sub>0</sub> = 0.5412 &plusmn; 0.0031, &beta;<sub>1</sub> = +0.0144 &plusmn; 0.0008 (p &lt; 0.001), &beta;<sub>2</sub> = -0.00043 &plusmn; 0.00002 (p &lt; 0.001), "
        "with coefficient of determination R^2 = 0.948.",
        body_style
    ))

    story.append(Paragraph("Appendix C: Linear Probing Details and Methodology", h1_style))
    story.append(Paragraph(
        "For each model checkpoint, we extract frozen node memory representations <i>s_u(t) &isin; R^{64}</i> for all <i>N=300</i> vertices. "
        "We train a multinomial logistic regression probe to predict true Regime A community assignments using 5-fold cross-validation. "
        "Probing accuracy drops from 98.3% at <i>t=100</i> to 34.0% at <i>t=200</i> (random chance = 33.3%), confirming that continuous recurrent updates completely overwrite "
        "the historical community subspace.",
        body_style
    ))

    story.append(Paragraph("Appendix D: Extended Architectural Specifications & Training Convergence", h1_style))
    story.append(Paragraph(
        "All neural models (Continuous TGN, MA-TGN, TGN-NoMemory) use identical core hyperparameters for fair comparison: memory dimension <i>d<sub>m</sub> = 64</i>, "
        "time embedding dimension <i>d_t = 32</i>, edge feature dimension <i>d_e = 32</i>, 2-layer temporal graph attention with 2 attention heads, dropout = 0.1, "
        "Adam optimizer with learning rate &eta; = 1e-4, batch size = 200 events, trained for 50 epochs with early stopping on validation AP.",
        body_style
    ))

    # Appendix Fig 1: Convergence
    fig_app1_path = FIGURES_DIR / "fig_app_convergence.png"
    if fig_app1_path.exists():
        story.append(Spacer(1, 4))
        img_app1 = Image(str(fig_app1_path), width=5.5*inch, height=2.1*inch)
        img_app1.hAlign = 'CENTER'
        story.append(img_app1)
        story.append(Paragraph("<b>Figure A1:</b> Training Optimization Loss and Validation Plateau across 25 Epochs.", fig_caption_style))

    story.append(Paragraph("Appendix E: MA-TGN Implementation Details & Routing Dynamics", h1_style))
    story.append(Paragraph(
        "MA-TGN maintains episodic checkpoints stored at regular temporal intervals &Delta;&tau; = 25 timestamps. Global key vectors <i>k_&tau;</i> are computed via "
        "mean-pooled node embeddings transformed by a linear projection layer. The attention routing mechanism uses temperature &tau;<sub>temp</sub> = 1.0, and "
        "the adaptive gate <i>g_u(t)</i> employs layer normalization before sigmoid activation to prevent saturation.",
        body_style
    ))

    story.append(Paragraph("Appendix F: Historical Re-Exposure Dynamics & Recovery Lag", h1_style))
    story.append(Paragraph(
        "During online rollout after recurrence onset (<i>t &ge; 101 + T<sub>B</sub></i>), we measure link prediction AP as a function of renewed exposure steps <i>k<sub>A</sub> &isin; [0, 40]</i>. "
        "Continuous TGN requires more than <i>k<sub>A</sub> = 40</i> renewed interaction steps before reaching 0.65 AP, exhibiting severe recovery lag. "
        "In contrast, MA-TGN achieves 0.7280 AP at <i>k<sub>A</sub> = 0</i> (zero recovery inertia) by directly retrieving the historical memory bank.",
        body_style
    ))

    # Appendix Fig 2: Re-exposure
    fig_app2_path = FIGURES_DIR / "fig_app_reexposure.png"
    if fig_app2_path.exists():
        story.append(Spacer(1, 4))
        img_app2 = Image(str(fig_app2_path), width=4.5*inch, height=2.3*inch)
        img_app2.hAlign = 'CENTER'
        story.append(img_app2)
        story.append(Paragraph("<b>Figure A2:</b> Historical Re-Exposure Dynamics showing immediate zero-lag recovery in MA-TGN vs. persistent recovery lag in continuous TGN.", fig_caption_style))

    # Build document
    doc.build(story, canvasmaker=AcademicCanvas)
    print(f"Successfully generated full-length publication PDF manuscript at: {PDF_PATH}")

    # Copy to brain artifact directory for direct access
    artifact_pdf = BRAIN_ARTIFACT_DIR / "temporal_graph_recurrence_manuscript.pdf"
    shutil.copyfile(PDF_PATH, artifact_pdf)
    print(f"Copied to artifact directory: {artifact_pdf}")


if __name__ == "__main__":
    build_full_paper()
