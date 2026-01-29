# TO-MHT Reference

This document collects references, links, and distilled notes about TO-MHT and related multi-hypothesis tracking methods. It’s a complement to the roadmap and architecture docs, focusing on theory and external publications.

Local PDFs are kept under:

- `papers/Classic & conceptual/`
- `papers/Implementation/`
- `papers/Survey/`

---

## 1. Classic & conceptual references

### 1.1 Reid – Original MHT (1979)

**Local file:**  
- `papers/Classic & conceptual/Reid - MHT (1979).pdf`

**Citation:**  
D. B. Reid, “An Algorithm for Tracking Multiple Targets,” *IEEE Transactions on Automatic Control*, vol. AC-24, no. 6, pp. 843–854, 1979. :contentReference[oaicite:0]{index=0}  

**Online:**  
- PDF (Stanford graphics / various mirrors):  
  https://graphics.stanford.edu/courses/cs428-03-spring/Papers/readings/CollaborativeProcessing/Reid_MHT_ieee_trans_ac_1979.pdf :contentReference[oaicite:1]{index=1}  

**Notes / what it’s good for:**

- Original measurement-oriented MHT formulation.
- Hypothesis relationship matrix, track probabilities, and early ideas on pruning and merging.
- Good for understanding the “pure” conceptual MHT (before TO-MHT refinements).

---

### 1.2 Blackman – MHT for Multiple Target Tracking (2004)

**Local file:**  
- `papers/Classic & conceptual/Blackman - MHT for MTT (2004).pdf`

**Citation:**  
S. S. Blackman, “Multiple Hypothesis Tracking for Multiple Target Tracking,” *IEEE Aerospace and Electronic Systems Magazine*, vol. 19, no. 1, pp. 5–18, Jan. 2004. :contentReference[oaicite:2]{index=2}  

**Online:**  
- IEEE Xplore / secondary mirrors (PDF available via institutional access):  
  https://ieeexplore.ieee.org/document/1263228 :contentReference[oaicite:3]{index=3}  

**Notes:**

- High-level conceptual overview of MHT and why it’s used.
- Discusses different implementation flavours in practice.
- Nice for “motivation and big picture” and for referencing terminology.

---

### 1.3 Hendeby & Karlsson – Lecture notes & handouts

**Local files:**  
- `papers/Classic & conceptual/Hendeby - Lecture notes - Le5.pdf`  
- `papers/Classic & conceptual/Hendeby - Lecture handouts - Le5.pdf`

**Online:**  
- Lecture notes “Multi-Target Tracking: multi-hypothesis tracking” (Le 5):  
  https://rt.isy.liu.se/student/graduate/targettracking/file/le5.pdf :contentReference[oaicite:4]{index=4}  

**Notes:**

- Very clear, modern summary of MHT (conceptual, HO-MHT, TO-MHT).
- Includes:
  - Derivation of hypothesis probabilities.
  - Complexity reduction: clustering, pruning, N-scan, merging.
  - Discussion of HO-MHT vs TO-MHT and K-best assignment (Murty’s).
- Great bridge between classical theory and more practical designs.

---

## 2. Implementation-focused TO-MHT papers

### 2.1 Sun et al. – Efficient TO-MHT via graphical models (2012 / 2017)

**Local file:**  
- `papers/Implementation/Sun-et-al - Efficient implementation of TO-MHT (2012).pdf`

**Citation (journal version):**  
J. Sun, Y. Li, S. Sun, X. Li, and X. Hu, “An Efficient Implementation of Track-Oriented Multiple Hypothesis Tracker Using Graphical Model Approaches,” *International Journal of Distributed Sensor Networks*, vol. 2017, Article ID 8061561. :contentReference[oaicite:5]{index=5}  

**Online:**

- Wiley / Hindawi PDF:  
  https://onlinelibrary.wiley.com/doi/10.1155/2017/8061561 :contentReference[oaicite:6]{index=6}  

**Notes:**

- Represents TO-MHT as a graphical model and uses message passing (MPBP) to approximate the MAP solution.
- Focuses on **efficient hypothesis generation and scoring**, given TO-MHT structure.
- Useful for:
  - Understanding how to map TO-MHT to a factor graph.
  - Ideas for improving the efficiency of our K-best / beam search framework.

---

### 2.2 He et al. – TO-MHT based on Tabu search and Gibbs sampling (2018)

**Local file:**  
- `papers/Implementation/He-et-al - TO-MHT based on Tabu search and Gibbs sampling (2018).pdf`

**Citation:**  
S. He, H.-S. Shin, and A. Tsourdos, “Track-Oriented Multiple Hypothesis Tracking Based on Tabu Search and Gibbs Sampling,” *IEEE Sensors Journal*, vol. 18, no. 17, pp. 7213–7226, 2018. :contentReference[oaicite:7]{index=7}  

**Online:**

- PDF (core / other mirrors):  
  https://core.ac.uk/download/pdf/188364739.pdf :contentReference[oaicite:8]{index=8}  

**Notes:**

- Casts TO-MHT hypothesis selection as a combinatorial optimisation problem.
- Uses Tabu search and Gibbs sampling to efficiently explore the hypothesis space.
- Particularly relevant if we later want:
  - Non-greedy global optimisation beyond K-best,
  - More sophisticated search in the space of track combinations.

---

## 3. Survey and “TO-MHT as graphical model” papers

### 3.1 Chong, Mori, Reid – Forty Years of MHT (2019)

**Local file:**  
- `papers/Survey/Chong-Mori-Reid - Forty Years of MHT (2019).pdf`

**Citation:**  
C.-Y. Chong, S. Mori, and D. B. Reid, “Forty Years of Multiple Hypothesis Tracking – A Review of Key Developments,” *Journal of Advances in Information Fusion*, vol. 14, no. 2, pp. 131–153, Dec. 2019. :contentReference[oaicite:9]{index=9}  

**Online:**

- JAIF PDF (ISIF):  
  https://isif.org/media/forty-years-multiple-hypothesis-tracking :contentReference[oaicite:10]{index=10}  

**Notes:**

- Broad historical overview from Reid (1979) through modern variations.
- Covers:
  - Measurement-oriented MHT, HO-MHT, TO-MHT.
  - N-scan, clustering, and various pruning strategies.
  - Key algorithmic and practical developments over 40 years.
- Great as a conceptual map of the whole literature.

---

### 3.2 Frank, Smyth, Ihler – Graphical model representation of TO-MHT (2012)

**Local file:**  
- `papers/Survey/Frank-et-al - Graphical model of TO-MHT (2012).pdf`  

**Citation:**  
A. Frank, P. Smyth, and A. Ihler, “A Graphical Model Representation of the Track-Oriented Multiple Hypothesis Tracker,” in *Proc. IEEE Statistical Signal Processing Workshop*, 2012. :contentReference[oaicite:11]{index=11}  

**Online:**

- PDF:  
  https://ics.uci.edu/~ihler/papers/ssp12.pdf :contentReference[oaicite:12]{index=12}  

**Notes:**

- Expresses TO-MHT as a factor graph:
  - Tracks and exclusivity constraints map naturally to graph structure.
  - Makes it possible to use belief propagation / variational methods.
- Very helpful for:
  - Understanding the relationship between our current implementation and a more principled graphical model view.
  - Designing improved scoring and inference schemes (e.g., message passing).

---

### 3.3 Frank, Smyth, Ihler – Beyond MAP with TO-MHT (2014)

**Local file:**  
- `papers/Survey/Frank-Smyth-Ihler - Beyond MAP with TO-MHT (2014).pdf`

**Citation:**  
A. Frank, P. Smyth, and A. Ihler, “Beyond MAP Estimation With the Track-Oriented Multiple Hypothesis Tracker,” *IEEE Transactions on Signal Processing*, vol. 62, no. 10, pp. 2413–2423, 2014. :contentReference[oaicite:13]{index=13}  

**Online:**

- Journal page / PDF via IEEE or publisher:  
  https://doi.org/10.1109/TSP.2014.2311962 :contentReference[oaicite:14]{index=14}  

**Notes:**

- Extends TO-MHT beyond pure MAP:
  - Uses a graphical model formulation to approximate track *marginals*.
  - Enables queries like “probability this target exists” or “probability track i is correct”, not just “best hypothesis”.
- Relevant if we later want:
  - More than a single MAP global hypothesis.
  - Better uncertainty quantification over track identities and existence.

---

## 4. How these references map to this project

Very briefly:

- **Reid (1979)** and **Hendeby notes** are our conceptual backbone for MHT, N-scan, and pruning strategies.
- **Blackman (2004)** and **Chong–Mori–Reid (2019)** give framing and context; they’re good to revisit when deciding which variant to emulate or approximate.
- **Sun et al. (2017)** and **He et al. (2018)** are our main “implementation inspiration” for:
  - efficient TO-MHT,
  - graphical model formulations,
  - and alternative search strategies (message passing, Tabu, Gibbs).
- **Frank/Smyth/Ihler (2012, 2014)** connect TO-MHT with graphical models and inference:
  - Helps when we want to refine our scoring model or think about marginals instead of just MAP.

As the implementation evolves (e.g. when we design Scoring v2 or N-scan-lite), we can link specific design decisions in `TO_MHT_NEXT_STEPS.md` back to these references and sections here.
