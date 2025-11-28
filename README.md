# 🕸️ High-Performance Maximum Common Subgraph Solver

> A highly optimized, parallelized solver for the Maximum Common Subgraph (MCS) problem, utilizing Weisfeiler-Lehman kernels, spectral alignment, and stochastic local search.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue) ![Multiprocessing](https://img.shields.io/badge/Concurrency-Multiprocessing-green) ![Algorithm](https://img.shields.io/badge/Algorithm-Heuristic%20Optimization-orange)

## 📖 Overview

This project implements a high-performance heuristic solver designed to find the largest isomorphic subgraph between two arbitrary graphs ($G$ and $H$). Unlike exact solvers that struggle with NP-Hard complexity on large graphs, this solver employs a **hybrid architecture** combining:

1.  **Deep Feature Extraction** (Spectral & Topological)
2.  **Greedy Constructive Heuristics** (Jaccard & Priority Queues)
3.  **Stochastic Local Search** (Simulated Annealing & Tabu-like strategies)
4.  **Evolutionary Fusion** (Consensus-based Warm Starts)

The solver is capable of breaking out of deep local optima (Basins of Attraction) using a novel **Stochastic Anchor Erosion** technique.

---

## 🚀 Key Features

### 🧠 Advanced Feature Engineering
*   **Cascading Weisfeiler-Lehman (WL) Kernel:** Implements a "Smart Cascade" (Depth 5 $\to$ 4 $\to$ 3) to balance strict structural matching with fuzzy fallback for noisy graphs.
*   **Spectral Alignment:** Computes **Eigenvector Centrality** (25 iterations) to align nodes based on global influence.
*   **Topological Signatures:** Utilizes triangle counting and degree bucketing for rapid candidate filtering.

### ⚡ Parallel Architecture
*   **Multiprocessing Pool:** Automatically scales to available CPU cores.
*   **Diverse Seed Generation:** Each worker thread operates with unique random seeds and "Erosion Rates" to explore different areas of the solution space.

### 🛠️ Robust Optimization Strategies
The solver cycles through **11 distinct repair strategies** during the refinement phase:
*   **Strategy 9 (SOTA): Association Graph Repair:** Maps local isomorphism conflicts to a Maximum Clique problem to solve complex structural mismatches perfectly.
*   **Strategy 10: Global Shakeup:** Periodically unmaps the bottom 5% of "unhappy" nodes to force structural reorganization.
*   **Leaf & Chain Snapping:** Specialized handling for degree-1 and degree-2 nodes to maximize edge coverage.
*   **Edge Hunter:** Aggressively targets broken edges to swap nodes into better positions.

---

## 🔬 Technical Deep Dive

### 1. The "Stochastic Anchor Erosion" (Warm Start)
Standard warm starts often trap solvers in local optima (e.g., getting stuck at a score of 3.79 when 3.85 is possible). This project implements **Erosion**:
*   **The Anchor Thread:** Keeps 100% of the warm start mapping and runs conservative hill-climbing.
*   **The Explorer Threads:** Randomly "erode" (delete) 1%, 5%, 20%, or 50% of the warm start anchors. This forces the solver to reconstruct parts of the graph, often discovering better structural alignments that were previously impossible to reach.

### 2. Weighted Consensus
In Phase 2, the solver aggregates solutions from all parallel workers. Instead of requiring unanimous agreement, it uses a **4/5 Majority Vote**. This locks in high-confidence node mappings ("Anchors") while discarding noise from rogue threads.

### 3. Jaccard-Enhanced Greedy Construction
During the initial mapping phase, candidates are prioritized not just by feature similarity, but by the **Jaccard Similarity** of their neighborhoods. This ensures that nodes are mapped to targets that share the highest overlap with already-mapped neighbors.

---

## ⚙️ Installation & Usage

### Prerequisites
*   Python 3.8+
*   Standard libraries only (`multiprocessing`, `heapq`, `collections`, `math`, `random`). No heavy external dependencies (like numpy/networkx) required for portability.

### Running the Solver
The solver expects a file named `graphs` in the root directory, or it can read from `stdin`.

```bash
# Standard run (180 seconds default)
python solver.py

# Run with a warm start file (previous best solution)
python solver.py best_ans
```
You can also run the file for multiple iterations easily using the bash scripts provided.

```bash
# Simply running the file for multiple iterations
./run_loop.sh

# Running the file for multiple iterations and utilize the warm start functionality
./run_loop_warmstart.sh
```

### Input Format
The input file (`graphs`) should follow this structure:
```text
N M           # Number of nodes, Number of edges
u v           # Edge 1 for Graph G
...
u v           # Edge M for Graph G
u v           # Edge 1 for Graph H
...
u v           # Edge M for Graph H
```

### Output
The solver generates a file named `ans` containing the mapping:
*   Line `i` contains the node index in Graph $H$ that Node `i` in Graph $G$ maps to.

---

## 📊 Performance Logic

The solver calculates a score based on the number of preserved edges. The internal grading formula used to evaluate success is:

$$ Grade = 5.333 \cdot x^3 - 4 \cdot x^2 + 2.667 \cdot x $$

Where $x$ is the ratio of mapped edges to total edges ($Score / M$).

---

## 📂 File Structure

*   `solver_final_v3.py`: The main executable containing the complete logic.
    *   `precompute_structures()`: WL Kernel & Eigenvector logic.
    *   `solve_instance()`: The worker thread logic (Construction + Repair).
    *   `solve_max_clique_heuristic()`: Helper for Association Graph repair.
    *   `PostProcessSolver`: Main class handling I/O and Multiprocessing.

---

## 📜 License

This project is open-source. Feel free to use it for educational purposes or optimization challenges.
