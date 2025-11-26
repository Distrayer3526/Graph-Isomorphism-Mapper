import sys
import random
import time
import heapq
import multiprocessing
import math
import gc
from collections import defaultdict

# ==========================================
# ⚙️ GLOBAL MEMORY
# ==========================================
G_N = 0
G_M = 0
G_ADJ_G = []
G_ADJ_H = []
G_ADJ_G_SETS = []
G_ADJ_H_SETS = []

G_COLORS_D4 = []
G_COLORS_D3 = []
G_COLORS_D2 = []

G_H_MAP_D4 = {}
G_H_MAP_D3 = {}
G_H_MAP_D2 = {}
G_DEGREE_BUCKETS_H = {}

G_ANCHORS = {} 
G_DEGREE_G = []
G_DEGREE_H = []
G_TRIANGLE_G = []
G_TRIANGLE_H = []
G_EIGEN_G = [] 
G_EIGEN_H = [] 

# ==========================================
# 🛠️ FEATURES
# ==========================================
def get_eigen_centrality(n, adj, steps=15):
    x = [1.0 / math.sqrt(n)] * n
    for _ in range(steps):
        new_x = [0.0] * n
        for u in range(n):
            for v in adj[u]:
                new_x[u] += x[v]
        norm_sq = sum(v*v for v in new_x)
        if norm_sq < 1e-12: break 
        norm = math.sqrt(norm_sq)
        x = [v/norm for v in new_x]
    return x

def count_triangles(n, adj):
    triangles = [0] * n
    for u in range(n):
        neighbors = set(adj[u])
        for v in adj[u]:
            if v > u:
                common = neighbors & set(adj[v])
                triangles[u] += len(common)
                triangles[v] += len(common)
    return triangles

def get_wl_cascade(n, adj, max_depth=4):
    triangles = count_triangles(n, adj)
    colors = [hash((len(adj[u]), triangles[u])) & 0xFFFFFFFFFFFFFFF for u in range(n)]
    history = [colors]
    
    for _ in range(max_depth):
        new_colors = []
        for u in range(n):
            s1 = 0
            s2 = 0
            for v in adj[u]:
                c = colors[v]
                s1 += c
                s2 ^= c 
            signature = (colors[u], s1, s2)
            new_colors.append(hash(signature) & 0xFFFFFFFFFFFFFFF)
        colors = new_colors
        history.append(colors)
    return history, triangles

def precompute_structures(n, adj_g, adj_h):
    print("   🌍 Calculating Global Eigenvector Centrality...")
    eigen_g = get_eigen_centrality(n, adj_g)
    eigen_h = get_eigen_centrality(n, adj_h)
    
    print("   🎨 Running WL Cascade (Depth 4 -> 3 -> 2)...")
    hist_g, tri_g = get_wl_cascade(n, adj_g, 4)
    hist_h, tri_h = get_wl_cascade(n, adj_h, 4)
    
    c_g_2, c_g_3, c_g_4 = hist_g[2], hist_g[3], hist_g[4]
    c_h_2, c_h_3, c_h_4 = hist_h[2], hist_h[3], hist_h[4]
    
    def build_map(colors_h):
        mapping = defaultdict(list)
        for v, c in enumerate(colors_h):
            mapping[c].append(v)
        final_map = {}
        for c, nodes in mapping.items():
            if len(nodes) <= 100: 
                final_map[c] = tuple(nodes) 
        return final_map

    print("   🚀 Indexing Maps (RAM Safe)...")
    map_d4 = build_map(c_h_4)
    map_d3 = build_map(c_h_3)
    map_d2 = build_map(c_h_2)
    
    deg_buckets = defaultdict(list)
    for v in range(n):
        d = len(adj_h[v])
        deg_buckets[d].append(v)
    final_deg_buckets = {d: tuple(vs) for d, vs in deg_buckets.items()}
    
    anchors = {}
    for u in range(n):
        c = c_g_4[u]
        if c in map_d4:
            matches = map_d4[c]
            if len(matches) == 1:
                v = matches[0]
                if abs(eigen_g[u] - eigen_h[v]) < 0.05:
                    anchors[u] = v

    return (c_g_4, c_g_3, c_g_2, 
            map_d4, map_d3, map_d2, 
            final_deg_buckets,
            anchors, tri_g, tri_h, eigen_g, eigen_h)

# ==========================================
# 🧠 WORKER LOGIC
# ==========================================
def worker_init(n, m, adj_g, adj_h, 
                cg4, cg3, cg2, 
                hm4, hm3, hm2, 
                deg_buckets,
                anchors, deg_g, deg_h, tri_g, tri_h, eig_g, eig_h):
    global G_N, G_M, G_ADJ_G, G_ADJ_H, G_ADJ_G_SETS, G_ADJ_H_SETS
    global G_COLORS_D4, G_COLORS_D3, G_COLORS_D2
    global G_H_MAP_D4, G_H_MAP_D3, G_H_MAP_D2, G_DEGREE_BUCKETS_H, G_ANCHORS
    global G_DEGREE_G, G_DEGREE_H, G_TRIANGLE_G, G_TRIANGLE_H, G_EIGEN_G, G_EIGEN_H
    
    G_N = n
    G_M = m
    G_ADJ_G = adj_g
    G_ADJ_H = adj_h
    G_ADJ_G_SETS = [set(x) for x in adj_g]
    G_ADJ_H_SETS = [set(x) for x in adj_h]
    
    G_COLORS_D4 = cg4
    G_COLORS_D3 = cg3
    G_COLORS_D2 = cg2
    G_H_MAP_D4 = hm4
    G_H_MAP_D3 = hm3
    G_H_MAP_D2 = hm2
    G_DEGREE_BUCKETS_H = deg_buckets
    
    G_ANCHORS = anchors
    G_DEGREE_G = deg_g
    G_DEGREE_H = deg_h
    G_TRIANGLE_G = tri_g
    G_TRIANGLE_H = tri_h
    G_EIGEN_G = eig_g
    G_EIGEN_H = eig_h

def solve_instance(args):
    seed_val, deadline, extra_anchors = args
    random.seed(seed_val)
    n = G_N
    
    mapping = [-1] * n
    reverse_mapping = [-1] * n
    unmapped_g = set(range(n))
    unmapped_h = set(range(n))
    
    combined_anchors = G_ANCHORS.copy()
    if extra_anchors: combined_anchors.update(extra_anchors)
        
    for u, v in combined_anchors.items():
        mapping[u] = v
        reverse_mapping[v] = u
        unmapped_g.discard(u)
        unmapped_h.discard(v)
        
    pq = []
    
    def add_candidates(u_mapped, v_mapped):
        nbrs_u = [x for x in G_ADJ_G[u_mapped] if mapping[x] == -1]
        nbrs_v = [y for y in G_ADJ_H[v_mapped] if reverse_mapping[y] == -1]
        
        if len(nbrs_u) > 10: nbrs_u = random.sample(nbrs_u, 10)
        if len(nbrs_v) > 10: nbrs_v = random.sample(nbrs_v, 10)
        
        for u_cand in nbrs_u:
            e_u = G_EIGEN_G[u_cand]
            valid_vs = []
            
            # D4
            c = G_COLORS_D4[u_cand]
            if c in G_H_MAP_D4:
                cands = G_H_MAP_D4[c]
                for v in nbrs_v:
                    if v in cands and abs(G_EIGEN_H[v] - e_u) < 0.04:
                        valid_vs.append(v)
            
            # D3
            if not valid_vs:
                c = G_COLORS_D3[u_cand]
                if c in G_H_MAP_D3:
                    cands = G_H_MAP_D3[c]
                    for v in nbrs_v:
                        if v in cands and abs(G_EIGEN_H[v] - e_u) < 0.08:
                            valid_vs.append(v)
            
            # Fallback: Strict Degree + Triangle
            if not valid_vs and nbrs_v:
                deg_u = G_DEGREE_G[u_cand]
                tri_u = G_TRIANGLE_G[u_cand]
                candidates = [v for v in nbrs_v if G_DEGREE_H[v] == deg_u]
                
                if candidates:
                    tri_candidates = [v for v in candidates if abs(G_TRIANGLE_H[v] - tri_u) <= 1]
                    if tri_candidates: valid_vs = tri_candidates
                    else: valid_vs = candidates
                else:
                    valid_vs = [v for v in nbrs_v if abs(G_DEGREE_H[v] - deg_u) <= 1]

            for v_cand in valid_vs:
                score = 0
                for nu in G_ADJ_G[u_cand]:
                    if mapping[nu] != -1 and mapping[nu] in G_ADJ_H_SETS[v_cand]:
                        score += 1
                
                eigen_diff = abs(G_EIGEN_G[u_cand] - G_EIGEN_H[v_cand]) * 20 
                tri_diff = abs(G_TRIANGLE_G[u_cand] - G_TRIANGLE_H[v_cand])
                
                heapq.heappush(pq, (-score, eigen_diff + tri_diff, u_cand, v_cand))

    # Bootstrap
    if combined_anchors:
        start_nodes = list(combined_anchors.keys())
        if len(start_nodes) > 40: start_nodes = random.sample(start_nodes, 40)
        for u in start_nodes: add_candidates(u, mapping[u])
    else:
        u_start = max(range(n), key=lambda x: G_EIGEN_G[x])
        possibles = list(unmapped_h)
        v_start = min(possibles, key=lambda v: abs(G_EIGEN_H[v] - G_EIGEN_G[u_start]))
        mapping[u_start] = v_start
        reverse_mapping[v_start] = u_start
        unmapped_g.discard(u_start)
        unmapped_h.discard(v_start)
        add_candidates(u_start, v_start)

    # Greedy Construction
    while unmapped_g:
        if len(unmapped_g) % 1000 == 0 and time.time() > deadline: break
        if not pq:
            u_next = max(unmapped_g, key=lambda x: G_EIGEN_G[x])
            
            possibles = []
            c = G_COLORS_D4[u_next]
            if c in G_H_MAP_D4:
                possibles = [v for v in G_H_MAP_D4[c] if reverse_mapping[v] == -1]
            
            if not possibles:
                c = G_COLORS_D3[u_next]
                if c in G_H_MAP_D3:
                    possibles = [v for v in G_H_MAP_D3[c] if reverse_mapping[v] == -1]
            
            if not possibles:
                deg = G_DEGREE_G[u_next]
                if deg in G_DEGREE_BUCKETS_H:
                    possibles = [v for v in G_DEGREE_BUCKETS_H[deg] if reverse_mapping[v] == -1]
            
            if not possibles: possibles = list(unmapped_h)
            
            e_u = G_EIGEN_G[u_next]
            if len(possibles) > 100: subset = random.sample(possibles, 100)
            else: subset = possibles
            
            v_next = min(subset, key=lambda v: abs(G_EIGEN_H[v] - e_u))
            
            mapping[u_next] = v_next
            reverse_mapping[v_next] = u_next
            unmapped_g.discard(u_next)
            unmapped_h.discard(v_next)
            add_candidates(u_next, v_next)
            continue

        score, _, u, v = heapq.heappop(pq)
        if mapping[u] != -1 or reverse_mapping[v] != -1: continue
        mapping[u] = v
        reverse_mapping[v] = u
        unmapped_g.discard(u)
        unmapped_h.discard(v)
        add_candidates(u, v)

    if unmapped_g:
        rem_g = list(unmapped_g)
        rem_h = list(unmapped_h)
        random.shuffle(rem_h)
        for i, u in enumerate(rem_g):
            if i < len(rem_h): mapping[u] = rem_h[i]

    # --- IMPROVED REPAIR PHASE ---
    # Key changes: Logarithmic cooling + edge-focused swaps
    repair_iterations = 8000000  # Increased from 6M
    last_improvement = 0
    
    best_score = 0
    for u in range(n):
        v = mapping[u]
        for nbr in G_ADJ_G[u]:
            if mapping[nbr] in G_ADJ_H_SETS[v]: best_score += 1
    best_score //= 2
    best_mapping = mapping[:]
    
    # IMPROVED: Logarithmic cooling schedule (better for graph problems) [[2]](#__2) [[3]](#__3)
    T0 = 2.0  # Higher initial temperature
    
    for i in range(repair_iterations):
        if i % 100 == 0 and time.time() > deadline: break
        
        # Logarithmic cooling: T = T0 / log(1 + k)
        T = T0 / math.log(2 + i / 1000.0)
        
        # REHEATING: If stuck for 15k iterations
        if i - last_improvement > 15000:
             T = 1.5
             last_improvement = i
        
        strategy = i % 10  # 10 strategies
        
        # Strategy 0: IMPROVED - Edge-preserving swap (VF2++ inspired) [[0]](#__0)
        if strategy == 0:
            # Find mismatched edge
            u1 = random.randint(0, n-1)
            v1 = mapping[u1]
            mismatched = []
            for nbr in G_ADJ_G[u1]:
                if mapping[nbr] not in G_ADJ_H_SETS[v1]:
                    mismatched.append(nbr)
            
            if mismatched:
                u2 = random.choice(mismatched)
                v2 = mapping[u2]
                
                # Quick delta calculation
                old_local = 0
                for nbr in G_ADJ_G[u1]:
                    if mapping[nbr] in G_ADJ_H_SETS[v1]: old_local += 1
                for nbr in G_ADJ_G[u2]:
                    if mapping[nbr] in G_ADJ_H_SETS[v2]: old_local += 1
                
                mapping[u1], mapping[u2] = v2, v1
                
                new_local = 0
                for nbr in G_ADJ_G[u1]:
                    if mapping[nbr] in G_ADJ_H_SETS[v2]: new_local += 1
                for nbr in G_ADJ_G[u2]:
                    if mapping[nbr] in G_ADJ_H_SETS[v1]: new_local += 1
                
                delta = new_local - old_local
                
                if delta > 0 or random.random() < math.exp(delta / T):
                    if new_local > old_local:
                        total = 0
                        for u in range(n):
                            v = mapping[u]
                            for nbr in G_ADJ_G[u]:
                                if mapping[nbr] in G_ADJ_H_SETS[v]: total += 1
                        total //= 2
                        if total > best_score:
                            best_score = total
                            best_mapping = mapping[:]
                            last_improvement = i
                else:
                    mapping[u1], mapping[u2] = v1, v2
        
        # Strategy 1: Cluster Monte Carlo
        elif strategy == 1:
            u_center = random.randint(0, n-1)
            cluster = {u_center}
            for nbr in G_ADJ_G[u_center]: cluster.add(nbr)
            target_size = random.randint(10, 40)
            frontier = list(cluster)
            while len(cluster) < target_size and frontier:
                curr = frontier.pop(0)
                for nbr in G_ADJ_G[curr]:
                    if nbr not in cluster:
                        cluster.add(nbr)
                        frontier.append(nbr)
                        if len(cluster) >= target_size: break
            
            cluster_list = list(cluster)
            old_local = 0
            involved = set(cluster_list)
            for u in cluster_list:
                for nbr in G_ADJ_G[u]: involved.add(nbr)
            for u in involved:
                v = mapping[u]
                for nbr in G_ADJ_G[u]:
                    if mapping[nbr] in G_ADJ_H_SETS[v]: old_local += 1
            old_local //= 2
            
            backup_map = {}
            freed_h = []
            for u in cluster_list:
                v = mapping[u]
                backup_map[u] = v
                mapping[u] = -1
                reverse_mapping[v] = -1
                freed_h.append(v)
            
            random.shuffle(freed_h)
            for idx, u in enumerate(cluster_list):
                if idx < len(freed_h):
                    mapping[u] = freed_h[idx]
                    reverse_mapping[freed_h[idx]] = u
            
            new_local = 0
            for u in involved:
                v = mapping[u]
                for nbr in G_ADJ_G[u]:
                    if mapping[nbr] in G_ADJ_H_SETS[v]: new_local += 1
            new_local //= 2
            
            delta = new_local - old_local
            if delta > 0 or random.random() < math.exp(delta / T):
                if new_local > old_local:
                    total = 0
                    for u in range(n):
                        v = mapping[u]
                        for nbr in G_ADJ_G[u]:
                            if mapping[nbr] in G_ADJ_H_SETS[v]: total += 1
                    total //= 2
                    if total > best_score:
                        best_score = total
                        best_mapping = mapping[:]
                        last_improvement = i
            else:
                for u in cluster_list:
                    v = mapping[u]
                    mapping[u] = backup_map[u]
                    reverse_mapping[backup_map[u]] = u
                    reverse_mapping[v] = -1
        
        # Strategies 2-9: Random swaps with varying neighborhood sizes
        else:
            u1, u2 = random.sample(range(n), 2)
            v1, v2 = mapping[u1], mapping[u2]
            
            old_local = 0
            for nbr in G_ADJ_G[u1]:
                if mapping[nbr] in G_ADJ_H_SETS[v1]: old_local += 1
            for nbr in G_ADJ_G[u2]:
                if mapping[nbr] in G_ADJ_H_SETS[v2]: old_local += 1
            
            mapping[u1], mapping[u2] = v2, v1
            
            new_local = 0
            for nbr in G_ADJ_G[u1]:
                if mapping[nbr] in G_ADJ_H_SETS[v2]: new_local += 1
            for nbr in G_ADJ_G[u2]:
                if mapping[nbr] in G_ADJ_H_SETS[v1]: new_local += 1
            
            delta = new_local - old_local
            
            if delta > 0 or random.random() < math.exp(delta / T):
                if new_local > old_local:
                    total = 0
                    for u in range(n):
                        v = mapping[u]
                        for nbr in G_ADJ_G[u]:
                            if mapping[nbr] in G_ADJ_H_SETS[v]: total += 1
                    total //= 2
                    if total > best_score:
                        best_score = total
                        best_mapping = mapping[:]
                        last_improvement = i
            else:
                mapping[u1], mapping[u2] = v1, v2

    return best_mapping, best_score

# ==========================================
# 🚀 MAIN
# ==========================================
def main():
    input_file = "graphs"
    output_file = "ans"
    
    with open(input_file, 'r') as f:
        n, m = map(int, f.readline().split())
        adj_g = [[] for _ in range(n)]
        adj_h = [[] for _ in range(n)]
        
        # Read edges for G (convert from 1-indexed to 0-indexed)
        for _ in range(m):
            u, v = map(int, f.readline().split())
            u -= 1  # Convert to 0-indexed
            v -= 1  # Convert to 0-indexed
            adj_g[u].append(v)
            adj_g[v].append(u)
        
        # Read edges for H (convert from 1-indexed to 0-indexed)
        for _ in range(m):
            u, v = map(int, f.readline().split())
            u -= 1  # Convert to 0-indexed
            v -= 1  # Convert to 0-indexed
            adj_h[u].append(v)
            adj_h[v].append(u)
    
    print(f"📊 Loaded: {n} nodes, {m} edges")
    print("="*70)
    
    deg_g = [len(adj_g[u]) for u in range(n)]
    deg_h = [len(adj_h[u]) for u in range(n)]
    
    (cg4, cg3, cg2, hm4, hm3, hm2, deg_buckets, anchors, 
     tri_g, tri_h, eig_g, eig_h) = precompute_structures(n, adj_g, adj_h)
    
    print(f"✅ Found {len(anchors)} high-confidence anchors")
    print("="*70)
    
    deadline = time.time() + 280  # 4:40 time limit
    
    num_workers = min(8, multiprocessing.cpu_count())
    pool = multiprocessing.Pool(
        processes=num_workers,
        initializer=worker_init,
        initargs=(n, m, adj_g, adj_h, cg4, cg3, cg2, hm4, hm3, hm2, 
                 deg_buckets, anchors, deg_g, deg_h, tri_g, tri_h, eig_g, eig_h)
    )
    
    tasks = [(42 + i, deadline, None) for i in range(num_workers)]
    results = pool.map(solve_instance, tasks)
    pool.close()
    pool.join()
    
    best_mapping, best_score = max(results, key=lambda x: x[1])
    
    # Write output (convert back to 1-indexed)
    with open(output_file, 'w') as f:
        for u in range(n):
            f.write(f"{best_mapping[u] + 1}\n")  # Convert back to 1-indexed
    
    match_ratio = best_score / m
    score = 5.333 * (match_ratio ** 3) - 4 * (match_ratio ** 2) + 2.667 * match_ratio
    
    print("="*70)
    print("BEST RESULT:")
    print(f"   Matched edges: {best_score}/{m}")
    print(f"   Match ratio: {match_ratio*100:.2f}%")
    print(f"   Score: {score:.4f}")
    print("="*70)
    print("Permutation written to ans")

if __name__ == "__main__":
    main()
