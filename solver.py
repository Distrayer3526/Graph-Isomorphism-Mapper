import sys
import random
import time
import heapq
import multiprocessing
import math
import gc
import itertools 
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
            
            c = G_COLORS_D4[u_cand]
            if c in G_H_MAP_D4:
                cands = G_H_MAP_D4[c]
                for v in nbrs_v:
                    if v in cands and abs(G_EIGEN_H[v] - e_u) < 0.04:
                        valid_vs.append(v)
            
            if not valid_vs:
                c = G_COLORS_D3[u_cand]
                if c in G_H_MAP_D3:
                    cands = G_H_MAP_D3[c]
                    for v in nbrs_v:
                        if v in cands and abs(G_EIGEN_H[v] - e_u) < 0.08:
                            valid_vs.append(v)
            
            if not valid_vs and nbrs_v:
                deg_u = G_DEGREE_G[u_cand]
                tri_u = G_TRIANGLE_G[u_cand]
                candidates = [v for v in nbrs_v if G_DEGREE_H[v] == deg_u]
                if candidates:
                    tri_candidates = [v for v in candidates if abs(G_TRIANGLE_H[v] - tri_u) <= 1]
                    valid_vs = tri_candidates if tri_candidates else candidates
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

    # --- REPAIR PHASE ---
    repair_iterations = 6000000 
    last_improvement = 0
    leaves_g = [u for u in range(n) if G_DEGREE_G[u] == 1]
    
    best_score = 0
    for u in range(n):
        v = mapping[u]
        for nbr in G_ADJ_G[u]:
            if mapping[nbr] in G_ADJ_H_SETS[v]: best_score += 1
    best_score //= 2
    best_mapping = mapping[:]
    
    T = 1.0
    cooling_rate = 0.99999

    for i in range(repair_iterations):
        if i % 100 == 0 and time.time() > deadline: break
        if i % 1000 == 0: T *= cooling_rate
        if i - last_improvement > 20000:
             T = 0.8
             last_improvement = i 
        
        strategy = i % 10 
        
        # Strategy 0/1: Monte Carlo
        if strategy <= 1:
            u_center = random.randint(0, n-1)
            cluster = {u_center}
            for nbr in G_ADJ_G[u_center]: cluster.add(nbr)
            target_size = random.randint(10, 50)
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
            
            best_mc_map = None
            best_mc_score = -1
            
            for _ in range(3):
                temp_map = {}
                temp_freed = list(freed_h)
                random.shuffle(cluster_list)
                cluster_list.sort(key=lambda x: -G_EIGEN_G[x] + random.random()*0.01)
                
                trial_ok = True
                for u in cluster_list:
                    c = G_COLORS_D4[u]
                    cands = G_H_MAP_D4.get(c, None)
                    if not cands:
                         c = G_COLORS_D3[u]
                         cands = G_H_MAP_D3.get(c, None)
                    if cands: valid = [v for v in temp_freed if v in cands]
                    else: valid = []
                    if not valid:
                        deg = G_DEGREE_G[u]
                        valid = [v for v in temp_freed if G_DEGREE_H[v] == deg]
                    if not valid: valid = temp_freed
                    
                    best_v = -1
                    best_v_sc = -1
                    for v in valid:
                        sc = 0
                        for nbr in G_ADJ_G[u]:
                            if mapping[nbr] != -1:
                                if mapping[nbr] in G_ADJ_H_SETS[v]: sc += 1
                            elif nbr in temp_map:
                                if temp_map[nbr] in G_ADJ_H_SETS[v]: sc += 1
                        if sc > best_v_sc:
                            best_v_sc = sc
                            best_v = v
                    if best_v != -1:
                        temp_map[u] = best_v
                        temp_freed.remove(best_v)
                    else:
                        trial_ok = False; break
                if trial_ok:
                    sc = 0
                    for u, v in temp_map.items():
                        for nbr in G_ADJ_G[u]:
                            if mapping[nbr] != -1:
                                if mapping[nbr] in G_ADJ_H_SETS[v]: sc += 1
                            elif nbr in temp_map:
                                if temp_map[nbr] in G_ADJ_H_SETS[v]: sc += 1
                    if sc > best_mc_score:
                        best_mc_score = sc
                        best_mc_map = temp_map
            
            if best_mc_map:
                for u, v in best_mc_map.items():
                    mapping[u] = v
                    reverse_mapping[v] = u
            else:
                for u, v in backup_map.items(): mapping[u]=v; reverse_mapping[v]=u
                continue
                
            new_local = 0
            for u in involved:
                v = mapping[u]
                for nbr in G_ADJ_G[u]:
                    if mapping[nbr] in G_ADJ_H_SETS[v]: new_local += 1
            new_local //= 2
            
            if new_local > old_local: last_improvement = i
            elif new_local < old_local:
                for u in cluster_list: 
                    if mapping[u]!=-1: reverse_mapping[mapping[u]]=-1
                for u, v in backup_map.items(): mapping[u]=v; reverse_mapping[v]=u

        # Strategy 2: Smart Swap
        elif strategy == 2:
            u1 = random.randint(0, n-1)
            if G_ADJ_G[u1]:
                u2 = random.choice(G_ADJ_G[u1])
                v1, v2 = mapping[u1], mapping[u2]
                if G_DEGREE_G[u1] == G_DEGREE_H[v2] and G_DEGREE_G[u2] == G_DEGREE_H[v1]:
                    if v2 not in G_ADJ_H_SETS[v1] and v1 in G_ADJ_H_SETS[v2]:
                        mapping[u1], mapping[u2] = v2, v1
                        reverse_mapping[v1], reverse_mapping[v2] = u2, u1
                        last_improvement = i
        
        # Strategy 6: Leaf Snapper
        elif strategy == 6:
            batch = random.sample(leaves_g, min(len(leaves_g), 100))
            for u in batch:
                v = mapping[u]
                parent_g = G_ADJ_G[u][0]
                parent_h_target = mapping[parent_g]
                if parent_h_target not in G_ADJ_H_SETS[v]:
                    candidates = G_ADJ_H[parent_h_target]
                    random.shuffle(candidates)
                    for candidate_v in candidates:
                        if G_DEGREE_H[candidate_v] == 1:
                            owner_u = reverse_mapping[candidate_v]
                            owner_parent_g = G_ADJ_G[owner_u][0]
                            if mapping[owner_parent_g] != parent_h_target:
                                mapping[u] = candidate_v
                                mapping[owner_u] = v
                                reverse_mapping[candidate_v] = u
                                reverse_mapping[v] = owner_u
                                last_improvement = i
                                break

        # Strategy 7: SA Swap
        elif strategy == 7:
            u1 = random.randint(0, n-1)
            u2 = random.randint(0, n-1)
            v1, v2 = mapping[u1], mapping[u2]
            if G_DEGREE_G[u1] != G_DEGREE_H[v2] or G_DEGREE_G[u2] != G_DEGREE_H[v1]: continue
            
            old_score = 0
            new_score = 0
            for nbr in G_ADJ_G[u1]:
                if mapping[nbr] in G_ADJ_H_SETS[v1]: old_score += 1
                if mapping[nbr] in G_ADJ_H_SETS[v2]: new_score += 1
            for nbr in G_ADJ_G[u2]:
                if mapping[nbr] in G_ADJ_H_SETS[v2]: old_score += 1
                if mapping[nbr] in G_ADJ_H_SETS[v1]: new_score += 1
            
            delta = new_score - old_score
            if delta > 0 or (delta > -2 and random.random() < math.exp(delta / T)):
                mapping[u1], mapping[u2] = v2, v1
                reverse_mapping[v1], reverse_mapping[v2] = u2, u1
                if delta > 0: last_improvement = i

        # Strategy 9: MICRO-BRUTE FORCE (NEW)
        elif strategy == 9:
            # 1. Pick a random center
            u_center = random.randint(0, n-1)
            # 2. Get 4 neighbors (Cluster size 5)
            cluster = [u_center]
            for nbr in G_ADJ_G[u_center]:
                if len(cluster) >= 5: break
                cluster.append(nbr)
            
            if len(cluster) < 2: continue

            # 3. Get current targets in H
            targets = [mapping[u] for u in cluster]
            
            # 4. Brute Force Permutations (Max 5! = 120 checks, very fast)
            best_perm_score = -1
            best_perm = None
            
            # Calculate current local score for this cluster
            current_local = 0
            for u in cluster:
                for nbr in G_ADJ_G[u]:
                    if mapping[nbr] in G_ADJ_H_SETS[mapping[u]]: current_local += 1
            
            # Try all permutations of targets
            for p in itertools.permutations(targets):
                # Temporarily map cluster -> p
                # Check score
                temp_score = 0
                valid_perm = True
                
                # Quick Degree Check
                for i, u in enumerate(cluster):
                    if G_DEGREE_G[u] != G_DEGREE_H[p[i]]:
                        valid_perm = False; break
                if not valid_perm: continue

                # Score Check
                # Note: We only check edges connected to the cluster nodes
                # We need a fast way to check.
                # Just assume the mapping for nodes OUTSIDE the cluster is fixed.
                
                # Create temp lookup for p
                temp_mapping = {cluster[i]: p[i] for i in range(len(cluster))}
                
                for i, u in enumerate(cluster):
                    v_new = p[i]
                    for nbr in G_ADJ_G[u]:
                        # Where is nbr mapped?
                        if nbr in temp_mapping: v_nbr = temp_mapping[nbr]
                        else: v_nbr = mapping[nbr]
                        
                        if v_nbr in G_ADJ_H_SETS[v_new]: temp_score += 1
                
                if temp_score > current_local:
                    current_local = temp_score
                    best_perm = p
            
            # Apply best permutation
            if best_perm:
                for i, u in enumerate(cluster):
                    v_old = mapping[u]
                    v_new = best_perm[i]
                    if v_old != v_new:
                        mapping[u] = v_new
                        reverse_mapping[v_new] = u
                        # Note: We need to be careful about swapping. 
                        # Ideally we swap the owners, but here we just permute a closed set.
                        # Since targets came from mapping[cluster], we are just shuffling ownership within the cluster.
                        # So reverse_mapping is safe.
                last_improvement = i

        if i % 100 == 0:
            curr = 0
            for u in range(n):
                v = mapping[u]
                for nbr in G_ADJ_G[u]:
                    if mapping[nbr] in G_ADJ_H_SETS[v]: curr += 1
            curr //= 2
            if curr > best_score:
                best_score = curr
                best_mapping = mapping[:]
                last_improvement = i
                if best_score == G_M: break

    return best_score, best_mapping

# ==========================================
# 🚀 MAIN
# ==========================================
class MicroBruteSolver:
    def __init__(self, filename):
        self.filename = filename
        self.n = 0
        self.m = 0
        self.adj_g = [] 
        self.adj_h = []
        self.best_mapping = []
        self.best_score = -1

    def read_input(self):
        print(f"📖 Reading {self.filename}...")
        with open(self.filename, 'r') as f:
            content = f.read().split()
        it = iter(content)
        try:
            self.n = int(next(it))
            self.m = int(next(it))
            self.adj_g = [[] for _ in range(self.n)]
            self.adj_h = [[] for _ in range(self.n)]
            for _ in range(self.m):
                u, v = int(next(it))-1, int(next(it))-1
                self.adj_g[u].append(v)
                self.adj_g[v].append(u)
            for _ in range(self.m):
                u, v = int(next(it))-1, int(next(it))-1
                self.adj_h[u].append(v)
                self.adj_h[v].append(u)
        except StopIteration: pass
        print(f"📊 Graph Loaded: {self.n} Nodes, {self.m} Edges")

    def solve(self, duration=180):
        structs = precompute_structures(self.n, self.adj_g, self.adj_h)
        (cg4, cg3, cg2, hm4, hm3, hm2, deg_buckets, anchors, tri_g, tri_h, eig_g, eig_h) = structs
        
        deg_g = [len(adj) for adj in self.adj_g]
        deg_h = [len(adj) for adj in self.adj_h]
        
        print(f"   🔒 Initial Anchors: {len(anchors)}")
        gc.collect()

        start_time = time.time()
        cores = max(1, min(6, multiprocessing.cpu_count())) 
        
        # PHASE 1
        phase1_duration = duration * 0.5
        deadline1 = start_time + phase1_duration
        
        print(f"🔥 Phase 1: Generating Population ({phase1_duration:.0f}s) on {cores} cores...")
        
        pool = multiprocessing.Pool(
            processes=cores, 
            initializer=worker_init, 
            initargs=(self.n, self.m, self.adj_g, self.adj_h, 
                      cg4, cg3, cg2, hm4, hm3, hm2, 
                      deg_buckets,
                      anchors, deg_g, deg_h, tri_g, tri_h, eig_g, eig_h)
        )
        
        seeds = ((random.randint(0, 10**9), deadline1, None) for _ in range(100))
        results = pool.imap_unordered(solve_instance, seeds)
        
        population = []
        
        try:
            for score, mapping in results:
                if score > self.best_score:
                    self.best_score = score
                    self.best_mapping = mapping
                    print(f"   ✨ [P1] Score: {score}/{self.m} ({(score/self.m)*100:.2f}%)")
                    if score == self.m:
                        print("   🏆 Perfect Match Found!")
                        pool.terminate()
                        return
                
                population.append((score, mapping))
                if time.time() > deadline1 + 5: break
        except KeyboardInterrupt: pass
        
        pool.terminate()
        pool.join()
        
        # PHASE 2
        print("🧬 Phase 2: Calculating Consensus...")
        population.sort(key=lambda x: x[0], reverse=True)
        top_k = population[:5] 
        
        consensus_anchors = {}
        if len(top_k) >= 2:
            for u in range(self.n):
                target = top_k[0][1][u]
                is_consensus = True
                for i in range(1, len(top_k)):
                    if top_k[i][1][u] != target:
                        is_consensus = False
                        break
                if is_consensus:
                    consensus_anchors[u] = target
        
        print(f"   🔒 Consensus Found: {len(consensus_anchors)} nodes locked.")
        
        # PHASE 3
        remaining_time = (start_time + duration) - time.time()
        if remaining_time < 10: remaining_time = 10
        deadline2 = time.time() + remaining_time
        
        print(f"🚀 Phase 3: Fusion Run ({remaining_time:.0f}s)...")
        
        pool = multiprocessing.Pool(
            processes=cores, 
            initializer=worker_init, 
            initargs=(self.n, self.m, self.adj_g, self.adj_h, 
                      cg4, cg3, cg2, hm4, hm3, hm2, 
                      deg_buckets,
                      anchors, deg_g, deg_h, tri_g, tri_h, eig_g, eig_h)
        )
        
        seeds = ((random.randint(0, 10**9), deadline2, consensus_anchors) for _ in range(100))
        results = pool.imap_unordered(solve_instance, seeds)
        
        try:
            for score, mapping in results:
                if score > self.best_score:
                    self.best_score = score
                    self.best_mapping = mapping
                    print(f"   ✨ [P3] Score: {score}/{self.m} ({(score/self.m)*100:.2f}%)")
                    if score == self.m:
                        print("   🏆 Perfect Match Found!")
                        pool.terminate()
                        return
                if time.time() > deadline2 + 5: break
        except KeyboardInterrupt: pass
        finally:
            pool.terminate()
            pool.join()

        print(f"✅ Final Best Score: {self.best_score}")

    def write_output(self):
        if not self.best_mapping: self.best_mapping = list(range(self.n))
        print("💾 Writing output...")
        with open("ans", "w") as f:
            for i in range(self.n):
                f.write(f"{self.best_mapping[i] + 1}\n")
        
        x = self.best_score / self.m if self.m > 0 else 0
        grade = 5.333 * (x**3) - 4 * (x**2) + 2.667 * x
        print("="*40)
        print(f"📈 Overlap: {x:.4f}")
        print(f"🎓 Grade:   {grade:.4f} / 4.0")
        print("="*40)

if __name__ == "__main__":
    multiprocessing.freeze_support()
    s = MicroBruteSolver("graphs")
    s.read_input()
    s.solve(duration=180)
    s.write_output()
