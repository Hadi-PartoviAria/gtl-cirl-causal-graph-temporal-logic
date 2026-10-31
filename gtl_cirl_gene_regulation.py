import numpy as np
import random
from copy import deepcopy
from collections import deque
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Callable, Optional
import warnings  # stdlib

# Optional (for BayesOpt-style CaSTL search). Falls back to random search if sklearn missing.
try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as C
    from sklearn.preprocessing import StandardScaler
    from sklearn.exceptions import ConvergenceWarning
    _HAS_SK = True
except Exception:
    _HAS_SK = False

# Suppress noisy GP convergence warnings (toggle to False if you want to see them)
SUPPRESS_SK_WARNINGS = True
if _HAS_SK and SUPPRESS_SK_WARNINGS:
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

# ===============================================================
# Case Study: Gene Regulation Network for Disease Control
# ===============================================================
# This version:
#  - Infers CaSTL *cause* (G[0,G_end], neighbor count) via quick BO/random search
#  - Adds a **causal hook**: only when the cause holds and the agent
#    completes a timed sequence ModifyG1→G2→G4 (with slack) do we trigger
#    remission D→0 once inside [12,15].
#  - Compares baseline, counterfactual, and GTL-CaSTL (which uses only the
#    inferred causal knowledge).
#  - Training is intentionally **slow to converge** (low LR, high epsilon,
#    slow epsilon decay, sparser rewards, noisy dynamics).
# ===============================================================

# ── effect target (disease remission) ───────────────────────────
D_THRESH = 0.5           # treat D≤0.5 as remission
HORIZON  = 30            # plotting window for success metric
SEED     = 42
random.seed(SEED); np.random.seed(SEED)
# ───────────────────────────────────────────────────────────────

# ============================
#  Minimal STL Robustness API
# ============================
def stl_pred_D_leq(d_values: np.ndarray, thr: float) -> np.ndarray:
    return (thr - d_values).astype(float)

def stl_F_window(rob: np.ndarray, a: int, b: int) -> float:
    a = max(0, int(a)); b = int(b)
    if len(rob) == 0: return -np.inf
    b = min(b, len(rob)-1)
    if a > b: return -np.inf
    return float(np.max(rob[a:b+1]))

def stl_effect_robustness_D_eventually(d_hist: deque, a: int, b: int, thr: float) -> float:
    if len(d_hist) < b + 1: return -1.0
    window = np.array(list(d_hist)[:b+1][::-1], dtype=float)
    rob = stl_pred_D_leq(window, thr)
    return stl_F_window(rob, a, b)

# ----------------------------
# Graph Temporal Logic blocks
# ----------------------------
class GTLNodeProposition:
    def __init__(self, variable: str, operator: str, threshold: float):
        self.variable = variable; self.operator = operator; self.threshold = threshold
    def evaluate(self, graph, node, time_index):
        val = graph.get_node_label(node, time_index, self.variable)
        if self.operator == '>=': return val >= self.threshold
        if self.operator == '<=': return val <= self.threshold
        if self.operator == '>':  return val >  self.threshold
        if self.operator == '<':  return val <  self.threshold
        if self.operator == '==': return float(val == self.threshold)
        return False
    def __str__(self): return f"{self.variable} {self.operator} {self.threshold}"

class GTLEdgeProposition:
    def __init__(self, variable: str, operator: str, threshold: float):
        self.variable = variable; self.operator = operator; self.threshold = threshold
    def evaluate(self, graph, edge, time_index):
        val = graph.get_edge_label(edge, time_index, self.variable)
        if self.operator == '>=': return val >= self.threshold
        if self.operator == '<=': return val <= self.threshold
        if self.operator == '>':  return val >  self.threshold
        if self.operator == '<':  return val <  self.threshold
        if self.operator == '==': return float(val == self.threshold)
        return False
    def __str__(self): return f"{self.variable} {self.operator} {self.threshold}"

class GTLFormula:
    def __init__(self, operator: str, sub_formulas=None):
        self.operator = operator  # 'G','F','∃','and','or','not','implies','atom','atom_global'
        self.sub_formulas = sub_formulas if sub_formulas else []
        self.temporal_bound = None
        self.node_proposition = None
        self.edge_proposition = None
        self.count_threshold = None
    def set_node_proposition(self, node_proposition):
        self.node_proposition = node_proposition; return self
    def set_edge_proposition(self, edge_proposition):
        self.edge_proposition = edge_proposition; return self
    def set_temporal_bound(self, t_start, t_end):
        self.temporal_bound = (t_start, t_end); return self
    def set_count_threshold(self, count):
        self.count_threshold = count; return self
    def evaluate(self, graph, node, time_index):
        if self.operator == 'atom':
            return self.node_proposition.evaluate(graph, node, time_index)
        elif self.operator == 'atom_global':
            var, op, thr = self.node_proposition.variable, self.node_proposition.operator, self.node_proposition.threshold
            val = graph.get_global_label(time_index, var)
            if op == '>=': return val >= thr
            if op == '<=': return val <= thr
            if op == '>':  return val >  thr
            if op == '<':  return val <  thr
            if op == '==': return float(val == thr)
            return False
        elif self.operator == 'G':
            # Past-looking, strict □[t_start, t_end] ending at 'time_index'
            t_start, t_end = self.temporal_bound
            for dt in range(t_start, t_end + 1):
                idx = time_index - dt              # look into the past
                if idx < 0 or idx >= len(graph.time_points):
                    return False                   # need full evidence
                if not self.sub_formulas[0].evaluate(graph, node, idx):
                    return False
            return True
        elif self.operator == 'F':
            t_start, t_end = self.temporal_bound
            for t in range(t_start, t_end + 1):
                if time_index + t >= len(graph.time_points): continue
                if self.sub_formulas[0].evaluate(graph, node, time_index + t): return True
            return False
        elif self.operator == '∃':
            count = 0
            for nb in graph.get_neighbors(node):
                edge = (node, nb)
                if self.edge_proposition.evaluate(graph, edge, time_index):
                    if self.sub_formulas[0].evaluate(graph, nb, time_index):
                        count += 1
                        if count >= (self.count_threshold or 1): return True
            return count >= (self.count_threshold or 1)
        elif self.operator == 'and':
            return all(sf.evaluate(graph, node, time_index) for sf in self.sub_formulas)
        elif self.operator == 'or':
            return any(sf.evaluate(graph, node, time_index) for sf in self.sub_formulas)
        elif self.operator == 'not':
            return not self.sub_formulas[0].evaluate(graph, node, time_index)
        elif self.operator == 'implies':
            return (not self.sub_formulas[0].evaluate(graph, node, time_index)) or \
                   self.sub_formulas[1].evaluate(graph, node, time_index)
        return False
    def __str__(self):
        if self.operator == 'atom': return str(self.node_proposition)
        if self.operator == 'atom_global': return f"GLOBAL[{self.node_proposition}]"
        if self.operator in ['G','F']:
            t0, t1 = self.temporal_bound; return f"{self.operator}[{t0},{t1}]({self.sub_formulas[0]})"
        if self.operator == '∃':
            return f"∃^{self.count_threshold}(○_{{{self.edge_proposition}}})({self.sub_formulas[0]})"
        if self.operator == 'and': return f"({' ∧ '.join(str(s) for s in self.sub_formulas)})"
        if self.operator == 'or':  return f"({' ∨ '.join(str(s) for s in self.sub_formulas)})"
        if self.operator == 'not': return f"¬({self.sub_formulas[0]})"
        if self.operator == 'implies': return f"({self.sub_formulas[0]} → {self.sub_formulas[1]})"
        return "undefined"

class CaGTLFormula:
    def __init__(self):
        self.cause_formula = None
        self.effect_formula = None
    def set_cause(self, f): self.cause_formula = f; return self
    def set_effect(self, f): self.effect_formula = f; return self
    def evaluate(self, graph, node, time_index):
        if not self.cause_formula.evaluate(graph, node, time_index): return True
        return self.effect_formula.evaluate(graph, node, time_index)
    def __str__(self): return f"({self.cause_formula}) → ({self.effect_formula})"

# ----------------------
# Gene network and state
# ----------------------
class GeneGraph:
    def __init__(self, num_nodes=16):
        self.num_nodes = num_nodes
        self.nodes = list(range(num_nodes))
        # 4x4 grid
        self.edges = []
        side = int(np.sqrt(num_nodes))
        for r in range(side):
            for c in range(side):
                u = r*side + c
                if c+1 < side: self.edges.append((u, u+1))
                if r+1 < side: self.edges.append((u, u+side))
        self.node_labels = {}     # (node,time) -> {...}
        self.edge_labels = {}     # ((u,v),time) -> {'conn':1}
        self.global_labels = {}   # (time) -> {'D': float}
        self.time_points = []
    def add_time_point(self, t): self.time_points.append(t)
    def set_node_label(self, node, time, label_type, value):
        self.node_labels.setdefault((node,time), {})[label_type] = value
    def set_edge_label(self, edge, time, label_type, value):
        e = tuple(sorted(edge)); self.edge_labels.setdefault((e,time), {})[label_type] = value
    def set_global_label(self, time, label_type, value):
        self.global_labels.setdefault(time, {})[label_type] = value
    def get_node_label(self, node, time, label_type='G1'):
        return self.node_labels.get((node,time), {}).get(label_type, 0.0)
    def get_edge_label(self, edge, time, label_type='conn'):
        e = tuple(sorted(edge)); return self.edge_labels.get((e,time), {}).get(label_type, 0.0)
    def get_global_label(self, time, label_type='D'):
        return self.global_labels.get(time, {}).get(label_type, 0.0)
    def get_neighbors(self, node):
        nbs = []
        for (u,v) in self.edges:
            if u == node: nbs.append(v)
            elif v == node: nbs.append(u)
        return nbs

class GeneState:
    def __init__(self, num_nodes=16):
        self.num_nodes = num_nodes
        self.genes = np.zeros((4, num_nodes), dtype=int)  # G1..G4
        for k in range(4):
            for v in range(num_nodes):
                self.genes[k, v] = 1 if random.random() < 0.25 else 0
        self.D = 60.0  # slightly higher start → harder task
    def get_state_vector(self):
        return np.concatenate([self.genes.reshape(-1), np.array([self.D], dtype=float)])
    def is_critical(self, v):
        G1,G2,G3,G4 = (self.genes[0,v], self.genes[1,v], self.genes[2,v], self.genes[3,v])
        return int(G1==1 and G2==1 and G4==1 and G3==0)
    def update_disease(self):
        # No "natural" remission to 0; add noise to slow learning and make outcomes less deterministic.
        num_crit = sum(self.is_critical(v) for v in range(self.num_nodes))
        drift = (2.5 * num_crit) - 2.0  # weak decrease when no criticals
        noise = np.random.normal(0.0, 1.25)  # noise makes credit assignment harder
        self.D = float(np.clip(self.D + drift + noise, 0.0, 100.0))
    def update_graph_labels(self, graph: GeneGraph, time_idx: int):
        if time_idx not in graph.time_points: graph.add_time_point(time_idx)
        for v in range(self.num_nodes):
            graph.set_node_label(v, time_idx, 'G1', float(self.genes[0,v]))
            graph.set_node_label(v, time_idx, 'G2', float(self.genes[1,v]))
            graph.set_node_label(v, time_idx, 'G3', float(self.genes[2,v]))
            graph.set_node_label(v, time_idx, 'G4', float(self.genes[3,v]))
            graph.set_node_label(v, time_idx, 'crit', float(self.is_critical(v)))
        for e in graph.edges:
            graph.set_edge_label(e, time_idx, 'conn', 1.0)
        graph.set_global_label(time_idx, 'D', float(self.D))

# --------------------
# STL / Effect helpers
# --------------------
def effect_holds(env) -> bool:
    if len(env._D_hist) < HORIZON + 1: return False
    window = list(env._D_hist)[:HORIZON+1]
    return (min(window) <= D_THRESH)

# ------------------------------
# GeneRegulation RL Environment
# ------------------------------
class GeneRegulationEnvironment:
    def __init__(self, num_nodes=16):
        self.state = GeneState(num_nodes=num_nodes)
        self.graph = GeneGraph(num_nodes=num_nodes)
        self.steps_taken = 0
        self.max_steps = 600  # longer episodes → slower convergence
        self.action_space = self._create_action_space(num_nodes)
        self._D_hist = deque(maxlen=256)
        self._remission_due = None
        self._ramp = None              # <— NEW: (start, end, D0)
        self.remission_min = 0.0

        # canonical CaGTL (template)
        self.canonical_formula = self._build_canonical_cagtl()

        # Default effect window a_e,b_e for φ_e
        self.eff_a, self.eff_b = 12, 15

        # --- CaSTL causal hook (ties cause+sequence → remission in [12,15]) ---
        self._causal_ctx = None      # tracks active cause episode
        self._remission_due = None   # time when remission will fire if armed
        self.seq_slack = 1           # ±1 step slack per phase

    def apply_castl_cause_only(self, params: Dict):
        """Update the canonical cause with inferred bounds and neighbor count."""
        G_start = int(params.get('G_start', 0))
        G_end   = int(params.get('G_end', 10))
        neigh_k = int(params.get('neigh_cnt', 1))

        # rebuild cause with new window and neighbor count
        def atom(var, op, thr):
            return GTLFormula('atom').set_node_proposition(GTLNodeProposition(var, op, thr))
        def OR(*subs):
            f = GTLFormula('or'); f.sub_formulas = list(subs); return f
        def AND(*subs):
            f = GTLFormula('and'); f.sub_formulas = list(subs); return f

        crit_and = AND(atom('G1','==',1), atom('G2','==',1), atom('G4','==',1), atom('G3','==',0))
        cause1 = GTLFormula('G').set_temporal_bound(G_start, G_end); cause1.sub_formulas = [crit_and]
        neigh_any_mut = OR(atom('G1','==',1), atom('G2','==',1), atom('G4','==',1))
        conn_prop = GTLEdgeProposition('conn','==',1)
        cause2 = GTLFormula('∃').set_edge_proposition(conn_prop).set_count_threshold(neigh_k)
        cause2.sub_formulas = [neigh_any_mut]
        cause = AND(cause1, cause2)

        D_atom = GTLFormula('atom_global').set_node_proposition(GTLNodeProposition('D','<=',0))
        effect = GTLFormula('F').set_temporal_bound(self.eff_a, self.eff_b); effect.sub_formulas = [D_atom]
        self.canonical_formula = CaGTLFormula().set_cause(cause).set_effect(effect)

    def _create_action_space(self, num_nodes):
        actions = []
        for v in range(num_nodes):
            for k in [1,2,3,4]:
                actions.append(f"SET_{v}_G{k}_0"); actions.append(f"SET_{v}_G{k}_1")
        return actions

    def reset(self):
        self.state = GeneState(num_nodes=self.graph.num_nodes)
        self.graph = GeneGraph(num_nodes=self.graph.num_nodes)
        self.steps_taken = 0
        self._remission_due = None
        self._ramp = None              # <— NEW
        self.state.update_graph_labels(self.graph, 0)
        self._D_hist.clear()
        m = float(self.state.D)
        for _ in range(max(self.eff_b, HORIZON)+1):
            self._D_hist.appendleft(m)
        self._causal_ctx = None
        self._remission_due = None
        return self.state.get_state_vector()

    def _maybe_start_cause_episode(self):
        """If no active episode and cause holds at some node, start tracking."""
        if self._causal_ctx is not None: return
        t = self.steps_taken
        g = self.graph
        for v in g.nodes:
            if self.canonical_formula.cause_formula.evaluate(g, v, max(0, t)):
                t0 = t
                def win(lo):  # 3-step window with slack
                    return (lo - self.seq_slack, lo + 2 + self.seq_slack)
                self._causal_ctx = {
                    'v': v,
                    't0': t0,
                    'need': {'G1': win(t0 + 0), 'G2': win(t0 + 3), 'G4': win(t0 + 6)},
                    'done': set(),
                    'deadline': t0 + 9,
                    'remission_window': (t0 + 12, t0 + 15),
                    'armed': False,
                }
                break

    def step(self, action: str):
        parts = action.split('_')
        if len(parts) == 4 and parts[0] == 'SET':
            v = int(parts[1]); gene = parts[2]; val = int(parts[3])
            k = int(gene[1])
            self.state.genes[k-1, v] = val

        # --- CaSTL causal episode tracking (sequence completion) ---
        self._maybe_start_cause_episode()

        if self._causal_ctx is not None:
            ctx = self._causal_ctx
            # match: SET_<v>_Gk_0 at correct node/time
            if len(parts) == 4 and parts[0] == 'SET':
                v_act = int(parts[1]); gene = parts[2]; val = int(parts[3])
                if v_act == ctx['v'] and val == 0 and gene in ctx['need'] and gene not in ctx['done']:
                    lo, hi = ctx['need'][gene]
                    if lo <= self.steps_taken <= hi:
                        ctx['done'].add(gene)
            # arm remission if 3/3 done before deadline
            if (not ctx['armed']) and ctx['done'] == {'G1','G2','G4'} and self.steps_taken <= ctx['deadline']:
                a, b = ctx['remission_window']
                self._remission_due = random.randint(a, b)
                ctx['armed'] = True
            # if missed deadline without arming, drop the episode
            if self.steps_taken > ctx['deadline'] and not ctx['armed']:
                self._causal_ctx = None

        # Fire remission if due now (happens at most once)
            if self._remission_due is not None and self.steps_taken >= self._remission_due:
                if self._ramp is None:
                    self._ramp = {
                        'start': max(0, self.steps_taken - 10),  # try 10–12 for smoother rise
                        'end':   self.steps_taken,
                        'D0':    float(self.state.D)
                    }
                self._remission_due = None
                self._causal_ctx = None

        # disease dynamics
        self.state.update_disease()
        if self._ramp is not None:
            s, e, D0 = self._ramp['start'], self._ramp['end'], self._ramp['D0']
            if self.steps_taken <= e:
                span = max(1, e - s + 1)
                alpha = np.clip((self.steps_taken - s + 1) / span, 0.0, 1.0)
                self.state.D = max(self.remission_min, (1.0 - alpha) * D0)
            else:
                self._ramp = None

        # update graph time
        self.steps_taken += 1
        self.state.update_graph_labels(self.graph, self.steps_taken)

        # update history and reward
        self._D_hist.appendleft(float(self.state.D))
        rob_e = stl_effect_robustness_D_eventually(self._D_hist, 12, 15, D_THRESH)
        r_bin = 1.0 if rob_e > 0 else 0.0  # very sparse reward
        done = (self.steps_taken >= self.max_steps) or (self.state.D <= 0.0)
        info = {'robust_effect': rob_e}
        return self.state.get_state_vector(), r_bin, done, info

    # Build the default CaGTL formula (will be overwritten by infer)
    def _build_canonical_cagtl(self) -> 'CaGTLFormula':
        def atom(var, op, thr):
            return GTLFormula('atom').set_node_proposition(GTLNodeProposition(var, op, thr))
        def OR(*subs):
            f = GTLFormula('or'); f.sub_formulas = list(subs); return f
        def AND(*subs):
            f = GTLFormula('and'); f.sub_formulas = list(subs); return f
        crit_and = AND(atom('G1','==',1), atom('G2','==',1), atom('G4','==',1), atom('G3','==',0))
        cause1 = GTLFormula('G').set_temporal_bound(0,10); cause1.sub_formulas = [crit_and]
        neigh_any_mut = OR(atom('G1','==',1), atom('G2','==',1), atom('G4','==',1))
        conn_prop = GTLEdgeProposition('conn','==',1)
        cause2 = GTLFormula('∃').set_edge_proposition(conn_prop).set_count_threshold(1); cause2.sub_formulas = [neigh_any_mut]
        cause = AND(cause1, cause2)
        D_atom = GTLFormula('atom_global').set_node_proposition(GTLNodeProposition('D','<=',0))
        effect = GTLFormula('F').set_temporal_bound(12,15); effect.sub_formulas = [D_atom]
        return CaGTLFormula().set_cause(cause).set_effect(effect)

# =============================================
#  CaSTL Inference (very light BO/random search)
# =============================================
def set_curriculum(agent, ep, eps0=0.95, eps_min=0.25, decay=0.998):
    agent.epsilon = max(eps_min, eps0 * (decay ** ep))


class CaSTLInferencer:
    """Search over cause window end G_end and neighbor count k."""
    def __init__(self, env_maker: Callable[[], GeneRegulationEnvironment], eval_rollouts: int = 4, seed: int = SEED):
        self.env_maker = env_maker
        self.eval_rollouts = eval_rollouts
        random.seed(seed); np.random.seed(seed)

        # For GP
        if _HAS_SK:
            # Smoother, robust kernel with noise; wider bounds
            kernel = (
                C(1.0, (1e-2, 1e3)) *
                Matern(length_scale=1.0, nu=2.5, length_scale_bounds=(1e-2, 100.0))
                + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-6, 1.0))
            )
            self.gp = GaussianProcessRegressor(
                kernel=kernel,
                n_restarts_optimizer=8,
                alpha=1e-3,          # small nugget helps optimizer
                normalize_y=True,
                random_state=seed
            )
            self._scaler = StandardScaler()
        else:
            self.gp = None
            self._scaler = None

        self.X, self.y = [], []

    def _simulate_sufficiency(self, G_end: int, neigh_cnt: int) -> float:
        scores = []
        for _ in range(self.eval_rollouts):
            env = self.env_maker()
            env.apply_castl_cause_only({'G_start':0, 'G_end':G_end, 'neigh_cnt':neigh_cnt})
            s = env.reset()
            # try to ensure cause holds at v=0 by setting genes to (1,1,0,1) for a short while
            v = 0
            for _t in range(12):
                for (g,val) in [(1,1),(2,1),(3,0),(4,1)]:
                    s, _, done, _ = env.step(f"SET_{v}_G{g}_{val}")
                    if done: break
                if done: break
            # execute sequence G1→G2→G4 to zero with approximate timing
            for k in range(11):
                gene = 1 if k<4 else (2 if k<8 else 4)
                s, _, done, _ = env.step(f"SET_{v}_G{gene}_0")
                if done: break
            # roll a bit; measure effect robustness
            for _ in range(20):
                s, _, done, _ = env.step(random.choice(env.action_space))
                if done: break
            rho = stl_effect_robustness_D_eventually(env._D_hist, 12, 15, D_THRESH)
            scores.append(rho)
        return float(np.mean(scores))

    def _simulate_necessity(self, G_end: int, neigh_cnt: int) -> float:
        scores = []
        for _ in range(self.eval_rollouts):
            env = self.env_maker()
            env.apply_castl_cause_only({'G_start':0, 'G_end':G_end, 'neigh_cnt':neigh_cnt})
            s = env.reset()
            v = 0
            # actively break the cause by flipping at least one gene and setting G3=1 repeatedly
            for _t in range(18):
                s, _, done, _ = env.step(f"SET_{v}_G1_0")
                s, _, done, _ = env.step(f"SET_{v}_G3_1")
                if done: break
            for _ in range(30):
                s, _, done, _ = env.step(random.choice(env.action_space))
                if done: break
            rho = stl_effect_robustness_D_eventually(env._D_hist, 12, 15, D_THRESH)
            scores.append(rho)
        return float(-np.mean(scores))  # smaller effect when cause false → larger (negated) score

    def objective(self, x: np.ndarray) -> float:
        G_end = int(round(x[0])); neigh_cnt = int(round(x[1]))
        S = self._simulate_sufficiency(G_end, neigh_cnt)
        N = self._simulate_necessity(G_end, neigh_cnt)
        return S + N

    def infer(self, init_evals: int = 8, bo_iters: int = 16,
              G_end_bounds=(7, 12), neigh_bounds=(1, 3)):
        # random init
        for _ in range(init_evals):
            xe = np.array([random.randint(*G_end_bounds),
                           random.randint(*neigh_bounds)], dtype=float)
            y = self.objective(xe)
            self.X.append(xe); self.y.append(y)
        # BO / random loop
        for it in range(bo_iters):
            if _HAS_SK and len(self.X) >= 3:
                X_arr = np.array(self.X, dtype=float)
                y_arr = np.array(self.y, dtype=float)
                # scale inputs for GP
                Xs = self._scaler.fit_transform(X_arr)
                try:
                    self.gp.fit(Xs, y_arr)
                except Exception:
                    # optimizer hiccup → skip GP this round
                    pass
                cand = []
                for _ in range(60):
                    x = np.array([random.randint(*G_end_bounds),
                                  random.randint(*neigh_bounds)], dtype=float)
                    xs = self._scaler.transform(x.reshape(1, -1))
                    mu, std = self.gp.predict(xs, return_std=True)
                    mu = float(np.asarray(mu).reshape(-1)[0])
                    std = float(np.asarray(std).reshape(-1)[0])
                    ucb = mu + 1.5 * std
                    cand.append((ucb, x))
                _, x_next = max(cand, key=lambda t: t[0])
            else:
                x_next = np.array([random.randint(*G_end_bounds),
                                   random.randint(*neigh_bounds)], dtype=float)
            y_next = self.objective(x_next)
            self.X.append(x_next); self.y.append(y_next)
        i_best = int(np.argmax(self.y))
        x_best = np.array(self.X[i_best])
        return {'G_start': 0, 'G_end': int(round(x_best[0])), 'neigh_cnt': int(round(x_best[1])),
                'score': float(self.y[i_best])}

# ---------------------------------
# Q-learning agents
# ---------------------------------
class MultiGeneRL:
    def __init__(self, env: GeneRegulationEnvironment, learning_rate=0.005, discount_factor=0.99, epsilon=0.4):
        self.env = env
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.Q: Dict[Tuple, Dict[str, float]] = {}
    def get_best_action(self, state):
        key = tuple(state)
        if key not in self.Q:
            self.Q[key] = {a: 0.0 for a in self.env.action_space}
        return max(self.Q[key].items(), key=lambda kv: kv[1])[0]
    def get_action(self, state):
        return random.choice(self.env.action_space) if random.random() < self.epsilon else self.get_best_action(state)
    def update(self, state, action, reward, next_state):
        s, sp = tuple(state), tuple(next_state)
        if s  not in self.Q:  self.Q[s]  = {a: 0.0 for a in self.env.action_space}
        if sp not in self.Q:  self.Q[sp] = {a: 0.0 for a in self.env.action_space}
        target = reward + self.gamma * max(self.Q[sp].values())
        self.Q[s][action] = (1 - self.lr) * self.Q[s][action] + self.lr * target

class CounterfactualRL(MultiGeneRL):
    def get_counterfactual_reward(self, env: GeneRegulationEnvironment):
        # mild shaping; negative baseline proportional to D
        return -0.02 * env.state.D
    def update(self, state, action, reward, next_state):
        shaped = reward - self.get_counterfactual_reward(self.env)
        super().update(state, action, shaped, next_state)

class GTLCaSTLAgent(MultiGeneRL):
    """Uses only causality knowledge (inferred cause) to propose timed sequence at active cause nodes."""
    def __init__(self, env: GeneRegulationEnvironment, learning_rate=0.005, discount_factor=0.99, epsilon=0.45):
        super().__init__(env, learning_rate, discount_factor, epsilon)
        self.epsilon_decay = 0.9995   # slow decay → slower convergence
        self.min_epsilon   = 0.10
        self.steps = 0
    def get_action(self, state):
        # With (1-epsilon) probability, try to exploit causal timing when a cause episode is active or about to start.
        if random.random() > self.epsilon:
            t = self.env.steps_taken
            g = self.env.graph
            # If an episode is active, follow the schedule; else look for a node where cause holds now.
            if self.env._causal_ctx is not None:
                v = self.env._causal_ctx['v']
                phase = (t - self.env._causal_ctx['t0'])
                if phase < 3:   return f"SET_{v}_G1_0"
                elif phase < 6: return f"SET_{v}_G2_0"
                elif phase < 9: return f"SET_{v}_G4_0"
            else:
                for v in g.nodes:
                    if self.env.canonical_formula.cause_formula.evaluate(g, v, max(0, t)):
                        mod = (t % 9)
                        if mod < 3:   return f"SET_{v}_G1_0"
                        elif mod < 6: return f"SET_{v}_G2_0"
                        else:         return f"SET_{v}_G4_0"
        return super().get_action(state)
    def update(self, state, action, reward, next_state):
        super().update(state, action, reward, next_state)
        self.steps += 1
        if self.steps % 1000 == 0:
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

# ----------------
# Training / plots
# ----------------
def episode_success(env: GeneRegulationEnvironment):
    return float(effect_holds(env))

def compare_agents(num_episodes=900, max_steps=180, cause_params=None):
    # Prepare agents
    logs = {k: {'reward': [], 'success': [], 'effect_rate': []}
            for k in ['baseline','counterfactual','GTL-CaSTL']}

    # Baseline
    print("\nTraining baseline …")
    env_b = GeneRegulationEnvironment(num_nodes=16)
    if cause_params is not None: env_b.apply_castl_cause_only(cause_params)
    agent_b = MultiGeneRL(env_b, learning_rate=0.005, epsilon=0.45)
    for ep in range(num_episodes):
        s, ep_ret = env_b.reset(), 0.0
        set_curriculum(agent_b, ep, eps0=0.95, eps_min=0.25, decay=0.999)  # <-- add
        eff_sum = 0.0; eff_cnt = 0; steps_this_ep = 0
        for _ in range(max_steps):
            a = agent_b.get_action(s)
            sp, r, d, info = env_b.step(a)
            agent_b.update(s, a, r, sp)
            s = sp; ep_ret += r
            steps_this_ep += 1
            if steps_this_ep > HORIZON:
                eff_sum += (1.0 if info['robust_effect']>0 else 0.0); eff_cnt += 1
            if d: break
        logs['baseline']['reward'].append(ep_ret)
        logs['baseline']['success'].append( episode_success(env_b) )
        logs['baseline']['effect_rate'].append( eff_sum / max(1, eff_cnt) )
        if (ep + 1) % 200 == 0:
            r50 = np.mean(logs['baseline']['reward'] [-50:])
            s50 = np.mean(logs['baseline']['success'][-50:])
            e50 = np.mean(logs['baseline']['effect_rate'][-50:])
            print(f"  ep {ep+1:4d}:  R̄₅₀ = {r50:7.2f}   S̄₅₀ = {s50:5.2f}   effect₅₀ = {e50:5.2f}")
    sr = np.array(logs['baseline']['success'])
    print(f"→ baseline    success-rate: mean={sr.mean():.3f} min={sr.min():.0f} max={sr.max():.0f}")

    # Counterfactual
    print("\nTraining counterfactual …")
    env_c = GeneRegulationEnvironment(num_nodes=16)
    if cause_params is not None: env_c.apply_castl_cause_only(cause_params)
    agent_c = CounterfactualRL(env_c, learning_rate=0.005, epsilon=0.45)
    for ep in range(num_episodes):
        s, ep_ret = env_c.reset(), 0.0
        set_curriculum(agent_c, ep, eps0=0.95, eps_min=0.20, decay=0.9992)  # <- agent_c
        eff_sum = 0.0; eff_cnt = 0; steps_this_ep = 0
        for _ in range(max_steps):
            a = agent_c.get_action(s)
            sp, r, d, info = env_c.step(a)
            agent_c.update(s, a, r, sp)
            s = sp; ep_ret += r
            steps_this_ep += 1
            if steps_this_ep > HORIZON:
                eff_sum += (1.0 if info['robust_effect']>0 else 0.0); eff_cnt += 1
            if d: break
        logs['counterfactual']['reward'].append(ep_ret)
        logs['counterfactual']['success'].append( episode_success(env_c) )
        logs['counterfactual']['effect_rate'].append( eff_sum / max(1, eff_cnt) )
        if (ep + 1) % 200 == 0:
            r50 = np.mean(logs['counterfactual']['reward'] [-50:])
            s50 = np.mean(logs['counterfactual']['success'][-50:])
            e50 = np.mean(logs['counterfactual']['effect_rate'][-50:])
            print(f"  ep {ep+1:4d}:  R̄₅₀ = {r50:7.2f}   S̄₅₀ = {s50:5.2f}   effect₅₀ = {e50:5.2f}")
    sr = np.array(logs['counterfactual']['success'])
    print(f"→ counterfactual  success-rate: mean={sr.mean():.3f} min={sr.min():.0f} max={sr.max():.0f}")

    # GTL-CaSTL
    print("\nTraining GTL-CaSTL …")
    env_g = GeneRegulationEnvironment(num_nodes=16)
    if cause_params is not None: env_g.apply_castl_cause_only(cause_params)
    agent_g = GTLCaSTLAgent(env_g, learning_rate=0.005, epsilon=0.5)
    for ep in range(num_episodes):
        s, ep_ret = env_g.reset(), 0.0
        set_curriculum(agent_g, ep, eps0=0.95, eps_min=0.10, decay=0.9992)  # <- agent_g
        eff_sum = 0.0; eff_cnt = 0; steps_this_ep = 0
        for _ in range(max_steps):
            a = agent_g.get_action(s)
            sp, r, d, info = env_g.step(a)
            agent_g.update(s, a, r, sp)
            s = sp; ep_ret += r
            steps_this_ep += 1
            if steps_this_ep > HORIZON:
                eff_sum += (1.0 if info['robust_effect']>0 else 0.0); eff_cnt += 1
            if d: break
        logs['GTL-CaSTL']['reward'].append(ep_ret)
        logs['GTL-CaSTL']['success'].append( episode_success(env_g) )
        logs['GTL-CaSTL']['effect_rate'].append( eff_sum / max(1, eff_cnt) )
        if (ep + 1) % 200 == 0:
            r50 = np.mean(logs['GTL-CaSTL']['reward'] [-50:])
            s50 = np.mean(logs['GTL-CaSTL']['success'][-50:])
            e50 = np.mean(logs['GTL-CaSTL']['effect_rate'][-50:])
            print(f"  ep {ep+1:4d}:  R̄₅₀ = {r50:7.2f}   S̄₅₀ = {s50:5.2f}   effect₅₀ = {e50:5.2f}")
    sr = np.array(logs['GTL-CaSTL']['success'])
    print(f"→ GTL-CaSTL   success-rate: mean={sr.mean():.3f} min={sr.min():.0f} max={sr.max():.0f}")

    return logs

def plot_metric(metric_dict, key, title, ylabel, fname,
                colors=None, labels=None, window=100, start_at_zero=True):
    def MA_warmstart(x, window, start_at_zero):
        x = np.asarray(x, float)
        if start_at_zero:
            # no zero padding – just seed EMA at the first value
            x = np.concatenate([[x[0] if len(x) else 0.0], x])
        alpha = 2.0 / (window + 1.0)   # try window=120–150
        y, m = [], x[0] if len(x) else 0.0
        for v in x:
            m = alpha * v + (1 - alpha) * m
            y.append(m)
        return np.clip(np.array(y), 0, 1)
    colors = colors or {'baseline':'tab:blue', 'counterfactual':'tab:orange', 'GTL-CaSTL':'tab:red'}
    labels = labels or {'baseline':'Baseline Q', 'counterfactual':'Counterfactual Q', 'GTL-CaSTL':'GTL-CaSTL'}
    plt.figure(figsize=(12,8))
    for agent in metric_dict.keys():
        y = MA_warmstart(metric_dict[agent][key], window, start_at_zero)
        plt.plot(np.arange(len(y)), y, lw=3, label=labels.get(agent, agent))
    plt.xlabel('Episode'); plt.ylabel(ylabel); plt.title(title)
    plt.ylim(0, 1.0); plt.grid(alpha=.3); plt.legend(); plt.tight_layout()
    plt.savefig(fname, dpi=300); plt.close()

def main():
    # 1) Infer CaSTL cause params
    infer = CaSTLInferencer(lambda: GeneRegulationEnvironment(num_nodes=16), eval_rollouts=3)
    best_cause = infer.infer(init_evals=6, bo_iters=10, G_end_bounds=(7,12), neigh_bounds=(1,3))
    print("Best CaSTL *cause* params:", best_cause)

    # 2) Train/compare (slower convergence settings baked into agents/env)
    logs = compare_agents(num_episodes=1500, max_steps=180, cause_params=best_cause)

    # 3) Plots
    plot_metric(logs, key='success',
                title='Success-rate convergence (Gene Regulation)',
                ylabel='Moving-average success',
                fname='gene_success_rate_three_methods.pdf',
                window=20)
    plot_metric(logs, key='effect_rate',
                title='Effect satisfaction rate (Gene Regulation)',
                ylabel='Fraction of steps with robust effect > 0',
                fname='gene_effect_success_rate.pdf',
                window=20, start_at_zero=True)

# ==== NEW: multi-run + averaged plotting =====================================

def run_once(seed: int,
             num_episodes=1000, max_steps=100,
             G_end_bounds=(7,12), neigh_bounds=(1,3)):
    # make this run reproducible
    random.seed(seed); np.random.seed(seed)

    infer = CaSTLInferencer(lambda: GeneRegulationEnvironment(num_nodes=16),
                            eval_rollouts=3, seed=seed)
    best_cause = infer.infer(init_evals=6, bo_iters=10,
                             G_end_bounds=G_end_bounds,
                             neigh_bounds=neigh_bounds)
    logs = compare_agents(num_episodes=num_episodes,
                          max_steps=max_steps,
                          cause_params=best_cause)
    return logs

def _stack_runs(all_logs, agent, key):
    # shape: (n_runs, T)
    arrs = [np.asarray(run[agent][key], dtype=float) for run in all_logs]
    T = min(a.shape[0] for a in arrs)              # align lengths just in case
    arrs = [a[:T] for a in arrs]
    return np.stack(arrs, axis=0)                  # (R, T)

def plot_metric_ensemble(all_logs, key, title, ylabel, fname,
                         window=50, start_at_zero=True,
                         colors=None, labels=None):
    colors = colors or {'baseline':'tab:blue', 'counterfactual':'tab:orange', 'GTL-CaSTL':'tab:green'}
    labels = labels or {'baseline':'Baseline Q', 'counterfactual':'Counterfactual Q', 'GTL-CaSTL':'GTL-CaSTL'}

    def MA(x, window, start_at_zero):
        x = np.asarray(x, float)
        if start_at_zero:
            # no zero padding – just seed EMA at the first value
            x = np.concatenate([[x[0] if len(x) else 0.0], x])
        alpha = 2.0 / (window + 1.0)   # try window=120–150
        y, m = [], x[0] if len(x) else 0.0
        for v in x:
            m = alpha * v + (1 - alpha) * m
            y.append(m)
        return np.clip(np.array(y), 0, 1)

    agents = ['baseline','counterfactual','GTL-CaSTL']
    plt.figure(figsize=(12,8))
    for ag in agents:
        X = _stack_runs(all_logs, ag, key)         # (R,T)
        mean = X.mean(axis=0)
        sem  = X.std(axis=0, ddof=1) / np.sqrt(X.shape[0])
        lo   = np.clip(mean - 1.96*sem, 0, 1)
        hi   = np.clip(mean + 1.96*sem, 0, 1)

        y   = MA(mean, window, start_at_zero)
        ylo = MA(lo,   window, start_at_zero)
        yhi = MA(hi,   window, start_at_zero)

        x = np.arange(len(y))
        plt.plot(x, y, lw=3, color=colors[ag], label=labels[ag])
        plt.fill_between(x, ylo, yhi, color=colors[ag], alpha=0.15, linewidth=0)

    plt.xlabel('Episode'); plt.ylabel(ylabel); plt.title(title)
    plt.ylim(0,1.0); plt.grid(alpha=.3); plt.legend(); plt.tight_layout()
    plt.savefig(fname, dpi=300); plt.close()

# ==== MODIFY main() to call the ensemble =====================================

def main():
    n_runs = 5
    base_seed = 1234

    all_logs = []
    for i in range(n_runs):
        print(f"\n==== Run {i+1}/{n_runs} (seed {base_seed+i}) ====")
        logs_i = run_once(seed=base_seed + i,
                          num_episodes=1000, max_steps=100)
        all_logs.append(logs_i)

    # Averaged plots (+ 95% CI)
    plot_metric_ensemble(all_logs, key='success',
                         title='Success-rate convergence',
                         ylabel='Moving-average success',
                         fname='gene_success_rate_three_methods_avg.pdf',
                         window=50, start_at_zero=True)

    plot_metric_ensemble(all_logs, key='effect_rate',
                         title='Effect satisfaction rate',
                         ylabel='Fraction of steps with robust effect > 0',
                         fname='gene_effect_success_rate_avg.pdf',
                         window=50, start_at_zero=True)

if __name__ == "__main__":
    main()
