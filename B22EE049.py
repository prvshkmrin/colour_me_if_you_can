# B22EE049.py
from collections import defaultdict
import random

class B22EE049:
    ...
    def __init__(self, initial_state):     # <-- fix here
        self.random = random.Random(0)
        self.colors = []
        self.current = None

        # Belief over seen graph
        self.known_nodes = set()
        self.neigh = defaultdict(set)  # node -> set(neighbor)
        self.color = dict()            # node -> color or None
        self.domain = dict()           # node -> set(colors) (or {fixed})

        self._ingest(initial_state)


    # ---------------- Runner API ----------------

    def get_next_move(self, visible_state):
        self._ingest(visible_state)
        vis = self._norm(visible_state)
        V = list(vis["nodes"])

        # visible & uncolored
        U = [u for u in V if self.color.get(u) is None]

        if U:
            best = self._mrv(U)
            # stay if current is as good or better (save -1 move penalty)
            if self.color.get(self.current) is None:
                cur_dom = len(self.domain.get(self.current, set(self.colors)))
                best_dom = len(self.domain.get(best, set(self.colors)))
                if cur_dom <= best_dom:
                    return {"action": "move", "node": self.current}  # 'stay' by moving to current
            return {"action": "move", "node": (best if best != self.current else self.current)}

        # ---- EXPLORATION: no uncoloured node visible -> move to expand frontier ----
        # Frontier score: prefer nodes that likely reveal progress (unknown/uncoloured neighbours).
        # Break ties preferring NOT current, then higher degree, then name.
        def frontier_score(n):
            return self._frontier_score(n)

        # Rank candidates
        ranked = sorted(
            V,
            key=lambda n: (
                -frontier_score(n),
                0 if n != self.current else 1,   # prefer leaving current if tied
                -len(self.neigh.get(n, [])),     # degree
                str(n)
            )
        )
        target = ranked[0] if ranked else self.current
        return {"action": "move", "node": target}

    def get_color_for_node(self, arg1, arg2=None):
        # Accept (visible_state,node) or (node,visible_state) or (node)
        visible_state = None
        node_to_color = None

        if isinstance(arg1, dict) and (arg2 is None or isinstance(arg2, (str, type(None)))):
            visible_state = arg1
            node_to_color = arg2 if isinstance(arg2, str) else (arg1.get("current_node") or self.current)
        elif isinstance(arg1, str) and (arg2 is None or isinstance(arg2, dict)):
            node_to_color = arg1
            visible_state = arg2
        else:
            if isinstance(arg1, dict): visible_state = arg1
            if isinstance(arg1, str):  node_to_color = arg1
            if isinstance(arg2, dict): visible_state = arg2
            if isinstance(arg2, str):  node_to_color = arg2

        if not isinstance(visible_state, dict):
            visible_state = self._synthetic_visible_state(node_to_color or self.current)

        self._ingest(visible_state)
        target = node_to_color or self.current

        # If pre-colored, echo it
        if self.color.get(target) is not None:
            return {"action": "color", "node": target, "color": self.color[target]}

        vis = self._norm(visible_state)
        V = set(vis["nodes"]); V.add(target)
        N = {u: [v for v in self.neigh[u] if v in V] for u in V}

        fixed = {u: c for u, c in self.color.items() if u in V and c is not None}
        D = {u: (set([fixed[u]]) if u in fixed else set(self.domain.get(u, set(self.colors)))) for u in V}

        self._forward_prune_all(N, D)

        ok, sol = self._backtrack(dict(fixed), D, N)
        if ok and target in sol and sol[target] is not None:
            c = sol[target]
            self._commit(target, c)
            return {"action": "color", "node": target, "color": c}

        # Fallbacks
        legal = [v for v in D.get(target, set(self.colors)) if self._consistent(target, v, fixed, N)]
        if legal:
            c = self._lcv(target, legal, D, N, fixed)
        else:
            candidates = (D.get(target) or set(self.colors))
            scored = []
            for v in candidates:
                conflicts = sum(1 for nb in N.get(target, []) if fixed.get(nb) == v or self.color.get(nb) == v)
                scored.append((conflicts, v))
            scored.sort()
            c = scored[0][1] if scored else self.colors[0]
        self._commit(target, c)
        return {"action": "color", "node": target, "color": c}

    # -------- Observation normalization & belief --------

    def _synthetic_visible_state(self, focus_node):
        if focus_node is None and self.current:
            focus_node = self.current
        nodes = list(self.known_nodes) if self.known_nodes else ([focus_node] if focus_node else [])
        return {
            "current_node": focus_node or self.current,
            "available_colors": list(self.colors),
            "visible_nodes": nodes,
            "neighbors": {u: list(self.neigh[u]) for u in nodes},
            "node_colors": {u: self.color.get(u) for u in nodes},
            "visible_edges": [(u, v) for u in nodes for v in self.neigh.get(u, []) if v in nodes],
        }

    def _norm(self, obs):
        cur = obs.get("current_node") or obs.get("agent_node")
        avail = obs.get("available_colors") or obs.get("colors") or self.colors or []

        nodes, edges = None, []

        if "visible_nodes" in obs:
            nodes = list(obs["visible_nodes"])
        elif "visible_graph" in obs and isinstance(obs["visible_graph"], dict):
            vg = obs["visible_graph"]
            nodes = list(vg.get("nodes", []))
            edges = [tuple(e) for e in vg.get("edges", [])]
        else:
            if isinstance(obs.get("neighbors"), dict):
                nodes = set(obs["neighbors"].keys())
                for u, nbs in obs["neighbors"].items():
                    for v in nbs:
                        nodes.add(v)
                        edges.append((u, v))
                nodes = list(nodes)
            else:
                nc = obs.get("node_colors", {}) or {}
                nodes = list(nc.keys()) if nc else ([cur] if cur else [])

        neighbors = None
        if isinstance(obs.get("neighbors"), dict):
            neighbors = {u: list(set(vs)) for u, vs in obs["neighbors"].items()}

        node_colors = obs.get("node_colors") or obs.get("visible_node_colors") or {}

        return {
            "current": cur,
            "nodes": nodes or ([] if cur is None else [cur]),
            "edges": edges,
            "neighbors": neighbors,
            "node_colors": node_colors,
            "available_colors": list(avail),
        }

    def _ingest(self, obs):
        vis = self._norm(obs)
        self.current = vis["current"]
        if not self.colors:
            self.colors = list(vis["available_colors"])

        for u in vis["nodes"]:
            self._ensure(u)

        if vis["neighbors"] is not None:
            for u, nbs in vis["neighbors"].items():
                self._ensure(u)
                for v in nbs:
                    self._ensure(v)
                    self.neigh[u].add(v)
                    self.neigh[v].add(u)

        for u, v in vis["edges"]:
            self._ensure(u); self._ensure(v)
            self.neigh[u].add(v); self.neigh[v].add(u)

        for u, c in vis["node_colors"].items():
            self._ensure(u)
            if c is not None:
                self._fix(u, c)
            else:
                if u not in self.domain or not self.domain[u]:
                    self.domain[u] = set(self.colors)
                self._prune_vs_colored_neighbors(u)

    def _ensure(self, u):
        if u in self.known_nodes:
            return
        self.known_nodes.add(u)
        self.color.setdefault(u, None)
        self.domain.setdefault(u, set(self.colors))

    def _fix(self, u, c):
        self.color[u] = c
        self.domain[u] = {c}
        for v in self.neigh.get(u, []):
            if self.color.get(v) is None:
                self.domain[v].discard(c)

    def _commit(self, u, c):
        self.color[u] = c
        self.domain[u] = {c}
        for v in self.neigh.get(u, []):
            if self.color.get(v) is None:
                self.domain[v].discard(c)

    def _prune_vs_colored_neighbors(self, u):
        if self.color.get(u) is not None:
            self.domain[u] = {self.color[u]}
            return
        pruned = set(self.domain[u])
        for nb in self.neigh.get(u, []):
            c = self.color.get(nb)
            if c is not None and c in pruned:
                pruned.discard(c)
        self.domain[u] = pruned or set(self.colors)

    # ---------------- CSP core ----------------

    def _forward_prune_all(self, neighbors, domains):
        changed = True
        while changed:
            changed = False
            for u, dom in domains.items():
                if len(dom) == 1:
                    (val,) = tuple(dom)
                    for v in neighbors.get(u, []):
                        if val in domains.get(v, set()) and len(domains[v]) > 1:
                            domains[v].discard(val)
                            changed = True

    def _mrv(self, V):
        return min(V, key=lambda u: (len(self.domain.get(u, set(self.colors))),
                                     -len(self.neigh.get(u, [])), str(u)))

    def _lcv(self, u, values, domains, neighbors, assignment):
        impacts = []
        for v in values:
            impact = 0
            for nb in neighbors.get(u, []):
                if nb in assignment:
                    continue
                if v in domains.get(nb, set()):
                    impact += 1
            impacts.append((impact, v))
        impacts.sort()
        return impacts[0][1]

    def _consistent(self, u, v, assignment, neighbors):
        for nb in neighbors.get(u, []):
            w = assignment.get(nb, None)
            if w is None:
                fixed = self.color.get(nb)
                if fixed is not None and fixed == v:
                    return False
            else:
                if w == v:
                    return False
        return True

    def _backtrack(self, assignment, domains, neighbors):
        # Done if every variable is assigned or singleton
        if all(u in assignment or (len(domains[u]) == 1 and next(iter(domains[u])) is not None)
               for u in domains):
            final = dict(assignment)
            for u, dom in domains.items():
                if u not in final and len(dom) == 1:
                    final[u] = next(iter(dom))
            return True, final

        unassigned = [u for u in domains if u not in assignment]
        if not unassigned:
            return True, dict(assignment)

        u = min(unassigned, key=lambda x: (len(domains[x]), -len(neighbors.get(x, [])), str(x)))

        values = [v for v in list(domains[u]) if self._consistent(u, v, assignment, neighbors)]
        if not values:
            return False, None

        ordered = []
        for v in values:
            impact = 0
            for nb in neighbors.get(u, []):
                if nb in assignment:
                    continue
                if v in domains.get(nb, set()):
                    impact += 1
            ordered.append((impact, v))
        ordered.sort()
        values = [v for _, v in ordered]

        for v in values:
            newA = dict(assignment); newA[u] = v
            newD = {x: set(domains[x]) for x in domains}

            fail = False
            for nb in neighbors.get(u, []):
                if nb in newA and newA[nb] == v:
                    fail = True; break
                if v in newD.get(nb, set()):
                    newD[nb].discard(v)
                    if len(newD[nb]) == 0:
                        fail = True; break
            if fail:
                continue

            self._forward_prune_all(neighbors, newD)
            ok, sol = self._backtrack(newA, newD, neighbors)
            if ok:
                return True, sol
        return False, None

    # ---------------- helpers ----------------

    def _frontier_score(self, n):
        """Count neighbours of n that are unknown or uncoloured in our belief."""
        score = 0
        for nb in self.neigh.get(n, []):
            if self.color.get(nb) is None:
                score += 1
        return score

    def _count_uncolored_neighbors(self, u):
        return sum(1 for v in self.neigh.get(u, []) if self.color.get(v) is None)

    def _argmax(self, iterable, key):
        best, bestk = None, None
        for x in iterable:
            k = key(x)
            if best is None or k > bestk:
                best, bestk = x, k
        return best