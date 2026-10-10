"""Utilities for generating Dubins-style recovery trajectories."""

from __future__ import annotations

from math import atan2, pi
from typing import Dict, Iterable, Sequence, Tuple

import numpy as np


class DubinsRecoveryPlanner:
    """Dubins-style recovery planner with optional departure-arc skipping."""

    def __init__(
        self,
        r_start: float,
        r_end: float,
        dist_thresh: float = 5.0,
        angle_thresh_deg: float = 90.0,
    ) -> None:
        self.r_start, self.r_end = r_start, r_end
        self.dist_thresh = dist_thresh
        self.angle_thresh = np.deg2rad(angle_thresh_deg)
        self.col = {"start L": "C0", "start R": "C1", "end L": "C2", "end R": "C3"}
        self.sty = {lbl: "-" for lbl in self.col}
        self.dir = {"start L": "ccw", "start R": "cw", "end L": "ccw", "end R": "cw"}

    # Geometry helpers -------------------------------------------------
    def _center(self, pose, R, left):
        x, y, th = pose
        s = 1 if left else -1
        return np.array([x - s * R * np.sin(th), y + s * R * np.cos(th)])

    @staticmethod
    def _angle(p, c):
        return atan2(p[1] - c[1], p[0] - c[0])

    @staticmethod
    def _angdiff(a, b):
        return (a - b + pi) % (2 * pi) - pi

    def _unit_tan(self, p, c, ccw):
        r = (p - c) / np.linalg.norm(p - c)
        return np.array([-r[1], r[0]]) if ccw else np.array([r[1], -r[0]])

    # Circles -----------------------------------------------------------
    def _circles(self, start, end):
        return {
            "start L": (self._center(start, self.r_start, True), self.r_start),
            "start R": (self._center(start, self.r_start, False), self.r_start),
            "end L": (self._center(end, self.r_end, True), self.r_end),
            "end R": (self._center(end, self.r_end, False), self.r_end),
        }

    # Tangents ----------------------------------------------------------
    def _tangents(self, c1, r1, c2, r2):
        d = c2 - c1
        D = np.linalg.norm(d)
        if D < 1e-9:
            return []
        u = d / D
        v = np.array([-u[1], u[0]])
        out = []
        for s in (-1, 1):
            r = (r1 - s * r2) / D
            disc = 1 - r**2
            if disc < 0:
                continue
            h = np.sqrt(disc)
            for t in (-1, 1):
                dvec = r * u + t * h * v
                out.append((c1 + r1 * dvec, c2 + s * r2 * dvec))
        return out

    def _forward(self, lbl1, p1, c1, lbl2, p2, c2):
        if np.allclose(p1, p2, atol=1e-9):
            return False
        line = (p2 - p1) / np.linalg.norm(p2 - p1)
        t1 = self._unit_tan(p1, c1, self.dir[lbl1] == "ccw")
        t2 = self._unit_tan(p2, c2, self.dir[lbl2] == "ccw")
        return (t1 @ line > 0) and (t2 @ line > 0)

    # Arc helpers ------------------------------------------------------
    def _arc_with_yaw(self, c, r, p_from, p_to, ccw, n=80):
        a, b = self._angle(p_from, c), self._angle(p_to, c)
        if ccw and b <= a:
            b += 2 * pi
        if (not ccw) and b >= a:
            b -= 2 * pi
        ang = np.linspace(a, b, n)
        x = c[0] + r * np.cos(ang)
        y = c[1] + r * np.sin(ang)
        yaw = ang + pi / 2 if ccw else ang - pi / 2
        yaw = (yaw + pi) % (2 * pi) - pi
        return np.column_stack([x, y, yaw])

    def _arc(self, c, r, p_from, p_to, ccw, n=80):
        return self._arc_with_yaw(c, r, p_from, p_to, ccw, n)[:, :2]

    def _arc_pts(self, c, r, p_from, p_to, ccw, n=80):
        return self._arc(c, r, p_from, p_to, ccw, n)

    # Path generation --------------------------------------------------
    def paths_with_yaw(self, start, end):
        circ = self._circles(start, end)
        sp = np.array(start[:2])
        start_yaw = start[2]
        ep = np.array(end[:2])
        end_yaw = end[2]
        heading_vec = np.array([np.cos(start[2]), np.sin(start[2])])
        combos = [("start L", "end L"), ("start L", "end R"), ("start R", "end L"), ("start R", "end R")]
        results: Dict[Tuple[str, str], dict] = {}
        for s_lbl, e_lbl in combos:
            c_s, r_s = circ[s_lbl]
            c_e, r_e = circ[e_lbl]
            feasible = []
            for p1, p2 in self._tangents(c_s, r_s, c_e, r_e):
                if self._forward(s_lbl, p1, c_s, e_lbl, p2, c_e):
                    feasible.append((np.linalg.norm(p1 - sp), p1, p2))
            if not feasible:
                continue
            skip_ok, other = [], []
            for dist, p1, p2 in feasible:
                tan_dir = (p2 - p1) / np.linalg.norm(p2 - p1)
                ang = np.arccos(np.clip(heading_vec @ tan_dir, -1.0, 1.0))
                if dist <= self.dist_thresh and ang <= self.angle_thresh:
                    skip_ok.append((dist, p1, p2))
                else:
                    other.append((dist, p1, p2))
            if skip_ok:
                _, p1, p2 = min(skip_ok, key=lambda t: t[0])
                skip = True
            elif other:
                _, p1, p2 = min(other, key=lambda t: t[0])
                skip = False
            else:
                continue
            if skip:
                line_yaw = np.arctan2(p2[1] - sp[1], p2[0] - sp[0])
                seg = np.array([[sp[0], sp[1], start_yaw], [p2[0], p2[1], line_yaw]])
                arc2 = self._arc_with_yaw(c_e, r_e, p2, ep, self.dir[e_lbl] == "ccw")
                arc2[-1, 2] = end_yaw
                path = np.vstack([seg, arc2])
            else:
                arc1 = self._arc_with_yaw(c_s, r_s, sp, p1, self.dir[s_lbl] == "ccw")
                arc1[0, 2] = start_yaw
                line_yaw = np.arctan2(p2[1] - p1[1], p2[0] - p1[0])
                seg = np.array([[p1[0], p1[1], line_yaw], [p2[0], p2[1], line_yaw]])
                arc2 = self._arc_with_yaw(c_e, r_e, p2, ep, self.dir[e_lbl] == "ccw")
                arc2[-1, 2] = end_yaw
                path = np.vstack([arc1, seg, arc2])
            results[(s_lbl, e_lbl)] = {"path": path, "p1": p1, "p2": p2, "skip": skip}
        return results

    def paths(self, start, end):
        paths_3d = self.paths_with_yaw(start, end)
        results = {}
        for key, data in paths_3d.items():
            results[key] = {
                "path": data["path"][:, :2],
                "p1": data["p1"],
                "p2": data["p2"],
                "skip": data["skip"],
            }
        return results

    # Plotting helpers (debug) ----------------------------------------
    def plot(self, start, end):
        data = self.paths(start, end)
        circ = self._circles(start, end)
        combos = [("start L", "end L"), ("start L", "end R"), ("start R", "end L"), ("start R", "end R")]
        titles = ["L→L", "L→R", "R→L", "R→R"]
        theta = np.linspace(0, 2 * pi, 181)
        arrow_scale = 0.5 * max(self.r_start, self.r_end)
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        for ax, (s_lbl, e_lbl), title in zip(axes.flat, combos, titles):
            for lbl in (s_lbl, e_lbl):
                c, r = circ[lbl]
                if lbl == s_lbl and (s_lbl, e_lbl) in data and data[(s_lbl, e_lbl)]["skip"]:
                    col = "0.8"
                else:
                    col = self.col[lbl]
                ax.plot(c[0] + r * np.cos(theta), c[1] + r * np.sin(theta), self.sty[lbl], color=col)
            if (s_lbl, e_lbl) in data:
                d = data[(s_lbl, e_lbl)]
                ax.plot([d["p1"][0], d["p2"][0]], [d["p1"][1], d["p2"][1]], "--", color="0.5", lw=1)
                ax.plot(d["path"][:, 0], d["path"][:, 1], color=self.col[s_lbl], lw=2)
                if d["skip"]:
                    ax.text(0.05, 0.9, "skipped", transform=ax.transAxes, fontsize=8, color=self.col[s_lbl])
            for pose, m in ((start, "o"), (end, "x")):
                x, y, th = pose
                ax.scatter(x, y, marker=m, color="k", zorder=5)
                ax.quiver(
                    x,
                    y,
                    np.cos(th),
                    np.sin(th),
                    angles="xy",
                    scale_units="xy",
                    scale=1.0 / arrow_scale,
                    color="k",
                    width=0.01,
                    zorder=6,
                )
            ax.set_title(title)
            ax.set_aspect("equal")
            ax.grid(True)
        fig.suptitle("Rectified paths")
        plt.show()

    def shortest_path(self, start, end):
        data = self.paths(start, end)
        if not data:
            return None
        key = min(data, key=lambda k: poly_length(data[k]["path"]))
        return key, data[key]

    def shortest_path_with_yaw(self, start, end):
        data = self.paths_with_yaw(start, end)
        if not data:
            return None
        key = min(data, key=lambda k: poly_length(data[k]["path"]))
        return key, data[key]


def poly_length(pts: np.ndarray) -> float:
    """Sum of segment lengths for a (N,2) or (N,3) poly-line."""
    return float(np.sum(np.linalg.norm(np.diff(pts[:, :2], axis=0), axis=1)))


def Dubins_find_best_radii(
    start_pose,
    end_pose,
    cand_r_start: Sequence[float] | np.ndarray,
    cand_r_end: Sequence[float] | np.ndarray,
    dist_thresh: float = 0.5,
    angle_thresh_deg: float = 15,
    n_jobs: int = -1,
    return_yaw: bool = False,
    reverse: bool = True,
):
    """Brute-force search over radius pairs (optionally considering reversed poses)."""

    from joblib import Parallel, delayed

    cand_r_start = np.asarray(list(cand_r_start), float)
    cand_r_end = np.asarray(list(cand_r_end), float)

    if reverse:
        def _reverse_pose(p):
            x, y, th = p
            return (x, y, (th + pi) % (2 * pi))

        start_rev = _reverse_pose(start_pose)
        end_rev = _reverse_pose(end_pose)
    else:
        start_rev = start_pose
        end_rev = end_pose

    def _evaluate(rs, re):
        out = []
        for rev_flag, s, e in ((False, start_pose, end_pose), (True, start_rev, end_rev)):
            plan = DubinsRecoveryPlanner(rs, re, dist_thresh=dist_thresh, angle_thresh_deg=angle_thresh_deg)
            shortest = plan.shortest_path_with_yaw(s, e) if return_yaw else plan.shortest_path(s, e)
            if shortest is None:
                continue
            _, data = shortest
            length = poly_length(data["path"])
            out.append((length, rev_flag, shortest, data))
        if not out:
            return np.inf, None
        return min(out, key=lambda t: t[0])

    pairs = [(rs, re) for rs in cand_r_start for re in cand_r_end]
    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_evaluate)(rs, re) for rs, re in pairs
    )

    clean = []
    for (length, rev, shortest, data), pair in zip(results, pairs):
        if shortest is None:
            continue
        clean.append((length, rev, pair[0], pair[1], shortest[0], data))

    if not clean:
        raise RuntimeError("No feasible path for any radius pair!")

    best_len, rev, rs, re, (combo_key, _), data = min(clean, key=lambda x: x[0])
    path = data["path"]
    if rev and return_yaw and path.shape[1] >= 3:
        path = path.copy()
        path[:, 2] = (path[:, 2] + pi) % (2 * pi)

    return {
        "r_start": rs,
        "r_end": re,
        "combo": combo_key,
        "reversed": rev,
        "length": best_len,
        "path": path,
    }


def resample_by_speed(poly: np.ndarray, speed: float | Sequence[float] | np.ndarray, dt: float) -> np.ndarray:
    """Resample a poly-line (with optional yaw) according to a speed profile."""

    poly = np.asarray(poly, float)
    has_yaw = poly.shape[1] >= 3

    if np.ndim(speed) == 0:
        step = float(speed) * dt
        speeds = None
    else:
        speeds = np.asarray(speed, float)
        if len(speeds) != len(poly):
            raise ValueError("speed array length must match poly length")
        step = None

    seg_len = np.linalg.norm(np.diff(poly[:, :2], axis=0), axis=1)
    cum = np.concatenate(([0.0], np.cumsum(seg_len)))
    total = cum[-1]

    if step:
        targets = np.arange(0.0, total, step)[1:]
    else:
        s = [0.0]
        while True:
            idx = np.searchsorted(cum, s[-1], side="right") - 1
            if idx >= len(seg_len):
                break
            v = speeds[idx]
            nxt = s[-1] + v * dt
            if nxt >= total:
                break
            s.append(nxt)
        targets = s[1:]

    out = [poly[0]]
    for d in targets:
        idx = np.searchsorted(cum, d, side="right") - 1
        rel = (d - cum[idx]) / seg_len[idx]
        p_xy = poly[idx, :2] * (1 - rel) + poly[idx + 1, :2] * rel
        if has_yaw:
            yaw1, yaw2 = poly[idx, 2], poly[idx + 1, 2]
            diff = yaw2 - yaw1
            if diff > pi:
                diff -= 2 * pi
            elif diff < -pi:
                diff += 2 * pi
            yaw_interp = yaw1 + rel * diff
            yaw_interp = (yaw_interp + pi) % (2 * pi) - pi
            p = np.concatenate([p_xy, [yaw_interp]])
        else:
            p = p_xy
        out.append(p)
    out.append(poly[-1])
    return np.vstack(out)


__all__ = [
    "DubinsRecoveryPlanner",
    "Dubins_find_best_radii",
    "poly_length",
    "resample_by_speed",
]

