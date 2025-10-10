from __future__ import annotations
import numpy as np
Float = np.float32

class P2Quantile:
    """P² online quantile estimator (Jain & Chlamtac, 1985)."""
    __slots__ = ("q","n","marker_positions","desired_positions","heights")
    def __init__(self, q: float):
        assert 0.0 < q < 1.0, "q in (0,1)"
        self.q = Float(q)
        self.n = 0
        self.marker_positions = np.zeros(5, dtype=Float)
        self.desired_positions = np.zeros(5, dtype=Float)
        self.heights = np.zeros(5, dtype=Float)

    def _parabolic(self, i: int, d: float) -> Float:
        hp = self.heights; mp = self.marker_positions
        num = d * (mp[i]-mp[i-1]) * (hp[i+1]-hp[i])/(mp[i+1]-mp[i]) + d * (mp[i+1]-mp[i]) * (hp[i]-hp[i-1])/(mp[i]-mp[i-1])
        return Float(hp[i] + num/(mp[i+1]-mp[i-1]))

    def _linear(self, i: int, d: int) -> Float:
        return Float(self.heights[i] + d*(self.heights[i + d] - self.heights[i])/(self.marker_positions[i + d] - self.marker_positions[i]))

    def update(self, x: float) -> float:
        x = Float(x); self.n += 1
        if self.n <= 5:
            self.heights[self.n - 1] = x
            if self.n == 5:
                self.heights.sort()
                self.marker_positions[:] = Float([1,2,3,4,5])
                self.desired_positions[:] = Float([1, 1 + 2*self.q, 1 + 4*self.q, 3 + 2*self.q, 5])
            return float(self.quantile)
        if x < self.heights[0]:
            self.heights[0] = x; k = 0
        elif x >= self.heights[4]:
            self.heights[4] = x; k = 3
        else:
            k = int(np.searchsorted(self.heights[1:4], x)) + 0
        self.marker_positions[0] += 1
        self.marker_positions[1:k+1] += 1
        self.marker_positions[4] += 1
        self.desired_positions[1] += self.q/2
        self.desired_positions[2] += self.q
        self.desired_positions[3] += (1 + self.q)/2
        self.desired_positions[4] += 1.0
        for i in (1,2,3):
            d = self.desired_positions[i] - self.marker_positions[i]
            if (d >= 1 and self.marker_positions[i+1] - self.marker_positions[i] > 1) or (d <= -1 and self.marker_positions[i-1] - self.marker_positions[i] < -1):
                d_int = 1 if d >= 1 else -1
                hp_try = self._parabolic(i, d_int)
                if self.heights[i-1] < hp_try < self.heights[i+1]:
                    self.heights[i] = hp_try
                else:
                    self.heights[i] = self._linear(i, d_int)
                self.marker_positions[i] += d_int
        return float(self.quantile)

    @property
    def quantile(self) -> float:
        if self.n == 0: return float("nan")
        if self.n <= 5:
            s = np.sort(self.heights[:self.n])
            idx = max(0, min(self.n-1, int(np.floor((self.n-1)*self.q))))
            return float(s[idx])
        return float(self.heights[2])
