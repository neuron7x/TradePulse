# SPDX-License-Identifier: MIT
# concrete adapter for SumPort
from ..ports.ports import SumPort

class LocalSum(SumPort):
    def sum(self,a:int,b:int)->int:
        return a+b
