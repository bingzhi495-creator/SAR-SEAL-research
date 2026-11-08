"""Collect fixed candidate alternatives without changing the native prediction stream."""


class Readout:
    def start(self, candidate, truth, matches, rles):
        native=candidate["native"]
        self.candidate=candidate
        self.truth=truth
        self.matches=matches
        self.native_rles=rles
        self.masks={"native":native,"offset":native.copy(),"coeff":native.copy(),"matched_gt":native.copy()}

    def replace(self, position, stock, truth, accepted, coefficient, offset):
        if stock["strict"]:
            return
        self.masks["matched_gt"][position]=truth
        if accepted[0][0]:
            self.masks["coeff"][position]=coefficient
        if accepted[1][0]:
            self.masks["offset"][position]=offset

