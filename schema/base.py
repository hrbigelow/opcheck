class RankConstraints(object):
    def __init__(self, op):
        self.rank_maxs = {}
        self.rank_mins = {}
        self.rank_equiv = {}
        self.op = op

    def equate_ranks(self, target_sig, source_sig):
        self.rank_equiv[target_sig] = source_sig

    def add_rank_limits(self, sig, min_val, max_val):
        if min_val is not None:
            prev_min_val = self.rank_mins[sig]
            self.rank_mins[sig] = max(prev_min_val, min_val)
        if max_val is not None:
            prev_max_val = self.rank_maxs[sig]
            self.rank_maxs[sig] = min(prev_max_val, max_val)

    def rmins_inds(self):
        return { self.op.sig_indices(s): r for s, r in self.rank_mins.items() }

    def rmaxs_inds(self):
        return { self.op.sig_indices(s): r for s, r in self.rank_maxs.items() }

    def req_inds(self):
        return { op.sig_indices(sig1): op.sig_indices(sig2) for sig1, sig2 in
                self.rank_equiv.items() }

class DTypeConstraints(object):
    def __init__(self):
        self.valid = {}
        self.equiv = {}

    def add_valid(self, tensor_name, dtypes):
        self.valid[tensor_name] = tuple(dtypes)

    def add_equiv(self, target_tensor, source_tensor):
        self.equiv[target_tensor] = source_tensor

