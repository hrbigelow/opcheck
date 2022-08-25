from . import predicates as pr
from . import generators as ge
from .fgraph import PredNode, GenNode

Kind = pr.Kind

class SchemaApi(object):

    def __init__(self):
        self.input_pred_graph = None
        self.return_pred_graph = None
        self.gen_graph = None
        self.index = {}
        self.rank_cons = base.RankConstraints(self)
        self.dtype_cons = base.DTypeComstraits(self)

    def _init_pred_graph(self):
        PredNode.add_node(Kind.SCHEMA, lambda: self)
        PredNode.add_node(Kind.DTYPES, ValidDTypes(self.dtype_cons))
        PredNode.add_node(Kind.IRANKS, IndexRanks(self, self.rank_cons))
        PredNode.add_node(Kind.IDIMS, input_index_dims, Kind.IRANKS)

    def _init_gen_graph(self):
        GenNode.add_node(Kind.SCHEMA, lambda: self)
        GenNode.add_node(Kind.DTYPES, GenDTypes(self.dtype_cons)) 
        GenNode.add_node(Kind.IRANKS, GenRanks(self, self.rank_cons)) 
        GenNode.add_node(Kind.DIMS, GenIndexDims(), Kind.IRANKS)

    def _add_pred_graph(self):
        pred_nodes = PredNode.get_ordered_nodes()
        # TODO
        ins = [n for n in pred_nodes if True ]
        self.input_pred_graph = ins
        # self.return_pred_graph = [n for n in pred_nodes if
                # n.name.startswith(IName.RETURN) or n.name == IName.DIMS]

    def add_index(self, idx, description):
        self.index[idx] = description

    def sig_indices(self, sig):
        inds = list(self.index.keys())
        return tuple(inds.index(idx) for idx in sig)

    def constraint(self, constraint_name, pred_func, *pred_arg_names):
        cons_name = pr.name(constraint_name, Kind.CONS)
        PredNode.add_node(cons_name, pred_func, *pred_arg_names) 

    def computed_dims(self, index, comp_func, *comp_arg_names):
        comp_name = pr.name(index, Kind.CDIMS)
        func_obj = pr.ComputedDims(index, comp_func) 
        PredNode.add_node(comp_name, func_obj, *comp_arg_names)

    def equate_ranks(self, target_sig, source_sig):
        self.check_sig(target_sig, 'equate ranks')
        self.check_sig(source_sig, 'equate ranks')
        self.rank_cons.equate_ranks(target_sig, source_sig)

    def limit_ranks(self, sig, min_val, max_val):
        self.p.check_sig(sig, 'rank limits')
        self.rank_cons.add_rank_limits(sig, min_val, max_val)

    def valid_dtypes(self, tensor_name, type_list):
        pass

    def equate_dtypes(self, trg_tensor, src_tensor):
        pass

    def arg_pseudo(self, pseudo_name, pred_func, arg_name):
        node_name = pr.name(pseudo_name, Kind.PSEUDO)
        func_obj = pr.ArgFunc(arg_name, pred_func)
        PredNode.add_node(node_name, func_obj, Kind.SCHEMA) 

    def arg_func(self, arg_name, pred_func, *func_arg_names):
        node_name = pr.name(arg_name, Kind.ARG)
        func_obj = pr.ArgFunc(arg_name, pred_func)
        PredNode.add_node(node_name, func_obj, *func_arg_names)

    def arg_tensor(self, arg_name, sig_func, *sig_arg_names):
        tensor_name = pr.name(arg_name, Kind.TENSOR) 
        dtype_name = pr.name(arg_name, Kind.DTYPE)
        shape_name = pr.name(arg_name, Kind.SHAPE)
        sig_name = pr.name(arg_name, Kind.SIG)

        PredNode.add_node(tensor_name, pr.GetTensor(arg_name), Kind.SCHEMA)
        dtype_pred = PredNode.add_node(dtype_name, pr.dtype, tensor_name)
        shape_pred = PredNode.add_node(shape_name, pr.shape, tensor_name)
        sig_pred = PredNode.add_node(sig_name, pr.Sig(sig_func), *sig_arg_names)

        sig_gen = GenNode.add_node(sig_name, ge.Sig(sig_func), *sig_arg_names)
        dtypes_gen = GenNode.get_node(Kind.DTYPES)
        dims_gen = GenNode.get_node(Kind.DIMS)
        GenNode.add_node(tensor_name, ge.GenTensor(arg_name), Kind.SIG,
                Kind.DIMS, Kind.DTYPES) 


        # Create edges
        dtypes_pred = PredNode.get_node(Kind.DTYPES)
        ranks_pred = PredNode.get_node(Kind.IRANKS)
        idims_pred = PredNode.get_node(Kind.IDIMS)
        dtypes_pred.append_parent(dtype_pred)
        ranks_pred.append_parent(shape_pred)
        ranks_pred.append_parent(sig_pred)
        idims_pred.append_parent(shape_pred)
        idims_pred.append_parent(sig_pred)


    def arg_shape(self, arg_name, sig_func, *sig_arg_names):
        shape_name = pr.name(arg_name, Kind.SHAPE)
        sig_name = pr.name(arg_name, Kind.SIG)
        PredNode.add_node(shape_name, pr.ArgShape(arg_name), Kind.SCHEMA)
        PredNode.add_node(sig_name, pr.Sig(sig_func), *sig_arg_names)

