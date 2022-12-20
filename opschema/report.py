import numpy as np
from . import base
from .base import ReportKind
"""
1 edit distance:

     df     rank    shape    dtype     <indexed by>           <report_type>
A     1        0        0        0     layout                 carat_tables
C     0        0        1        0     --                     carat_tables
B     0        1        0        0     index_ranks            arrow_table
D     0        0        0        1     --                     dtype_report


2 edit distance:

     df     rank    shape    dtype     <indexed by>           <report_type>
D     1        1        0        0     layout and index_ranks arrow_table
E     1        0        1        0     layout                 carat_tables
F     1        0        0        1     layout                 ??
G     0        1        0        1     index_ranks            dtype_report
H     0        0        1        1     --                     dtype_report

For arrow_table, how do we deal with both the rank and data_format error?
Just put two arrows.

"""

class Report(object):
    def __init__(self, op, fixes, obs_dtypes, obs_shapes, obs_args):
        self.op = op
        self.fixes = fixes
        self.obs_dtypes = obs_dtypes
        self.obs_shapes = obs_shapes
        self.obs_args = obs_args

    def report(self):
        items = []
        items.append(self.carat_tables())
        items.append(self.arrow_table())
        items.append(self.dtype_report())
        items = list(filter(None, items))
        return '\n\n'.join(items)

    def dtype_report(self):
        dtype_fixes = [f for f in self.fixes if f.kind() == ReportKind.DType]
        uniq_dt = { hash(f.dtype): f.dtype for f in dtype_fixes }.values()
        items = []
        for dtype_edit in uniq_dt:
            item = self._fix_dtype(dtype_edit)
            items.append(item)
        return '\n'.join(items)

    def _fix_dtype(self, dtype_edit):
        """
        Called if a dtype fix is available.
        """
        rules = self.op.dtype_rules

        if dtype_edit.kind == 'indiv':
            arg = dtype_edit.info
            obs_dtype = self.obs_dtypes[arg]
            valid_dtypes = rules.indiv_rules[arg]
            valid_phrase = grammar_list(valid_dtypes)

            final =  f'Received {arg}.dtype = {obs_dtype}.  Valid dtypes for '
            final += f'{arg} are: {valid_phrase}'

        elif dtype_edit.kind == 'equate':
            arg = dtype_edit.info
            dtype = self.obs_dtypes[arg]
            source_arg = rules.equate_rules[arg]
            source_dtype = self.obs_dtypes[source_arg]

            final =  f'Received {arg}.dtype = {dtype} and '
            final += f'{source_arg}.dtype = {source_dtype}.  '
            final += f'dtypes of {arg} and {source_arg} must match.'

        elif dtype_edit.kind == 'combo':
            rule = dtype_edit.info

            items = []
            if rule.dtypes is not None:
                for arg, dtypes in rule.dtypes.items():
                    dtype_str = ', '.join(dtypes)
                    item = f'{arg}.dtype in ({dtype_str})'
                    items.append(item)

            if rule.ranks is not None:
                for idx, rank in rule.ranks.items():
                    desc = self.op.index[idx].snake()
                    item = f'{rank} {desc} dimensions'
                    items.append(item)

            if rule.layouts is not None:
                formats = self.op.data_formats.formats
                exc_layouts = rule.layouts
                exc_formats = [df for df, (l, _) in formats.items() if l in
                        exc_layouts and df is not None]
                fmt_list = ', '.join(exc_formats)
                arg_name = self.op.data_formats.arg_name
                item = f'{arg_name} in ({fmt_list})'
                items.append(item)

            set_msg = grammar_list(items)
            final = f'This combination is not implemented: {set_msg}'

        code = base.FixKind.codestring(dtype_edit.code())
        return f'{code}\n{final}'

    def _carat_table(self, fix):
        """
        Produce a table as depicted in 'FixIndexUsage' in report.txt 
        """
        column_names = _get_shape_columns(self.op, self.obs_shapes)
        header_row = [''] + _get_headers(self.op, column_names)
        df_name = self.op.data_formats.arg_name

        shape_edit = fix.shape
        arg_templ = shape_edit.arg_templates()
        usage_map = shape_edit.usage_map
        index_ranks = shape_edit.index_ranks 
        layout = shape_edit.layout

        leader_column = ['received', 'template', 'error']
        columns = [leader_column]
        for arg in column_names:
            if arg == df_name:
                used_fmt = fix.df.used
                imp_fmt = fix.df.imputed
                # the template should only appear if differing
                if used_fmt != imp_fmt:
                    hl = '^' * max(len(used_fmt), len(imp_fmt))
                    col_str = [ used_fmt, imp_fmt, hl ]
                else:
                    col_str = [ fix.df.observed, '', '' ]
            else:
                # all signature-bearing arguments (or returns)
                sig = shape_edit.arg_sigs[arg]
                if arg in self.obs_shapes:
                    obs_shape = self.obs_shapes[arg]
                    if isinstance(obs_shape, int):
                        obs_shape = [obs_shape]
                else:
                    obs_shape = []
                    for idx in sig:
                        dims = shape_edit.maybe_get_index_dim(idx) 
                        obs_shape.extend(dims)

                # shape_str = dims_string(obs_shape)
                obs_repr = ['?' if d is None else str(d) for d in obs_shape]
                col = [ obs_repr, arg_templ[arg] ]
                hl_row = []
                for idx in sig:
                    do_hl = shape_edit.highlighted(arg, idx)
                    hl = [do_hl] * index_ranks[idx]
                    hl_row.extend(hl)
                widths = [max(len(str(t)), len(str(s))) for s, t in zip(*col)]
                hl_row_str = ['^' * w if h else '' for h, w in zip(hl_row,
                    widths)]
                col.append(hl_row_str)
                col_str, _ = base.tabulate(col, ' ', False)

            # add the header
            columns.append(col_str)

        rows = [header_row] + np.array(columns).transpose().tolist()
        main, _ = base.tabulate(rows, '   ', False)
        codestring = base.FixKind.codestring(fix.code())
        table = '\n'.join([codestring] + main)
        return table

    def carat_tables(self):
        fixes = [f for f in self.fixes if f.kind() == ReportKind.CaratTable]
        items = []
        for fix in fixes:
            table = self._carat_table(fix)
            items.append(table)

            df_change = _change_data_format_msg(fix)
            if df_change is not None:
                items.append(df_change)

            # index usage error messages 
            usage_msgs = _change_usage_msgs(self.op, fix)
            usage_items = [ f'=> {item}' for item in usage_msgs]
            items.extend(usage_items)

            idx_cons_msg = _idx_constraint_msg(self.op, fix, self.obs_args)
            if idx_cons_msg is not None:
                items.append(idx_cons_msg)

        # This is too verbose to include
        # if len(items) > 0:
            # index_msg = index_definitions(self.op)
            # items.append(index_msg)

        final = None if len(items) == 0 else '\n\n'.join(items)
        return final

    def _obs_rank_msg(self):
        # form a user-facing message describing all observed ranks
        items = []
        for arg, shape in self.obs_shapes.items():
            if isinstance(shape, int):
                continue
            item = f'{arg} rank = {len(shape)}'
            items.append(item)

        if self.op.data_formats.arg_name is not None:
            df_name = self.op.data_formats.arg_name
            item = f'{df_name} = {self.obs_args[df_name]}'
            items.append(item)

        item_str = grammar_list(items)
        return item_str

    def _template_table(self, fixes):
        """
        Produce a table with the input row, then a set of valid template rows.
        The last column consists of a suggested edit
        """
        columns = _get_shape_columns(self.op, self.obs_shapes)
        header_row = [''] + _get_headers(self.op, columns)
        df_name = self.op.data_formats.arg_name
        rows = [header_row]
        input_row = ['received']
        for col in columns:
            if col in self.obs_shapes:
                obs_shape = self.obs_shapes[col]
                shape_str = dims_string(obs_shape)
                input_row.append(shape_str)
            elif col == df_name:
                input_row.append(self.obs_args[col])
        rows.append(input_row)

        used_indexes = set()
        tips = []
        code = 0
        for i, fix in enumerate(fixes, 1):
            code |= fix.code()
            edit = fix.shape
            template_map = edit.arg_templates()

            edit_tips = []
            row = [f'config {i}'] 
            for col in columns:
                if col == df_name:
                    if fix.df.cost() == 0:
                        cell = ''
                    else:
                        tip = f'Change {fix.df.arg_name} to {fix.df.imputed}'
                        edit_tips.append(tip)
                        cell = f'=> {fix.df.imputed}'
                else:
                    cell = ' '.join(template_map[col])
                    if col in edit.arg_delta:
                        delta = edit.arg_delta[col]
                        cell = '=> ' + cell
                        sfx = '' if abs(delta) == 1 else 's'
                        if delta < 0:
                            tip = f'remove {abs(delta)} dimension{sfx} from {col}'
                        else:
                            tip = f'add {delta} dimension{sfx} to {col}'
                        edit_tips.append(tip)
                row.append(cell)

            final_tip = f'=> config {i}: {", ".join(edit_tips)}'
            tips.append(final_tip)
            rows.append(row)

        codestring = base.FixKind.codestring(code)
        main, _ = base.tabulate(rows, '   ', False)
        table = '\n'.join(main)
        tip_msg = '\n'.join(tips)
        return f'{codestring}\n{table}\n\n{tip_msg}'

    def arrow_table(self):
        """
        Produce an 'arrow table' format.  It is used for rank errors in the
        absence of any dtype errors.
        """
        fixes = [f for f in self.fixes if f.kind() == ReportKind.ArrowTable]
        if len(fixes) == 0:
            return ''

        ranks_msg = self._obs_rank_msg()
        leader_msg = f'Received invalid configuration: {ranks_msg}.  '
        leader_msg += f'Closest valid configurations:'
        table = self._template_table(fixes)
        # index_defn = index_definitions(self.op)
        # tail_msg = 'For the list of all valid configurations, use: '
        # tail_msg += f'opschema.explain(\'{self.op.op_path}\')'
        
        # this is too verbose
        # final = f'{leader_msg}\n\n{table}\n\n{index_defn}\n\n{tail_msg}\n'
        final = f'{leader_msg}\n\n{table}\n'
        return final

def grammar_list(items):
    # generate a grammatically correct English list
    if len(items) == 0:
        return None
    elif len(items) < 3:
        return ' and '.join(items)
    else:
        return ', '.join(items[:-1]) + ' and ' + items[-1]

def index_definitions(op):
    msg = 'index definitions:'
    items = [msg]
    for idx, ind in op.index.items():
        item = f'{idx}: {ind.snake()}'
        items.append(item)
    tab = '\n'.join(items)
    return tab 

def _get_shape_columns(op, obs_shapes):
    s = set(obs_shapes.keys()) 
    # s = { arg for arg, shp in obs_shapes.items() if isinstance(shp, list) }

    df_name = op.data_formats.arg_name
    if df_name is not None:
        s.add(df_name)
    columns = [ arg for arg in op.arg_order if arg in s ]

    # append output columns
    for i in range(op.num_returns):
        columns.append(f'return[{i}]')

    return columns 

def _get_headers(op, columns):
    return [op._arg_shape_name(c) for c in columns]

def _change_usage_msgs(op, fix):
    """
    Issue messages of the form:
    Change {arg}.shape[{slice}] or {arg2}.shape[{slice}] to the same values.
    """
    index_msgs = []
    for idx, usage in fix.shape.usage_map.items():
        items = []
        if len(usage) == 1:
            continue
        desc = op.index[idx].desc

        all_args = set()
        for dims, arg_list in usage.items():
            for arg in arg_list:
                arg_shape_name = op._arg_shape_name(arg)
                beg, end = fix.shape.arg_index_slice(arg, idx)
                if end - beg == 1:
                    item = f'{arg_shape_name}[{beg}] = {dims[0]}'
                else:
                    item = f'{arg_shape_name}[{beg}:{end}] = {list(dims)}'
                items.append(item)
                all_args.add(arg)

        arg_list_msg = grammar_list(list(all_args)) 
        item_str = grammar_list(items)
        index_msg =  f'{op.index[idx].display_name(True)} ({idx}) has inconsistent '
        index_msg += f'dimensions in {arg_list_msg}. {item_str}'
        index_msgs.append(index_msg)
    return index_msgs

def _change_data_format_msg(fix):
    if fix.df.cost() != 0:
        msg = f'=> Change {fix.df.arg_name} to {fix.df.imputed}'
    else:
        msg = None
    return msg

def dims_string(dims):
    # create a string representation of dimensions, 
    if isinstance(dims, int):
        return str(dims)
    else:
        s = ','.join('?' if d is None else str(d) for d in dims)
        return f'[{s}]'

def _idx_constraint_msg(op, fix, obs_args):
    """
    Called when index dimensions violate a constraint added with
    API function add_index_predicate.

    Produces a message like:

    "output spatial" dimensions must be >= 0

    output spatial = ceil((input_spatial - (filter_spatial - 1) * dilations) / strides)
    [-2] = ceil(([16] - ([10] - 1) * 2) / 1)
    """
    if fix.shape.pred_cost() == 0:
        return None

    pred = fix.shape.index_pred_error
    templ_args = [ f'{op.index[idx].snake()}' for idx in pred.indices ]

    items = []
    for idx in pred.indices:
        item =  f'{op.index[idx].snake()} ({idx}) = '
        idx_dims = fix.shape.maybe_get_index_dim(idx)
        item += dims_string(idx_dims)
        items.append(item)
    values_msg = grammar_list(items)

    constraint_msg = pred.pfunc_t(*templ_args)

    notice = f'{values_msg}.  {constraint_msg}'

    # show all formulas preceding and up to any of pred.indices
    formulas = fix.shape.formulas
    if len(formulas) == 0:
        path_msg = None
    else:
        desc_msg = '\n'.join(frm.desc_path for frm in formulas)
        code_msg = '\n'.join(frm.code_path for frm in formulas)
        dims_msg = '\n'.join(frm.dims_path for frm in formulas)
        path_msg = 'Dimensions computed as:\n'
        path_msg += '\n\n'.join((desc_msg, code_msg, dims_msg))

    main = list(filter(None, (notice, path_msg)))
    return '\n\n'.join(main)

