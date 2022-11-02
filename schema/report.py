import numpy as np
from . import base
"""
               df   rank   idx   dtype
rank       A    0      1     0       ?  (single layout, multiple index_ranks)
fmt_idx    B    1      0     1       ?  (multiple layouts)
idx        C    0      0     1       ?  (single layout, single index_ranks)
fmt        D    1      0     0       ?  (can be up to 2, if) 
dtype      E    0      0     0       1
           F    1      1     0       ?  (ignored)
pass       G    0      0     0       0  (success)


df: data_format error
rank: arg rank error
idx: index usage error
dtype: dtype error

The six combinations above constitute all possible record types from the graph.

FixRank - if <rank> is present but <fmt_idx> and <fmt> are absent, issue the
message showing closest signature from each <rank> record of the declared data
format

FixDfOrRank - if both <rank> and one of <fmt_idx>, <fmt> records are present,
issue message showing closest signature from each <rank> record of declared format.
Show the single signature for each <fmt_idx> or <fmt> record, showing the <fmt>
record first.  (maybe, if <fmt> is present, skip showing the <fmt_idx>
records?)

FixIndexUsage - if <idx> is present but not <fmt>, issue the single signature
from <idx> of declared data_format, and highlight index usage error

FixDfOrIndexUsage - if <idx> and <fmt> are present, issue the single signature
from <idx> of declared data_format, highlighting index usage error.  Also, show
the single signature from each <fmt> record. 

FixDType - if <dtype> is present, issue the error which is one of: dtype not
allowed for arg, 2) dtype of arg1 must be equal to dtype of arg2, 3) dtype __
not implemented for x {idx description} dimensions (and layout ___)

Success - if <pass> is present, it should be the only record.  Do not print any
message.

# the only record that should be present is G, the success record


Breakdown of Mutual Exclusive categories:

df   rank   idx   dtype
 0      0     0       0  <pass>
 0      0     0       1  <dtype>
 0      0     1       0  <idx>
 0      0     1       1  <idx>
 0      1     0       0  <rank>
 0      1     0       1  <rank>
 0      1     1       0  excluded by graph
 0      1     1       1  excluded by graph
 1      0     0       0  <fmt>
 1      0     0       1  <fmt>
 1      0     1       0  <fmt_idx> 
 1      0     1       1  <fmt_idx>
 1      1     0       0  ignored
 1      1     0       1  ignored
 1      1     1       0  excluded by graph
 1      1     1       1  excluded by graph

          df   rank   idx   dtype
rank       0      1     0       ?  (single layout, multiple index_ranks)
fmt_idx    1      0     1       ?  (multiple layouts)
idx        0      0     1       ?  (single layout, single index_ranks)
fmt        1      0     0       ?  (one alternate layout) 
dtype      0      0     0       1
           1      1     0       ?  (ignored)
"""

def report(op, fixes, obs_dtypes, obs_shapes, obs_args):
    """
    Implement the messages templated in report.txt
    """
    rank_err = []
    fmt_err = []
    idx_err = []
    fmt_idx_err = []
    dtype_err = []
    dtype_fix = None

    for fix in fixes:
        df = fix['data_format'].cost()
        rank = fix['shape'].indel_cost()
        idx = fix['shape'].idx_usage_cost()
        dtype = fix['dtypes'].cost()
        tup = (df, rank, idx)
        # print(tup)

        if tup == (0,1,0):
            rank_err.append(fix)
        elif tup == (1,0,1):
            fmt_idx_err.append(fix)
        elif tup == (0,0,1):
            idx_err.append(fix)
        elif tup == (1,0,0):
            fmt_err.append(fix)
        elif tup == (0,0,0) and dtype == 1:
            assert dtype_fix is None
            dtype_fix = fix['dtypes']
        else:
            continue

    assert len(idx_err) < 2
    idx_fix = None if len(idx_err) == 0 else idx_err[0]

    if dtype_fix is not None:
        return fix_dtype(op, dtype_fix, obs_dtypes)

    if len(idx_err) > 0:
        if len(fmt_err) != 0:
            return fix_df_or_index_usage(op, idx_fix, fmt_err, obs_shapes,
                    obs_args)
        else:
            return fix_index_usage(op, idx_fix, obs_shapes, obs_args)

    if len(fmt_err) > 0:
        return fix_data_format(op, fmt_err, obs_shapes, obs_args)

    if len(rank_err) > 0:
        if len(fmt_err) > 0 or len(fmt_idx_err) > 0:
            return fix_df_or_rank(op, rank_err, fmt_err, fmt_idx_err,
                    obs_shapes, obs_args)
        else:
            return fix_rank(op, rank_err, obs_shapes, obs_args)

    return None

def grammar_list(items):
    # generate a grammatically correct English list
    if len(items) < 3:
        return ' and '.join(items)
    else:
        return ', '.join(items[:-1]) + ' and ' + items[-1]

def obs_rank_msg(op, obs_shapes, obs_args):
    # form a user-facing message describing all observed ranks
    items = []
    for arg, shape in obs_shapes.items():
        if isinstance(shape, int):
            continue
        item = f'{arg} rank = {len(shape)}'
        items.append(item)

    if op.data_formats.arg_name is not None:
        df_name = op.data_formats.arg_name
        item = f'{df_name} = {obs_args[df_name]}'
        items.append(item)

    item_str = grammar_list(items)
    return item_str

def index_abbreviations(op):
    msg = 'index abbreviations:'
    items = [msg]
    for idx, desc in op.index.items():
        item = f'{idx}: {desc}'
        items.append(item)
    tab = '\n'.join(items)
    return tab 

def _get_shape_columns(op, obs_shapes):
    s = { arg for arg, shp in obs_shapes.items() if isinstance(shp, list) }

    df_name = op.data_formats.arg_name
    if df_name is not None:
        s.add(df_name)
    columns = [ arg for arg in op.arg_order if arg in s ]
    return columns 

def _get_headers(op, columns):
    return [f'{c}.shape' if c in op.data_tensors else c for c in columns]

def template_table(op, fixes, obs_shapes, obs_args):
    """
    Produce a table with the input row, then a set of valid template rows.
    The last column consists of a suggested edit
    """
    columns = _get_shape_columns(op, obs_shapes)
    header_row = [''] + _get_headers(op, columns)
    df_name = op.data_formats.arg_name
    rows = [header_row]
    input_row = ['received']
    for col in columns:
        if col in obs_shapes:
            cell = ','.join(str(dim) for dim in obs_shapes[col])
            input_row.append(f'[{cell}]')
        elif col == df_name:
            input_row.append(obs_args[col])
    rows.append(input_row)

    used_indexes = set()
    tips = []
    for i, fix in enumerate(fixes, 1):
        edit = fix['shape']
        template_map = edit.arg_templates()

        edit_tips = []
        row = [f'config {i}'] 
        for col in columns:
            if col == df_name:
                df_edit = fix[df_name]
                if df_edit.cost() == 0:
                    cell = ''
                else:
                    tip = f'Change {df_name} to {df_edit.imp_val}'
                    edit_tips.append(tip)
                    cell = f'=> {df_edit.imp_val}'
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

        final_tip = f'=> config {i} tip: {", ".join(edit_tips)}'
        tips.append(final_tip)
        rows.append(row)

    main, _ = base.tabulate(rows, '   ', False)
    table = '\n'.join(main)
    tip_msg = '\n'.join(tips)
    return f'{table}\n\n{tip_msg}'

def _fix_config(op, all_fixes, obs_shapes, obs_args):
    """
    """
    ranks_msg = obs_rank_msg(op, obs_shapes, obs_args)
    leader_msg = f'Received invalid configuration: {ranks_msg}.  '
    leader_msg += f'Closest valid configurations:'
    table = template_table(op, all_fixes, obs_shapes, obs_args)
    index_abbrev = index_abbreviations(op)
    tail_msg = 'For the list of all valid configurations, use: '
    tail_msg += f'opgrind.explain(\'{op.op_path}\')'
    final = f'{leader_msg}\n\n{table}\n\n{index_abbrev}\n\n{tail_msg}\n'
    return final

def _index_usage_table(op, idx_or_fmt_fix, obs_shapes, obs_args):
    """
    Produce a table as depicted in 'FixIndexUsage' in report.txt 
    """
    fix = idx_or_fmt_fix
    column_names = _get_shape_columns(op, obs_shapes)
    header_row = [''] + _get_headers(op, column_names)
    df_name = op.data_formats.arg_name

    edit = fix['shape']
    arg_templ = edit.arg_templates()
    usage_map = edit.usage_map
    index_ranks = edit.index_ranks 
    layout = edit.layout

    leader_column = ['received', 'template', 'error']
    columns = [leader_column]
    for arg in column_names:
        if arg == df_name:
            obs_df = obs_args[df_name]
            imp_df = op.data_formats.data_format(layout, index_ranks)
            # the template should only appear if differing
            if obs_df != imp_df:
                hl = '^' * max(len(obs_df), len(imp_df))
                col_str = [ obs_df, imp_df, hl ]
            else:
                col_str = [ obs_df, '', '' ]
        else:
            col = [ obs_shapes[arg], arg_templ[arg] ]
            sig = edit.arg_sigs[arg]
            hl_row = []
            for idx in sig:
                do_hl = len(usage_map[idx]) > 1
                hl = [do_hl] * index_ranks[idx]
                hl_row.extend(hl)
            widths = [ max(len(str(t)), len(str(s))) for s, t in zip(*col) ]
            hl_row_str = ['^' * w if h else '' for h, w in zip(hl_row, widths)]
            col.append(hl_row_str)
            col_str, _ = base.tabulate(col, ' ', False)

        # add the header
        columns.append(col_str)

    rows = [header_row] + np.array(columns).transpose().tolist()
    main, _ = base.tabulate(rows, '   ', False)
    table = '\n'.join(main)
    return table

def _index_usage_leader(op, idx_fix, obs_shapes, obs_args):
    usage_map = idx_fix['shape'].usage_map
    # idx => (dims => [arg1, ...])

    items = []
    for idx, usage in usage_map.items():
        if len(usage) == 1:
            continue
        desc = op.index[idx]
        args = [ arg for l in usage.values() for arg in l ]
        ord_args = [ arg for arg in op.arg_order if arg in args ]
        shape_list = grammar_list(ord_args)
        item = f'\'{desc}\' dimensions (index {idx}) differ in {shape_list}'
        items.append(item)
    leader_msg = '\n'.join(items)
    return leader_msg

def _index_usage_change(op, edit):
    """
    Issue messages of the form:
    Change {arg}.shape[{slice}] or {arg2}.shape[{slice}] to the same values.
    """
    index_msgs = []
    for idx, usage in edit.usage_map.items():
        items = []
        if len(usage) == 1:
            continue
        desc = op.index[idx]

        for dims, arg_list in usage.items():
            for arg in arg_list:
                beg, end = edit.arg_index_slice(arg, idx)
                if end - beg == 1:
                    item = f'{arg}.shape[{beg}] = {dims[0]}'
                else:
                    item = f'{arg}.shape[{beg}:{end}] = {dims}'
                items.append(item)
        item_str = grammar_list(items)
        index_msg =  f'Change {item_str} to the same value(s) to fix '
        index_msg += f'index {idx} ({desc}) usage conflict.'
        index_msgs.append(index_msg)
    return index_msgs

def fix_dtype(op, dtype_fix, obs_dtypes):
    """
    Called if a dtype fix is available.
    """
    rules = op.dtype_rules

    if dtype_fix.kind == 'indiv':
        arg = dtype_fix.info
        obs_dtype = obs_dtypes[arg]
        valid_dtypes = rules.indiv_rules[arg]
        valid_phrase = grammar_list(valid_dtypes)

        final = f'Received {arg}.dtype = {obs_dtype}.  Valid dtypes for {arg} '
        final += f'{valid_phrase}'
    elif dtype_fix.kind == 'equate':
        arg = dtype_fix.info
        dtype = obs_dtypes[arg]
        source_arg = rules.equate_rules[arg]
        source_dtype = obs_dtypes[source_arg]

        final =  f'Received {arg}.dtype = {dtype} and '
        final += f'{source_arg}.dtype = {source_dtype}.  '
        final += f'dtypes of {arg} and {source_arg} must match.'

    elif dtype_fix.kind == 'combo':
        rule = dtype_fix.info

        items = []
        for arg, dtypes in rule.dtypes.items():
            dtype_str = ', '.join(dtypes)
            item = f'{arg}.dtype in ({dtype_str})'
            items.append(item)

        for idx, rank in rule.ranks.items():
            desc = op.index[idx]
            item = f'{rank} {desc} dimensions'
            items.append(item)

        formats = op.data_formats.formats
        exc_layouts = rule.excluded_layouts
        exc_formats = [df for df, (l, _) in formats.items() if l in exc_layouts]
        fmt_list = ', '.join(exc_formats)
        item = f'data formats ({fmt_list})'
        items.append(item)
        final = grammar_list(items)

    return f'Fix DType\n{final}'

def fix_index_usage(op, idx_fix, obs_shapes, obs_args):
    """
    Called when idx fix is present but no fmt, and no dtype fix
    """
    table = _index_usage_table(op, idx_fix, obs_shapes, obs_args)
    leader = _index_usage_leader(op, idx_fix, obs_shapes, obs_args)
    usage_msgs = _index_usage_change(op, idx_fix['shape'])
    usage_items = [ f'=> {item}' for item in usage_msgs]
    usage_msg = '\n'.join(usage_items)
    final = f'Fix Index Usage\n{leader}\n\n{table}\n\n{usage_msg}\n'
    return final

def fix_df_or_index_usage(op, idx_fix, fmt_fixes, obs_shapes, obs_args):
    """
    Called when idx fix and fmt fix are both available, but no dtype fix
    """
    df_name = op.data_formats.arg_name
    leader = f'Shapes are inconsistent for the provided {df_name}.  '
    leader += 'Suggestions:'
    usage_table = _index_usage_table(op, idx_fix, obs_shapes, obs_args)
    usage_msgs = _index_usage_change(op, idx_fix['shape'])

    final =  f'Fix Data Format or Index Usage\n'
    final += f'{leader}\n\n'

    for fmt_fix in fmt_fixes:
        df_table = _index_usage_table(op, fmt_fix, obs_shapes, obs_args)
        inf_df = fmt_fix[df_name].imp_val
        df_msg = f'=> Change {df_name} to {inf_df}'
        final += f'{df_table}\n\n{df_msg}\n\n'

    usage_items = [ f'=> {item}' for item in usage_msgs]
    usage_msg = '\n'.join(usage_items)
    final += f'{usage_table}\n\n{usage_msg}\n'
    return final

def fix_data_format(op, fmt_fixes, obs_shapes, obs_args):
    """
    Called when there is no idx fix
    """
    final = 'Fix Data Format\n'
    final += _fix_config(op, fmt_fixes, obs_shapes, obs_args)
    return final


def fix_rank(op, rank_fixes, obs_shapes, obs_args):
    """
    Called when rank fixes are present but fmt_idx and fmt fixes are both
    absent.
    """
    final = 'Fix Ranks\n'
    final += _fix_config(op, rank_fixes, obs_shapes, obs_args)
    return final

def fix_df_or_rank(op, rank_fixes, fmt_fixes, fmt_idx_fixes, obs_shapes,
        obs_args):
    """
    Called when rank fixes are present and one of fmt_idx or fmt fixes are also
    present.
    """
    if len(fmt_fixes) == 0:
        all_fixes = rank_fixes + fmt_idx_fixes
    else:
        all_fixes = fmt_fixes + rank_fixes
    final = 'Fix Data Format or Ranks\n'
    final += _fix_config(op, all_fixes, obs_shapes, obs_args)
    return final

