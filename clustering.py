import numpy as np
import pandas as pd
import json

BENCHMARK = 'TPC_DS_100'
DEFAULT_CHUNK_SIZE = 65535


class Operator:
    def __init__(self, t, operator):
        self.t = t
        self.operator = operator
        self.output_operator = None
        self.left_input_operator = None
        self.right_input_operator = None
        self.output_can_be_sorted = None


def read_operator(operator_type):
    return [Operator(operator_type, r) for r in
            pd.read_csv(BENCHMARK + '/' + operator_type + '.csv', sep='|').itertuples()]


def read_operators():
    operator_types = ['get_tables', 'aggregates', 'joins', 'projections', 'table_scans', 'validates']
    operators = []
    for op in operator_types:
        operators += read_operator(op)
    return operators


def build_tree(operators):
    hashes_to_operators = {op.operator.OPERATOR_HASH: op for op in operators}
    for op in operators:
        h = op.operator.LEFT_INPUT_OPERATOR_HASH
        if h is not None:
            o = hashes_to_operators.get(h)
            if o is not None:
                o.output_operator = op
                op.left_input_operator = o
        h = op.operator.RIGHT_INPUT_OPERATOR_HASH
        if h is not None:
            o = hashes_to_operators.get(h)
            if o is not None:
                o.output_operator = op
                op.right_input_operator = o


def sort_check(operator):

    def check_left_child(op):
        if hasattr(op, 'left_input_operator') and op.left_input_operator is not None:
            return operator_output_can_be_sorted(op.left_input_operator)
        else:
            return False

    type = operator.t

    if type == 'get_tables':
        return True

    if type == 'table_scans':
        if operator.operator.COLUMN_TYPE == 'DATA':
            return True
        else:
            return check_left_child(operator)
    elif type in ['validates', 'projections']:
        return check_left_child(operator)
    elif type == 'aggregates':
        return False
    elif type == 'joins':
        if operator.operator.JOIN_MODE == 'Semi':
            return check_left_child(operator)
        else:
            return False
    else:
        return False

def operator_output_can_be_sorted(operator):

    if hasattr(operator, 'output_can_be_sorted') and operator.output_can_be_sorted is not None:
        return operator.output_can_be_sorted
    else:
        operator.output_can_be_sorted = sort_check(operator)

def compute_table_scan_pruning_savings_ns(table_scans, table_name, column):
    affected_scans = table_scans[table_scans['COLUMN_NAME'] == column]
    affected_scans = affected_scans[affected_scans['TABLE_NAME'] == table_name]

    runtimes = affected_scans['RUNTIME_NS']
    output_rows = affected_scans['OUTPUT_ROW_COUNT']
    output_rows = output_rows + (output_rows % DEFAULT_CHUNK_SIZE)  # round it to the next multiple
    scores = runtimes - (runtimes * (output_rows / affected_scans['INPUT_ROW_COUNT']))
    return scores.sum()

def compute_join_savings_ns(operators, table_name, sort_column):
    joins = [j for j in operators if j.t == 'joins']
    affected_joins = [j for j in joins if (j.operator.LEFT_TABLE_NAME == table_name and j.operator.LEFT_COLUMN_NAME == sort_column) or (j.operator.RIGHT_TABLE_NAME == table_name and j.operator.RIGHT_COLUMN_NAME == sort_column)]
    savings = 0
    for j in affected_joins:
        if hasattr(j, 'left_input_operator') and j.left_input_operator is not None:
            if j.left_input_operator.output_can_be_sorted:
                savings += j.operator.RUNTIME_NS * 0.2307
                continue
        if hasattr(j, 'right_input_operator') and j.right_input_operator is not None:
            if j.right_input_operator.output_can_be_sorted:
                savings += j.operator.RUNTIME_NS * 0.2307
                continue

    return savings

def compute_aggregate_savings_ns(operators, table_name, sort_column):
    aggregates = [a for a in operators if a.t == 'aggregates']
    affected_aggregates = [a for a in aggregates if a.operator.AGGREGATE_COLUMN_COUNT == sort_column and a.operator.TABLE_NAME == table_name] # todo this should be a.operator.COLUMN_NAME
    savings = 0
    for a in affected_aggregates:
        if hasattr(a, 'left_input_operator') and a.left_input_operator is not None:
            if a.left_input_operator.output_can_be_sorted:
                savings += a.operator.RUNTIME_NS * 0.2307
    return savings



def get_best_columns_for_scores(scores, columns, max_num_columns = 5):
    s = np.array(scores)
    if len(scores) == 0:
        return None

    m = s.mean()
    if (m == 0):
        print("all scores were zero")
        return [list(columns)[0]]
    else:
        print("not all zero")
    s /= m  # can't make sense out of such big numbers..
    scores = s.tolist()

    scores = sorted([(column, sc) for sc, column in zip(scores, columns)], key=lambda s: s[1], reverse=True)
    print(scores)

    cluster_columns = []

    previous_score = scores[0][1]
    for i in range(len(scores)):
        sc = scores[i]
        if (sc[1] / previous_score) / max(i, 1) < 0.25 or len(cluster_columns) >= max_num_columns:
            break
        else:
            cluster_columns.append(sc)
            previous_score = sc[1]

    print(cluster_columns)
    return [c[0] for c in cluster_columns]

def run():
    all_operators = read_operators()
    query_hashes = np.unique(np.array([op.operator.QUERY_HASH for op in all_operators]))
    operators_for_query = {h: [op for op in all_operators if op.operator.QUERY_HASH == h] for h in query_hashes}

    for operators in operators_for_query.values():
        build_tree(operators)
        for op in operators:
            operator_output_can_be_sorted(op)



    clusters = {}

    tables = pd.read_csv(BENCHMARK + "/table_meta_data.csv", sep='|')
    tables = tables[tables['ROW_COUNT'] > DEFAULT_CHUNK_SIZE]
    for table_name in tables['TABLE_NAME']:

        columns = pd.read_csv(BENCHMARK + "/column_meta_data.csv", sep='|')
        columns = columns[columns['TABLE_NAME'] == table_name]['COLUMN_NAME']

        scores = []
        for column in columns:
            score = 0
            table_scans = pd.read_csv(BENCHMARK + "/table_scans.csv", sep='|')
            table_scans = table_scans[table_scans['TABLE_NAME'] == table_name]
            score += compute_table_scan_pruning_savings_ns(table_scans, table_name, column)
            score += 2 * compute_join_savings_ns(all_operators, table_name, column)
            score += compute_aggregate_savings_ns(all_operators, table_name, column)
            scores.append(score)
        cluster_column_names = get_best_columns_for_scores(scores, columns, 1)

        scores = []
        for column in columns:
            score = 0
            score += compute_join_savings_ns(all_operators, table_name, column)
            score += compute_aggregate_savings_ns(all_operators, table_name, column)
            scores.append(score)
        if np.array(scores).sum() == 0.0:
            sort_column_name = cluster_column_names[0]
        else:
            sort_column_name = get_best_columns_for_scores(scores, columns, 1)[0]


        clusters[table_name] = {
            'cluster_columns': cluster_column_names,
            'sort_column': sort_column_name
        }

    with open(BENCHMARK + "/" + "clustering.json", "w") as f:
        f.write(json.dumps(clusters))


if __name__ == '__main__':
    run()
