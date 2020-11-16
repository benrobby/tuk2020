import numpy as np
import pandas as pd
import json
from typing import List, Mapping


def get_speedup_saving_factor(speedup):
    return 1 - 1 / speedup


BENCHMARK = 'TPC_DS_100'
DEFAULT_CHUNK_SIZE = 65535
SORTED_JOIN_SPEEDUP = 1.8
SORTED_AGGREGATE_SPEEDUP = 1.3
PRUNING_SAVINGS_FACTOR = 1.0
SORTED_JOIN_SPEEDUP_SAVING_FACTOR = get_speedup_saving_factor(SORTED_JOIN_SPEEDUP)
SORTED_AGGREGATE_SPEEDUP_SAVING_FACTOR = get_speedup_saving_factor(SORTED_AGGREGATE_SPEEDUP)


class Operator:
    def __init__(self, operator_type: str, operator):
        self.operator_type: str = operator_type
        self.operator: any = operator
        self.output_operator: Operator = None
        self.left_input_operator: Operator = None
        self.right_input_operator: Operator = None
        self.output_can_be_sorted: bool = None


def read_operator(operator_type: str) -> List[Operator]:
    return [Operator(operator_type, r) for r in
            pd.read_csv(BENCHMARK + '/' + operator_type + '.csv', sep='|').itertuples()]


def read_operators() -> List[Operator]:
    operator_types = ['get_tables', 'aggregates', 'joins', 'projections', 'table_scans', 'validates']
    operators = []
    for operator_type in operator_types:
        operators += read_operator(operator_type)
    return operators


def build_tree(operators: List[Operator]) -> None:
    hashes_to_operators: Mapping[str, Operator] = {op.operator.OPERATOR_HASH: op for op in operators}
    for current_operator in operators:

        left_hash = current_operator.operator.LEFT_INPUT_OPERATOR_HASH
        if left_hash is not None:
            left_input_operator = hashes_to_operators.get(left_hash)
            if left_input_operator is not None:
                left_input_operator.output_operator = current_operator
                current_operator.left_input_operator = left_input_operator

        right_hash = current_operator.operator.RIGHT_INPUT_OPERATOR_HASH
        if right_hash is not None:
            right_input_operator = hashes_to_operators.get(right_hash)
            if right_input_operator is not None:
                right_input_operator.output_operator = current_operator
                current_operator.right_input_operator = right_input_operator


def sort_check(operator: Operator) -> bool:
    def check_left_child(op):
        if hasattr(op, 'left_input_operator') and op.left_input_operator is not None:
            return operator_output_can_be_sorted(op.left_input_operator)
        else:
            return False

    t = operator.operator_type

    if t == 'get_tables':
        return True

    if t == 'table_scans':
        if operator.operator.COLUMN_TYPE == 'DATA':
            return True
        else:
            return check_left_child(operator)
    elif t in ['validates', 'projections']:
        return check_left_child(operator)
    elif t == 'aggregates':
        return False
    elif t == 'joins':
        if operator.operator.JOIN_MODE == 'Semi':
            return check_left_child(operator)
        else:
            return False
    else:
        return False


def operator_output_can_be_sorted(operator) -> bool:
    if hasattr(operator, 'output_can_be_sorted') and operator.output_can_be_sorted is not None:
        return operator.output_can_be_sorted
    else:
        operator.output_can_be_sorted = sort_check(operator)
        return operator.output_can_be_sorted


def compute_table_scan_pruning_savings_ns(table_scans, table_name: str, column_name: str):
    affected_scans = table_scans[table_scans['COLUMN_NAME'] == column_name]
    affected_scans = affected_scans[affected_scans['TABLE_NAME'] == table_name]

    runtimes = affected_scans['RUNTIME_NS']
    output_rows = affected_scans['OUTPUT_ROW_COUNT']
    output_rows = output_rows + (output_rows % DEFAULT_CHUNK_SIZE)  # round it to the next multiple
    scores = runtimes - (runtimes * (output_rows / affected_scans['INPUT_ROW_COUNT']))
    return PRUNING_SAVINGS_FACTOR * scores.sum()


def compute_join_savings_ns(operators, table_name: str, sort_column_name: str):
    joins = [j for j in operators if j.operator_type == 'joins']
    affected_joins = [j for j in joins if
                      (
                                  j.operator.LEFT_TABLE_NAME == table_name and j.operator.LEFT_COLUMN_NAME ==
                                  sort_column_name) or (
                              j.operator.RIGHT_TABLE_NAME == table_name and j.operator.RIGHT_COLUMN_NAME ==
                              sort_column_name)]
    savings = 0
    for j in affected_joins:
        if hasattr(j, 'left_input_operator') and j.left_input_operator is not None:
            if j.left_input_operator.output_can_be_sorted:
                savings += j.operator.RUNTIME_NS * SORTED_JOIN_SPEEDUP_SAVING_FACTOR
                continue
        if hasattr(j, 'right_input_operator') and j.right_input_operator is not None:
            if j.right_input_operator.output_can_be_sorted:
                savings += j.operator.RUNTIME_NS * SORTED_JOIN_SPEEDUP_SAVING_FACTOR
                continue

    return savings


def compute_aggregate_savings_ns(operators, table_name, sort_column):
    aggregates = [a for a in operators if a.operator_type == 'aggregates']
    affected_aggregates: List[Operator] = [a for a in aggregates if
                           a.operator.AGGREGATE_COLUMN_COUNT == sort_column and a.operator.TABLE_NAME == table_name]
    # todo this should be a.operator.COLUMN_NAME
    savings = 0
    for a in affected_aggregates:
        if hasattr(a, 'left_input_operator') and a.left_input_operator is not None:
            if a.left_input_operator.output_can_be_sorted:
                savings += a.operator.RUNTIME_NS * SORTED_AGGREGATE_SPEEDUP_SAVING_FACTOR
    return savings


def get_best_columns_for_scores(scores: List[float], column_name: List[str], max_num_columns=5) -> List[str]:
    s = np.array(scores)
    if len(scores) == 0:
        return None

    m = s.mean()
    if m == 0:
        print("all scores were zero")
        return [list(column_name)[0]]
    else:
        print("not all zero")
    s /= m  # can't make sense out of such big numbers..
    scores = s.tolist()

    scores = sorted([(column, sc) for sc, column in zip(scores, column_name)], key=lambda s: s[1], reverse=True)
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


def find_cluster_columns_for_table(all_operators: List[Operator], column_names: List[str], table_name) -> List[str]:
    table_scans = pd.read_csv(BENCHMARK + "/table_scans.csv", sep='|')
    table_scans = table_scans[table_scans['TABLE_NAME'] == table_name]

    scores = [
        compute_table_scan_pruning_savings_ns(table_scans, table_name, column_name) +
        compute_join_savings_ns(all_operators, table_name, column_name) +
        compute_aggregate_savings_ns(all_operators, table_name, column_name)

        for column_name in column_names
    ]

    cluster_column_names = get_best_columns_for_scores(scores, column_names, 1)
    return cluster_column_names


def find_best_sort_column_for_table(all_operators, cluster_column_names, column_names, table_name):
    scores: List[float] = [
        compute_join_savings_ns(all_operators, table_name, column) +
        compute_aggregate_savings_ns(all_operators, table_name, column)

        for column in column_names]

    if np.array(scores).sum() == 0.0:
        sort_column_name = cluster_column_names[0]
    else:
        sort_column_name = get_best_columns_for_scores(scores, column_names, 1)[0]
    return sort_column_name


def determine_clustering_for_table(all_operators, table_name):
    columns = pd.read_csv(BENCHMARK + "/column_meta_data.csv", sep='|')
    column_names = columns[columns['TABLE_NAME'] == table_name]['COLUMN_NAME']

    cluster_column_names = find_cluster_columns_for_table(all_operators, column_names, table_name)
    sort_column_name = find_best_sort_column_for_table(all_operators, cluster_column_names, column_names, table_name)

    return {
        'cluster_columns': cluster_column_names,
        'sort_column': sort_column_name
    }


def run():
    all_operators: List[Operator] = read_operators()
    query_hashes = np.unique(np.array([op.operator.QUERY_HASH for op in all_operators]))
    queries_to_operators: Mapping[str, List[Operator]] = {
        query_hash: [operator for operator in all_operators if operator.operator.QUERY_HASH == query_hash]
        for query_hash in query_hashes
    }

    for operators_for_query in queries_to_operators.values():
        build_tree(operators_for_query)
        for op in operators_for_query:
            operator_output_can_be_sorted(op)

    clusters = {}

    tables = pd.read_csv(BENCHMARK + "/table_meta_data.csv", sep='|')
    tables = tables[tables['ROW_COUNT'] > DEFAULT_CHUNK_SIZE]

    for table_name in tables['TABLE_NAME']:
        clusters[table_name] = determine_clustering_for_table(all_operators, table_name)

    with open(BENCHMARK + "/" + "clustering.json", "w") as f:
        f.write(json.dumps(clusters))


if __name__ == '__main__':
    run()
