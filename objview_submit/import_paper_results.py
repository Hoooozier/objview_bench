"""Offline, strict extraction of the four curated paper tables; never runs TeX.

Run from the repository root with ``python3 -m objview_submit.import_paper_results``.
The web service reads only the generated snapshot, not the manuscript directory.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re


FAMILIES = {
    'Random+TSP': 'Classical', 'RSE': 'Classical', 'RSE-Mov': 'Classical',
    'NBV-Net': 'Learned', 'BENBV-Net': 'Learned', 'MA-SCVP': 'Learned',
    'PoinTr-C+NBV': 'Hybrid', 'PoinTr-C+SCP': 'Hybrid', 'PoinTr-C+MCP': 'Hybrid',
}
METRICS = {
    'Views': 'views', 'NSC@0.01': 'nsc01', 'NSC@0.02': 'nsc02', 'NSC@0.03': 'nsc03',
    'SC@0.01': 'sc01', 'SC@0.02': 'sc02', 'SC@0.03': 'sc03',
    'CD': 'cd', 'Path': 'path', 'Time bin': 'time_bin',
}
TABLES = (
    ('tab:app_budgeted_plan_results_transposed', 'budgeted', ['K=5', 'K=10', 'K=30'], 2),
    ('tab:app_setcover_plan_results_transposed', 'set_cover', ['plan_end'], 2),
    ('tab:app_iterative_results_transposed', 'iterative',
     ['K=5', 'K=10', 'K=30', 'MS@0.01', 'MS@0.02', 'MS@0.03'], 5),
)
DEFAULT_SOURCE = Path('pannips/main_post_rebuttal.tex')
DEFAULT_OUTPUT = Path(__file__).parent / 'data' / 'paper_results.json'


def _table(source, label):
    tables = re.findall(r'\\begin\{table\*?\}.*?\\end\{table\*?\}', source, re.S)
    matches = [table for table in tables if '\\label{' + label + '}' in table]
    if len(matches) != 1:
        raise ValueError('Expected exactly one table: ' + label)
    return matches[0]


def _value(cell, *, time_bin=False):
    value = cell.strip().replace('$', '').replace('^\\dagger', '').replace('--', '–')
    pattern = r'(?:<0\.1|0\.1–0\.5|0\.5–1|1–3|3–5|>5)' if time_bin else r'\d+(?:\.\d+)?'
    if not re.fullmatch(pattern, value):
        raise ValueError('Unexpected table cell: ' + cell)
    return value


def extract(source_bytes, source_name):
    # Ignore commented-out LaTeX; no include expansion, macros or code execution.
    source = re.sub(r'(?m)(?<!\\)%.*$', '', source_bytes.decode('utf-8'))
    rows = []
    for label, mode, checkpoints, expected_methods in TABLES:
        blocks = re.findall(r'\\begin\{tabular\}.*?\\end\{tabular\}', _table(source, label), re.S)
        if len(blocks) != expected_methods:
            raise ValueError('Unexpected method count: ' + label)
        for block in blocks:
            method = re.search(r'\\textbf\{([^}]+)\}', block).group(1)
            family = FAMILIES[method]
            metrics = {}
            for line in block.splitlines():
                if '&' not in line:
                    continue
                cells = line.split('&')
                name = cells[0].split('$')[0].strip()
                if name not in METRICS:
                    continue
                if not line.rstrip().endswith('\\\\') or name in metrics:
                    raise ValueError('Unexpected metric row: ' + name)
                values = [*cells[1:-1], cells[-1].rstrip()[:-2]]
                if len(values) != 2 * len(checkpoints):
                    raise ValueError('Unexpected column count: ' + method)
                metrics[name] = [_value(v, time_bin=name == 'Time bin') for v in values]
            if set(metrics) != set(METRICS):
                raise ValueError('Incomplete metrics: ' + method)
            for index, (space, checkpoint) in enumerate(
                    (s, c) for s in ('whole', 'quarter') for c in checkpoints):
                rows.append(dict(method=method, family=family, mode=mode, space=space,
                                 checkpoint=checkpoint, source_table=label,
                                 **{METRICS[name]: values[index] for name, values in metrics.items()}))

    real = []
    label = 'tab:real_rgbd_replay'
    checkpoint = None
    for line in _table(source, label).splitlines():
        if 'Fixed-budget cap, $K=5$' in line:
            checkpoint = 'K=5'
        elif 'Fixed-budget cap, $K=30$' in line:
            checkpoint = 'K=30'
        elif 'Native finite-plan termination' in line:
            checkpoint = 'plan_end'
        elif line.split('&')[0].strip() in FAMILIES:
            cells = line.rstrip().removesuffix('\\\\').split('&')
            if len(cells) != 5 or checkpoint is None:
                raise ValueError('Unexpected RGB-D row')
            method = cells[0].strip()
            real.append(dict(method=method, family=FAMILIES[method], checkpoint=checkpoint,
                             source_table=label, early_termination=method == 'NBV-Net',
                             **dict(zip(('views', 'crc01', 'crc02', 'path'), map(_value, cells[1:])))))
    if len(rows) != 76 or len(real) != 14:
        raise ValueError('Incomplete paper snapshot')
    if len({(r['method'], r['space'], r['checkpoint']) for r in rows}) != len(rows):
        raise ValueError('Duplicate results')
    return {
        'schema_version': 1, 'kind': 'paper_reported_results',
        'source': {'file': source_name, 'sha256': hashlib.sha256(source_bytes).hexdigest(),
                   'title': 'ObjView-Bench: Rethinking Difficulty and Deployment for Object-Centric View Planning'},
        'synthetic': rows, 'real_rgbd': real,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--check', action='store_true', help='Verify snapshot without modifying it')
    args = parser.parse_args()
    snapshot = extract(args.source.read_bytes(), args.source.as_posix())
    if args.check:
        if json.loads(args.output.read_text()) != snapshot:
            parser.error('Paper snapshot differs; review and regenerate it')
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + '\n')
    print('Verified 76 synthetic results and 14 real RGB-D replay results.')


if __name__ == '__main__':
    main()
