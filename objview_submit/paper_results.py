"""Public manuscript results, independent of private submissions and smoke jobs."""
from functools import lru_cache
from html import escape
import json
from pathlib import Path

from .validation import IntakeError


SPACES = {'whole': 'Whole · Full view sphere', 'quarter': 'Quarter · Quarter sphere'}
SETTINGS = {'K=5': 'K=5 · Small budget', 'K=30': 'K=30 · Large budget',
            'automatic': 'Automatic · Automatic stopping', 'K=10': 'K=10 · Additional budget',
            'MS@0.01': 'MS@0.01 · Stopping sensitivity', 'MS@0.03': 'MS@0.03 · Stopping sensitivity'}
SORTS = {'quality': 'NSC@0.02: highest first', 'path': 'Path: lowest first, NSC ≥ 0.95', 'method': 'Method name'}
FAMILIES = {'Classical': 'Classical', 'Learned': 'Learned', 'Hybrid': 'Hybrid'}


@lru_cache(maxsize=1)
def load_results():
    return json.loads((Path(__file__).parent / 'data/paper_results.json').read_text())


def select_results(space='whole', setting='K=5', order='quality'):
    if space not in SPACES or setting not in SETTINGS or order not in SORTS:
        raise IntakeError('INVALID_RESULT_FILTER', 422)
    checkpoints = {'MS@0.02', 'plan_end'} if setting == 'automatic' else {setting}
    rows = [row for row in load_results()['synthetic']
            if row['space'] == space and row['checkpoint'] in checkpoints]
    if order == 'path':
        rows = [row for row in rows if float(row['nsc02']) >= .95]
        return sorted(rows, key=lambda row: (float(row['path']), row['method']))
    if order == 'quality':
        return sorted(rows, key=lambda row: (-float(row['nsc02']), row['method']))
    return sorted(rows, key=lambda row: row['method'])


def _select(name, label, choices, selected):
    options = ''.join('<option value="' + escape(value, quote=True) + '"' +
                      (' selected' if value == selected else '') + '>' + escape(text) + '</option>'
                      for value, text in choices.items())
    return '<label>' + label + '<select name="' + name + '">' + options + '</select></label>'


def _table(caption, headers, rows):
    head = ''.join('<th scope="col">' + escape(header) + '</th>' for header in headers)
    body = ''.join('<tr><th scope="row">' + escape(str(row[0])) + '</th>' +
                   ''.join('<td>' + escape(str(cell)) + '</td>' for cell in row[1:]) + '</tr>' for row in rows)
    return ('<div class="table-scroll" tabindex="0" role="region" aria-label="' + escape(caption, quote=True) +
            '"><table><caption>' + escape(caption) + '</caption><thead><tr>' + head +
            '</tr></thead><tbody>' + body + '</tbody></table></div>')


def render_results(space='whole', setting='K=5', order='quality'):
    rows = select_results(space, setting, order)
    data = load_results()
    content = '''<section class="hero"><p class="eyebrow">OBJVIEW-BENCH / PAPER RESULTS</p>
<h1>View planning under real constraints.</h1>
<p class="lead">Explore how view budgets and reachability affect reconstruction quality, camera motion and planning efficiency.</p>
<div class="actions"><a class="button" href="#results">Explore results</a>
<a class="button secondary" href="/ui/submit">Submit Dockerfile</a></div>
<dl class="stats"><div><dt>Baseline methods</dt><dd>9</dd></div><div><dt>Main-Hidden-Test</dt><dd>528 <small>objects</small></dd></div>
<div><dt>View spaces</dt><dd>Whole / Quarter</dd></div></dl></section>'''
    content += '''<section id="results" class="panel"><p class="eyebrow">01 / SYNTHETIC BENCHMARK</p>
<h2>Paper baseline results</h2><p>Browse results reported in the paper. Results from future submissions will be reviewed and published separately.</p>'''
    content += '<form class="filters" method="get" action="/ui#results">'
    content += _select('space', 'Reachable view space', SPACES, space)
    content += _select('setting', 'Budget and stopping rule', SETTINGS, setting)
    content += _select('order', 'Compare by', SORTS, order)
    content += '<button type="submit">Update results</button></form>'
    content += '<p class="muted">Only methods reported under the selected protocol are shown. Missing settings are not zeros. Sorting uses rounded paper values; equal values are not assigned distinct ranks.</p>'
    display = [[row['method'], FAMILIES[row['family']],
                'Native plan end' if row['checkpoint'] == 'plan_end' else row['checkpoint'],
                row['views'], row['nsc02'], row['sc02'], row['path'], row['time_bin']] for row in rows]
    content += _table(SPACES[space] + ' / ' + SETTINGS[setting] + ' · ' + str(len(rows)) + ' methods',
                      ['Method', 'Family', 'Stopping protocol', 'Views', 'NSC@0.02 ↑', 'SC@0.02 ↑', 'Path Cost ↓', 'PR bin (s)'], display)
    if not rows:
        content += '<p>No reported results meet NSC@0.02 ≥ 0.95 under this setting.</p>'
    content += '''<div class="notes"><p><strong>Views include the initial observation.</strong>K limits additional acquisitions after the initial observation, so K=5 normally means 6 total views.</p>
<p><strong>Automatic respects each method’s execution mode.</strong>Iterative methods use MS@0.02: relative voxel-count gain below 1% for three consecutive steps.
MA-SCVP and PoinTr-C+SCP stop at native plan end. Budgeted methods have no automatic-stopping results.</p>
<p>NSC normalizes coverage by the observable reference for the same view space; SC measures coverage of the full object surface. High NSC in Quarter does not imply high full-surface coverage.
Some methods terminate early. Interpret short paths alongside coverage and Views, especially for NBV-Net in Quarter.</p></div>'''
    content += '<details><summary>Additional thresholds and Chamfer Distance</summary>'
    content += _table('Additional metrics for this selection', ['Method', 'NSC@0.01 ↑', 'NSC@0.03 ↑', 'SC@0.01 ↑', 'SC@0.03 ↑', 'CD ↓'],
                      [[r['method'], r['nsc01'], r['nsc03'], r['sc01'], r['sc03'], r['cd']] for r in rows])
    content += '</details></section>'
    content += '''<section class="panel" id="protocol"><p class="eyebrow">02 / READING THE RESULTS</p>
<h2>Metrics and evaluation protocol</h2><dl class="definitions">
<div><dt>NSC ↑ / SC ↑</dt><dd>NSC@0.02 is the primary quality metric. Thresholds 0.01, 0.02 and 0.03 are in normalized object coordinates, not physical centimeters.</dd></div>
<div><dt>Path Cost ↓</dt><dd>Accumulated camera travel in normalized object coordinates, with collision-avoiding distance around the unit object. This is not robot joint-space trajectory length.</dd></div>
<div><dt>Planning Runtime</dt><dd>Algorithm-reported total planning time from the paper, recorded in coarse second bins. This is distinct from trusted server-side timing in the submission platform.</dd></div>
<div><dt>Execution modes</dt><dd>Random+TSP and PoinTr-C+MCP use budgeted plans; MA-SCVP and PoinTr-C+SCP use finite set-cover plans; the others replan iteratively.</dd></div>
</dl><p class="muted">Paper hardware: Intel Core i7-14700HX, 32 GB RAM, RTX 4060 Laptop GPU (8 GB).
The PC-NBV adaptation did not converge reliably and is excluded from the main comparison. The 168 slow-saturation objects form a separate diagnostic split, outside the 528-object main results.</p></section>'''
    content += '''<section class="panel" id="real-rgbd"><p class="eyebrow">03 / REAL RGB-D REPLAY</p>
<h2>Real RGB-D replay</h2><p>25 object–placement configurations: average over reachable starts within each configuration, then macro-average over configurations.
CRC measures coverage of a captured reference and is not directly comparable to synthetic SC / NSC. This study replays shared pre-captured observations; it is not repeated closed-loop robot execution for each method.</p>'''
    for checkpoint, label in (('K=5', 'Small budget K=5'), ('K=30', 'Large budget K=30'), ('plan_end', 'Native finite-plan termination')):
        real_rows = [row for row in data['real_rgbd'] if row['checkpoint'] == checkpoint]
        content += '<details><summary>' + label + ' · ' + str(len(real_rows)) + ' methods</summary>'
        content += _table('Real RGB-D / ' + label, ['Method', 'Views', 'CRC@1cm ↑', 'CRC@2cm ↑', 'Path Cost ↓'],
                          [[r['method'] + (' †' if r['early_termination'] else ''), r['views'], r['crc01'],
                            r['crc02'], r['path']] for r in real_rows]) + '</details>'
    content += '''<p class="muted">† NBV-Net’s low path cost results from early invalid-action termination, not higher motion efficiency.
BENBV-Net generates continuous online viewpoints and is excluded from this fixed-view replay pool. MS automatic stopping and planning runtime are not reported for real replay.</p></section>'''
    content += '<section class="panel" id="sources"><h2>Data provenance</h2><p>' + escape(data['source']['title']) + '</p>'
    content += '<p>Source version: <code>' + escape(data['source']['file']) + '</code>. The full synthetic tables match the corresponding tables in main.tex.</p>'
    content += '<ul>' + ''.join('<li><code>' + escape(label) + '</code></li>' for label in
                              dict.fromkeys(row['source_table'] for row in data['synthetic'] + data['real_rgbd'])) + '</ul>'
    content += '<p><a href="/ui/paper-results.json" download="objview-paper-results.json">Download paper results (JSON)</a></p>'
    content += '<details><summary>Source file checksum</summary><p class="digest">SHA256: ' + escape(data['source']['sha256']) + '</p></details></section>'
    return content
