"""Build gui/job_sequence_dashboard.html from the notebook's job sequence export.

Usage: python gui/build_dashboard.py [path/to/job_sequences.json]
Default input: notebooks/results/partc/job_sequences.json (written by dashboard_export_cell.py).
The output is a single self-contained HTML file with the data embedded; it needs no server.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
src = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent / 'notebooks' / 'results' / 'partc' / 'job_sequences.json'
data = src.read_text(encoding='utf-8').replace('</', '<\\/')
html = (HERE / 'dashboard_template.html').read_text(encoding='utf-8').replace('/*__DATA__*/null', data)
out = HERE / 'job_sequence_dashboard.html'
out.write_text(html, encoding='utf-8')
print(f'{out}  ({out.stat().st_size / 1e6:.1f} MB)')
