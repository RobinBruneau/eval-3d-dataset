"""LaTeX tables: coverage per configuration, Chamfer per object, failures."""

from src.figures.cells import Results
from src.figures.tables import chamfer_table, coverage_table, failures_table

from .helpers import make_cell, make_config, make_spec

TABLES = {
    "object_labels": {"obj1": "first\\_obj"},
    "tables": {
        "coverage": {"methods": ["alpha", "beta"], "views": ["all", 10], "tiers": ["d2", "d4"],
                     "caption": "Cov.", "label": "tab:coverage", "output": "coverage.tex"},
        "chamfer": {"methods": ["alpha", "beta"], "tiers": ["d1", "d2", "d4"], "caption": "CD.",
                    "label": "tab:chamfer", "output": "cd.tex"},
        "failures": {"caption": "Fail.", "label": "tab:failures", "output": "failures.tex"},
    },
}


def _results(tmp_path):
    config = make_config(tmp_path)
    make_cell(config, "obj1", "alpha_d4", chamfer=0.2)
    make_cell(config, "obj2", "alpha_d4", chamfer=0.4)
    make_cell(config, "obj1", "alpha_d2", chamfer=0.3)
    make_cell(config, "obj1", "beta_d4", chamfer=0.1)
    make_cell(config, "obj2", "beta_d4", chamfer=0.5)
    make_cell(config, "obj1", "alpha_d4_v10", chamfer=0.9)
    spec = make_spec(**TABLES)
    return spec, Results(config, spec)


def test_coverage_counts_and_not_run(tmp_path):
    spec, res = _results(tmp_path)
    tex = coverage_table(spec, res)
    assert "Alpha & 2 & 1 & 1 & 2 \\\\" in tex
    assert "Beta & 2 & -- & -- & 2 \\\\" in tex
    assert "Method & 84 & 10 & d2 & d4 \\\\" in tex
    assert "\\label{tab:coverage}" in tex and tex.endswith("\\end{table}\n")


def test_chamfer_ranks_rows_and_reports_means(tmp_path):
    spec, res = _results(tmp_path)
    tex = chamfer_table(spec, res)
    assert "Object & d2 & d4 & d4 \\\\" in tex  # alpha d2, alpha d4, beta d4 (d1 never run: no column)
    assert "first\\_obj & \\underline{0.300} & 0.200 & \\textbf{0.100} \\\\" not in tex
    assert "first\\_obj & 0.300 & \\underline{0.200} & \\textbf{0.100} \\\\" in tex
    assert "obj2 & -- & \\textbf{0.400} & \\underline{0.500} \\\\" in tex
    assert "\\textit{Mean} & 0.300 & 0.300 & 0.300 \\\\" in tex
    assert "\\textit{Reconstructed} & 1/2 & 2/2 & 2/2 \\\\" in tex


def test_failures_ordered_dense_to_sparse(tmp_path):
    spec, res = _results(tmp_path)
    tex = failures_table(spec, res)
    lines = [l for l in tex.splitlines() if "/2 &" in l]
    assert lines == ["Alpha d2 & 1/2 & obj2 \\\\", "Alpha d4, 10 views & 1/2 & obj2 \\\\"]


def _cd_f_results(tmp_path, **chamfer):
    config = make_config(tmp_path)
    for obj, cell, cd, f in (("obj1", "alpha_d2", 0.3, 0.4), ("obj1", "alpha_d4", 0.2, 0.7), ("obj1", "beta_d4", 0.1, 0.6),
                             ("obj2", "alpha_d4", 0.4, 0.5), ("obj2", "beta_d4", 0.5, 0.8)):
        make_cell(config, obj, cell, chamfer=cd, fscore=(f, 0.9))
    tables = {**TABLES["tables"], "chamfer": {**TABLES["tables"]["chamfer"], "method_tiers": {"alpha": ["d4"]},
                                              "fscore": 0.5, **chamfer}}
    spec = make_spec(object_labels=TABLES["object_labels"], tables=tables)
    return spec, Results(config, spec)


def test_chamfer_and_fscore_in_each_cell(tmp_path):
    spec, res = _cd_f_results(tmp_path)
    tex = chamfer_table(spec, res)
    assert "Object & d4 & d4 \\\\" in tex  # alpha restricted to d4: its d2 column is gone
    # each metric ranked on its own: lowest CD, highest F
    assert "first\\_obj & \\underline{0.200} / \\textbf{70.0} & \\textbf{0.100} / \\underline{60.0} \\\\" in tex
    assert "obj2 & \\textbf{0.400} / \\underline{50.0} & \\underline{0.500} / \\textbf{80.0} \\\\" in tex
    assert "\\textit{Mean} & 0.300 / 60.0 & 0.300 / 70.0 \\\\" in tex
    assert "\\textit{Reconstructed} & 2/2 & 2/2 \\\\" in tex


def test_chamfer_and_fscore_stacked(tmp_path):
    spec, res = _cd_f_results(tmp_path, layout="stacked")
    lines = chamfer_table(spec, res).splitlines()
    cd = lines.index("\\multicolumn{3}{c}{Chamfer distance (mm) $\\downarrow$} \\\\")
    f = lines.index("\\multicolumn{3}{c}{F-score at 0.5~mm (\\%) $\\uparrow$} \\\\")
    assert cd < lines.index("first\\_obj & \\underline{0.200} & \\textbf{0.100} \\\\") < f
    assert f < lines.index("first\\_obj & \\textbf{70.0} & \\underline{60.0} \\\\")
    assert lines.index("\\textit{Mean} & 0.300 & 0.300 \\\\") < f < lines.index("\\textit{Mean} & 60.0 & 70.0 \\\\")
    assert lines.count("\\textit{Reconstructed} & 2/2 & 2/2 \\\\") == 2
