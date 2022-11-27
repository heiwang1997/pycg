"""
Copyright 2022 by Jiahui Huang. All rights reserved.
This file is part of PyCG toolbox and is released under "MIT License Agreement".
Please see the LICENSE file that should have been included as part of this package.
"""


class HElement:
    def render(self):
        raise NotImplementedError


class HText(HElement):
    def __init__(self, text: str, boldface: bool = False):
        self.text = text
        self.boldface = boldface

    def render(self):
        rendered = self.text
        if self.boldface:
            rendered = f'<strong>{rendered}</strong>'
        return rendered


def convert_canonical(content):
    if isinstance(content, HElement):
        return content
    elif isinstance(content, str):
        return HText(content)
    elif isinstance(content, list):
        canonicalized = []
        for c in content:
            canonicalized.append(convert_canonical(c))
        return canonicalized
    else:
        raise NotImplementedError


def render_attributes(attr_dict: dict):
    return ' '.join([
        f'{k}="{v}"' for k, v in attr_dict.items()
    ])


class HDiv(HElement):
    def __init__(self, content, attributes=None):
        self.content = convert_canonical(content)
        if attributes is None:
            attributes = {}
        self.attributes = attributes

    def render(self):
        all_content = [f'<div {render_attributes(self.attributes)}>']
        for c in self.content:
            all_content += [c.render()]
        all_content += ['</div>']
        return ''.join(all_content)


class HImage(HElement):
    def __init__(self, path, attributes=None):
        self.path = path
        if attributes is None:
            attributes = {}
        self.attributes = attributes

    def render(self):
        self.attributes.update({"src": self.path})
        return f'<img {render_attributes(self.attributes)}>'


class HTable(HElement):
    def __init__(self, content, header=None, attributes=None):
        assert isinstance(content, list)
        assert isinstance(content[0], list)
        if attributes is None:
            attributes = {}
        self.content = convert_canonical(content)
        self.header = convert_canonical(header) if header is not None else None
        self.attributes = attributes

    def render(self):
        all_content = [f'<table {render_attributes(self.attributes)}>']
        if self.header is not None:
            all_content += ['<thead>', '<tr>']
            all_content += ['<td>' + t.render() + '</td>' for t in self.header]
            all_content += ['</tr>', '</thead>']
        all_content += ['<tbody>']
        for row in self.content:
            row_content = ['<td>' + t.render() + '</td>' for t in row]
            row_content = ''.join(['<tr>'] + row_content + ['</tr>'])
            all_content.append(row_content)
        all_content += ['</tbody>']
        all_content += ['</table>']
        return ''.join(all_content)


def render_html(content: list, css: str = None, title: str = "PYCG HTML"):
    contents = ['<!DOCTYPE html><html><head>']
    contents += [f'<title>{title}</title>']
    if css is not None:
        contents += [f'<style>{css}</style>']
    contents += ['</head><body>']
    contents += [c.render() for c in content]
    contents += ['</body></html>']
    return ''.join(contents)


TEMPLATE_TABLE_CSS = '''
html,
body {
  height: 100%;
}
body {
  margin: 0;
  background: rgba(240,240,240,0.5);
  font-family: sans-serif;
  font-weight: 100;
}
table {
  border-collapse: collapse;
  overflow: hidden;
  box-shadow: 0 0 20px rgba(0,0,0,0.1);
}
th,
td {
  padding: 15px;
  background-color: rgba(255,255,255,0.2);
  color: #000;
}
th {
  text-align: left;
}
thead th {
  background-color: #55608f;
}
tbody tr:hover {
  background-color: rgba(0,0,0,0.3);
}
tbody td {
  position: relative;
}
tbody td:hover:before {
  content: "";
  position: absolute;
  left: 0;
  right: 0;
  top: -9999px;
  bottom: -9999px;
  background-color: rgba(0,0,0,0.2);
  z-index: -1;
}
'''


def render_template_table(htable: HTable, title: str = None):
    # This would be useful: https://uicookies.com/css-table-templates/
    hdiv = HDiv(content=[htable], attributes={"class": "container"})
    return render_html([hdiv], css=TEMPLATE_TABLE_CSS, title="Table View")
