async function loadPlugins() {
  const res = await fetch('/plugin_list');
  if (!res.ok) return [];
  return await res.json();
}

let plugins = [];

function createPluginSelect() {
  const sel = document.createElement('select');
  sel.className = 'plugin-select';
  plugins.forEach(p => {
    const opt = document.createElement('option');
    opt.value = p;
    opt.textContent = p;
    sel.appendChild(opt);
  });
  return sel;
}

function addFiles() {
  const files = document.getElementById('file-input').files;
  const tbody = document.querySelector('#file-table tbody');
  for (const file of files) {
    const tr = document.createElement('tr');
    tr.file = file;
    const selectTd = document.createElement('td');
    const checkbox = document.createElement('input');
    checkbox.type = 'checkbox';
    checkbox.className = 'row-select';
    selectTd.appendChild(checkbox);
    const nameTd = document.createElement('td');
    nameTd.textContent = file.name;
    const pluginTd = document.createElement('td');
    pluginTd.appendChild(createPluginSelect());
    const dimsTd = document.createElement('td');
    const dimsInput = document.createElement('input');
    dimsInput.type = 'text';
    dimsInput.placeholder = '{"old_dim": "new_dim"}';
    dimsInput.className = 'dims-input';
    dimsTd.appendChild(dimsInput);
    const coordsTd = document.createElement('td');
    const coordsInput = document.createElement('input');
    coordsInput.type = 'text';
    coordsInput.placeholder = '{"coord": [1,2]}';
    coordsInput.className = 'coords-input';
    coordsTd.appendChild(coordsInput);

    tr.appendChild(selectTd);
    tr.appendChild(nameTd);
    tr.appendChild(pluginTd);
    tr.appendChild(dimsTd);
    tr.appendChild(coordsTd);

    tbody.appendChild(tr);
  }
}

async function combineSelected() {
  const rows = document.querySelectorAll('#file-table tbody tr');
  for (const row of rows) {
    const cb = row.querySelector('.row-select');
    if (!cb.checked) continue;
    const file = row.file;
    const plugin = row.querySelector('.plugin-select').value;
    const dims = row.querySelector('.dims-input').value || '{}';
    const coords = row.querySelector('.coords-input').value || '{}';
    const buf = await file.arrayBuffer();
    const bytes = Array.from(new Uint8Array(buf));
    await fetch('/upload_data', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({plugin, filename: file.name, file_bytes: bytes, dims, coords})
    });
  }

  const res = await fetch('/get_dataset_html');
  if (res.ok) {
    const html = await res.text();
    document.getElementById('dataset-output').innerHTML = html;
  }
}

window.addEventListener('DOMContentLoaded', async () => {
  plugins = await loadPlugins();
  document.getElementById('add-files-btn').addEventListener('click', addFiles);
  document.getElementById('combine-btn').addEventListener('click', combineSelected);
});
