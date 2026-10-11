// Shared folder tree for the raw viewer and manual DBM workbench.
window.OptCarBinExplorer = {
  expandTo(opened, relativePath) {
    const parts = relativePath.split('/');
    parts.pop();
    let path = '';
    for (const folder of parts) { path += folder + '/'; opened.add(path); }
  },
  render({tree, entries, filter = '', opened, selected, onSelect, formatDuration, scrollToSelection = true}) {
    tree.replaceChildren();
    filter = filter.trim().toLowerCase();
    const visible = entries.filter(entry => entry.relative_path.toLowerCase().includes(filter));
    const root = {folders: new Map(), files: []};
    for (const entry of visible) {
      const parts = entry.relative_path.split('/');
      parts.pop();
      let node = root;
      for (const folder of parts) {
        if (!node.folders.has(folder)) node.folders.set(folder, {folders: new Map(), files: []});
        node = node.folders.get(folder);
      }
      node.files.push(entry);
    }
    function append(node, parent, prefix = '') {
      for (const [name, child] of node.folders) {
        const path = prefix + name + '/';
        const details = document.createElement('details');
        details.className = 'bin-folder'; details.dataset.folder = path;
        details.open = Boolean(filter) || opened.has(path);
        const summary = document.createElement('summary'); summary.textContent = name;
        const contents = document.createElement('div'); contents.className = 'bin-folder-contents';
        append(child, contents, path); details.append(summary, contents);
        details.addEventListener('toggle', () => details.open ? opened.add(path) : opened.delete(path));
        parent.append(details);
      }
      for (const entry of node.files) {
        const button = document.createElement('button');
        button.className = 'bin-file'; button.dataset.path = entry.path;
        button.disabled = Boolean(entry.error); button.title = entry.error || entry.path;
        button.setAttribute('aria-pressed', String(selected instanceof Set ? selected.has(entry.path) : selected === entry.path));
        const name = document.createElement('span'); name.textContent = entry.name;
        const meta = document.createElement('small');
        meta.textContent = entry.error || `${formatDuration(entry.duration_s)} · ${entry.samples.toLocaleString()} samples${entry.split ? ' · '+entry.split : ''}${entry.num_envs>1?' · '+entry.num_envs+' trajectories':''}`;
        button.append(name, meta); button.onclick = () => onSelect(entry.path); parent.append(button);
      }
    }
    append(root, tree);
    if (!visible.length) tree.textContent = 'No raw bins match this folder or filter.';
    if (scrollToSelection) tree.querySelector('.bin-file[aria-pressed=true]')?.scrollIntoView({block: 'nearest'});
    return visible.length;
  },
};
