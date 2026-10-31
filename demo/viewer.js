"use strict";
(() => {
  const data = JSON.parse(document.getElementById("demo-data").textContent);
  const el = (id) => document.getElementById(id);
  const controls = {model:el("model-select"), task:el("task-select"), cases:el("case-select"),
    search:el("search"), example:el("example-select"), gold:el("show-gold")};
  const colors = ["#fac9c6", "#ddd0f3", "#ffe5aa", "#bce8ec"];
  const bookmark = new URLSearchParams(location.hash.slice(1));
  let filtered = [], currentId = bookmark.get("id");
  function option(select, value, label) {
    const node = document.createElement("option"); node.value=value; node.textContent=label; select.append(node);
  }
  data.models.forEach((name) => option(controls.model, name, name));
  if (data.models.includes(bookmark.get("model"))) controls.model.value=bookmark.get("model");
  [...new Set(data.examples.map((row) => row.task_type))].sort().forEach((name) => option(controls.task,name,name));
  data.label_types.forEach((kind,index) => {
    const entry=document.createElement("span"); entry.className="legend-item";
    const swatch=document.createElement("span"); swatch.className=`swatch type-${index}`;
    const label=document.createElement("span"); label.textContent=kind; entry.append(swatch,label); el("legend").append(entry);
  });
  function renderAnswer(target, panel) {
    target.replaceChildren();
    panel.segments.forEach((segment) => {
      if (!segment.types.length) { target.append(document.createTextNode(segment.text)); return; }
      const node=document.createElement("mark"); node.textContent=segment.text;
      node.title=segment.types.map((kind) => data.label_types[kind]).join(" + ") + ` · characters [${segment.start}, ${segment.end})`;
      if (segment.types.length===1) node.style.background=colors[segment.types[0]];
      else {
        const stops=segment.types.map((kind,index) => `${colors[kind]} ${100*index/segment.types.length}% ${100*(index+1)/segment.types.length}%`);
        node.style.background=`linear-gradient(to bottom,${stops.join(",")})`;
      }
      target.append(node);
    });
    if (!panel.segments.length) { const empty=document.createElement("span"); empty.className="muted"; empty.textContent="(Empty response)"; target.append(empty); }
  }
  function showExample() {
    const row=filtered.find((item) => item.id===controls.example.value);
    const visible=Boolean(row); el("empty-state").hidden=visible; el("example-content").hidden=!visible;
    const index=row?filtered.indexOf(row):-1;
    el("previous").disabled=index<=0; el("next").disabled=index<0||index>=filtered.length-1;
    if (!row) return;
    currentId=row.id;
    // Same-file fragments are useful for sharing an inspected example offline.
    try { history.replaceState(null,"","#"+new URLSearchParams({id:row.id,model:controls.model.value})); } catch (_) {}
    el("example-title").textContent=`Response ${row.id}`;
    el("example-meta").textContent=`${row.task_type} · ${row.generator} · source ${row.source_id} · ${row.response_characters.toLocaleString()} characters`;
    el("source-text").textContent=row.source_text;
    const name=controls.model.value, predicted=row.predictions[name];
    el("prediction-title").textContent=name + " prediction";
    el("prediction-count").textContent=`${predicted.spans.length} span${predicted.spans.length===1?"":"s"}`;
    el("gold-count").textContent=`${row.gold.spans.length} span${row.gold.spans.length===1?"":"s"}`;
    renderAnswer(el("prediction-text"),predicted); renderAnswer(el("gold-text"),row.gold);
    el("gold-card").hidden=!controls.gold.checked; el("response-grid").classList.toggle("single",!controls.gold.checked);
    el("match-note").textContent=row.exact_match[name]
      ? "Exact typed span sets match the human annotations for this example. This is not a dataset-level score."
      : "Exact typed span sets differ from the human annotations. Partial matches may still receive credit in character-level evaluation.";
  }
  function filterExamples() {
    const query=controls.search.value.trim().toLocaleLowerCase(), name=controls.model.value;
    filtered=data.examples.filter((row) => {
      if (controls.task.value && row.task_type!==controls.task.value) return false;
      if (query && ![row.id,row.source_id,row.task_type,row.generator,row.source_text,row.response].join("\n").toLocaleLowerCase().includes(query)) return false;
      const count=row.predictions[name].spans.length;
      if (controls.cases.value==="positive" && !count) return false;
      if (controls.cases.value==="negative" && count) return false;
      if (controls.cases.value==="mismatch" && row.exact_match[name]) return false;
      return true;
    });
    controls.example.replaceChildren();
    filtered.forEach((row,index) => option(controls.example,row.id,`${index+1} / ${filtered.length} · ${row.id} · ${row.task_type}`));
    if (filtered.some((row) => row.id===currentId)) controls.example.value=currentId;
    el("collection-note").textContent=`${filtered.length.toLocaleString()} matching examples · ${data.displayed_records.toLocaleString()} of ${data.validated_records.toLocaleString()} quality=good validation responses included · All prediction ID sets were checked before display selection.`;
    showExample();
  }
  [controls.model,controls.task,controls.cases].forEach((node) => node.addEventListener("change",filterExamples));
  controls.search.addEventListener("input",filterExamples); controls.example.addEventListener("change",showExample);
  controls.gold.addEventListener("change",showExample);
  [["previous",-1],["next",1]].forEach(([id,delta]) => el(id).addEventListener("click",() => {
    const next=controls.example.selectedIndex+delta;
    if (next>=0 && next<controls.example.options.length) {controls.example.selectedIndex=next;showExample();}
  }));
  filterExamples();
})();
