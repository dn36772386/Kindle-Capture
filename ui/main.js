/* pywebviewのAPIが注入されてから動くようにする */
const $ = sel => document.querySelector(sel);
const q = $("#q");
const list = $("#list");
const btnStart = $("#start");
const btnClose = $("#close");
const chkSplit = $("#split");
const chkAuto = $("#autocrop");
const err = $("#err");

let items = [];
let selected = -1;

async function refresh() {
  err.style.display = "none";
  const res = await window.pywebview.api.list_windows(q.value||"");
  if(!res.ok){
    err.textContent = res.message || "検索に失敗しました。";
    err.style.display = "block";
    return;
  }
  items = res.items || [];
  render();
}

function render(){
  list.innerHTML = "";
  items.forEach((x, i) => {
    const btn = document.createElement("fluent-button");
    btn.textContent = x.title;
    btn.setAttribute("appearance", "stealth");
    btn.style.width = "100%";
    btn.addEventListener("click", () => {
      selected = i;
      start();
    });
    list.appendChild(btn);
  });
  selected = items.length ? 0 : -1;
}

async function start(){
  if(selected < 0 || !items.length){
    err.textContent = "対象ウィンドウを選択してください。";
    err.style.display = "block";
    return;
  }
  const hwnd = items[selected].hwnd;
  const opt = {
    split_double_page: chkSplit.checked,
    auto_crop: chkAuto.checked
  };
  // 開始フィードバック
  btnStart.disabled = true;
  const oldText = btnStart.textContent;
  btnStart.textContent = "撮影中…";
  const res = await window.pywebview.api.start_capture({ hwnd, options: opt });
  if(!res.ok){
    err.textContent = res.message || "開始に失敗しました。";
    err.style.display = "block";
    btnStart.disabled = false;
    btnStart.textContent = oldText || "開始";
    return;
  }
  // ここでは閉じない。完了はトーストとログで確認。
}

q.addEventListener("input", refresh);
q.addEventListener("keydown", async (e)=>{
  if(e.key === "Enter"){ 
    if(selected === -1 && items.length){ selected = 0; }
    start();
  } else if(e.key === "Escape"){
    await window.pywebview.api.hide();
  } else if(e.key === "ArrowDown"){
    if(items.length){ selected = Math.min(items.length-1, selected+1); }
  } else if(e.key === "ArrowUp"){
    if(items.length){ selected = Math.max(0, selected-1); }
  }
});
btnStart.addEventListener("click", start);
btnClose.addEventListener("click", async ()=> { try { await window.pywebview.api.quit(); } catch(e){} });

// 初期表示（pywebviewが準備できてから）
async function boot(){
  try {
    if (!window.pywebview || !window.pywebview.api) {
      throw new Error("pywebview API が未準備です");
    }
    const s = await window.pywebview.api.get_settings();
    chkAuto.checked = !!s.auto_crop;
    chkSplit.checked = !!s.split_double_page;
    q.focus();
    await refresh();
  } catch (e) {
    err.textContent = (e && e.message) ? e.message : "初期化に失敗しました。";
    err.style.display = "block";
  }
}
if (window.pywebview && window.pywebview.api) { boot(); }
else { window.addEventListener('pywebviewready', boot); }

// Escでも終了したい場合（任意）
document.addEventListener('keydown', async (e)=>{
  if(e.key === 'Escape'){
    try { await window.pywebview.api.quit(); } catch(_){}}
});
