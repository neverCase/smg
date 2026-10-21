import { Universe } from './scene.js';
import { DEFAULT_ENDPOINT, normalizePayload, summarize, demoPayload, escapeHTML as esc, relativeLoads, heatColor, validateEndpoint } from './data.js';

const $ = id => document.getElementById(id);
const fmt = value => typeof value !== 'number' || !Number.isFinite(value) ? '—' : value.toLocaleString('en-US');
const pct = value => value === null ? '未提供' : `${(value * 100).toFixed(1)}%`;
const store = { get(key){try{return localStorage.getItem(key);}catch{return null;}}, set(key,value){try{localStorage.setItem(key,value);}catch{/* Private browsing can disable storage. */}} };
let saved = {};try { saved = JSON.parse(store.get('smg-orbital-settings') || '{}'); } catch { /* Use defaults. */ }
let endpoint = DEFAULT_ENDPOINT;try { endpoint = validateEndpoint(saved.endpoint || DEFAULT_ENDPOINT); } catch { /* Ignore invalid stored URLs. */ }
const allowedIntervals = [2000,3000,5000,10000];
const state = { endpoint, interval:allowedIntervals.includes(saved.interval)?saved.interval:3000, mode:saved.mode==='demo'?'demo':'live', data:null, selected:null, group:null, search:'', collapsed:new Set(), history:[], lastSuccess:null, connection:'connecting', error:'', paused:matchMedia('(prefers-reduced-motion: reduce)').matches, labels:true };
let universe=null, timer=null, controller=null, generation=0;
let panels={left:matchMedia('(max-width:640px)').matches,right:true,bottom:false,immersive:false};
try{const savedPanels=JSON.parse(store.get('xhive-orbital-panels')||'{}');for(const key of Object.keys(panels))if(typeof savedPanels[key]==='boolean')panels[key]=savedPanels[key];}catch{/* Use default panel visibility. */}
function applyPanels(){
  for(const side of ['left','right','bottom']){document.body.classList.toggle(`${side}-collapsed`,panels[side]);$(`expand-${side}`).setAttribute('aria-expanded',!panels[side]);}
  document.body.classList.toggle('immersive',panels.immersive);$('immersive-toggle').setAttribute('aria-pressed',panels.immersive);$('immersive-toggle').querySelector('span').textContent=panels.immersive?'退出沉浸':'沉浸模式';
  store.set('xhive-orbital-panels',JSON.stringify(panels));
}
function setPanel(side,collapsed){panels[side]=collapsed;applyPanels();}
function toggleImmersive(){panels.immersive=!panels.immersive;applyPanels();}

function notice(message,action=false) {
  $('scene-notice').hidden=!message;
  $('scene-notice').replaceChildren(document.createTextNode(message || ''));
  if(action){const button=document.createElement('button');button.textContent='连接设置';button.addEventListener('click',openSettings);$('scene-notice').append(button);}
}
try {
  universe=new Universe($('universe'),{onSelect:selectWorker,onGateway:showOverview,onHover:showTooltip,onError:message=>notice(message)});
} catch (error) {
  console.error('Unable to initialize WebGL',error);
  const fallback=document.createElement('div');fallback.className='webgl-error';fallback.innerHTML='<strong>星空暂时无法点亮</strong>浏览器未能创建 WebGL 2 渲染环境。<br>请开启硬件加速或更换支持 WebGL 的浏览器。<br>左侧列表和实时数据仍可使用。';$('universe').append(fallback);
}

function selectedWorker(){return state.data?.workers.find(w=>w.id===state.selected);}
function selectWorker(id,groupId){state.selected=id;state.group=groupId||state.data?.workers.find(w=>w.id===id)?.modelIds[0]||null;panels.right=false;panels.immersive=false;if(matchMedia('(max-width:640px)').matches)panels.left=true;applyPanels();document.body.classList.add('detail-open');universe?.focusWorker(id,state.group);renderTree();renderDetail();$('scene-caption-text').textContent='WORKER FOCUS · 点击空白并拖拽环绕查看';}
function selectGroup(id){state.selected=null;state.group=id;document.body.classList.remove('detail-open');universe?.focusGroup(id);renderTree();renderDetail();$('scene-caption-text').textContent=`${id} · 星系视角`;}
function showOverview(){state.selected=null;state.group=null;document.body.classList.remove('detail-open');universe?.resetView();renderTree();renderDetail();$('scene-caption-text').textContent='探索你的推理宇宙';}

function renderTree(){
  if(!state.data)return;
  const active=document.activeElement;const focusKey=active?.closest('#galaxy-list')?{action:active.dataset.action,id:active.dataset.id,group:active.dataset.group}:null;
  const query=state.search.toLowerCase();let visible=0;
  $('galaxy-list').innerHTML=state.data.groups.map(group=>{
    const matches=group.id.toLowerCase().includes(query);
    const workers=group.workers.filter(w=>matches||`${w.address} ${w.runtime} ${w.labels['smg.ai/pod-name']||''}`.toLowerCase().includes(query));
    if(!workers.length)return '';visible+=workers.length;const heats=relativeLoads(group.workers),range=heats.values().next().value;
    const closed=state.collapsed.has(group.id)&&!query;
    return `<div class="model-group ${state.group===group.id?'active':''} ${closed?'collapsed':''}" style="--group-color:${group.color}"><div class="model-header"><button class="group-toggle" data-action="toggle" data-id="${esc(group.id)}" aria-label="${closed?'展开':'收起'} ${esc(group.id)}" aria-expanded="${!closed}">▼</button><button class="model-focus" data-action="group" data-id="${esc(group.id)}" title="定位 ${esc(group.id)} 星系"><span class="model-symbol">✦</span><span class="model-name">${esc(group.id)}</span><span class="model-count">${group.workers.length.toString().padStart(2,'0')}</span></button></div><div class="worker-tree">${workers.map(w=>{
      const color=heatColor(heats.get(w.id).heat);return `<button class="worker-row ${state.selected===w.id&&state.group===group.id?'active':''}" data-action="worker" data-id="${esc(w.id)}" data-group="${esc(group.id)}" aria-label="查看 ${esc(w.address)}，在途 ${fmt(w.load)}" title="${esc(w.labels['smg.ai/pod-name']||w.id)}"><i class="worker-dot ${w.status}" style="--heat-color:${color}"></i><span class="worker-text"><strong>${esc(w.address)}</strong><small>${esc(w.runtime)} · ${w.status==='offline'?'不可用':w.waiting>0?`排队 ${w.waiting}`:'在线'}</small></span><span class="worker-load" style="color:${color}">${fmt(w.load)}<small>INFLIGHT</small></span></button>`;
    }).join('')}</div>${closed?'':`<div class="worker-heat-note">组内负载 ${range?.min??0} — ${range?.max??0} <span style="color:#70dce6">●</span> → <span style="color:#ff656c">●</span></div>`}</div>`;
  }).join('')||`<div class="empty-list">${state.data.workers.length?'没有匹配的模型或节点':'尚未发现 worker<br>等待节点加入星网'}</div>`;
  $('visible-count').textContent=`${visible} NODES`;
  if(focusKey){const replacement=[...$('galaxy-list').querySelectorAll('button')].find(b=>b.dataset.action===focusKey.action&&b.dataset.id===focusKey.id&&b.dataset.group===focusKey.group);replacement?.focus({preventScroll:true});}
}
function metricValue(metric){return `${fmt(metric.value)}${metric.partial?'*':''}`;}
function renderSummary(){
  if(!state.data)return;const stats=summarize(state.data.workers);
  $('stat-models').textContent=state.data.groups.length;
  $('galaxy-count').textContent=String(state.data.groups.length).padStart(2,'0');
  $('stat-workers').textContent=stats.healthy;
  $('stat-health').textContent=`/ ${stats.total} 节点`;
  $('stat-inflight').textContent=metricValue(stats.inflight);
  $('stat-queued').textContent=metricValue(stats.waiting);
  $('stat-queued').style.color=stats.waiting.value>0?'var(--gold)':'';
  $('queue-description').textContent=stats.waiting.partial?'* 部分数据缺失':stats.waiting.value>0?'请求等待中':'队列畅通';
}
function renderDetail(){
  const worker=selectedWorker();const group=state.data?.groups.find(g=>g.id===state.group);
  if(!worker){
    const workers=group?.workers||state.data?.workers||[];const stats=summarize(workers);
    $('detail-title').textContent=group?'模型观测站':'网关观测站';$('detail-icon').textContent=group?'✦':'◎';
    $('worker-detail').innerHTML=`<div class="detail-hero"><div class="gateway-emblem" ${group?`style="color:${group.color};border-color:${group.color}55"`:''}>${group?'✦':'◎'}</div><div><strong>${group?esc(group.id):'ai-gateway'}</strong><small>${group?'MODEL CONSTELLATION':'XHIVE-PRO · CENTRAL CORE'}</small></div></div><p class="gateway-description">${group?'这个星系中的星球服务于同一个模型。点击左侧节点，或直接选择场景中的星球。':'所有推理链路的起点。模型汇聚成星系，worker 化为星球，在同一个智能宇宙中协同运转。'}</p><div class="detail-stats"><div><strong>${state.data?fmt(stats.total):'—'}</strong><span>节点</span></div><div><strong>${state.data?metricValue(stats.running):'—'}</strong><span>执行中</span></div><div><strong>${state.data?metricValue(stats.waiting):'—'}</strong><span>排队中</span></div></div><div class="orbit-readout"><span>LINK STATUS</span><span>${state.connection==='live'?'CONNECTED':state.connection==='demo'?'SIMULATED':state.connection==='error'?'STALE / OFFLINE':'CONNECTING'}</span></div><div class="detail-section"><dl class="detail-rows"><dt>数据来源</dt><dd>${state.mode==='demo'?'演示数据':'ai-gateway /workers'}</dd><dt>节点着色</dt><dd>同星系 · 相对在途负载</dd><dt>星球光环</dt><dd>KV Cache 使用比例</dd><dt>空间导航</dt><dd>拖拽旋转 · 滚轮缩放</dd></dl><p class="detail-note">蓝 → 琥珀 → 红代表组内负载由低到高。颜色不等同于过载判定。${stats.running.partial?'* 部分 worker 未提供引擎快照。':''}</p></div>${group?'<button class="detail-close" data-action="overview">← 返回网关全景</button>':''}`;
    return;
  }
  const heat=relativeLoads(group?.workers||[worker]).get(worker.id);const color=heatColor(heat.heat);
  $('detail-title').textContent='节点观测站';$('detail-icon').textContent='◉';$('detail-icon').style.color=color;
  const status=worker.status==='offline'?'节点不可用':worker.waiting>0?`在线 · ${worker.waiting} 个请求等待`:'在线 · 队列畅通';
  const timestamp=worker.engineTimestamp?new Date(worker.engineTimestamp).toLocaleTimeString('zh-CN',{hour12:false}):'后端未提供';
  const memory=worker.raw.engine_load?.loads?.[0]?.memory;
  $('worker-detail').innerHTML=`<div class="detail-hero"><div class="gateway-emblem" style="color:${color};border-color:${color}55;background:radial-gradient(circle,${color}25,transparent 70%)">◉</div><div><strong>${esc(worker.runtime)} <span style="font:9px var(--mono);color:#6c819b">${esc(worker.version)}</span></strong><small>${esc(worker.address)}</small></div></div><div class="detail-status ${worker.status}"><i></i>${status}</div><div class="detail-stats"><div><strong>${fmt(worker.load)}</strong><span>网关在途</span></div><div><strong>${fmt(worker.running)}</strong><span>引擎执行</span></div><div><strong style="color:${worker.waiting>0?'var(--gold)':'inherit'}">${fmt(worker.waiting)}</strong><span>引擎排队</span></div></div><div class="detail-section"><div class="detail-label"><span>KV CACHE 占用</span><b style="color:${worker.kv>.8?'var(--gold)':'var(--cyan)'}">${pct(worker.kv)}</b></div><div class="meter"><i style="width:${(worker.kv??0)*100}%;background:${worker.kv>.8?'var(--gold)':'var(--cyan)'}"></i></div><div class="detail-label"><span>${worker.capacity?`${fmt(worker.usedTokens)} / ${fmt(worker.capacity)} tokens`:'绝对 token 容量未提供'}</span></div><p class="detail-note">组内在途范围 ${heat.min}–${heat.max}，当前 ${fmt(worker.load)}。${heat.min===heat.max?'本组负载相同，统一使用冰蓝色。':'颜色随同组 worker 相对负载变化。'}</p></div><div class="detail-divider"></div><div class="detail-section"><dl class="detail-rows"><dt>所属模型</dt><dd>${esc(state.group||worker.modelIds[0])}</dd><dt>运行状态</dt><dd>${esc(worker.raw.status||'unknown')}</dd><dt>等待 token</dt><dd class="mono">${fmt(worker.waitingTokens)}</dd><dt>引擎并发上限</dt><dd class="mono">${worker.maxRunning||'未提供'}</dd>${memory?`<dt>权重 / KV</dt><dd class="mono">${fmt(memory.weight_gb)} / ${fmt(memory.kv_cache_gb)} GB</dd>`:''}<dt>引擎采样时间</dt><dd class="mono">${timestamp}</dd><dt>采集状态</dt><dd>${state.connection==='error'?'连接中断 · 上次数据':state.mode==='demo'?'模拟快照':'最近一次返回'}</dd></dl><div class="capabilities">${worker.capabilities.map(c=>`<span>${esc(c)}</span>`).join('')}</div><p class="detail-note">引擎快照与 网关在途计数的时间和口径不同；未提供的值显示为 —。本接口不提供历史成功 / 失败累计。</p></div><div class="node-id">${esc(worker.labels['smg.ai/pod-name']||worker.id)}</div><button class="detail-close" data-action="overview">← 返回网关全景</button>`;
}
function renderChart(){
  const history=state.history.filter(p=>p.value!==null);if(!history.length){$('chart-line').setAttribute('d','');$('chart-area').setAttribute('d','');$('chart-current').textContent='—';$('history-window').textContent='本页观测 · 等待数据';return;}
  const max=Math.max(5,...history.map(h=>h.value))*1.15;const span=Math.max(state.interval,history.at(-1).time-history[0].time);const points=history.map(p=>({x:(p.time-history[0].time)/span*560,y:55-p.value/max*48}));
  if(points.length===1)points[0].x=560;
  const d=points.map((p,i)=>`${i?'L':'M'}${p.x.toFixed(2)},${p.y.toFixed(2)}`).join(' ');
  $('chart-line').setAttribute('d',d);$('chart-area').setAttribute('d',points.length>1?`${d} L${points.at(-1).x},58 L${points[0].x},58 Z`:'');
  $('chart-current').textContent=fmt(history.at(-1).value);
  const seconds=Math.round((history.at(-1).time-history[0].time)/1000);$('history-window').textContent=`本页观测 · ${seconds<60?`${seconds} 秒`:`${Math.floor(seconds/60)} 分 ${seconds%60} 秒`}`;
}
function renderConnection(){
  const badge=$('connection-badge');badge.className=`connection-badge ${state.connection}`;
  badge.querySelector('span').textContent={connecting:'连接中',live:'LIVE · 已连接',error:state.data?'数据已过期':'连接失败',demo:'DEMO · 演示'}[state.connection];
  document.body.classList.toggle('demo-mode',state.mode==='demo');
  $('source-label').textContent=state.mode==='demo'?'DEMO / SIMULATED':state.connection==='error'?'STALE / OFFLINE':'LIVE SOURCE';
  $('endpoint-label').textContent=state.mode==='demo'?'DEMO UNIVERSE':new URL(state.endpoint).host;
  $('endpoint-label').title=state.endpoint;$('poll-label').textContent=`每 ${state.interval/1000} 秒`;
  if(state.connection==='demo')notice('演示宇宙 · 当前星系和负载均为模拟数据。切换“实时星网”查看你的集群。',true);
  else if(state.connection==='error')notice(`${state.error}${state.data?' 保留最后一次成功采集的数据。':''}`,true);
  else if(state.connection==='live'&&state.data?.workers.length===0)notice('连接成功，当前没有注册的 worker。节点加入后会自动出现在星图中。');
  else notice('');
  updateClock();
}
function applyData(payload){
  const data=normalizePayload(payload);state.data=data;state.lastSuccess=Date.now();
  if(state.selected&&!data.workers.some(w=>w.id===state.selected)){state.selected=null;document.body.classList.remove('detail-open');}
  if(state.group&&!data.groups.some(g=>g.id===state.group))state.group=null;
  for(const id of state.collapsed)if(!data.groups.some(g=>g.id===id))state.collapsed.delete(id);
  state.history.push({time:Date.now(),value:summarize(data.workers).inflight.value});if(state.history.length>120)state.history.shift();
  universe?.setData(data);renderSummary();renderTree();renderDetail();renderChart();renderConnection();
}
async function refresh(){
  clearTimeout(timer);controller?.abort();const current=++generation;controller=new AbortController();
  if(document.hidden){timer=setTimeout(refresh,state.interval);return;}
  $('refresh-now').disabled=true;
  const requestController=controller;
  const timeout=setTimeout(()=>requestController.abort(),8000);
  try{
    if(state.mode==='demo'){state.connection='demo';applyData(demoPayload(Date.now()));}
    else{
      const response=await fetch(state.endpoint,{signal:controller.signal,mode:'cors',cache:'no-store',credentials:'omit',headers:{Accept:'application/json'}});
      if(!response.ok)throw new Error(`ai-gateway 返回 HTTP ${response.status}。`);
      const payload=await response.json();if(current!==generation)return;
      // Validate before claiming the new response is live.
      normalizePayload(payload);state.connection='live';state.error='';applyData(payload);
    }
  }catch(error){if(current!==generation)return;state.connection='error';state.error=error.name==='AbortError'?'请求超过 8 秒未完成。':error instanceof TypeError?'无法连接 ai-gateway，请检查网络、跨域设置或浏览器的本地网络访问权限。':error instanceof SyntaxError?'接口返回的内容不是有效 JSON。':error.message;renderConnection();renderDetail();}
  finally{clearTimeout(timeout);if(current===generation){$('refresh-now').disabled=false;timer=setTimeout(refresh,state.interval);}}
}
function openSettings(){ $('endpoint-input').value=state.endpoint;$('interval-input').value=state.interval;document.querySelector(`input[name="mode"][value="${state.mode}"]`).checked=true;$('settings-error').hidden=true;$('settings-dialog').showModal(); }
function updateClock(){const now=new Date();$('local-time').textContent=now.toLocaleTimeString('en-GB',{hour12:false});if(!state.lastSuccess){$('last-update').textContent='等待首次采集';return;}const age=Math.max(0,Math.floor((Date.now()-state.lastSuccess)/1000));$('last-update').textContent=`${state.connection==='error'?'上次成功':'更新于'} ${age<2?'刚刚':`${age} 秒前`}`;}
function showTooltip(worker,x,y,groupId){const tooltip=$('tooltip');if(!worker){tooltip.hidden=true;return;}const group=state.data?.groups.find(g=>g.id===groupId),heat=relativeLoads(group?.workers||[worker]).get(worker.id);tooltip.innerHTML=`${esc(worker.address)}<br><b>${esc(worker.runtime)}</b> · 在途 ${fmt(worker.load)}<small>执行 ${fmt(worker.running)} / 排队 ${fmt(worker.waiting)} · KV ${pct(worker.kv)}</small><small>组内负载范围 ${heat.min}–${heat.max} · 点击定位</small>`;tooltip.hidden=false;tooltip.style.left=`${Math.max(8,Math.min(x+16,innerWidth-tooltip.offsetWidth-12))}px`;tooltip.style.top=`${Math.max(8,Math.min(y+14,innerHeight-tooltip.offsetHeight-12))}px`;}
function renderMotionButton(){const button=$('motion-toggle');button.setAttribute('aria-pressed',state.paused);button.setAttribute('aria-label',state.paused?'恢复公转':'暂停公转');button.title=state.paused?'恢复公转':'暂停公转';$('motion-icon').innerHTML=state.paused?'<path d="m8 5 10 7-10 7z"/>':'<path d="M9 5v14M15 5v14"/>';}

$('galaxy-list').addEventListener('click',e=>{const button=e.target.closest('button[data-action]');if(!button)return;const {action,id,group}=button.dataset;if(action==='toggle'){state.collapsed.has(id)?state.collapsed.delete(id):state.collapsed.add(id);renderTree();}else if(action==='group')selectGroup(id);else selectWorker(id,group);});
$('worker-detail').addEventListener('click',e=>{if(e.target.closest('[data-action="overview"]'))showOverview();});
$('model-search').addEventListener('input',e=>{state.search=e.target.value;renderTree();});
for(const id of ['overview-button','show-all','reset-view'])$(id).addEventListener('click',showOverview);
$('zoom-in').addEventListener('click',()=>universe?.zoom(.8));$('zoom-out').addEventListener('click',()=>universe?.zoom(1.25));
$('motion-toggle').addEventListener('click',()=>{state.paused=!state.paused;universe?.setPaused(state.paused);renderMotionButton();});
$('labels-toggle').addEventListener('click',()=>{state.labels=!state.labels;universe?.setLabels(state.labels);$('labels-toggle').setAttribute('aria-pressed',state.labels);});
$('refresh-now').addEventListener('click',refresh);$('settings-button').addEventListener('click',openSettings);$('close-settings').addEventListener('click',()=>$('settings-dialog').close());
$('settings-dialog').addEventListener('click',e=>{if(e.target===$('settings-dialog')){const r=e.target.getBoundingClientRect();if(e.clientX<r.left||e.clientX>r.right||e.clientY<r.top||e.clientY>r.bottom)e.target.close();}});
$('settings-form').addEventListener('submit',e=>{e.preventDefault();try{const nextEndpoint=validateEndpoint($('endpoint-input').value.trim()),nextMode=document.querySelector('input[name="mode"]:checked').value;const sourceChanged=state.endpoint!==nextEndpoint||state.mode!==nextMode;state.endpoint=nextEndpoint;state.interval=Number($('interval-input').value);state.mode=nextMode;store.set('smg-orbital-settings',JSON.stringify({endpoint:state.endpoint,interval:state.interval,mode:state.mode}));if(sourceChanged){state.data=null;state.selected=null;state.group=null;state.lastSuccess=null;state.history=[];state.collapsed.clear();state.connection='connecting';showOverview();universe?.setData({workers:[],groups:[]});for(const id of ['stat-models','stat-workers','stat-inflight','stat-queued'])$(id).textContent='—';$('galaxy-list').innerHTML='<div class="empty-list">正在寻找星系…</div>';renderChart();renderConnection();}$('settings-dialog').close();refresh();}catch(error){$('settings-error').textContent=error.message;$('settings-error').hidden=false;}});
for(const side of ['left','right','bottom']){$(`collapse-${side}`).addEventListener('click',()=>setPanel(side,true));$(`expand-${side}`).addEventListener('click',()=>setPanel(side,false));}
$('immersive-toggle').addEventListener('click',toggleImmersive);
document.addEventListener('keydown',e=>{const editing=['INPUT','TEXTAREA','SELECT'].includes(document.activeElement?.tagName);if(e.key==='/'&&!editing&&!$('settings-dialog').open){e.preventDefault();panels.left=false;panels.immersive=false;applyPanels();$('model-search').focus();}if(e.key.toLowerCase()==='f'&&!editing&&!$('settings-dialog').open){e.preventDefault();toggleImmersive();}if(e.key==='Escape'&&!$('settings-dialog').open){showTooltip(null);if(panels.immersive){panels.immersive=false;applyPanels();}else if(state.selected||state.group)showOverview();}});
document.addEventListener('visibilitychange',()=>{if(!document.hidden)refresh();});
applyPanels();renderMotionButton();renderDetail();renderConnection();refresh();setInterval(updateClock,1000);
