/* Common process output viewer. No backend calls, HTML interpretation or invented timestamps. */
"use strict";
(() => {
  const MAX_ROWS = 2000;
  function terminalText(text) {
    return text.replace(/\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)/g, '')
      .replace(/\x1b\[[0-?]*[ -/]*[@-~]/g, '')
      .replace(/[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]/g, '')
      .replace(/\r\n/g, '\n').split('\n')
      .map(line => line.replace(/\r$/, '').split('\r').pop()).join('\n');
  }
  const levels = {info:'info', inf:'info', warn:'warn', warning:'warn', wrn:'warn', error:'error', err:'error', fatal:'error', critical:'error', debug:'debug', dbg:'debug', trace:'debug'};
  function parseLine(raw, index) {
    let message = raw, time = '', source = '', level = 'plain';
    const timestamp = message.match(/^\[?((?:\d{4}-\d{2}-\d{2}[T ])?\d{2}:\d{2}:\d{2}(?:[.,]\d{1,6})?(?:Z|[+-]\d{2}:\d{2})?)\]?\s+/);
    if (timestamp) {time = timestamp[1]; message = message.slice(timestamp[0].length);}
    function severity() {
      const match = message.match(/^(?:\[\s*(trace|debug|dbg|info|inf|warn|warning|wrn|error|err|fatal|critical)\s*\]|(trace|debug|dbg|info|inf|warn|warning|wrn|error|err|fatal|critical)(?=\s|:))\s*:?\s*/i);
      if (match) {level = levels[(match[1] || match[2]).toLowerCase()]; message = message.slice(match[0].length);}
    }
    severity();
    // NInfer's [module] / "module |", llama.cpp's "slot print_timing:",
    // and PolyStrata's [strata] all keep their source labels.
    const module = message.match(/^\[([\w./-]{1,64})\]\s*/)
      || message.match(/^([a-zA-Z_][\w.-]*(?:\s+[a-zA-Z_][\w.-]*)?)\s*(?::|\s\|)\s*/);
    if (module) {source = module[1]; message = message.slice(module[0].length); severity();}
    return {raw, index, time, source, level, message};
  }
  function parseLog(text) {
    const normalized = terminalText(text);
    const lines = normalized ? normalized.replace(/\n$/, '').split('\n').map((raw, index) => ({raw, index:index + 1})) : [];
    return {total:lines.length, rows:lines.slice(-MAX_ROWS).map(line => parseLine(line.raw, line.index))};
  }
  // Highlight in place, without removing prefixes, punctuation or whitespace.
  // Only explicit level labels receive severity colors; arbitrary prose is left alone.
  function logTokens(text) {
    const pattern = /\[?(?:\d{4}-\d{2}-\d{2}[T ])?\d{2}:\d{2}:\d{2}(?:[.,]\d{1,6})?(?:Z|[+-]\d{2}:\d{2})?\]?|\[[\w./-]{1,64}\]|\b(?:INFO|WARN|WARNING|ERROR|FATAL|CRITICAL|DEBUG|TRACE)\b|\b\d+(?:\.\d+)?\s*(?:tok\/s|tokens\/s|tokens per second|tokens?|[KMGT]i?B|ms|µs|us|seconds?|s)\b|\b\d+(?:\.\d+)?%/g;
    const tokens = []; let from = 0;
    for (const match of text.matchAll(pattern)) {
      if (match.index > from) tokens.push({text:text.slice(from, match.index),kind:''});
      const value = match[0], label = value.replace(/^\[|\]$/g,'').toLowerCase();
      const kind = /\d{2}:\d{2}:\d{2}/.test(value) ? 'time' : levels[label] || (value.startsWith('[') ? 'tag' : 'metric');
      tokens.push({text:value,kind}); from = match.index + value.length;
    }
    if (from < text.length) tokens.push({text:text.slice(from),kind:''});
    return tokens;
  }
  if (typeof module !== 'undefined' && module.exports) module.exports = {parseLog, terminalText, logTokens};
  if (typeof document === 'undefined') return;
  const root = document.querySelector('[data-view="logs"]');
  if (!root) return;
  const $ = selector => root.querySelector(selector);
  const names = {ninfer:'NInfer', llamacpp:'llama.cpp', polystrata:'PolyStrata'};
  const words = {
    title:['后台日志','Backend logs'], subtitle:['当前引擎的运行输出','Output from the running engine'],
    export:['导出日志','Export logs'], started:['本次启动','Started'], all:['全部','All'],
    info:['信息','Info'], warn:['警告','Warnings'], error:['错误','Errors'], debug:['调试','Debug'], plain:['未分级','Unclassified'],
    stream:['引擎输出 · stdout / stderr','Engine output · stdout / stderr'], styled:['高亮','Highlight'], raw:['纯文本','Plain text'], wrap:['自动换行','Wrap lines'],
    follow:['跟随最新输出','Follow latest output'], latest:['回到最新 ↓','Jump to latest ↓'],
    limit:['最近 2,000 行 / 256 KiB · 导出当前缓存原文','Latest 2,000 lines / 256 KiB · Export the cached original'],
    search:['搜索日志、模块或请求 ID…','Search logs, modules or request IDs…'],
    pause:['暂停刷新','Pause refresh'], resume:['继续刷新','Resume refresh'],
    paused:['刷新已暂停','Refresh paused'], live:['实时接收','Receiving output'],
    stopped:['已停止 · 保留最后日志','Stopped · Last output retained'], waiting:['等待引擎启动','Waiting for engine startup'],
    previous:['上次运行日志','Previous run'], unknown:['引擎信息未记录','Engine identity unavailable'],
    connecting:['正在连接','Connecting'], stale:['连接中断 · 显示上次数据','Disconnected · Showing previous output'],
    unsupported:['此服务未提供日志接口。请使用更新后的 KVMem 本地控制台打开后台日志。','This server has no log API. Open logs in the updated KVMem local console.'],
    retry:['暂时无法读取日志，将自动重试。','Unable to read logs. Retrying automatically.'],
    empty:['还没有日志。启动引擎后，输出会显示在这里。','No logs yet. Start an engine to see its output here.'],
    pending:['等待完整日志行…','Waiting for a complete log line…'],
    noMatch:['没有匹配的日志','No matching log entries'],
    noError:['当前缓存中没有错误日志','No error entries in the current buffer'],
    capped:['较早的日志已超出显示范围；当前仅保留最近一段。','Earlier output is outside the display limit; only the latest portion is retained.'],
    region:['日志内容','Log content'], engine:['运行引擎','Running engine'], filters:['日志级别','Log level'], console:['引擎日志','Engine logs']
  };
  const t = key => words[key][window.monitorI18n.language === 'en' ? 1 : 0];
  const tr = (zh,en) => window.monitorI18n.language === 'en' ? en : zh;
  let meta = null, rawText = '', rows = [], total = 0, revision = '', query = '', raw = false;
  let paused = false, follow = true, busy = false, timer, controller, failures = 0, error = '', scrolling = false;
  let legacy = false, runtime = null;
  const viewport = $('[data-log-scroll]'), content = $('[data-log-rows]');
  function make(tag, className, text) {
    const node = document.createElement(tag); if (className) node.className = className;
    if (text !== undefined) node.textContent = text; return node;
  }
  function localTime(ms) {
    return ms ? new Date(ms).toLocaleString(window.monitorI18n.language, {month:'2-digit', day:'2-digit', hour:'2-digit', minute:'2-digit', second:'2-digit'}) : '—';
  }
  function appearance() {
    document.querySelector('[data-page="logs"]').textContent = t('title');
    root.querySelectorAll('[data-log-label]').forEach(node => node.textContent = t(node.dataset.logLabel));
    $('[data-log-search]').placeholder = t('search');
    $('[data-log-search]').setAttribute('aria-label',t('search'));
    viewport.setAttribute('aria-label',t('region'));
    root.querySelector('.s-log-engine').setAttribute('aria-label',t('engine'));
    root.querySelector('.s-log-console').setAttribute('aria-label',t('console'));
    status(); render();
  }
  function status() {
    const state = $('[data-log-state]');
    state.dataset.state = error ? 'error' : paused ? 'paused' : meta?.running ? 'live' : 'stopped';
    state.textContent = error ? t('stale') : paused ? t('paused') : meta?.running ? t('live') : meta?.available ? t('stopped') : meta ? t('waiting') : t('connecting');
    if (!error && !paused && meta?.exit_code) {
      state.dataset.state = 'error';
      state.textContent = tr('进程已退出 · 退出码 ','Process exited · Code ') + meta.exit_code;
    }
    $('[data-log-backend]').textContent = names[meta?.backend] || t('previous');
    $('[data-log-monogram]').textContent = {ninfer:'N',llamacpp:'L',polystrata:'P'}[meta?.backend] || '—';
    $('[data-log-model]').textContent = meta?.model || (meta?.available ? t('unknown') : t('waiting'));
    $('[data-log-started]').textContent = localTime(meta?.started_at);
    $('[data-log-pause]').textContent = t(paused ? 'resume' : 'pause');
    $('[data-log-pause]').setAttribute('aria-pressed', String(paused));
    $('[data-log-export]').disabled = !rawText;
    $('[data-log-follow]').checked = follow;
    $('[data-log-latest]').hidden = follow;
    $('[data-log-updated]').textContent = meta?.updated_at ? tr('日志更新于 ','Log updated ') + localTime(meta.updated_at) : '';
    if (meta?.legacy) {
      $('[data-log-updated]').textContent = tr('兼容旧版控制台','Legacy console compatibility');
      $('[data-log-label="limit"]').textContent = tr('最近 2,000 行 / 32 KiB · 导出当前缓存原文','Latest 2,000 lines / 32 KiB · Export the cached original');
    }
    const notice = $('[data-log-notice]');
    notice.hidden = !error && !meta?.truncated && total <= MAX_ROWS;
    notice.textContent = error ? t(error) : t('capped');
  }
  function goBottom() {
    scrolling = true; viewport.scrollTop = viewport.scrollHeight;
    requestAnimationFrame(() => {scrolling = false;});
  }
  function highlight(node, text, needle) {
    // Calculate search ranges before tokenization, so matches can span colors.
    const ranges = [], lower = text.toLowerCase(); let at = 0;
    if (needle) while ((at = lower.indexOf(needle, at)) !== -1) {ranges.push([at, at + needle.length]); at += needle.length;}
    let offset = 0, rangeIndex = 0;
    for (const token of raw ? [{text,kind:''}] : logTokens(text)) {
      const span = token.kind ? make('span','s-log-token-' + token.kind) : document.createDocumentFragment();
      const end = offset + token.text.length; let from = offset;
      while (rangeIndex < ranges.length && ranges[rangeIndex][1] <= offset) rangeIndex++;
      for (let i = rangeIndex; i < ranges.length && ranges[i][0] < end; i++) {
        const start = Math.max(offset,ranges[i][0]), stop = Math.min(end,ranges[i][1]);
        span.append(document.createTextNode(text.slice(from,start)),make('mark','',text.slice(start,stop))); from = stop;
      }
      span.append(document.createTextNode(text.slice(from,end))); node.append(span); offset = end;
    }
  }
  function render(reset = false) {
    const needle = query.trim().toLowerCase();
    const visible = rows.filter(row => !needle || row.raw.toLowerCase().includes(needle));
    root.classList.toggle('s-log-plain', raw);
    let anchor;
    if (!follow && !reset) {
      const top = viewport.getBoundingClientRect().top;
      const row = [...content.children].find(node => node.logRaw !== undefined && node.getBoundingClientRect().bottom > top);
      if (row) anchor = {raw:row.logRaw, top:row.getBoundingClientRect().top};
    }
    const scrollTop = viewport.scrollTop, fragment = document.createDocumentFragment();
    if (!visible.length) {
      fragment.append(make('div','s-log-empty', rows.length ? t('noMatch') : t(meta?.pending ? 'pending' : 'empty')));
    } else {
      for (const row of visible) {
        const item = make('div','s-log-row'); item.dataset.level = row.level; item.logRaw = row.raw;
        const message = make('span','s-log-message'); highlight(message,row.raw,needle);
        item.append(message,document.createTextNode('\n'));
        fragment.append(item);
      }
    }
    content.replaceChildren(fragment);
    root.querySelectorAll('[data-log-format]').forEach(button => button.setAttribute('aria-pressed',String(raw === (button.dataset.logFormat === 'raw'))));
    $('[data-log-summary]').textContent = tr('显示 ','Showing ') + visible.length.toLocaleString() + ' / ' + rows.length.toLocaleString() + tr(' 行',' lines');
    if (follow) goBottom();
    else if (reset) viewport.scrollTop = 0;
    else {
      viewport.scrollTop = scrollTop;
      const replacement = anchor && [...content.children].find(node => node.logRaw === anchor.raw);
      if (replacement) viewport.scrollTop += replacement.getBoundingClientRect().top - anchor.top;
    }
  }
  function active() {return !root.hidden && !document.hidden && !paused;}
  async function snapshot(signal) {
    if (!legacy) {
      const response = await fetch('/api/launcher/logs' + (revision ? '?revision=' + encodeURIComponent(revision) : ''), {cache:'no-store',signal});
      if (response.status !== 404) {
        if (!response.ok) throw new Error('logs');
        return response.json();
      }
      // Older running controllers can serve the new view without killing their
      // owned worker. Their existing tail API is explicitly limited to 32 KiB.
      legacy = true;
    }
    const response = await fetch('/api/launcher/log', {cache:'no-store',signal});
    if (!response.ok) {const failure = new Error('logs'); failure.unsupported = response.status === 404; throw failure;}
    const body = await response.json();
    if (typeof body.text !== 'string') throw new Error('Invalid log tail');
    const p = runtime?.active_config?.parameters, running = ['starting','ready'].includes(runtime?.state);
    let text = body.text;
    if (running && !/[\r\n]$/.test(text)) text = text.slice(0,Math.max(text.lastIndexOf('\n'),text.lastIndexOf('\r')) + 1);
    return {protocol:1, legacy:true, session:p ? runtime.instance : meta?.session || 'previous', revision:'legacy',
      backend:p?.Backend || meta?.backend, model:p?.Model?.split(/[\\/]/).pop() || meta?.model,
      running, exit_code:runtime?.exit_code, available:!!text, limit_bytes:32768, text,
      truncated:new TextEncoder().encode(text).length >= 32768};
  }
  async function poll() {
    clearTimeout(timer);
    if (!active() || busy) return;
    busy = true;
    const request = controller = new AbortController(), timeout = setTimeout(() => request.abort(), 5000);
    try {
      const next = await snapshot(request.signal);
      if (next.protocol !== 1 || typeof next.revision !== 'string' || (next.text !== null && typeof next.text !== 'string')) throw new Error('Invalid log snapshot');
      if (!active()) return;
      const restarted = meta && meta.session !== next.session;
      meta = next; error = ''; failures = 0; revision = next.revision;
      if (next.text !== null) {
        const changed = rawText !== next.text || restarted;
        rawText = next.text;
        if (changed) {
          ({rows,total} = parseLog(rawText));
          if (restarted) follow = true;
          render(!!restarted);
        } else if (!rows.length) render();
      }
      status();
    } catch (failure) {
      if (active()) {error = failure.unsupported ? 'unsupported' : 'retry'; failures++; status();}
    } finally {
      clearTimeout(timeout); busy = false; controller = null;
      if (active()) timer = setTimeout(poll,Math.min(1000 * 2 ** failures,10000));
    }
  }
  function wake() {
    clearTimeout(timer);
    if (!active()) {controller?.abort(); return;}
    poll();
  }
  $('[data-log-search]').addEventListener('input',event => {query = event.target.value; render(true);});
  $('[data-log-wrap]').addEventListener('change',event => {root.classList.toggle('s-log-nowrap',!event.target.checked); if (follow) goBottom();});
  root.querySelectorAll('[data-log-format]').forEach(button => button.addEventListener('click',() => {raw = button.dataset.logFormat === 'raw'; render(true);}));
  $('[data-log-pause]').addEventListener('click',() => {paused = !paused; status(); wake();});
  $('[data-log-follow]').addEventListener('change',event => {follow = event.target.checked; status(); if (follow) goBottom();});
  $('[data-log-latest]').addEventListener('click',() => {follow = true; status(); goBottom();});
  viewport.addEventListener('scroll',() => {
    if (follow && !scrolling && viewport.scrollHeight - viewport.clientHeight - viewport.scrollTop > 28) {follow = false; status();}
  },{passive:true});
  $('[data-log-export]').addEventListener('click',() => {
    const url = URL.createObjectURL(new Blob([rawText],{type:'text/plain;charset=utf-8'})), link = make('a');
    link.href = url; link.download = 'kvmem-' + (Object.hasOwn(names,meta?.backend) ? meta.backend : 'engine') + '-' + new Date().toISOString().replace(/[:.]/g,'-') + '.log';
    link.click(); setTimeout(() => URL.revokeObjectURL(url),1000);
  });
  window.addEventListener('kvmem-page-change',wake);
  window.addEventListener('kvmem-runtime-status',event => {runtime = event.detail;});
  document.addEventListener('visibilitychange',wake);
  window.addEventListener('kvmem-language-change',appearance);
  const resize = new ResizeObserver(() => {if (follow && !root.hidden) goBottom();});
  resize.observe(viewport); resize.observe(content);
  window.addEventListener('pagehide',() => {clearTimeout(timer); controller?.abort();});
  window.addEventListener('pageshow',wake);
  appearance(); wake();
  document.documentElement.dataset.logsReady = "true";
})();
