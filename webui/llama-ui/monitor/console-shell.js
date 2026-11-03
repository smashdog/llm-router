/* One persistent chat view; navigation does not destroy its streaming session. */
(() => {
  const frame=document.querySelector('#chat-frame');
  if(!frame)return;
  const appearance=document.querySelector('.s-appearance');
  if(appearance){
    const narrow=matchMedia('(max-width:760px)');
    const resize=()=>{appearance.open=!narrow.matches;};
    resize();narrow.addEventListener('change',resize);
    const close=event=>{if(narrow.matches&&!appearance.contains(event.target))appearance.open=false;};
    const escape=event=>{if(narrow.matches&&appearance.open&&event.key==='Escape'){appearance.open=false;appearance.querySelector('summary').focus();}};
    document.addEventListener('pointerdown',close);document.addEventListener('keydown',escape);
    frame.addEventListener('load',()=>{
      frame.contentDocument?.addEventListener('pointerdown',()=>{if(narrow.matches)appearance.open=false;});
    });
  }
  new ResizeObserver(entries=>document.querySelector('#app').style.setProperty('--console-header-height',entries[0].borderBoxSize[0].blockSize+'px')).observe(document.querySelector('.s-header'));
  const sync=()=>{
    if(frame.contentDocument?.querySelector('#app')) {
      const host=document.querySelector('#app'),child=frame.contentDocument.querySelector('#app');
      child.dataset.theme=host.dataset.theme;child.dataset.density=host.dataset.density;
    }
    frame.contentWindow?.postMessage({type:'kvmem-appearance',theme:document.querySelector('#app').dataset.theme,
      density:document.querySelector('#app').dataset.density,language:document.documentElement.lang},location.origin);
  };
  function open(name,replace=false){
    document.querySelector('#app').dataset.currentPage=name;
    window.dispatchEvent(new CustomEvent('kvmem-navigate',{detail:name}));
    const url=name==='chat'?'/chat/':name==='monitor'?'/monitor/':'/monitor/#'+name;
    if(location.pathname+location.hash!==url)history[replace?'replaceState':'pushState']({},'',url);
  }
  window.addEventListener('kvmem-page-change',event=>{
    if(event.detail==='chat'&&!frame.getAttribute('src')) frame.src=frame.dataset.src;
    sync();
  });
  document.querySelectorAll('[data-page]').forEach(el=>el.addEventListener('click',()=>open(el.dataset.page)));
  document.querySelector('[data-chat-link]').addEventListener('click',e=>{e.preventDefault();open('chat');});
  new MutationObserver(sync).observe(document.querySelector('#app'),{attributes:true,attributeFilter:['data-theme','data-density']});
  window.addEventListener('kvmem-language-change',sync);
  frame.addEventListener('load',sync);
  window.addEventListener('message',event=>{
    if(event.origin!==location.origin||event.source!==frame.contentWindow)return;
    if(event.data?.type==='kvmem-navigate'&&['config','monitor','chat'].includes(event.data.page))open(event.data.page);
    if(event.data?.type==='kvmem-chat-ready')sync();
  });
  const current=()=>location.pathname.startsWith('/chat')?'chat':['config','about','logs'].includes(location.hash.slice(1))?location.hash.slice(1):'monitor';
  window.addEventListener('popstate',()=>open(current(),true));
  // Launcher initialization may already have selected config before this
  // deferred script loads. Preserve that selection on an unqualified URL.
  const selected=document.querySelector('[data-page][aria-pressed="true"]')?.dataset.page;
  open(location.pathname.startsWith('/chat')||location.hash?current():selected||current(),true);
  document.documentElement.dataset.shellReady = "true";
})();
