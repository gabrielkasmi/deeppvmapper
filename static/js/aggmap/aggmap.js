/* aggmap.js — aggregated viewer of OpenPVMapper (PV installations on a square grid, optional INSEE covariates).
   Mount: <div class="aggmap" data-mode="simple|full" data-base="…/static/data/agg/"> + <script class="am-i18n" type="application/json">.
   Data (built by scripts/build_agg.py): meta.json, outline.json, pv.bin.gz, pvcls.bin.gz, insee.bin.gz; the tables are fetched on demand.
   No dependency. Grid: INSPIRE 1 km cells (EPSG:3035) pre-summed in 2×2 blocks; every stored column is a sum, ratios are sum/sum. */
(function(){
"use strict";
const roots=document.querySelectorAll('.aggmap'); if(!roots.length) return;
const start=r=>{ if(r._am) return; r._am=true; init(r).catch(e=>{ console.error('[aggmap]',e);
  const o=r.querySelector('.am-loading'); let I={}; try{ I=JSON.parse(r.querySelector('script.am-i18n').textContent); }catch(_){}
  if(o){ o.classList.remove('off'); o.textContent=I.loadErr||'Error'; } }); };
if('IntersectionObserver' in window){
  const io=new IntersectionObserver(es=>es.forEach(e=>{ if(e.isIntersecting){ io.unobserve(e.target); start(e.target); } }),{rootMargin:'300px'});
  roots.forEach(r=>io.observe(r));
} else roots.forEach(start);

/* ---------- interface ---------- */
function ui(I,FULL){
  const sel=(id,l)=>`<label class="am-ctl"><span class="am-lab" data-id="${id}Lab">${l}</span><select data-id="${id}"></select></label>`;
  const seg=(id,lab,btns)=>`<div class="am-ctl"><span class="am-lab" data-id="${id}Lab">${lab}</span><div class="am-seg" data-id="${id}" role="group" aria-label="${lab}">${btns.map(([v,t],i)=>`<button type="button" data-v="${v}" aria-pressed="${i===0}">${t}</button>`).join('')}</div></div>`;
  const res=`<div class="am-ctl"><label class="am-lab"><span>${I.res}</span><span class="am-val" data-id="resVal"></span></label><input type="range" data-id="res" min="0" max="1000" step="1" value="0" aria-label="${I.res}"><div class="am-ticks" data-id="resTicks"></div></div>`;
  const stage=`<div class="am-stage" data-id="stage">
    <canvas class="am-map" data-id="map" role="img" aria-label="${I.mapAria}"></canvas>
    <canvas class="am-legend" data-id="legend"></canvas>
    <div class="am-cap"><div class="am-capres" data-id="capRes"></div><div class="am-capvar" data-id="capVar"></div><div class="am-capnote" data-id="capNote"></div></div>
    <div class="am-attr" data-id="attr"></div>
    <div class="am-zoom"><button type="button" data-id="zin" aria-label="${I.zin}">+</button><button type="button" data-id="zout" aria-label="${I.zout}">−</button><button type="button" data-id="zfit" aria-label="${I.zfit}">⌂</button></div>
    <div class="am-tip" data-id="tip"></div>
    <div class="am-wheel" data-id="wheel">${I.wheelHint}</div>
    <div class="am-loading" data-id="loading">${I.loading}</div>
  </div>`;
  if(!FULL) return `<div class="am-wrap">${stage}<div class="am-panel">${res}${sel('selA',I.variable)}${sel('denA',I.normBy)}<label class="am-chk" style="grid-column:1/-1"><input type="checkbox" data-id="chkBase" checked> ${I.baseLabel}</label><p class="am-hint" data-id="resHint"></p></div></div>`;
  const adv=`<details class="am-adv" data-id="adv"><summary data-id="advSum">${I.adv}</summary><div class="am-advbody">
    <p class="am-hint">${I.advHint}</p>
    <div class="am-ctl"><span class="am-lab">${I.clsLabel}</span><div class="am-clsgrid">${I.classes.map((c,i)=>`<label class="am-chk"><input type="checkbox" data-id="cls${i}" data-c="${i}" checked> ${c.short}</label>`).join('')}</div><p class="am-hint">${I.clsHint}</p></div>
    <label class="am-chk" data-id="lblHc"><input type="checkbox" data-id="chkHc"> ${I.hc}</label>
    ${seg('segM',I.display,[['one',I.dispOne],['biv',I.dispBiv]])}
    <div class="am-ctl" data-id="boxB" hidden>${sel('selB',I.explVar)}${sel('denB',I.normBy)}</div>
    <div class="am-ctl" data-id="boxScale">${seg('segS',I.scale,[['log',I.scLog],['lin',I.scLin],['quant',I.scQuant]])}<p class="am-hint" data-id="scaleHint"></p></div>
    <div class="am-scatterbox" data-id="scatterBox"><span class="am-lab">${I.scatterTitle}</span><canvas data-id="scatter" aria-label="${I.scatterAria}"></canvas><div class="am-rho" data-id="rho"></div></div>
    <div class="am-ctl"><label class="am-lab"><span>${I.opacity}</span><span class="am-val" data-id="alphaVal"></span></label><input type="range" data-id="alpha" min="20" max="100" step="5" value="80" aria-label="${I.opacity}"></div>
  </div></details>`;
  return `<div class="am-wrap">${stage}<div class="am-panel">
    <p class="am-lead" data-id="lead"></p>${res}<p class="am-hint" data-id="resHint"></p>
    ${sel('selA',I.variable)}${sel('denA',I.normBy)}
    <label class="am-chk"><input type="checkbox" data-id="chkBase" checked> ${I.baseLabel}</label><p class="am-hint" data-id="baseHint"></p>
    ${adv}
    <div class="am-ctl"><span class="am-lab">${I.exportTitle}</span><div class="am-exp"><button type="button" class="am-btn" data-id="btnPng">${I.png}</button><button type="button" class="am-btn2" data-id="btnCsv" title="${I.csvTitle}">${I.csv}</button><button type="button" class="am-btn2" data-id="btnGeo" title="${I.geoTitle}">${I.geo}</button></div><p class="am-hint">${I.exportHint}</p></div>
    <p class="am-foot"><span>${I.foot1}</span><span>${I.foot2}</span></p>
  </div></div>`;
}

async function init(root){
  const I=JSON.parse(root.querySelector('script.am-i18n').textContent);
  if(typeof DecompressionStream==='undefined'){ root.textContent=I.oldBrowser; return; }
  const FULL=root.dataset.mode==='full', DATA=root.dataset.base;
  root.insertAdjacentHTML('beforeend',ui(I,FULL));
  const ghost=document.createElement('select');          // absorbs the calls on controls that this mode does not have
  const $=id=>root.querySelector(`[data-id="${id}"]`)||ghost;
  const tr=(s,o)=>s.replace(/\{(\w+)\}/g,(m,k)=>o&&k in o?o[k]:m);
  const NF=(v,d=0)=>v.toLocaleString(I.locale,{maximumFractionDigits:d,minimumFractionDigits:0});
  const FONT=getComputedStyle(root).fontFamily;
  const getJSON=async u=>{ const r=await fetch(u); if(!r.ok) throw new Error(u+' '+r.status); return r.json(); };

  /* ---------- 1. données : méta, contour, tables creuses chargées à la demande ---------- */
  const META=await getJSON(DATA+'meta.json');
  const VQ=META.version?'?v='+encodeURIComponent(META.version):'';
  const {W,H,x0,y0}=META, TABS=META.tables, VARTAB={};
  const unshuf=(buf,off,n,sz)=>{ const pl=new Uint8Array(buf,off,n*sz), raw=new Uint8Array(n*sz);
    for(let j=0;j<sz;j++){ const o=j*n; for(let i=0;i<n;i++) raw[i*sz+j]=pl[o+i]; } return raw.buffer; };
  function loadTable(name){
    const t=TABS[name]; if(t.v) return Promise.resolve(); if(t.p) return t.p;
    return t.p=(async()=>{
      const r=await fetch(DATA+t.file+VQ); if(!r.ok) throw new Error(t.file+' '+r.status);
      const ab=await r.arrayBuffer(), u=new Uint8Array(ab);          // un serveur peut déjà avoir décodé le gzip (Content-Encoding)
      const buf=(u[0]===0x1f&&u[1]===0x8b)?await new Response(new Blob([ab]).stream().pipeThrough(new DecompressionStream('gzip'))).arrayBuffer():ab;
      const n=t.n, d=new Uint32Array(unshuf(buf,t.idx,n,4)), R=new Uint16Array(n), C=new Uint16Array(n); let s=0;
      for(let i=0;i<n;i++){ s+=d[i]; const r2=(s/W)|0; R[i]=r2; C[i]=s-r2*W; }
      t.R=R; t.C=C; const v={};
      for(const c of t.cols){ v[c.id]=c.t==='u16'?new Uint16Array(unshuf(buf,c.off,n,2)):new Float32Array(unshuf(buf,c.off,n,4)); VARTAB[c.id]=t; }
      t.v=v;
    })().catch(e=>{ t.p=null; throw e; });
  }
  const OUTLINE_P=getJSON(DATA+'outline.json'+VQ);
  const [OUTLINE]=await Promise.all([OUTLINE_P,loadTable('pv')]);

  /* ---------- 2. agrégation en blocs de k × k km, ancrés au coin sud-ouest ---------- */
  const gCache=new Map(), aCache=new Map(); let aBytes=0, LANDKM=null;
  const KMAX=Math.max(W,H);
  function geom(k){
    if(gCache.has(k)) return gCache.get(k);
    const g={k,Wk:Math.ceil(W/k),Hk:Math.ceil(H/k),ci:new Int32Array(W),ri:new Int32Array(H)};
    for(let c=0;c<W;c++) g.ci[c]=Math.floor(c/k);
    for(let r=0;r<H;r++) g.ri[r]=Math.floor(r/k);
    gCache.set(k,g); return g;
  }
  function agg(id,g){
    const key=g.k+':'+id; if(aCache.has(key)){ const v=aCache.get(key); aCache.delete(key); aCache.set(key,v); return v; }
    const ri=g.ri, ci=g.ci, Wk=g.Wk, out=new Float32Array(Wk*g.Hk);
    if(id.startsWith('sel:')){ const [,w,q,m]=id.split(':'), p=w==='n'?'n':'k', mask=+m; for(let c=0;c<5;c++) if(mask>>c&1){ const a=agg(`${p}${c}_${q}`,g); for(let i=0;i<out.length;i++) out[i]+=a[i]; } }
    else if(id==='land'){ for(let r=0;r<H;r++){ const ro=ri[r]*Wk, off=r*W; for(let c=0;c<W;c++){ const v=LANDKM[off+c]; if(v) out[ro+ci[c]]+=v; } } }
    else { const t=VARTAB[id], a=t.v[id], R=t.R, C=t.C; for(let i=0,n=t.n;i<n;i++){ const v=a[i]; if(v) out[ri[R[i]]*Wk+ci[C[i]]]+=v; } }
    aCache.set(key,out); aBytes+=out.byteLength;
    while(aBytes>120e6&&aCache.size>1){ const k0=aCache.keys().next().value; aBytes-=aCache.get(k0).byteLength; aCache.delete(k0); }
    return out;
  }

  /* ---------- 3. variables : mesure ÷ dénominateur ---------- */
  const MEAS={
    n:{g:'pv',fam:'pv',parts:['n']}, kwp:{g:'pv',fam:'kwp',parts:['kwp']},
    ind:{g:'pop',fam:'ind',parts:['ind']},
    snv:{g:'pop',fam:'snv',parts:['ind_snv'],fixed:'ind',mult:1},
    men:{g:'pop',fam:'men',parts:['men']},
    men_pauv:{g:'pop',fam:'men',parts:['men_pauv'],def:'men'},
    men_prop:{g:'hh',fam:'men',parts:['men_prop'],def:'men'},
    men_mais:{g:'hh',fam:'men',parts:['men_mais'],def:'men'},
  };
  for(const id in MEAS) Object.assign(MEAS[id],I.meas[id]);
  I.classes.forEach((c,i)=>{
    MEAS['nc'+i]={g:'cls',fam:'pv',unit:I.meas.n.unit,parts:['nc'+i],def:'pvall',label:tr(I.clsInst,{c:c.long}),short:tr(I.clsInstShort,{s:c.short})};
    MEAS['kc'+i]={g:'cls',fam:'kwp',unit:I.meas.kwp.unit,parts:['kc'+i],def:'kall',label:tr(I.clsKwp,{c:c.long}),short:tr(I.clsKwpShort,{s:c.short})};
  });
  const DEN={
    none:{}, ind:{parts:['ind'],fam:'ind',thr:20,mult:1000}, men:{parts:['men'],fam:'men',thr:10,mult:100},
    mais:{parts:['men_mais'],fam:'men',thr:10,mult:100}, rev:{parts:['ind_snv'],fam:'snv',thr:4e5,mult:1e6},
    pvall:{parts:['nall'],fam:'pv',thr:1,mult:1}, kall:{parts:['kall'],fam:'kwp',thr:1,mult:1},
  };
  for(const id in DEN) Object.assign(DEN[id],I.den[id]);
  const MCACHE=new Map();
  function getMetric(mk,dk){
    const M=MEAS[mk]; if(M.fixed) dk=M.fixed; const id=mk+'|'+dk; if(MCACHE.has(id)) return MCACHE.get(id);
    const D=dk==='none'?null:DEN[dk], nn=M.parts.length, nd=D?D.parts.length:0, pct=!!D&&D.fam===M.fam;
    const mult=M.mult!==undefined?M.mult:!D?1:pct?100:D.mult, thr=D?D.thr:0;
    let unit=M.unit, label=M.label;
    if(D&&pct){ unit='%'; label=tr(I.fmt.pct,{short:M.short,pct:D.pctLab}); }
    else if(D&&M.fam==='kwp'&&dk==='ind'){ unit=I.kwpPerCap.unit; label=I.kwpPerCap.label; }
    else if(D){ unit=(M.unit+' '+D.per).trim(); label=tr(I.fmt.per,{short:M.short,per:D.perLab}); }
    if(M.fam==='snv'){ unit=I.unitSnv; label=M.label; }
    const met={id,mk,dk,M,D,label,short:M.short+(D&&!M.fixed?' ÷ '+D.short:''),unit,needs:M.parts.concat(D?D.parts:[]),
      zeroEmpty:!D, int:!D&&dk==='none', scale:(!D&&M.fam!=='snv')?'log':'lin',
      note:D&&thr>1?tr(I.thrNote,{thr:NF(thr),what:D.fam==='ind'?I.thrInd:D.fam==='men'?I.thrMen:I.thrUnit}):'',
      f:(A,i)=>{ let nu=0; for(let j=0;j<nn;j++) nu+=A[j][i];
        if(!D) return nu;
        let de=0; for(let j=0;j<nd;j++) de+=A[nn+j][i];
        return de>=thr?mult*nu/de:NaN; }};
    MCACHE.set(id,met); return met;
  }
  const S={k:4,q:'all',cls:31,mode:'one',a:{m:'n',d:'none'},b:{m:'snv',d:'none'},scale:'log',hover:null,base:'none',alpha:.8};
  const NEED_ID=n=>{
    if(n==='n'||n==='kwp') return S.cls===31?(n==='n'?`n_${S.q}`:`kwp_${S.q}`):`sel:${n}:${S.q}:${S.cls}`;
    if(n==='nall') return `n_${S.q}`; if(n==='kall') return `kwp_${S.q}`;
    if(/^[nk]c\d$/.test(n)) return `${n[0]}${n[2]}_${S.q}`;
    return n; };
  const tabOf=id=>id.startsWith('sel:')||/^[nk]\d_/.test(id)?'pvcls':/^(n|kwp)_(all|hc)$/.test(id)?'pv':id==='land'?null:'insee';
  const metA=()=>getMetric(S.a.m,S.a.d), metB=()=>getMetric(S.b.m,S.b.d);
  function missingTables(){
    const ms=[metA()].concat(S.mode==='biv'?[metB()]:[]), need=new Set(['pv']);
    for(const m of ms) for(const n of m.needs){ const t=tabOf(NEED_ID(n)); if(t) need.add(t); }
    return [...need].filter(t=>!TABS[t].v);
  }
  const hasPop=()=>!!TABS.insee&&!!TABS.insee.v;
  function metricArray(m,g){
    const A=m.needs.map(n=>agg(NEED_ID(n),g)), N=g.Wk*g.Hk, out=new Float32Array(N), f=m.f;
    for(let i=0;i<N;i++) out[i]=f(A,i);
    return out;
  }

  /* ---------- 4. couleurs ---------- */
  const hex2=h=>[parseInt(h.slice(1,3),16),parseInt(h.slice(3,5),16),parseInt(h.slice(5,7),16)];
  function makeLUT(stops,t0=0){
    const lut=new Uint32Array(256);
    for(let i=0;i<256;i++){
      const t=t0+(1-t0)*i/255; let j=0; while(j<stops.length-2&&t>stops[j+1][0]) j++;
      const [p0,c0]=stops[j],[p1,c1]=stops[j+1], u=Math.min(1,Math.max(0,(t-p0)/(p1-p0))), a=hex2(c0), b=hex2(c1);
      lut[i]=(255<<24)|(Math.round(a[2]+(b[2]-a[2])*u)<<16)|(Math.round(a[1]+(b[1]-a[1])*u)<<8)|Math.round(a[0]+(b[0]-a[0])*u);
    }
    return lut;
  }
  const INFERNO=[[0,'#000004'],[.125,'#1b0c41'],[.25,'#4a0c6b'],[.375,'#781c6d'],[.5,'#a52c60'],[.625,'#cf4446'],[.75,'#ed6925'],[.875,'#fb9b06'],[.94,'#f7d13d'],[1,'#fcffa4']];
  const TEAL=[[0,'#10243a'],[.2,'#1d4f7c'],[.45,'#2a82a3'],[.7,'#44bdae'],[.88,'#a6e3b3'],[1,'#f3fbd2']];
  const LUTS={pv:makeLUT(INFERNO,.12), cov:makeLUT(TEAL,0)};
  const grpOf=m=>(m.M.fam==='pv'||m.M.fam==='kwp')?'pv':'cov';
  const BIV=[['#e8e8e8','#ace4e4','#5ac8c8'],['#dfb0d6','#a5add3','#5698b9'],['#be64ac','#8c62aa','#3b4994']];
  const GREYPX=(255<<24)|(0xbd<<16)|(0xbd<<8)|0xbd;
  const BIVLUT=BIV.map(r=>r.map(h=>{const c=hex2(h);return (255<<24)|(c[2]<<16)|(c[1]<<8)|c[0];}));
  const css=(lut,i)=>{const v=lut[i];return `rgb(${v&255},${(v>>8)&255},${(v>>16)&255})`;};

  /* ---------- 5. état courant : valeurs, statistiques, raster de couleurs ---------- */
  let CUR=null, CURID=0;
  const OFF=document.createElement('canvas');
  const pct=(sorted,p)=>sorted[Math.min(sorted.length-1,Math.max(0,Math.round(p*(sorted.length-1))))];
  const lowerBound=(a,v)=>{ let lo=0,hi=a.length; while(lo<hi){const mid=(lo+hi)>>1; if(a[mid]<v) lo=mid+1; else hi=mid;} return lo; };
  function recompute(){
    const g=geom(S.k), N=g.Wk*g.Hk, mA=metA(), VA=metricArray(mA,g);
    const cur={id:++CURID,g,mA,VA,mode:S.mode};
    OFF.width=g.Wk; OFF.height=g.Hk;
    const ctx=OFF.getContext('2d'), img=ctx.createImageData(g.Wk,g.Hk), px=new Uint32Array(img.data.buffer); cur.px=px;
    if(S.mode==='one'){
      const sc=S.scale, vals=[];
      const ok=v=>isFinite(v)&&!(mA.zeroEmpty&&v<=0)&&!(sc==='log'&&v<=0);
      for(let i=0;i<N;i++) if(ok(VA[i])) vals.push(VA[i]);
      const sorted=Float32Array.from(vals).sort(); cur.n=sorted.length;
      if(sorted.length){
        let lo,hi;
        if(sc==='log'){ lo=mA.zeroEmpty?sorted[0]:pct(sorted,.01); hi=pct(sorted,.98); if(hi<=lo) hi=lo*1.0001+1e-9; }
        else if(sc==='lin'){ lo=pct(sorted,.02); hi=pct(sorted,.98); if(hi<=lo) hi=lo+1e-9; }
        else { lo=sorted[0]; hi=sorted[sorted.length-1]; }
        let scE=sc;
        if(sorted[0]===sorted[sorted.length-1]){ const v=sorted[0]; if(sc==='log'){ lo=v/2; hi=v*2; } else { scE='lin'; lo=v-.5*Math.abs(v)-1e-9; hi=v+.5*Math.abs(v)+1e-9; } }
        cur.lo=lo; cur.hi=hi; cur.sorted=sorted; cur.sc=scE;
        const llo=Math.log(lo), lhi=Math.log(hi), lut=LUTS[grpOf(mA)], nS=sorted.length;
        for(let r=0;r<g.Hk;r++){ const yr=(g.Hk-1-r)*g.Wk, ro=r*g.Wk;
          for(let c=0;c<g.Wk;c++){ const v=VA[ro+c]; if(!ok(v)) continue;
            let t=scE==='log'?(Math.log(v)-llo)/(lhi-llo):scE==='lin'?(v-lo)/(hi-lo):lowerBound(sorted,v)/nS;
            t=t<0?0:t>1?1:t; px[yr+c]=lut[Math.round(t*255)]; } }
      }
    } else {
      const mB=metB(), VB=metricArray(mB,g); cur.mB=mB; cur.VB=VB;
      const va=[],vb=[],idx=[];
      for(let i=0;i<N;i++){ const a=VA[i], b=VB[i]; if(isFinite(a)&&isFinite(b)&&(a>0||b>0)){ va.push(a); vb.push(b); idx.push(i); } }
      const sa=Float32Array.from(va).sort(), sb=Float32Array.from(vb).sort();
      cur.n=sa.length; cur.va=Float32Array.from(va); cur.vb=Float32Array.from(vb);
      if(sa.length){
        const ea=[pct(sa,1/3),pct(sa,2/3)], eb=[pct(sb,1/3),pct(sb,2/3)]; cur.ea=ea; cur.eb=eb; cur.sa=sa; cur.sb=sb;
        const cl=(v,e)=>v<=e[0]?0:v<=e[1]?1:2;
        for(let j=0;j<idx.length;j++){ const i=idx[j], r=(i/g.Wk)|0, c=i-r*g.Wk; px[(g.Hk-1-r)*g.Wk+c]=BIVLUT[cl(va[j],ea)][cl(vb[j],eb)]; }
      }
    }
    /* carreaux sans couleur mais en France : zéro réel -> couleur la plus sombre ; valeur non calculable (dénominateur trop faible) -> gris */
    { const LA=agg('land',g), NA=agg(NEED_ID('n'),g), lut0=LUTS[grpOf(mA)][0], biv=S.mode==='biv', VB=cur.VB; let nF=0, nG=0;
      for(let r=0;r<g.Hk;r++){ const yr=(g.Hk-1-r)*g.Wk, ro=r*g.Wk;
        for(let c=0;c<g.Wk;c++){ const i=ro+c; if(px[yr+c]||!(LA[i]>0||NA[i]>0)) continue;
          const def=biv?(isFinite(VA[i])&&isFinite(VB[i])):isFinite(VA[i]);
          if(def){ px[yr+c]=biv?BIVLUT[0][0]:lut0; nF++; } else { px[yr+c]=GREYPX; nG++; } } }
      cur.nFloor=nF; cur.nGrey=nG; }
    ctx.putImageData(img,0,0); CUR=cur;
  }

  /* ---------- 6. projections : LAEA Europe (EPSG:3035, GRS80) et Web Mercator ---------- */
  const D2R=Math.PI/180, CIRC=40075.016686;
  const LAEA=(()=>{
    const a=6378137, f=1/298.257222101, e2=f*(2-f), e=Math.sqrt(e2), p1=52*D2R, l0=10*D2R, FE=4321000, FN=3210000;
    const q=p=>{const s=Math.sin(p); return (1-e2)*(s/(1-e2*s*s)-Math.log((1-e*s)/(1+e*s))/(2*e));};
    const qp=q(Math.PI/2), q1=q(p1), m1=Math.cos(p1)/Math.sqrt(1-e2*Math.sin(p1)**2);
    const bc=Math.asin(q1/qp), Rq=a*Math.sqrt(qp/2), D=a*m1/(Rq*Math.cos(bc)), sbc=Math.sin(bc), cbc=Math.cos(bc);
    const c2=e2/3+31*e2**2/180+517*e2**3/5040, c4=23*e2**2/360+251*e2**3/3780, c6=761*e2**3/45360;
    return {
      fwd(lon,lat){ const dl=lon*D2R-l0, b=Math.asin(q(lat*D2R)/qp), sb=Math.sin(b), cb=Math.cos(b), cd=Math.cos(dl), B=Rq*Math.sqrt(2/(1+sbc*sb+cbc*cb*cd));
        return [FE+B*D*cb*Math.sin(dl), FN+(B/D)*(cbc*sb-sbc*cb*cd)]; },
      inv(E,N){ const x=E-FE, y=N-FN, rho=Math.hypot(x/D,D*y); if(rho<1e-9) return [10,52];
        const ce=2*Math.asin(rho/(2*Rq)), sce=Math.sin(ce), cce=Math.cos(ce), b=Math.asin(cce*sbc+D*y*sce*cbc/rho),
              lon=l0+Math.atan2(x*sce,D*rho*cbc*cce-D*D*y*sbc*sce), lat=b+c2*Math.sin(2*b)+c4*Math.sin(4*b)+c6*Math.sin(6*b);
        return [lon/D2R,lat/D2R]; }
    };
  })();
  const MERC={ toM(lon,lat){ const s=Math.sin(lat*D2R); return [(lon+180)/360,.5-Math.log((1+s)/(1-s))/(4*Math.PI)]; },
               fromM(mx,my){ return [mx*360-180, Math.atan(Math.sinh(Math.PI-2*Math.PI*my))/D2R]; } };

  /* ---------- 7. contours, vues et dessin ---------- */
  const LAND_LL=OUTLINE.land;
  const LAND=LAND_LL.map(r=>r.map(([lo,la])=>{const p=LAEA.fwd(lo,la);return [p[0]/1000,p[1]/1000];}));
  LANDKM=(()=>{ // km² de terre dans chaque carreau de 1 km (fraction de pixel couverte par le contour), indexé comme les tables
    const c=document.createElement('canvas'); c.width=W; c.height=H; const x=c.getContext('2d',{willReadFrequently:true});
    x.setTransform(1,0,0,-1,-x0,y0+H); x.fillStyle='#000'; x.beginPath();
    for(const r of LAND){ r.forEach(([X,Y],i)=>i?x.lineTo(X,Y):x.moveTo(X,Y)); x.closePath(); } x.fill();
    const d=x.getImageData(0,0,W,H).data, out=new Float32Array(W*H);
    for(let r=0;r<H;r++){ const po=(H-1-r)*W; for(let cc=0;cc<W;cc++) out[r*W+cc]=d[(po+cc)*4+3]/255; }
    return out; })();
  let bx0=1e9,bx1=-1e9,by0=1e9,by1=-1e9, mx0=1,mx1=0,my0=1,my1=0;
  for(const r of LAND) for(const [x,y] of r){ bx0=Math.min(bx0,x); bx1=Math.max(bx1,x); by0=Math.min(by0,y); by1=Math.max(by1,y); }
  for(const r of LAND_LL) for(const [lo,la] of r){ const [x,y]=MERC.toM(lo,la); mx0=Math.min(mx0,x); mx1=Math.max(mx1,x); my0=Math.min(my0,y); my1=Math.max(my1,y); }
  const view={cx:(bx0+bx1)/2,cy:(by0+by1)/2,s:1}, mv={mx:.5,my:.5,z:5};
  let MODE='laea', FAST=false, hqTimer=0, fitted=false;
  const fitS=(w,h,pad=.04)=>Math.min(w*(1-2*pad)/(bx1-bx0),h*(1-2*pad)/(by1-by0));
  const fitM=(w,h,pad=.04)=>({z:Math.log2(Math.min(w*(1-2*pad)/(mx1-mx0),h*(1-2*pad)/(my1-my0))/256),mx:(mx0+mx1)/2,my:(my0+my1)/2});
  const kmPerPx=(z,lat)=>CIRC*Math.cos(lat*D2R)/(256*Math.pow(2,z));
  function laeaToMerc(){ const [lo,la]=LAEA.inv(view.cx*1000,view.cy*1000), [x,y]=MERC.toM(lo,la);
    mv.mx=x; mv.my=y; mv.z=Math.log2(view.s*CIRC*Math.cos(la*D2R)/256); }
  function mercToLaea(){ const [lo,la]=MERC.fromM(mv.mx,mv.my), [E,N]=LAEA.fwd(lo,la);
    view.cx=E/1000; view.cy=N/1000; view.s=256*Math.pow(2,mv.z)/(CIRC*Math.cos(la*D2R)); }

  function ringsPath(ctx,rings,w,h,v){
    ctx.beginPath();
    for(const r of rings){ for(let i=0;i<r.length;i++){ const x=w/2+(r[i][0]-v.cx)*v.s, y=h/2-(r[i][1]-v.cy)*v.s; i?ctx.lineTo(x,y):ctx.moveTo(x,y); } ctx.closePath(); }
  }
  /* — rendu LAEA : grille nette, sans fond de carte — */
  function drawMap(ctx,w,h,v,opt={}){
    ctx.fillStyle=opt.bg||'#eceff1'; ctx.fillRect(0,0,w,h);
    ctx.save();
    ringsPath(ctx,LAND,w,h,v); ctx.fillStyle='#1f1f25'; ctx.fill(); ctx.clip();
    if(CUR&&CUR.n){
      const g=CUR.g,k=g.k, sx=w/2+(x0-v.cx)*v.s, sy=h/2-(y0+g.Hk*k-v.cy)*v.s;
      ctx.globalAlpha=S.alpha; ctx.imageSmoothingEnabled=(v.s*k<1.2); ctx.imageSmoothingQuality='high';
      ctx.drawImage(OFF,sx,sy,g.Wk*k*v.s,g.Hk*k*v.s); ctx.globalAlpha=1;
    }
    ctx.restore();
    ringsPath(ctx,LAND,w,h,v); ctx.strokeStyle='rgba(20,20,26,.9)'; ctx.lineWidth=.9; ctx.stroke();
    if(opt.hover&&CUR){ const g=CUR.g,k=g.k,hv=opt.hover, x=w/2+(x0+hv.bc*k-v.cx)*v.s, y=h/2-(y0+(hv.br+1)*k-v.cy)*v.s, d=k*v.s;
      ctx.lineWidth=2; ctx.strokeStyle='#fff'; ctx.strokeRect(x,y,d,d); ctx.lineWidth=1; ctx.strokeStyle='#000'; ctx.strokeRect(x-1.5,y-1.5,d+3,d+3); }
  }

  /* — fond de carte OpenStreetMap (tuiles Web Mercator, zoom choisi automatiquement) — */
  const OSM={attr:'© <a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noopener">OpenStreetMap contributors</a>',attrTxt:'© OpenStreetMap contributors',max:19,bg:'#e5e3df',
    url:(z,x,y)=>`https://tile.openstreetmap.org/${z}/${x}/${y}.png`};
  const TILES=new Map(); let tileOK=0, tileErr=0;
  function getTile(key,url){
    let t=TILES.get(key); if(t) return t;
    const img=new Image(); t={img,ok:false,fail:false};
    img.crossOrigin='anonymous'; img.referrerPolicy='strict-origin-when-cross-origin';   // Referer envoyé (politique d'usage OSM) ; CORS pour l'export PNG
    img.onload=()=>{ t.ok=true; tileOK++; schedule(); };
    img.onerror=()=>{ t.fail=true; tileErr++; if(tileErr>=6&&!tileOK) $('baseHint').textContent=I.tilesErr; };
    img.src=url;
    TILES.set(key,t); if(TILES.size>500) TILES.delete(TILES.keys().next().value);
    return t;
  }
  function drawTiles(ctx,w,h,v,dpr){
    const ws=256*Math.pow(2,v.z), zt=Math.max(0,Math.min(OSM.max,Math.round(v.z+(dpr>=2?.5:0)))), n=1<<zt, ts=ws/n; let pending=0;
    const xa=Math.floor((v.mx*ws-w/2)/ts), xb=Math.floor((v.mx*ws+w/2)/ts), ya=Math.max(0,Math.floor((v.my*ws-h/2)/ts)), yb=Math.min(n-1,Math.floor((v.my*ws+h/2)/ts));
    for(let ty=ya;ty<=yb;ty++) for(let tx=xa;tx<=xb;tx++){
      const wx=((tx%n)+n)%n, t=getTile(`${zt}/${wx}/${ty}`,OSM.url(zt,wx,ty));
      if(t.ok) ctx.drawImage(t.img,w/2+tx*ts-v.mx*ws,h/2+ty*ts-v.my*ws,ts+.5,ts+.5); else if(!t.fail) pending++;
    }
    return pending;
  }
  /* — couche de données reprojetée : chaque pixel de l'écran est ramené dans la grille LAEA (maillage 8 px, interpolation bilinéaire) — */
  const LC={key:'',cv:document.createElement('canvas')};
  function buildLayer(w,h,v,rs){
    const g=CUR.g,k=g.k, lw=Math.max(1,Math.round(w*rs)), lh=Math.max(1,Math.round(h*rs)), cv=LC.cv; cv.width=lw; cv.height=lh;
    const c=cv.getContext('2d'), img=c.createImageData(lw,lh), out=new Uint32Array(img.data.buffer), ws=256*Math.pow(2,v.z), Bk=8, nx=Math.ceil(lw/Bk), ny=Math.ceil(lh/Bk), NW=nx+1;
    const U=new Float64Array(NW*(ny+1)), V=new Float64Array(NW*(ny+1));
    for(let j=0;j<=ny;j++) for(let i=0;i<=nx;i++){
      const [lo,la]=MERC.fromM(v.mx+(i*Bk/rs-w/2)/ws, v.my+(j*Bk/rs-h/2)/ws), [E,N]=LAEA.fwd(lo,la);
      U[j*NW+i]=(E/1000-x0)/k; V[j*NW+i]=(N/1000-y0)/k; }
    const PX=CUR.px, Wk=g.Wk, Hk=g.Hk;
    for(let by=0;by<ny;by++) for(let bx=0;bx<nx;bx++){
      const i00=by*NW+bx, i10=i00+1, i01=i00+NW, i11=i01+1, xs=bx*Bk, ys=by*Bk, xe=Math.min(xs+Bk,lw), ye=Math.min(ys+Bk,lh);
      for(let y=ys;y<ye;y++){ const fy=(y-ys)/Bk,
        ul=U[i00]+(U[i01]-U[i00])*fy, vl=V[i00]+(V[i01]-V[i00])*fy, ur=U[i10]+(U[i11]-U[i10])*fy, vr=V[i10]+(V[i11]-V[i10])*fy,
        du=(ur-ul)/Bk, dv=(vr-vl)/Bk, ro=y*lw; let u=ul, vv=vl;
        for(let x=xs;x<xe;x++,u+=du,vv+=dv){ const ui=Math.floor(u), vi=Math.floor(vv); if(ui<0||vi<0||ui>=Wk||vi>=Hk) continue; const col=PX[(Hk-1-vi)*Wk+ui]; if(col) out[ro+x]=col; } }
    }
    c.putImageData(img,0,0); return cv;
  }
  function drawMapMerc(ctx,w,h,v,opt={}){
    const ws=256*Math.pow(2,v.z), dpr=opt.dpr||window.devicePixelRatio||1;
    ctx.fillStyle=OSM.bg; ctx.fillRect(0,0,w,h);
    const pending=drawTiles(ctx,w,h,v,dpr);
    const P=(lo,la)=>{const [x,y]=MERC.toM(lo,la); return [w/2+(x-v.mx)*ws, h/2+(y-v.my)*ws];};
    const path=rings=>{ ctx.beginPath(); for(const r of rings){ for(let i=0;i<r.length;i++){ const [x,y]=P(r[i][0],r[i][1]); i?ctx.lineTo(x,y):ctx.moveTo(x,y); } ctx.closePath(); } };
    ctx.save(); path(LAND_LL); ctx.clip();
    if(CUR&&CUR.n){
      const lat=MERC.fromM(v.mx,v.my)[1], cellPx=CUR.g.k/kmPerPx(v.z,lat);
      const rs=opt.fast?.5:(cellPx<1.6?Math.min(2,Math.sqrt(6e6/(w*h))):1), key=[CUR.id,v.mx.toFixed(9),v.my.toFixed(9),v.z.toFixed(6),w,h,rs].join('|');
      if(LC.key!==key){ buildLayer(w,h,v,rs); LC.key=key; }
      ctx.globalAlpha=S.alpha; ctx.imageSmoothingEnabled=true; ctx.imageSmoothingQuality='high'; ctx.drawImage(LC.cv,0,0,w,h); ctx.globalAlpha=1;
    }
    ctx.restore();
    path(LAND_LL); ctx.strokeStyle='rgba(20,20,26,.55)'; ctx.lineWidth=.9; ctx.stroke();
    if(opt.hover&&CUR){ const k=CUR.g.k, hv=opt.hover;
      const pts=[[0,0],[1,0],[1,1],[0,1]].map(([a,b])=>{ const [lo,la]=LAEA.inv((x0+(hv.bc+a)*k)*1000,(y0+(hv.br+b)*k)*1000); return P(lo,la); });
      ctx.beginPath(); pts.forEach((p,i)=>i?ctx.lineTo(p[0],p[1]):ctx.moveTo(p[0],p[1])); ctx.closePath();
      ctx.lineWidth=3; ctx.strokeStyle='#000'; ctx.stroke(); ctx.lineWidth=1.6; ctx.strokeStyle='#fff'; ctx.stroke(); }
    return pending;
  }

  /* ---------- 8. légende ---------- */
  function fmtV(v,m){ if(!isFinite(v)) return '—'; if(m.int||Math.abs(v)>=100) return NF(v,0); if(Math.abs(v)>=10) return NF(v,1); return NF(v,2); }
  function invT(t){ const c=CUR; if(c.sc==='log') return Math.exp(Math.log(c.lo)+t*(Math.log(c.hi)-Math.log(c.lo))); if(c.sc==='lin') return c.lo+t*(c.hi-c.lo); return pct(c.sorted,t); }
  function ell(ctx,t,mw){ while(t.length>3&&ctx.measureText(t).width>mw) t=t.slice(0,-2).trimEnd()+'…'; return t; }
  function drawLegend(ctx,x,y,s=1){
    const fs=12*s; ctx.textBaseline='alphabetic';
    if(!CUR||!CUR.n) return {w:0,h:0};
    if(CUR.mode==='one'){
      const m=CUR.mA, bw=(FULL?250:220)*s, bh=11*s, pad=12*s, lut=LUTS[grpOf(m)], W0=bw+2*pad, H0=pad+fs+6*s+bh+fs+10*s+(CUR.sc==='quant'?fs+4*s:0);
      ctx.font=`600 ${fs}px ${FONT}`; ctx.fillStyle='#15171c'; ctx.fillText(ell(ctx,m.label,bw),x+pad,y+pad+fs*.85);
      const by=y+pad+fs+6*s;
      for(let i=0;i<bw;i++){ ctx.fillStyle=css(lut,Math.round(i/(bw-1)*255)); ctx.fillRect(x+pad+i,by,1.2,bh); }
      ctx.font=`${fs*.95}px ${FONT}`;
      const labs=[0,1,2,3,4].map(q=>{ const t=q/4, txt=fmtV(invT(t),m), tw=ctx.measureText(txt).width, tx=x+pad+t*bw;
        return {txt,tw,tx,px:Math.min(Math.max(tx-tw/2,x+pad-4*s),x+W0-pad-tw+4*s)}; });
      const keep=[labs[0],labs[4]];
      for(const L of [labs[2],labs[1],labs[3]]) if(keep.every(K=>L.px>K.px+K.tw+6*s||L.px+L.tw+6*s<K.px)) keep.push(L);
      for(const L of keep){ ctx.fillStyle='#15171c'; ctx.fillRect(L.tx-.5*s,by+bh,s,3*s); ctx.fillStyle='#434a56'; ctx.fillText(L.txt,L.px,by+bh+fs+3*s); }
      if(CUR.sc==='quant'){ ctx.fillStyle='#6a7280'; ctx.fillText(I.quantNote,x+pad,y+H0-pad+2*s); }
      return {w:W0,h:H0};
    }
    const a=CUR.mA, b=CUR.mB, cell=30*s, pad=12*s, lw=cell*3;
    ctx.font=`600 ${fs}px ${FONT}`; const title=I.bivTitle, tw=ctx.measureText(title).width;
    ctx.font=`${fs*.92}px ${FONT}`; const bl=ell(ctx,b.short+' →',lw+40*s), al=ell(ctx,a.short+' →',lw+fs), bw=ctx.measureText(bl).width;
    const gx=x+pad+fs*1.6, gy=y+pad+fs+10*s, W0=Math.max(tw,gx-x-pad+Math.max(lw,bw))+2*pad, H0=(gy-y)+lw+fs+8*s+pad;
    ctx.font=`600 ${fs}px ${FONT}`; ctx.fillStyle='#15171c'; ctx.fillText(title,x+pad,y+pad+fs*.85);
    for(let r=0;r<3;r++) for(let q=0;q<3;q++){ ctx.fillStyle=BIV[r][q]; ctx.fillRect(gx+q*cell,gy+(2-r)*cell,cell-1.5*s,cell-1.5*s); }
    ctx.font=`${fs*.92}px ${FONT}`; ctx.fillStyle='#434a56'; ctx.fillText(bl,gx,gy+lw+fs+3*s);
    ctx.save(); ctx.translate(gx-6*s,gy+lw); ctx.rotate(-Math.PI/2); ctx.fillText(al,0,0); ctx.restore();
    return {w:W0,h:H0};
  }
  const legCv=$('legend');
  function renderLegend(){
    const dpr=window.devicePixelRatio||1, sz=drawLegend(document.createElement('canvas').getContext('2d'),0,0,1);
    if(!sz.w){ legCv.style.display='none'; return; }
    legCv.style.display='block'; legCv.width=Math.ceil(sz.w*dpr); legCv.height=Math.ceil(sz.h*dpr); legCv.style.width=sz.w+'px'; legCv.style.height=sz.h+'px';
    const c=legCv.getContext('2d'); c.setTransform(dpr,0,0,dpr,0,0); drawLegend(c,0,0,1);
  }

  /* ---------- 9. nuage de points 2D et corrélation (mode croisement) ---------- */
  function ranks(a){ const n=a.length, idx=new Uint32Array(n); for(let i=0;i<n;i++) idx[i]=i; idx.sort((p,q)=>a[p]-a[q]);
    const r=new Float64Array(n); let i=0; while(i<n){ let j=i; while(j+1<n&&a[idx[j+1]]===a[idx[i]]) j++; const rk=(i+j)/2; for(let t=i;t<=j;t++) r[idx[t]]=rk; i=j+1; } return r; }
  function spearman(a,b){ const n=a.length, ra=ranks(a), rb=ranks(b); let ma=0,mb=0; for(let i=0;i<n;i++){ma+=ra[i];mb+=rb[i];} ma/=n; mb/=n;
    let sab=0,saa=0,sbb=0; for(let i=0;i<n;i++){ const da=ra[i]-ma, db=rb[i]-mb; sab+=da*db; saa+=da*da; sbb+=db*db; } return sab/Math.sqrt(saa*sbb); }
  function renderScatter(){
    const box=$('scatterBox'); if(S.mode!=='biv'||!CUR||CUR.n<10){ box.style.display='none'; return; }
    box.style.display='flex';
    const cv=$('scatter'), dpr=window.devicePixelRatio||1, w=cv.clientWidth, h=cv.clientHeight; if(!w) return; cv.width=w*dpr; cv.height=h*dpr;
    const c=cv.getContext('2d'); c.setTransform(dpr,0,0,dpr,0,0); c.fillStyle='#17171c'; c.fillRect(0,0,w,h);
    const a=CUR.mA, b=CUR.mB, isLog=m=>m.scale==='log', tx=(v,m)=>isLog(m)?Math.log(Math.max(v,1e-9)):v;
    const L=44,R=8,T=8,B=34, pw=w-L-R, ph=h-T-B, NB=56, sa=CUR.sa, sb=CUR.sb;
    const lo=(s,m)=>tx(isLog(m)?(s.find(x=>x>0)||1):pct(s,.01),m), hi=(s,m)=>tx(pct(s,.99),m);
    const ax0=lo(sa,a), ax1=hi(sa,a), bx0_=lo(sb,b), bx1_=hi(sb,b), H2=new Uint32Array(NB*NB); let mx=0;
    for(let i=0;i<CUR.va.length;i++){ const va=CUR.va[i], vb=CUR.vb[i]; if(isLog(a)&&va<=0) continue; if(isLog(b)&&vb<=0) continue;
      const u=(tx(vb,b)-bx0_)/(bx1_-bx0_), v=(tx(va,a)-ax0)/(ax1-ax0); if(u<0||u>=1||v<0||v>=1) continue;
      const k=((v*NB)|0)*NB+((u*NB)|0); if(++H2[k]>mx) mx=H2[k]; }
    const lut=LUTS.pv, lm=Math.log(1+mx);
    for(let j=0;j<NB;j++) for(let i=0;i<NB;i++){ const n=H2[j*NB+i]; if(!n) continue; c.fillStyle=css(lut,Math.round(Math.log(1+n)/lm*255)); c.fillRect(L+i*pw/NB,T+ph-(j+1)*ph/NB,pw/NB+.5,ph/NB+.5); }
    c.strokeStyle='rgba(255,255,255,.35)'; c.setLineDash([3,3]); c.lineWidth=1;
    for(const e of CUR.eb){ if(isLog(b)&&e<=0) continue; const u=(tx(e,b)-bx0_)/(bx1_-bx0_); if(u>0&&u<1){ c.beginPath(); c.moveTo(L+u*pw,T); c.lineTo(L+u*pw,T+ph); c.stroke(); } }
    for(const e of CUR.ea){ if(isLog(a)&&e<=0) continue; const v=(tx(e,a)-ax0)/(ax1-ax0); if(v>0&&v<1){ c.beginPath(); c.moveTo(L,T+ph-v*ph); c.lineTo(L+pw,T+ph-v*ph); c.stroke(); } }
    c.setLineDash([]); c.fillStyle='#c9ced6'; c.font=`11px ${FONT}`;
    const inv=(t,l,h_,m)=>isLog(m)?Math.exp(l+t*(h_-l)):l+t*(h_-l);
    c.textAlign='left'; c.fillText(fmtV(inv(0,bx0_,bx1_,b),b),L,h-18); c.textAlign='right'; c.fillText(fmtV(inv(1,bx0_,bx1_,b),b),L+pw,h-18);
    c.textAlign='center'; c.fillText(ell(c,b.short,pw)+(isLog(b)?' · '+I.log:''),L+pw/2,h-4);
    c.textAlign='right'; c.fillText(fmtV(inv(1,ax0,ax1,a),a),L-4,T+9); c.fillText(fmtV(inv(0,ax0,ax1,a),a),L-4,T+ph);
    c.save(); c.translate(11,T+ph/2); c.rotate(-Math.PI/2); c.textAlign='center'; c.fillText(ell(c,a.short,ph)+(isLog(a)?' · '+I.log:''),0,0); c.restore();
    let va=CUR.va, vb=CUR.vb; const n=va.length, step=Math.ceil(n/40000);
    if(step>1){ const A=[],Bv=[]; for(let i=0;i<n;i+=step){A.push(va[i]);Bv.push(vb[i]);} va=Float32Array.from(A); vb=Float32Array.from(Bv); }
    $('rho').innerHTML='<b>'+tr(I.rho,{rho:NF(spearman(va,vb),2)})+'</b>'+tr(I.rhoTail,{n:NF(n),sample:step>1?tr(I.rhoSample,{n:NF(va.length)}):''});
  }

  /* ---------- 10. vue et interactions ---------- */
  const cv=$('map'), stage=$('stage'), tip=$('tip');
  let CW=0, CH=0, raf=0;
  function resize(){ const dpr=window.devicePixelRatio||1, r=stage.getBoundingClientRect(); CW=r.width; CH=r.height; cv.width=Math.round(CW*dpr); cv.height=Math.round(CH*dpr); }
  function schedule(){ if(raf||!CW) return; raf=requestAnimationFrame(()=>{ raf=0;
    const dpr=window.devicePixelRatio||1, c=cv.getContext('2d'); c.setTransform(dpr,0,0,dpr,0,0);
    if(MODE==='laea') drawMap(c,CW,CH,view,{hover:S.hover}); else drawMapMerc(c,CW,CH,mv,{hover:S.hover,fast:FAST,dpr}); }); }
  function interact(){ if(MODE!=='merc') return; FAST=true; clearTimeout(hqTimer); hqTimer=setTimeout(()=>{ FAST=false; schedule(); },180); }
  function fitView(){ if(!CW) return; if(MODE==='laea'){ view.cx=(bx0+bx1)/2; view.cy=(by0+by1)/2; view.s=fitS(CW,CH); } else { const f=fitM(CW,CH); mv.mx=f.mx; mv.my=f.my; mv.z=f.z; } interact(); schedule(); }
  function zoomAt(px,py,f){
    if(MODE==='laea'){ const X=view.cx+(px-CW/2)/view.s, Y=view.cy-(py-CH/2)/view.s, s=Math.min(60,Math.max(fitS(CW,CH)*.7,view.s*f)); view.s=s; view.cx=X-(px-CW/2)/s; view.cy=Y+(py-CH/2)/s; }
    else { const ws=256*Math.pow(2,mv.z), x=mv.mx+(px-CW/2)/ws, y=mv.my+(py-CH/2)/ws; mv.z=Math.min(17,Math.max(fitM(CW,CH).z-.8,mv.z+Math.log2(f)));
      const w2=256*Math.pow(2,mv.z); mv.mx=x-(px-CW/2)/w2; mv.my=y-(py-CH/2)/w2; interact(); }
    schedule();
  }
  function panBy(dx,dy){ if(MODE==='laea'){ view.cx-=dx/view.s; view.cy+=dy/view.s; } else { const ws=256*Math.pow(2,mv.z); mv.mx-=dx/ws; mv.my=Math.min(.98,Math.max(.02,mv.my-dy/ws)); interact(); } }
  function pointerToKm(px,py){
    if(MODE==='laea') return {X:view.cx+(px-CW/2)/view.s, Y:view.cy-(py-CH/2)/view.s};
    const ws=256*Math.pow(2,mv.z), [lo,la]=MERC.fromM(mv.mx+(px-CW/2)/ws, mv.my+(py-CH/2)/ws), [E,N]=LAEA.fwd(lo,la); return {X:E/1000,Y:N/1000};
  }
  const ptrs=new Map(); let pinch0=0, moved=0, wheelT=0;
  cv.addEventListener('pointerdown',e=>{ cv.setPointerCapture(e.pointerId); ptrs.set(e.pointerId,{x:e.offsetX,y:e.offsetY}); moved=0; cv.classList.add('drag'); if(ptrs.size===2){ const [a,b]=[...ptrs.values()]; pinch0=Math.hypot(a.x-b.x,a.y-b.y); } });
  cv.addEventListener('pointermove',e=>{
    const p=ptrs.get(e.pointerId);
    if(p){
      if(ptrs.size===1){ moved+=Math.abs(e.offsetX-p.x)+Math.abs(e.offsetY-p.y); panBy(e.offsetX-p.x,e.offsetY-p.y); p.x=e.offsetX; p.y=e.offsetY; tip.style.display='none'; S.hover=null; schedule(); }
      else if(ptrs.size===2){ moved=99; p.x=e.offsetX; p.y=e.offsetY; const [a,b]=[...ptrs.values()], d=Math.hypot(a.x-b.x,a.y-b.y); if(pinch0>0) zoomAt((a.x+b.x)/2,(a.y+b.y)/2,d/pinch0); pinch0=d; }
    } else hoverAt(e.offsetX,e.offsetY);
  });
  const endp=e=>{ const had=ptrs.size===1&&ptrs.has(e.pointerId); ptrs.delete(e.pointerId); if(!ptrs.size) cv.classList.remove('drag');
    if(had&&e.type==='pointerup'&&moved<6&&e.pointerType!=='mouse') hoverAt(e.offsetX,e.offsetY); };      // tactile : un appui affiche l'infobulle
  cv.addEventListener('pointerup',endp); cv.addEventListener('pointercancel',endp);
  cv.addEventListener('pointerleave',()=>{ if(!ptrs.size){ tip.style.display='none'; S.hover=null; schedule(); } });
  /* la molette ne zoome qu'avec Ctrl / ⌘ (sinon elle fait défiler la page) ; sur mobile, glisser verticalement fait défiler la page */
  cv.addEventListener('wheel',e=>{
    if(!(e.ctrlKey||e.metaKey)){ const w=$('wheel'); w.classList.add('on'); clearTimeout(wheelT); wheelT=setTimeout(()=>w.classList.remove('on'),1200); return; }
    e.preventDefault(); zoomAt(e.offsetX,e.offsetY,Math.exp(-e.deltaY*(Math.abs(e.deltaY)>=50?.0016:.01)));
  },{passive:false});
  $('zin').onclick=()=>zoomAt(CW/2,CH/2,1.5); $('zout').onclick=()=>zoomAt(CW/2,CH/2,1/1.5); $('zfit').onclick=fitView;

  function hoverAt(px,py){
    if(!CUR) return;
    const g=CUR.g, k=g.k, {X,Y}=pointerToKm(px,py), bc=Math.floor((X-x0)/k), br=Math.floor((Y-y0)/k), c=bc, r=br;
    const hide=()=>{ tip.style.display='none'; if(S.hover){S.hover=null;schedule();} };
    if(c<0||r<0||c>=g.Wk||r>=g.Hk) return hide();
    const i=r*g.Wk+c, A=CUR.VA[i], B=CUR.VB?CUR.VB[i]:NaN, nn=agg(NEED_ID('n'),g)[i], pop=hasPop()?agg('ind',g)[i]:0;
    const empty=CUR.mode==='one'?!(isFinite(A)&&!(CUR.mA.zeroEmpty&&A<=0)):!(isFinite(A)&&isFinite(B));
    if(empty&&!nn&&!(pop>0)) return hide();
    S.hover={bc,br};
    const row=(m,v)=>`<div><span class="k">${m.short} :</span> <b>${fmtV(v,m)}</b> ${m.unit}</div>`, shown=id=>CUR.mA.id===id||(CUR.mB&&CUR.mB.id===id);
    let h=`<div class="k">${k>=KMAX?I.tipOne:tr(I.tipK,{k})}</div>`+row(CUR.mA,A)+(CUR.mB?row(CUR.mB,B):'');
    if(!shown('n|none')) h+=`<div><span class="k">${I.tipInst} :</span> <b>${NF(nn)}</b></div>`;
    if(hasPop()&&!shown('ind|none')) h+=`<div><span class="k">${I.tipPop} :</span> <b>${NF(pop)}</b></div>`;
    tip.innerHTML=h; tip.style.display='block';
    const tw=tip.offsetWidth, th=tip.offsetHeight; tip.style.left=Math.max(4,Math.min(CW-tw-8,px+14))+'px'; tip.style.top=Math.max(4,Math.min(CH-th-8,py+14))+'px';
    schedule();
  }

  /* ---------- 11. contrôles ---------- */
  function seg(id,cb){ const el=$(id); el.querySelectorAll('button').forEach(b=>b.addEventListener('click',()=>{ el.querySelectorAll('button').forEach(x=>x.setAttribute('aria-pressed',x===b)); cb(b.dataset.v); })); }
  function setSeg(id,v){ $(id).querySelectorAll('button').forEach(b=>b.setAttribute('aria-pressed',b.dataset.v===v)); }
  const GROUPS=['pv','cls','pop','hh'];
  const DEN_A=['none','ind','men','mais','rev'], DEN_B=['none','ind','men','mais','rev','pvall','kall'];
  function fillMeas(sel,cur,pick,flat){ sel.innerHTML='';
    for(const g of GROUPS){ const ids=Object.keys(MEAS).filter(k=>MEAS[k].g===g&&pick(k)); if(!ids.length) continue;
      const host=flat?sel:Object.assign(document.createElement('optgroup'),{label:I.groups[g]});
      for(const id of ids){ const o=document.createElement('option'); o.value=id; o.textContent=MEAS[id].label; host.appendChild(o); }
      if(!flat) sel.appendChild(host); }
    sel.value=cur; }
  function fillDen(sel,mk,cur,allowed){ const M=MEAS[mk]; sel.innerHTML='';
    for(const id of allowed){ const o=document.createElement('option'); o.value=id; o.textContent=DEN[id].label; sel.appendChild(o); }
    const v=M.fixed||cur; sel.value=[...sel.options].some(o=>o.value===v)?v:'none'; sel.disabled=!!M.fixed; }
  function syncSelects(){ fillDen($('denA'),S.a.m,S.a.d,DEN_A); fillDen($('denB'),S.b.m,S.b.d,DEN_B); }
  const clsText=()=>S.cls===31?'':I.classes.filter((_,c)=>S.cls>>c&1).map(x=>x.short).join(' + ');
  function refresh(){
    const m=metA(), k=S.k, one=k>=KMAX;
    $('scaleHint').textContent=I.scaleHint[S.scale]+(m.note?' '+m.note:'');
    $('boxScale').style.display=S.mode==='one'?'flex':'none'; $('boxB').hidden=S.mode!=='biv';
    $('advSum').textContent=I.adv+(S.mode==='biv'?' · '+I.advBiv:'')+(S.cls!==31?' · '+clsText():'')+(S.q==='hc'?' · '+I.advHc:'');
    $('selALab').textContent=S.mode==='biv'?I.variableRose:I.variable;
    recompute();
    const lab=one?I.resOne:k+' km'; $('resVal').textContent=lab; $('res').setAttribute('aria-valuetext',lab);
    $('resHint').textContent=(one?I.resHintOne:tr(I.resHint,{k,area:NF(k*k)})+' '+tr(I.nCells,{n:NF(CUR.n)}))+(!FULL&&m.note?' '+m.note:'');
    $('resTicks').querySelectorAll('button').forEach(b=>b.setAttribute('aria-pressed',+b.dataset.k===k));
    { const nt=[]; if(CUR.nFloor) nt.push(S.mode==='one'?I.noteZero:I.noteZero2); if(CUR.nGrey) nt.push(I.noteGrey); $('capNote').textContent=nt.join(' · '); }
    $('capRes').textContent=one?I.capOne:tr(I.capK,{k});
    $('capVar').textContent=S.mode==='biv'?`${CUR.mA.label}${S.cls!==31?' ('+clsText()+')':''} × ${CUR.mB.label}`:CUR.mA.label+(S.cls!==31?' · '+clsText():'')+(S.q==='hc'?' · '+I.capHc:'');
    renderLegend(); renderScatter(); schedule();
  }
  /* mise à jour : charge au besoin les tables manquantes (classes de puissance, INSEE), puis redessine */
  let tok=0;
  function go(){
    const miss=missingTables(), my=++tok;
    if(!miss.length){ $('loading').classList.add('off'); refresh(); return; }
    $('loading').classList.remove('off'); $('loading').textContent=I.loading;
    Promise.all(miss.map(loadTable)).then(()=>{ if(my!==tok) return; $('loading').classList.add('off'); refresh(); })
      .catch(e=>{ console.error('[aggmap]',e); if(my===tok) $('loading').textContent=I.loadErr; });
  }
  function onMeas(which,mk){ const o=S[which]; o.m=mk; o.d=MEAS[mk].fixed||MEAS[mk].def||'none'; if(which==='a'&&S.mode==='one'){ S.scale=metA().scale; setSeg('segS',S.scale); } syncSelects(); go(); }
  function onDen(which,dk){ S[which].d=dk; if(which==='a'&&S.mode==='one'){ S.scale=metA().scale; setSeg('segS',S.scale); } go(); }
  const L2=Math.log(2), LK=Math.log(KMAX);
  const kOfPos=p=>p>=1000?KMAX:Math.max(2,Math.min(KMAX,2*Math.round(Math.exp(L2+p/1000*(LK-L2))/2))), posOfK=k=>Math.round((Math.log(Math.max(2,k))-L2)/(LK-L2)*1000);
  function setK(k,src){ S.k=k; if($('res')!==src) $('res').value=posOfK(k); go(); }
  $('res').addEventListener('input',e=>{ const k=kOfPos(+e.target.value); if(k!==S.k) setK(k,e.target); });
  [2,4,10,20,50,100,...(FULL?[300]:[]),KMAX].forEach(k=>{ const b=document.createElement('button'); b.type='button'; b.textContent=k===KMAX?I.resOneTick:k+' km'; b.dataset.k=k; b.onclick=()=>setK(k); $('resTicks').appendChild(b); });
  I.classes.forEach((_,c)=>{ const el=$('cls'+c); el.onchange=()=>{ let m=0; I.classes.forEach((_,j)=>{ if($('cls'+j).checked) m|=1<<j; }); if(!m){ el.checked=true; m=1<<c; } S.cls=m; go(); }; });
  $('chkHc').onchange=e=>{ S.q=e.target.checked?'hc':'all'; go(); };
  seg('segM',v=>{ S.mode=v; go(); });
  seg('segS',v=>{ S.scale=v; go(); });
  fillMeas($('selA'),S.a.m,k=>k==='n'||k==='kwp',true);
  fillMeas($('selB'),S.b.m,k=>k!=='n'&&k!=='kwp',false);
  syncSelects();
  $('selA').onchange=e=>onMeas('a',e.target.value); $('selB').onchange=e=>onMeas('b',e.target.value);
  $('denA').onchange=e=>onDen('a',e.target.value); $('denB').onchange=e=>onDen('b',e.target.value);
  function setAlpha(a){ S.alpha=a; $('alpha').value=Math.round(a*100); $('alphaVal').textContent=Math.round(a*100)+' %'; }
  function setBase(on){
    const nm=on?'merc':'laea'; S.base=on?'osm':'none';
    if(nm!==MODE){ if(nm==='merc') laeaToMerc(); else mercToLaea(); MODE=nm; }
    $('baseHint').hidden=!on; $('baseHint').textContent=on?I.baseHint:'';
    const at=$('attr'); at.style.display=on?'block':'none'; if(on) at.innerHTML=OSM.attr;
    tileErr=0; interact(); schedule();
  }
  $('chkBase').onchange=e=>setBase(e.target.checked);
  $('alpha').addEventListener('input',e=>{ setAlpha(+e.target.value/100); schedule(); });
  setAlpha(S.alpha); $('res').value=posOfK(S.k);
  const T=META.totals;
  $('lead').textContent=tr(I.lead,{n:NF(T.n_all),gwp:NF(T.gwp_all,1)});
  $('lblHc').title=tr(I.hcTitle,{n:NF(T.n_hc),gwp:NF(T.gwp_hc,1)});

  /* ---------- 12. export des données et de l'image ---------- */
  const slug=t=>t.normalize('NFD').replace(/[̀-ͯ]/g,'').replace(/%/g,'pct').replace(/[^A-Za-z0-9]+/g,'_').replace(/^_|_$/g,'').toLowerCase();
  function saveText(name,text,type){ const a=document.createElement('a'); a.href=URL.createObjectURL(new Blob([text],{type})); a.download=name; a.click(); setTimeout(()=>URL.revokeObjectURL(a.href),4000); }
  function exportRows(){
    const c=CUR, g=c.g, k=g.k, biv=c.mode==='biv', nA=agg(NEED_ID('n'),g), kA=agg(NEED_ID('kwp'),g), lA=agg('land',g), pA=hasPop()?agg('ind',g):null, rows=[];
    for(let r=0;r<g.Hk;r++) for(let q=0;q<g.Wk;q++){
      const i=r*g.Wk+q, a=c.VA[i], b=biv?c.VB[i]:NaN;
      const ok=biv?(isFinite(a)&&isFinite(b)&&(a>0||b>0)):(isFinite(a)&&!(c.mA.zeroEmpty&&a<=0));
      if(!ok) continue;
      rows.push({X:(x0+q*k)*1000,Y:(y0+r*k)*1000,a,b,n:nA[i],kwp:kA[i],land:lA[i],pop:pA?pA[i]:null});
    }
    return rows;
  }
  const expName=ext=>`pv_${slug(CUR.mA.label)}${S.mode==='biv'?'_x_'+slug(CUR.mB.label):''}_${S.k>=KMAX?'france':S.k+'km'}.${ext}`;
  const rnd=(v,d=4)=>isFinite(v)?+v.toFixed(d):'';
  const sig=v=>isFinite(v)?+(+v).toPrecision(6):'';
  /* l'export a besoin de la population : on la charge à la demande */
  async function withPop(f){ try{ await loadTable('insee'); }catch(_){} f(); }
  $('btnCsv').onclick=()=>withPop(()=>{
    const c=CUR, k=c.g.k, biv=c.mode==='biv', rows=exportRows(), P=hasPop();
    const head=['idcar','x_sw_epsg3035','y_sw_epsg3035','lon_centre','lat_centre','a_'+slug(c.mA.label)].concat(biv?['b_'+slug(c.mB.label)]:[],['nb_installations','puissance_kwc','surface_terre_km2'],P?['population']:[]);
    const out=[head.join(',')];
    for(const t of rows){
      const [lo,la]=LAEA.inv(t.X+k*500,t.Y+k*500);
      out.push([`CRS3035RES${k*1000}mN${t.Y}E${t.X}`,t.X,t.Y,rnd(lo,5),rnd(la,5),sig(t.a)].concat(biv?[sig(t.b)]:[],[sig(t.n),sig(t.kwp),rnd(t.land,3)],P?[sig(t.pop)]:[]).join(','));
    }
    saveText(expName('csv'),out.join('\n')+'\n','text/csv;charset=utf-8');
  });
  $('btnGeo').onclick=()=>withPop(()=>{
    const c=CUR, k=c.g.k, biv=c.mode==='biv', rows=exportRows(), m=1000*k, feats=[], P=hasPop();
    const pt=(X,Y)=>{ const [lo,la]=LAEA.inv(X,Y); return `[${lo.toFixed(5)},${la.toFixed(5)}]`; };
    for(const t of rows){
      const p={idcar:`CRS3035RES${m}mN${t.Y}E${t.X}`,a:sig(t.a)}; if(biv) p.b=sig(t.b); p.nb_installations=sig(t.n); p.puissance_kwc=sig(t.kwp); p.surface_terre_km2=rnd(t.land,3); if(P) p.population=sig(t.pop);
      feats.push(`{"type":"Feature","properties":${JSON.stringify(p)},"geometry":{"type":"Polygon","coordinates":[[${pt(t.X,t.Y)},${pt(t.X+m,t.Y)},${pt(t.X+m,t.Y+m)},${pt(t.X,t.Y+m)},${pt(t.X,t.Y)}]]}}`);
    }
    const lab=mm=>mm.label+(mm.unit&&!mm.label.includes('(')?' ('+mm.unit+')':'');
    const meta={a:lab(c.mA),b:biv?lab(c.mB):null,cell_km:k,detections:S.q==='hc'?I.geoHc:I.geoAll,grid:I.geoGrid,sources:I.geoSrc};
    saveText(expName('geojson'),`{"type":"FeatureCollection","metadata":${JSON.stringify(meta)},"features":[\n${feats.join(',\n')}\n]}\n`,'application/geo+json');
  });
  $('btnPng').onclick=async()=>{
    const btn=$('btnPng'); btn.disabled=true; const txt0=btn.textContent; btn.textContent=I.preparing;
    try{
      const Wp=2400, Hp=2700, head=300, foot=200, mh=Hp-head-foot, c=document.createElement('canvas'); c.width=Wp; c.height=Hp; const x=c.getContext('2d');
      x.fillStyle='#eceff1'; x.fillRect(0,0,Wp,Hp);
      const m=CUR.mA, drawBody=()=>{ x.save(); x.translate(0,head); x.beginPath(); x.rect(0,0,Wp,mh); x.clip();
        let pend=0; if(MODE==='laea') drawMap(x,Wp,mh,{cx:(bx0+bx1)/2,cy:(by0+by1)/2,s:fitS(Wp,mh,.02)},{bg:'#eceff1'});
        else { const f=fitM(Wp,mh,.02); pend=drawMapMerc(x,Wp,mh,{mx:f.mx,my:f.my,z:f.z},{dpr:1}); } x.restore(); return pend; };
      let pend=drawBody(); for(let t=0;t<40&&pend>0;t++){ await new Promise(r=>setTimeout(r,250)); pend=drawBody(); }
      const ff=FONT; x.fillStyle='#15171c'; x.textBaseline='alphabetic'; x.font=`700 74px ${ff}`;
      x.fillText(ell(x,S.mode==='one'?m.label:`${m.short} × ${CUR.mB.short}`,Wp-180),90,130);
      x.font=`400 40px ${ff}`; x.fillStyle='#434a56'; x.fillText(tr(I.pngSub,{k:S.k})+(S.cls!==31?' · '+clsText():'')+(S.q==='hc'?I.pngHc:''),90,200);
      const sz=drawLegend(document.createElement('canvas').getContext('2d'),0,0,2.4);
      x.fillStyle='rgba(255,255,255,.92)'; x.fillRect(70,head+mh-sz.h-70,sz.w,sz.h); drawLegend(x,70,head+mh-sz.h-70,2.4);
      x.font=`400 30px ${ff}`; x.fillStyle='#434a56';
      x.fillText(I.pngCredit1+(S.base!=='none'?tr(I.pngBase,{a:OSM.attrTxt}):''),90,Hp-110);
      x.fillText(I.pngCredit2,90,Hp-62);
      const blob=await new Promise(res=>c.toBlob(res)); if(!blob) throw new Error('canvas export');
      const a=document.createElement('a'); a.href=URL.createObjectURL(blob);
      a.download=`pv_${S.a.m}-${S.a.d}${S.mode==='biv'?'_x_'+S.b.m+'-'+S.b.d:''}_${S.k}km.png`; a.click(); setTimeout(()=>URL.revokeObjectURL(a.href),4000);
    } catch(e){ console.error('[aggmap] png',e); } finally { btn.disabled=false; btn.textContent=txt0; }
  };

  /* ---------- 13. démarrage ---------- */
  new ResizeObserver(()=>{ resize(); if(CW>0&&CH>0){ if(!fitted){ fitted=true; fitView(); } schedule(); if(S.mode==='biv') renderScatter(); } }).observe(stage);
  resize(); if(CW>0&&CH>0){ fitted=true; view.s=fitS(CW,CH); }
  setBase(true); $('chkBase').checked=true; fitView();
  refresh();
  $('loading').classList.add('off');
  const fb=root.querySelector('.am-fallback'); if(fb) fb.remove();
  root.classList.add('am-ready');
  root._am={S,go,setBase,LAEA,MERC,agg,geom,project:(lo,la)=>{ if(MODE==='laea'){ const [E,N]=LAEA.fwd(lo,la); return [CW/2+(E/1000-view.cx)*view.s, CH/2-(N/1000-view.cy)*view.s]; } const [x,y]=MERC.toM(lo,la), ws=256*Math.pow(2,mv.z); return [CW/2+(x-mv.mx)*ws, CH/2+(y-mv.my)*ws]; },
    tiles:()=>({ok:tileOK,err:tileErr}), cur:()=>CUR, tabs:()=>Object.fromEntries(Object.entries(TABS).map(([k,t])=>[k,!!t.v]))};
  root.dispatchEvent(new CustomEvent('aggmap:ready',{bubbles:true}));
}
})();
