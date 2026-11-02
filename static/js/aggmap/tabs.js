/* Tabs of the Data page: "per installation" (Leaflet viewer) | "aggregated" (aggmap). Deep link: #aggregated. */
(function(){
  var tabs=document.querySelector('.am-tabs'); if(!tabs) return;
  var btns=[].slice.call(tabs.querySelectorAll('button'));
  function show(id,push){
    btns.forEach(function(b){ var on=b.dataset.tab===id; b.setAttribute('aria-selected',on); b.tabIndex=on?0:-1; var p=document.getElementById(b.getAttribute('aria-controls')); if(p) p.hidden=!on; });
    if(push){ try{ history.replaceState(null,'',id==='agg'?'#aggregated':location.pathname+location.search+'#map-section'); }catch(e){} }
    window.dispatchEvent(new Event('resize'));          // Leaflet and the aggregated map re-measure their container
  }
  btns.forEach(function(b,i){
    b.addEventListener('click',function(){ show(b.dataset.tab,true); });
    b.addEventListener('keydown',function(e){ var j=e.key==='ArrowRight'?i+1:e.key==='ArrowLeft'?i-1:-1; if(j<0||j>=btns.length) return; btns[j].focus(); show(btns[j].dataset.tab,true); });
  });
  if(location.hash==='#aggregated'){ show('agg',false); var s=document.getElementById('map-section'); if(s) s.scrollIntoView(); }
})();
