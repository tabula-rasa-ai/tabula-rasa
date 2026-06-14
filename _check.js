var document=null;

    var API='http://localhost:8000';
    var TOKENS=[];
    TOKENS[0]='<PAD>';TOKENS[1]='<BOS>';TOKENS[2]='<EOS>';TOKENS[3]='<UNK>';
    TOKENS[4]='0';TOKENS[5]='1';TOKENS[6]='2';TOKENS[7]='3';TOKENS[8]='4';TOKENS[9]='5';
    TOKENS[10]='6';TOKENS[11]='7';TOKENS[12]='8';TOKENS[13]='9';TOKENS[14]='+';TOKENS[15]='-';
    TOKENS[16]='*';TOKENS[17]='/';TOKENS[18]='=';TOKENS[19]='(';TOKENS[20]=')';TOKENS[21]='.';
    TOKENS[22]='%';TOKENS[23]=' ';
    var REV={};
    for(var i=0;i<TOKENS.length;i++)if(TOKENS[i])REV[TOKENS[i]]=i;

    function renderVocab(){
        var h='';
        for(var i=0;i<24;i++){
            var c=TOKENS[i]||'<UNK>';
            var d=c==' '?'SPC':c=='<PAD>'?'PAD':c=='<BOS>'?'BOS':c=='<EOS>'?'EOS':c=='<UNK>'?'UNK':c;
            h+='<div style="background:#0f172a;border:1px solid var(--border);border-radius:4px;padding:6px;text-align:center;">';
            h+='<div style="font-size:0.6rem;color:var(--muted);">'+i+'</div>';
            h+='<div style="font-weight:600;color:var(--accent);font-size:0.95rem;">'+d+'</div></div>';
        }
        document.getElementById('vocab-table').innerHTML=h;
    }

    function encode(){
        var t=document.getElementById('encode-input').value;
        var ids=[];
        for(var i=0;i<t.length;i++){
            var id=REV[t[i]];
            if(id===undefined)id=3;
            ids.push(id);
        }
        var h='';
        ids.forEach(function(id){
            var c=TOKENS[id]||'?';
            var d=c==' '?'SPC':c=='<PAD>'?'PAD':c=='<BOS>'?'BOS':c=='<EOS>'?'EOS':c;
            h+='<div class="token-chip"><span class="chr">'+d+'</span><span class="id">'+id+'</span></div>';
        });
        document.getElementById('encode-result').innerHTML=h;
        document.getElementById('encode-count').textContent=ids.length;
    }

    function decode(){
        var t=document.getElementById('decode-input').value.trim();
        var parts=t.split(',').map(function(s){return parseInt(s.trim());});
        var text='';
        parts.forEach(function(id){text+=TOKENS[id]||'?';});
        document.getElementById('decode-result').textContent=text||'(empty)';
    }

    function loadAPI(){
        fetch(API+'/health').then(function(r){return r.json()}).then(function(d){
            if(d.vocab_size)document.getElementById('vocab-size').textContent=d.vocab_size;
            if(d.max_seq_len)document.getElementById('seq-len').textContent=d.max_seq_len;
        }).catch(function(){});
    }

    renderVocab();loadAPI();encode();
