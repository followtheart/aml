"""Post-hoc agent review, kept separate from the frozen model judge scores.

Labels are explicit per answer, not a replacement judge or a tuning signal.
Order: C00/v4, C00/v5, C01/v4, C01/v5, C10/v4, C10/v5, C11/v4, C11/v5.
"""
import hashlib
import json
from pathlib import Path
from .grounding_benchmark import ROOT, rows
from .analyze_grounding import paired


LABELS = {
    'gpt4_c27434e8': ('00000001', '00110111'),
    'gpt4_1e4a8aeb': ('11111111', '11111111'),
    'gpt4_fe651585': ('00001100', '10000010'),
    'gpt4_1d80365e': ('01010101', '01011111'),
    '9aaed6a3': ('11110000', '11110000'),
    '6cb6f249': ('11111111', '11111111'),
    'a11281a2': ('00000000', '00000000'),
    '2318644b': ('11110010', '11110011'),
    '9ea5eabc': ('01100100', '10001011'),
    '08e075c7': ('11111111', '11111111'),
    '19b5f2b3': ('11111111', '11111111'),
    '5d3d2817': ('11111111', '11111111'),
}

NOTES = {
    'gpt4_c27434e8': '以日本零式先開始為正確；不接受結論說 Ferrari 先的答案。',
    'gpt4_1e4a8aeb': '六天符合參考答案；全數計算一致。',
    'gpt4_fe651585': 'Alex 為參考答案；拒絕同時說無法比較或推理明示 Rachel 更早的自相矛盾答案。after C00/v4 與 C11/v4 的結論符合參考，但缺少 Rachel 日期支持，另列為證據不足，不能視作已證明推理正確。',
    'gpt4_1d80365e': '參考明確接受二天或包含末日的三天；五月十五至十七日符合兩種計數，不能因此判錯。棄答仍為零。',
    '9aaed6a3': '75 × 1% = 0.75；拒絕將消費額 75 當成回饋。',
    '6cb6f249': '七天加十天為十七天；全數正確。',
    'a11281a2': '350 是兩週後總量，增量為 350 − 250 = 100；全部答案將總量誤當增量。',
    '2318644b': '按發布參考接受 270；300 − 30 = 270。210／210 至 255／300 或棄答不符合。來源 over 300 的精度限制另列於報告，不選擇性處罰特定組。',
    '9ea5eabc': '問的是最新家庭旅行地點；Paris 符合，Hawaii 不符。額外月份敘述不作完整來源認證。',
    '08e075c7': '九個月符合更新後參考；九月二日至提問九月四日不足以改變整月答案。',
    '19b5f2b3': '兩週符合參考；全數正確。',
    '5d3d2817': '小型新創公司的 marketing specialist 符合參考；before C11/v4 的額外管理實習生敘述由同一返回包的 answer_235eb6fb:8:0 明確支持，並非捏造。',
}


def main():
    root=Path(ROOT);results=rows(root/'results.jsonl')
    analysis=json.loads((root/'analysis.json').read_text(encoding='utf-8'))
    assert len(results)==192 and len(LABELS)==12, 'Only publish a complete all-answer review'
    reviewed=[];groups={}
    for r in results:
        version,arm=r['arm'].split('/');idx=('C00','C01','C10','C11').index(arm)*2+('v4','v5').index(r['reader'])
        score=int(LABELS[r['conversation_id']][version=='after'][idx])
        if r['reader_errors']:assert score==0
        row={'conversation_id':r['conversation_id'],'arm':r['arm'],'reader':r['reader'],
             'raw_judge_score':r['score'],'review_score':score,
             'prediction_sha256':hashlib.sha256(r['prediction'].encode()).hexdigest(),
             'note':NOTES[r['conversation_id']]}
        reviewed.append(row);groups.setdefault(r['arm']+'/'+r['reader'],{})[r['conversation_id']]=score
    report={'method':'本代理事後逐答核對全部 192 筆；非盲測、非獨立人類標註；不覆寫凍結 Judge，未用於修改程式或重跑。',
            'criterion':'要求完整符合參考答案、數值及題目明許的容差；拒絕錯誤列表成員與自相矛盾的結論。這是答案與參考的一致性覆核，不是對全部額外主張的完整來源證明。',
            'rows':reviewed,'changed':sum(r['raw_judge_score']!=r['review_score'] for r in reviewed),
            'scores':{key:{'correct':sum(group.values()),'n':len(group)} for key,group in groups.items()},
            'paired_end_to_end':{a:paired(groups['after/'+a+'/v5'],groups['before/'+a+'/v4']) for a in ('C00','C10','C01','C11')},
            'dual_comparisons':{a:paired(groups['after/C11/v5'],groups['after/'+a+'/v5']) for a in ('C00','C10','C01')},
            'retrieval_with_fixed_reader':{a+'/'+rd:paired(groups['after/'+a+'/'+rd],groups['before/'+a+'/'+rd]) for a in ('C00','C10','C01','C11') for rd in ('v4','v5')},
            'reader_with_fixed_retrieval':{v+'/'+a:paired(groups[v+'/'+a+'/v5'],groups[v+'/'+a+'/v4']) for v in ('before','after') for a in ('C00','C10','C01','C11')}}
    anomalies={x['conversation_id'] for x in analysis['same_day_clock_anomalies']}
    report['excluding_clock_anomalies']={a:paired(
        {cid:s for cid,s in groups['after/'+a+'/v5'].items() if cid not in anomalies},
        {cid:s for cid,s in groups['before/'+a+'/v4'].items() if cid not in anomalies}) for a in ('C00','C10','C01','C11')}
    report['nondegraded_comparisons']={}
    for a in ('C00','C10','C01','C11'):
        bad={d['conversation_id'] for d in analysis['question_pipeline'] if d['arm'].endswith('/'+a) and (d['degradation'] or d['verification_errors'] or d['search_errors'])}
        report['nondegraded_comparisons'][a]=paired(
            {cid:s for cid,s in groups['after/'+a+'/v5'].items() if cid not in bad},
            {cid:s for cid,s in groups['before/'+a+'/v4'].items() if cid not in bad})
    (root/'answer-audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='rows'},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
