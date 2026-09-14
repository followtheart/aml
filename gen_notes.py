# -*- coding: utf-8 -*-
"""Split DeepSeek shared conversation (layout-preserved PDF text) into per-paper notes."""
import re, os

SRC = r'C:/Users/aapoo/Desktop/aml/deepseek_share_layout.txt'
OUT = r'C:/Users/aapoo/Desktop/aml/llm-memory-survey'
NOTES = os.path.join(OUT, 'notes')
os.makedirs(NOTES, exist_ok=True)

TITLES = {
 'ace.pdf': 'Agentic Context Engineering (ACE): Evolving Contexts for Self-Improving Language Models',
 'agent-workflow-memor…': 'Agent Workflow Memory (AWM)',
 'a-mem.pdf': 'A-Mem: Agentic Memory for LLM Agents',
 'dynamic-cheatsheet.pdf': 'Dynamic Cheatsheet: Test-Time Learning with Adaptive Memory',
 'evermemos.pdf': 'EverMemOS: A Self-Organizing Memory Operating System for Structured Long-Horizon Reasoning',
 'evo-memory.pdf': 'Evo-Memory: Benchmarking LLM Agent Test-Time Learning with Self-Evolving Memory',
 'expel.pdf': 'ExpeL: LLM Agents Are Experiential Learners',
 'g-memory.pdf': 'G-Memory: Tracing Hierarchical Memory for Multi-Agent Systems',
 'hipporag.pdf': 'HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models',
 'hipporag2.pdf': 'HippoRAG 2 — From RAG to Memory: Non-Parametric Continual Learning for Large Language Models',
 'larimar.pdf': 'Larimar: Large Language Models with Episodic Memory Control',
 'locomo.pdf': 'LoCoMo: Evaluating Very Long-Term Conversational Memory of LLM Agents',
 'longmemeval.pdf': 'LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory',
 'mem0.pdf': 'Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory',
 'mem1.pdf': 'MEM1: Learning to Synergize Memory and Reasoning for Efficient Long-Horizon Agents',
 'memagent.pdf': 'MemAgent: Reshaping Long-Context LLM with Multi-Conv RL-Based Memory Agent',
 'memgpt.pdf': 'MemGPT: Towards LLMs as Operating Systems',
 'memorybank.pdf': 'MemoryBank: Enhancing Large Language Models with Long-Term Memory',
 'memoryllm.pdf': 'MemoryLLM: Towards Self-Updatable Large Language Models',
 'memoryos.pdf': 'MemoryOS: Memory OS of AI Agent',
 'memory-r1.pdf': 'Memory-R1: Enhancing Large Language Model Agents to Manage and Utilize Memories via Reinforcement Learning',
 'memos.pdf': 'MemOS: A Memory OS for AI System',
 'mirix.pdf': 'MIRIX: Multi-Agent Memory System for LLM-Based Agents',
 'm-plus.pdf': 'M+: Extending MemoryLLM with Scalable Long-Term Memory',
 'reasoningbank.pdf': 'ReasoningBank: Scaling Agent Self-Evolving with Reasoning Memory',
 'secom.pdf': 'SeCom: On Memory Construction and Retrieval for Personalized Conversational Agents',
 'structural-memory.pdf': 'On the Structural Memory of LLM Agents',
 'titans.pdf': 'Titans: Learning to Memorize at Test Time',
 'voyager.pdf': 'Voyager: An Open-Ended Embodied Agent with Large Language Models',
 'zep.pdf': 'Zep: A Temporal Knowledge Graph Architecture for Agent Memory',
}

CJK = r'一-鿿'
TERM = tuple('。！？；：!?;')
HEAD_RE = re.compile(r'^(\d+\.\s*\S|\d+\.\d+\s*\S|\d+\.\d+\.\d+\s*\S|总结$|关键区别|以下是对|【)')
H2_RE = re.compile(r'^\d+\.\s*\S')
H3_RE = re.compile(r'^\d+\.\d+\s*\S')

def promote(line):
    if line == '总结':
        return '## 总结'
    if len(line) <= 46:
        if H3_RE.match(line):
            return '### ' + line
        if H2_RE.match(line):
            return '## ' + line
    return line

def reflow_prose(lines):
    """Join hard-wrapped prose lines into paragraphs; promote headings.

    Blank lines only end a paragraph when the buffer ends with terminal
    punctuation (layout extraction inserts blanks between wrapped lines).
    """
    out, buf = [], []
    def flush2():
        nonlocal buf
        if buf:
            out.append(promote(buf[0])); out.append(''); buf = []
    for raw in lines:
        l = re.sub(r'\s{2,}', ' ', raw.strip())
        if not l:
            if buf and buf[0].endswith(TERM):
                flush2()
            continue
        if HEAD_RE.match(l):
            flush2(); out.append(promote(l)); out.append(''); continue
        if not buf:
            buf = [l]; continue
        prev = buf[0]
        joinable = (not prev.endswith(TERM)
                    and re.search(r'[A-Za-z0-9' + CJK + r'）、，：:—\-]$', prev)
                    and re.match(r'^[A-Za-z0-9' + CJK + r'“‘（(《]', l))
        if joinable:
            if prev.endswith('-'):
                buf[0] = prev + l
            elif re.search(r'[' + CJK + r']$', prev) or re.match(r'^[' + CJK + r']', l):
                buf[0] = prev + l
            else:
                buf[0] = prev + ' ' + l
        else:
            flush2(); buf = [l]
    flush2()
    return out

GRID_RUNS = re.compile(r'(?<=\S)\s{2,}(?=\S)')

def process(text):
    """Split text into prose (reflowed) and table regions (fenced, alignment kept).

    A grid line contains >=2 runs of 2+ spaces. A table block needs >=2 non-blank
    grid/indented lines; isolated grid lines are treated as prose.
    """
    raw_lines = [l.rstrip() for l in text.split('\n')]
    n = len(raw_lines)
    blocks = []
    i = 0
    prose = []
    def flush_prose():
        nonlocal prose
        if prose:
            blocks.extend(reflow_prose(prose)); blocks.append(''); prose = []
    while i < n:
        l = raw_lines[i]
        stripped = l.strip()
        is_grid = len(GRID_RUNS.findall(l)) >= 2
        if is_grid:
            # collect candidate block
            tbl = [l]; j = i + 1
            while j < n:
                l2 = raw_lines[j]
                s2 = l2.strip()
                if not s2:
                    tbl.append(l2); j += 1; continue
                if len(GRID_RUNS.findall(l2)) >= 2 or l2.startswith('   '):
                    tbl.append(l2); j += 1; continue
                break
            nonblank = [x for x in tbl if x.strip()]
            grids = [x for x in nonblank if len(GRID_RUNS.findall(x)) >= 2]
            if len(nonblank) >= 2 and len(grids) >= 2:
                flush_prose()
                while tbl and not tbl[0].strip(): tbl.pop(0)
                while tbl and not tbl[-1].strip(): tbl.pop()
                # de-indent: remove common leading spaces
                indents = [len(x) - len(x.lstrip()) for x in tbl if x.strip()]
                cut = min(indents) if indents else 0
                tbl = [x[cut:] if x.strip() else '' for x in tbl]
                blocks.append('```'); blocks.extend(tbl); blocks.append('```'); blocks.append('')
            else:
                prose.extend(tbl)
            i = j
        else:
            prose.append(l); i += 1
    flush_prose()
    # collapse >2 consecutive blank lines
    res, blank = [], 0
    for b in blocks:
        if b == '':
            blank += 1
            if blank > 1: continue
        else:
            blank = 0
        res.append(b)
    return '\n'.join(res).strip() + '\n'

t = open(SRC, encoding='utf-8').read()
marks = [(m.start(), m.group(1).strip()) for m in re.finditer(r'(?m)^\s*(\S[^\n]*?)\s*\n\s*PDF [\d.]+[KM]B\s*\n(\s*\n)*\s*阅读并分析该论文的主要内容', t)]
print('markers:', len(marks))
assert len(marks) == 30

ANS_RE = re.compile(r'(?m)^\s*好的，我(已|已经)')
def slugify(name):
    return 'agent-workflow-memory' if name.startswith('agent-workflow') else name[:-4]

for i, (pos, name) in enumerate(marks):
    end = marks[i+1][0] if i+1 < len(marks) else len(t)
    seg = t[pos:end]
    if name == 'zep.pdf':
        cut = seg.find('输出一份上述诸多方法的综述')
        if cut > 0: seg = seg[:cut]
    m = ANS_RE.search(seg)
    body = seg[m.start():] if m else seg
    body = body.replace('和 DeepSeek 继续聊', '').strip()
    title = TITLES[name]
    slug = slugify(name)
    fname = f'{i+1:02d}-{slug}.md'
    disp = 'agent-workflow-memory.pdf' if name.startswith('agent-workflow') else name
    md = [f'# {title}', '',
          f'> 来源文件：`{disp}` · 整理自 DeepSeek 对话阅读笔记 · [返回综述](../survey.md)', '',
          process(body), '']
    open(os.path.join(NOTES, fname), 'w', encoding='utf-8').write('\n'.join(md))
    print(f'{fname}: {len(body)} chars')

# raw survey draft (final regenerated survey with 缺陷 column)
seg = t[marks[-1][0]:]
i = seg.find('重新生成综述')
sub = seg[i:]
j = sub.find('1. 引言')
survey = sub[j:].replace('和 DeepSeek 继续聊', '').strip()
open(os.path.join(OUT, 'survey_raw.txt'), 'w', encoding='utf-8').write(survey)
print('survey raw chars:', len(survey))
